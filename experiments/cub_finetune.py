"""E2. Full fine-tuning on CUB-200-2011 from the RAM cache, bf16, everything augmented on the GPU.

Two recipes:
  naive  a fresh random head with a 10x learning rate, AdamW, one warm-up epoch then cosine decay
  LP-FT  (--lpft) the head is first fitted as a linear probe on the frozen features (train split only),
         then the whole network is fine-tuned from there with one learning rate, so the first updates
         do not push large random-head gradients into good pretrained features (Kumar et al., 2022)
Both: label smoothing 0.1, RandomResizedCrop + flip + light colour jitter on the GPU, bf16.
Evaluation: centre crop plus its mirror, probabilities averaged. Test probabilities are stored for the
ensemble and the paired bootstrap comparisons.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision.common import CACHE, bootstrap_ci, log_result, seed_all  # noqa: E402
from vision.datasets import build_cache, load_cub  # noqa: E402
from vision.common import read_results  # noqa: E402
from vision.gpu import center_resize, color_jitter, hflip_random, normalize, random_resized_crop, to_gpu_batch  # noqa: E402
from vision.linear import fit_logreg, fold_standardization  # noqa: E402
from vision.models import SHORT, load_backbone  # noqa: E402

PRED_DIR = CACHE / "preds"


@torch.no_grad()
def evaluate(bb, arr: np.ndarray, batch: int = 64) -> np.ndarray:
    bb.model.eval()
    out = []
    for i in range(0, len(arr), batch):
        x = to_gpu_batch(arr, np.arange(i, min(i + batch, len(arr))))
        x = normalize(center_resize(x, bb.img_size), bb.mean, bb.std)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.softmax(bb.model(x).float(), 1) + torch.softmax(bb.model(x.flip(-1)).float(), 1)
        out.append((p / 2).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def init_head_from_probe(bb, arr: np.ndarray, y: torch.Tensor, l2: float, batch: int = 64) -> float:
    """LP step: fit a logistic regression on the frozen pooled features and copy it into the head."""
    bb.model.eval()
    feats = []
    for i in range(0, len(arr), batch):
        x = normalize(center_resize(to_gpu_batch(arr, np.arange(i, min(i + batch, len(arr)))), bb.img_size), bb.mean, bb.std)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = bb.model.forward_head(bb.model.forward_features(x), pre_logits=True).float()
            f = (f + bb.model.forward_head(bb.model.forward_features(x.flip(-1)), pre_logits=True).float()) / 2
        feats.append(f)
    X = torch.cat(feats)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    with torch.enable_grad():
        W = fit_logreg((X - mu) / sd, y, 200, l2)
    w, b = fold_standardization(W, mu, sd)
    head = bb.model.get_classifier()
    head.weight.copy_(w.to(head.weight.dtype))
    head.bias.copy_(b.to(head.bias.dtype))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        acc = float((head(X).argmax(1) == y).float().mean())
    return acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--lpft", action="store_true", help="initialise the head from a linear probe, single LR")
    ap.add_argument("--scale_min", type=float, default=0.35, help="RandomResizedCrop minimum area")
    args = ap.parse_args()
    seed_all(args.seed)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    train, test, _ = load_cub()
    tr_arr = build_cache("cub_train", train, 256)
    te_arr = build_cache("cub_test", test, 256)
    ytr = torch.tensor(train.labels, device="cuda")

    bb = load_backbone(args.model, num_classes=200)
    head = list(bb.model.get_classifier().parameters())
    head_ids = {id(p) for p in head}
    body = [p for p in bb.model.parameters() if id(p) not in head_ids]
    lp_train_acc = None
    if args.lpft:
        probe = [r for r in read_results("cub_linear_probe") if r["model"] == args.model]
        l2 = probe[-1]["l2"] if probe else 1e-3  # the L2 strength E1 chose on its train-split hold-out
        lp_train_acc = init_head_from_probe(bb, tr_arr, ytr, l2)
        print(f"LP init: head fitted on frozen features (l2={l2}), train accuracy {lp_train_acc:.4f}", flush=True)
    head_lr = args.lr if args.lpft else args.lr * 10
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr}, {"params": head, "lr": head_lr}], weight_decay=args.wd)
    steps_per_epoch = len(tr_arr) // args.bs
    total = steps_per_epoch * args.epochs
    warm = steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))
    )
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.cuda.reset_peak_memory_stats()
    t_train = time.perf_counter()
    seen = 0
    for ep in range(args.epochs):
        bb.model.train()
        perm = rng.permutation(len(tr_arr))
        t0 = time.perf_counter()
        run_loss = 0.0
        for s in range(steps_per_epoch):
            idx = np.sort(perm[s * args.bs:(s + 1) * args.bs])
            x = to_gpu_batch(tr_arr, idx)
            x = color_jitter(hflip_random(random_resized_crop(x, bb.img_size, scale=(args.scale_min, 1.0), gen=gen), gen=gen), gen=gen)
            x = normalize(x, bb.mean, bb.std)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(bb.model(x).float(), ytr[idx], label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bb.model.parameters(), 1.0)
            opt.step()
            sched.step()
            run_loss += loss.item()
            seen += len(idx)
        torch.cuda.synchronize()
        print(f"epoch {ep + 1}/{args.epochs} loss {run_loss / steps_per_epoch:.3f} {steps_per_epoch * args.bs / (time.perf_counter() - t0):.0f} img/s", flush=True)
    train_s = time.perf_counter() - t_train
    prob = evaluate(bb, te_arr)
    key = args.model + (f"__{args.tag}" if args.tag else "")
    np.save(PRED_DIR / f"cub_ft__{key}.npy", prob.astype(np.float16))
    torch.save(bb.model.state_dict(), CACHE / f"cub_ft__{key}.pt")
    correct = prob.argmax(1) == test.labels
    acc, lo, hi = bootstrap_ci(correct)
    log_result(
        "cub_finetune", model=args.model, label=SHORT.get(args.model, args.model), tag=args.tag, recipe="LP-FT" if args.lpft else "naive",
        lp_train_acc=lp_train_acc, scale_min=args.scale_min, epochs=args.epochs,
        batch=args.bs, lr=args.lr, wd=args.wd, seed=args.seed, test_acc=round(acc, 4), ci95=[round(lo, 4), round(hi, 4)],
        train_minutes=round(train_s / 60, 2), train_img_per_s=round(seen / train_s, 1),
        peak_gpu_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2), n_test=int(len(correct)),
    )
    print(f"{SHORT.get(args.model, args.model)}: test {acc:.4f} [{lo:.4f}, {hi:.4f}] in {train_s / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
