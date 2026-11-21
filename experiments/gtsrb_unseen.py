"""E5. Recognising traffic-sign classes that have zero real training images (GTSRB, 43 classes).

Ten classes are removed from the real training data entirely (every image of them). The only way the
model can learn them is from procedurally rendered signs (vision/synth_signs.py). Scored on the full
official GTSRB test set (12,630 real images, held-out classes included).

Conditions
  oracle             real images of all 43 classes (the usual supervised setting, an upper bound)
  real_seen_only     real images of the 33 seen classes only (held-out accuracy is 0 by construction)
  synth_unseen_only  + synthetic images for the 10 held-out classes only
  synth_all          + synthetic images for every drawable class, seen ones included, so that
                     'looks synthetic' stops being a cue for 'is a held-out class'
  synth_all + knn    same network, but classified by cosine k-NN over embeddings of the real seen
                     training images and the synthetic held-out images (no retraining)

Model: torchvision ResNet-50 (ImageNet weights), 64 px ROI crops, bf16, GPU augmentation without flips
(left/right arrows must stay left/right).
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
from vision.datasets import GTSRB_NAMES, build_cache, load_cub, load_gtsrb  # noqa: E402
from vision.gpu import color_jitter, normalize, random_resized_crop, to_gpu_batch  # noqa: E402
from vision.synth_signs import DRAWABLE, render_sample  # noqa: E402

UNSEEN = [0, 3, 8, 13, 14, 15, 17, 32, 34, 39]
SIZE = 64
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def synth_bank(classes: list[int], n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    out = CACHE / f"gtsrb_synth_s{seed}_n{n}_{SIZE}.npz"
    have = {}
    if out.exists():
        z = np.load(out)
        have = {int(k[1:]): z[k] for k in z.files}
    need = [c for c in classes if c not in have]
    if need:
        cub_train, _, _ = load_cub()
        bgs = build_cache("cub_train", cub_train, 256)[::12]  # ~500 natural photos as backgrounds
        for c in need:
            rng = np.random.default_rng(seed * 1000 + c)
            have[c] = np.stack([render_sample(c, SIZE, rng, bgs) for _ in range(n)])
            print(f"  rendered class {c:2d} ({GTSRB_NAMES[c]}): {n}", flush=True)
        np.savez(out, **{f"c{c}": v for c, v in have.items()})
    X = np.concatenate([have[c] for c in classes])
    y = np.concatenate([np.full(len(have[c]), c) for c in classes]).astype(np.int64)
    return X, y


def make_model():
    import torchvision

    m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = torch.nn.Linear(m.fc.in_features, 43)
    return m.cuda()


@torch.no_grad()
def embed_and_predict(m, arr: np.ndarray, batch: int = 512) -> tuple[np.ndarray, np.ndarray]:
    m.eval()
    feats, probs = [], []
    body = torch.nn.Sequential(*list(m.children())[:-1])
    for i in range(0, len(arr), batch):
        x = normalize(to_gpu_batch(arr, np.arange(i, min(i + batch, len(arr)))), MEAN, STD)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = body(x).flatten(1).float()
            p = torch.softmax(m.fc(f), 1)
        feats.append(F.normalize(f, dim=1).cpu())
        probs.append(p.cpu())
    return torch.cat(feats).numpy(), torch.cat(probs).numpy()


def train(X: np.ndarray, y: np.ndarray, epochs: int, bs: int, lr: float, seed: int):
    seed_all(seed)
    m = make_model()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.05)
    steps = len(X) // bs
    total, warm = steps * epochs, steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    yt = torch.tensor(y, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(seed)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    for ep in range(epochs):
        m.train()
        perm = rng.permutation(len(X))
        tot = 0.0
        for s in range(steps):
            idx = np.sort(perm[s * bs:(s + 1) * bs])
            x = to_gpu_batch(X, idx)
            x = color_jitter(random_resized_crop(x, SIZE, scale=(0.7, 1.0), ratio=(0.9, 1.1), gen=gen), 0.3, 0.3, 0.3, gen=gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(m(normalize(x, MEAN, STD)).float(), yt[idx], label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        print(f"    epoch {ep + 1}/{epochs} loss {tot / steps:.3f}", flush=True)
    return m, (time.perf_counter() - t0) / 60


def score(name: str, pred: np.ndarray, y: np.ndarray, seed: int, extra: dict) -> None:
    unseen = np.isin(y, UNSEEN)
    correct = pred == y
    acc, lo, hi = bootstrap_ci(correct, seed=seed)
    ua, ulo, uhi = bootstrap_ci(correct[unseen], seed=seed)
    sa, slo, shi = bootstrap_ci(correct[~unseen], seed=seed)
    per_cls = {GTSRB_NAMES[c]: round(float(correct[y == c].mean()), 4) for c in UNSEEN}
    log_result("gtsrb_unseen", condition=name, seed=seed, overall=round(acc, 4), overall_ci95=[round(lo, 4), round(hi, 4)],
               unseen=round(ua, 4), unseen_ci95=[round(ulo, 4), round(uhi, 4)], seen=round(sa, 4), seen_ci95=[round(slo, 4), round(shi, 4)],
               n_test=int(len(y)), n_test_unseen=int(unseen.sum()), unseen_classes=UNSEEN, per_unseen_class=per_cls, **extra)
    print(f"  {name:20s} seed {seed}: overall {acc:.4f}  seen {sa:.4f}  held-out {ua:.4f} [{ulo:.4f}, {uhi:.4f}]", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="*", default=["oracle", "real_seen_only", "synth_unseen_only", "synth_all"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_synth", type=int, default=1500)
    args = ap.parse_args()

    tr, te = load_gtsrb()
    Xr = build_cache("gtsrb_train_roi", tr, SIZE, use_box=True, margin=0.1)
    Xt = build_cache("gtsrb_test_roi", te, SIZE, use_box=True, margin=0.1)
    seen_mask = ~np.isin(tr.labels, UNSEEN)
    print(f"real train {len(Xr)} ({seen_mask.sum()} of seen classes), test {len(Xt)} ({np.isin(te.labels, UNSEEN).sum()} of held-out classes)", flush=True)

    for seed in args.seeds:
        for cond in args.conditions:
            if cond == "oracle":
                X, y = Xr, tr.labels
            else:
                X, y = Xr[seen_mask], tr.labels[seen_mask]
                if cond in ("synth_unseen_only", "synth_all"):
                    cls = UNSEEN if cond == "synth_unseen_only" else DRAWABLE
                    Xs, ys = synth_bank(cls, args.n_synth, seed)
                    X, y = np.concatenate([X, Xs]), np.concatenate([y, ys])
            print(f"{cond} seed {seed}: {len(X)} training images", flush=True)
            m, minutes = train(X, y, args.epochs, args.bs, args.lr, seed)
            fte, pte = embed_and_predict(m, Xt)
            extra = {"train_images": int(len(X)), "epochs": args.epochs, "train_minutes": round(minutes, 2), "n_synth_per_class": args.n_synth if "synth" in cond else 0}
            score(cond, pte.argmax(1), te.labels, seed, extra)
            if cond == "synth_all":
                # cosine k-NN over real seen-class embeddings + synthetic held-out embeddings
                Xs_u, ys_u = synth_bank(UNSEEN, args.n_synth, seed)
                f_seen, _ = embed_and_predict(m, Xr[seen_mask])
                f_syn, _ = embed_and_predict(m, Xs_u)
                bank = torch.tensor(np.concatenate([f_seen, f_syn]), device="cuda")
                bank_y = torch.tensor(np.concatenate([tr.labels[seen_mask], ys_u]), device="cuda")
                preds = []
                for i in range(0, len(fte), 1024):
                    sim = torch.tensor(fte[i:i + 1024], device="cuda") @ bank.T
                    top = sim.topk(10, dim=1)
                    votes = torch.zeros(len(sim), 43, device="cuda").scatter_add_(1, bank_y[top.indices], top.values)
                    preds.append(votes.argmax(1).cpu())
                score("synth_all + knn", torch.cat(preds).numpy(), te.labels, seed, {**extra, "k": 10})
            del m
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
