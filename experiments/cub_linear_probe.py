"""E1. Backbone sweep on CUB-200-2011 (all 200 species): frozen features + a linear classifier.

For each backbone: features of every train/test image (centre crop + horizontal flip, averaged),
a multinomial logistic regression with its L2 strength picked on a 20% stratified hold-out of the
train split, then refit on the full train split and scored once on the official test split.
Test probabilities are kept for the ensemble and the paired comparisons in E2.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision.common import CACHE, bootstrap_ci, log_result, seed_all  # noqa: E402
from vision.datasets import build_cache, load_cub  # noqa: E402
from vision.gpu import center_resize, normalize, to_gpu_batch  # noqa: E402
from vision.linear import fit_logreg, predict  # noqa: E402
from vision.models import BACKBONES, SHORT, load_backbone  # noqa: E402

PRED_DIR = CACHE / "preds"


@torch.no_grad()
def extract(bb, arr: np.ndarray, batch: int = 64) -> tuple[np.ndarray, float]:
    feats = []
    t0 = time.perf_counter()
    for i in range(0, len(arr), batch):
        x = to_gpu_batch(arr, np.arange(i, min(i + batch, len(arr))))
        x = normalize(center_resize(x, bb.img_size), bb.mean, bb.std)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = bb.model(x).float() + bb.model(x.flip(-1)).float()
        feats.append((f / 2).cpu())
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return torch.cat(feats).numpy(), len(arr) * 2 / dt  # images/s counts both views


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=BACKBONES)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seed_all(args.seed)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    train, test, classes = load_cub()
    t0 = time.perf_counter()
    tr_arr = build_cache("cub_train", train, 256)
    te_arr = build_cache("cub_test", test, 256)
    print(f"cache ready: train {tr_arr.shape}, test {te_arr.shape} in {time.perf_counter() - t0:.0f} s", flush=True)

    rng = np.random.default_rng(args.seed)
    val_mask = np.zeros(len(train.labels), bool)
    for c in range(200):  # stratified 20% hold-out for choosing the L2 strength
        idx = np.flatnonzero(train.labels == c)
        val_mask[rng.choice(idx, size=max(1, round(0.2 * len(idx))), replace=False)] = True

    for name in args.models:
        bb = load_backbone(name)
        ftr, ips = extract(bb, tr_arr)
        fte, _ = extract(bb, te_arr)
        del bb
        torch.cuda.empty_cache()
        mu, sd = ftr.mean(0), ftr.std(0) + 1e-6
        Xtr = torch.tensor((ftr - mu) / sd, device="cuda")
        Xte = torch.tensor((fte - mu) / sd, device="cuda")
        ytr = torch.tensor(train.labels, device="cuda")
        vm = torch.tensor(val_mask, device="cuda")
        best = None
        for wd in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
            W = fit_logreg(Xtr[~vm], ytr[~vm], 200, wd)
            acc = (predict(W, Xtr[vm]).argmax(1) == ytr[vm]).float().mean().item()
            if best is None or acc > best[1]:
                best = (wd, acc)
        W = fit_logreg(Xtr, ytr, 200, best[0])
        prob = predict(W, Xte).cpu().numpy()
        np.save(PRED_DIR / f"cub_probe__{name}.npy", prob.astype(np.float16))
        correct = prob.argmax(1) == test.labels
        acc, lo, hi = bootstrap_ci(correct)
        rec = log_result(
            "cub_linear_probe", model=name, label=SHORT[name], feat_dim=int(ftr.shape[1]), img_size=224,
            l2=best[0], val_acc=round(best[1], 4), test_acc=round(acc, 4), ci95=[round(lo, 4), round(hi, 4)],
            n_test=int(len(correct)), extract_img_per_s=round(ips, 1),
        )
        print(f"{SHORT[name]:38s} test {acc:.4f} [{lo:.4f}, {hi:.4f}]  l2={best[0]}  {ips:.0f} img/s", flush=True)


if __name__ == "__main__":
    main()
