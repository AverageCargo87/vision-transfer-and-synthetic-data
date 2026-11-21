"""E2b. Does an ensemble really help, or is the gain inside the noise?

The members are fixed BEFORE looking at the test split: every fine-tuned model from E2 (one CNN, one
ViT, chosen in E2 by their E1 hold-out accuracy), probabilities averaged with equal weights. No
search over the test set. The ensemble is compared with the best single fine-tuned model by a paired
bootstrap on the same 5,794 test images: if the CI of the gain touches zero, it is not a real gain.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision.common import CACHE, bootstrap_ci, log_result, paired_bootstrap  # noqa: E402
from vision.datasets import load_cub  # noqa: E402
from vision.models import SHORT  # noqa: E402


def label(stem: str) -> str:
    kind, name = stem.split("__", 1)
    base = name.split("__")[0]
    return f"{SHORT.get(base, base)} {'fine-tuned' if kind == 'cub_ft' else 'linear probe'}"


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="lpft", help="which fine-tuned runs form the ensemble (fixed in advance)")
    args = ap.parse_args()
    _, test, _ = load_cub()
    y = test.labels
    members = sorted((CACHE / "preds").glob(f"cub_ft__*__{args.tag}.npy" if args.tag else "cub_ft__*.npy"))
    if len(members) < 2:
        raise SystemExit("need at least two fine-tuned models from experiments/cub_finetune.py")
    preds = {p.stem: np.load(p).astype(np.float32) for p in members}
    singles = {k: v.argmax(1) == y for k, v in preds.items()}
    best = max(singles, key=lambda k: singles[k].mean())
    ens = sum(preds.values()) / len(preds)
    correct = ens.argmax(1) == y
    a, lo, hi = bootstrap_ci(correct)
    pb = paired_bootstrap(singles[best], correct)
    log_result("cub_ensemble", members=[label(k) for k in preds], best_single=label(best),
               best_single_acc=round(float(singles[best].mean()), 4), test_acc=round(a, 4), ci95=[round(lo, 4), round(hi, 4)],
               gain=round(pb["diff"], 4), gain_ci95=[round(pb["ci_lo"], 4), round(pb["ci_hi"], 4)], p_not_better=round(pb["p_not_better"], 4))
    print(f"members: {', '.join(label(k) for k in preds)}")
    print(f"best single {label(best)} {singles[best].mean():.4f}; ensemble {a:.4f} [{lo:.4f}, {hi:.4f}]")
    print(f"paired gain {pb['diff'] * 100:+.2f} pp, 95% CI [{pb['ci_lo'] * 100:+.2f}, {pb['ci_hi'] * 100:+.2f}], P(not better) {pb['p_not_better']:.3f}")


if __name__ == "__main__":
    main()
