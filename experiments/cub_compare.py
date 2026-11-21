"""E2c. Paired comparisons between recipes for the same backbone, on the same 5,794 test images.

For every backbone that has more than one stored prediction (linear probe, naive fine-tune, LP-FT),
log the accuracy difference with a paired-bootstrap 95% CI. A difference whose CI includes zero is
reported as no difference.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision.common import CACHE, log_result, paired_bootstrap  # noqa: E402
from vision.datasets import load_cub  # noqa: E402
from vision.models import SHORT  # noqa: E402

RECIPE = {"cub_probe": "linear probe", "cub_ft": "naive fine-tune", "cub_ft_lpft": "LP-FT"}


def main() -> None:
    _, test, _ = load_cub()
    y = test.labels
    runs: dict[str, dict[str, np.ndarray]] = {}
    for p in (CACHE / "preds").glob("cub_*.npy"):
        kind, rest = p.stem.split("__", 1)
        parts = rest.split("__")
        model, tag = parts[0], (parts[1] if len(parts) > 1 else "")
        recipe = RECIPE["cub_ft_lpft"] if tag == "lpft" else RECIPE[kind]
        runs.setdefault(model, {})[recipe] = np.load(p).astype(np.float32).argmax(1) == y
    order = ["linear probe", "naive fine-tune", "LP-FT"]
    for model, rec in sorted(runs.items()):
        for a, b in itertools.combinations([r for r in order if r in rec], 2):
            pb = paired_bootstrap(rec[a], rec[b])
            log_result("cub_compare", model=model, label=SHORT.get(model, model), a=a, b=b, acc_a=round(float(rec[a].mean()), 4),
                       acc_b=round(float(rec[b].mean()), 4), diff=round(pb["diff"], 4), diff_ci95=[round(pb["ci_lo"], 4), round(pb["ci_hi"], 4)],
                       p_not_better=round(pb["p_not_better"], 4))
            print(f"{SHORT.get(model, model):30s} {b:>15s} vs {a:<15s} {pb['diff'] * 100:+.2f} pp  CI [{pb['ci_lo'] * 100:+.2f}, {pb['ci_hi'] * 100:+.2f}]", flush=True)


if __name__ == "__main__":
    main()
