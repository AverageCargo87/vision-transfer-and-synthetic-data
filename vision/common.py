"""Shared helpers: paths, seeding, result logging and bootstrap statistics."""
from __future__ import annotations

import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("VISION_DATA", ROOT / "data"))
CACHE = Path(os.environ.get("VISION_CACHE", DATA / "cache"))
RESULTS = ROOT / "results"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def env_info() -> dict:
    info = {"python": platform.python_version(), "os": platform.system(), "cpu_count": os.cpu_count()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    except ImportError:
        pass
    try:
        import timm

        info["timm"] = timm.__version__
    except ImportError:
        pass
    return info


def log_result(experiment: str, **record) -> dict:
    """Append one record to results/results.jsonl (the single source for every number in the README)."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    rec = {"experiment": experiment, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **record, "env": env_info()}
    with open(RESULTS / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def read_results(experiment: str | None = None) -> list[dict]:
    p = RESULTS / "results.jsonl"
    if not p.exists():
        return []
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if experiment is None or r["experiment"] == experiment]


def bootstrap_ci(correct, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Accuracy and a percentile bootstrap CI over test items."""
    c = np.asarray(correct, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(c), size=(n_boot, len(c)))
    accs = c[idx].mean(axis=1)
    lo, hi = np.quantile(accs, [alpha / 2, 1 - alpha / 2])
    return float(c.mean()), float(lo), float(hi)


def paired_bootstrap(correct_a, correct_b, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Difference in accuracy (b minus a) on the same test items, with a CI and the share of resamples where b <= a."""
    a = np.asarray(correct_a, dtype=np.float64)
    b = np.asarray(correct_b, dtype=np.float64)
    assert a.shape == b.shape
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    d = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    return {"diff": float(b.mean() - a.mean()), "ci_lo": float(lo), "ci_hi": float(hi), "p_not_better": float((d <= 0).mean())}
