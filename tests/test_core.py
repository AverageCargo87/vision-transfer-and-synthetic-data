"""Offline CPU tests: statistics, GPU-side augmentation maths, dataset readers, the sign renderer,
and a check that every number quoted in the README exists in results/results.jsonl."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision.common import bootstrap_ci, paired_bootstrap  # noqa: E402


def test_bootstrap_ci_brackets_the_mean():
    c = np.r_[np.ones(900), np.zeros(100)]
    acc, lo, hi = bootstrap_ci(c, n_boot=1000)
    assert acc == pytest.approx(0.9)
    assert lo < 0.9 < hi
    assert 0.87 < lo and hi < 0.93  # binomial SE is ~0.0095


def test_paired_bootstrap_detects_a_real_gain_and_ignores_noise():
    rng = np.random.default_rng(0)
    a = rng.random(5000) < 0.80
    b = a.copy()
    flip = rng.choice(np.flatnonzero(~a), 200, replace=False)
    b[flip] = True  # b fixes 200 of a's errors
    r = paired_bootstrap(a, b)
    assert r["diff"] == pytest.approx(0.04)
    assert r["ci_lo"] > 0 and r["p_not_better"] == 0.0
    same = paired_bootstrap(a, a)
    assert same["diff"] == 0 and same["p_not_better"] == 1.0


def test_gpu_aug_runs_on_cpu_and_keeps_shapes():
    torch = pytest.importorskip("torch")
    from vision.gpu import center_resize, color_jitter, hflip_random, normalize, random_resized_crop, to_gpu_batch

    arr = (np.random.default_rng(0).random((4, 32, 32, 3)) * 255).astype(np.uint8)
    x = to_gpu_batch(arr, np.arange(4), device="cpu")
    assert x.shape == (4, 3, 32, 32) and 0 <= x.min() and x.max() <= 1
    g = torch.Generator().manual_seed(0)
    y = color_jitter(hflip_random(random_resized_crop(x, 24, gen=g), gen=g), gen=g)
    assert y.shape == (4, 3, 24, 24) and float(y.min()) >= 0 and float(y.max()) <= 1
    assert center_resize(x, 16).shape == (4, 3, 16, 16)
    n = normalize(torch.full((1, 3, 2, 2), 0.5), (0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
    assert torch.allclose(n, torch.zeros_like(n))


def test_full_crop_is_identity():
    torch = pytest.importorskip("torch")
    from vision.gpu import random_resized_crop

    x = torch.rand(2, 3, 16, 16)
    y = random_resized_crop(x, 16, scale=(1.0, 1.0), ratio=(1.0, 1.0))
    assert torch.allclose(x, y, atol=1e-5)


def _fake_cub(root: Path):
    from PIL import Image

    r = root / "cub" / "CUB_200_2011"
    (r / "images" / "001.A").mkdir(parents=True)
    (r / "images" / "002.B").mkdir(parents=True)
    rows = [(1, "001.A/a1.jpg", 1, 1), (2, "001.A/a2.jpg", 1, 0), (3, "002.B/b1.jpg", 2, 1), (4, "002.B/b2.jpg", 2, 0)]
    for _, p, _, _ in rows:
        Image.new("RGB", (40, 30), (120, 50, 10)).save(r / "images" / p)
    (r / "images.txt").write_text("\n".join(f"{i} {p}" for i, p, _, _ in rows))
    (r / "image_class_labels.txt").write_text("\n".join(f"{i} {c}" for i, _, c, _ in rows))
    (r / "train_test_split.txt").write_text("\n".join(f"{i} {t}" for i, _, _, t in rows))
    (r / "classes.txt").write_text("1 001.Black_footed_Albatross\n2 002.Laysan_Albatross")


def test_cub_reader_and_cache(tmp_path, monkeypatch):
    _fake_cub(tmp_path)
    import vision.datasets as ds

    monkeypatch.setattr(ds, "DATA", tmp_path)
    monkeypatch.setattr(ds, "CACHE", tmp_path / "cache")
    tr, te, classes = ds.load_cub()
    assert [p.name for p in tr.paths] == ["a1.jpg", "b1.jpg"] and list(tr.labels) == [0, 1]
    assert list(te.labels) == [0, 1] and classes == ["Black footed Albatross", "Laysan Albatross"]
    arr = ds.build_cache("t", tr, 16)
    assert arr.shape == (2, 16, 16, 3) and arr.dtype == np.uint8


def test_gtsrb_tracks_are_deterministic():
    from vision.datasets import gtsrb_track_ids

    p = [Path("Images/00003/00012_00029.ppm"), Path("Images/00003/00012_00000.ppm"), Path("Images/00004/00012_00000.ppm")]
    ids = gtsrb_track_ids(p)
    assert ids[0] == ids[1] != ids[2]
    assert list(ids) == list(gtsrb_track_ids(p))


def test_every_drawable_sign_renders_deterministically():
    pytest.importorskip("cv2")
    from vision.synth_signs import DRAWABLE, draw_sign, render_sample

    for c in DRAWABLE:
        im = np.asarray(draw_sign(c, 64))
        assert im.shape == (64, 64, 4) and im[..., 3].mean() > 60, c  # something opaque was drawn
    a = render_sample(14, 48, np.random.default_rng(5))
    b = render_sample(14, 48, np.random.default_rng(5))
    assert a.shape == (48, 48, 3) and np.array_equal(a, b)
    with pytest.raises(ValueError):
        draw_sign(9, 64)  # 'no passing' is not in the drawable set


def test_mirrored_arrows_are_mirror_images():
    from vision.synth_signs import draw_sign

    for right, left in ((33, 34), (36, 37), (38, 39)):
        r = np.asarray(draw_sign(right, 96)).astype(int)
        l_ = np.asarray(draw_sign(left, 96)).astype(int)
        assert np.abs(r[:, ::-1] - l_).mean() < 6, (right, left)


def test_readme_numbers_come_from_results():
    """Every 'NN.N%' in the README tables must match a value logged in results/results.jsonl."""
    res = ROOT / "results" / "results.jsonl"
    if not res.exists() or not (ROOT / "README.md").exists():
        pytest.skip("no results yet")
    vals = set()
    records = [json.loads(line) for line in res.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = ROOT / "results" / "summary.json"
    if summary.exists():
        records.append(json.loads(summary.read_text(encoding="utf-8")))
    for rec in records:
        stack = [rec]
        while stack:
            v = stack.pop()
            if isinstance(v, dict):
                stack.extend(v.values())
            elif isinstance(v, list):
                stack.extend(v)
            elif isinstance(v, float) and 0 <= v <= 1:
                vals.add(round(v * 100, 1))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    quoted = [float(m) for m in re.findall(r"\|\s*(\d{1,3}\.\d)%", readme)]
    missing = [q for q in quoted if not any(abs(q - v) < 0.051 for v in vals)]
    assert not missing, f"README percentages not found in results.jsonl: {missing}"
