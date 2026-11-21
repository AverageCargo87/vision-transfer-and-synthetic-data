"""Dataset readers for CUB-200-2011 and GTSRB, plus a uint8 image cache.

Both datasets are read from their official archives as published by the authors; nothing is re-split
except where an experiment says so (the GTSRB held-out classes).
"""
from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .common import CACHE, DATA


@dataclass
class Split:
    paths: list[Path]
    labels: np.ndarray  # int64, 0-based
    boxes: list[tuple[int, int, int, int]] | None = None  # optional ROI (x1, y1, x2, y2)


# ---------------------------------------------------------------- CUB-200-2011
def cub_root() -> Path:
    return DATA / "cub" / "CUB_200_2011"


def load_cub() -> tuple[Split, Split, list[str]]:
    """Official split: 5,994 train and 5,794 test images over 200 species."""
    r = cub_root()
    paths = dict(line.split(" ", 1) for line in (r / "images.txt").read_text().splitlines())
    labels = {k: int(v) - 1 for k, v in (line.split() for line in (r / "image_class_labels.txt").read_text().splitlines())}
    is_train = {k: v == "1" for k, v in (line.split() for line in (r / "train_test_split.txt").read_text().splitlines())}
    classes = [line.split(" ", 1)[1].split(".", 1)[1].replace("_", " ") for line in (r / "classes.txt").read_text().splitlines()]
    tr, te = ([], []), ([], [])
    for k in sorted(paths, key=int):
        dst = tr if is_train[k] else te
        dst[0].append(r / "images" / paths[k].strip())
        dst[1].append(labels[k])
    return (Split(tr[0], np.array(tr[1], dtype=np.int64)), Split(te[0], np.array(te[1], dtype=np.int64)), classes)


# ---------------------------------------------------------------- GTSRB
GTSRB_NAMES = [
    "speed limit 20", "speed limit 30", "speed limit 50", "speed limit 60", "speed limit 70", "speed limit 80",
    "end of speed limit 80", "speed limit 100", "speed limit 120", "no passing", "no passing for trucks",
    "right of way at next intersection", "priority road", "yield", "stop", "no vehicles", "trucks prohibited",
    "no entry", "general caution", "dangerous curve left", "dangerous curve right", "double curve", "bumpy road",
    "slippery road", "road narrows on the right", "road work", "traffic signals", "pedestrians", "children crossing",
    "bicycles crossing", "beware of ice and snow", "wild animals crossing", "end of all limits", "turn right ahead",
    "turn left ahead", "ahead only", "go straight or right", "go straight or left", "keep right", "keep left",
    "roundabout mandatory", "end of no passing", "end of no passing for trucks",
]


def gtsrb_root() -> Path:
    return DATA / "gtsrb"


def load_gtsrb() -> tuple[Split, Split]:
    """Official GTSRB: 39,209 training images (43 classes) and 12,630 test images with ground truth."""
    r = gtsrb_root() / "GTSRB"
    tr_p, tr_y, tr_b = [], [], []
    for d in sorted((r / "Final_Training" / "Images").iterdir()):
        if not d.is_dir():
            continue
        with open(d / f"GT-{d.name}.csv", newline="") as f:
            for row in csv.DictReader(f, delimiter=";"):
                tr_p.append(d / row["Filename"])
                tr_y.append(int(row["ClassId"]))
                tr_b.append((int(row["Roi.X1"]), int(row["Roi.Y1"]), int(row["Roi.X2"]), int(row["Roi.Y2"])))
    te_p, te_y, te_b = [], [], []
    with open(gtsrb_root() / "GT-final_test.csv", newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            te_p.append(r / "Final_Test" / "Images" / row["Filename"])
            te_y.append(int(row["ClassId"]))
            te_b.append((int(row["Roi.X1"]), int(row["Roi.Y1"]), int(row["Roi.X2"]), int(row["Roi.Y2"])))
    return (Split(tr_p, np.array(tr_y, dtype=np.int64), tr_b), Split(te_p, np.array(te_y, dtype=np.int64), te_b))


def gtsrb_track_ids(paths: list[Path]) -> np.ndarray:
    """GTSRB training images come in tracks of ~30 frames of one physical sign (file name prefix).

    Validation splits must be made by track, otherwise near-duplicate frames leak between train and val.
    """
    return np.array([int(p.parent.name) * 100_000 + int(p.name.split("_")[0]) for p in paths], dtype=np.int64)


# ---------------------------------------------------------------- uint8 cache
def _load_square(path: Path, size: int, box=None, margin: float = 0.0) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if box is not None:
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        m = int(round(margin * max(w, h)))
        im = im.crop((max(0, x1 - m), max(0, y1 - m), min(im.width, x2 + m + 1), min(im.height, y2 + m + 1)))
        return np.asarray(im.resize((size, size), Image.BILINEAR), dtype=np.uint8)
    # resize the shorter side to `size`, centre-crop a square
    s = size / min(im.size)
    im = im.resize((max(size, round(im.width * s)), max(size, round(im.height * s))), Image.BILINEAR)
    left, top = (im.width - size) // 2, (im.height - size) // 2
    return np.asarray(im.crop((left, top, left + size, top + size)), dtype=np.uint8)


def build_cache(name: str, split: Split, size: int, use_box: bool = False, margin: float = 0.0, threads: int = 8) -> np.ndarray:
    """Decode and resize every image once into an (N, size, size, 3) uint8 array saved as .npy.

    This is the 'RAM cache' in the throughput study: training then never touches JPEG decoding again.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"{name}_{size}.npy"
    if out.exists():
        arr = np.load(out, mmap_mode="r")
        if arr.shape[0] == len(split.paths):
            return np.ascontiguousarray(arr)
    boxes = split.boxes if use_box else [None] * len(split.paths)
    with ThreadPoolExecutor(threads) as ex:
        imgs = list(ex.map(lambda pb: _load_square(pb[0], size, pb[1], margin), zip(split.paths, boxes)))
    arr = np.stack(imgs)
    np.save(out, arr)
    return arr
