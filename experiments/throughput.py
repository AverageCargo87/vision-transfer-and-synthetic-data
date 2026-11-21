"""E3. Where does a Windows training loop lose its time? Same model, same step, three ways to feed it.

  files, 0 workers : PIL decode + torchvision transforms in the main process
  files, N workers : the same in N DataLoader worker processes (Windows spawns them)
  RAM cache + GPU  : decode once into a uint8 array, then slice, copy and augment on the GPU

Two workloads, because the answer depends on how heavy the GPU step is:
  cub    ConvNeXt-T at 224 px, batch 32 (a heavy step: the GPU is the bottleneck)
  gtsrb  ResNet-50 at 64 px on traffic-sign crops, batch 256 (a light step: input can be the bottleneck)
Each step is a full training step (bf16 forward, backward, AdamW). GPU utilisation is sampled by
nvidia-smi every 200 ms during the measured window.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision.common import log_result, seed_all  # noqa: E402
from vision.datasets import build_cache, load_cub, load_gtsrb  # noqa: E402
from vision.gpu import color_jitter, hflip_random, normalize, random_resized_crop, to_gpu_batch  # noqa: E402
from vision.models import load_backbone  # noqa: E402


class FileDataset(torch.utils.data.Dataset):
    def __init__(self, paths, labels, mean, std):
        from torchvision import transforms as T

        self.paths, self.labels = paths, labels
        self.tf = T.Compose([
            T.RandomResizedCrop(224, scale=(0.35, 1.0)), T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(), T.Normalize(mean, std),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image

        return self.tf(Image.open(self.paths[i]).convert("RGB")), int(self.labels[i])


class SignDataset(torch.utils.data.Dataset):
    """GTSRB PPM files: ROI crop with a 10% margin, resize to 64 px, colour jitter, normalise."""

    def __init__(self, split):
        from torchvision import transforms as T

        self.split = split
        self.tf = T.Compose([T.RandomResizedCrop(64, scale=(0.7, 1.0), ratio=(0.9, 1.1)), T.ColorJitter(0.3, 0.3, 0.3),
                             T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def __len__(self):
        return len(self.split.paths)

    def __getitem__(self, i):
        from PIL import Image

        im = Image.open(self.split.paths[i]).convert("RGB")
        x1, y1, x2, y2 = self.split.boxes[i]
        m = int(round(0.1 * max(x2 - x1, y2 - y1)))
        im = im.crop((max(0, x1 - m), max(0, y1 - m), min(im.width, x2 + m + 1), min(im.height, y2 + m + 1))).resize((64, 64))
        return self.tf(im), int(self.split.labels[i])


class GpuSampler:
    def __init__(self):
        self.samples, self._stop = [], False

    def start(self):
        self.p = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", "-lms", "200"],
            stdout=subprocess.PIPE, text=True)
        self.t = threading.Thread(target=self._read, daemon=True)
        self.t.start()

    def _read(self):
        for line in self.p.stdout:
            if self._stop:
                break
            try:
                self.samples.append(float(line.strip()))
            except ValueError:
                pass

    def stop(self) -> float:
        self._stop = True
        self.p.terminate()
        return float(np.mean(self.samples)) if self.samples else float("nan")


def run(task: str, mode: str, workers: int, steps: int, warmup: int, bs: int) -> dict:
    seed_all(0)
    if task == "cub":
        train, _, _ = load_cub()
        bb = load_backbone("convnext_tiny.in12k", num_classes=200)
        model, mean, std, size = bb.model, bb.mean, bb.std, 224
    else:
        import torchvision

        from vision.models import configure_cuda

        configure_cuda()
        train, _ = load_gtsrb()
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = torch.nn.Linear(model.fc.in_features, 43)
        model = model.cuda()
        mean, std, size = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 64
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    y_all = torch.tensor(train.labels, device="cuda")

    def step(x, y):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(x).float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    if mode == "files":
        ds = FileDataset(train.paths, train.labels, mean, std) if task == "cub" else SignDataset(train)
        dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, num_workers=workers, pin_memory=True,
                                         persistent_workers=workers > 0, drop_last=True)

        def batches():
            while True:
                for x, y in dl:
                    yield x.to("cuda", non_blocking=True), y.to("cuda", non_blocking=True)
    else:
        arr = build_cache("cub_train", train, 256) if task == "cub" else build_cache("gtsrb_train_roi", train, 64, use_box=True, margin=0.1)
        gen = torch.Generator(device="cuda").manual_seed(0)
        rng = np.random.default_rng(0)

        def batches():
            while True:
                idx = np.sort(rng.choice(len(arr), bs, replace=False))
                x = to_gpu_batch(arr, idx)
                if task == "cub":
                    x = color_jitter(hflip_random(random_resized_crop(x, size, gen=gen), gen=gen), gen=gen)
                else:
                    x = color_jitter(random_resized_crop(x, size, scale=(0.7, 1.0), ratio=(0.9, 1.1), gen=gen), 0.3, 0.3, 0.3, gen=gen)
                yield normalize(x, mean, std), y_all[idx]

    it = batches()
    for _ in range(warmup):
        step(*next(it))
    torch.cuda.synchronize()
    gs = GpuSampler()
    gs.start()
    t0 = time.perf_counter()
    for _ in range(steps):
        step(*next(it))
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    util = gs.stop()
    return {"img_per_s": round(steps * bs / dt, 1), "gpu_util_pct": round(util, 1), "seconds": round(dt, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--task", choices=["cub", "gtsrb"], default="cub")
    args = ap.parse_args()
    bs = args.bs or (32 if args.task == "cub" else 256)
    fmt = "JPEG" if args.task == "cub" else "PPM"
    for mode, workers in (("files", 0), ("files", 2), ("files", 4), ("files", 8), ("cache", 0)):
        r = run(args.task, mode, workers, args.steps, args.warmup, bs)
        label = "RAM cache + GPU augmentation" if mode == "cache" else f"{fmt} files, {workers} DataLoader workers"
        log_result("throughput", task=args.task, loader=label, mode=mode, workers=workers, batch=bs, steps=args.steps,
                   model="convnext_tiny.in12k" if args.task == "cub" else "resnet50 (64 px)", **r)
        print(f"{label:40s} {r['img_per_s']:7.1f} img/s  GPU {r['gpu_util_pct']:.0f}%", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
