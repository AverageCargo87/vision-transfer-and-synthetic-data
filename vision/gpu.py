"""GPU-side batching and augmentation on top of a uint8 cache.

The loader never decodes a file: it slices the cached array, moves the batch to the GPU and does
crop / flip / colour / normalise there. That is what removes the Windows DataLoader bottleneck
measured in experiments/throughput.py.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def to_gpu_batch(arr: np.ndarray, idx: np.ndarray, device: str = "cuda") -> torch.Tensor:
    """uint8 NHWC slice -> float NCHW in [0, 1] on the device (pinned copy)."""
    x = torch.from_numpy(np.ascontiguousarray(arr[idx]))
    if device != "cpu":
        x = x.pin_memory().to(device, non_blocking=True)
    return x.permute(0, 3, 1, 2).float().div_(255.0)


def normalize(x: torch.Tensor, mean, std) -> torch.Tensor:
    m = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s


def center_resize(x: torch.Tensor, out: int, crop_pct: float = 0.875) -> torch.Tensor:
    """Centre crop `crop_pct` of the square cache and resize to `out` (eval transform)."""
    n, c, h, w = x.shape
    ch = int(round(h * crop_pct))
    top = (h - ch) // 2
    x = x[:, :, top: top + ch, top: top + ch]
    return F.interpolate(x, size=(out, out), mode="bilinear", antialias=True, align_corners=False)


def random_resized_crop(x: torch.Tensor, out: int, scale=(0.35, 1.0), ratio=(3 / 4, 4 / 3), gen: torch.Generator | None = None) -> torch.Tensor:
    """Per-sample RandomResizedCrop done with one batched grid_sample call."""
    n, _, h, w = x.shape
    dev = x.device
    area = torch.empty(n, device=dev).uniform_(*scale, generator=gen)
    logr = torch.empty(n, device=dev).uniform_(math.log(ratio[0]), math.log(ratio[1]), generator=gen)
    r = torch.exp(logr)
    cw = torch.sqrt(area * r).clamp(max=1.0)
    chh = torch.sqrt(area / r).clamp(max=1.0)
    cx = torch.rand(n, device=dev, generator=gen) * (1 - cw) + cw / 2
    cy = torch.rand(n, device=dev, generator=gen) * (1 - chh) + chh / 2
    theta = torch.zeros(n, 2, 3, device=dev)
    theta[:, 0, 0] = cw
    theta[:, 1, 1] = chh
    theta[:, 0, 2] = cx * 2 - 1
    theta[:, 1, 2] = cy * 2 - 1
    grid = F.affine_grid(theta, (n, 3, out, out), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def hflip_random(x: torch.Tensor, p: float = 0.5, gen: torch.Generator | None = None) -> torch.Tensor:
    m = torch.rand(x.shape[0], device=x.device, generator=gen) < p
    x = x.clone()
    x[m] = x[m].flip(-1)
    return x


def color_jitter(x: torch.Tensor, b: float = 0.2, c: float = 0.2, s: float = 0.2, gen: torch.Generator | None = None) -> torch.Tensor:
    n = x.shape[0]
    dev = x.device
    fb = 1 + torch.empty(n, 1, 1, 1, device=dev).uniform_(-b, b, generator=gen)
    fc = 1 + torch.empty(n, 1, 1, 1, device=dev).uniform_(-c, c, generator=gen)
    fs = 1 + torch.empty(n, 1, 1, 1, device=dev).uniform_(-s, s, generator=gen)
    x = x * fb
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * fc + mean
    gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    x = (x - gray) * fs + gray
    return x.clamp_(0, 1)
