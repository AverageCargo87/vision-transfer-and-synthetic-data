"""E4. Deployment check: what do the candidates cost on a CPU-only machine with no GPU?

For each model: parameters, fp32 weight size, single-image latency (median of 50 runs after warm-up)
and batched throughput with 8 threads, then the time to classify the whole 5,794-image CUB test split.
Accuracy comes from E1/E2 so the result is an accuracy versus latency table.
"""
from __future__ import annotations

import argparse
import io
import os
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision.common import log_result  # noqa: E402
from vision.models import SHORT, load_backbone  # noqa: E402

DEFAULT = [
    "mobilenetv2_100.ra_in1k", "tf_efficientnetv2_s.in21k_ft_in1k", "convnext_tiny.in12k", "eva02_small_patch14_336.mim_in22k_ft_in1k",
    "vit_small_patch14_reg4_dinov2.lvd142m", "vit_base_patch14_reg4_dinov2.lvd142m", "vit_base_patch16_clip_224.openai_ft_in12k_in1k",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    for name in args.models:
        bb = load_backbone(name, num_classes=200, device="cpu")
        m = bb.model.eval()
        buf = io.BytesIO()
        torch.save(m.state_dict(), buf)
        size_mb = buf.tell() / 2**20
        x1 = torch.randn(1, 3, 224, 224)
        x32 = torch.randn(32, 3, 224, 224)
        with torch.inference_mode():
            for _ in range(5):
                m(x1)
            lat = []
            for _ in range(50):
                t = time.perf_counter()
                m(x1)
                lat.append((time.perf_counter() - t) * 1000)
            m(x32)
            t = time.perf_counter()
            for _ in range(3):
                m(x32)
            ips = 96 / (time.perf_counter() - t)
        rec = log_result("cpu_budget", model=name, label=SHORT[name], params_m=bb.params_m, weights_mb=round(size_mb, 1),
                         threads=args.threads, latency_ms_b1=round(statistics.median(lat), 1), batch32_img_per_s=round(ips, 1),
                         cub_test_seconds=round(5794 / ips, 0))
        print(f"{SHORT[name]:38s} {rec['latency_ms_b1']:7.1f} ms/img  {ips:6.1f} img/s  {size_mb:6.1f} MB  test split in {rec['cub_test_seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
