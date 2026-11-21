"""E3b. The 13x slowdown that showed up in the first sweep run, isolated and measured.

Same model (ConvNeXt-T), same batch (64 x 224 x 224), forward pass only, in four settings:
precision fp32 or bf16 autocast, cuDNN autotuning (torch.backends.cudnn.benchmark) off or on.
Each setting runs in a fresh process so a previous autotune cannot leak into the next one.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision.common import log_result  # noqa: E402

CHILD = r"""
import os, sys, time, json, torch, timm
os.environ.setdefault("HF_HUB_OFFLINE", "1")
prec, bench = sys.argv[1], sys.argv[2] == "1"
torch.backends.cudnn.benchmark = bench
torch.backends.cuda.matmul.allow_tf32 = bench
torch.backends.cudnn.allow_tf32 = bench
m = timm.create_model("convnext_tiny.in12k", pretrained=True, num_classes=0).cuda().eval()
x = torch.randn(64, 3, 224, 224, device="cuda")
def fwd():
    with torch.no_grad():
        if prec == "bf16":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return m(x)
        return m(x)
for _ in range(3): fwd()
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(10): fwd()
torch.cuda.synchronize()
print(json.dumps({"ms_per_batch": (time.perf_counter() - t) / 10 * 1000}))
"""


def main() -> None:
    for prec in ("fp32", "bf16"):
        for bench in ("0", "1"):
            out = subprocess.run([sys.executable, "-c", CHILD, prec, bench], capture_output=True, text=True, check=True).stdout
            ms = json.loads(out.strip().splitlines()[-1])["ms_per_batch"]
            label = f"{prec}, autotune {'on (+TF32)' if bench == '1' else 'off'}"
            log_result("cudnn_autotune", model="convnext_tiny.in12k", batch=64, precision=prec, autotune=bench == "1",
                       label=label, ms_per_batch=round(ms, 1), img_per_s=round(64 / ms * 1000, 1))
            print(f"{label:28s} {ms:8.1f} ms/batch  {64 / ms * 1000:8.1f} img/s", flush=True)


if __name__ == "__main__":
    main()
