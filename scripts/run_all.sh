#!/usr/bin/env bash
# Reproduce every number in the README, in order. One RTX 3060 (12 GB) is enough.
# Set VISION_DATA to the folder that holds cub/ and gtsrb/ (see scripts/get_data.py) and PY to a
# Python with torch (CUDA), timm and opencv. Each step appends to results/results.jsonl.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
mkdir -p logs

# E1: 11 backbones, frozen features + linear classifier, CUB-200-2011
[ "${SKIP_E1:-0}" = "1" ] || $PY -u experiments/cub_linear_probe.py | tee logs/e1_probe.log

# E2: fine-tune the best CNN and the best ViT, picked by their E1 hold-out accuracy (train split only)
read -r CNN VIT < <($PY - <<'EOF'
import json
rows = [json.loads(l) for l in open("results/results.jsonl", encoding="utf-8") if '"cub_linear_probe"' in l]
cnn = [r for r in rows if r["model"].startswith(("convnext", "tf_efficientnet", "mobilenet"))]
vit = [r for r in rows if r["model"].startswith(("vit_", "eva02", "beit"))]
print(max(cnn, key=lambda r: r["val_acc"])["model"], max(vit, key=lambda r: r["val_acc"])["model"])
EOF
)
CNN=$(printf %s "$CNN" | tr -d '\r'); VIT=$(printf %s "$VIT" | tr -d '\r')  # Windows Python prints CRLF
echo "fine-tuning $CNN and $VIT"
# naive fine-tuning (random head, 10x head LR), then LP-FT (head initialised from the linear probe, lower LR)
[ "${SKIP_NAIVE:-0}" = "1" ] || $PY -u experiments/cub_finetune.py --model "$CNN" --epochs 15 | tee logs/e2_ft_cnn.log
$PY -u experiments/cub_finetune.py --model "$CNN" --lpft --epochs 10 --lr 3e-5 --tag lpft | tee logs/e2_lpft_cnn.log
$PY -u experiments/cub_finetune.py --model "$VIT" --lpft --epochs 10 --lr 2e-5 --tag lpft | tee logs/e2_lpft_vit.log
$PY -u experiments/cub_ensemble.py --tag lpft | tee logs/e2_ensemble.log
$PY -u experiments/cub_compare.py | tee logs/e2_compare.log

# E3: input pipeline throughput and the cuDNN autotune effect
$PY -u experiments/throughput.py --task cub | tee logs/e3_throughput_cub.log
$PY -u experiments/throughput.py --task gtsrb | tee logs/e3_throughput_gtsrb.log
$PY -u experiments/cudnn_autotune.py | tee logs/e3_autotune.log

# E4: CPU-only deployment budget
$PY -u experiments/cpu_budget.py --models mobilenetv2_100.ra_in1k tf_efficientnetv2_s.in21k_ft_in1k "$CNN" "$VIT" vit_small_patch14_reg4_dinov2.lvd142m | tee logs/e4_cpu.log

# E5: GTSRB classes with zero real training images
$PY -u experiments/gtsrb_unseen.py --conditions oracle real_seen_only --seeds 0 | tee logs/e5_baselines.log
$PY -u experiments/gtsrb_unseen.py --conditions synth_unseen_only synth_all --seeds 0 1 2 | tee logs/e5_synth.log

echo ALL-EXPERIMENTS-DONE
