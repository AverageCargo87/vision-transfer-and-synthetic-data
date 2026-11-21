"""Backbone zoo used in the sweep. All weights are public timm checkpoints from the Hugging Face hub."""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch

# Families chosen to span the design space a practitioner actually picks from:
# mobile CNN, modern CNN at two pretraining scales, MIM ViT, self-supervised ViT, CLIP ViT.
BACKBONES = [
    "mobilenetv2_100.ra_in1k",
    "tf_efficientnetv2_s.in21k_ft_in1k",
    "convnext_tiny.in12k_ft_in1k",
    "convnext_tiny.in12k",
    "convnext_small.in12k",
    "eva02_small_patch14_336.mim_in22k_ft_in1k",
    "beitv2_base_patch16_224.in1k_ft_in22k",
    "vit_small_patch14_reg4_dinov2.lvd142m",
    "vit_base_patch14_reg4_dinov2.lvd142m",
    "vit_base_patch16_clip_224.openai_ft_in12k_in1k",
    "vit_base_patch16_clip_224.laion2b_ft_in12k",
]

SHORT = {
    "mobilenetv2_100.ra_in1k": "MobileNetV2 (IN-1k)",
    "tf_efficientnetv2_s.in21k_ft_in1k": "EfficientNetV2-S (IN-21k)",
    "convnext_tiny.in12k_ft_in1k": "ConvNeXt-T (IN-12k, ft 1k)",
    "convnext_tiny.in12k": "ConvNeXt-T (IN-12k)",
    "convnext_small.in12k": "ConvNeXt-S (IN-12k)",
    "eva02_small_patch14_336.mim_in22k_ft_in1k": "EVA-02-S (MIM IN-22k)",
    "beitv2_base_patch16_224.in1k_ft_in22k": "BEiT-v2-B (IN-22k)",
    "vit_small_patch14_reg4_dinov2.lvd142m": "DINOv2-S/14 reg",
    "vit_base_patch14_reg4_dinov2.lvd142m": "DINOv2-B/14 reg",
    "vit_base_patch16_clip_224.openai_ft_in12k_in1k": "CLIP ViT-B/16 (OpenAI, ft 12k+1k)",
    "vit_base_patch16_clip_224.laion2b_ft_in12k": "CLIP ViT-B/16 (LAION-2B, ft 12k)",
}


@dataclass
class Backbone:
    name: str
    model: torch.nn.Module
    mean: tuple
    std: tuple
    img_size: int
    params_m: float


def configure_cuda() -> None:
    """Let cuDNN autotune and allow TF32 matmuls. Measured in experiments/cudnn_autotune.py: worth about
    1.3x for fp32 and a few percent under bf16 autocast on an RTX 3060."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def load_backbone(name: str, img_size: int = 224, num_classes: int = 0, device: str = "cuda") -> Backbone:
    """Load a pretrained timm model. num_classes=0 returns pooled features; img_size is forced for ViTs."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # weights must already be in the local HF cache
    import timm

    configure_cuda()

    kw = {"pretrained": True, "num_classes": num_classes}
    if any(k in name for k in ("vit_", "eva02", "beit")):
        kw["img_size"] = img_size
    m = timm.create_model(name, **kw)
    cfg = timm.data.resolve_data_config({}, model=m)
    params = sum(p.numel() for p in m.parameters()) / 1e6
    return Backbone(name, m.to(device).eval(), tuple(cfg["mean"]), tuple(cfg["std"]), img_size, round(params, 1))
