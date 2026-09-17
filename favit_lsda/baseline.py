"""Conventional RGB-only timm baselines, independent of FA-ViT/LSDA."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import timm
import torch
from torch import Tensor, nn

from .forensic import _PIPELINE_MEAN, _PIPELINE_STD, _backbone_normalization


SUPPORTED_BASELINE_BACKBONES = frozenset(
    {
        "tf_efficientnet_b4.ns_jft_in1k",
        "resnet50",
        "vit_base_patch16_224.augreg_in21k",
        "swin_tiny_patch4_window7_224.ms_in1k",
        "xception",
    }
)
BASELINE_ARCHITECTURE = "rgb_timm_binary_baseline"
BASELINE_FORMAT_VERSION = 1


def validate_baseline_model_config(model_config: dict[str, Any]) -> None:
    backbone = model_config.get("backbone")
    if backbone not in SUPPORTED_BASELINE_BACKBONES:
        raise ValueError(
            f"unsupported baseline backbone: {backbone!r}; supported: "
            f"{sorted(SUPPORTED_BASELINE_BACKBONES)}"
        )
    pretrained = model_config.get("pretrained", True)
    if not isinstance(pretrained, bool):
        raise ValueError("model.pretrained must be a boolean")
    if int(model_config.get("num_classes", 2)) != 2:
        raise ValueError("baseline model.num_classes must be 2")


class TimmBinaryClassifier(nn.Module):
    """A pretrained timm classifier consuming the repository's RGB mapping."""

    def __init__(self, model_name: str, pretrained: bool = True) -> None:
        super().__init__()
        self.model_name = model_name
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=2
        )
        # FaceTransform emits [-1, 1] using mean/std 0.5. Baseline input
        # normalization remains tied to the architecture even when pretrained
        # downloads are disabled while restoring a checkpoint.
        mean, std = _backbone_normalization(self.backbone)
        scale = tuple(_PIPELINE_STD[index] / std[index] for index in range(3))
        shift = tuple(
            (_PIPELINE_MEAN[index] - mean[index]) / std[index]
            for index in range(3)
        )
        self.register_buffer(
            "_renorm_scale",
            torch.tensor(scale).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_renorm_shift",
            torch.tensor(shift).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, inputs: Mapping[str, Tensor] | Tensor) -> Tensor:
        images = inputs["rgb"] if isinstance(inputs, Mapping) else inputs
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                f"{self.model_name} expects [B, 3, H, W], got {tuple(images.shape)}"
            )
        if not images.is_floating_point() or not torch.isfinite(images).all():
            raise ValueError(f"{self.model_name} input must be finite floating point")
        logits = self.backbone(images * self._renorm_scale + self._renorm_shift)
        if logits.ndim != 2 or logits.shape[1] != 2:
            raise RuntimeError(
                f"{self.model_name} must return [B, 2] logits, got {tuple(logits.shape)}"
            )
        return logits


def build_baseline_model(
    model_config: dict[str, Any], pretrained: bool | None = None
) -> TimmBinaryClassifier:
    validate_baseline_model_config(model_config)
    initialize_pretrained = (
        model_config.get("pretrained", True) if pretrained is None else pretrained
    )
    return TimmBinaryClassifier(
        str(model_config["backbone"]), pretrained=bool(initialize_pretrained)
    )


def baseline_checkpoint_metadata(model: TimmBinaryClassifier) -> dict[str, Any]:
    return {
        "format_version": BASELINE_FORMAT_VERSION,
        "architecture": BASELINE_ARCHITECTURE,
        "backbone": model.model_name,
    }


def validate_baseline_checkpoint(
    checkpoint: dict[str, Any], model_config: dict[str, Any], path: Path
) -> None:
    validate_baseline_model_config(model_config)
    expected = {
        "format_version": BASELINE_FORMAT_VERSION,
        "architecture": BASELINE_ARCHITECTURE,
        "backbone": model_config["backbone"],
    }
    actual = {key: checkpoint.get(key) for key in expected}
    if actual != expected:
        raise ValueError(
            f"baseline checkpoint/config mismatch: checkpoint={actual!r}, "
            f"expected={expected!r}, path={path}"
        )
