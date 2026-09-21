"""Three-branch timm baselines, independent of FA-ViT/LSDA."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import timm
import torch
from torch import Tensor, nn

from .forensic import (
    _PIPELINE_MEAN,
    _PIPELINE_STD,
    _backbone_normalization,
    _backbone_output_width,
)


SUPPORTED_BASELINE_BACKBONES = frozenset(
    {
        "tf_efficientnet_b4.ns_jft_in1k",
        "resnet50",
        "vit_base_patch16_224.augreg_in21k",
        "swin_tiny_patch4_window7_224.ms_in1k",
        "xception",
    }
)
BASELINE_BRANCHES = ("rgb", "srm", "fft")
BASELINE_FUSION = "concat"
BASELINE_BACKBONE_SHARING = "independent"
BASELINE_ARCHITECTURE = "rgb_srm_fft_timm_concat_baseline"
BASELINE_FORMAT_VERSION = 2


def _resolve_baseline_branches(model_config: dict[str, Any]) -> tuple[str, ...]:
    configured = model_config.get("branches", BASELINE_BRANCHES)
    if not isinstance(configured, Sequence) or isinstance(configured, (str, bytes)):
        raise ValueError("baseline model.branches must be a sequence")
    branches = tuple(configured)
    if branches != BASELINE_BRANCHES:
        raise ValueError(
            "baseline model.branches must be exactly "
            f"{list(BASELINE_BRANCHES)!r}, got {list(branches)!r}"
        )
    return branches


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
    _resolve_baseline_branches(model_config)
    sharing = model_config.get("backbone_sharing", BASELINE_BACKBONE_SHARING)
    if sharing != BASELINE_BACKBONE_SHARING:
        raise ValueError(
            "baseline model.backbone_sharing must be "
            f"{BASELINE_BACKBONE_SHARING!r}, got {sharing!r}"
        )
    fusion = model_config.get("fusion", BASELINE_FUSION)
    if fusion != BASELINE_FUSION:
        raise ValueError(
            f"baseline model.fusion must be {BASELINE_FUSION!r}, got {fusion!r}"
        )


class TimmThreeBranchClassifier(nn.Module):
    """Encode RGB, SRM and FFT independently, then classify concatenated features."""

    def __init__(self, model_name: str, pretrained: bool = True) -> None:
        super().__init__()
        self.model_name = model_name
        self.enabled_branches = BASELINE_BRANCHES
        self.fusion_name = BASELINE_FUSION
        self.backbone_sharing = BASELINE_BACKBONE_SHARING
        self.backbones = nn.ModuleDict(
            {
                branch: timm.create_model(
                    model_name,
                    pretrained=pretrained,
                    num_classes=0,
                    global_pool="avg",
                )
                for branch in self.enabled_branches
            }
        )
        widths = {
            branch: _backbone_output_width(backbone, model_name)
            for branch, backbone in self.backbones.items()
        }
        if len(set(widths.values())) != 1:
            raise ValueError(
                f"{model_name} branches expose different output widths: {widths!r}"
            )
        self.feature_dim = next(iter(widths.values()))
        self.head = nn.Linear(self.feature_dim * len(self.enabled_branches), 2)

        # FaceTransform emits every branch in [-1, 1] using mean/std 0.5.
        # Keep normalization tied to the architecture even when pretrained
        # downloads are disabled while restoring a checkpoint.
        reference_backbone = self.backbones[self.enabled_branches[0]]
        mean, std = _backbone_normalization(reference_backbone)
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

    def _validate_inputs(self, inputs: Mapping[str, Tensor]) -> None:
        observed = tuple(inputs)
        if observed != self.enabled_branches:
            raise ValueError(
                f"{self.model_name} expects branches {self.enabled_branches}, "
                f"got {observed}"
            )
        reference_shape: tuple[int, ...] | None = None
        for branch in self.enabled_branches:
            images = inputs[branch]
            if images.ndim != 4 or images.shape[1] != 3:
                raise ValueError(
                    f"{self.model_name} branch {branch!r} expects [B, 3, H, W], "
                    f"got {tuple(images.shape)}"
                )
            if not images.is_floating_point() or not torch.isfinite(images).all():
                raise ValueError(
                    f"{self.model_name} branch {branch!r} must be finite floating point"
                )
            if reference_shape is None:
                reference_shape = tuple(images.shape)
            elif tuple(images.shape) != reference_shape:
                raise ValueError(
                    f"{self.model_name} branches must have the same shape; "
                    f"rgb={reference_shape}, {branch}={tuple(images.shape)}"
                )

    def forward(self, inputs: Mapping[str, Tensor]) -> Tensor:
        if not isinstance(inputs, Mapping):
            raise ValueError(
                f"{self.model_name} expects an RGB/SRM/FFT input mapping"
            )
        self._validate_inputs(inputs)
        features = []
        for branch in self.enabled_branches:
            images = inputs[branch] * self._renorm_scale + self._renorm_shift
            branch_features = self.backbones[branch](images)
            if branch_features.ndim != 2 or branch_features.shape[1] != self.feature_dim:
                raise RuntimeError(
                    f"{self.model_name} branch {branch!r} must return "
                    f"[B, {self.feature_dim}] features, got {tuple(branch_features.shape)}"
                )
            features.append(branch_features)
        logits = self.head(torch.cat(features, dim=1))
        if logits.ndim != 2 or logits.shape[1] != 2:
            raise RuntimeError(
                f"{self.model_name} must return [B, 2] logits, got {tuple(logits.shape)}"
            )
        return logits


# Preserve the public import used by earlier callers while changing its
# semantics to the new three-branch baseline.
TimmBinaryClassifier = TimmThreeBranchClassifier


def build_baseline_model(
    model_config: dict[str, Any], pretrained: bool | None = None
) -> TimmThreeBranchClassifier:
    validate_baseline_model_config(model_config)
    initialize_pretrained = (
        model_config.get("pretrained", True) if pretrained is None else pretrained
    )
    return TimmThreeBranchClassifier(
        str(model_config["backbone"]), pretrained=bool(initialize_pretrained)
    )


def baseline_checkpoint_metadata(
    model: TimmThreeBranchClassifier,
) -> dict[str, Any]:
    return {
        "format_version": BASELINE_FORMAT_VERSION,
        "architecture": BASELINE_ARCHITECTURE,
        "backbone": model.model_name,
        "enabled_branches": list(model.enabled_branches),
        "backbone_sharing": model.backbone_sharing,
        "fusion": model.fusion_name,
    }


def validate_baseline_checkpoint(
    checkpoint: dict[str, Any], model_config: dict[str, Any], path: Path
) -> None:
    validate_baseline_model_config(model_config)
    expected = {
        "format_version": BASELINE_FORMAT_VERSION,
        "architecture": BASELINE_ARCHITECTURE,
        "backbone": model_config["backbone"],
        "enabled_branches": list(_resolve_baseline_branches(model_config)),
        "backbone_sharing": BASELINE_BACKBONE_SHARING,
        "fusion": BASELINE_FUSION,
    }
    actual = {key: checkpoint.get(key) for key in expected}
    if actual != expected:
        raise ValueError(
            f"baseline checkpoint/config mismatch: checkpoint={actual!r}, "
            f"expected={expected!r}, path={path}"
        )
