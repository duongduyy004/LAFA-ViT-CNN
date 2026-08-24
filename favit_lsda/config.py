from __future__ import annotations

from dataclasses import dataclass
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


@dataclass(frozen=True)
class BranchConfig:
    enable_srm: bool
    enable_fft: bool
    srm_backbone: str
    fft_backbone: str
    forensic_pretrained: bool

    @property
    def enabled_branches(self) -> tuple[str, ...]:
        return (
            "rgb",
            *(("srm",) if self.enable_srm else ()),
            *(("fft",) if self.enable_fft else ()),
        )


def resolve_branch_config(model_config: dict[str, Any]) -> BranchConfig:
    legacy = sorted({"artifact_mode", "cnn_in_channels"} & model_config.keys())
    if legacy:
        raise ValueError(
            f"obsolete model field(s) {legacy}; use enable_srm_branch and "
            "enable_fft_branch"
        )
    value = BranchConfig(
        enable_srm=bool(model_config.get("enable_srm_branch", False)),
        enable_fft=bool(model_config.get("enable_fft_branch", False)),
        srm_backbone=str(model_config.get("srm_backbone", "xception")),
        fft_backbone=str(
            model_config.get("fft_backbone", "mobilenetv3_small_100")
        ),
        forensic_pretrained=bool(model_config.get("forensic_pretrained", True)),
    )
    if value.srm_backbone != "xception":
        raise ValueError(f"unsupported srm_backbone: {value.srm_backbone!r}")
    if value.fft_backbone != "mobilenetv3_small_100":
        raise ValueError(f"unsupported fft_backbone: {value.fft_backbone!r}")
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def validate_model_config(model_config: dict[str, Any]) -> None:
    resolve_branch_config(model_config)


def build_model_from_config(model_config: dict[str, Any], pretrained: bool | None = None):
    from .model import create_favit_lsda

    branch_config = resolve_branch_config(model_config)
    encoder_pretrained = model_config.get("pretrained", True) if pretrained is None else pretrained
    forensic_pretrained = branch_config.forensic_pretrained
    if pretrained is False:
        forensic_pretrained = False

    return create_favit_lsda(
        model_name=model_config["backbone"],
        pretrained=encoder_pretrained,
        num_classes=model_config.get("num_classes", 2),
        forgery_methods=model_config.get(
            "forgery_methods",
            ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"],
        ),
        gam_reduction=model_config.get("gam_reduction", 2),
        inject_layers=model_config.get("inject_layers", [0, 3, 6]),
        train_backbone_norms=model_config.get("train_backbone_norms", True),
        train_cls_token=model_config.get("train_cls_token", True),
        latent_transforms=model_config.get(
            "latent_transforms",
            [
                "hard_interpolation",
                "centrifugal",
                "gaussian",
                "rotation",
                "difference",
            ],
        ),
        max_rotation_degrees=model_config.get("max_rotation_degrees", 30.0),
        latent_noise_std=model_config.get("latent_noise_std", 1.0),
        mixup_concentration_min=model_config.get("mixup_concentration_min", 0.5),
        mixup_concentration_max=model_config.get("mixup_concentration_max", 2.0),
        feature_dropout=model_config.get("feature_dropout", 0.0),
        unfreeze_last_blocks=model_config.get("unfreeze_last_blocks", 0),
        domain_adversarial_strength=model_config.get(
            "domain_adversarial_strength", 1.0
        ),
        enable_srm_branch=branch_config.enable_srm,
        enable_fft_branch=branch_config.enable_fft,
        srm_backbone=branch_config.srm_backbone,
        fft_backbone=branch_config.fft_backbone,
        forensic_pretrained=forensic_pretrained,
    )
