from __future__ import annotations

import timm
import torch
from torch import Tensor, nn


#: Mean/std the data pipeline already normalizes every branch input to
#: (RGB via ``TF.normalize``, SRM/FFT via ``_normalize_artifact``), i.e. the
#: input arrives in [-1, 1] under a 0.5/0.5 assumption.
_PIPELINE_MEAN = (0.5, 0.5, 0.5)
_PIPELINE_STD = (0.5, 0.5, 0.5)


def _backbone_output_width(backbone: nn.Module, model_name: str) -> int:
    """Resolve the width returned by a pooled timm backbone without a probe."""
    for attribute in ("head_hidden_size", "num_features"):
        value = getattr(backbone, attribute, None)
        if value is None:
            continue
        try:
            width = int(value)
        except (TypeError, ValueError):
            continue
        if width > 0:
            return width
    raise ValueError(
        f"{model_name} backbone must expose a positive head_hidden_size or num_features"
    )


def _backbone_normalization(backbone: nn.Module) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Resolve a timm backbone's expected mean/std, defaulting to the pipeline's."""
    cfg = getattr(backbone, "pretrained_cfg", None) or getattr(backbone, "default_cfg", None)
    if not cfg:
        return _PIPELINE_MEAN, _PIPELINE_STD
    mean = cfg.get("mean") if hasattr(cfg, "get") else getattr(cfg, "mean", None)
    std = cfg.get("std") if hasattr(cfg, "get") else getattr(cfg, "std", None)
    if not mean or not std or len(mean) != 3 or len(std) != 3:
        return _PIPELINE_MEAN, _PIPELINE_STD
    mean = tuple(float(value) for value in mean)
    std = tuple(float(value) for value in std)
    if any(value <= 0.0 for value in std):
        return _PIPELINE_MEAN, _PIPELINE_STD
    return mean, std


class ProjectedForensicEncoder(nn.Module):
    """Adapt a pooled forensic backbone to the FA-ViT embedding width."""

    def __init__(
        self,
        model_name: str,
        embed_dim: int,
        pretrained: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        output_width = _backbone_output_width(self.backbone, model_name)
        self.project = nn.Sequential(
            nn.Linear(output_width, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Re-normalize from the pipeline's [-1, 1] (0.5/0.5) convention to
        # whatever mean/std this backbone was pretrained on, so a backbone
        # whose pretrained stats differ (e.g. ImageNet mean/std) isn't fed
        # mismatched inputs. x_backbone = x_pipeline * scale + shift.
        # Only meaningful with pretrained weights: a randomly initialized
        # backbone has no expected input distribution, so the pipeline's own
        # convention is kept and the re-normalization degenerates to identity.
        mean, std = (
            _backbone_normalization(self.backbone)
            if pretrained
            else (_PIPELINE_MEAN, _PIPELINE_STD)
        )
        scale = tuple(_PIPELINE_STD[i] / std[i] for i in range(3))
        shift = tuple(
            (_PIPELINE_MEAN[i] - mean[i]) / std[i] for i in range(3)
        )
        # Derived from the backbone's config, not learned: keeping them out of
        # ``state_dict`` leaves the checkpoint contract unchanged.
        self.register_buffer(
            "_renorm_scale", torch.tensor(scale).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "_renorm_shift", torch.tensor(shift).view(1, 3, 1, 1), persistent=False
        )

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                f"{self.model_name} expects [B, 3, H, W], got {tuple(images.shape)}"
            )
        if not images.is_floating_point() or not torch.isfinite(images).all():
            raise ValueError(f"{self.model_name} input must be finite floating point")
        images = images * self._renorm_scale + self._renorm_shift
        features = self.backbone(images)
        if features.ndim != 2:
            raise RuntimeError(
                f"{self.model_name} pooled output must be rank 2, got {features.shape}"
            )
        return self.project(features)
