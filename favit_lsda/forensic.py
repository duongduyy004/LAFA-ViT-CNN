from __future__ import annotations

import timm
import torch
from torch import Tensor, nn


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

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                f"{self.model_name} expects [B, 3, H, W], got {tuple(images.shape)}"
            )
        if not images.is_floating_point() or not torch.isfinite(images).all():
            raise ValueError(f"{self.model_name} input must be finite floating point")
        features = self.backbone(images)
        if features.ndim != 2:
            raise RuntimeError(
                f"{self.model_name} pooled output must be rank 2, got {features.shape}"
            )
        return self.project(features)
