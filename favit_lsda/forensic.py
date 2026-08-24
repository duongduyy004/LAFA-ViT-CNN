from __future__ import annotations

import timm
import torch
from torch import Tensor, nn


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
        self.project = nn.Sequential(
            nn.Linear(int(self.backbone.num_features), embed_dim),
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
