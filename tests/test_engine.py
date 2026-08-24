"""Integration coverage for mapping-based train/evaluation wiring."""

from __future__ import annotations

import csv
import math

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from favit_lsda.config import resolve_branch_config
from favit_lsda.data import FaceTransform, FrameFaceDataset, GroupedForgeryDataset
from favit_lsda.engine import evaluate_at_level, train_one_epoch
from favit_lsda.losses import FineGrainedAdaptiveLoss
from favit_lsda.model import create_favit_lsda
from train import build_optimizer

METHODS = ("DF", "F2F")


class _FakeForensicEncoder(nn.Module):
    def __init__(self, model_name, embed_dim, pretrained, dropout):
        super().__init__()
        assert pretrained is False
        self.model_name = model_name
        self.backbone = nn.Conv2d(3, 3, kernel_size=1, bias=False)
        self.project = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, images):
        pooled = self.backbone(images).mean(dim=(-2, -1))
        return self.project(pooled)


def _write_fixture(tmp_path):
    """Two complete LSDA groups plus a two-class frame manifest."""
    reals = ("real_a.jpg", "real_b.jpg")
    for name in reals:
        Image.new("RGB", (24, 24), "white").save(tmp_path / name)
    pair_rows = []
    for real in reals:
        for method in METHODS:
            fake = f"{real[:-4]}_{method}.jpg"
            Image.new("RGB", (24, 24), "red").save(tmp_path / fake)
            pair_rows.append({"fake_path": fake, "real_path": real, "method": method})
    pairs = tmp_path / "pairs.csv"
    with pairs.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fake_path", "real_path", "method"])
        writer.writeheader()
        writer.writerows(pair_rows)

    frames = tmp_path / "frames.csv"
    with frames.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "video_id"])
        writer.writeheader()
        writer.writerows(
            [
                {"path": "real_a.jpg", "label": "0", "video_id": "v_real"},
                {"path": "real_b.jpg", "label": "0", "video_id": "v_real"},
                {"path": "real_a_DF.jpg", "label": "1", "video_id": "v_fake"},
                {"path": "real_b_F2F.jpg", "label": "1", "video_id": "v_fake"},
            ]
        )
    return pairs, frames


def test_engine_loops_forward_enabled_branch_mappings(monkeypatch, tmp_path):
    pairs, frames = _write_fixture(tmp_path)
    monkeypatch.setattr("favit_lsda.model.ProjectedForensicEncoder", _FakeForensicEncoder)
    device = torch.device("cpu")
    transform = FaceTransform(224, enable_srm=True, enable_fft=True)
    model = create_favit_lsda(
        model_name="vit_tiny_patch16_224",
        pretrained=False,
        forgery_methods=METHODS,
        train_backbone_norms=False,
        train_cls_token=False,
        enable_srm_branch=True,
        enable_fft_branch=True,
        forensic_pretrained=False,
    ).to(device)

    train_loader = DataLoader(
        GroupedForgeryDataset(pairs, tmp_path, transform, METHODS),
        batch_size=2,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
    )
    train_metrics = train_one_epoch(
        model,
        train_loader,
        optimizer,
        FineGrainedAdaptiveLoss(),
        {"binary": 1.0, "domain": 1.0, "invariance": 1.0, "distill": 1.0, "fal": 1.0},
        device,
    )
    assert math.isfinite(train_metrics["loss"])
    for key in ("binary", "domain", "invariance", "distill", "fal", "binary_accuracy"):
        assert math.isfinite(train_metrics[key])

    eval_loader = DataLoader(
        FrameFaceDataset(frames, tmp_path, transform),
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    metrics = evaluate_at_level(model, eval_loader, device, level="frame")
    assert metrics["level"] == "frame"
    assert metrics["num_frames"] == 4
    assert 0.0 <= metrics["auc"] <= 1.0
    assert math.isfinite(metrics["accuracy"])


def test_optimizer_reduces_learning_rate_for_every_pretrained_backbone(monkeypatch):
    monkeypatch.setattr("favit_lsda.model.ProjectedForensicEncoder", _FakeForensicEncoder)
    model = create_favit_lsda(
        model_name="vit_tiny_patch16_224",
        pretrained=False,
        forgery_methods=METHODS,
        train_backbone_norms=False,
        train_cls_token=False,
        enable_srm_branch=True,
        enable_fft_branch=True,
        forensic_pretrained=False,
    )
    optimizer = build_optimizer(
        model,
        {"learning_rate": 1e-3, "backbone_lr_multiplier": 0.2},
    )
    rates_by_name = {
        name: group["lr"]
        for group in optimizer.param_groups
        for parameter in group["params"]
        for name, candidate in model.named_parameters()
        if candidate is parameter
    }
    assert rates_by_name["srm_encoder.backbone.weight"] == 2e-4
    assert rates_by_name["fft_encoder.backbone.weight"] == 2e-4
    assert rates_by_name["srm_encoder.project.0.weight"] == 1e-3
    assert rates_by_name["fft_encoder.project.0.weight"] == 1e-3
    assert rates_by_name["late_fusion.0.weight"] == 1e-3


def test_branch_config_order_is_the_transform_contract():
    assert resolve_branch_config(
        {"enable_srm_branch": True, "enable_fft_branch": True}
    ).enabled_branches == ("rgb", "srm", "fft")
