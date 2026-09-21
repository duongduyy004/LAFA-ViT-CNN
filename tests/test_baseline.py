from __future__ import annotations

import csv
from pathlib import Path

import pytest
import timm
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from favit_lsda.baseline import (
    BASELINE_ARCHITECTURE,
    BASELINE_BACKBONE_SHARING,
    BASELINE_BRANCHES,
    BASELINE_FORMAT_VERSION,
    BASELINE_FUSION,
    SUPPORTED_BASELINE_BACKBONES,
    TimmThreeBranchClassifier,
    baseline_checkpoint_metadata,
    validate_baseline_checkpoint,
)
from favit_lsda.baseline_engine import train_baseline_one_epoch
from favit_lsda.config import load_config
from favit_lsda.data import BinaryFrameDataset, FaceTransform


CONFIG_BACKBONES = {
    "efficientnet_b4.yaml": "tf_efficientnet_b4.ns_jft_in1k",
    "resnet50.yaml": "resnet50",
    "vit_b16.yaml": "vit_base_patch16_224.augreg_in21k",
    "swin_t.yaml": "swin_tiny_patch4_window7_224.ms_in1k",
    "xception.yaml": "xception",
}


class _TinyEncoder(nn.Module):
    pretrained_cfg = {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    }
    num_features = 3

    def __init__(self):
        super().__init__()
        self.observed = None

    def forward(self, images):
        self.observed = images.detach().clone()
        return images.mean(dim=(-2, -1))


def test_baseline_configs_cover_exact_requested_timm_models():
    observed = {}
    for name, expected_backbone in CONFIG_BACKBONES.items():
        config = load_config(Path("configs/baselines") / name)
        observed[name] = config["model"]["backbone"]
        assert observed[name] == expected_backbone
        assert timm.is_model(expected_backbone)
        assert tuple(config["model"]["branches"]) == BASELINE_BRANCHES
        assert config["model"]["backbone_sharing"] == BASELINE_BACKBONE_SHARING
        assert config["model"]["fusion"] == BASELINE_FUSION
        assert config["data"]["celebdf_validation_frames"]
        assert config["data"]["ffpp_test_frames"]
    assert set(observed.values()) == SUPPORTED_BASELINE_BACKBONES


def test_baseline_encodes_and_concatenates_three_renormalized_branches(monkeypatch):
    encoders = []
    calls = []

    def fake_create(name, **kwargs):
        calls.append((name, kwargs))
        encoder = _TinyEncoder()
        encoders.append(encoder)
        return encoder

    monkeypatch.setattr("favit_lsda.baseline.timm.create_model", fake_create)
    model = TimmThreeBranchClassifier("resnet50", pretrained=False)
    images = torch.zeros(2, 3, 8, 8)
    logits = model({"rgb": images, "srm": images + 0.25, "fft": images + 0.5})

    expected_rgb = torch.tensor(
        [
            (0.5 - 0.485) / 0.229,
            (0.5 - 0.456) / 0.224,
            (0.5 - 0.406) / 0.225,
        ]
    ).view(1, 3, 1, 1).expand_as(images)
    scale = model._renorm_scale.expand_as(images)
    torch.testing.assert_close(encoders[0].observed, expected_rgb)
    torch.testing.assert_close(encoders[1].observed, expected_rgb + 0.25 * scale)
    torch.testing.assert_close(encoders[2].observed, expected_rgb + 0.5 * scale)
    assert model.head.in_features == 9
    assert logits.shape == (2, 2)
    assert calls == [
        (
            "resnet50",
            {"pretrained": False, "num_classes": 0, "global_pool": "avg"},
        )
    ] * 3


def test_baseline_rejects_missing_forensic_branch(monkeypatch):
    monkeypatch.setattr(
        "favit_lsda.baseline.timm.create_model",
        lambda *_args, **_kwargs: _TinyEncoder(),
    )
    model = TimmThreeBranchClassifier("resnet50", pretrained=False)
    with pytest.raises(ValueError, match="expects branches"):
        model({"rgb": torch.zeros(1, 3, 8, 8)})


def _write_pair_manifest(root: Path) -> Path:
    for name, color in (
        ("real.jpg", "white"),
        ("fake_df.jpg", "red"),
        ("fake_f2f.jpg", "blue"),
    ):
        Image.new("RGB", (20, 20), color).save(root / name)
    manifest = root / "pairs.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["fake_path", "real_path", "method"]
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "fake_path": "fake_df.jpg",
                    "real_path": "real.jpg",
                    "method": "Deepfakes",
                },
                {
                    "fake_path": "fake_f2f.jpg",
                    "real_path": "real.jpg",
                    "method": "Face2Face",
                },
            ]
        )
    return manifest


def test_binary_dataset_flattens_pairs_and_deduplicates_real_frames(tmp_path):
    dataset = BinaryFrameDataset(
        _write_pair_manifest(tmp_path), tmp_path, FaceTransform(16)
    )
    assert len(dataset) == 3
    assert dataset.labels.count(0) == 1
    assert dataset.labels.count(1) == 2
    inputs, label, video_id = dataset[0]
    assert tuple(inputs) == ("rgb",)
    assert inputs["rgb"].shape == (3, 16, 16)
    assert label in (0, 1)
    assert video_id


def test_baseline_train_loop_is_plain_binary_cross_entropy(tmp_path):
    dataset = BinaryFrameDataset(
        _write_pair_manifest(tmp_path), tmp_path, FaceTransform(16)
    )
    loader = DataLoader(dataset, batch_size=3, shuffle=False)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(3, 2)

        def forward(self, inputs):
            return self.head(inputs["rgb"].mean(dim=(-2, -1)))

    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metrics = train_baseline_one_epoch(
        model,
        loader,
        optimizer,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
    )
    assert metrics["num_frames"] == 3
    assert metrics["loss"] > 0.0
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_baseline_checkpoint_rejects_a_different_backbone(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "favit_lsda.baseline.timm.create_model",
        lambda *_args, **_kwargs: _TinyEncoder(),
    )
    model = TimmThreeBranchClassifier("resnet50", pretrained=False)
    checkpoint = {
        **baseline_checkpoint_metadata(model),
        "model": model.state_dict(),
    }
    assert checkpoint["architecture"] == BASELINE_ARCHITECTURE
    assert checkpoint["format_version"] == BASELINE_FORMAT_VERSION
    assert checkpoint["enabled_branches"] == list(BASELINE_BRANCHES)
    assert checkpoint["backbone_sharing"] == BASELINE_BACKBONE_SHARING
    assert checkpoint["fusion"] == BASELINE_FUSION
    with pytest.raises(ValueError, match="checkpoint/config mismatch"):
        validate_baseline_checkpoint(
            checkpoint,
            {"backbone": "xception", "pretrained": True, "num_classes": 2},
            tmp_path / "best.pt",
        )
