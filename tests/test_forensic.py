import pytest
import timm
import torch
from torch.nn.parameter import UninitializedParameter

from favit_lsda.forensic import (
    ProjectedForensicEncoder,
    _backbone_normalization,
)


class TinyBackbone(torch.nn.Module):
    num_features = 5

    def forward(self, images):
        return images.mean((-2, -1)).mean(1, keepdim=True).expand(-1, 5)


@pytest.mark.parametrize("name", ["xception", "efficientnet_b0"])
def test_encoder_builds_requested_pretrained_model(monkeypatch, name):
    calls = []

    def fake_create(model_name, **kwargs):
        calls.append((model_name, kwargs))
        return TinyBackbone()

    monkeypatch.setattr("favit_lsda.forensic.timm.create_model", fake_create)
    model = ProjectedForensicEncoder(name, 7, True, 0.0)
    output = model(torch.randn(2, 3, 16, 16))

    assert output.shape == (2, 7)
    assert torch.isfinite(output).all()
    assert calls == [
        (name, {"pretrained": True, "num_classes": 0, "global_pool": "avg"})
    ]
    output.square().mean().backward()
    assert model.project[0].weight.grad is not None


@pytest.mark.parametrize(
    ("name", "expected_width"),
    [("xception", 2048), ("efficientnet_b0", 1280)],
)
@pytest.mark.filterwarnings("ignore:Mapping deprecated model name xception")
def test_encoder_projects_real_backbone_output_width(name, expected_width):
    model = ProjectedForensicEncoder(name, 7, False, 0.0).eval()

    with torch.no_grad():
        output = model(torch.randn(1, 3, 64, 64))

    assert output.shape == (1, 7)
    assert torch.isfinite(output).all()
    assert model.project[0].in_features == expected_width


@pytest.mark.parametrize(
    ("name", "expected_width"),
    [("xception", 2048), ("efficientnet_b0", 1280)],
)
@pytest.mark.filterwarnings("ignore:Mapping deprecated model name xception")
def test_encoder_parameters_are_initialized_before_forward(name, expected_width):
    model = ProjectedForensicEncoder(name, 7, False, 0.0)

    assert not any(
        isinstance(parameter, UninitializedParameter) for parameter in model.parameters()
    )
    assert sum(parameter.numel() for parameter in model.parameters()) > 0
    state = model.state_dict()
    assert state["project.0.weight"].shape == (7, expected_width)


@pytest.mark.parametrize(
    "images",
    [
        torch.randn(3, 16, 16),
        torch.randn(2, 1, 16, 16),
    ],
)
def test_encoder_rejects_non_batched_three_channel_inputs(monkeypatch, images):
    monkeypatch.setattr(
        "favit_lsda.forensic.timm.create_model",
        lambda *_args, **_kwargs: TinyBackbone(),
    )
    model = ProjectedForensicEncoder("xception", 7, False, 0.0)

    with pytest.raises(ValueError, match=r"expects \[B, 3, H, W\]"):
        model(images)


@pytest.mark.parametrize(
    "images",
    [
        torch.randint(0, 255, (2, 3, 16, 16), dtype=torch.uint8),
        torch.full((2, 3, 16, 16), float("nan")),
    ],
)
def test_encoder_rejects_nonfinite_or_nonfloating_inputs(monkeypatch, images):
    monkeypatch.setattr(
        "favit_lsda.forensic.timm.create_model",
        lambda *_args, **_kwargs: TinyBackbone(),
    )
    model = ProjectedForensicEncoder("xception", 7, False, 0.0)

    with pytest.raises(ValueError, match=r"input must be finite floating point"):
        model(images)


def _capture_backbone_input(model, images):
    """Run the encoder and return the tensor its backbone actually received."""
    captured = {}

    def hook(_module, args):
        captured["images"] = args[0]

    handle = model.backbone.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            model(images)
    finally:
        handle.remove()
    return captured["images"]


def _stub_backbone(monkeypatch, mean=None, std=None):
    """Install a tiny backbone advertising the given pretrained stats."""

    class StubBackbone(TinyBackbone):
        pass

    if mean is not None and std is not None:
        StubBackbone.pretrained_cfg = {"mean": mean, "std": std}
    monkeypatch.setattr(
        "favit_lsda.forensic.timm.create_model",
        lambda *_args, **_kwargs: StubBackbone(),
    )


@pytest.mark.parametrize(
    ("name", "mean", "std"),
    [
        ("xception", (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        (
            "efficientnet_b0",
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
        (
            "tf_efficientnet_b4",
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
    ],
)
@pytest.mark.filterwarnings("ignore:Mapping deprecated model name xception")
def test_real_backbone_normalization_stats_are_read_from_timm(name, mean, std):
    """Pin the stats the renorm is derived from, per allowlisted backbone."""
    backbone = timm.create_model(name, pretrained=False, num_classes=0)

    assert _backbone_normalization(backbone) == (mean, std)


def test_encoder_is_identity_renorm_for_zero_five_mean_std_backbones(monkeypatch):
    """xception's stats equal the pipeline's, so its input passes through."""
    _stub_backbone(monkeypatch, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    model = ProjectedForensicEncoder("xception", 7, True, 0.0)
    images = torch.randn(2, 3, 16, 16)

    torch.testing.assert_close(_capture_backbone_input(model, images), images)


def test_encoder_renormalizes_for_imagenet_mean_std_backbones(monkeypatch):
    """efficientnet_b0 expects ImageNet stats, not the pipeline's 0.5/0.5."""
    _stub_backbone(
        monkeypatch, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    model = ProjectedForensicEncoder("efficientnet_b0", 7, True, 0.0)
    images = torch.randn(2, 3, 16, 16)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    expected = ((images * 0.5 + 0.5) - mean) / std
    torch.testing.assert_close(_capture_backbone_input(model, images), expected)


def test_encoder_falls_back_to_pipeline_stats_without_a_backbone_config(monkeypatch):
    _stub_backbone(monkeypatch)
    model = ProjectedForensicEncoder("xception", 7, True, 0.0)
    images = torch.randn(2, 3, 16, 16)

    torch.testing.assert_close(_capture_backbone_input(model, images), images)


def test_encoder_skips_renorm_without_pretrained_weights(monkeypatch):
    """Random init has no expected input distribution to re-normalize to."""
    _stub_backbone(
        monkeypatch, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    model = ProjectedForensicEncoder("efficientnet_b0", 7, False, 0.0)
    images = torch.randn(2, 3, 16, 16)

    torch.testing.assert_close(_capture_backbone_input(model, images), images)


@pytest.mark.parametrize("pretrained", [True, False])
def test_renorm_buffers_stay_out_of_the_state_dict(monkeypatch, pretrained):
    """They are derived from the backbone config, so they are not checkpointed."""
    _stub_backbone(
        monkeypatch, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    model = ProjectedForensicEncoder("efficientnet_b0", 7, pretrained, 0.0)

    assert not [key for key in model.state_dict() if "_renorm_" in key]
    model.load_state_dict(model.state_dict(), strict=True)


def test_encoder_ignores_non_positive_backbone_std(monkeypatch):
    """A degenerate std must not produce inf/nan scaling."""
    _stub_backbone(monkeypatch, mean=(0.5, 0.5, 0.5), std=(0.0, 0.5, 0.5))
    model = ProjectedForensicEncoder("efficientnet_b0", 7, True, 0.0)
    images = torch.randn(2, 3, 16, 16)

    torch.testing.assert_close(_capture_backbone_input(model, images), images)


@pytest.mark.filterwarnings("ignore:Mapping deprecated model name xception")
def test_encoder_projects_efficientnet_b4_output_width():
    model = ProjectedForensicEncoder(
        "tf_efficientnet_b4", 7, False, 0.0
    ).eval()

    with torch.no_grad():
        output = model(torch.randn(1, 3, 32, 32))

    assert output.shape == (1, 7)
    assert torch.isfinite(output).all()
    assert model.project[0].in_features == 1792


def test_encoder_rejects_non_rank_two_backbone_output(monkeypatch):
    class UnpooledBackbone(TinyBackbone):
        def forward(self, images):
            return images.mean(1, keepdim=True)

    monkeypatch.setattr(
        "favit_lsda.forensic.timm.create_model",
        lambda *_args, **_kwargs: UnpooledBackbone(),
    )
    model = ProjectedForensicEncoder("xception", 7, False, 0.0)

    with pytest.raises(RuntimeError, match=r"pooled output must be rank 2"):
        model(torch.randn(2, 3, 16, 16))
