import pytest
import torch
from torch.nn.parameter import UninitializedParameter

from favit_lsda.forensic import ProjectedForensicEncoder


class TinyBackbone(torch.nn.Module):
    num_features = 5

    def forward(self, images):
        return images.mean((-2, -1)).mean(1, keepdim=True).expand(-1, 5)


@pytest.mark.parametrize("name", ["xception", "mobilenetv3_small_100"])
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
    [("xception", 2048), ("mobilenetv3_small_100", 1024)],
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
    [("xception", 2048), ("mobilenetv3_small_100", 1024)],
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


def test_encoder_is_identity_renorm_for_zero_five_mean_std_backbones(monkeypatch):
    """xception's default_cfg is mean=std=0.5, matching the pipeline exactly."""
    captured = {}

    class RecordingBackbone(TinyBackbone):
        def forward(self, images):
            captured["images"] = images
            return super().forward(images)

    monkeypatch.setattr(
        "favit_lsda.forensic.timm.create_model",
        lambda *_args, **_kwargs: RecordingBackbone(),
    )
    model = ProjectedForensicEncoder("xception", 7, False, 0.0)
    images = torch.randn(2, 3, 16, 16)

    model(images)

    torch.testing.assert_close(captured["images"], images)


def test_encoder_renormalizes_for_imagenet_mean_std_backbones(monkeypatch):
    """mobilenetv3_small_100 expects ImageNet stats, not the pipeline's 0.5/0.5."""
    captured = {}

    class RecordingBackbone(TinyBackbone):
        pretrained_cfg = {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
        }

        def forward(self, images):
            captured["images"] = images
            return super().forward(images)

    monkeypatch.setattr(
        "favit_lsda.forensic.timm.create_model",
        lambda *_args, **_kwargs: RecordingBackbone(),
    )
    model = ProjectedForensicEncoder("mobilenetv3_small_100", 7, False, 0.0)
    images = torch.randn(2, 3, 16, 16)

    model(images)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    expected = ((images * 0.5 + 0.5) - mean) / std
    torch.testing.assert_close(captured["images"], expected)


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
