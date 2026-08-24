import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from torch.nn.parameter import UninitializedParameter


def _install_data_import_compatibility():
    """Load migrated data.py and bridge the legacy model import in-process."""
    if "favit_lsda" in sys.modules:
        return
    data_path = Path(__file__).parents[1] / "favit_lsda" / "data.py"
    spec = importlib.util.spec_from_file_location("favit_lsda.data", data_path)
    data_module = importlib.util.module_from_spec(spec)
    sys.modules["favit_lsda.data"] = data_module
    spec.loader.exec_module(data_module)
    if not hasattr(data_module, "artifact_channels"):
        data_module.artifact_channels = lambda mode: {
            "rgb": 3,
            "rgb_srm": 6,
            "rgb_fft": 6,
            "rgb_srm_fft": 9,
        }[mode]


_install_data_import_compatibility()

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
