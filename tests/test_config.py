import pytest

from favit_lsda.config import resolve_branch_config, validate_model_config


def test_branch_config_defaults_to_rgb_only():
    value = resolve_branch_config({})
    assert value.enabled_branches == ("rgb",)
    assert value.srm_backbone == "xception"
    assert value.fft_backbone == "mobilenetv3_small_100"
    assert value.forensic_pretrained is True


def test_branch_config_enables_both_forensic_paths():
    value = resolve_branch_config(
        {"enable_srm_branch": True, "enable_fft_branch": True}
    )
    assert value.enabled_branches == ("rgb", "srm", "fft")


@pytest.mark.parametrize("key", ["artifact_mode", "cnn_in_channels"])
def test_legacy_artifact_fields_are_rejected(key):
    with pytest.raises(ValueError, match=rf"obsolete.*{key}"):
        validate_model_config({key: "legacy"})


@pytest.mark.parametrize(
    ("field", "value"),
    [("srm_backbone", "resnet50"), ("fft_backbone", "freqnet")],
)
def test_unsupported_backbones_are_rejected(field, value):
    with pytest.raises(ValueError, match=rf"{field}.*{value}"):
        validate_model_config({field: value})
