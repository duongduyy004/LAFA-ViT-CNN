import sys
from pathlib import Path

import pytest

from favit_lsda.config import (
    load_config,
    resolve_branch_config,
    validate_model_config,
)


APPROVED_ABLATION_CONFIGS = {
    "favit_lsda_rgb.yaml",
    "favit_lsda_rgb_srm.yaml",
    "favit_lsda_rgb_fft.yaml",
    "favit_lsda_rgb_srm_fft.yaml",
}
PRIMARY_CONFIG = Path("configs/favit_lsda_ffpp_c23_celebdf.yaml")


def test_branch_config_defaults_to_rgb_only():
    value = resolve_branch_config({})
    assert value.enabled_branches == ("rgb",)
    assert value.srm_backbone == "xception"
    assert value.fft_backbone == "efficientnet_b0"
    assert value.forensic_pretrained is True


def test_branch_config_enables_both_forensic_paths():
    value = resolve_branch_config(
        {"enable_srm_branch": True, "enable_fft_branch": True}
    )
    assert value.enabled_branches == ("rgb", "srm", "fft")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("enable_srm_branch", "false"),
        ("enable_fft_branch", "true"),
        ("forensic_pretrained", "false"),
    ],
)
def test_branch_config_rejects_non_boolean_toggles(field, value):
    with pytest.raises(ValueError, match=rf"{field}.*boolean"):
        resolve_branch_config({field: value})


@pytest.mark.parametrize(
    "key", ["artifact_mode", "cnn_in_channels", "enable_rgb_cnn_branch"]
)
def test_legacy_artifact_fields_are_rejected(key):
    with pytest.raises(ValueError, match=rf"obsolete.*{key}"):
        validate_model_config({key: "legacy"})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("srm_backbone", "resnet50"),
        ("fft_backbone", "freqnet"),
        ("fft_backbone", "mobilenetv3_small_100"),
        # Vetted for SRM only; the FFT branch was never evaluated with it.
        ("fft_backbone", "tf_efficientnet_b4"),
        ("fft_backbone", "tf_efficientnet_b4.ns_jft_in1k"),
    ],
)
def test_unsupported_backbones_are_rejected(field, value):
    with pytest.raises(ValueError, match=rf"{field}.*{value}"):
        validate_model_config({field: value})


@pytest.mark.parametrize(
    ("name", "srm", "fft"),
    [
        ("rgb", False, False),
        ("rgb_srm", True, False),
        ("rgb_fft", False, True),
        ("rgb_srm_fft", True, True),
    ],
)
def test_experiment_config_has_exact_branches(name, srm, fft):
    config = load_config(Path("configs") / f"favit_lsda_{name}.yaml")
    validate_model_config(config["model"])
    assert config["model"]["enable_srm_branch"] is srm
    assert config["model"]["enable_fft_branch"] is fft
    assert config["model"]["srm_backbone"] == "tf_efficientnet_b4.ns_jft_in1k"
    assert config["model"]["fft_backbone"] == "efficientnet_b0"
    assert config["model"]["forensic_pretrained"] is True
    assert config["output_dir"] == f"outputs/favit_lsda_{name}"


def test_ablation_config_filenames_are_exact():
    observed = {
        path.name
        for path in Path("configs").glob("favit_lsda_*.yaml")
        if path != PRIMARY_CONFIG
    }
    assert observed == APPROVED_ABLATION_CONFIGS


def test_primary_config_is_explicitly_rgb_only():
    config = load_config(PRIMARY_CONFIG)
    validate_model_config(config["model"])
    assert config["model"]["enable_srm_branch"] is False
    assert config["model"]["enable_fft_branch"] is False
    assert config["model"]["srm_backbone"] == "tf_efficientnet_b4.ns_jft_in1k"
    assert config["model"]["fft_backbone"] == "efficientnet_b0"
    assert config["model"]["forensic_pretrained"] is True


def test_build_model_from_config_always_constructs_the_rgb_cnn_branch():
    """The branch is part of the architecture, not an opt-in slot."""
    from favit_lsda.config import build_model_from_config

    model = build_model_from_config(
        {"backbone": "vit_tiny_patch16_224", "forgery_methods": ["DF", "F2F"]},
        pretrained=False,
    )
    assert model.rgb_cnn_encoder is not None


def test_ffpp_test_runner_covers_every_ablation_config():
    """Catches a new ablation config silently missing from the batch runner."""
    import run_ffpp_tests

    assert set(run_ffpp_tests.CASES) == APPROVED_ABLATION_CONFIGS


def test_ffpp_test_runner_case_option_restricts_to_selected_configs(
    monkeypatch, capsys, tmp_path
):
    """``--case`` should skip every case not explicitly selected."""
    import run_ffpp_tests

    monkeypatch.setattr(run_ffpp_tests, "CONFIGS_DIR", tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["run_ffpp_tests.py", "--case", "favit_lsda_rgb_srm.yaml"]
    )

    run_ffpp_tests.main()

    output = capsys.readouterr().out
    assert "favit_lsda_rgb_srm.yaml" in output
    for other in run_ffpp_tests.CASES:
        if other != "favit_lsda_rgb_srm.yaml":
            assert other not in output


def test_ffpp_test_runner_rejects_unknown_case():
    import run_ffpp_tests

    with pytest.raises(SystemExit):
        run_ffpp_tests.build_parser().parse_args(["--case", "not_a_config.yaml"])
