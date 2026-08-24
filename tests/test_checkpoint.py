import csv
import json
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

import train
from favit_lsda.config import build_model_from_config
from favit_lsda.config import resolve_branch_config
from train import load_favit_initialization, save_checkpoint

TINY_MODEL_CONFIG = {
    "backbone": "vit_tiny_patch16_224",
    "pretrained": False,
    "num_classes": 2,
    "forgery_methods": ["DF", "F2F"],
    "train_backbone_norms": False,
    "train_cls_token": False,
    "enable_srm_branch": False,
    "enable_fft_branch": False,
}


def test_checkpoint_is_atomic(tmp_path):
    path = tmp_path / "model.pt"
    save_checkpoint(path, {"architecture": "favit_lsda", "epoch": 1})
    assert torch.load(path, weights_only=False)["epoch"] == 1
    assert not (tmp_path / "model.pt.tmp").exists()


def test_favit_initialization_loads_only_compatible_tensors(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
    model.artifact_cnn = torch.nn.Module()
    model.artifact_cnn.stem = torch.nn.Sequential(torch.nn.Linear(4, 4))
    original = model[0].weight.detach().clone()
    original_cnn_weight = model.artifact_cnn.stem[0].weight.detach().clone()
    checkpoint = {
        "model": {
            "0.weight": torch.full_like(model[0].weight, 7.0),
            "0.bias": torch.zeros(99),
            "not_in_target": torch.ones(1),
        }
    }
    path = tmp_path / "favit.pt"
    torch.save(checkpoint, path)
    assert load_favit_initialization(model, path) == 1
    assert not torch.equal(model[0].weight, original)
    assert torch.equal(model[0].weight, torch.full_like(model[0].weight, 7.0))
    # The CNN artifact branch has no counterpart in a FA-ViT-only source
    # checkpoint, so it must stay freshly initialized rather than being
    # partially/incorrectly loaded.
    assert torch.equal(model.artifact_cnn.stem[0].weight, original_cnn_weight)


def test_favit_initialization_excludes_head_even_when_shape_matches(tmp_path):
    class Detector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.body = torch.nn.Linear(3, 2)
            self.head = torch.nn.Linear(2, 2, bias=False)
            self.srm_encoder = torch.nn.Linear(3, 2, bias=False)
            self.fft_encoder = torch.nn.Linear(3, 2, bias=False)
            self.late_fusion = torch.nn.Linear(4, 2, bias=False)

    model = Detector()
    original_head_weight = model.head.weight.detach().clone()
    original_srm_weight = model.srm_encoder.weight.detach().clone()
    original_fft_weight = model.fft_encoder.weight.detach().clone()
    original_fusion_weight = model.late_fusion.weight.detach().clone()
    checkpoint = {
        "model": {
            "body.weight": torch.full_like(model.body.weight, 7.0),
            "head.weight": torch.full_like(model.head.weight, 3.0),
            "srm_encoder.weight": torch.full_like(model.srm_encoder.weight, 5.0),
            "fft_encoder.weight": torch.full_like(model.fft_encoder.weight, 6.0),
            "late_fusion.weight": torch.full_like(model.late_fusion.weight, 8.0),
        }
    }
    path = tmp_path / "favit.pt"
    torch.save(checkpoint, path)
    loaded = load_favit_initialization(model, path)
    assert loaded == 1
    assert torch.equal(model.body.weight, torch.full_like(model.body.weight, 7.0))
    assert torch.equal(model.head.weight, original_head_weight)
    assert torch.equal(model.srm_encoder.weight, original_srm_weight)
    assert torch.equal(model.fft_encoder.weight, original_fft_weight)
    assert torch.equal(model.late_fusion.weight, original_fusion_weight)


def test_checkpoint_rejects_artifact_metadata_mismatch(tmp_path):
    from favit_lsda.checkpoints import validate_checkpoint_branches
    checkpoint = _tiny_checkpoint({"enable_srm_branch": False, "enable_fft_branch": True})
    with pytest.raises(ValueError, match=r"checkpoint/config branch mismatch.*rgb"):
        validate_checkpoint_branches(
            checkpoint,
            {"enable_srm_branch": True, "enable_fft_branch": False},
            tmp_path / "last.pt",
        )


def test_checkpoint_rejects_legacy_architecture(tmp_path):
    from favit_lsda.checkpoints import validate_checkpoint_branches
    checkpoint = {
        "format_version": 4,
        "architecture": "favit_lsda_cnn",
    }
    with pytest.raises(ValueError, match=r"legacy architecture.*--init-favit"):
        validate_checkpoint_branches(
            checkpoint, {}, tmp_path / "old.pt"
        )


@pytest.mark.parametrize("version", [5, 2, None, "4"])
def test_checkpoint_rejects_unsupported_format_version(tmp_path, version):
    """Catches a future on-disk format bump loading silently against old code."""
    from favit_lsda.checkpoints import validate_checkpoint_branches
    checkpoint = {
        "format_version": version,
        "architecture": "favit_lsda_multibranch",
        "enabled_branches": ["rgb"],
        "srm_backbone": None,
        "fft_backbone": None,
        "fusion": "fixed_slot_concat",
    }
    with pytest.raises(ValueError, match=r"unsupported format_version"):
        validate_checkpoint_branches(
            checkpoint, {}, tmp_path / "future.pt"
        )


def test_checkpoint_rejects_legacy_v3_with_migration_message(tmp_path):
    from favit_lsda.checkpoints import validate_checkpoint_branches

    checkpoint = {
        "format_version": 3,
        "architecture": "favit_lsda_cnn",
    }
    with pytest.raises(ValueError, match=r"legacy.*format_version 3.*--init-favit"):
        validate_checkpoint_branches(checkpoint, {}, tmp_path / "old.pt")


def test_checkpoint_rejects_absent_top_level_branch_metadata(tmp_path):
    from favit_lsda.checkpoints import validate_checkpoint_branches

    checkpoint = _tiny_checkpoint(TINY_MODEL_CONFIG)
    del checkpoint["enabled_branches"]
    with pytest.raises(ValueError, match=r"checkpoint/config branch mismatch"):
        validate_checkpoint_branches(
            checkpoint, TINY_MODEL_CONFIG, tmp_path / "missing-branches.pt"
        )


def test_checkpoint_rejects_malformed_top_level_branch_metadata(tmp_path):
    from favit_lsda.checkpoints import validate_checkpoint_branches

    checkpoint = _tiny_checkpoint(TINY_MODEL_CONFIG)
    checkpoint["enabled_branches"] = "rgb"
    checkpoint["fusion"] = {"name": "fixed_slot_concat"}
    with pytest.raises(ValueError, match=r"checkpoint/config branch mismatch"):
        validate_checkpoint_branches(
            checkpoint, TINY_MODEL_CONFIG, tmp_path / "malformed-branches.pt"
        )


def _tiny_checkpoint(model_config: dict, state: dict | None = None) -> dict:
    branches = resolve_branch_config(model_config)
    return {
        "format_version": 4,
        "architecture": "favit_lsda_multibranch",
        "enabled_branches": list(branches.enabled_branches),
        "srm_backbone": branches.srm_backbone if branches.enable_srm else None,
        "fft_backbone": branches.fft_backbone if branches.enable_fft else None,
        "fusion": "fixed_slot_concat",
        "config": {"model": model_config},
        "model": state if state is not None else {},
    }


def test_evaluation_validates_checkpoint_against_cli_config_not_embedded_config(tmp_path):
    """Catches the no-op self-comparison of a checkpoint against its own config.

    The checkpoint is internally consistent (rgb_fft everywhere), so validating
    it against its own embedded config can never fail. What must be checked is
    the config the user actually passed to ``--config``.
    """
    from favit_lsda import evaluation

    checkpoint_config = {**TINY_MODEL_CONFIG, "enable_srm_branch": False, "enable_fft_branch": True}
    path = tmp_path / "best.pt"
    torch.save(_tiny_checkpoint(checkpoint_config), path)
    cli_config = {"model": {**TINY_MODEL_CONFIG, "enable_srm_branch": True, "enable_fft_branch": False}}
    with pytest.raises(ValueError, match=r"checkpoint/config branch mismatch.*rgb"):
        evaluation._load_model(path, cli_config, torch.device("cpu"))


def test_evaluation_rejects_mismatched_checkpoint_before_touching_model_state(
    tmp_path, monkeypatch
):
    """Catches validation being reordered after construction or state loading."""
    from favit_lsda import evaluation

    model = build_model_from_config(TINY_MODEL_CONFIG, pretrained=False)
    snapshot = {key: value.detach().clone() for key, value in model.state_dict().items()}
    assert any(value.abs().sum() > 0 for value in snapshot.values())
    poisoned = {key: torch.zeros_like(value) for key, value in snapshot.items()}

    checkpoint_config = {**TINY_MODEL_CONFIG, "enable_srm_branch": True, "enable_fft_branch": False}
    path = tmp_path / "best.pt"
    torch.save(_tiny_checkpoint(checkpoint_config, poisoned), path)

    builder_calls: list[dict] = []

    def recording_builder(model_config, *args, **kwargs):
        builder_calls.append(model_config)
        return model

    monkeypatch.setattr(evaluation, "build_model_from_config", recording_builder)
    with pytest.raises(ValueError, match=r"checkpoint.*config.*branches"):
        evaluation._load_model(path, {"model": dict(TINY_MODEL_CONFIG)}, torch.device("cpu"))

    assert builder_calls == []
    for key, value in model.state_dict().items():
        assert torch.equal(value, snapshot[key])


def _write_train_fixture(tmp_path, epochs: int = 2, **extra_data) -> tuple[str, dict]:
    """Write a runnable tiny train config with real images and manifests."""
    for name in ("real_a.jpg", "real_b.jpg"):
        Image.new("RGB", (24, 24), "white").save(tmp_path / name)
    pair_rows = []
    for real in ("real_a.jpg", "real_b.jpg"):
        for method in ("DF", "F2F"):
            fake = f"{real[:-4]}_{method}.jpg"
            Image.new("RGB", (24, 24), "red").save(tmp_path / fake)
            pair_rows.append({"fake_path": fake, "real_path": real, "method": method})
    pairs = tmp_path / "pairs.csv"
    with pairs.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fake_path", "real_path", "method"])
        writer.writeheader()
        writer.writerows(pair_rows)

    def _write_frames(name: str) -> Path:
        path = tmp_path / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "label", "video_id"])
            writer.writeheader()
            writer.writerow({"path": "real_a.jpg", "label": "0", "video_id": "v0"})
            writer.writerow({"path": "real_a_DF.jpg", "label": "1", "video_id": "v1"})
        return path

    frames = _write_frames("frames.csv")

    config = {
        "seed": 0,
        "device": "cpu",
        "output_dir": str(tmp_path / "out"),
        "data": {
            "root": str(tmp_path),
            "train_pairs": str(pairs),
            "validation_frames": str(frames),
            "image_size": 224,
            "num_workers": 0,
            # Each extra manifest is its own file (identical rows) rather than
            # reusing `frames`, so it is never mistaken for the same file as
            # validation_frames by path-equality checks in train.py.
            **{key: str(_write_frames(f"{key}.csv")) for key in extra_data},
        },
        "model": dict(TINY_MODEL_CONFIG),
        "loss": {},
        "train": {
            "epochs": epochs,
            "group_batch_size": 2,
            "eval_image_batch_size": 2,
            "learning_rate": 0.0001,
            "warmup_epochs": 0,
            "amp": False,
        },
    }
    config_path = tmp_path / "config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return str(config_path), config


def _record_train_models(monkeypatch) -> list[tuple[torch.nn.Module, dict]]:
    """Capture every model ``train.main`` builds, with a snapshot of its weights."""
    captured: list[tuple[torch.nn.Module, dict]] = []
    real_builder = train.build_model_from_config

    def recording_builder(model_config, *args, **kwargs):
        model = real_builder(model_config, *args, **kwargs)
        captured.append(
            (model, {k: v.detach().clone() for k, v in model.state_dict().items()})
        )
        return model

    monkeypatch.setattr(train, "build_model_from_config", recording_builder)
    return captured


def test_train_resume_rejects_mismatched_checkpoint_before_loading_state(
    tmp_path, monkeypatch
):
    """Catches --resume loading a wrong-mode checkpoint's weights before validating."""
    config_path, _ = _write_train_fixture(tmp_path)

    reference = build_model_from_config(TINY_MODEL_CONFIG, pretrained=False)
    poisoned = {
        key: torch.zeros_like(value) for key, value in reference.state_dict().items()
    }
    checkpoint_config = {**TINY_MODEL_CONFIG, "enable_srm_branch": True, "enable_fft_branch": False}
    resume_path = tmp_path / "resume.pt"
    torch.save(
        {
            **_tiny_checkpoint(checkpoint_config, poisoned),
            "epoch": 0,
            "optimizer": {},
            "scheduler": {},
            "scaler": None,
        },
        resume_path,
    )

    captured = _record_train_models(monkeypatch)
    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "--config", config_path, "--resume", str(resume_path)],
    )
    with pytest.raises(ValueError, match=r"checkpoint.*config.*branches"):
        train.main()

    assert len(captured) == 1
    model, snapshot = captured[0]
    # Non-vacuous: the rejected checkpoint is all zeros, so a load that slipped
    # through ahead of validation would be visible in this comparison.
    assert any(value.abs().sum() > 0 for value in snapshot.values())
    for key, value in model.state_dict().items():
        assert torch.equal(value, snapshot[key])


def test_final_target_evaluation_runs_at_video_level_and_persists_model_metadata(
    tmp_path, monkeypatch
):
    """Pins the post-selection protocol and the source of checkpoint metadata.

    Checkpoint selection and the final Celeb-DF target evaluation both
    aggregate at video level, matching fa_vit_lsda's baseline training flow
    -- not per frame.

    The same run also proves the persisted artifact metadata comes from the
    constructed model instance rather than from a second, independently
    resolved copy of the config.
    """
    config_path, config = _write_train_fixture(
        tmp_path, epochs=1, celebdf_test_frames=True
    )
    captured = _record_train_models(monkeypatch)
    monkeypatch.setattr("sys.argv", ["train.py", "--config", config_path])
    train.main()

    output_dir = Path(config["output_dir"])
    records = [
        json.loads(line)
        for line in (output_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    final = [
        record for record in records if record.get("event") == "final_target_evaluation"
    ]
    assert len(final) == 1
    assert final[0]["celebdf_test"]["level"] == "video"
    assert final[0]["celebdf_test"]["num_videos"] == 2

    assert len(captured) == 1
    model, _ = captured[0]
    best = torch.load(output_dir / "best.pt", weights_only=False)
    assert best["format_version"] == 4
    assert best["architecture"] == "favit_lsda_multibranch"
    assert best["enabled_branches"] == list(model.enabled_branches)
    assert best["srm_backbone"] == model.srm_backbone_name
    assert best["fft_backbone"] == model.fft_backbone_name
    assert best["fusion"] == model.fusion_name
    assert best["celebdf_test_metrics"]["level"] == "video"
