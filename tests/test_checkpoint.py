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
    original = model[0].weight.detach().clone()
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


def test_favit_initialization_excludes_head_even_when_shape_matches(tmp_path):
    class Detector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.body = torch.nn.Linear(3, 2)
            self.head = torch.nn.Linear(2, 2, bias=False)
            self.rgb_cnn_encoder = torch.nn.Linear(3, 2, bias=False)
            self.srm_encoder = torch.nn.Linear(3, 2, bias=False)
            self.fft_encoder = torch.nn.Linear(3, 2, bias=False)
            self.late_fusion = torch.nn.Linear(4, 2, bias=False)

    model = Detector()
    original_head_weight = model.head.weight.detach().clone()
    original_rgb_cnn_weight = model.rgb_cnn_encoder.weight.detach().clone()
    original_srm_weight = model.srm_encoder.weight.detach().clone()
    original_fft_weight = model.fft_encoder.weight.detach().clone()
    original_fusion_weight = model.late_fusion.weight.detach().clone()
    checkpoint = {
        "model": {
            "body.weight": torch.full_like(model.body.weight, 7.0),
            "head.weight": torch.full_like(model.head.weight, 3.0),
            "rgb_cnn_encoder.weight": torch.full_like(model.rgb_cnn_encoder.weight, 4.0),
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
    assert torch.equal(model.rgb_cnn_encoder.weight, original_rgb_cnn_weight)
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


@pytest.mark.parametrize("version", [6, 2, None, "5"])
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
        "format_version": 5,
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


def test_evaluation_deserializes_and_loads_on_cpu_before_moving_model(
    tmp_path, monkeypatch
):
    """Catches checkpoint tensors being materialized on the accelerator."""
    from favit_lsda import evaluation

    path = tmp_path / "best.pt"
    path.write_bytes(b"checkpoint placeholder")
    checkpoint = _tiny_checkpoint(TINY_MODEL_CONFIG)
    load_calls = []
    events = []

    def fake_load(checkpoint_path, **kwargs):
        load_calls.append((checkpoint_path, kwargs))
        return checkpoint

    class RecordingModel:
        def load_state_dict(self, state, strict):
            events.append(("load_state_dict", state, strict))

        def to(self, device):
            events.append(("to", device))
            return self

    monkeypatch.setattr(evaluation.torch, "load", fake_load)
    monkeypatch.setattr(
        evaluation, "build_model_from_config", lambda *_args, **_kwargs: RecordingModel()
    )

    evaluation._load_model(path, {"model": TINY_MODEL_CONFIG}, torch.device("cpu"))

    assert load_calls == [
        (path, {"map_location": "cpu", "weights_only": False})
    ]
    assert events == [
        ("load_state_dict", {}, True),
        ("to", torch.device("cpu")),
    ]


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


@pytest.mark.parametrize("legacy", [False, True], ids=("branch-mismatch", "v3"))
def test_train_resume_rejects_checkpoint_before_builders_or_output_side_effects(
    tmp_path, monkeypatch, legacy
):
    """Catches resume compatibility checks occurring after startup side effects."""
    config_path, _ = _write_train_fixture(tmp_path)
    resume_path = tmp_path / "resume.pt"
    checkpoint = (
        {"format_version": 3, "architecture": "favit_lsda_cnn"}
        if legacy
        else _tiny_checkpoint(
            {
                **TINY_MODEL_CONFIG,
                "enable_srm_branch": True,
                "enable_fft_branch": False,
            }
        )
    )
    torch.save(checkpoint, resume_path)

    builder_calls = []
    real_grouped_dataset = train.GroupedForgeryDataset
    real_frame_dataset = train.FrameFaceDataset
    real_model_builder = train.build_model_from_config
    real_optimizer_builder = train.build_optimizer

    def record(name, function):
        def wrapper(*args, **kwargs):
            builder_calls.append(name)
            return function(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(
        train, "GroupedForgeryDataset", record("grouped_dataset", real_grouped_dataset)
    )
    monkeypatch.setattr(
        train, "FrameFaceDataset", record("frame_dataset", real_frame_dataset)
    )
    monkeypatch.setattr(
        train, "build_model_from_config", record("model", real_model_builder)
    )
    monkeypatch.setattr(
        train, "build_optimizer", record("optimizer", real_optimizer_builder)
    )
    real_torch_load = train.torch.load
    load_calls = []

    def recording_load(path, **kwargs):
        load_calls.append((path, kwargs.get("map_location")))
        return real_torch_load(path, **kwargs)

    monkeypatch.setattr(train.torch, "load", recording_load)
    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "--config", config_path, "--resume", str(resume_path)],
    )
    expected_error = (
        r"legacy.*format_version 3"
        if legacy
        else r"checkpoint.*config.*branches"
    )
    with pytest.raises(ValueError, match=expected_error):
        train.main()

    assert load_calls == [(resume_path, "cpu")]
    assert builder_calls == []
    assert not (tmp_path / "out").exists()


def test_train_resume_releases_deserialized_checkpoint_before_epoch_loop(
    tmp_path, monkeypatch
):
    """Catches CPU optimizer state remaining live through the training loop."""
    import gc
    import weakref

    config_path, config = _write_train_fixture(tmp_path)
    model = build_model_from_config(TINY_MODEL_CONFIG, pretrained=False)
    optimizer = train.build_optimizer(model, config["train"])
    scheduler = train.build_scheduler(optimizer, config["train"])
    resume_path = tmp_path / "resume.pt"
    torch.save(
        {
            **_tiny_checkpoint(TINY_MODEL_CONFIG, model.state_dict()),
            "epoch": 0,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": None,
            "random_state": None,
        },
        resume_path,
    )

    class WeakCheckpoint(dict):
        __slots__ = ("__weakref__",)

    real_torch_load = train.torch.load
    checkpoint_ref = None

    def tracked_load(path, **kwargs):
        nonlocal checkpoint_ref
        checkpoint = WeakCheckpoint(real_torch_load(path, **kwargs))
        checkpoint_ref = weakref.ref(checkpoint)
        return checkpoint

    def observe_release(*args, **_kwargs):
        gc.collect()
        assert checkpoint_ref is not None
        assert checkpoint_ref() is None
        args[2].step()
        return {}

    monkeypatch.setattr(train.torch, "load", tracked_load)
    monkeypatch.setattr(train, "train_one_epoch", observe_release)
    monkeypatch.setattr(train, "evaluate_at_level", lambda *_args, **_kwargs: {"auc": 0.5})
    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "--config", config_path, "--resume", str(resume_path)],
    )

    train.main()


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
    assert best["format_version"] == 5
    assert best["architecture"] == "favit_lsda_multibranch"
    assert best["enabled_branches"] == list(model.enabled_branches)
    assert best["srm_backbone"] == model.srm_backbone_name
    assert best["fft_backbone"] == model.fft_backbone_name
    assert best["fusion"] == model.fusion_name
    assert best["celebdf_test_metrics"]["level"] == "video"


def test_checkpoint_metadata_is_version_five(tmp_path):
    """The mandatory RGB CNN slot widened late_fusion, so v4 state cannot load."""
    from favit_lsda.checkpoints import model_branch_metadata

    model = build_model_from_config(TINY_MODEL_CONFIG, pretrained=False)
    metadata = model_branch_metadata(model)
    assert metadata["format_version"] == 5
    # The branch reads inputs['rgb'], so it must not widen the input contract.
    assert metadata["enabled_branches"] == ["rgb"]


def test_checkpoint_rejects_legacy_v4_with_migration_message(tmp_path):
    """A v4 checkpoint predates the RGB CNN slot; its late_fusion is narrower."""
    from favit_lsda.checkpoints import validate_checkpoint_branches

    checkpoint = {
        "format_version": 4,
        "architecture": "favit_lsda_multibranch",
    }
    with pytest.raises(ValueError, match=r"legacy.*format_version 4.*--init-favit"):
        validate_checkpoint_branches(checkpoint, {}, tmp_path / "old.pt")


def test_train_round_trip_carries_the_rgb_cnn_branch(tmp_path, monkeypatch):
    """Trains, saves and reloads a run whose fourth fusion slot is populated.

    Exercises the branch through forward_group, frame-level evaluation and the
    checkpoint contract in one pass, which unit coverage of the module alone
    cannot do.
    """
    config_path, config = _write_train_fixture(tmp_path, epochs=1)
    captured = _record_train_models(monkeypatch)
    monkeypatch.setattr("sys.argv", ["train.py", "--config", config_path])
    train.main()

    model, _ = captured[0]
    assert model.rgb_cnn_encoder is not None
    best = torch.load(Path(config["output_dir"]) / "best.pt", weights_only=False)
    assert any(key.startswith("rgb_cnn_encoder.") for key in best["model"])

    reloaded = build_model_from_config(TINY_MODEL_CONFIG, pretrained=False)
    reloaded.load_state_dict(best["model"])
    assert reloaded.late_fusion[0].in_features == reloaded.embed_dim * 4


def _history_records(output_dir) -> list[dict]:
    path = Path(output_dir) / "history.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_history_records_the_best_model_with_its_target_metrics(tmp_path, monkeypatch):
    """History must name the selected epoch, not just the per-epoch stream.

    Without this record the file says which epochs happened but never which one
    ``best.pt`` actually holds, so a finished run cannot be read back.
    """
    config_path, config = _write_train_fixture(
        tmp_path, epochs=2, celebdf_test_frames=True
    )
    monkeypatch.setattr("sys.argv", ["train.py", "--config", config_path])
    train.main()

    records = _history_records(config["output_dir"])
    best = [record for record in records if record.get("event") == "best_model"]
    assert len(best) == 1
    best = best[0]

    epochs = [record for record in records if "epoch" in record and "train" in record]
    selected = [
        record for record in epochs if record["epoch"] == best["epoch"]
    ]
    assert len(selected) == 1
    assert best["selection_name"] == "validation"
    assert best["selection_metrics"] == selected[0]["validation"]
    assert best["best_selection_auc"] == max(
        record["validation"]["auc"] for record in epochs
    )
    assert best["celebdf_test"]["level"] == "video"
    assert best["checkpoint"] == "best.pt"


def test_best_model_record_is_written_without_a_target_dataset(tmp_path, monkeypatch):
    """Catches the record being emitted only from the target-evaluation branch."""
    config_path, config = _write_train_fixture(tmp_path, epochs=1)
    monkeypatch.setattr("sys.argv", ["train.py", "--config", config_path])
    train.main()

    records = _history_records(config["output_dir"])
    assert not [r for r in records if r.get("event") == "final_target_evaluation"]
    best = [record for record in records if record.get("event") == "best_model"]
    assert len(best) == 1
    assert best[0]["epoch"] == 1
    assert best[0]["celebdf_test"] is None
