from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from favit_lsda.baseline import (
    baseline_checkpoint_metadata,
    build_baseline_model,
    validate_baseline_checkpoint,
    validate_baseline_model_config,
)
from favit_lsda.baseline_engine import train_baseline_one_epoch
from favit_lsda.config import load_config, resolve_device, seed_everything
from favit_lsda.data import BinaryFrameDataset, FaceTransform, FrameFaceDataset
from favit_lsda.engine import evaluate_at_level


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a conventional RGB-only timm baseline on FF++"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _append_history(path: Path, record: dict) -> None:
    print(json.dumps(record, indent=2))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _capture_random_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_random_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _build_scheduler(
    optimizer: torch.optim.Optimizer, train_config: dict
) -> torch.optim.lr_scheduler.LambdaLR:
    epochs = int(train_config["epochs"])
    warmup_epochs = int(train_config.get("warmup_epochs", 0))
    min_lr_ratio = float(train_config.get("min_lr_ratio", 0.05))
    if not 0 <= warmup_epochs < epochs:
        raise ValueError("train.warmup_epochs must be in [0, epochs)")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("train.min_lr_ratio must be in [0, 1]")

    def multiplier(epoch: int) -> float:
        if warmup_epochs and epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        span = max(1, epochs - warmup_epochs - 1)
        progress = min(1.0, max(0.0, (epoch - warmup_epochs) / span))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _make_eval_loader(
    manifest: str | Path,
    data_config: dict,
    batch_size: int,
    device: torch.device,
) -> DataLoader:
    dataset = FrameFaceDataset(
        manifest,
        data_config["root"],
        FaceTransform(int(data_config.get("image_size", 224))),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(data_config.get("num_workers", 8)),
        pin_memory=device.type == "cuda",
    )


def _class_weights(dataset: BinaryFrameDataset, device: torch.device) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(dataset.labels), minlength=2).float()
    if bool((counts == 0).any()):
        raise ValueError("FF++ training data must contain both binary classes")
    return (counts.sum() / (2.0 * counts)).to(device)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    model_config = config["model"]
    train_config = config["train"]
    validate_baseline_model_config(model_config)
    if "train_manifest" not in data_config:
        raise ValueError("baseline config requires data.train_manifest")
    if "celebdf_validation_frames" not in data_config:
        raise ValueError(
            "baseline config requires data.celebdf_validation_frames for "
            "video-level checkpoint selection"
        )
    resume_value = args.resume or train_config.get("resume")
    resume_path = Path(resume_value) if resume_value else None
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")

    seed_everything(int(config.get("seed", 42)))
    device = resolve_device(args.device or config.get("device", "cuda"))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    augmentation = data_config.get("train_augmentation", {})
    train_transform = FaceTransform(
        image_size=int(data_config.get("image_size", 224)),
        horizontal_flip=float(augmentation.get("horizontal_flip", 0.0)),
        crop_scale_min=float(augmentation.get("crop_scale_min", 1.0)),
        color_jitter=float(augmentation.get("color_jitter", 0.0)),
        grayscale_probability=float(
            augmentation.get("grayscale_probability", 0.0)
        ),
        blur_probability=float(augmentation.get("blur_probability", 0.0)),
        degradation_probability=float(
            augmentation.get("degradation_probability", 0.0)
        ),
        jpeg_probability=float(augmentation.get("jpeg_probability", 0.0)),
        jpeg_quality_min=int(augmentation.get("jpeg_quality_min", 40)),
    )
    train_dataset = BinaryFrameDataset(
        data_config["train_manifest"], data_config["root"], train_transform
    )
    num_workers = int(data_config.get("num_workers", 8))
    batch_size = int(train_config.get("batch_size", 32))
    if batch_size <= 0:
        raise ValueError("train.batch_size must be positive")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    eval_batch_size = int(train_config.get("eval_batch_size", batch_size))
    celebdf_loader = _make_eval_loader(
        data_config["celebdf_validation_frames"],
        data_config,
        eval_batch_size,
        device,
    )
    ffpp_test_manifest = data_config.get("ffpp_test_frames")
    ffpp_test_loader = (
        _make_eval_loader(
            ffpp_test_manifest, data_config, eval_batch_size, device
        )
        if ffpp_test_manifest
        else None
    )

    resume_checkpoint = None
    if resume_path is not None:
        resume_checkpoint = torch.load(
            resume_path, map_location="cpu", weights_only=False
        )
        validate_baseline_checkpoint(resume_checkpoint, model_config, resume_path)
    model = build_baseline_model(
        model_config, pretrained=False if resume_checkpoint is not None else None
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config.get("weight_decay", 0.01)),
        betas=tuple(train_config.get("adam_betas", [0.9, 0.999])),
    )
    scheduler = _build_scheduler(optimizer, train_config)
    amp_enabled = bool(train_config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    criterion = nn.CrossEntropyLoss(
        weight=(
            _class_weights(train_dataset, device)
            if train_config.get("class_balanced_loss", True)
            else None
        ),
        label_smoothing=float(train_config.get("label_smoothing", 0.0)),
    )

    start_epoch = 0
    best_auc = float("-inf")
    epochs_without_improvement = 0
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model"], strict=True)
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        scheduler.load_state_dict(resume_checkpoint["scheduler"])
        if resume_checkpoint.get("scaler"):
            scaler.load_state_dict(resume_checkpoint["scaler"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_auc = float(resume_checkpoint.get("best_selection_auc", best_auc))
        epochs_without_improvement = int(
            resume_checkpoint.get("epochs_without_improvement", 0)
        )
        _restore_random_state(resume_checkpoint.get("random_state"))
        print(f"resume_checkpoint: path={resume_path} next_epoch={start_epoch + 1}")

    history_path = output_dir / "history.jsonl"
    for epoch in range(start_epoch, int(train_config["epochs"])):
        train_metrics = train_baseline_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            max_grad_norm=float(train_config.get("max_grad_norm", 0.0)) or None,
        )
        celebdf_metrics = evaluate_at_level(
            model,
            celebdf_loader,
            device,
            level="video",
            description="validate Celeb-DF (video)",
        )
        record = {
            "epoch": epoch + 1,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "celebdf_validation": celebdf_metrics,
        }
        _append_history(history_path, record)
        current_auc = float(celebdf_metrics["auc"])
        improved = current_auc > best_auc
        best_auc = max(best_auc, current_auc)
        epochs_without_improvement = 0 if improved else epochs_without_improvement + 1
        scheduler.step()
        state = {
            **baseline_checkpoint_metadata(model),
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_selection_auc": best_auc,
            "selection_name": "celebdf_validation",
            "selection_metrics": celebdf_metrics,
            "ffpp_test_metrics": None,
            "epochs_without_improvement": epochs_without_improvement,
            "random_state": _capture_random_state(),
            "config": config,
        }
        if improved:
            _save_checkpoint(output_dir / "best.pt", state)
            print(
                f"save_best_checkpoint: epoch={epoch + 1} "
                f"celebdf_video_auc={current_auc:.6f}"
            )
        _save_checkpoint(output_dir / "last.pt", state)
        patience = int(train_config.get("early_stopping_patience", 0))
        if patience > 0 and epochs_without_improvement >= patience:
            print(
                "early_stopping: no Celeb-DF video AUC improvement for "
                f"{patience} epochs"
            )
            break

    if ffpp_test_loader is not None:
        best_path = output_dir / "best.pt"
        if not best_path.is_file():
            raise FileNotFoundError("no best baseline checkpoint is available")
        best_state = torch.load(best_path, map_location="cpu", weights_only=False)
        validate_baseline_checkpoint(best_state, model_config, best_path)
        model.load_state_dict(best_state["model"], strict=True)
        ffpp_metrics = evaluate_at_level(
            model,
            ffpp_test_loader,
            device,
            level="video",
            description="test FF++ (video)",
        )
        best_state["ffpp_test_metrics"] = ffpp_metrics
        _save_checkpoint(best_path, best_state)
        _append_history(
            history_path,
            {"event": "final_ffpp_test", "ffpp_test": ffpp_metrics},
        )


if __name__ == "__main__":
    main()
