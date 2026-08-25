from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from favit_lsda.checkpoints import model_branch_metadata, validate_checkpoint_branches
from favit_lsda.config import (
    build_model_from_config,
    load_config,
    resolve_branch_config,
    resolve_device,
    seed_everything,
    validate_model_config,
)
from favit_lsda.data import (
    FaceTransform,
    FrameFaceDataset,
    GroupedForgeryDataset,
)
from favit_lsda.engine import evaluate_at_level, train_one_epoch
from favit_lsda.losses import FineGrainedAdaptiveLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FA-ViT with LSDA")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--init-favit",
        type=Path,
        default=None,
        help="Initialize compatible layers from a fa_vit_remake checkpoint",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def capture_random_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_random_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def load_favit_initialization(model: torch.nn.Module, checkpoint_path: Path) -> int:
    """Load only matching name-and-shape FA-ViT tensors from a source checkpoint.

    Detector-specific heads, the parallel RGB CNN branch, forensic encoders,
    and late fusion are excluded because they have no counterpart worth
    transplanting from a FA-ViT source.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint.get("model", checkpoint)
    target = model.state_dict()
    excluded_prefixes = (
        "head.",
        "rgb_cnn_encoder.",
        "srm_encoder.",
        "fft_encoder.",
        "late_fusion.",
    )
    compatible = {
        key: value
        for key, value in source.items()
        if key in target
        and target[key].shape == value.shape
        and not key.startswith(excluded_prefixes)
    }
    shape_mismatches = [
        key
        for key, value in source.items()
        if key in target
        and target[key].shape != value.shape
        and not key.startswith(excluded_prefixes)
    ]
    if shape_mismatches:
        print(
            f"init_favit_checkpoint: skipped {len(shape_mismatches)} shape-mismatched "
            f"tensor(s), not transferred: {sorted(shape_mismatches)[:10]}"
        )
    model.load_state_dict(compatible, strict=False)
    return len(compatible)


def make_frame_loader(
    manifest: str | Path,
    data_config: dict,
    transform: FaceTransform,
    batch_size: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        FrameFaceDataset(manifest, data_config["root"], transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(data_config.get("num_workers", 8)),
        pin_memory=device.type == "cuda",
    )


def build_optimizer(
    model: torch.nn.Module, train_config: dict
) -> torch.optim.Optimizer:
    """AdamW with lower LR on pretrained ViT parameters and no decay on scales/norms."""

    learning_rate = float(train_config["learning_rate"])
    backbone_multiplier = float(train_config.get("backbone_lr_multiplier", 0.2))
    weight_decay = float(train_config.get("weight_decay", 0.01))
    groups: dict[tuple[bool, bool], list[torch.nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_backbone = name.startswith(
            ("backbone.", "srm_encoder.backbone.", "fft_encoder.backbone.")
        )
        use_decay = parameter.ndim > 1 and not name.endswith(".bias")
        groups.setdefault((is_backbone, use_decay), []).append(parameter)
    parameter_groups = [
        {
            "params": parameters,
            "lr": learning_rate * (backbone_multiplier if is_backbone else 1.0),
            "weight_decay": weight_decay if use_decay else 0.0,
        }
        for (is_backbone, use_decay), parameters in groups.items()
    ]
    return torch.optim.AdamW(
        parameter_groups,
        lr=learning_rate,
        betas=tuple(train_config.get("adam_betas", [0.9, 0.999])),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, train_config: dict
) -> torch.optim.lr_scheduler.LambdaLR:
    total_epochs = int(train_config["epochs"])
    warmup_epochs = int(train_config.get("warmup_epochs", 2))
    min_lr_ratio = float(train_config.get("min_lr_ratio", 0.05))
    if not 0 <= warmup_epochs < total_epochs:
        raise ValueError("warmup_epochs must be in [0, epochs)")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1]")

    def multiplier(epoch: int) -> float:
        if warmup_epochs and epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        span = max(1, total_epochs - warmup_epochs - 1)
        progress = min(1.0, max(0.0, (epoch - warmup_epochs) / span))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def ramped_weight(
    epoch: int, target: float, warmup_epochs: int, ramp_epochs: int
) -> float:
    if epoch < warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return target
    progress = min(1.0, (epoch - warmup_epochs + 1) / ramp_epochs)
    return target * progress


def main() -> None:
    args = parse_args()
    if args.resume is not None and args.init_favit is not None:
        raise ValueError("--resume and --init-favit are mutually exclusive")
    config = load_config(args.config)
    data_config = config["data"]
    model_config = config["model"]
    train_config = config["train"]
    loss_config = config["loss"]
    validate_model_config(model_config)
    branch_config = resolve_branch_config(model_config)

    resume_value = args.resume or train_config.get("resume")
    resume_path = Path(resume_value) if resume_value else None
    init_value = args.init_favit or train_config.get("init_favit")
    init_path = Path(init_value) if init_value else None
    if resume_path is not None and init_path is not None:
        raise ValueError("resume and FA-ViT initialization are mutually exclusive")
    for path, name in ((resume_path, "resume"), (init_path, "FA-ViT initialization")):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"{name} checkpoint does not exist: {path}")
    resume_checkpoint = None
    if resume_path is not None:
        resume_checkpoint = torch.load(
            resume_path, map_location="cpu", weights_only=False
        )
        validate_checkpoint_branches(
            resume_checkpoint, model_config, resume_path
        )

    seed_everything(int(config.get("seed", 42)))
    device = resolve_device(args.device or config.get("device", "cuda"))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    methods = tuple(model_config.get(
        "forgery_methods", ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    ))
    augmentation = data_config.get("train_augmentation", {})
    train_transform = FaceTransform(
        image_size=int(data_config.get("image_size", 224)),
        horizontal_flip=float(
            augmentation.get(
                "horizontal_flip", data_config.get("horizontal_flip", 0.0)
            )
        ),
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
        enable_srm=branch_config.enable_srm,
        enable_fft=branch_config.enable_fft,
    )
    train_dataset = GroupedForgeryDataset(
        data_config["train_pairs"], data_config["root"], train_transform, methods
    )
    print(
        f"lsda_groups: complete={len(train_dataset)} "
        f"dropped_incomplete={train_dataset.dropped_incomplete_groups}"
    )
    group_batch_size = int(train_config.get("group_batch_size", 4))
    if group_batch_size < 2:
        raise ValueError("group_batch_size must be at least 2 for LSDA hard-example mining")
    num_workers = int(data_config.get("num_workers", 8))
    train_loader = DataLoader(
        train_dataset,
        batch_size=group_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    clean_transform = FaceTransform(
        int(data_config.get("image_size", 224)),
        enable_srm=branch_config.enable_srm,
        enable_fft=branch_config.enable_fft,
    )
    selection_manifest = data_config.get("validation_frames")
    if selection_manifest:
        selection_name = "validation"
    else:
        selection_manifest = data_config["celebdf_test_frames"]
        selection_name = "celebdf_test"
        print(
            "warning: data.validation_frames is not configured; selecting the best "
            "checkpoint on Celeb-DF test AUC leaks target-domain information"
        )
    selection_loader = make_frame_loader(
        selection_manifest,
        data_config,
        clean_transform,
        int(train_config.get("eval_image_batch_size", 32)),
        device,
    )
    target_manifest = data_config.get("celebdf_test_frames")
    target_loader = (
        make_frame_loader(
            target_manifest,
            data_config,
            clean_transform,
            int(train_config.get("eval_image_batch_size", 32)),
            device,
        )
        if target_manifest and str(target_manifest) != str(selection_manifest)
        else None
    )

    model = build_model_from_config(
        model_config, pretrained=False if resume_path or init_path else None
    )
    if init_path is not None:
        loaded = load_favit_initialization(model, init_path)
        print(f"init_favit_checkpoint: path={init_path} compatible_tensors={loaded}")
    model = model.to(device)
    print(json.dumps(model.trainable_parameter_summary(), indent=2))
    optimizer = build_optimizer(model, train_config)
    scheduler = build_scheduler(optimizer, train_config)
    fal_criterion = FineGrainedAdaptiveLoss(
        scale=float(loss_config.get("fal_scale", 24.0)),
        margin=float(loss_config.get("fal_margin", 0.25)),
    )
    amp_enabled = bool(train_config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    start_epoch = 0
    best_auc = float("-inf")
    epochs_without_improvement = 0
    if resume_path is not None:
        checkpoint = resume_checkpoint
        assert checkpoint is not None
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_auc = float(checkpoint.get("best_selection_auc", best_auc))
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        restore_random_state(checkpoint.get("random_state"))
        resume_checkpoint = None
        del checkpoint
        print(f"resume_checkpoint: path={resume_path} next_epoch={start_epoch + 1}")

    history_path = output_dir / "history.jsonl"
    for epoch in range(start_epoch, int(train_config["epochs"])):
        lsda_warmup = int(loss_config.get("lsda_warmup_epochs", 0))
        lsda_ramp = int(loss_config.get("lsda_ramp_epochs", 0))
        invariance_warmup = int(
            loss_config.get("invariance_warmup_epochs", lsda_warmup)
        )
        invariance_ramp = int(
            loss_config.get("invariance_ramp_epochs", lsda_ramp)
        )
        fal_warmup = int(loss_config.get("fal_warmup_epochs", 1))
        fal_ramp = int(loss_config.get("fal_ramp_epochs", 0))
        weights = {
            "binary": float(loss_config.get("binary_weight", 0.5)),
            "domain": ramped_weight(
                epoch,
                float(loss_config.get("domain_weight", 1.0)),
                lsda_warmup,
                lsda_ramp,
            ),
            "distill": ramped_weight(
                epoch,
                float(loss_config.get("distill_weight", 1.0)),
                lsda_warmup,
                lsda_ramp,
            ),
            "invariance": ramped_weight(
                epoch,
                float(loss_config.get("invariance_weight", 0.0)),
                invariance_warmup,
                invariance_ramp,
            ),
            "fal": ramped_weight(
                epoch,
                float(loss_config.get("fal_weight_after_warmup", 1.0)),
                fal_warmup,
                fal_ramp,
            ),
        }
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            fal_criterion,
            weights,
            device,
            scaler,
            label_smoothing=float(loss_config.get("label_smoothing", 0.0)),
            max_grad_norm=float(train_config.get("max_grad_norm", 0.0)) or None,
        )
        selection_metrics = evaluate_at_level(
            model,
            selection_loader,
            device,
            description=f"evaluate {selection_name}",
        )
        record = {
            "epoch": epoch + 1,
            "learning_rates": sorted(
                {float(group["lr"]) for group in optimizer.param_groups}
            ),
            "loss_weights": weights,
            "train": train_metrics,
            selection_name: selection_metrics,
        }
        print(json.dumps(record, indent=2))
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        current_auc = float(selection_metrics["auc"])
        scheduler.step()
        improved = current_auc > best_auc
        best_auc = max(best_auc, current_auc)
        epochs_without_improvement = (
            0 if improved else epochs_without_improvement + 1
        )
        state = {
            **model_branch_metadata(model),
            # Read off the model instance rather than the config-derived locals
            # so the persisted metadata is definitionally what the model is,
            # and cannot drift from it via a second resolution site.
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_auc": best_auc,
            "best_selection_auc": best_auc,
            "selection_name": selection_name,
            "selection_metrics": selection_metrics,
            "celebdf_test_metrics": None,
            "epochs_without_improvement": epochs_without_improvement,
            "random_state": capture_random_state(),
            "config": config,
        }
        if improved:
            save_checkpoint(output_dir / "best.pt", state)
            print(
                f"save_best_checkpoint: epoch={epoch + 1} "
                f"{selection_name}_auc={current_auc:.6f}"
            )
        save_checkpoint(output_dir / "last.pt", state)
        patience = int(train_config.get("early_stopping_patience", 0))
        if patience > 0 and epochs_without_improvement >= patience:
            print(
                f"early_stopping: no {selection_name} AUC improvement for "
                f"{patience} epochs"
            )
            break

    # Evaluate an external target only once, after source-validation model
    # selection is complete. This prevents target AUC from steering epochs.
    if target_loader is not None:
        best_path = output_dir / "best.pt"
        if not best_path.is_file():
            raise FileNotFoundError("no best checkpoint is available for target evaluation")
        best_state = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(best_state["model"])
        target_metrics = evaluate_at_level(
            model, target_loader, device, description="final test CelebDF"
        )
        best_state["celebdf_test_metrics"] = target_metrics
        save_checkpoint(best_path, best_state)
        final_record = {"event": "final_target_evaluation", "celebdf_test": target_metrics}
        print(json.dumps(final_record, indent=2))
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(final_record) + "\n")


if __name__ == "__main__":
    main()
