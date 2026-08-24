from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .checkpoints import validate_checkpoint_branches
from .config import build_model_from_config, load_config, resolve_device
from .data import FaceTransform, FrameFaceDataset
from .engine import evaluate_at_level


def build_evaluation_parser(
    description: str, manifest_help: str
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=None, help=manifest_help)
    parser.add_argument(
        "--level",
        choices=("frame", "video"),
        default="video",
        help="Metric aggregation level (default: video)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Fake-probability threshold for accuracy/F1/precision/recall",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser


def _load_model(
    checkpoint_path: Path, config: dict, device: torch.device
) -> nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    # Validate against the user-supplied --config, never against the config the
    # checkpoint carries: comparing the checkpoint's metadata to its own
    # embedded config is a self-comparison that can never fail, and would let
    # an evaluation run under a mode/width the caller never asked for.
    validate_checkpoint_branches(checkpoint, config["model"], checkpoint_path)
    # Only once the request is known to match may the checkpoint's own config
    # be trusted for construction; it is the more complete record of how the
    # model was actually built. Re-validate it against the checkpoint's
    # top-level metadata (already proven to match the caller's request above)
    # so a checkpoint whose nested config was edited out of step with its
    # top-level fields is still caught before construction.
    model_config = checkpoint.get("config", config)["model"]
    validate_checkpoint_branches(checkpoint, model_config, checkpoint_path)
    model = build_model_from_config(model_config, pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device)


def _resolve_manifest(
    explicit_manifest: Path | None,
    data_config: dict,
    default_keys: Sequence[str],
) -> Path:
    if explicit_manifest is not None:
        manifest = explicit_manifest
    else:
        configured = next(
            (data_config.get(key) for key in default_keys if data_config.get(key)),
            None,
        )
        if configured is None:
            keys = ", ".join(f"data.{key}" for key in default_keys)
            raise ValueError(
                f"no evaluation manifest is configured; set one of {keys} "
                "or pass --manifest"
            )
        manifest = Path(configured)
    if not manifest.is_file():
        raise FileNotFoundError(f"evaluation manifest does not exist: {manifest}")
    return manifest


def run_evaluation(
    args: argparse.Namespace,
    dataset_name: str,
    default_manifest_keys: Sequence[str],
) -> dict[str, float | int | str]:
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")

    config = load_config(args.config)
    device = resolve_device(args.device or config.get("device", "cuda"))
    data_config = config["data"]
    manifest = _resolve_manifest(
        args.manifest, data_config, default_manifest_keys
    )
    model = _load_model(args.checkpoint, config, device)
    enabled_branches = tuple(model.enabled_branches)

    dataset = FrameFaceDataset(
        manifest,
        data_config["root"],
        FaceTransform(
            int(data_config.get("image_size", 224)),
            enable_srm="srm" in enabled_branches,
            enable_fft="fft" in enabled_branches,
        ),
    )
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else int(config["train"].get("eval_image_batch_size", 32))
    )
    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else int(data_config.get("num_workers", 8))
    )
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    metrics = evaluate_at_level(
        model,
        loader,
        device,
        level=args.level,
        threshold=args.threshold,
        description=f"evaluate {dataset_name} ({args.level})",
    )
    result: dict[str, float | int | str] = {
        "dataset": dataset_name,
        "manifest": str(manifest),
        **metrics,
    }
    print(json.dumps(result, indent=2))
    return result
