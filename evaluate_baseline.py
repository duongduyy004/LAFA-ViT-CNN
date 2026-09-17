from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from favit_lsda.baseline import build_baseline_model, validate_baseline_checkpoint
from favit_lsda.config import load_config, resolve_device
from favit_lsda.data import FaceTransform, FrameFaceDataset
from favit_lsda.engine import evaluate_at_level


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an RGB timm baseline on an FF++ frame manifest"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--level", choices=("frame", "video"), default="video")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")
    config = load_config(args.config)
    data_config = config["data"]
    checkpoint_path = args.checkpoint or Path(config["output_dir"]) / "best.pt"
    manifest = args.manifest or Path(data_config["ffpp_test_frames"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    if not manifest.is_file():
        raise FileNotFoundError(f"FF++ test manifest does not exist: {manifest}")
    device = resolve_device(args.device or config.get("device", "cuda"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_baseline_checkpoint(checkpoint, config["model"], checkpoint_path)
    model = build_baseline_model(config["model"], pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device)
    dataset = FrameFaceDataset(
        manifest,
        data_config["root"],
        FaceTransform(int(data_config.get("image_size", 224))),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or int(config["train"].get("eval_batch_size", 32)),
        shuffle=False,
        num_workers=(
            args.num_workers
            if args.num_workers is not None
            else int(data_config.get("num_workers", 8))
        ),
        pin_memory=device.type == "cuda",
    )
    metrics = evaluate_at_level(
        model,
        loader,
        device,
        level=args.level,
        threshold=args.threshold,
        description=f"test FF++ ({args.level})",
    )
    result = {
        "dataset": "FaceForensics++",
        "backbone": config["model"]["backbone"],
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest),
        **metrics,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
