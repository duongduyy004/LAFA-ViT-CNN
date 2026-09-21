from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


CONFIGS = (
    "efficientnet_b4.yaml",
    "resnet50.yaml",
    "vit_b16.yaml",
    "swin_t.yaml",
    "xception.yaml",
)
CONFIG_DIR = Path("configs/baselines")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test all RGB/SRM/FFT baseline best checkpoints on FF++"
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--level", choices=("frame", "video"), default="video")
    parser.add_argument("--case", action="append", choices=CONFIGS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failures = 0
    for name in args.case or CONFIGS:
        config_path = CONFIG_DIR / name
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        output_dir = Path(config["output_dir"])
        checkpoint = output_dir / "best.pt"
        if not checkpoint.is_file():
            print(f"skip_missing_checkpoint: {checkpoint}")
            continue
        command = [
            sys.executable,
            "evaluate_baseline.py",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--level",
            args.level,
        ]
        if args.manifest is not None:
            command.extend(("--manifest", str(args.manifest)))
        if args.device is not None:
            command.extend(("--device", args.device))
        completed = subprocess.run(command, text=True, capture_output=True)
        result_path = output_dir / f"ffpp_test_result_{args.level}.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        if completed.returncode == 0:
            result_path.write_text(completed.stdout, encoding="utf-8")
            print(completed.stdout.strip())
        else:
            failures += 1
            error = {
                "config": str(config_path),
                "checkpoint": str(checkpoint),
                "error": completed.stderr.strip() or completed.stdout.strip(),
            }
            result_path.write_text(json.dumps(error, indent=2), encoding="utf-8")
            print(json.dumps(error, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
