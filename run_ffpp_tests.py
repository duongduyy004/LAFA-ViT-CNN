from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from favit_lsda.config import load_config

CONFIGS_DIR = Path("configs")
DEFAULT_MANIFEST = (
    r"E:\Deepfake_Data_Chien\ffpp_celebdf_data\processed\manifests\ffpp_c23_test_frames.csv"
)

CASES = [
    "favit_lsda_rgb.yaml",
    "favit_lsda_rgb_srm.yaml",
    "favit_lsda_rgb_fft.yaml",
    "favit_lsda_rgb_srm_fft.yaml",
]


def output_dir_of(config_path: Path) -> Path:
    output_dir = load_config(config_path).get("output_dir")
    if not output_dir:
        raise ValueError(f"output_dir not found in {config_path}")
    return Path(str(output_dir))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run evaluate_ffpp.py for one or every case config"
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--level", default="video", choices=("frame", "video"))
    parser.add_argument("--checkpoint-name", default="best.pt")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--case",
        action="append",
        choices=CASES,
        dest="cases",
        metavar="CONFIG.yaml",
        help=(
            "Restrict the run to one case config filename (repeatable). "
            "Default: run every case in CASES."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cases = args.cases if args.cases else CASES

    failures = []
    for name in cases:
        config_path = CONFIGS_DIR / name
        if not config_path.is_file():
            print(f"[skip] config not found: {config_path}")
            continue

        out_dir = output_dir_of(config_path)
        checkpoint = out_dir / args.checkpoint_name
        result_file = out_dir / f"ffpp_test_result_{args.level}.json"

        print(f"\n=== {name} ===")
        if not checkpoint.is_file():
            print(f"[skip] checkpoint not found: {checkpoint}")
            failures.append(name)
            continue

        cmd = [
            args.python,
            "evaluate_ffpp.py",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--manifest",
            args.manifest,
            "--level",
            args.level,
        ]
        print("$", " ".join(cmd))
        # stderr inherited (not captured) so tqdm's progress bar renders live;
        # only stdout (the final JSON blob) is piped back for saving.
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        print(proc.stdout)
        if proc.returncode != 0:
            failures.append(name)
            result_file.write_text(
                json.dumps({"case": name, "error": "non-zero exit code"}, indent=2),
                encoding="utf-8",
            )
            continue

        # evaluate_ffpp.py prints a JSON blob to stdout; take the last
        # top-level JSON object in case of extra prints.
        stdout = proc.stdout.strip()
        json_start = stdout.rfind("\n{")
        json_text = stdout[json_start + 1 :] if json_start != -1 else stdout
        try:
            metrics = json.loads(json_text)
        except json.JSONDecodeError:
            metrics = {"raw_stdout": stdout}

        result_file.parent.mkdir(parents=True, exist_ok=True)
        result_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"[saved] {result_file}")

    if failures:
        print(f"\nFailed/skipped cases: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
