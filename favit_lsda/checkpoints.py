"""Checkpoint metadata validation shared by training and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

EXPECTED_ARCHITECTURE = "favit_lsda_multibranch"
SUPPORTED_FORMAT_VERSION = 5
#: On-disk formats this build can name but not read. Each predates a fixed-slot
#: fusion width change, so their ``late_fusion`` tensors cannot be transplanted.
LEGACY_FORMAT_VERSIONS = (3, 4)


def model_branch_metadata(model) -> dict[str, Any]:
    """Return the authoritative v5 branch metadata for a constructed model."""
    return {
        "format_version": SUPPORTED_FORMAT_VERSION,
        "architecture": EXPECTED_ARCHITECTURE,
        "enabled_branches": list(model.enabled_branches),
        "srm_backbone": model.srm_backbone_name,
        "fft_backbone": model.fft_backbone_name,
        "fusion": model.fusion_name,
    }


def validate_checkpoint_branches(
    checkpoint: dict[str, Any],
    model_config: dict[str, Any],
    checkpoint_path: Path,
) -> None:
    """Reject incompatible multibranch checkpoints before state loading."""
    architecture = checkpoint.get("architecture")
    format_version = checkpoint.get("format_version")
    if architecture != EXPECTED_ARCHITECTURE:
        if format_version in LEGACY_FORMAT_VERSIONS or architecture == "favit_lsda_cnn":
            raise ValueError(
                f"checkpoint at {checkpoint_path} has legacy architecture "
                f"{architecture!r} (legacy format_version {format_version!r}); "
                "migrate it by starting a new run and loading it with "
                "--init-favit instead of --resume or evaluation."
            )
        raise ValueError(
            f"checkpoint at {checkpoint_path} has unsupported architecture "
            f"{architecture!r}; expected {EXPECTED_ARCHITECTURE!r}"
        )
    if format_version in LEGACY_FORMAT_VERSIONS:
        raise ValueError(
            f"checkpoint at {checkpoint_path} is a legacy format_version "
            f"{format_version} checkpoint; migrate it by starting a new run "
            "and loading it with --init-favit instead of --resume or "
            "evaluation."
        )
    if format_version != SUPPORTED_FORMAT_VERSION:
        raise ValueError(
            f"checkpoint at {checkpoint_path} has unsupported format_version "
            f"{format_version!r}; this build only reads format_version "
            f"{SUPPORTED_FORMAT_VERSION}. Use a build matching the checkpoint, "
            "or start a new run and load it with --init-favit."
        )

    from .config import resolve_branch_config

    branches = resolve_branch_config(model_config)
    expected = {
        "enabled_branches": list(branches.enabled_branches),
        "srm_backbone": branches.srm_backbone if branches.enable_srm else None,
        "fft_backbone": branches.fft_backbone if branches.enable_fft else None,
        "fusion": "fixed_slot_concat",
    }
    actual = {key: checkpoint.get(key) for key in expected}
    if actual != expected:
        raise ValueError(
            "checkpoint/config branch mismatch: "
            f"checkpoint={actual!r}, config={expected!r}, path={checkpoint_path}"
        )
