"""Training loop for conventional RGB-only binary classifiers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
from torch import Tensor, nn
from tqdm import tqdm

from .engine import move_branch_inputs


def train_baseline_one_epoch(
    model: nn.Module,
    loader: Iterable[tuple[Mapping[str, Tensor], Tensor, list[str]]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    max_grad_norm: float | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    use_amp = scaler is not None and scaler.is_enabled()
    for inputs, labels, _video_ids in tqdm(loader, desc="train", leave=False):
        inputs = move_branch_inputs(inputs, device)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(inputs)
            loss = criterion(logits, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            if max_grad_norm is not None and max_grad_norm > 0.0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm is not None and max_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum())
        total_samples += batch_size
    if total_samples == 0:
        raise ValueError("baseline training loader produced no samples")
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "num_frames": total_samples,
    }
