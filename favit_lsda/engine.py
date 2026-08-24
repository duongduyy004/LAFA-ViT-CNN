from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm import tqdm

from .losses import FineGrainedAdaptiveLoss, balanced_binary_cross_entropy
from .metrics import EvaluationLevel, evaluation_metrics


def move_branch_inputs(
    inputs: Mapping[str, Tensor], device: torch.device
) -> dict[str, Tensor]:
    """Move each enabled branch tensor without changing the mapping contract."""

    return {
        name: value.to(device, non_blocking=True)
        for name, value in inputs.items()
    }


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[tuple[Mapping[str, Tensor], Tensor]],
    optimizer: torch.optim.Optimizer,
    fal_criterion: FineGrainedAdaptiveLoss,
    weights: dict[str, float],
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    label_smoothing: float = 0.0,
    max_grad_norm: float | None = None,
) -> dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "binary": 0.0,
        "domain": 0.0,
        "invariance": 0.0,
        "distill": 0.0,
        "distill_real": 0.0,
        "distill_fake": 0.0,
        "fal": 0.0,
        "binary_accuracy": 0.0,
        "domain_accuracy": 0.0,
        "invariance_accuracy": 0.0,
    }
    batches = 0
    use_amp = scaler is not None and scaler.is_enabled()
    for grouped_inputs, domain_labels in tqdm(loader, desc="train", leave=False):
        grouped_inputs = move_branch_inputs(grouped_inputs, device)
        domain_labels = domain_labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            output = model.forward_group(grouped_inputs)
            logits = output["logits"]
            features = output["features"]
            binary_labels = (domain_labels > 0).long()
            binary_loss = balanced_binary_cross_entropy(
                logits.flatten(0, 1),
                binary_labels.flatten(),
                label_smoothing=label_smoothing,
            )
            domain_loss = F.cross_entropy(
                output["domain_logits"].flatten(0, 1),
                domain_labels.flatten(),
                label_smoothing=label_smoothing,
            )
            fake_domain_labels = domain_labels[:, 1:] - 1
            invariance_loss = F.cross_entropy(
                output["invariance_logits"].flatten(0, 1),
                fake_domain_labels.flatten(),
                label_smoothing=label_smoothing,
            )
            distill_loss = output["distill_real"] + output["distill_fake"]

            fake_domains = features.shape[1] - 1
            fake_features = features[:, 1:].reshape(-1, features.shape[-1])
            real_features = features[:, :1].expand(-1, fake_domains, -1).reshape_as(
                fake_features
            )
            fal_loss = fal_criterion(
                model.head.weight[0], real_features, fake_features
            )
            loss = (
                weights.get("binary", 0.5) * binary_loss
                + weights.get("domain", 1.0) * domain_loss
                + weights.get("invariance", 0.0) * invariance_loss
                + weights.get("distill", 1.0) * distill_loss
                + weights.get("fal", 1.0) * fal_loss
            )
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
        totals["loss"] += float(loss.detach())
        totals["binary"] += float(binary_loss.detach())
        totals["domain"] += float(domain_loss.detach())
        totals["invariance"] += float(invariance_loss.detach())
        totals["distill"] += float(distill_loss.detach())
        totals["distill_real"] += float(output["distill_real"].detach())
        totals["distill_fake"] += float(output["distill_fake"].detach())
        totals["fal"] += float(fal_loss.detach())
        totals["binary_accuracy"] += float(
            (logits.argmax(dim=-1) == binary_labels).float().mean()
        )
        totals["domain_accuracy"] += float(
            (output["domain_logits"].argmax(dim=-1) == domain_labels).float().mean()
        )
        totals["invariance_accuracy"] += float(
            (
                output["invariance_logits"].argmax(dim=-1) == fake_domain_labels
            ).float().mean()
        )
        batches += 1
    if batches == 0:
        raise ValueError("training loader produced no batches")
    return {key: value / batches for key, value in totals.items()}


@torch.inference_mode()
def evaluate_at_level(
    model: nn.Module,
    loader: Iterable[tuple[Mapping[str, Tensor], Tensor, list[str]]],
    device: torch.device,
    level: EvaluationLevel = "video",
    threshold: float = 0.5,
    description: str = "evaluate",
) -> dict[str, float | int | str]:
    """Run inference and calculate frame- or video-level binary metrics."""

    model.eval()
    probabilities: list[float] = []
    labels: list[int] = []
    video_ids: list[str] = []
    for inputs, batch_labels, batch_video_ids in tqdm(
        loader, desc=description, leave=False
    ):
        inputs = move_branch_inputs(inputs, device)
        logits = model(inputs)
        probabilities.extend(logits.softmax(dim=1)[:, 1].cpu().tolist())
        labels.extend(batch_labels.tolist())
        video_ids.extend(batch_video_ids)
    return evaluation_metrics(
        probabilities, labels, video_ids, level=level, threshold=threshold
    )
