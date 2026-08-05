"""Validation utilities for a caller-supplied sequence of batches."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from standalone_nnunet2d.metrics.segmentation_metrics import binary_segmentation_metrics


@dataclass(frozen=True)
class ValidationEpochResult:
    batch_count: int
    mean_loss: float
    dice: float
    iou: float


def _full_resolution_logits(outputs: Tensor | Iterable[Tensor]) -> Tensor:
    """Select the full-resolution logits from a regular or deep-supervision output."""
    if isinstance(outputs, Tensor):
        return outputs

    try:
        logits = next(iter(outputs))
    except (StopIteration, TypeError) as exc:
        raise ValueError("validation model output must contain logits") from exc
    if not isinstance(logits, Tensor):
        raise ValueError("validation model output logits must be a Tensor")
    return logits


def _validate_logits_and_target(logits: Tensor, target: Tensor) -> None:
    if logits.ndim != 4:
        raise ValueError(f"validation logits must be 4D (N, C, H, W), got shape {tuple(logits.shape)}")
    if logits.shape[1] != 2:
        raise ValueError(f"validation logits must have 2 channels, got shape {tuple(logits.shape)}")
    if target.ndim != 3:
        raise ValueError(f"validation target must be 3D (N, H, W), got shape {tuple(target.shape)}")
    if logits.shape[0] != target.shape[0] or logits.shape[2:] != target.shape[1:]:
        raise ValueError(
            "validation logits and target batch/spatial shapes must match: "
            f"got logits {tuple(logits.shape)} and target {tuple(target.shape)}"
        )


def run_validation_epoch(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
    loss_fn: nn.Module,
    device: torch.device,
) -> ValidationEpochResult:
    """Evaluate binary segmentation batches without changing model parameters or mode."""
    was_training = model.training
    try:
        model.eval()
        batch_count = 0
        total_loss = 0.0
        true_positive = 0
        false_positive = 0
        false_negative = 0

        with torch.no_grad():
            for batch in batches:
                image, target = (value.to(device) for value in batch)
                logits = _full_resolution_logits(model(image))
                _validate_logits_and_target(logits, target)

                loss = loss_fn(logits, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite validation loss")

                metrics = binary_segmentation_metrics(
                    torch.argmax(logits, dim=1).cpu().numpy(),
                    target.cpu().numpy(),
                )
                true_positive += int(metrics["TP"])
                false_positive += int(metrics["FP"])
                false_negative += int(metrics["FN"])
                total_loss += float(loss.cpu())
                batch_count += 1

        if batch_count == 0:
            raise ValueError("empty validation batches")

        dice_denominator = 2 * true_positive + false_positive + false_negative
        iou_denominator = true_positive + false_positive + false_negative
        return ValidationEpochResult(
            batch_count=batch_count,
            mean_loss=total_loss / batch_count,
            dice=1.0 if dice_denominator == 0 else 2 * true_positive / dice_denominator,
            iou=1.0 if iou_denominator == 0 else true_positive / iou_denominator,
        )
    finally:
        model.train(was_training)
