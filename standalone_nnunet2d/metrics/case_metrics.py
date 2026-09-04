"""Whole-volume binary lesion metrics for smoke-only validation."""

from __future__ import annotations

import numpy as np


def volume_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    """Compute whole-volume metrics; zero denominators use explicit zero precision/recall."""
    if prediction.shape != target.shape or not np.isin(prediction, (0, 1)).all() or not np.isin(target, (0, 1)).all():
        raise ValueError("prediction and target must be equal-shaped binary volumes")
    tp = int(((prediction == 1) & (target == 1)).sum()); fp = int(((prediction == 1) & (target == 0)).sum()); fn = int(((prediction == 0) & (target == 1)).sum())
    denominator = 2 * tp + fp + fn; union = tp + fp + fn
    return {"dice": 1.0 if denominator == 0 else 2 * tp / denominator, "iou": 1.0 if union == 0 else tp / union, "precision": 0.0 if tp + fp == 0 else tp / (tp + fp), "recall": 0.0 if tp + fn == 0 else tp / (tp + fn), "gt_voxels": int((target == 1).sum()), "pred_voxels": int((prediction == 1).sum()), "tp": tp, "fp": fp, "fn": fn}
