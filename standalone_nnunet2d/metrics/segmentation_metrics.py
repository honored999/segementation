"""Strict, explicit metrics for already-discrete binary segmentation masks."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

METRIC_POLICY: dict[str, int | str] = {
    "foreground": 1,
    "postprocessing": "argmax",
    "both_empty": "dice=1",
    "one_empty": "dice=0",
    "aggregation": "case_macro_mean",
}


def binary_segmentation_metrics(prediction: ArrayLike, reference: ArrayLike) -> dict[str, float | int]:
    """Return confusion counts, Dice, and IoU for equal-shaped 0/1 masks."""
    predicted_mask = np.asarray(prediction)
    reference_mask = np.asarray(reference)
    if predicted_mask.shape != reference_mask.shape:
        raise ValueError("prediction and reference must have equal shapes")
    if not np.isin(predicted_mask, (0, 1)).all() or not np.isin(reference_mask, (0, 1)).all():
        raise ValueError("prediction and reference masks must contain only 0 and 1")
    true_positive = int(((predicted_mask == 1) & (reference_mask == 1)).sum())
    false_positive = int(((predicted_mask == 1) & (reference_mask == 0)).sum())
    false_negative = int(((predicted_mask == 0) & (reference_mask == 1)).sum())
    true_negative = int(((predicted_mask == 0) & (reference_mask == 0)).sum())
    dice_denominator = 2 * true_positive + false_positive + false_negative
    iou_denominator = true_positive + false_positive + false_negative
    return {
        "TP": true_positive,
        "FP": false_positive,
        "FN": false_negative,
        "TN": true_negative,
        "Dice": 1.0 if dice_denominator == 0 else 2 * true_positive / dice_denominator,
        "IoU": 1.0 if iou_denominator == 0 else true_positive / iou_denominator,
    }


def case_metric_record(case_id: str, prediction: ArrayLike, reference: ArrayLike) -> dict[str, str | float | int]:
    """Attach a required case identity to JSON-safe binary metrics."""
    if not case_id:
        raise ValueError("case_id must not be empty")
    return {"case_id": case_id, **binary_segmentation_metrics(prediction, reference)}
