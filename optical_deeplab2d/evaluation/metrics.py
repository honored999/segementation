from __future__ import annotations
import numpy as np
def dice_score(target: np.ndarray, prediction: np.ndarray) -> float:
    """Binary Dice with explicit empty-mask rules."""
    target, prediction = np.asarray(target, dtype=bool), np.asarray(prediction, dtype=bool)
    if not target.any() or not prediction.any(): return 1.0 if not target.any() and not prediction.any() else 0.0
    return float(2 * np.logical_and(target, prediction).sum() / (target.sum() + prediction.sum()))

def binary_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    """Compute pixel-level binary segmentation metrics with safe zero denominators."""
    target, prediction = np.asarray(target, bool).ravel(), np.asarray(prediction, bool).ravel()
    tp, tn = int((target & prediction).sum()), int((~target & ~prediction).sum())
    fp, fn = int((~target & prediction).sum()), int((target & ~prediction).sum())
    safe = lambda numerator, denominator: float(numerator / denominator) if denominator else 0.0
    return {"dice": dice_score(target, prediction), "iou": safe(tp, tp + fp + fn), "precision": safe(tp, tp + fp), "recall": safe(tp, tp + fn), "specificity": safe(tn, tn + fp), "false_positive_pixels": fp, "predicted_lesion_area": int(prediction.sum()), "ground_truth_lesion_area": int(target.sum())}

def summarize_by_patient(rows: list[dict]) -> list[dict]:
    """Concatenate all slices from each patient before computing metrics."""
    grouped: dict[str, list[dict]] = {}
    for row in rows: grouped.setdefault(str(row["patient"]), []).append(row)
    return [{"patient": patient, **binary_metrics(np.concatenate([r["target"].ravel() for r in items]), np.concatenate([r["prediction"].ravel() for r in items]))} for patient, items in sorted(grouped.items())]
