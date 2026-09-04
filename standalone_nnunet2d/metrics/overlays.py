"""Case-level smoke-only diagnostic overlays."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def select_overlay_slice(target: np.ndarray, prediction: np.ndarray) -> int:
    """Prefer max GT lesion slice, then prediction, then central z slice."""
    target_counts = target.sum(axis=(1, 2)); prediction_counts = prediction.sum(axis=(1, 2))
    if target_counts.max() > 0: return int(target_counts.argmax())
    if prediction_counts.max() > 0: return int(prediction_counts.argmax())
    return target.shape[0] // 2


def write_overlay(path: Path, image: np.ndarray, target: np.ndarray, prediction: np.ndarray, *, case_id: str, dice: float) -> None:
    z_index = select_overlay_slice(target, prediction)
    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    for axis, data, title in zip(axes[:3], (image[z_index], target[z_index], prediction[z_index]), ("DWI", "GT", "Prediction")):
        axis.imshow(data, cmap="gray"); axis.set_title(title); axis.axis("off")
    axes[3].imshow(image[z_index], cmap="gray"); axes[3].imshow(target[z_index], cmap="Greens", alpha=0.45); axes[3].imshow(prediction[z_index], cmap="Reds", alpha=0.45); axes[3].set_title(f"{case_id} z={z_index} Dice={dice:.4f}"); axes[3].axis("off")
    path.parent.mkdir(parents=True, exist_ok=True); figure.savefig(path, dpi=150, bbox_inches="tight"); plt.close(figure)
