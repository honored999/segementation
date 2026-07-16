"""Small, selected diagnostic image and optical-kernel visualizations."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import random
from .metrics import dice_score

def select_representative_rows(rows: list[dict], seed: int = 2026, limit: int = 6) -> list[dict]:
    """Select unique largest/smallest/lowest-Dice/false-positive representative slices."""
    if not rows: return []
    area = lambda row: int(np.asarray(row["target"], bool).sum())
    false_positive = lambda row: int((np.asarray(row["prediction"], bool) & ~np.asarray(row["target"], bool)).sum())
    dice = lambda row: dice_score(row["target"], row["prediction"])
    positive = [row for row in rows if area(row) > 0]
    empty = [row for row in rows if area(row) == 0]
    candidates = []
    if positive: candidates += [max(positive, key=area), min(positive, key=area), min(positive, key=dice)]
    if empty: candidates.append(max(empty, key=false_positive))
    remaining = [row for row in rows if row not in candidates]; random.Random(seed).shuffle(remaining); candidates += remaining
    selected = []
    for row in candidates:
        if row["sample_id"] not in {item["sample_id"] for item in selected}: selected.append(row)
        if len(selected) == limit: break
    return selected

def select_random_rows(rows: list[dict], count: int, seed: int = 2026) -> list[dict]:
    """Return deterministic unique random rows without replacement."""
    return random.Random(seed).sample(rows, k=min(count, len(rows)))

def save_validation_grid(rows: list[dict], output: Path) -> None:
    """Save one DWI/GT/prediction row per representative validation slice."""
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(max(1, len(rows)), 3, figsize=(12, 4 * max(1, len(rows))), squeeze=False)
    for index, row in enumerate(rows):
        target, prediction = np.asarray(row["target"]).squeeze(), np.asarray(row["prediction"]).squeeze()
        image = np.asarray(row["image"]).squeeze(); label = f"{row['patient']} {row['timepoint']} z={row['slice_index']}\nGT={int(target.sum())} Pred={int(prediction.sum())} Dice={dice_score(target,prediction):.3f}"
        for axis, value, title in zip(axes[index], [image,target,prediction], ["DWI", "GT Mask", "Prediction"]): axis.imshow(value, cmap="gray"); axis.set_title(f"{title}\n{label}"); axis.axis("off")
    figure.tight_layout(); figure.savefig(output, dpi=150, bbox_inches="tight"); plt.close(figure)

def save_optical_kernels(weights: np.ndarray, output_dir: Path) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True); stats = {"min":float(weights.min()),"max":float(weights.max()),"mean":float(weights.mean()),"l1":float(np.abs(weights).sum()),"l2":float(np.sqrt((weights**2).sum()))}
    for index, kernel in enumerate(weights[:, 0]): plt.imsave(output_dir / f"kernel_{index:02d}.png", kernel, cmap="coolwarm")
    return stats

def save_prediction_panel(image: np.ndarray, target: np.ndarray, probability: np.ndarray, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True); figure, axes=plt.subplots(1,4,figsize=(12,3));
    for axis, array, title in zip(axes,[image,target,probability,probability>=.5],["DWI","GT","Probability","Prediction"]): axis.imshow(array,cmap="gray");axis.set_title(title);axis.axis("off")
    figure.tight_layout();figure.savefig(output,dpi=150);plt.close(figure)
