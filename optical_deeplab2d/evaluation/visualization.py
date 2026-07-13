"""Small, selected diagnostic image and optical-kernel visualizations."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def save_optical_kernels(weights: np.ndarray, output_dir: Path) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True); stats = {"min":float(weights.min()),"max":float(weights.max()),"mean":float(weights.mean()),"l1":float(np.abs(weights).sum()),"l2":float(np.sqrt((weights**2).sum()))}
    for index, kernel in enumerate(weights[:, 0]): plt.imsave(output_dir / f"kernel_{index:02d}.png", kernel, cmap="coolwarm")
    return stats

def save_prediction_panel(image: np.ndarray, target: np.ndarray, probability: np.ndarray, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True); figure, axes=plt.subplots(1,4,figsize=(12,3));
    for axis, array, title in zip(axes,[image,target,probability,probability>=.5],["DWI","GT","Probability","Prediction"]): axis.imshow(array,cmap="gray");axis.set_title(title);axis.axis("off")
    figure.tight_layout();figure.savefig(output,dpi=150);plt.close(figure)
