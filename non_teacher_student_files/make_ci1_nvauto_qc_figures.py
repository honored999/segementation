"""Generate DWI/ADC/label QC preview figures for converted CI-1 cases."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


def normalize(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    lo, hi = np.percentile(array[np.isfinite(array)], [1, 99]) if np.any(np.isfinite(array)) else (0, 1)
    if hi <= lo:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - lo) / (hi - lo), 0, 1)


def overlay(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = np.stack([gray, gray, gray], axis=-1)
    out = base.copy()
    color = np.asarray([1.0, 0.1, 0.0], dtype=np.float32)
    alpha = 0.45
    selected = mask.astype(bool)
    out[selected] = (1 - alpha) * out[selected] + alpha * color
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make CI-1 NVAUTO QC figures.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument("--slices", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qc_dir = args.dataset_root / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    case_dirs = sorted((args.dataset_root / "cases").glob("case_*"))[: args.max_cases]
    for case_dir in case_dirs:
        dwi = sitk.GetArrayFromImage(sitk.ReadImage(str(case_dir / "dwi.nii.gz")))
        adc = sitk.GetArrayFromImage(sitk.ReadImage(str(case_dir / "adc.nii.gz")))
        label = sitk.GetArrayFromImage(sitk.ReadImage(str(case_dir / "label.nii.gz"))) != 0
        areas = label.reshape(label.shape[0], -1).sum(axis=1)
        indices = np.argsort(areas)[::-1][: args.slices]
        indices = [int(index) for index in indices if areas[index] > 0] or [int(np.argmax(areas))]
        fig, axes = plt.subplots(len(indices), 4, figsize=(12, 3 * len(indices)), squeeze=False)
        for row_idx, z in enumerate(indices):
            dwi_slice = normalize(dwi[z])
            adc_slice = normalize(adc[z])
            label_slice = label[z]
            panels = [dwi_slice, adc_slice, label_slice.astype(np.float32), overlay(dwi_slice, label_slice)]
            titles = [f"DWI z={z}", "ADC", "Label", "DWI + label"]
            for col_idx, panel in enumerate(panels):
                axes[row_idx, col_idx].imshow(panel, cmap=None if panel.ndim == 3 else "gray")
                axes[row_idx, col_idx].set_title(titles[col_idx])
                axes[row_idx, col_idx].axis("off")
        fig.tight_layout()
        output_path = qc_dir / f"{case_dir.name}_preview.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

