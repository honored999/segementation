"""Create comparison figures for DeepISLES prediction outputs.

中文说明：这个脚本用于查看 DeepISLES 输出的 lesion_msk.nii.gz。
如果提供 DWI 原图和人工 GT label，会生成 DWI / GT / DeepISLES prediction
的对比图；如果没有提供原图和 GT，则生成 prediction-only 预览图。
注意：DeepISLES 输出只用于可视化对比，不能作为训练标签。
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


DEFAULT_DEEPISLES_OUTPUT = Path("E:/学习/研一/work14-图像分割/deepisles_output")
DEFAULT_OUTPUT_DIR = Path("results") / "deepisles_prediction_comparisons"


def ascii_nifti_name(path: Path) -> str:
    """Return an ASCII temporary name preserving NIfTI extension."""
    lower_name = path.name.lower()
    if lower_name.endswith(".nii.gz"):
        return "image.nii.gz"
    if lower_name.endswith(".nii"):
        return "image.nii"
    return "image.nii.gz"


def read_nifti(path: Path) -> sitk.Image:
    """Read NIfTI robustly on Windows paths with non-ASCII characters."""
    try:
        return sitk.ReadImage(str(path))
    except RuntimeError:
        with tempfile.TemporaryDirectory(prefix="deepisles_nifti_") as tmp_dir:
            tmp_path = Path(tmp_dir) / ascii_nifti_name(path)
            shutil.copy2(path, tmp_path)
            return sitk.ReadImage(str(tmp_path))


def image_to_array(path: Path) -> np.ndarray:
    """Read a NIfTI file into a z/y/x numpy array."""
    return sitk.GetArrayFromImage(read_nifti(path))


def image_geometry(image: sitk.Image) -> str:
    """Return a compact geometry description for logging."""
    return (
        f"size={image.GetSize()} "
        f"spacing={tuple(round(value, 6) for value in image.GetSpacing())} "
        f"origin={tuple(round(value, 6) for value in image.GetOrigin())} "
        f"direction={tuple(round(value, 6) for value in image.GetDirection())}"
    )


def geometry_matches(image: sitk.Image, reference: sitk.Image, tol: float = 1e-5) -> bool:
    """Return True when image geometry matches the reference image."""
    if image.GetSize() != reference.GetSize():
        return False
    checks = [
        np.allclose(image.GetSpacing(), reference.GetSpacing(), atol=tol),
        np.allclose(image.GetOrigin(), reference.GetOrigin(), atol=tol),
        np.allclose(image.GetDirection(), reference.GetDirection(), atol=tol),
    ]
    return all(checks)


def resample_mask_to_reference(
    mask: sitk.Image,
    reference: sitk.Image,
    name: str,
) -> sitk.Image:
    """Resample a mask-like image into reference physical space."""
    if geometry_matches(mask, reference):
        print(f"{name} geometry already matches DWI.")
        return mask
    print(f"[WARN] {name} geometry differs from DWI, resampling with nearest neighbor.")
    print(f"  {name}: {image_geometry(mask)}")
    print(f"  DWI: {image_geometry(reference)}")
    return sitk.Resample(
        mask,
        reference,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        mask.GetPixelID(),
    )


def normalize_slice(image: np.ndarray) -> np.ndarray:
    """Normalize a 2D image slice to 0..1."""
    image = image.astype(np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    low, high = np.percentile(finite, [1.0, 99.0])
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def choose_prediction_path(deepisles_output: Path, use_xy_fixed: bool) -> Path:
    """Choose DeepISLES prediction mask path."""
    if use_xy_fixed:
        candidate = deepisles_output / "lesion_msk_xy_fixed.nii.gz"
        if candidate.exists():
            return candidate
    candidate = deepisles_output / "lesion_msk.nii.gz"
    if candidate.exists():
        return candidate
    nested = deepisles_output / "output_teams" / "nvauto" / "lesion_msk.nii.gz"
    if nested.exists():
        return nested
    raise FileNotFoundError(f"No DeepISLES lesion_msk.nii.gz found in {deepisles_output}")


def largest_positive_slices(mask: np.ndarray, max_slices: int) -> list[int]:
    """Return z slices with largest nonzero prediction area."""
    mask = mask != 0
    areas = mask.reshape(mask.shape[0], -1).sum(axis=1)
    positive = [index for index, area in enumerate(areas) if area > 0]
    ranked = sorted(positive, key=lambda index: areas[index], reverse=True)
    return sorted(ranked[:max_slices])


def resize_nearest(slice_2d: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a 2D array with nearest-neighbor indexing."""
    if slice_2d.shape == shape:
        return slice_2d
    rows = np.linspace(0, slice_2d.shape[0] - 1, shape[0]).round().astype(int)
    cols = np.linspace(0, slice_2d.shape[1] - 1, shape[1]).round().astype(int)
    return slice_2d[np.ix_(rows, cols)]


def overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Overlay a red mask on grayscale image."""
    base = normalize_slice(image)
    mask = resize_nearest(mask.astype(bool), base.shape)
    rgb = np.stack([base, base, base], axis=-1)
    alpha = 0.45
    red = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * red
    return np.clip(rgb, 0.0, 1.0)


def draw_prediction_only(prediction: np.ndarray, output_path: Path, max_slices: int) -> None:
    """Draw DeepISLES prediction-only grid."""
    indices = largest_positive_slices(prediction, max_slices)
    if not indices:
        indices = list(range(min(max_slices, prediction.shape[0])))
    fig, axes = plt.subplots(len(indices), 1, figsize=(3, 3 * len(indices)))
    axes = np.asarray(axes).reshape(-1)
    for axis, z_index in zip(axes, indices):
        axis.imshow(prediction[z_index] != 0, cmap="gray")
        axis.set_title(f"DeepISLES prediction z={z_index}")
        axis.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_full_comparison(
    dwi: np.ndarray,
    label: np.ndarray,
    prediction: np.ndarray,
    output_path: Path,
    max_slices: int,
) -> None:
    """Draw DWI/GT/DeepISLES prediction/overlay comparison grid."""
    label_mask = label != 0
    prediction_mask = prediction != 0
    selection_mask = label_mask if label_mask.any() else (label_mask | prediction_mask)
    indices = largest_positive_slices(selection_mask.astype(np.uint8), max_slices)
    if not indices:
        indices = list(range(min(max_slices, prediction.shape[0])))

    fig, axes = plt.subplots(len(indices), 4, figsize=(12, 3 * len(indices)))
    if len(indices) == 1:
        axes = axes.reshape(1, -1)

    for row, z_index in enumerate(indices):
        z_index = min(z_index, dwi.shape[0] - 1, label.shape[0] - 1, prediction.shape[0] - 1)
        image_slice = dwi[z_index]
        label_slice = resize_nearest((label[z_index] != 0), image_slice.shape)
        pred_slice = resize_nearest((prediction[z_index] != 0), image_slice.shape)
        panels = [
            (normalize_slice(image_slice), "DWI"),
            (label_slice, "GT Mask"),
            (pred_slice, "DeepISLES"),
            (overlay_mask(image_slice, pred_slice), "Prediction Overlay"),
        ]
        for col, (array, title) in enumerate(panels):
            axes[row, col].imshow(array, cmap="gray" if array.ndim == 2 else None)
            axes[row, col].set_title(f"{title}\nz={z_index}")
            axes[row, col].axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create comparison figures for DeepISLES output."
    )
    parser.add_argument("--deepisles-output", type=Path, default=DEFAULT_DEEPISLES_OUTPUT)
    parser.add_argument("--dwi-path", type=Path, default=None)
    parser.add_argument("--label-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-slices", type=int, default=6)
    parser.add_argument(
        "--use-xy-fixed",
        action="store_true",
        help="Use lesion_msk_xy_fixed.nii.gz when present.",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    prediction_path = choose_prediction_path(args.deepisles_output, args.use_xy_fixed)
    prediction_image = read_nifti(prediction_path)
    prediction = sitk.GetArrayFromImage(prediction_image)

    print(f"DeepISLES prediction: {prediction_path}")
    print(f"prediction {image_geometry(prediction_image)}")
    print(f"shape={prediction.shape}, positive_voxels={int((prediction != 0).sum())}")

    if args.dwi_path is None or args.label_path is None:
        output_path = args.output_dir / "deepisles_prediction_only.png"
        draw_prediction_only(prediction, output_path, args.max_slices)
        print(f"Wrote prediction-only figure: {output_path}")
        print("Pass --dwi-path and --label-path to create DWI/GT/prediction comparison.")
        return

    dwi_image = read_nifti(args.dwi_path)
    label_image = read_nifti(args.label_path)
    print(f"DWI image: {args.dwi_path}")
    print(f"DWI {image_geometry(dwi_image)}")
    print(f"GT label: {args.label_path}")
    print(f"GT {image_geometry(label_image)}")

    label_image = resample_mask_to_reference(label_image, dwi_image, "GT")
    prediction_image = resample_mask_to_reference(prediction_image, dwi_image, "DeepISLES")
    dwi = sitk.GetArrayFromImage(dwi_image)
    label = sitk.GetArrayFromImage(label_image)
    prediction = sitk.GetArrayFromImage(prediction_image)
    output_path = args.output_dir / "deepisles_prediction_comparison.png"
    draw_full_comparison(dwi, label, prediction, output_path, args.max_slices)
    print(f"Wrote DeepISLES comparison figure: {output_path}")


if __name__ == "__main__":
    main()
