"""Render a read-only Dataset501/Dataset504 coarse-to-fine case comparison."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from numbers import Real
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from standalone_nnunet2d.data.nifti_io import NiftiVolume, read_nifti


_GT_OVERLAY_CMAP = ListedColormap(
    [(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 1.0)]
)
_PRED_OVERLAY_CMAP = ListedColormap(
    [(0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 1.0)]
)


def select_slice_index(
    label: np.ndarray, slice_index: int | None = None
) -> int:
    """Select an axial slice from a ``(z, y, x)`` label volume.

    A supplied index takes precedence.  Otherwise, the slice with the largest
    number of positive label voxels is selected; an empty label falls back to
    the middle slice.
    """
    if label.ndim != 3:
        raise ValueError(f"label must be a 3D array ordered as (z, y, x), got {label.shape}")
    z_count = label.shape[0]
    if z_count == 0:
        raise ValueError("label must contain at least one z slice")

    if slice_index is not None:
        if not isinstance(slice_index, (int, np.integer)):
            raise ValueError(f"slice_index must be an integer or None, got {slice_index!r}")
        chosen = int(slice_index)
        if not 0 <= chosen < z_count:
            raise ValueError(
                f"slice_index must satisfy 0 <= index < {z_count}, got {chosen}"
            )
        return chosen

    lesion_area = np.sum(label > 0, axis=(1, 2), dtype=np.int64)
    if not np.any(lesion_area):
        fallback = z_count // 2
        print(
            "Dataset501 GT is empty; falling back to middle "
            f"z slice {fallback}."
        )
        return fallback
    return int(np.argmax(lesion_area))


def _expected_case_paths(root: Path, case_id: str) -> tuple[Path, Path]:
    return (
        root / "imagesTr" / f"{case_id}_0000.nii.gz",
        root / "labelsTr" / f"{case_id}.nii.gz",
    )


def _require_case_files(
    dataset501_root: Path, dataset504_root: Path, case_id: str
) -> tuple[tuple[Path, Path], tuple[Path, Path]]:
    dataset501_paths = _expected_case_paths(dataset501_root, case_id)
    dataset504_paths = _expected_case_paths(dataset504_root, case_id)
    missing = [
        path
        for path in (*dataset501_paths, *dataset504_paths)
        if not path.is_file()
    ]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required NIfTI file(s): {missing_text}")
    return dataset501_paths, dataset504_paths


def _read_and_validate_case(
    image_path: Path, label_path: Path, dataset_name: str
) -> tuple[NiftiVolume, NiftiVolume]:
    image = read_nifti(image_path)
    label = read_nifti(label_path)
    if image.array.shape != label.array.shape:
        raise ValueError(
            f"{dataset_name} image/label shape mismatch: "
            f"image={image.array.shape}, label={label.array.shape}"
        )
    return image, label


def _display_window(image_slice: np.ndarray) -> tuple[float, float]:
    finite_nonzero = image_slice[
        np.isfinite(image_slice) & (image_slice != 0)
    ]
    if finite_nonzero.size == 0:
        return 0.0, 1.0

    vmin, vmax = np.percentile(finite_nonzero, (1, 99))
    vmin = float(vmin)
    vmax = float(vmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        center = float(np.median(finite_nonzero))
        if not np.isfinite(center):
            return 0.0, 1.0
        return center - 0.5, center + 0.5
    return vmin, vmax


def _show_dwi(axis: plt.Axes, image_slice: np.ndarray, title: str) -> None:
    vmin, vmax = _display_window(image_slice)
    axis.imshow(image_slice, cmap="gray", vmin=vmin, vmax=vmax)
    axis.set_title(f"{title}\nshape={image_slice.shape}")
    axis.axis("off")


def _show_gt(axis: plt.Axes, label_slice: np.ndarray, title: str) -> None:
    axis.imshow(label_slice > 0, cmap="gray", vmin=0, vmax=1)
    axis.set_title(title)
    axis.axis("off")


def _show_overlay(
    axis: plt.Axes,
    image_slice: np.ndarray,
    label_slice: np.ndarray,
    title: str,
) -> None:
    vmin, vmax = _display_window(image_slice)
    axis.imshow(image_slice, cmap="gray", vmin=vmin, vmax=vmax)
    axis.imshow(
        label_slice > 0,
        cmap=_GT_OVERLAY_CMAP,
        vmin=0,
        vmax=1,
        alpha=0.35,
    )
    axis.set_title(title)
    axis.axis("off")


def roi_to_rectangle(
    roi: Sequence[int | float],
) -> tuple[int | float, int | float, int | float, int | float]:
    """Convert ``[x_min, y_min, x_max, y_max]`` to Matplotlib rectangle args."""
    if isinstance(roi, (str, bytes)) or len(roi) != 4:
        raise ValueError(f"ROI must contain four coordinates, got {roi!r}")
    x_min, y_min, x_max, y_max = roi
    return x_min, y_min, x_max - x_min, y_max - y_min


def _validate_bbox(
    bbox: object,
    *,
    name: str,
    original_shape: tuple[int, int, int],
) -> tuple[int, int, int, int]:
    if isinstance(bbox, (str, bytes)) or not isinstance(bbox, Sequence):
        raise ValueError(f"{name} must be a sequence of four numbers, got {bbox!r}")
    if len(bbox) != 4:
        raise ValueError(f"{name} must contain four numbers, got {bbox!r}")

    coordinates: list[int] = []
    for value in bbox:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must contain only numeric coordinates, got {bbox!r}")
        numeric = float(value)
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(
                f"{name} coordinates must be finite integer-valued numbers, got {bbox!r}"
            )
        coordinates.append(int(numeric))

    x_min, y_min, x_max, y_max = coordinates
    _, original_y, original_x = original_shape
    if not (0 <= x_min < x_max <= original_x and 0 <= y_min < y_max <= original_y):
        raise ValueError(
            f"{name} must be a non-empty box within original XY shape "
            f"(y={original_y}, x={original_x}), got {bbox!r}"
        )
    return x_min, y_min, x_max, y_max


def _load_manifest_rois(
    manifest_path: Path,
    *,
    case_id: str,
    original_shape: tuple[int, int, int],
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int] | None]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"ROI manifest does not exist: {manifest_path}")
    try:
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid ROI manifest JSON: {manifest_path}") from exc

    if not isinstance(manifest, dict) or not isinstance(manifest.get("cases"), dict):
        raise ValueError("ROI manifest must contain a cases dictionary")
    record = manifest["cases"].get(case_id)
    if not isinstance(record, dict):
        raise ValueError(f"ROI manifest has no cases record for {case_id!r}")

    if "final_roi" in record:
        final_roi = record["final_roi"]
    elif "roi" in record:
        final_roi = record["roi"]
    else:
        raise ValueError(
            f"ROI manifest record for {case_id!r} has neither final_roi nor roi"
        )
    validated_final_roi = _validate_bbox(
        final_roi, name="final_roi/roi", original_shape=original_shape
    )

    raw_bbox = record.get("raw_prediction_bbox")
    validated_raw_bbox = (
        None
        if raw_bbox is None
        else _validate_bbox(
            raw_bbox, name="raw_prediction_bbox", original_shape=original_shape
        )
    )
    return validated_final_roi, validated_raw_bbox


def _show_prediction(axis: plt.Axes, prediction_slice: np.ndarray, title: str) -> None:
    axis.imshow(prediction_slice > 0, cmap="gray", vmin=0, vmax=1)
    axis.set_title(title)
    axis.axis("off")


def _add_bbox(
    axis: plt.Axes,
    bbox: tuple[int, int, int, int],
    *,
    color: str,
    linestyle: str,
    linewidth: float,
) -> None:
    x_min, y_min, width, height = roi_to_rectangle(bbox)
    axis.add_patch(
        Rectangle(
            (x_min, y_min),
            width,
            height,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            linestyle=linestyle,
        )
    )


def _show_key_overlay(
    axis: plt.Axes,
    image_slice: np.ndarray,
    label_slice: np.ndarray,
    prediction_slice: np.ndarray,
    final_roi: tuple[int, int, int, int],
    raw_prediction_bbox: tuple[int, int, int, int] | None,
) -> None:
    vmin, vmax = _display_window(image_slice)
    axis.imshow(image_slice, cmap="gray", vmin=vmin, vmax=vmax)
    axis.imshow(
        label_slice > 0,
        cmap=_GT_OVERLAY_CMAP,
        vmin=0,
        vmax=1,
        alpha=0.35,
    )
    axis.imshow(
        prediction_slice > 0,
        cmap=_PRED_OVERLAY_CMAP,
        vmin=0,
        vmax=1,
        alpha=0.35,
    )
    _add_bbox(axis, final_roi, color="yellow", linestyle="-", linewidth=2.0)
    if raw_prediction_bbox is not None:
        _add_bbox(
            axis,
            raw_prediction_bbox,
            color="orange",
            linestyle="--",
            linewidth=1.5,
        )
    legend_handles = [
        Patch(facecolor="red", edgecolor="none", alpha=0.35, label="Red = GT"),
        Patch(
            facecolor="green",
            edgecolor="none",
            alpha=0.35,
            label="Green = Stage 1 prediction",
        ),
        Line2D(
            [0], [0], color="yellow", linewidth=2.0, label="Yellow box = Dataset504 ROI"
        ),
    ]
    if raw_prediction_bbox is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="orange",
                linewidth=1.5,
                linestyle="--",
                label="Orange dashed = raw prediction bbox",
            )
        )
    axis.legend(handles=legend_handles, loc="upper right", fontsize=6, framealpha=0.8)
    axis.set_title("Dataset501 GT + Stage1 + ROI")
    axis.axis("off")


def _show_roi_info(
    axis: plt.Axes,
    *,
    final_roi: tuple[int, int, int, int],
    raw_prediction_bbox: tuple[int, int, int, int] | None,
    cropped_shape: tuple[int, int, int],
    coverage: float | None,
) -> None:
    coverage_text = "N/A" if coverage is None else f"{coverage:.3f}"
    lines = [
        "Dataset504 ROI information",
        f"final ROI [x0,y0,x1,y1]: {list(final_roi)}",
        f"ROI XY: ({cropped_shape[2]}, {cropped_shape[1]})",
        f"GT coverage: {coverage_text}",
    ]
    if raw_prediction_bbox is not None:
        lines.append(f"raw prediction bbox: {list(raw_prediction_bbox)}")
    axis.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=axis.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "pad": 0.6},
    )
    axis.set_title("Dataset504 ROI information")
    axis.axis("off")


def _render_enhanced_comparison(
    *,
    case_id: str,
    z_index: int,
    original_image: np.ndarray,
    original_label: np.ndarray,
    stage1_prediction: np.ndarray,
    cropped_image: np.ndarray,
    cropped_label: np.ndarray,
    final_roi: tuple[int, int, int, int],
    raw_prediction_bbox: tuple[int, int, int, int] | None,
    output: Path,
) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=200, squeeze=False)
    try:
        original_image_slice = original_image[z_index]
        original_label_slice = original_label[z_index]
        prediction_slice = stage1_prediction[z_index]
        cropped_image_slice = cropped_image[z_index]
        cropped_label_slice = cropped_label[z_index]
        original_gt_voxels = int(np.count_nonzero(original_label))
        cropped_gt_voxels = int(np.count_nonzero(cropped_label))
        coverage = (
            None
            if original_gt_voxels == 0
            else cropped_gt_voxels / original_gt_voxels
        )

        _show_dwi(axes[0, 0], original_image_slice, "Dataset501 DWI")
        _show_gt(axes[0, 1], original_label_slice, "Dataset501 GT")
        _show_prediction(axes[0, 2], prediction_slice, "Stage 1 prediction")
        _show_key_overlay(
            axes[0, 3],
            original_image_slice,
            original_label_slice,
            prediction_slice,
            final_roi,
            raw_prediction_bbox,
        )
        _show_dwi(axes[1, 0], cropped_image_slice, "Dataset504 Cropped DWI")
        _show_gt(axes[1, 1], cropped_label_slice, "Dataset504 Cropped GT")
        _show_overlay(
            axes[1, 2],
            cropped_image_slice,
            cropped_label_slice,
            "Dataset504 Overlay",
        )
        _show_roi_info(
            axes[1, 3],
            final_roi=final_roi,
            raw_prediction_bbox=raw_prediction_bbox,
            cropped_shape=cropped_image.shape,
            coverage=coverage,
        )
        coverage_text = "N/A" if coverage is None else f"{coverage:.3f}"
        figure.suptitle(
            f"{case_id} | z={z_index} | "
            f"Original={original_image.shape[2]}×{original_image.shape[1]} | "
            f"ROI={cropped_image.shape[2]}×{cropped_image.shape[1]} | "
            f"GT coverage={coverage_text}"
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
        figure.savefig(output, dpi=200, bbox_inches="tight")
    finally:
        plt.close(figure)


def _render_comparison(
    *,
    case_id: str,
    z_index: int,
    original_image: np.ndarray,
    original_label: np.ndarray,
    cropped_image: np.ndarray,
    cropped_label: np.ndarray,
    output: Path,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=200, squeeze=False)
    try:
        original_image_slice = original_image[z_index]
        original_label_slice = original_label[z_index]
        cropped_image_slice = cropped_image[z_index]
        cropped_label_slice = cropped_label[z_index]

        _show_dwi(axes[0, 0], original_image_slice, "Dataset501 DWI")
        _show_gt(axes[0, 1], original_label_slice, "Dataset501 GT")
        _show_overlay(
            axes[0, 2],
            original_image_slice,
            original_label_slice,
            "Dataset501 Overlay",
        )
        _show_dwi(axes[1, 0], cropped_image_slice, "Dataset504 Cropped DWI")
        _show_gt(axes[1, 1], cropped_label_slice, "Dataset504 Cropped GT")
        _show_overlay(
            axes[1, 2],
            cropped_image_slice,
            cropped_label_slice,
            "Dataset504 Overlay",
        )
        figure.suptitle(
            f"case ID: {case_id} | z slice: {z_index} | "
            f"Dataset501 XY shape: {original_image.shape[1:]} | "
            f"Dataset504 XY shape: {cropped_image.shape[1:]}"
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
        figure.savefig(output, dpi=200, bbox_inches="tight")
    finally:
        plt.close(figure)


def compare_case(
    *,
    dataset501_root: Path,
    dataset504_root: Path,
    case_id: str,
    output: Path,
    slice_index: int | None = None,
    stage1_prediction_dir: Path | None = None,
    roi_manifest: Path | None = None,
) -> int:
    """Validate and render one matching Dataset501/Dataset504 case pair."""
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("case_id must be a non-empty string")

    dataset501_root = Path(dataset501_root)
    dataset504_root = Path(dataset504_root)
    output = Path(output)
    if (stage1_prediction_dir is None) != (roi_manifest is None):
        raise ValueError(
            "stage1_prediction_dir and roi_manifest must be provided together"
        )
    dataset501_paths, dataset504_paths = _require_case_files(
        dataset501_root, dataset504_root, case_id
    )

    original_image, original_label = _read_and_validate_case(
        *dataset501_paths, "Dataset501"
    )
    cropped_image, cropped_label = _read_and_validate_case(
        *dataset504_paths, "Dataset504"
    )
    original_z = original_image.array.shape[0]
    cropped_z = cropped_image.array.shape[0]
    if original_z != cropped_z:
        raise ValueError(
            "Dataset501 and Dataset504 z counts differ: "
            f"Dataset501={original_z}, Dataset504={cropped_z}"
        )

    z_index = select_slice_index(original_label.array, slice_index=slice_index)
    if not 0 <= z_index < cropped_z:
        raise ValueError(
            f"chosen slice index {z_index} is invalid for Dataset504 z count {cropped_z}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    if stage1_prediction_dir is None and roi_manifest is None:
        _render_comparison(
            case_id=case_id,
            z_index=z_index,
            original_image=original_image.array,
            original_label=original_label.array,
            cropped_image=cropped_image.array,
            cropped_label=cropped_label.array,
            output=output,
        )
    else:
        stage1_prediction_path = Path(stage1_prediction_dir) / f"{case_id}.nii.gz"
        if not stage1_prediction_path.is_file():
            raise FileNotFoundError(
                f"Stage 1 prediction does not exist: {stage1_prediction_path}"
            )
        stage1_prediction = read_nifti(stage1_prediction_path)
        if stage1_prediction.array.shape != original_image.array.shape:
            raise ValueError(
                "Stage 1 prediction shape mismatch: "
                f"prediction={stage1_prediction.array.shape}, "
                f"Dataset501={original_image.array.shape}"
            )
        final_roi, raw_prediction_bbox = _load_manifest_rois(
            Path(roi_manifest),
            case_id=case_id,
            original_shape=original_image.array.shape,
        )
        original_gt_voxels = int(np.count_nonzero(original_label.array))
        cropped_gt_voxels = int(np.count_nonzero(cropped_label.array))
        coverage = (
            None
            if original_gt_voxels == 0
            else cropped_gt_voxels / original_gt_voxels
        )
        _render_enhanced_comparison(
            case_id=case_id,
            z_index=z_index,
            original_image=original_image.array,
            original_label=original_label.array,
            stage1_prediction=stage1_prediction.array,
            cropped_image=cropped_image.array,
            cropped_label=cropped_label.array,
            final_roi=final_roi,
            raw_prediction_bbox=raw_prediction_bbox,
            output=output,
        )
    return z_index


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare one Dataset501 case with its Dataset504 ROI crop."
    )
    parser.add_argument("--dataset501-root", type=Path, required=True)
    parser.add_argument("--dataset504-root", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slice-index", type=int, default=None)
    parser.add_argument("--stage1-prediction-dir", type=Path, default=None)
    parser.add_argument("--roi-manifest", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    z_index = compare_case(
        dataset501_root=arguments.dataset501_root,
        dataset504_root=arguments.dataset504_root,
        case_id=arguments.case_id,
        output=arguments.output,
        slice_index=arguments.slice_index,
        stage1_prediction_dir=arguments.stage1_prediction_dir,
        roi_manifest=arguments.roi_manifest,
    )
    print(
        f"Compared {arguments.case_id}: z={z_index}; "
        f"output={arguments.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
