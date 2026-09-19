#!/usr/bin/env python
"""Compare three binary nnU-Net predictions with read-only error overlays.

This tool is intended for server-side error analysis. It never resamples or
repairs input data, and it requires a new report directory outside every input
directory.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import quote

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import SimpleITK as sitk

from standalone_nnunet2d.metrics.case_metrics import volume_metrics


GEOMETRY_ATOL = 1e-6
MODEL_DIRECTORIES = (
    ("topk10", "TopK10"),
    ("foreground50", "Foreground50"),
    ("grouped_topk10", "GroupedTopK10"),
)
ERROR_COLORS = {
    "TP": np.asarray((0.0, 0.80, 0.0, 0.58), dtype=np.float32),
    "FP": np.asarray((1.0, 0.0, 0.0, 0.58), dtype=np.float32),
    "FN": np.asarray((0.0, 0.25, 1.0, 0.58), dtype=np.float32),
}


def _nifti_case_id(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    if name.lower().endswith(".nii"):
        return name[:-4]
    raise ValueError(f"not a NIfTI filename: {path}")


def _nifti_files(directory: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file()
            and (path.name.lower().endswith(".nii.gz") or path.name.lower().endswith(".nii"))
        ),
        key=lambda path: path.name,
    )


def _require_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {resolved}")
    return resolved


def discover_image_files(directory: Path) -> dict[str, Path]:
    """Discover only nnU-Net channel-0 images named ``<case>_0000.nii[.gz]``."""
    result: dict[str, Path] = {}
    for path in _nifti_files(directory):
        stem = _nifti_case_id(path)
        if not stem.endswith("_0000"):
            continue
        case_id = stem[:-5]
        if not case_id:
            raise ValueError(f"invalid channel-0 DWI filename: {path}")
        if case_id in result:
            raise ValueError(
                f"DWI: duplicate case ID {case_id!r}: {result[case_id]} and {path}"
            )
        result[case_id] = path
    return result


def discover_mask_files(directory: Path, label: str) -> dict[str, Path]:
    """Discover binary masks named ``<case>.nii[.gz]`` with duplicate checks."""
    result: dict[str, Path] = {}
    for path in _nifti_files(directory):
        case_id = _nifti_case_id(path)
        if not case_id:
            raise ValueError(f"{label}: empty case ID in filename {path}")
        if case_id in result:
            raise ValueError(
                f"{label}: duplicate case ID {case_id!r}: {result[case_id]} and {path}"
            )
        result[case_id] = path
    return result


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_output_boundary(output_dir: Path, input_dirs: Iterable[Path]) -> Path:
    """Require a fresh output path with no ancestor/descendant input relationship."""
    output = output_dir.expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError(
            f"output directory already exists; choose a new path to avoid overwriting: {output}"
        )
    for input_dir in input_dirs:
        if _is_relative_to(output, input_dir) or _is_relative_to(input_dir, output):
            raise ValueError(
                "output directory must not equal, contain, or be inside an input directory: "
                f"output={output}, input={input_dir}"
            )
    return output


def _read_scalar_3d(path: Path, label: str) -> sitk.Image:
    image = sitk.ReadImage(str(path))
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError(
            f"{label} must be a scalar 3D NIfTI image, got dimension="
            f"{image.GetDimension()} components={image.GetNumberOfComponentsPerPixel()}"
        )
    return image


def assert_same_geometry(reference: sitk.Image, other: sitk.Image, label: str) -> None:
    """Reject any size or physical-geometry mismatch without resampling."""
    if reference.GetSize() != other.GetSize():
        raise ValueError(
            f"{label}: size mismatch: reference={reference.GetSize()}, other={other.GetSize()}"
        )
    checks = (
        ("spacing", reference.GetSpacing(), other.GetSpacing()),
        ("origin", reference.GetOrigin(), other.GetOrigin()),
        ("direction", reference.GetDirection(), other.GetDirection()),
    )
    for name, expected, actual in checks:
        if not np.allclose(
            np.asarray(expected, dtype=np.float64),
            np.asarray(actual, dtype=np.float64),
            rtol=0.0,
            atol=GEOMETRY_ATOL,
        ):
            raise ValueError(
                f"{label}: {name} mismatch (rtol=0, atol={GEOMETRY_ATOL:g}): "
                f"reference={expected}, other={actual}"
            )


def validate_axis_aligned_direction(direction: Sequence[float], label: str) -> None:
    """Accept only signed axis permutations that can be reoriented without interpolation."""
    matrix = np.asarray(direction, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label}: direction contains non-finite values")
    absolute = np.abs(matrix)
    dominant_rows = np.argmax(absolute, axis=0)
    signed_permutation = np.zeros((3, 3), dtype=np.float64)
    for column, row in enumerate(dominant_rows):
        signed_permutation[row, column] = 1.0 if matrix[row, column] >= 0 else -1.0
    if len(set(int(row) for row in dominant_rows)) != 3 or not np.allclose(
        matrix, signed_permutation, rtol=0.0, atol=GEOMETRY_ATOL
    ):
        raise ValueError(
            f"{label}: direction is not axis-aligned within rtol=0, "
            f"atol={GEOMETRY_ATOL:g}; oblique data cannot be displayed reliably "
            "without resampling, which this tool forbids"
        )


def _binary_array(image: sitk.Image, label: str) -> np.ndarray:
    array = sitk.GetArrayFromImage(image)
    unique = np.unique(array)
    if not np.isin(unique, (0, 1)).all():
        preview = unique[:20].tolist()
        raise ValueError(f"{label} must contain only binary labels 0 and 1, got {preview}")
    return array.astype(np.uint8, copy=False)


def _orient_to_lps(image: sitk.Image, label: str) -> sitk.Image:
    validate_axis_aligned_direction(image.GetDirection(), label)
    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation("LPS")
    oriented = orienter.Execute(image)
    if not np.allclose(
        np.asarray(oriented.GetDirection()), np.eye(3).reshape(-1), rtol=0.0, atol=GEOMETRY_ATOL
    ):
        raise ValueError(f"{label}: failed to obtain an axis-aligned LPS view")
    return oriented


def _parse_case_ids(values: Sequence[str] | None) -> list[str] | None:
    if values is None:
        return None
    parsed: list[str] = []
    for value in values:
        parsed.extend(part.strip() for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError("--case-ids did not contain any case IDs")
    duplicates = sorted({case_id for case_id in parsed if parsed.count(case_id) > 1})
    if duplicates:
        raise ValueError(f"--case-ids contains duplicates: {duplicates}")
    return parsed


def _format_set_difference(reference: set[str], other: set[str], label: str) -> str:
    missing = sorted(reference - other)
    extra = sorted(other - reference)
    return f"{label}: missing={missing[:20]}, extra={extra[:20]}"


def _resolve_candidate_ids(
    explicit_ids: list[str] | None,
    images: dict[str, Path],
    labels: dict[str, Path],
    predictions: dict[str, dict[str, Path]],
) -> list[str]:
    if explicit_ids is None:
        reference_ids = set(predictions["topk10"])
        if not reference_ids:
            raise ValueError("TopK10 directory contains no prediction NIfTI files")
        mismatches = []
        for key in ("foreground50", "grouped_topk10"):
            current = set(predictions[key])
            if current != reference_ids:
                mismatches.append(_format_set_difference(reference_ids, current, key))
        if mismatches:
            raise ValueError("prediction case-ID sets differ:\n" + "\n".join(mismatches))
        candidate_ids = sorted(reference_ids)
    else:
        candidate_ids = explicit_ids

    sources = {
        "DWI": images,
        "GT": labels,
        "TopK10": predictions["topk10"],
        "Foreground50": predictions["foreground50"],
        "GroupedTopK10": predictions["grouped_topk10"],
    }
    missing_messages = []
    requested = set(candidate_ids)
    for label, mapping in sources.items():
        missing = sorted(requested - set(mapping))
        if missing:
            missing_messages.append(f"{label} missing {missing[:20]}")
    if missing_messages:
        raise ValueError("selected cases are incomplete: " + "; ".join(missing_messages))
    return candidate_ids


def _evaluate_cases(
    case_ids: Sequence[str],
    images: dict[str, Path],
    labels: dict[str, Path],
    predictions: dict[str, dict[str, Path]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case_id in case_ids:
        dwi_image = _read_scalar_3d(images[case_id], f"{case_id} DWI")
        validate_axis_aligned_direction(dwi_image.GetDirection(), f"{case_id} DWI")
        dwi_array = sitk.GetArrayViewFromImage(dwi_image)
        if not np.isfinite(dwi_array).any():
            raise ValueError(f"{case_id} DWI contains no finite voxel values")

        gt_image = _read_scalar_3d(labels[case_id], f"{case_id} GT")
        assert_same_geometry(dwi_image, gt_image, f"{case_id} GT vs DWI")
        gt = _binary_array(gt_image, f"{case_id} GT")
        row: dict[str, object] = {"case_id": case_id}
        for key, display_name in MODEL_DIRECTORIES:
            prediction_image = _read_scalar_3d(
                predictions[key][case_id], f"{case_id} {display_name}"
            )
            assert_same_geometry(
                dwi_image, prediction_image, f"{case_id} {display_name} vs DWI"
            )
            prediction = _binary_array(prediction_image, f"{case_id} {display_name}")
            metrics = volume_metrics(prediction, gt)
            for metric_name in (
                "dice",
                "precision",
                "recall",
                "tp",
                "fp",
                "fn",
                "pred_voxels",
                "gt_voxels",
            ):
                row[f"{key}_{metric_name}"] = metrics[metric_name]
        row["grouped_topk10_dice_minus_topk10_dice"] = (
            float(row["grouped_topk10_dice"]) - float(row["topk10_dice"])
        )
        rows.append(row)
    return rows


def select_cases(
    rows: Sequence[dict[str, object]], max_cases: int
) -> tuple[list[str], dict[str, list[str]]]:
    """Apply deterministic automatic case-selection rules."""
    if max_cases < 1:
        raise ValueError("--max-cases must be at least 1")
    reasons: dict[str, list[str]] = defaultdict(list)
    first_seen: list[str] = []

    def add(group: Sequence[dict[str, object]], reason: str) -> None:
        for row in group:
            case_id = str(row["case_id"])
            if case_id not in reasons:
                first_seen.append(case_id)
            reasons[case_id].append(reason)

    negative = [
        row for row in rows if float(row["grouped_topk10_dice_minus_topk10_dice"]) < 0
    ]
    negative.sort(
        key=lambda row: (
            float(row["grouped_topk10_dice_minus_topk10_dice"]),
            str(row["case_id"]),
        )
    )
    add(negative[:3], "grouped_minus_topk10_most_negative")

    lowest_mean = sorted(
        rows,
        key=lambda row: (
            (float(row["topk10_dice"]) + float(row["grouped_topk10_dice"])) / 2.0,
            str(row["case_id"]),
        ),
    )
    add(lowest_mean[:3], "topk10_grouped_mean_dice_lowest")

    improved = [
        row for row in rows if float(row["grouped_topk10_dice_minus_topk10_dice"]) > 0
    ]
    improved.sort(
        key=lambda row: (
            -float(row["grouped_topk10_dice_minus_topk10_dice"]),
            str(row["case_id"]),
        )
    )
    add(improved[:2], "grouped_minus_topk10_largest_positive")
    return first_seen[:max_cases], reasons


def _slice_counts(prediction: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axes = (1, 2)
    tp = ((prediction == 1) & (gt == 1)).sum(axis=axes)
    fp = ((prediction == 1) & (gt == 0)).sum(axis=axes)
    fn = ((prediction == 0) & (gt == 1)).sum(axis=axes)
    return tp.astype(np.int64), fp.astype(np.int64), fn.astype(np.int64)


def select_slices(
    gt: np.ndarray,
    topk10: np.ndarray,
    foreground50: np.ndarray,
    grouped_topk10: np.ndarray,
    limit: int,
) -> tuple[list[int], dict[int, list[str]]]:
    """Select informative LPS axial slices without padding with all-empty slices."""
    if limit < 1:
        raise ValueError("--slices-per-case must be at least 1")
    gt_area = (gt == 1).sum(axis=(1, 2)).astype(np.int64)
    top_counts = _slice_counts(topk10, gt)
    foreground_counts = _slice_counts(foreground50, gt)
    grouped_counts = _slice_counts(grouped_topk10, gt)
    grouped_fp = grouped_counts[1]
    grouped_fn = grouped_counts[2]
    relative_error_increase = (grouped_fp + grouped_fn) - (top_counts[1] + top_counts[2])

    reasons: dict[int, list[str]] = defaultdict(list)
    order: list[int] = []

    def add_best(values: np.ndarray, reason: str, *, require_positive: bool = True) -> None:
        maximum = int(values.max()) if values.size else 0
        if require_positive and maximum <= 0:
            return
        index = int(np.flatnonzero(values == maximum)[0])
        if index not in reasons:
            order.append(index)
        reasons[index].append(reason)

    add_best(gt_area, "gt_area_max")
    add_best(grouped_fn, "grouped_topk10_fn_max")
    add_best(grouped_fp, "grouped_topk10_fp_max")
    add_best(relative_error_increase, "grouped_vs_topk10_fp_fn_increase_max")

    all_error = (
        top_counts[1]
        + top_counts[2]
        + foreground_counts[1]
        + foreground_counts[2]
        + grouped_fp
        + grouped_fn
    )
    information_score = all_error + gt_area
    supplementation = sorted(
        (index for index in range(gt.shape[0]) if information_score[index] > 0),
        key=lambda index: (-int(information_score[index]), index),
    )
    for index in supplementation:
        if len(order) >= limit:
            break
        if index not in reasons:
            order.append(index)
            reasons[index].append("informative_error_or_gt_area_fill")
    return order[:limit], reasons


def _display_window(image: np.ndarray) -> tuple[float, float]:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise ValueError("DWI contains no finite values")
    nonzero = finite[finite != 0]
    values = nonzero if nonzero.size else finite
    low, high = np.percentile(values.astype(np.float64), (1.0, 99.0))
    if high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        low -= 0.5
        high += 0.5
    return float(low), float(high)


def _error_overlay(prediction: np.ndarray, gt: np.ndarray) -> np.ndarray:
    rgba = np.zeros((*gt.shape, 4), dtype=np.float32)
    rgba[(prediction == 1) & (gt == 1)] = ERROR_COLORS["TP"]
    rgba[(prediction == 1) & (gt == 0)] = ERROR_COLORS["FP"]
    rgba[(prediction == 0) & (gt == 1)] = ERROR_COLORS["FN"]
    return rgba


def _draw_orientation_labels(axis: plt.Axes) -> None:
    labels = (
        (0.01, 0.50, "R", "left", "center"),
        (0.99, 0.50, "L", "right", "center"),
        (0.50, 0.99, "A", "center", "top"),
        (0.50, 0.01, "P", "center", "bottom"),
    )
    for x, y, text, horizontal, vertical in labels:
        axis.text(
            x,
            y,
            text,
            transform=axis.transAxes,
            color="white",
            fontsize=8,
            fontweight="bold",
            ha=horizontal,
            va=vertical,
            bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 1},
        )


def _safe_case_filename(case_id: str) -> str:
    return quote(case_id, safe="._-") or "case"


def render_slice(
    output_path: Path,
    case_id: str,
    slice_index: int,
    physical_z_mm: float,
    image: np.ndarray,
    gt: np.ndarray,
    predictions: dict[str, np.ndarray],
    spacing_xy: tuple[float, float],
    window: tuple[float, float],
) -> None:
    """Render one full-field, four-column LPS axial comparison PNG."""
    figure, axes = plt.subplots(1, 4, figsize=(16, 4.6), sharex=True, sharey=True)
    aspect = spacing_xy[1] / spacing_xy[0]
    for axis in axes:
        axis.imshow(
            image[slice_index],
            cmap="gray",
            vmin=window[0],
            vmax=window[1],
            origin="upper",
            interpolation="nearest",
            aspect=aspect,
        )
        axis.set_xlim(-0.5, image.shape[2] - 0.5)
        axis.set_ylim(image.shape[1] - 0.5, -0.5)
        axis.axis("off")
        _draw_orientation_labels(axis)

    if gt[slice_index].any():
        axes[0].contour(
            gt[slice_index],
            levels=(0.5,),
            colors=("#FFD700",),
            linewidths=1.2,
            origin="upper",
        )
    axes[0].set_title(f"DWI + GT contour\n{case_id} | LPS axial z={slice_index}", fontsize=10)

    for axis, (key, display_name) in zip(axes[1:], MODEL_DIRECTORIES):
        prediction = predictions[key]
        axis.imshow(
            _error_overlay(prediction[slice_index], gt[slice_index]),
            origin="upper",
            interpolation="nearest",
            aspect=aspect,
        )
        tp, fp, fn = (int(values[slice_index]) for values in _slice_counts(prediction, gt))
        axis.set_title(f"{display_name}\nTP={tp}  FP={fp}  FN={fn}", fontsize=10)

    legend = [
        Patch(facecolor=ERROR_COLORS["TP"], label="TP (green)"),
        Patch(facecolor=ERROR_COLORS["FP"], label="FP (red)"),
        Patch(facecolor=ERROR_COLORS["FN"], label="FN (blue)"),
        Line2D((0,), (0,), color="#FFD700", linewidth=1.5, label="GT contour (yellow)"),
    ]
    figure.legend(handles=legend, loc="lower center", ncol=4, frameon=False)
    figure.suptitle(
        f"LPS axial plane | physical z={physical_z_mm:.3f} mm | "
        f"case-wide DWI window [{window[0]:.3g}, {window[1]:.3g}]",
        fontsize=11,
    )
    figure.tight_layout(rect=(0.0, 0.10, 1.0, 0.91))
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _load_oriented_case(
    case_id: str,
    images: dict[str, Path],
    labels: dict[str, Path],
    prediction_paths: dict[str, dict[str, Path]],
) -> tuple[sitk.Image, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    dwi = _orient_to_lps(_read_scalar_3d(images[case_id], f"{case_id} DWI"), f"{case_id} DWI")
    gt = _orient_to_lps(_read_scalar_3d(labels[case_id], f"{case_id} GT"), f"{case_id} GT")
    assert_same_geometry(dwi, gt, f"{case_id} oriented GT vs DWI")
    predictions: dict[str, np.ndarray] = {}
    for key, display_name in MODEL_DIRECTORIES:
        prediction = _orient_to_lps(
            _read_scalar_3d(prediction_paths[key][case_id], f"{case_id} {display_name}"),
            f"{case_id} {display_name}",
        )
        assert_same_geometry(dwi, prediction, f"{case_id} oriented {display_name} vs DWI")
        predictions[key] = _binary_array(prediction, f"{case_id} {display_name}")
    return dwi, sitk.GetArrayFromImage(dwi), _binary_array(gt, f"{case_id} GT"), predictions


def _write_case_csv(
    path: Path,
    rows: Sequence[dict[str, object]],
    selected_ids: Sequence[str],
    reasons: dict[str, list[str]],
) -> None:
    selected = set(selected_ids)
    metric_names = ("dice", "precision", "recall", "tp", "fp", "fn", "pred_voxels", "gt_voxels")
    fieldnames = ["case_id", "selected_for_visualization", "selection_reasons"]
    for key, _ in MODEL_DIRECTORIES:
        fieldnames.extend(f"{key}_{metric}" for metric in metric_names)
    fieldnames.append("grouped_topk10_dice_minus_topk10_dice")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for original in sorted(rows, key=lambda row: str(row["case_id"])):
            row = dict(original)
            case_id = str(row["case_id"])
            row["selected_for_visualization"] = "yes" if case_id in selected else "no"
            row["selection_reasons"] = ";".join(reasons.get(case_id, ()))
            writer.writerow(row)


def _write_slice_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames = (
        "case_id",
        "slice_index_lps_z",
        "physical_z_mm",
        "selection_reasons",
        "gt_voxels",
        "topk10_tp",
        "topk10_fp",
        "topk10_fn",
        "foreground50_tp",
        "foreground50_fp",
        "foreground50_fn",
        "grouped_topk10_tp",
        "grouped_topk10_fp",
        "grouped_topk10_fn",
        "png_file",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(
    path: Path,
    input_dirs: dict[str, Path],
    case_count: int,
    selected_ids: Sequence[str],
    slices_per_case: int,
) -> None:
    lines = [
        "SEGMENTATION ERROR COMPARISON REPORT",
        "",
        "Purpose",
        "- Read-only error analysis for TopK10, Foreground50, and GroupedTopK10 predictions.",
        "- PNGs provide visual evidence; they do not diagnose the cause or automatically classify error types.",
        "- This report is not a new formal experiment and does not change predictions, labels, or images.",
        "",
        "Inputs",
    ]
    lines.extend(f"- {name}: {directory}" for name, directory in input_dirs.items())
    lines.extend(
        [
            "",
            "Geometry and orientation",
            f"- Size must match exactly. Spacing/origin/direction use rtol=0 and atol={GEOMETRY_ATOL:g}.",
            "- No resampling, interpolation, flipping repair, or geometry repair is performed.",
            "- Only axis-aligned directions that can be converted to LPS by permutation/flips are accepted.",
            "- PNGs show full-field LPS axial source planes: R at left, L at right, A at top, P at bottom.",
            "",
            "Display",
            "- DWI window is the case-wide 1st/99th percentile of finite nonzero voxels (fallback: all finite voxels).",
            "- Every panel and selected slice for a case uses that same window, field of view, and physical aspect.",
            "- Yellow line: GT contour. Green: TP. Red: FP. Blue: FN.",
            "- No GT-based cropping is used, so distant false positives remain visible.",
            "",
            "Metrics",
            "- Binary class convention: background=0, foreground=1.",
            "- Dice=1 when prediction and GT are both empty; Dice=0 when exactly one is empty.",
            "- Precision=0 when TP+FP=0; recall=0 when TP+FN=0.",
            "- Counts and metrics are computed over each complete 3D volume.",
            "",
            "Selection and outputs",
            f"- Evaluated cases: {case_count}.",
            f"- Visualized cases ({len(selected_ids)}): {', '.join(selected_ids) if selected_ids else '(none)'}.",
            f"- Requested maximum slices per case: {slices_per_case}; all-empty slices are never used as filler.",
            "- case_comparison.csv: all evaluated cases, metrics, Dice delta, and case-selection reasons.",
            "- slice_selection.csv: displayed LPS-z indices, physical z, reasons, slice counts, and PNG names.",
            "- <case>_slice_<index>.png: one full-field four-column comparison.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate strict, read-only three-model segmentation error comparison PNGs and CSVs.",
        epilog=(
            "Use explicit server paths. Example placeholders: --images-dir <IMAGES_TR_DIR> "
            "--labels-dir <LABELS_TR_DIR> --topk10-dir <TOPK10_PRED_DIR> "
            "--foreground50-dir <FOREGROUND50_PRED_DIR> "
            "--grouped-topk-dir <GROUPED_TOPK10_PRED_DIR> --output-dir <NEW_REPORT_DIR>"
        ),
    )
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--topk10-dir", type=Path, required=True)
    parser.add_argument("--foreground50-dir", type=Path, required=True)
    parser.add_argument("--grouped-topk-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--case-ids",
        nargs="+",
        default=None,
        help="Explicit case IDs (space- or comma-separated); bypass automatic case selection.",
    )
    parser.add_argument("--slices-per-case", type=int, default=4)
    parser.add_argument(
        "--max-cases",
        type=int,
        default=8,
        help="Maximum automatically selected cases; ignored when --case-ids is provided.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.slices_per_case < 1:
        raise ValueError("--slices-per-case must be at least 1")
    if args.max_cases < 1:
        raise ValueError("--max-cases must be at least 1")

    input_dirs = {
        "images_dir": _require_directory(args.images_dir, "--images-dir"),
        "labels_dir": _require_directory(args.labels_dir, "--labels-dir"),
        "topk10_dir": _require_directory(args.topk10_dir, "--topk10-dir"),
        "foreground50_dir": _require_directory(args.foreground50_dir, "--foreground50-dir"),
        "grouped_topk_dir": _require_directory(args.grouped_topk_dir, "--grouped-topk-dir"),
    }
    normalized_inputs = list(input_dirs.values())
    if len(set(normalized_inputs)) != len(normalized_inputs):
        raise ValueError("all five input directories must resolve to distinct paths")
    output_dir = validate_output_boundary(args.output_dir, normalized_inputs)

    images = discover_image_files(input_dirs["images_dir"])
    labels = discover_mask_files(input_dirs["labels_dir"], "GT")
    predictions = {
        "topk10": discover_mask_files(input_dirs["topk10_dir"], "TopK10"),
        "foreground50": discover_mask_files(input_dirs["foreground50_dir"], "Foreground50"),
        "grouped_topk10": discover_mask_files(input_dirs["grouped_topk_dir"], "GroupedTopK10"),
    }
    explicit_ids = _parse_case_ids(args.case_ids)
    case_ids = _resolve_candidate_ids(explicit_ids, images, labels, predictions)
    metric_rows = _evaluate_cases(case_ids, images, labels, predictions)

    if explicit_ids is None:
        selected_ids, selection_reasons = select_cases(metric_rows, args.max_cases)
    else:
        selected_ids = list(explicit_ids)
        selection_reasons = {case_id: ["explicit_case_id"] for case_id in selected_ids}

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_case_csv(
        output_dir / "case_comparison.csv",
        metric_rows,
        selected_ids,
        selection_reasons,
    )

    slice_rows: list[dict[str, object]] = []
    for case_id in selected_ids:
        dwi_image, dwi, gt, oriented_predictions = _load_oriented_case(
            case_id, images, labels, predictions
        )
        selected_slices, slice_reasons = select_slices(
            gt,
            oriented_predictions["topk10"],
            oriented_predictions["foreground50"],
            oriented_predictions["grouped_topk10"],
            args.slices_per_case,
        )
        window = _display_window(dwi)
        safe_case_id = _safe_case_filename(case_id)
        per_model_counts = {
            key: _slice_counts(prediction, gt)
            for key, prediction in oriented_predictions.items()
        }
        for slice_index in selected_slices:
            physical_z_mm = float(
                dwi_image.TransformIndexToPhysicalPoint((0, 0, int(slice_index)))[2]
            )
            png_name = f"{safe_case_id}_slice_{slice_index:04d}.png"
            render_slice(
                output_dir / png_name,
                case_id,
                slice_index,
                physical_z_mm,
                dwi,
                gt,
                oriented_predictions,
                (float(dwi_image.GetSpacing()[0]), float(dwi_image.GetSpacing()[1])),
                window,
            )
            slice_row: dict[str, object] = {
                "case_id": case_id,
                "slice_index_lps_z": slice_index,
                "physical_z_mm": f"{physical_z_mm:.6f}",
                "selection_reasons": ";".join(slice_reasons[slice_index]),
                "gt_voxels": int((gt[slice_index] == 1).sum()),
                "png_file": png_name,
            }
            for key, _ in MODEL_DIRECTORIES:
                tp, fp, fn = per_model_counts[key]
                slice_row[f"{key}_tp"] = int(tp[slice_index])
                slice_row[f"{key}_fp"] = int(fp[slice_index])
                slice_row[f"{key}_fn"] = int(fn[slice_index])
            slice_rows.append(slice_row)

    _write_slice_csv(output_dir / "slice_selection.csv", slice_rows)
    _write_readme(
        output_dir / "README.txt",
        input_dirs,
        len(metric_rows),
        selected_ids,
        args.slices_per_case,
    )
    print(f"Evaluated {len(metric_rows)} cases and visualized {len(selected_ids)} cases.")
    print(f"Wrote report: {output_dir}")
    if any(case_id for case_id in selected_ids if not any(row["case_id"] == case_id for row in slice_rows)):
        print("Warning: at least one selected case had no informative non-empty slice to render.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
