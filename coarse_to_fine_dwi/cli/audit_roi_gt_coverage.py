"""Audit final prediction-guided ROIs against complete full-volume GT labels."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean, median
from typing import Any

import numpy as np

from ..dataset import _validate_fixed_splits
from ..nifti import NiftiVolume
from ..roi import compute_prediction_roi, validate_binary_prediction
from .evaluate_fold_preflight import _manifest_rows


EXPECTED_FOLD_COUNT = 5
CSV_FIELDS = (
    "case_id",
    "gt_total_foreground_voxels",
    "gt_roi_foreground_voxels",
    "gt_outside_roi_foreground_voxels",
    "coverage",
    "gt_lesion_components",
    "has_complete_missed_lesion",
    "complete_missed_lesion_count",
    "complete_missed_lesion_voxels",
)


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _validate_output_dir(
    output_dir: Path,
    *,
    labels_dir: Path,
    manifest: Path,
    splits: Path,
) -> None:
    if _paths_overlap(output_dir, labels_dir):
        raise ValueError("output directory overlaps labels directory")
    for source_path, source_name in ((manifest, "manifest"), (splits, "splits")):
        source_parent = source_path.parent
        if output_dir == source_parent or source_parent in output_dir.parents:
            raise ValueError(f"output directory is inside {source_name} parent directory")
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"output directory is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise FileExistsError(f"output directory must be empty: {output_dir}")


def _nii_case_id(path: Path) -> str | None:
    if path.name.endswith(".nii.gz"):
        return path.name[:-7]
    if path.name.endswith(".nii"):
        return path.name[:-4]
    return None


def _selected_label_paths(labels_dir: Path, selected_ids: Sequence[str]) -> dict[str, Path]:
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"labels directory does not exist: {labels_dir}")
    expected = set(selected_ids)
    found: dict[str, Path] = {}
    duplicates: set[str] = set()
    for path in labels_dir.iterdir():
        if not path.is_file():
            continue
        case_id = _nii_case_id(path)
        if case_id is None or case_id not in expected:
            continue
        if case_id in found:
            duplicates.add(case_id)
        else:
            found[case_id] = path
    missing = sorted(expected - set(found))
    if duplicates or missing:
        raise ValueError(
            f"labels directory must contain one NIfTI for each selected case; "
            f"missing={missing}, duplicates={sorted(duplicates)}"
        )
    return found


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must contain integer coordinates")
    return int(value)


def _bbox(value: object, name: str) -> tuple[int, int, int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError(f"{name} must contain four integer coordinates")
    return tuple(_integer(item, name) for item in value)  # type: ignore[return-value]


def _in_bounds_bbox(bbox: tuple[int, int, int, int], shape_zyx: tuple[int, int, int], name: str) -> None:
    _, height, width = shape_zyx
    x0, y0, x1, y1 = bbox
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(
            f"{name} must be a non-empty in-bounds half-open XY bbox for shape {shape_zyx}"
        )


def _positive_integer(value: object, name: str) -> int:
    result = _integer(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _final_bbox(row: Mapping[str, Any]) -> tuple[int, int, int, int]:
    values: list[tuple[str, tuple[int, int, int, int]]] = []
    for field in ("roi", "bbox_xy"):
        if field in row:
            if row[field] is None:
                raise ValueError(f"manifest field {field} cannot be null")
            values.append((field, _bbox(row[field], field)))
    if not values:
        raise ValueError("manifest case has no final roi or bbox_xy")
    if len(values) == 2 and values[0][1] != values[1][1]:
        raise ValueError("manifest roi and bbox_xy do not match")
    return values[0][1]


def _validate_roi_provenance(
    row: Mapping[str, Any],
    final_bbox: tuple[int, int, int, int],
    shape_zyx: tuple[int, int, int],
) -> None:
    _, height, width = shape_zyx
    _in_bounds_bbox(final_bbox, shape_zyx, "final ROI")

    if "roi_width" in row and _positive_integer(row["roi_width"], "roi_width") != final_bbox[2] - final_bbox[0]:
        raise ValueError("manifest roi_width does not match final ROI")
    if "roi_height" in row and _positive_integer(row["roi_height"], "roi_height") != final_bbox[3] - final_bbox[1]:
        raise ValueError("manifest roi_height does not match final ROI")

    raw_value = row.get("raw_prediction_bbox")
    if raw_value is None:
        if row.get("fallback") is not True:
            raise ValueError("ROI without raw_prediction_bbox must declare fallback=true")
        if final_bbox != (0, 0, width, height):
            raise ValueError("fallback ROI must cover the full XY image")
        return

    raw_bbox = _bbox(raw_value, "raw_prediction_bbox")
    _in_bounds_bbox(raw_bbox, shape_zyx, "raw_prediction_bbox")
    if "roi_margin" not in row or "min_roi_width" not in row or "min_roi_height" not in row:
        raise ValueError(
            "raw_prediction_bbox requires serialized roi_margin, min_roi_width, and min_roi_height"
        )
    min_width = _positive_integer(row["min_roi_width"], "min_roi_width")
    min_height = _positive_integer(row["min_roi_height"], "min_roi_height")
    margin = row["roi_margin"]

    # Construct a synthetic prediction whose foreground union is exactly the
    # serialized raw bbox, then use the existing ROI implementation. This
    # checks the final margin/minimum-ROI policy rather than trusting final_bbox.
    prediction = np.zeros((1, height, width), dtype=np.uint8)
    x0, y0, x1, y1 = raw_bbox
    prediction[:, y0:y1, x0:x1] = 1
    recomputed = compute_prediction_roi(
        prediction,
        margin=margin,
        min_width=min_width,
        min_height=min_height,
    )
    if recomputed.bbox != final_bbox:
        raise ValueError(
            f"final ROI does not match recomputed margin/minimum ROI: "
            f"expected {recomputed.bbox}, got {final_bbox}"
        )
    if "fallback" in row and row["fallback"] is not False:
        raise ValueError("ROI with raw_prediction_bbox must declare fallback=false")


def _component_metrics(
    foreground: np.ndarray,
    final_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int]:
    import SimpleITK as sitk

    image = sitk.GetImageFromArray(foreground.astype(np.uint8, copy=False))
    labels_image = sitk.ConnectedComponent(image, True)
    labels = np.asarray(sitk.GetArrayFromImage(labels_image), dtype=np.int64)
    label_count = int(labels.max(initial=0))
    if label_count == 0:
        return 0, 0, 0
    counts = np.bincount(labels.reshape(-1), minlength=label_count + 1)
    x0, y0, x1, y1 = final_bbox
    in_roi_labels = labels[:, y0:y1, x0:x1]
    in_roi_counts = np.bincount(in_roi_labels.reshape(-1), minlength=label_count + 1)
    missed = [label for label in range(1, label_count + 1) if in_roi_counts[label] == 0]
    return label_count, len(missed), int(counts[missed].sum()) if missed else 0


def _read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"manifest does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"manifest is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("manifest must be a JSON object")
    return _manifest_rows(payload)


def _select_fold_rows(manifest: Path, splits: Path, fold: int) -> tuple[list[str], dict[str, dict[str, Any]]]:
    if fold not in range(EXPECTED_FOLD_COUNT):
        raise ValueError("fold must be between 0 and 4")
    fixed_splits, _ = _validate_fixed_splits(splits.resolve())
    try:
        selected_ids = list(fixed_splits[fold]["val"])
    except (IndexError, KeyError, TypeError) as error:
        raise ValueError(f"fixed splits do not contain fold {fold} validation IDs") from error
    if not selected_ids or not all(isinstance(case_id, str) for case_id in selected_ids):
        raise ValueError(f"fold {fold} validation IDs must be a non-empty list of strings")
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError(f"fold {fold} validation IDs contain duplicates")

    rows = _read_manifest(manifest)
    missing = sorted(set(selected_ids) - set(rows))
    if missing:
        raise ValueError(f"manifest is missing selected case IDs: {missing}")
    for case_id in selected_ids:
        if "fold" in rows[case_id] and rows[case_id]["fold"] != fold:
            raise ValueError(f"manifest fold metadata mismatch for {case_id}: expected {fold}")
    return sorted(selected_ids), rows


def _audit_rows(
    selected_ids: Sequence[str],
    rows: Mapping[str, Mapping[str, Any]],
    label_paths: Mapping[str, Path],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    output_rows: list[dict[str, Any]] = []
    sample_info: dict[str, dict[str, Any]] = {}
    for case_id in selected_ids:
        volume = NiftiVolume.read(label_paths[case_id])
        shape = volume.shape_zyx
        final_bbox = _final_bbox(rows[case_id])
        _validate_roi_provenance(rows[case_id], final_bbox, shape)
        foreground = validate_binary_prediction(volume.array).astype(bool, copy=False)
        total = int(foreground.sum())
        x0, y0, x1, y1 = final_bbox
        in_roi = int(foreground[:, y0:y1, x0:x1].sum())
        outside = total - in_roi
        coverage = 1.0 if total == 0 else in_roi / total
        components, missed_count, missed_voxels = _component_metrics(foreground, final_bbox)
        output_rows.append(
            {
                "case_id": case_id,
                "gt_total_foreground_voxels": total,
                "gt_roi_foreground_voxels": in_roi,
                "gt_outside_roi_foreground_voxels": outside,
                "coverage": float(coverage),
                "gt_lesion_components": components,
                "has_complete_missed_lesion": missed_count > 0,
                "complete_missed_lesion_count": missed_count,
                "complete_missed_lesion_voxels": missed_voxels,
            }
        )
        sample_info[case_id] = {"shape_zyx": shape, "bbox_xy": final_bbox}
    return output_rows, sample_info


def _summary(fold: int, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    coverages = [float(row["coverage"]) for row in rows]
    worst = sorted(rows, key=lambda row: (float(row["coverage"]), str(row["case_id"])))[:5]
    return {
        "fold": fold,
        "case_ids": [str(row["case_id"]) for row in rows],
        "case_count": len(rows),
        "protocol": {
            "space": "original full GT space",
            "array_order": "zyx",
            "bbox_order": "xy",
            "bbox_upper_bound": "half-open",
            "connectivity": 26,
            "final_roi_provenance_check": True,
            "empty_gt_coverage": 1.0,
        },
        "mean_coverage": float(fmean(coverages)),
        "median_coverage": float(median(coverages)),
        "min_coverage": float(min(coverages)),
        "coverage_100_percent_case_count": sum(value == 1.0 for value in coverages),
        "coverage_below_99_percent_case_count": sum(value < 0.99 for value in coverages),
        "coverage_below_95_percent_case_count": sum(value < 0.95 for value in coverages),
        "empty_gt_case_count": sum(int(row["gt_total_foreground_voxels"]) == 0 for row in rows),
        "complete_miss_case_count": sum(bool(row["has_complete_missed_lesion"]) for row in rows),
        "total_missed_lesions": sum(int(row["complete_missed_lesion_count"]) for row in rows),
        "total_missed_lesion_voxels": sum(int(row["complete_missed_lesion_voxels"]) for row in rows),
        "worst_cases": [
            {"case_id": str(row["case_id"]), "coverage": float(row["coverage"])} for row in worst
        ],
    }


def audit_roi_gt_coverage(
    *,
    manifest: str | Path,
    labels_dir: str | Path,
    splits: str | Path,
    fold: int = 0,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Audit one fixed validation fold without modifying any input data."""
    manifest_path = Path(manifest).resolve()
    labels_root = Path(labels_dir).resolve()
    splits_path = Path(splits).resolve()
    output_root = Path(output_dir).resolve()
    _validate_output_dir(
        output_root,
        labels_dir=labels_root,
        manifest=manifest_path,
        splits=splits_path,
    )
    selected_ids, rows = _select_fold_rows(manifest_path, splits_path, fold)
    label_paths = _selected_label_paths(labels_root, selected_ids)
    output_rows, sample_info = _audit_rows(selected_ids, rows, label_paths)
    summary = _summary(fold, output_rows)

    for case_id in random.Random(0).sample(selected_ids, min(3, len(selected_ids))):
        info = sample_info[case_id]
        print(f"sanity case_id={case_id} shape_zyx={info['shape_zyx']} bbox_xy={info['bbox_xy']}")

    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / f"roi_gt_coverage_fold{fold}.csv"
    summary_path = output_root / f"roi_gt_coverage_fold{fold}_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"csv={csv_path}")
    print(f"summary={summary_path}")
    return csv_path, summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(EXPECTED_FOLD_COUNT), default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        audit_roi_gt_coverage(
            manifest=args.manifest,
            labels_dir=args.labels_dir,
            splits=args.splits,
            fold=args.fold,
            output_dir=args.output_dir,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ROI GT coverage audit failed: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
