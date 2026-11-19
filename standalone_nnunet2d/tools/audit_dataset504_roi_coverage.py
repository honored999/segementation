"""Audit Dataset504 ROI sizes and ground-truth coverage against Dataset501."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from standalone_nnunet2d.data.nifti_io import read_nifti


_IMAGE_SUFFIX = "_0000.nii.gz"
_NIFTI_SUFFIX = ".nii.gz"
_BASIC_FIELDS = (
    "case_id",
    "original_shape_z",
    "original_shape_y",
    "original_shape_x",
    "cropped_shape_z",
    "cropped_shape_y",
    "cropped_shape_x",
    "roi_width",
    "roi_height",
    "roi_area",
    "original_gt_voxels",
    "cropped_gt_voxels",
    "gt_coverage",
)


def _discover_images(root: Path) -> dict[str, Path]:
    image_dir = root / "imagesTr"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"missing imagesTr directory: {image_dir}")

    discovered: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in sorted(image_dir.glob(f"*{_IMAGE_SUFFIX}")):
        if not path.is_file():
            continue
        case_id = path.name[: -len(_IMAGE_SUFFIX)]
        if not case_id:
            raise ValueError(f"invalid image filename with empty case ID: {path}")
        if case_id in discovered:
            duplicates.setdefault(case_id, [discovered[case_id]]).append(path)
        else:
            discovered[case_id] = path
    if duplicates:
        details = "; ".join(
            f"{case_id}: {[str(path) for path in paths]}"
            for case_id, paths in sorted(duplicates.items())
        )
        raise ValueError(f"duplicate image case IDs in {image_dir}: {details}")
    return discovered


def _discover_labels(root: Path) -> dict[str, Path]:
    label_dir = root / "labelsTr"
    if not label_dir.is_dir():
        raise FileNotFoundError(f"missing labelsTr directory: {label_dir}")

    discovered: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in sorted(label_dir.glob(f"*{_NIFTI_SUFFIX}")):
        if not path.is_file():
            continue
        case_id = path.name[: -len(_NIFTI_SUFFIX)]
        if not case_id:
            raise ValueError(f"invalid label filename with empty case ID: {path}")
        if case_id in discovered:
            duplicates.setdefault(case_id, [discovered[case_id]]).append(path)
        else:
            discovered[case_id] = path
    if duplicates:
        details = "; ".join(
            f"{case_id}: {[str(path) for path in paths]}"
            for case_id, paths in sorted(duplicates.items())
        )
        raise ValueError(f"duplicate label case IDs in {label_dir}: {details}")
    return discovered


def _format_case_ids(case_ids: set[str]) -> str:
    return ", ".join(sorted(case_ids)) or "none"


def _validate_pair_case_sets(
    root: Path,
    images: dict[str, Path],
    labels: dict[str, Path],
) -> None:
    missing_labels = set(images) - set(labels)
    extra_labels = set(labels) - set(images)
    if missing_labels or extra_labels:
        raise ValueError(
            f"image/label case IDs do not match in {root}; "
            f"missing labels: {_format_case_ids(missing_labels)}; "
            f"extra labels: {_format_case_ids(extra_labels)}"
        )


def _load_manifest(root: Path) -> dict[str, dict[str, Any]]:
    manifest_path = root / "roi_manifest.json"
    if not manifest_path.is_file():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read ROI manifest {manifest_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"ROI manifest must be a JSON object: {manifest_path}")
    cases = payload.get("cases", {})
    if cases is None:
        return {}
    if not isinstance(cases, dict):
        raise ValueError(f"ROI manifest 'cases' must be an object: {manifest_path}")

    result: dict[str, dict[str, Any]] = {}
    for case_id, fields in cases.items():
        if not isinstance(case_id, str) or not isinstance(fields, dict):
            raise ValueError(
                f"ROI manifest 'cases' must map case IDs to objects: {manifest_path}"
            )
        result[case_id] = fields
    return result


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return str(value).lower()
    return value


def _stat(values: list[int]) -> tuple[int | float | None, int | float | None, int | float | None]:
    if not values:
        return None, None, None
    return (
        min(values),
        float(np.median(np.asarray(values, dtype=np.int64))),
        max(values),
    )


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    coverage_rows = [row for row in rows if row["gt_coverage"] is not None]

    def threshold_case_ids(threshold: float) -> list[str]:
        return sorted(
            row["case_id"]
            for row in coverage_rows
            if row["gt_coverage"] < threshold
        )

    width_min, width_median, width_max = _stat(
        [row["roi_width"] for row in rows]
    )
    height_min, height_median, height_max = _stat(
        [row["roi_height"] for row in rows]
    )
    area_min, area_median, area_max = _stat([row["roi_area"] for row in rows])

    empty_case_ids = sorted(
        row["case_id"] for row in rows if row["cropped_gt_voxels"] == 0
    )
    result: dict[str, Any] = {
        "total_cases": len(rows),
        "cases_with_gt": sum(row["original_gt_voxels"] > 0 for row in rows),
        "cropped_gt_empty_count": len(empty_case_ids),
        "cropped_gt_empty_case_ids": empty_case_ids,
        "roi_width_min": width_min,
        "roi_width_median": width_median,
        "roi_width_max": width_max,
        "roi_height_min": height_min,
        "roi_height_median": height_median,
        "roi_height_max": height_max,
        "roi_area_min": area_min,
        "roi_area_median": area_median,
        "roi_area_max": area_max,
    }
    for threshold, label in ((0.5, "0_5"), (0.9, "0_9"), (1.0, "1_0")):
        case_ids = threshold_case_ids(threshold)
        result[f"gt_coverage_lt_{label}_count"] = len(case_ids)
        result[f"gt_coverage_lt_{label}_case_ids"] = case_ids

    for threshold in (32, 64, 128):
        result[f"roi_smaller_than_{threshold}x{threshold}_count"] = sum(
            row["roi_width"] < threshold or row["roi_height"] < threshold
            for row in rows
        )
    return result


def audit_roi_coverage(
    dataset501_root: Path,
    dataset504_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Audit paired Dataset501 and Dataset504 cases and write CSV/JSON outputs."""
    dataset501_root = Path(dataset501_root)
    dataset504_root = Path(dataset504_root)
    output_dir = Path(output_dir)

    images501 = _discover_images(dataset501_root)
    labels501 = _discover_labels(dataset501_root)
    images504 = _discover_images(dataset504_root)
    labels504 = _discover_labels(dataset504_root)
    _validate_pair_case_sets(dataset501_root, images501, labels501)
    _validate_pair_case_sets(dataset504_root, images504, labels504)

    cases501 = set(images501)
    cases504 = set(images504)
    if cases501 != cases504:
        raise ValueError(
            "Dataset501 and Dataset504 case IDs do not match; "
            f"missing in Dataset504: {_format_case_ids(cases501 - cases504)}; "
            f"extra in Dataset504: {_format_case_ids(cases504 - cases501)}"
        )

    manifest_cases = _load_manifest(dataset504_root)
    rows: list[dict[str, Any]] = []
    for case_id in sorted(cases501):
        image501 = read_nifti(images501[case_id])
        label501 = read_nifti(labels501[case_id])
        image504 = read_nifti(images504[case_id])
        label504 = read_nifti(labels504[case_id])

        if image501.array.shape != label501.array.shape:
            raise ValueError(
                f"Dataset501 image/label shape mismatch for {case_id}: "
                f"image={image501.array.shape}, label={label501.array.shape}"
            )
        if image504.array.shape != label504.array.shape:
            raise ValueError(
                f"Dataset504 image/label shape mismatch for {case_id}: "
                f"image={image504.array.shape}, label={label504.array.shape}"
            )
        if image501.array.shape[0] != image504.array.shape[0]:
            raise ValueError(
                f"z mismatch for {case_id}: Dataset501 z={image501.array.shape[0]}, "
                f"Dataset504 z={image504.array.shape[0]}"
            )

        original_gt_voxels = int(np.count_nonzero(label501.array > 0))
        cropped_gt_voxels = int(np.count_nonzero(label504.array > 0))
        coverage = (
            cropped_gt_voxels / original_gt_voxels
            if original_gt_voxels
            else None
        )
        original_z, original_y, original_x = image501.array.shape
        cropped_z, cropped_y, cropped_x = image504.array.shape
        row: dict[str, Any] = {
            "case_id": case_id,
            "original_shape_z": int(original_z),
            "original_shape_y": int(original_y),
            "original_shape_x": int(original_x),
            "cropped_shape_z": int(cropped_z),
            "cropped_shape_y": int(cropped_y),
            "cropped_shape_x": int(cropped_x),
            "roi_width": int(cropped_x),
            "roi_height": int(cropped_y),
            "roi_area": int(cropped_x * cropped_y),
            "original_gt_voxels": original_gt_voxels,
            "cropped_gt_voxels": cropped_gt_voxels,
            "gt_coverage": coverage,
        }
        row.update(
            {
                key: _csv_value(value)
                for key, value in manifest_cases.get(case_id, {}).items()
                if key not in _BASIC_FIELDS
            }
        )
        rows.append(row)

    manifest_fields = sorted(
        {
            key
            for row in rows
            for key in row
            if key not in _BASIC_FIELDS
        }
    )
    fieldnames = [*(_BASIC_FIELDS), *manifest_fields]
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "roi_coverage_cases.csv"
    summary_path = output_dir / "roi_coverage_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = _summary(rows)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit Dataset504 ROI sizes and GT coverage against Dataset501."
    )
    parser.add_argument("--dataset501-root", type=Path, required=True)
    parser.add_argument("--dataset504-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = audit_roi_coverage(
        args.dataset501_root,
        args.dataset504_root,
        args.output_dir,
    )
    print(
        f"Audited {summary['total_cases']} cases; "
        f"wrote {args.output_dir / 'roi_coverage_cases.csv'} and "
        f"{args.output_dir / 'roi_coverage_summary.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
