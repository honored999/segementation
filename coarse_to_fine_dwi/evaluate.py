"""Original-volume binary evaluation for the coarse-to-fine pipeline."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .nifti import NiftiVolume, assert_compatible
from .roi import validate_binary_prediction


CSV_FIELDS = [
    "case_id",
    "gt_voxels",
    "stage1_voxels",
    "stage2_voxels",
    "stage1_tp",
    "stage1_fp",
    "stage1_fn",
    "stage1_dice",
    "stage1_iou",
    "stage1_precision",
    "stage1_recall",
    "stage2_tp",
    "stage2_fp",
    "stage2_fn",
    "stage2_dice",
    "stage2_iou",
    "stage2_precision",
    "stage2_recall",
    "dice_delta",
    "iou_delta",
    "precision_delta",
    "recall_delta",
]


def _case_id(path: Path) -> str | None:
    if path.name.endswith(".nii.gz"):
        return path.name[:-7]
    if path.name.endswith(".nii"):
        return path.name[:-4]
    return None


def _discover_nifti_files(directory: Path, *, kind: str) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"{kind} directory does not exist: {directory}")
    found: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        case_id = _case_id(path)
        if case_id is None:
            continue
        if case_id in found:
            raise ValueError(f"{kind} contains duplicate case ID: {case_id}")
        found[case_id] = path
    return found


def _require_exact_ids(
    found: dict[str, Path], expected: set[str], *, kind: str
) -> None:
    actual = set(found)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"{kind} IDs mismatch: missing={missing}, extra={extra}")


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _has_verified_formal_provenance(
    provenance: Mapping[str, Any] | None, *, expected_case_count: int
) -> bool:
    if expected_case_count != 95 or not isinstance(provenance, Mapping):
        return False
    required = {
        "verified": True,
        "stage1_trainer": "nnUNetTrainer",
        "stage1_prediction_source": "complete_5_fold_oof",
        "roi_source": "stage1_prediction_only",
        "split_policy": "fixed_5_fold_patient_level",
        "num_folds": 5,
        "case_count": 95,
    }
    return all(provenance.get(key) == value for key, value in required.items())


def _metrics_from_counts(
    *,
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    gt_voxels: int,
    prediction_voxels: int,
) -> dict[str, Any]:
    union = tp + fp + fn
    both_empty = gt_voxels == 0 and prediction_voxels == 0
    if both_empty:
        dice = iou = precision = recall = 1.0
    else:
        dice = (2.0 * tp) / (2 * tp + fp + fn)
        iou = tp / union if union else 0.0
        precision = tp / (tp + fp) if prediction_voxels else 0.0
        recall = tp / (tp + fn) if gt_voxels else 0.0
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "gt_voxels": int(gt_voxels),
        "prediction_voxels": int(prediction_voxels),
    }


def binary_case_metrics(ground_truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    """Calculate foreground metrics for one complete 3-D binary case."""
    ground_truth_array = np.asarray(ground_truth)
    prediction_array = np.asarray(prediction)
    if ground_truth_array.ndim != 3 or prediction_array.ndim != 3:
        raise ValueError("ground truth and prediction must be 3-dimensional")
    if ground_truth_array.shape != prediction_array.shape:
        raise ValueError(
            f"ground truth and prediction shape mismatch: {ground_truth_array.shape} vs {prediction_array.shape}"
        )
    try:
        gt = validate_binary_prediction(ground_truth_array).astype(bool)
        pred = validate_binary_prediction(prediction_array).astype(bool)
    except ValueError as error:
        raise ValueError("ground truth and prediction must be binary masks") from error
    tp = int(np.count_nonzero(gt & pred))
    fp = int(np.count_nonzero(~gt & pred))
    fn = int(np.count_nonzero(gt & ~pred))
    tn = int(np.count_nonzero(~gt & ~pred))
    return _metrics_from_counts(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        gt_voxels=int(np.count_nonzero(gt)),
        prediction_voxels=int(np.count_nonzero(pred)),
    )


def _case_row(case_id: str, stage1: dict[str, Any], stage2: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_id": case_id,
        "gt_voxels": stage1["gt_voxels"],
        "stage1_voxels": stage1["prediction_voxels"],
        "stage2_voxels": stage2["prediction_voxels"],
    }
    for prefix, metrics in (("stage1", stage1), ("stage2", stage2)):
        for name in ("tp", "fp", "fn", "dice", "iou", "precision", "recall"):
            row[f"{prefix}_{name}"] = metrics[name]
    for name in ("dice", "iou", "precision", "recall"):
        row[f"{name}_delta"] = stage2[name] - stage1[name]
    return row


def _macro(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    return {
        name: float(np.mean([row[f"{prefix}_{name}"] for row in rows]))
        for name in ("dice", "iou", "precision", "recall")
    }


def _pooled(rows: list[dict[str, Any]], prefix: str, total_voxels: int) -> dict[str, Any]:
    tp = sum(int(row[f"{prefix}_tp"]) for row in rows)
    fp = sum(int(row[f"{prefix}_fp"]) for row in rows)
    fn = sum(int(row[f"{prefix}_fn"]) for row in rows)
    gt_voxels = sum(int(row["gt_voxels"]) for row in rows)
    prediction_voxels = sum(int(row[f"{prefix}_voxels"]) for row in rows)
    tn = total_voxels - tp - fp - fn
    return _metrics_from_counts(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        gt_voxels=gt_voxels,
        prediction_voxels=prediction_voxels,
    )


def compare_full_volume_predictions(
    *,
    labels_dir: Path,
    stage1_dir: Path,
    stage2_restored_dir: Path,
    output_dir: Path,
    expected_case_count: int = 95,
    provenance: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Compare predictions in original space with optional verified provenance."""
    if expected_case_count <= 0:
        raise ValueError("expected_case_count must be positive")
    labels = _discover_nifti_files(Path(labels_dir).resolve(), kind="labels")
    stage1 = _discover_nifti_files(Path(stage1_dir).resolve(), kind="Stage1 predictions")
    stage2 = _discover_nifti_files(Path(stage2_restored_dir).resolve(), kind="Stage2 restored predictions")
    expected_ids = set(labels)
    if len(expected_ids) != expected_case_count:
        raise ValueError(
            f"labels must contain exactly {expected_case_count} cases, found {len(expected_ids)}"
        )
    _require_exact_ids(stage1, expected_ids, kind="Stage1 predictions")
    _require_exact_ids(stage2, expected_ids, kind="Stage2 restored predictions")

    rows: list[dict[str, Any]] = []
    total_voxels = 0
    for case_id in sorted(expected_ids):
        reference = NiftiVolume.read(labels[case_id])
        stage1_volume = NiftiVolume.read(stage1[case_id])
        stage2_volume = NiftiVolume.read(stage2[case_id])
        assert_compatible(reference, stage1_volume)
        assert_compatible(reference, stage2_volume)
        stage1_metrics = binary_case_metrics(reference.array, stage1_volume.array)
        stage2_metrics = binary_case_metrics(reference.array, stage2_volume.array)
        rows.append(_case_row(case_id, stage1_metrics, stage2_metrics))
        total_voxels += int(reference.array.size)

    output_root = Path(output_dir).resolve()
    input_roots = {
        "labels": Path(labels_dir).resolve(),
        "Stage1 predictions": Path(stage1_dir).resolve(),
        "Stage2 restored predictions": Path(stage2_restored_dir).resolve(),
    }
    for kind, input_root in input_roots.items():
        if _paths_overlap(output_root, input_root):
            raise ValueError(f"output_dir overlaps {kind} directory: {output_root}")
    if output_root.exists():
        if not output_root.is_dir():
            raise ValueError(f"output_dir is not a directory: {output_root}")
        if any(output_root.iterdir()):
            raise ValueError(f"output_dir must be empty or absent: {output_root}")
    else:
        output_root.mkdir(parents=True)
    csv_path = output_root / "stage1_vs_stage2_case_metrics.csv"
    json_path = output_root / "stage1_vs_stage2_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    stage1_macro = _macro(rows, "stage1")
    stage2_macro = _macro(rows, "stage2")
    deltas = {name: stage2_macro[name] - stage1_macro[name] for name in stage1_macro}
    summary = {
        "protocol": {
            "space": "original_full_volume",
            "gt_source": "Dataset501_labelsTr",
            "case_aggregation": "equal_case_macro",
        },
        "formal_eligible": _has_verified_formal_provenance(
            provenance, expected_case_count=expected_case_count
        ),
        "case_count": len(rows),
        "case_ids": sorted(expected_ids),
        "stage1_case_macro": stage1_macro,
        "stage2_case_macro": stage2_macro,
        "stage2_minus_stage1": deltas,
        "global": {
            "stage1": _pooled(rows, "stage1", total_voxels),
            "stage2": _pooled(rows, "stage2", total_voxels),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return csv_path, json_path
