"""Case-level full-volume validation for one supplied cross-validation fold."""

from __future__ import annotations

import csv
import copy
import json
from pathlib import Path
from typing import Any

import torch

from standalone_nnunet2d import alignment_evidence as alignment_evidence_module
from standalone_nnunet2d.data.dataset import load_fold_cases, validate_raw_root
from standalone_nnunet2d.data.nifti_io import read_nifti
from standalone_nnunet2d.engine.predictor import predict_volume, save_and_validate_prediction
from standalone_nnunet2d.metrics.segmentation_metrics import (
    METRIC_POLICY,
    case_metric_record,
)
from standalone_nnunet2d.training.official_config import DEFAULT_RUN_STATE


CASE_METRIC_FIELDS = ("case_id", "TP", "FP", "FN", "TN", "Dice", "IoU")


def _case_paths(raw_root: Path, case_id: str) -> tuple[Path, Path]:
    root = raw_root.resolve()
    return (
        root / "imagesTr" / f"{case_id}_0000.nii.gz",
        root / "labelsTr" / f"{case_id}.nii.gz",
    )


def _write_case_metrics(path: Path, records: list[dict[str, str | float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CASE_METRIC_FIELDS))
        writer.writeheader()
        writer.writerows({field: record[field] for field in CASE_METRIC_FIELDS} for record in records)


def _write_fold_report(
    path: Path,
    *,
    fold: int,
    records: list[dict[str, str | float | int]],
    failed_cases: list[dict[str, str]],
    run_state: str = DEFAULT_RUN_STATE,
    alignment_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_state, validated_evidence = (
        alignment_evidence_module.validate_checkpoint_alignment_metadata(
            {
                "run_type": run_state,
                "run_state": run_state,
                "alignment_evidence": alignment_evidence,
            }
        )
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "fold": fold,
        "case_count": len(records),
        "metric_per_case": records,
        "metric_policy": dict(METRIC_POLICY),
        "aggregation": METRIC_POLICY["aggregation"],
        "failed_case_count": len(failed_cases),
        "failed_cases": failed_cases,
        "run_state": resolved_state,
    }
    if validated_evidence is not None:
        report["alignment_evidence"] = copy.deepcopy(validated_evidence)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def validate_fold(
    model: torch.nn.Module,
    raw_root: Path,
    *,
    fold: int,
    output_root: Path,
    device: torch.device,
    run_state: str = DEFAULT_RUN_STATE,
    alignment_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate every held-out case with source-space full-volume inference.

    This path intentionally operates on one complete NIfTI volume at a time,
    independently of the online patch-level validation helper.
    """
    resolved_state, validated_evidence = (
        alignment_evidence_module.validate_checkpoint_alignment_metadata(
            {
                "run_type": run_state,
                "run_state": run_state,
                "alignment_evidence": alignment_evidence,
            }
        )
    )
    root = validate_raw_root(raw_root)
    case_ids = load_fold_cases(fold, "val")
    destination = output_root.resolve()
    prediction_root = destination / "predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str | float | int]] = []
    failed_cases: list[dict[str, str]] = []
    for case_id in case_ids:
        try:
            image_path, label_path = _case_paths(root, case_id)
            image = read_nifti(image_path)
            label = read_nifti(label_path)
            prediction = predict_volume(model, image, device)
            prediction_path = prediction_root / f"{case_id}.nii.gz"
            save_and_validate_prediction(prediction_path, prediction, image)
            records.append(case_metric_record(case_id, prediction, label.array))
        except Exception as exc:  # keep a deterministic failure record for the fold report
            failed_cases.append({"case_id": case_id, "error": f"{type(exc).__name__}: {exc}"})

    case_metrics_path = destination / f"fold_{fold}_case_metrics.csv"
    report_path = destination / f"fold_{fold}_report.json"
    _write_case_metrics(case_metrics_path, records)
    return _write_fold_report(
        report_path,
        fold=fold,
        records=records,
        failed_cases=failed_cases,
        run_state=resolved_state,
        alignment_evidence=validated_evidence,
    )
