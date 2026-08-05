"""Strict aggregation helpers for caller-supplied out-of-fold case metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path

import numpy as np

from standalone_nnunet2d.metrics.segmentation_metrics import METRIC_POLICY
from standalone_nnunet2d.training.official_config import DEFAULT_RUN_STATE


def summarize_oof_cases(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate unique case records without pretending they are predictions."""
    case_ids = [str(record.get("case_id", "")) for record in records]
    if not case_ids or any(not case_id for case_id in case_ids):
        raise ValueError("OOF records require non-empty case_id values")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate case IDs are not valid OOF results")
    required_metrics = ("Dice", "IoU", "TP", "FP", "FN", "TN")
    for record in records:
        missing_metrics = [metric for metric in required_metrics if metric not in record]
        if missing_metrics:
            raise ValueError(f"OOF record is missing metrics: {missing_metrics}")
    copied_records = [dict(record) for record in records]
    return {
        "case_count": len(copied_records),
        "foreground_mean": {
            metric: float(np.mean([float(record[metric]) for record in copied_records]))
            for metric in ("Dice", "IoU")
        },
        "metric_per_case": copied_records,
        "metric_policy": dict(METRIC_POLICY),
        "aggregation": METRIC_POLICY["aggregation"],
        "run_state": DEFAULT_RUN_STATE,
    }


def extract_reference_baseline(summary_path: Path) -> dict[str, float | int]:
    """Read only the official foreground baseline and its case count from JSON."""
    if not summary_path.is_file():
        raise FileNotFoundError(f"reference summary does not exist: {summary_path}")
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    foreground = summary.get("foreground_mean")
    per_case = summary.get("metric_per_case")
    if not isinstance(foreground, dict) or not isinstance(per_case, list):
        raise ValueError("summary must contain foreground_mean and metric_per_case")
    if "Dice" not in foreground or "IoU" not in foreground:
        raise ValueError("summary foreground_mean must contain Dice and IoU")
    return {"Dice": float(foreground["Dice"]), "IoU": float(foreground["IoU"]), "case_count": len(per_case)}
