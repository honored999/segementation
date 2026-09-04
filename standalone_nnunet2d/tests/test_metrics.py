from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from standalone_nnunet2d.metrics.crossval_summary import extract_reference_baseline, summarize_oof_cases
from standalone_nnunet2d.metrics.segmentation_metrics import binary_segmentation_metrics, case_metric_record


def test_binary_metrics_report_perfect_overlap() -> None:
    result = binary_segmentation_metrics(np.array([[0, 1], [1, 0]]), np.array([[0, 1], [1, 0]]))

    assert result["TP"] == 2
    assert result["FP"] == 0
    assert result["FN"] == 0
    assert result["Dice"] == pytest.approx(1.0)
    assert result["IoU"] == pytest.approx(1.0)


def test_binary_metrics_define_empty_masks_as_perfect_agreement() -> None:
    result = binary_segmentation_metrics(np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8))

    assert result["Dice"] == 1.0
    assert result["IoU"] == 1.0


def test_case_record_is_json_safe_and_oof_summary_averages_metrics() -> None:
    records = [
        case_metric_record("case001", np.array([0, 1]), np.array([0, 1])),
        case_metric_record("case002", np.array([0, 0]), np.array([0, 1])),
    ]

    summary = summarize_oof_cases(records)

    assert summary["case_count"] == 2
    assert summary["foreground_mean"]["Dice"] == pytest.approx(0.5)
    assert summary["metric_per_case"] == records


def test_oof_summary_rejects_duplicate_case_ids() -> None:
    record = {"case_id": "case001", "Dice": 1.0, "IoU": 1.0, "TP": 1, "FP": 0, "FN": 0, "TN": 1}

    with pytest.raises(ValueError, match="duplicate"):
        summarize_oof_cases([record, record])


def test_oof_summary_records_metric_policy_and_pending_run_state() -> None:
    record = {"case_id": "case001", "Dice": 1.0, "IoU": 1.0, "TP": 0, "FP": 0, "FN": 0, "TN": 4}

    summary = summarize_oof_cases([record])

    assert summary["metric_policy"] == {
        "foreground": 1,
        "postprocessing": "argmax",
        "both_empty": "dice=1",
        "one_empty": "dice=0",
        "aggregation": "case_macro_mean",
    }
    assert summary["aggregation"] == "case_macro_mean"
    assert summary["run_state"] == "official_alignment_pending"


def test_reference_baseline_matches_supplied_summary() -> None:
    baseline = extract_reference_baseline(Path("standalone_nnunet2d/reference/summary.json"))

    assert baseline["Dice"] == pytest.approx(0.731103738314918)
    assert baseline["IoU"] == pytest.approx(0.5923877518050135)
    assert baseline["case_count"] == 95
