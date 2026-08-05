from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from standalone_nnunet2d.data.dataset import load_fold_cases
from standalone_nnunet2d.metrics.segmentation_metrics import METRIC_POLICY
from standalone_nnunet2d.validate_cv import aggregate_oof


def _case_record(case_id: str) -> dict[str, str | float | int]:
    return {
        "case_id": case_id,
        "Dice": 1.0,
        "IoU": 1.0,
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "TN": 12,
    }


def _write_fold_reports(
    root: Path,
    *,
    missing_case: bool = False,
    duplicate_case: bool = False,
) -> None:
    for fold in range(5):
        case_ids = list(load_fold_cases(fold, "val"))
        if missing_case and fold == 4:
            case_ids.pop()
        if duplicate_case and fold == 4:
            case_ids[-1] = load_fold_cases(0, "val")[0]
        payload = {
            "fold": fold,
            "case_count": len(case_ids),
            "metric_per_case": [_case_record(case_id) for case_id in case_ids],
            "metric_policy": dict(METRIC_POLICY),
            "aggregation": "case_macro_mean",
            "failed_case_count": 0,
            "run_state": "official_alignment_pending",
        }
        (root / f"fold_{fold}_report.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )


def test_aggregate_oof_reads_five_fold_reports_and_writes_strict_95_case_outputs(tmp_path: Path) -> None:
    _write_fold_reports(tmp_path)

    aggregate_oof(tmp_path)

    per_case_path = tmp_path / "oof_per_case_metrics.csv"
    summary_path = tmp_path / "oof_summary.json"
    assert per_case_path.is_file()
    assert summary_path.is_file()

    with per_case_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 95
    assert len({row["case_id"] for row in rows}) == 95
    assert {row["case_id"] for row in rows} == {
        case_id for fold in range(5) for case_id in load_fold_cases(fold, "val")
    }

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["case_count"] == 95
    assert summary["aggregation"] == "case_macro_mean"
    assert summary["metric_policy"] == dict(METRIC_POLICY)
    assert summary["failed_case_count"] == 0
    assert summary["run_state"] == "official_alignment_pending"


@pytest.mark.parametrize("invalid_report", ["missing_case", "duplicate_case"])
def test_aggregate_oof_rejects_missing_or_duplicate_validation_ids_with_95_unique_error(
    tmp_path: Path, invalid_report: str
) -> None:
    _write_fold_reports(
        tmp_path,
        missing_case=invalid_report == "missing_case",
        duplicate_case=invalid_report == "duplicate_case",
    )

    with pytest.raises(ValueError, match="95 unique"):
        aggregate_oof(tmp_path)


def test_aggregate_oof_requires_exactly_five_fold_reports(tmp_path: Path) -> None:
    _write_fold_reports(tmp_path)
    (tmp_path / "fold_5_report.json").write_text(
        (tmp_path / "fold_4_report.json").read_text(encoding="utf-8"), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="exactly five|5 fold"):
        aggregate_oof(tmp_path)
