from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

import pytest

from standalone_nnunet2d.data.dataset import load_fold_cases
from standalone_nnunet2d.alignment_evidence import build_alignment_evidence
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
    run_state: str = "official_alignment_pending",
    alignment_evidence: dict[str, object] | None = None,
    failed_fold: int | None = None,
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
            "failed_case_count": 1 if fold == failed_fold else 0,
            "failed_cases": [f"failed_case_fold_{fold}"] if fold == failed_fold else [],
            "run_state": run_state,
        }
        if alignment_evidence is not None:
            payload["alignment_evidence"] = deepcopy(alignment_evidence)
        (root / f"fold_{fold}_report.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )


def _aligned_evidence(tmp_path: Path, *, suffix: str = "") -> dict[str, object]:
    components = {
        name: {"status": "passed", "diagnostics": []}
        for name in ("image", "label", "manifest", "mask")
    }
    transform_path = tmp_path / f"transform{suffix}.json"
    inference_path = tmp_path / f"inference{suffix}.json"
    transform_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "run_state": "official_alignment_pending",
                "oracle_root": f"/oracle/transform{suffix}",
                "standalone_root": f"/standalone/transform{suffix}",
                "image_atol": 0.0,
                "components": components,
                "diagnostics": [],
            }
        ),
        encoding="utf-8",
    )
    inference_path.write_text(
        json.dumps(
            {
                "parity_policy": "repeat_oracle_stability_v1",
                "oracle_roots": [
                    f"/oracle{suffix}/0",
                    f"/oracle{suffix}/1",
                    f"/oracle{suffix}/2",
                ],
                "oracle_repeat_count": 3,
                "stable_mask_mismatch_count": 0,
                "stable_mask_mismatch_coordinates": [],
                "unobserved_standalone_label_count": 0,
                "unobserved_standalone_label_coordinates": [],
                "status": "passed",
                "run_state": "official_alignment_pending",
                "standalone_root": f"/standalone/inference{suffix}",
                "image_atol": 0.0,
                "components": components,
                "diagnostics": [],
            }
        ),
        encoding="utf-8",
    )
    return build_alignment_evidence(transform_path, inference_path)


def _write_csv_fold_report(root: Path, fold: int) -> None:
    path = root / f"fold_{fold}_report.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case_id", "TP", "FP", "FN", "TN", "Dice", "IoU"),
        )
        writer.writeheader()
        for case_id in load_fold_cases(fold, "val"):
            writer.writerow(_case_record(case_id))


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


def test_aggregate_oof_promotes_five_identical_aligned_evidence_reports(
    tmp_path: Path,
) -> None:
    evidence = _aligned_evidence(tmp_path)
    _write_fold_reports(
        tmp_path,
        run_state="official_aligned",
        alignment_evidence=evidence,
    )

    summary = aggregate_oof(tmp_path)

    assert summary["run_state"] == "official_aligned"
    assert summary["alignment_evidence"] == evidence
    persisted = json.loads((tmp_path / "oof_summary.json").read_text(encoding="utf-8"))
    assert persisted["alignment_evidence"] == evidence


def test_aggregate_oof_rejects_aligned_reports_with_failed_cases(
    tmp_path: Path,
) -> None:
    evidence = _aligned_evidence(tmp_path)
    _write_fold_reports(
        tmp_path,
        run_state="official_aligned",
        alignment_evidence=evidence,
        failed_fold=2,
    )

    with pytest.raises(ValueError, match="failed_case_count|failed_cases|failed"):
        aggregate_oof(tmp_path)


def test_aggregate_oof_rejects_failed_case_count_length_mismatch(tmp_path: Path) -> None:
    evidence = _aligned_evidence(tmp_path)
    _write_fold_reports(
        tmp_path,
        run_state="official_aligned",
        alignment_evidence=evidence,
    )
    report_path = tmp_path / "fold_1_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["failed_case_count"] = 1
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="failed_case_count|failed_cases"):
        aggregate_oof(tmp_path)


def test_aggregate_oof_rejects_non_list_failed_cases(tmp_path: Path) -> None:
    evidence = _aligned_evidence(tmp_path)
    _write_fold_reports(
        tmp_path,
        run_state="official_aligned",
        alignment_evidence=evidence,
    )
    report_path = tmp_path / "fold_1_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["failed_cases"] = {"case_id": "case001"}
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="failed_cases"):
        aggregate_oof(tmp_path)


@pytest.mark.parametrize("failed_case_count", [0.0, -1, True])
def test_aggregate_oof_rejects_non_strict_failed_case_count(
    tmp_path: Path, failed_case_count: object
) -> None:
    evidence = _aligned_evidence(tmp_path)
    _write_fold_reports(
        tmp_path,
        run_state="official_aligned",
        alignment_evidence=evidence,
    )
    report_path = tmp_path / "fold_1_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["failed_case_count"] = failed_case_count
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="failed_case_count"):
        aggregate_oof(tmp_path)


def test_aggregate_oof_rejects_mixed_or_missing_aligned_evidence(tmp_path: Path) -> None:
    evidence = _aligned_evidence(tmp_path)
    _write_fold_reports(
        tmp_path,
        run_state="official_aligned",
        alignment_evidence=evidence,
    )
    fold_zero_path = tmp_path / "fold_0_report.json"
    fold_zero = json.loads(fold_zero_path.read_text(encoding="utf-8"))
    fold_zero["run_state"] = "official_alignment_pending"
    fold_zero.pop("alignment_evidence")
    fold_zero_path.write_text(json.dumps(fold_zero), encoding="utf-8")

    with pytest.raises(ValueError, match="mixed|pending|aligned"):
        aggregate_oof(tmp_path)

    _write_fold_reports(
        tmp_path,
        run_state="official_aligned",
        alignment_evidence=evidence,
    )
    fold_zero = json.loads(fold_zero_path.read_text(encoding="utf-8"))
    fold_zero.pop("alignment_evidence")
    fold_zero_path.write_text(json.dumps(fold_zero), encoding="utf-8")
    with pytest.raises(ValueError, match="alignment evidence"):
        aggregate_oof(tmp_path)


def test_aggregate_oof_rejects_different_aligned_evidence_records(tmp_path: Path) -> None:
    evidence = _aligned_evidence(tmp_path, suffix="-one")
    different = _aligned_evidence(tmp_path, suffix="-two")
    _write_fold_reports(
        tmp_path,
        run_state="official_aligned",
        alignment_evidence=evidence,
    )
    fold_four_path = tmp_path / "fold_4_report.json"
    fold_four = json.loads(fold_four_path.read_text(encoding="utf-8"))
    fold_four["alignment_evidence"] = different
    fold_four_path.write_text(json.dumps(fold_four), encoding="utf-8")

    with pytest.raises(ValueError, match="identical|differ|evidence"):
        aggregate_oof(tmp_path)


def test_aggregate_oof_treats_csv_fold_report_as_pending(tmp_path: Path) -> None:
    _write_fold_reports(tmp_path)
    (tmp_path / "fold_4_report.json").unlink()
    _write_csv_fold_report(tmp_path, 4)

    summary = aggregate_oof(tmp_path)

    assert summary["run_state"] == "official_alignment_pending"
    assert "alignment_evidence" not in summary


def test_validate_cv_rejects_wrong_raw_physical_channels_before_model_load(
    monkeypatch, tmp_path: Path
) -> None:
    import standalone_nnunet2d.validate_cv as validate_cv

    checkpoint = tmp_path / "tiny_c4_checkpoint.pt"
    checkpoint.touch()
    metadata = {
        "run_type": "official_alignment_pending",
        "run_state": "official_alignment_pending",
        "input_channels": 4,
        "resolved_config": {
            "input_mode": "dwi_adc_bilateral",
            "physical_input_channels": 2,
            "effective_model_input_channels": 4,
        },
    }
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    (raw_root / "dataset.json").write_text(
        json.dumps({"channel_names": {"0": "ADC", "1": "DWI", "2": "FLAIR"}}),
        encoding="utf-8",
    )
    load_model_called = False

    monkeypatch.setattr(validate_cv, "_load_checkpoint_metadata", lambda path: metadata)

    def fail_load_model(*args, **kwargs):
        nonlocal load_model_called
        load_model_called = True
        raise AssertionError("_load_model must not run for invalid raw channel metadata")

    monkeypatch.setattr(validate_cv, "_load_model", fail_load_model)

    with pytest.raises(ValueError, match=r"channel declaration.*must exactly match"):
        validate_cv.main(
            [
                "fold",
                "--checkpoint",
                str(checkpoint),
                "--raw-root",
                str(raw_root),
                "--fold",
                "0",
                "--output-root",
                str(tmp_path / "output"),
                "--allow-pending",
                "--input-mode",
                "dwi_adc_bilateral",
            ]
        )

    assert not load_model_called


def test_validate_cv_forwards_dwi_adc_bilateral_and_rejects_runtime_mismatch(
    monkeypatch, tmp_path
):
    import standalone_nnunet2d.validate_cv as validate_cv

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    metadata = {
        "run_type": "official_alignment_pending",
        "run_state": "official_alignment_pending",
        "input_channels": 4,
        "resolved_config": {
            "input_mode": "dwi_adc_bilateral",
            "physical_input_channels": 2,
            "effective_model_input_channels": 4,
        },
    }
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    (raw_root / "dataset.json").write_text(
        json.dumps({"channel_names": {"0": "DWI", "1": "ADC"}}),
        encoding="utf-8",
    )
    forwarded = {}
    load_model_called = []

    monkeypatch.setattr(validate_cv, "_load_checkpoint_metadata", lambda *args, **kwargs: metadata)

    def fake_validate_fold(*args, **kwargs):
        forwarded.update(kwargs)
        return None

    def fake_load_model(*args, **kwargs):
        load_model_called.append(True)
        return object(), metadata

    monkeypatch.setattr(validate_cv, "validate_fold", fake_validate_fold)
    monkeypatch.setattr(
        validate_cv,
        "_load_model",
        fake_load_model,
    )

    validate_cv.main(
        [
            "fold",
                "--checkpoint",
                str(checkpoint),
                "--raw-root",
                str(raw_root),
                "--fold",
                "0",
                "--output-root",
                str(tmp_path / "output"),
                "--allow-pending",
                "--input-mode",
                "dwi_adc_bilateral",
        ]
    )
    assert metadata["input_channels"] == 4
    assert metadata["resolved_config"] == {
        "input_mode": "dwi_adc_bilateral",
        "physical_input_channels": 2,
        "effective_model_input_channels": 4,
    }
    assert forwarded["input_mode"] == validate_cv.InputMode.DWI_ADC_BILATERAL
    assert "bilateral_asymmetry_channel" not in forwarded
    load_model_called.clear()

    with pytest.raises(ValueError):
        validate_cv.main(
            [
                "fold",
                "--checkpoint",
                str(checkpoint),
                "--raw-root",
                str(raw_root),
                "--fold",
                "0",
                "--output-root",
                str(tmp_path / "output"),
                "--allow-pending",
                "--input-mode",
                "dwi_bilateral",
            ]
        )
    assert not load_model_called
