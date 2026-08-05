"""Fold-validation CLI and strict five-fold out-of-fold aggregation."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

from standalone_nnunet2d.data.dataset import load_fold_cases
from standalone_nnunet2d.engine.formal_validation import CASE_METRIC_FIELDS, validate_fold
from standalone_nnunet2d.metrics.crossval_summary import summarize_oof_cases
from standalone_nnunet2d.metrics.segmentation_metrics import METRIC_POLICY
from standalone_nnunet2d.predict import _load_model, _read_checkpoint
from standalone_nnunet2d.training.official_config import DEFAULT_RUN_STATE


_REPORT_NAME = re.compile(r"fold_(\d+)_report\.(json|csv)$")
_CONFUSION_FIELDS = {"TP", "FP", "FN", "TN"}


def _report_paths(root: Path) -> list[Path]:
    candidates = sorted((*root.glob("fold_*_report.json"), *root.glob("fold_*_report.csv")))
    if len(candidates) != 5:
        raise ValueError(
            "OOF aggregation requires exactly five fold reports and exactly 95 unique "
            f"supplied validation IDs; found {len(candidates)} fold reports"
        )
    folds: list[int] = []
    for path in candidates:
        match = _REPORT_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"invalid fold report filename: {path.name}")
        folds.append(int(match.group(1)))
    if sorted(folds) != list(range(5)):
        raise ValueError(f"fold reports must cover exactly folds 0-4, found {sorted(folds)}")
    return candidates


def _coerce_json_records(payload: Mapping[str, Any], path: Path) -> tuple[list[dict[str, Any]], int]:
    raw_records = payload.get("metric_per_case", payload.get("case_metrics"))
    if not isinstance(raw_records, list):
        raise ValueError(f"fold report is missing metric_per_case: {path}")
    records: list[dict[str, Any]] = []
    for raw_record in raw_records:
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"fold report contains a non-object case record: {path}")
        records.append(dict(raw_record))
    failed_case_count = payload.get("failed_case_count", 0)
    if isinstance(failed_case_count, bool) or not isinstance(failed_case_count, (int, float)):
        raise ValueError(f"fold report has an invalid failed_case_count: {path}")
    return records, int(failed_case_count)


def _read_fold_report(path: Path) -> tuple[list[dict[str, Any]], int]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"fold report must contain a JSON object: {path}")
        return _coerce_json_records(payload, path)

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    records: list[dict[str, Any]] = []
    for row in rows:
        if any(field not in row or row[field] == "" for field in CASE_METRIC_FIELDS):
            raise ValueError(f"fold report CSV is missing case metrics: {path}")
        record: dict[str, Any] = {"case_id": row["case_id"]}
        record.update({field: int(row[field]) for field in _CONFUSION_FIELDS})
        record.update({field: float(row[field]) for field in ("Dice", "IoU")})
        records.append(record)
    return records, 0


def _validate_supplied_oof_ids(records: list[dict[str, Any]]) -> None:
    expected_ids = {
        case_id for fold in range(5) for case_id in load_fold_cases(fold, "val")
    }
    actual_ids = [str(record.get("case_id", "")) for record in records]
    unique_ids = set(actual_ids)
    if len(actual_ids) != 95 or len(unique_ids) != 95 or unique_ids != expected_ids:
        raise ValueError(
            "OOF aggregation requires exactly 95 unique supplied validation IDs; "
            f"got {len(actual_ids)} records and {len(unique_ids)} unique IDs"
        )


def _write_oof_csv(path: Path, records: list[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CASE_METRIC_FIELDS))
        writer.writeheader()
        writer.writerows({field: record[field] for field in CASE_METRIC_FIELDS} for record in records)


def aggregate_oof(output_root: Path) -> dict[str, Any]:
    """Aggregate exactly five fold reports over the supplied 95 validation IDs."""
    root = output_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"OOF report directory does not exist: {root}")

    records: list[dict[str, Any]] = []
    failed_case_count = 0
    for report_path in _report_paths(root):
        fold_records, failed_count = _read_fold_report(report_path)
        records.extend(fold_records)
        failed_case_count += failed_count

    _validate_supplied_oof_ids(records)
    summary = summarize_oof_cases(records)
    summary["metric_policy"] = dict(METRIC_POLICY)
    summary["aggregation"] = METRIC_POLICY["aggregation"]
    summary["failed_case_count"] = failed_case_count
    summary["run_state"] = DEFAULT_RUN_STATE

    _write_oof_csv(root / "oof_per_case_metrics.csv", records)
    (root / "oof_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Formal fold validation and strict OOF aggregation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fold_parser = subparsers.add_parser("fold", help="validate one held-out fold")
    fold_parser.add_argument("--checkpoint", required=True, type=Path)
    fold_parser.add_argument("--raw-root", required=True, type=Path)
    fold_parser.add_argument("--fold", required=True, type=int)
    fold_parser.add_argument("--output-root", required=True, type=Path)
    fold_parser.add_argument("--device", default="cpu")
    fold_parser.add_argument("--allow-pending", action="store_true")

    aggregate_parser = subparsers.add_parser("aggregate", help="aggregate five fold reports")
    aggregate_parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "aggregate":
        aggregate_oof(arguments.output_root)
        return 0

    _, metadata = _read_checkpoint(arguments.checkpoint)
    run_state = str(metadata.get("run_state", metadata.get("run_type", "")))
    if run_state == DEFAULT_RUN_STATE and not arguments.allow_pending:
        raise ValueError("pending checkpoint requires explicit --allow-pending")
    if run_state == "official_aligned":
        raise ValueError("official alignment cannot be claimed without a passed parity report")

    import torch

    model, _ = _load_model(arguments.checkpoint, torch.device(arguments.device))
    validate_fold(
        model,
        arguments.raw_root,
        fold=arguments.fold,
        output_root=arguments.output_root,
        device=torch.device(arguments.device),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
