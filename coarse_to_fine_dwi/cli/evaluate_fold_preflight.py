"""Evaluate one fixed fold as a non-formal coarse-to-fine screening run."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..dataset import _validate_fixed_splits
from ..evaluate import compare_full_volume_predictions
from ..provenance import validate_stage1_provenance
from .restore_predictions import restore_predictions


EXPECTED_FOLD_COUNT = 5
EXPECTED_CASE_COUNT = 19


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _manifest_rows(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    cases = payload.get("cases")
    if isinstance(cases, dict):
        rows = []
        for case_id, row in cases.items():
            if not isinstance(row, dict):
                raise ValueError(f"manifest case {case_id} must be an object")
            rows.append({"case_id": case_id, **row})
    elif isinstance(cases, list):
        rows = cases
    else:
        raise ValueError("manifest must contain a cases object or list")

    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("case_id"), str):
            raise ValueError("manifest cases must contain string case_id values")
        case_id = row["case_id"]
        if case_id in result:
            raise ValueError(f"manifest contains duplicate case ID: {case_id}")
        result[case_id] = row
    return result


def _selected_fold_case_ids(manifest: Path, splits: Path, fold: int) -> list[str]:
    if fold not in range(EXPECTED_FOLD_COUNT):
        raise ValueError("fold must be between 0 and 4")
    fixed_splits, _ = _validate_fixed_splits(splits.resolve())
    selected_ids = list(fixed_splits[fold]["val"])
    if len(selected_ids) != EXPECTED_CASE_COUNT:
        raise ValueError(f"fold {fold} must contain exactly {EXPECTED_CASE_COUNT} validation IDs")

    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"manifest does not exist: {manifest}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"manifest is not valid JSON: {manifest}") from error
    if not isinstance(payload, dict):
        raise ValueError("manifest must be a JSON object")
    rows = _manifest_rows(payload)
    missing = sorted(set(selected_ids) - set(rows))
    if missing:
        raise ValueError(f"manifest is missing selected case IDs: {missing}")
    for case_id in selected_ids:
        if rows[case_id].get("fold") != fold:
            raise ValueError(f"manifest fold metadata mismatch for {case_id}: expected {fold}")
    return selected_ids


def _validated_provenance(
    provenance_path: Path,
    *,
    dataset501_raw: Path,
    splits: Path,
    stage1_oof_dir: Path,
) -> dict[str, Any]:
    provenance = validate_stage1_provenance(provenance_path)
    expected = {
        "dataset501_raw": dataset501_raw.resolve(),
        "splits_path": splits.resolve(),
        "stage1_oof_dir": stage1_oof_dir.resolve(),
    }
    for field, expected_path in expected.items():
        actual = Path(provenance[field]).resolve()
        if actual != expected_path:
            raise ValueError(f"{field} does not match verified provenance")
    return provenance


def evaluate_fold_preflight(
    *,
    manifest: str | Path,
    stage2_cropped_predictions: str | Path,
    dataset501_raw: str | Path,
    splits: str | Path,
    fold: int,
    stage1_oof_dir: str | Path,
    stage1_provenance: str | Path,
    restored_output_dir: str | Path,
    evaluation_output_dir: str | Path,
) -> tuple[Path, Path]:
    """Restore and compare exactly one fixed validation fold.

    This function deliberately passes a selected case set to both existing
    subset-capable APIs, so its comparison is a screening diagnostic and not
    a formal 95-case evaluation.
    """
    manifest_path = Path(manifest).resolve()
    cropped_predictions = Path(stage2_cropped_predictions).resolve()
    raw_root = Path(dataset501_raw).resolve()
    splits_path = Path(splits).resolve()
    stage1_root = Path(stage1_oof_dir).resolve()
    provenance_path = Path(stage1_provenance).resolve()
    restored_root = Path(restored_output_dir).resolve()
    evaluation_root = Path(evaluation_output_dir).resolve()

    if _paths_overlap(restored_root, evaluation_root):
        raise ValueError("restored and evaluation output directories overlap")

    selected_ids = _selected_fold_case_ids(manifest_path, splits_path, fold)
    provenance = _validated_provenance(
        provenance_path,
        dataset501_raw=raw_root,
        splits=splits_path,
        stage1_oof_dir=stage1_root,
    )

    restored = restore_predictions(
        manifest=manifest_path,
        cropped_predictions=cropped_predictions,
        dataset501_raw=raw_root,
        output_dir=restored_root,
        selected_case_ids=selected_ids,
    )
    csv_path, summary_path = compare_full_volume_predictions(
        labels_dir=raw_root / "labelsTr",
        stage1_dir=stage1_root,
        stage2_restored_dir=restored,
        output_dir=evaluation_root,
        expected_case_count=EXPECTED_CASE_COUNT,
        provenance=provenance,
        selected_case_ids=selected_ids,
    )

    metadata = {
        "run_kind": "fold_preflight",
        "formal_eligible": False,
        "fold": fold,
        "case_count": EXPECTED_CASE_COUNT,
        "case_ids": sorted(selected_ids),
        "manifest": str(manifest_path),
        "splits": str(splits_path),
        "stage1_oof_dir": str(stage1_root),
        "stage1_provenance": str(provenance_path),
        "stage2_cropped_predictions": str(cropped_predictions),
        "restored_output_dir": str(restored_root),
        "evaluation_output_dir": str(evaluation_root),
    }
    (evaluation_root / "fold_preflight_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return csv_path, summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage2-cropped-predictions", type=Path, required=True)
    parser.add_argument("--dataset501-raw", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(EXPECTED_FOLD_COUNT), required=True)
    parser.add_argument("--stage1-oof-dir", type=Path, required=True)
    parser.add_argument("--stage1-provenance", type=Path, required=True)
    parser.add_argument("--restored-output-dir", type=Path, required=True)
    parser.add_argument("--evaluation-output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        csv_path, summary_path = evaluate_fold_preflight(
            manifest=args.manifest,
            stage2_cropped_predictions=args.stage2_cropped_predictions,
            dataset501_raw=args.dataset501_raw,
            splits=args.splits,
            fold=args.fold,
            stage1_oof_dir=args.stage1_oof_dir,
            stage1_provenance=args.stage1_provenance,
            restored_output_dir=args.restored_output_dir,
            evaluation_output_dir=args.evaluation_output_dir,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"fold preflight failed: {error}")
        return 2
    print(f"case_metrics={csv_path}")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
