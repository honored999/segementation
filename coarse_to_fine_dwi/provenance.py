"""Build and revalidate evidence-backed Stage-1 OOF provenance."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import EXPECTED_NUM_CASES, EXPECTED_NUM_FOLDS, _validate_fixed_splits
from .nifti import NiftiVolume


PROVENANCE_VERSION = 1
STAGE1_TRAINER = "nnUNetTrainer"
STAGE1_RESULT_FOLDER = "nnUNetTrainer__nnUNetPlans__2d"
STAGE1_OOF_FOLDER = "crossval_results_folds_0_1_2_3_4"


def _case_id(path: Path) -> str | None:
    if path.name.endswith(".nii.gz"):
        return path.name[:-7]
    if path.name.endswith(".nii"):
        return path.name[:-4]
    return None


def _discover_files(directory: Path, *, kind: str) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"required {kind} directory does not exist: {directory}")
    found: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        case_id = _case_id(path)
        if case_id is None:
            continue
        if case_id in found:
            raise ValueError(f"{kind} contains duplicate case ID: {case_id}")
        found[case_id] = path.resolve()
    return found


def _discover_dataset_files(directory: Path, *, kind: str, image: bool) -> dict[str, Path]:
    files = _discover_files(directory, kind=kind)
    normalized: dict[str, Path] = {}
    for file_case_id, path in files.items():
        if image and file_case_id.endswith("_0000"):
            case_id = file_case_id[:-5]
        else:
            case_id = file_case_id
        if case_id in normalized:
            raise ValueError(f"{kind} contains duplicate case ID: {case_id}")
        normalized[case_id] = path
    return normalized


def _require_exact(found: Mapping[str, Path], expected: set[str], *, kind: str) -> None:
    actual = set(found)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"{kind} IDs mismatch: missing={missing}, extra={extra}")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def nifti_fingerprint(path: Path) -> dict[str, Any]:
    """Read a NIfTI and return reproducible content and spatial evidence."""
    volume = NiftiVolume.read(path)
    array = np.ascontiguousarray(volume.array)
    content_hasher = hashlib.sha256()
    content_hasher.update(array.dtype.str.encode("ascii"))
    content_hasher.update(_canonical_json(list(array.shape)))
    content_hasher.update(array.tobytes(order="C"))
    spatial = {
        "shape_zyx": list(volume.shape_zyx),
        "spacing_xyz": list(volume.spacing_xyz),
        "origin_xyz": list(volume.origin_xyz),
        "direction": list(volume.direction),
    }
    spatial_sha256 = hashlib.sha256(_canonical_json(spatial)).hexdigest()
    content_sha256 = content_hasher.hexdigest()
    fingerprint_sha256 = hashlib.sha256(
        _canonical_json({"content_sha256": content_sha256, "spatial_sha256": spatial_sha256})
    ).hexdigest()
    return {
        "content_sha256": content_sha256,
        "spatial_sha256": spatial_sha256,
        "fingerprint_sha256": fingerprint_sha256,
        "dtype": array.dtype.str,
        **spatial,
    }


def _fingerprint_pair(first: Path, second: Path, *, case_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    first_fingerprint = nifti_fingerprint(first)
    second_fingerprint = nifti_fingerprint(second)
    if first_fingerprint != second_fingerprint:
        raise ValueError(f"case {case_id} fold-to-combined fingerprint mismatch")
    return first_fingerprint, second_fingerprint


def _validate_fold_directory(fold_index: int, directory: Path) -> None:
    if directory.name != "validation" or directory.parent.name != f"fold_{fold_index}":
        raise ValueError(
            f"fold {fold_index} validation directory must end in fold_{fold_index}\\validation: {directory}"
        )
    if directory.parent.parent.name != STAGE1_RESULT_FOLDER:
        raise ValueError(
            f"fold {fold_index} validation directory must come from {STAGE1_RESULT_FOLDER}"
        )


def _collect_provenance(
    dataset501_raw: Path,
    splits_path: Path,
    stage1_oof_dir: Path,
    fold_validation_dirs: Mapping[int, Path],
) -> dict[str, Any]:
    if set(fold_validation_dirs) != set(range(EXPECTED_NUM_FOLDS)):
        raise ValueError("fold validation directories must contain exactly fold 0 through fold 4")

    normalized_splits, case_ids_tuple = _validate_fixed_splits(splits_path.resolve())
    expected_ids = set(case_ids_tuple)
    raw_root = dataset501_raw.resolve()
    combined_root = stage1_oof_dir.resolve()
    if combined_root.name != STAGE1_OOF_FOLDER:
        raise ValueError(
            f"combined OOF directory must be named {STAGE1_OOF_FOLDER}: {combined_root}"
        )
    images = _discover_dataset_files(raw_root / "imagesTr", kind="imagesTr", image=True)
    labels = _discover_dataset_files(raw_root / "labelsTr", kind="labelsTr", image=False)
    _require_exact(images, expected_ids, kind="imagesTr")
    _require_exact(labels, expected_ids, kind="labelsTr")
    combined = _discover_files(combined_root, kind="combined OOF predictions")
    _require_exact(combined, expected_ids, kind="combined OOF predictions")

    fold_evidence: list[dict[str, Any]] = []
    case_evidence: dict[str, dict[str, Any]] = {}
    seen_validation_ids: set[str] = set()
    result_roots: set[Path] = set()
    for fold_index, fold in enumerate(normalized_splits):
        validation_dir = fold_validation_dirs[fold_index].resolve()
        _validate_fold_directory(fold_index, validation_dir)
        result_roots.add(validation_dir.parent.parent)
        expected_fold_ids = set(fold["val"])
        validation = _discover_files(validation_dir, kind=f"fold {fold_index} validation")
        _require_exact(validation, expected_fold_ids, kind=f"fold {fold_index} validation")
        overlap = seen_validation_ids & set(validation)
        if overlap:
            raise ValueError(f"validation folds contain duplicate case IDs: {sorted(overlap)}")
        seen_validation_ids.update(validation)

        prediction_files: dict[str, dict[str, Any]] = {}
        for case_id in sorted(expected_fold_ids):
            validation_fingerprint, combined_fingerprint = _fingerprint_pair(
                validation[case_id], combined[case_id], case_id=case_id
            )
            evidence = {
                "fold": fold_index,
                "validation_path": str(validation[case_id]),
                "combined_path": str(combined[case_id]),
                "validation_fingerprint": validation_fingerprint,
                "combined_fingerprint": combined_fingerprint,
            }
            prediction_files[case_id] = evidence
            case_evidence[case_id] = evidence
        fold_evidence.append(
            {
                "fold": fold_index,
                "validation_dir": str(validation_dir),
                "expected_case_ids": sorted(expected_fold_ids),
                "case_count": len(prediction_files),
                "prediction_files": prediction_files,
            }
        )

    if seen_validation_ids != expected_ids or len(seen_validation_ids) != EXPECTED_NUM_CASES:
        raise ValueError("five validation folds must union to exactly the fixed 95 unique case IDs")
    if len(result_roots) != 1:
        raise ValueError("all fold validation directories must use one Stage-1 result folder")
    result_root = next(iter(result_roots))
    if combined_root.parent != result_root:
        raise ValueError("combined OOF directory must come from the same Stage-1 result root")

    dataset_evidence = {
        "imagesTr": {
            "directory": str((raw_root / "imagesTr").resolve()),
            "case_ids": sorted(images),
            "files": {
                case_id: {"path": str(images[case_id]), "fingerprint": nifti_fingerprint(images[case_id])}
                for case_id in sorted(images)
            },
        },
        "labelsTr": {
            "directory": str((raw_root / "labelsTr").resolve()),
            "case_ids": sorted(labels),
            "files": {
                case_id: {"path": str(labels[case_id]), "fingerprint": nifti_fingerprint(labels[case_id])}
                for case_id in sorted(labels)
            },
        },
    }
    return {
        "verified": True,
        "stage1_trainer": STAGE1_TRAINER,
        "stage1_prediction_source": "complete_5_fold_oof",
        "roi_source": "stage1_prediction_only",
        "split_policy": "fixed_5_fold_patient_level",
        "num_folds": EXPECTED_NUM_FOLDS,
        "case_count": EXPECTED_NUM_CASES,
        "provenance_version": PROVENANCE_VERSION,
        "dataset501_raw": str(raw_root),
        "splits_path": str(splits_path.resolve()),
        "stage1_oof_dir": str(combined_root),
        "stage1_result_folder": str(result_root),
        "dataset501": dataset_evidence,
        "folds": fold_evidence,
        "cases": {case_id: case_evidence[case_id] for case_id in sorted(case_evidence)},
    }


def build_stage1_provenance(
    dataset501_raw: str | Path,
    splits_path: str | Path,
    stage1_oof_dir: str | Path,
    fold_validation_dirs: Mapping[int, str | Path],
    output: str | Path,
) -> Path:
    """Validate all Stage-1 inputs, then write a generated provenance artifact."""
    normalized_folds = {int(index): Path(path) for index, path in fold_validation_dirs.items()}
    payload = _collect_provenance(
        Path(dataset501_raw),
        Path(splits_path),
        Path(stage1_oof_dir),
        normalized_folds,
    )
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"provenance output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return destination


def _load_payload(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        payload = dict(source)
    else:
        path = Path(source)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise FileNotFoundError(f"Stage1 provenance file does not exist: {path}") from error
        except json.JSONDecodeError as error:
            raise ValueError(f"Stage1 provenance is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("Stage1 provenance must be a JSON object")
    return payload


def validate_stage1_provenance(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Re-read every referenced input and reject static or tampered artifacts."""
    payload = _load_payload(source)
    required_evidence = {
        "provenance_version",
        "dataset501_raw",
        "splits_path",
        "stage1_oof_dir",
        "stage1_result_folder",
        "dataset501",
        "folds",
        "cases",
    }
    if not required_evidence.issubset(payload):
        raise ValueError("Stage1 provenance evidence is missing")
    if payload.get("verified") is not True:
        raise ValueError("Stage1 provenance must have generated verified=true")
    try:
        fold_records = payload["folds"]
        if not isinstance(fold_records, list) or len(fold_records) != EXPECTED_NUM_FOLDS:
            raise ValueError
        fold_dirs = {int(record["fold"]): Path(record["validation_dir"]) for record in fold_records}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Stage1 provenance evidence is malformed") from error
    fresh = _collect_provenance(
        Path(payload["dataset501_raw"]),
        Path(payload["splits_path"]),
        Path(payload["stage1_oof_dir"]),
        fold_dirs,
    )
    if payload != fresh:
        raise ValueError("Stage1 provenance evidence does not match current inputs")
    return fresh
