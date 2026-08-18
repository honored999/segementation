"""Restore cropped Stage 2 predictions using the Dataset504 manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..nifti import NiftiVolume, crop_xy, restore_xy
from ..roi import validate_binary_prediction


def _case_id(path: Path) -> str | None:
    if path.name.endswith(".nii.gz"):
        return path.name[:-7]
    if path.name.endswith(".nii"):
        return path.name[:-4]
    return None


def _discover(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"prediction directory does not exist: {directory}")
    found: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        case_id = _case_id(path)
        if case_id is None:
            continue
        if case_id in found:
            raise ValueError(f"cropped predictions contain duplicate case ID: {case_id}")
        found[case_id] = path
    return found


def _manifest_rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
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
    if not result:
        raise ValueError("manifest contains no cases")
    declared = payload.get("case_ids")
    if declared is not None and sorted(declared) != sorted(result):
        raise ValueError("manifest case_ids do not match manifest cases")
    return result


def _validate_protocol(payload: dict[str, Any]) -> None:
    protocol = payload.get("protocol")
    if isinstance(protocol, dict):
        if protocol.get("roi_source") != "stage1_oof_prediction":
            raise ValueError("manifest protocol must use stage1_oof_prediction ROIs")
        if protocol.get("gt_used_for_roi") is not False:
            raise ValueError("manifest protocol must state gt_used_for_roi=false")
        return
    if payload.get("stage1_prediction_source") != "complete_5_fold_oof":
        raise ValueError("manifest must declare complete_5_fold_oof Stage1 predictions")
    if payload.get("roi_source") not in {"stage1_prediction_only", "stage1_oof_prediction"}:
        raise ValueError("manifest must declare prediction-only ROIs")


def _raw_image_path(raw_root: Path, case_id: str) -> Path:
    candidates = [
        raw_root / "imagesTr" / f"{case_id}_0000.nii.gz",
        raw_root / "imagesTr" / f"{case_id}_0000.nii",
    ]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise ValueError(f"raw Dataset501 must contain exactly one DWI image for {case_id}")
    return existing[0]


def _check_optional_metadata(row: dict[str, Any], reference: NiftiVolume, cropped: NiftiVolume) -> None:
    if "source_shape_zyx" in row and tuple(row["source_shape_zyx"]) != reference.shape_zyx:
        raise ValueError(f"manifest source shape mismatch for {row['case_id']}")
    if "cropped_shape_zyx" in row and tuple(row["cropped_shape_zyx"]) != cropped.shape_zyx:
        raise ValueError(f"manifest cropped shape mismatch for {row['case_id']}")
    for field, actual in (
        ("spacing_xyz", cropped.spacing_xyz),
        ("origin_xyz", cropped.origin_xyz),
        ("direction", cropped.direction),
    ):
        if field in row and not np.allclose(tuple(row[field]), actual, atol=1e-6, rtol=0.0):
            raise ValueError(f"manifest {field} mismatch for {row['case_id']}")


def restore_predictions(
    *,
    manifest: Path,
    cropped_predictions: Path,
    dataset501_raw: Path,
    output_dir: Path,
) -> Path:
    """Restore one exact cropped prediction per manifest case to original space."""
    manifest_path = Path(manifest).resolve()
    raw_root = Path(dataset501_raw).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest must be a JSON object")
    _validate_protocol(payload)
    rows = _manifest_rows(payload)
    predictions = _discover(Path(cropped_predictions).resolve())
    expected_ids = set(rows)
    if set(predictions) != expected_ids:
        raise ValueError(
            f"cropped predictions IDs mismatch: missing={sorted(expected_ids - set(predictions))}, "
            f"extra={sorted(set(predictions) - expected_ids)}"
        )

    destination = Path(output_dir).resolve()
    if destination == raw_root or raw_root in destination.parents:
        raise ValueError("restored output must be outside the raw Dataset501 root")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"restored output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    for case_id in sorted(expected_ids):
        row = rows[case_id]
        bbox_value = row.get("bbox_xy", row.get("roi"))
        if bbox_value is None:
            raise ValueError(f"manifest case {case_id} has no bbox_xy/roi")
        bbox = tuple(bbox_value)
        reference = NiftiVolume.read(_raw_image_path(raw_root, case_id))
        expected_cropped = crop_xy(reference, bbox)
        prediction = NiftiVolume.read(predictions[case_id])
        prediction = NiftiVolume(
            array=validate_binary_prediction(prediction.array),
            spacing_xyz=prediction.spacing_xyz,
            origin_xyz=prediction.origin_xyz,
            direction=prediction.direction,
        )
        _check_optional_metadata(row, reference, expected_cropped)
        restored = restore_xy(prediction, reference, bbox)
        if restored.shape_zyx != reference.shape_zyx:
            raise ValueError(f"restored shape mismatch for {case_id}")
        restored.write(destination / f"{case_id}.nii.gz")
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cropped-predictions", type=Path, required=True)
    parser.add_argument("--dataset501-raw", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        destination = restore_predictions(
            manifest=args.manifest,
            cropped_predictions=args.cropped_predictions,
            dataset501_raw=args.dataset501_raw,
            output_dir=args.output_dir,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"restore failed: {error}")
        return 2
    print(f"restored_predictions={destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
