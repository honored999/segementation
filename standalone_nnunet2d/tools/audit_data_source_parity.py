"""Audit raw online preprocessing against root-level nnU-Net 2D b2nd cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from standalone_nnunet2d.data.data_source import PreprocessedB2ndCaseSource
from standalone_nnunet2d.data.dataset import StrokeSliceDataset, load_fold_cases


IMAGE_PARITY_RTOL = 1e-5
IMAGE_PARITY_ATOL = 1e-6


def _split_for_case(fold: int, case_id: str) -> str:
    memberships = [
        split for split in ("train", "val") if case_id in load_fold_cases(fold, split)
    ]
    if len(memberships) != 1:
        raise ValueError(f"case {case_id!r} is not in exactly one split of fold {fold}")
    return memberships[0]


def _array_stats(image: np.ndarray, label: np.ndarray) -> dict[str, object]:
    if image.ndim != 3 or label.ndim != 3 or image.shape != label.shape:
        raise ValueError(
            "raw/preprocessed case arrays must be matched 3D, "
            f"got {image.shape} and {label.shape}"
        )
    if any(size <= 0 for size in image.shape):
        raise ValueError(f"case arrays must be non-empty, got {image.shape}")
    if not np.isfinite(image).all():
        raise ValueError("image contains non-finite values")
    unique = np.unique(label)
    if not np.isin(unique, (0, 1)).all():
        raise ValueError(f"labels contain invalid values: {unique.tolist()}")
    return {
        "shape": list(image.shape),
        "image_finite": True,
        "image_mean": float(image.mean()),
        "image_std": float(image.std()),
        "label_unique": [int(value) for value in unique.tolist()],
        "foreground_voxel_count": int(np.count_nonzero(label)),
        "foreground_voxel_count_by_z": [
            int(np.count_nonzero(label[z])) for z in range(label.shape[0])
        ],
    }


def _image_difference(raw: np.ndarray, preprocessed: np.ndarray) -> dict[str, float]:
    difference = np.asarray(preprocessed, dtype=np.float64) - np.asarray(raw, dtype=np.float64)
    absolute = np.abs(difference)
    return {
        "mean_abs": float(absolute.mean()),
        "max_abs": float(absolute.max()),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
    }


def audit_data_source_parity(
    *,
    raw_root: Path,
    preprocessed_root: Path,
    fold: int,
    case_ids: Sequence[str],
    output: Path,
) -> dict[str, object]:
    if not 3 <= len(case_ids) <= 5:
        raise ValueError("provide 3 to 5 --case-id values")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("case IDs must be unique")
    raw_root = Path(raw_root).expanduser().resolve()
    preprocessed_root = Path(preprocessed_root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if any(output == root or root in output.parents for root in (raw_root, preprocessed_root)):
        raise ValueError("output must be outside both input roots")
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw root does not exist: {raw_root}")
    if not preprocessed_root.is_dir():
        raise FileNotFoundError(f"preprocessed root does not exist: {preprocessed_root}")

    records: list[dict[str, object]] = []
    errors: list[str] = []
    for case_id in case_ids:
        split = _split_for_case(fold, case_id)
        raw_dataset = StrokeSliceDataset(
            raw_root, fold=fold, split=split, case_ids=(case_id,)
        )
        raw_image, raw_label = raw_dataset.load_case(case_id)
        preprocessed_case = PreprocessedB2ndCaseSource(
            preprocessed_root, fold=fold, split=split, case_ids=(case_id,)
        ).prepare_case(case_id)
        preprocessed_image = np.stack(
            [preprocessed_case.image_slice(z) for z in range(preprocessed_case.shape[0])],
            axis=0,
        )
        preprocessed_label = preprocessed_case.label
        raw_stats = _array_stats(raw_image, raw_label)
        preprocessed_stats = _array_stats(preprocessed_image, preprocessed_label)
        if raw_image.shape != preprocessed_image.shape:
            errors.append(
                f"{case_id}: shape mismatch {raw_image.shape} != {preprocessed_image.shape}"
            )
        if raw_label.shape != preprocessed_label.shape:
            errors.append(
                f"{case_id}: label shape mismatch {raw_label.shape} != {preprocessed_label.shape}"
            )
        label_equal = bool(np.array_equal(raw_label, preprocessed_label))
        z_counts_equal = raw_stats["foreground_voxel_count_by_z"] == preprocessed_stats["foreground_voxel_count_by_z"]
        image_shape_match = raw_image.shape == preprocessed_image.shape
        image_close = bool(
            image_shape_match
            and np.allclose(
                raw_image,
                preprocessed_image,
                rtol=IMAGE_PARITY_RTOL,
                atol=IMAGE_PARITY_ATOL,
            )
        )
        image_difference = (
            _image_difference(raw_image, preprocessed_image)
            if image_shape_match
            else None
        )
        if not image_close:
            errors.append(
                f"{case_id}: image value/axis correspondence exceeds tolerance"
            )
        if not label_equal or not z_counts_equal:
            errors.append(f"{case_id}: label/z-slice correspondence mismatch")
        records.append(
            {
                "case_id": case_id,
                "split": split,
                "raw": raw_stats,
                "preprocessed": preprocessed_stats,
                "shape_match": raw_image.shape == preprocessed_image.shape and raw_label.shape == preprocessed_label.shape,
                "image_close": image_close,
                "label_equal": label_equal,
                "z_foreground_counts_equal": z_counts_equal,
                "image_difference": image_difference,
            }
        )

    report: dict[str, object] = {
        "audit": "raw_nifti_online_vs_nnunet_preprocessed_b2nd",
        "synthetic_or_real": "unknown_input_provenance",
        "fold": fold,
        "raw_root": str(raw_root),
        "preprocessed_root": str(preprocessed_root),
        "image_tolerance": {
            "rtol": IMAGE_PARITY_RTOL,
            "atol": IMAGE_PARITY_ATOL,
        },
        "cases": records,
        "passed": not errors,
        "errors": errors,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if errors:
        raise ValueError("parity audit failed: " + "; ".join(errors))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--preprocessed-root", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--case-id", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        audit_data_source_parity(
            raw_root=arguments.raw_root,
            preprocessed_root=arguments.preprocessed_root,
            fold=arguments.fold,
            case_ids=arguments.case_id,
            output=arguments.output,
        )
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {"audit": "completed", "output": str(arguments.output.resolve())},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
