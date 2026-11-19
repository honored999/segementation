from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from standalone_nnunet2d.data.nifti_io import NiftiVolume, write_nifti
from standalone_nnunet2d.tools.audit_dataset504_roi_coverage import audit_roi_coverage


_BASIC_FIELDS = {
    "case_id",
    "original_shape_z",
    "original_shape_y",
    "original_shape_x",
    "cropped_shape_z",
    "cropped_shape_y",
    "cropped_shape_x",
    "roi_width",
    "roi_height",
    "roi_area",
    "original_gt_voxels",
    "cropped_gt_voxels",
    "gt_coverage",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_pair(
    root: Path,
    case_id: str,
    shape: tuple[int, int, int],
    label: np.ndarray,
) -> tuple[Path, Path]:
    image_path = root / "imagesTr" / f"{case_id}_0000.nii.gz"
    label_path = root / "labelsTr" / f"{case_id}.nii.gz"
    metadata = {"spacing_xyz": (1.0, 1.0, 2.0), "origin_xyz": (0.0, 0.0, 0.0)}
    image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    write_nifti(image_path, NiftiVolume(array=image, **metadata))
    write_nifti(label_path, NiftiVolume(array=label.astype(np.uint8), **metadata))
    return image_path, label_path


def _label(shape: tuple[int, int, int], coordinates: tuple[tuple[int, int, int], ...]) -> np.ndarray:
    result = np.zeros(shape, dtype=np.uint8)
    for coordinate in coordinates:
        result[coordinate] = 1
    return result


def test_audit_roi_coverage_writes_exact_synthetic_summary_and_preserves_inputs(
    tmp_path: Path,
) -> None:
    dataset501_root = tmp_path / "Dataset501_StrokeLesion"
    dataset504_root = tmp_path / "Dataset504_StrokeLesion_CoarseToFine"
    output_dir = tmp_path / "audit_output"
    shape501 = (5, 100, 120)

    cases = {
        "case_full": {
            "shape504": (5, 100, 120),
            "label501": _label(shape501, ((1, 10, 10), (1, 10, 11), (2, 20, 20), (2, 20, 21))),
            "label504": _label((5, 100, 120), ((1, 10, 10), (1, 10, 11), (2, 20, 20), (2, 20, 21))),
            "roi": [0, 0, 120, 100],
            "fallback": False,
        },
        "case_partial": {
            "shape504": (5, 40, 50),
            "label501": _label(shape501, ((1, 10, 10), (1, 10, 11), (2, 20, 20), (2, 20, 21))),
            "label504": _label((5, 40, 50), ((1, 10, 10),)),
            "roi": [10, 20, 60, 60],
            "fallback": False,
        },
        "case_cropped_empty": {
            "shape504": (5, 3, 3),
            "label501": _label(shape501, ((1, 10, 10), (1, 10, 11), (2, 20, 20))),
            "label504": _label((5, 3, 3), ()),
            "roi": [224, 305, 227, 308],
            "fallback": False,
        },
        "case_original_empty": {
            "shape504": (5, 64, 64),
            "label501": _label(shape501, ()),
            "label504": _label((5, 64, 64), ()),
            "roi": [0, 0, 64, 64],
            "fallback": True,
        },
        "case_3x3": {
            "shape504": (5, 3, 3),
            "label501": _label(shape501, ((3, 40, 40),)),
            "label504": _label((5, 3, 3), ((3, 1, 1),)),
            "roi": [40, 40, 43, 43],
            "fallback": False,
        },
    }

    input_paths: list[Path] = []
    manifest_cases: dict[str, dict[str, object]] = {}
    for case_id, values in cases.items():
        input_paths.extend(
            _write_pair(dataset501_root, case_id, shape501, values["label501"])
        )
        input_paths.extend(
            _write_pair(dataset504_root, case_id, values["shape504"], values["label504"])
        )
        manifest_cases[case_id] = {
            "fold": 0,
            "split": "val",
            "roi": values["roi"],
            "fallback": values["fallback"],
            "source_image": f"imagesTr/{case_id}_0000.nii.gz",
            "source_label": f"labelsTr/{case_id}.nii.gz",
            "stage1_prediction": f"{case_id}.nii.gz",
        }

    (dataset504_root / "roi_manifest.json").write_text(
        json.dumps(
            {
                "dataset_name": "Dataset504_StrokeLesion_CoarseToFine",
                "source_dataset": "Dataset501_StrokeLesion",
                "case_ids": list(cases),
                "cases": manifest_cases,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    before_hashes = {path: _sha256(path) for path in input_paths}

    audit_roi_coverage(dataset501_root, dataset504_root, output_dir)

    csv_path = output_dir / "roi_coverage_cases.csv"
    summary_path = output_dir / "roi_coverage_summary.json"
    assert csv_path.is_file() and csv_path.stat().st_size > 0
    assert summary_path.is_file() and summary_path.stat().st_size > 0
    assert {path: _sha256(path) for path in input_paths} == before_hashes

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = {row["case_id"]: row for row in csv.DictReader(handle)}
    assert set(rows) == set(cases)
    assert _BASIC_FIELDS <= set(rows["case_full"])
    assert {rows["case_full"]["gt_coverage"], rows["case_3x3"]["gt_coverage"]} == {"1.0"}
    assert float(rows["case_partial"]["gt_coverage"]) == 0.25
    assert rows["case_original_empty"]["gt_coverage"] in {"", "nan", "NaN", "null", "None"}
    assert {"roi", "fallback"} <= set(rows["case_full"])
    assert rows["case_3x3"]["roi"] == "[40, 40, 43, 43]"
    assert rows["case_original_empty"]["fallback"].lower() == "true"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["total_cases"] == 5
    assert summary["cases_with_gt"] == 4
    assert summary["cropped_gt_empty_count"] == 2
    assert summary["cropped_gt_empty_case_ids"] == ["case_cropped_empty", "case_original_empty"]
    assert summary["gt_coverage_lt_0_5_count"] == 2
    assert summary["gt_coverage_lt_0_5_case_ids"] == ["case_cropped_empty", "case_partial"]
    assert summary["gt_coverage_lt_0_9_count"] == 2
    assert summary["gt_coverage_lt_0_9_case_ids"] == ["case_cropped_empty", "case_partial"]
    assert summary["gt_coverage_lt_1_0_count"] == 2
    assert summary["gt_coverage_lt_1_0_case_ids"] == ["case_cropped_empty", "case_partial"]
    assert summary["roi_width_min"] == 3
    assert summary["roi_width_median"] == 50
    assert summary["roi_width_max"] == 120
    assert summary["roi_height_min"] == 3
    assert summary["roi_height_median"] == 40
    assert summary["roi_height_max"] == 100
    assert summary["roi_area_min"] == 9
    assert summary["roi_area_median"] == 2000
    assert summary["roi_area_max"] == 12000
    assert summary["roi_smaller_than_32x32_count"] == 2
    assert summary["roi_smaller_than_64x64_count"] == 3
    assert summary["roi_smaller_than_128x128_count"] == 5
    if "formal_performance_claim" in summary:
        assert summary["formal_performance_claim"] in (None, False, "")


def test_audit_roi_coverage_preserves_fractional_even_case_medians(
    tmp_path: Path,
) -> None:
    dataset501_root = tmp_path / "Dataset501_StrokeLesion"
    dataset504_root = tmp_path / "Dataset504_StrokeLesion_CoarseToFine"
    output_dir = tmp_path / "audit_output"
    shape501 = (2, 10, 10)
    cases = {
        "case_even_3": ((2, 5, 3), (1, 1, 1)),
        "case_even_4": ((2, 6, 4), (1, 2, 2)),
    }

    for case_id, (shape504, coordinate) in cases.items():
        _write_pair(dataset501_root, case_id, shape501, _label(shape501, ((0, 1, 1),)))
        _write_pair(
            dataset504_root,
            case_id,
            shape504,
            _label(shape504, (coordinate,)),
        )

    audit_roi_coverage(dataset501_root, dataset504_root, output_dir)
    summary = json.loads(
        (output_dir / "roi_coverage_summary.json").read_text(encoding="utf-8")
    )

    assert summary["roi_width_median"] == 3.5
    assert summary["roi_height_median"] == 5.5
    assert summary["roi_area_median"] == 19.5
