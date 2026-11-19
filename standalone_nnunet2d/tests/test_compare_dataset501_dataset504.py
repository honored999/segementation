from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from standalone_nnunet2d.data.nifti_io import NiftiVolume, write_nifti
import standalone_nnunet2d.tools.compare_dataset501_dataset504 as compare_module
from standalone_nnunet2d.tools.compare_dataset501_dataset504 import (
    compare_case,
    select_slice_index,
)


def _write_case(root: Path, image: np.ndarray, label: np.ndarray) -> tuple[Path, Path]:
    image_path = root / "imagesTr" / "case018_0000.nii.gz"
    label_path = root / "labelsTr" / "case018.nii.gz"
    metadata = {
        "spacing_xyz": (1.0, 1.0, 2.0),
        "origin_xyz": (0.0, 0.0, 0.0),
    }
    write_nifti(image_path, NiftiVolume(array=image, **metadata))
    write_nifti(label_path, NiftiVolume(array=label, **metadata))
    return image_path, label_path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_prediction(root: Path, prediction: np.ndarray) -> Path:
    prediction_path = root / "case018.nii.gz"
    metadata = {
        "spacing_xyz": (1.0, 1.0, 2.0),
        "origin_xyz": (0.0, 0.0, 0.0),
    }
    write_nifti(prediction_path, NiftiVolume(array=prediction, **metadata))
    return prediction_path


def _write_manifest(root: Path) -> Path:
    manifest_path = root / "roi_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "cases": {
                    "case018": {
                        "roi": [40, 20, 60, 40],
                        "fallback": False,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_compare_case_selects_largest_gt_slice_supports_override_and_preserves_inputs(
    tmp_path: Path,
) -> None:
    dataset501_root = tmp_path / "Dataset501_StrokeLesion"
    dataset504_root = tmp_path / "Dataset504_StrokeLesion_CoarseToFine"

    original_image = np.linspace(0.0, 1.0, 5 * 100 * 120, dtype=np.float32).reshape(5, 100, 120)
    original_label = np.zeros((5, 100, 120), dtype=np.uint8)
    original_label[1, 10:12, 10:13] = 1
    original_label[3, 30:37, 40:45] = 1
    original_label[3, 70:72, 80:84] = 1
    cropped_image = np.linspace(0.0, 1.0, 5 * 40 * 50, dtype=np.float32).reshape(5, 40, 50)
    cropped_label = np.zeros((5, 40, 50), dtype=np.uint8)
    cropped_label[3, 8:18, 12:22] = 1

    original_paths = _write_case(dataset501_root, original_image, original_label)
    cropped_paths = _write_case(dataset504_root, cropped_image, cropped_label)
    input_hashes = {path: _sha256(path) for path in (*original_paths, *cropped_paths)}

    assert select_slice_index(original_label) == 3
    assert select_slice_index(original_label, slice_index=1) == 1

    automatic_output = tmp_path / "comparison" / "case018_auto.png"
    assert (
        compare_case(
            dataset501_root=dataset501_root,
            dataset504_root=dataset504_root,
            case_id="case018",
            output=automatic_output,
        )
        == 3
    )
    assert automatic_output.is_file()
    assert automatic_output.stat().st_size > 0

    override_output = tmp_path / "comparison" / "case018_override.png"
    assert (
        compare_case(
            dataset501_root=dataset501_root,
            dataset504_root=dataset504_root,
            case_id="case018",
            output=override_output,
            slice_index=1,
        )
        == 1
    )
    assert override_output.is_file()
    assert override_output.stat().st_size > 0

    assert {path: _sha256(path) for path in input_hashes} == input_hashes


def test_enhanced_comparison_uses_matching_stage1_and_manifest_without_modifying_inputs(
    tmp_path: Path,
) -> None:
    dataset501_root = tmp_path / "Dataset501_StrokeLesion"
    dataset504_root = tmp_path / "Dataset504_StrokeLesion_CoarseToFine"
    stage1_prediction_dir = tmp_path / "stage1_oof"

    original_image = np.linspace(
        0.0, 1.0, 5 * 100 * 120, dtype=np.float32
    ).reshape(5, 100, 120)
    original_label = np.zeros((5, 100, 120), dtype=np.uint8)
    original_label[1, 10:12, 10:13] = 1
    original_label[3, 30:37, 40:45] = 1
    cropped_image = np.linspace(
        0.0, 1.0, 5 * 40 * 50, dtype=np.float32
    ).reshape(5, 40, 50)
    cropped_label = np.zeros((5, 40, 50), dtype=np.uint8)
    cropped_label[3, 8:18, 12:22] = 1
    stage1_prediction = np.zeros((5, 100, 120), dtype=np.uint8)
    stage1_prediction[0, 10:14, 10:15] = 1

    original_paths = _write_case(dataset501_root, original_image, original_label)
    cropped_paths = _write_case(dataset504_root, cropped_image, cropped_label)
    stage1_path = _write_prediction(stage1_prediction_dir, stage1_prediction)
    manifest_path = _write_manifest(tmp_path)
    input_paths = (*original_paths, *cropped_paths, stage1_path, manifest_path)
    input_hashes = {path: _sha256(path) for path in input_paths}

    output = tmp_path / "comparison" / "case018_enhanced.png"
    expected_z = select_slice_index(original_label)
    selected_z = compare_case(
        dataset501_root=dataset501_root,
        dataset504_root=dataset504_root,
        case_id="case018",
        output=output,
        stage1_prediction_dir=stage1_prediction_dir,
        roi_manifest=manifest_path,
    )

    assert selected_z == expected_z
    assert output.is_file()
    assert output.stat().st_size > 0
    assert {path: _sha256(path) for path in input_paths} == input_hashes


def test_enhanced_comparison_rejects_stage1_shape_mismatch(tmp_path: Path) -> None:
    dataset501_root = tmp_path / "Dataset501_StrokeLesion"
    dataset504_root = tmp_path / "Dataset504_StrokeLesion_CoarseToFine"
    stage1_prediction_dir = tmp_path / "stage1_oof"

    original_image = np.zeros((5, 100, 120), dtype=np.float32)
    original_label = np.zeros((5, 100, 120), dtype=np.uint8)
    original_label[2, 20:24, 30:34] = 1
    cropped_image = np.zeros((5, 40, 50), dtype=np.float32)
    cropped_label = np.zeros((5, 40, 50), dtype=np.uint8)
    _write_case(dataset501_root, original_image, original_label)
    _write_case(dataset504_root, cropped_image, cropped_label)
    _write_prediction(stage1_prediction_dir, np.zeros((4, 100, 120), dtype=np.uint8))
    manifest_path = _write_manifest(tmp_path)

    with pytest.raises(ValueError, match="Stage 1 prediction.*shape mismatch"):
        compare_case(
            dataset501_root=dataset501_root,
            dataset504_root=dataset504_root,
            case_id="case018",
            output=tmp_path / "comparison" / "case018_mismatch.png",
            stage1_prediction_dir=stage1_prediction_dir,
            roi_manifest=manifest_path,
        )


def test_roi_to_rectangle_uses_xy_matplotlib_order() -> None:
    assert compare_module.roi_to_rectangle([40, 20, 60, 40]) == (40, 20, 20, 20)
