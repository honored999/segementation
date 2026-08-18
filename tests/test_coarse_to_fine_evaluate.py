import csv
import json
from pathlib import Path

import numpy as np
import pytest

from coarse_to_fine_dwi.nifti import NiftiVolume, crop_xy


IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def _volume(array: np.ndarray, case_index: int = 0) -> NiftiVolume:
    return NiftiVolume(
        array=array,
        spacing_xyz=(0.7 + case_index, 0.8, 2.0),
        origin_xyz=(10.0 + case_index, 20.0, 30.0),
        direction=IDENTITY,
    )


def _write(volume: NiftiVolume, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    volume.write(path)


def test_binary_case_metrics_cover_overlap_and_empty_mask_semantics():
    from coarse_to_fine_dwi.evaluate import binary_case_metrics

    ground_truth = np.zeros((2, 3, 4), dtype=np.uint8)
    prediction = np.zeros_like(ground_truth)
    ground_truth[0, 0, 0] = 1
    ground_truth[0, 0, 1] = 1
    prediction[0, 0, 0] = 1
    prediction[0, 1, 1] = 1

    metrics = binary_case_metrics(ground_truth, prediction)
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["dice"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)

    both_empty = binary_case_metrics(np.zeros_like(ground_truth), np.zeros_like(ground_truth))
    assert both_empty["dice"] == 1.0
    assert both_empty["iou"] == 1.0
    assert both_empty["precision"] == 1.0
    assert both_empty["recall"] == 1.0

    gt_only = binary_case_metrics(ground_truth, np.zeros_like(ground_truth))
    pred_only = binary_case_metrics(np.zeros_like(ground_truth), ground_truth)
    for result in (gt_only, pred_only):
        assert result["dice"] == 0.0
        assert result["iou"] == 0.0
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0


def test_binary_case_metrics_rejects_shape_and_non_binary_masks():
    from coarse_to_fine_dwi.evaluate import binary_case_metrics

    with pytest.raises(ValueError, match="shape"):
        binary_case_metrics(np.zeros((2, 3, 4), dtype=np.uint8), np.zeros((1, 3, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="binary"):
        binary_case_metrics(np.zeros((2, 3, 4), dtype=np.uint8), np.full((2, 3, 4), 2, dtype=np.uint8))


def _make_synthetic_case_data(tmp_path: Path) -> tuple[Path, Path, Path, Path, dict[str, list[int]]]:
    raw_root = tmp_path / "dataset501"
    labels_dir = raw_root / "labelsTr"
    stage1_dir = tmp_path / "stage1_oof"
    stage2_cropped_dir = tmp_path / "stage2_cropped"
    for directory in (labels_dir, stage1_dir, stage2_cropped_dir):
        directory.mkdir(parents=True)

    cases = {
        "caseA": [1, 1, 4, 4],
        "caseB": [0, 0, 5, 4],
    }
    for case_index, (case_id, bbox) in enumerate(cases.items()):
        image = _volume(np.full((2, 4, 5), case_index + 1, dtype=np.float32), case_index)
        label_array = np.zeros((2, 4, 5), dtype=np.uint8)
        stage1_array = np.zeros_like(label_array)
        stage2_array = np.zeros_like(label_array)
        if case_id == "caseA":
            label_array[:, 1, 1] = 1
            stage1_array[:, 1, 1] = 1
            stage2_array[:, 1, 1] = 1
            stage2_array[:, 2, 2] = 1
        label = NiftiVolume(label_array, image.spacing_xyz, image.origin_xyz, image.direction)
        stage1 = NiftiVolume(stage1_array, image.spacing_xyz, image.origin_xyz, image.direction)
        stage2 = NiftiVolume(stage2_array, image.spacing_xyz, image.origin_xyz, image.direction)
        _write(image, raw_root / "imagesTr" / f"{case_id}_0000.nii.gz")
        _write(label, labels_dir / f"{case_id}.nii.gz")
        _write(stage1, stage1_dir / f"{case_id}.nii.gz")
        _write(crop_xy(stage2, tuple(bbox)), stage2_cropped_dir / f"{case_id}.nii.gz")

    manifest = {
        "dataset_name": "Dataset504_StrokeLesion_CoarseToFine",
        "stage1_prediction_source": "complete_5_fold_oof",
        "roi_source": "stage1_prediction_only",
        "num_cases": len(cases),
        "cases": {
            case_id: {
                "roi": bbox,
                "source_image": f"imagesTr/{case_id}_0000.nii.gz",
                "source_label": f"labelsTr/{case_id}.nii.gz",
            }
            for case_id, bbox in cases.items()
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return raw_root, labels_dir, stage1_dir, stage2_cropped_dir, cases | {"__manifest__": [str(manifest_path)]}


def test_restore_predictions_uses_manifest_and_restores_original_shape_metadata(tmp_path):
    from coarse_to_fine_dwi.cli.restore_predictions import restore_predictions

    raw_root, _, _, cropped_dir, case_info = _make_synthetic_case_data(tmp_path)
    manifest_path = Path(case_info["__manifest__"][0])
    restored_dir = tmp_path / "stage2_restored"

    result = restore_predictions(
        manifest=manifest_path,
        cropped_predictions=cropped_dir,
        dataset501_raw=raw_root,
        output_dir=restored_dir,
    )

    assert sorted(path.name for path in result.glob("*.nii.gz")) == ["caseA.nii.gz", "caseB.nii.gz"]
    restored = NiftiVolume.read(result / "caseA.nii.gz")
    reference = NiftiVolume.read(raw_root / "imagesTr" / "caseA_0000.nii.gz")
    assert restored.shape_zyx == reference.shape_zyx == (2, 4, 5)
    assert restored.spacing_xyz == reference.spacing_xyz
    assert restored.origin_xyz == reference.origin_xyz
    assert restored.direction == reference.direction
    assert restored.array[0, 1, 1] == 1
    assert restored.array[0, 2, 2] == 1


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_restore_predictions_requires_exact_cropped_prediction_ids(tmp_path, mutation):
    from coarse_to_fine_dwi.cli.restore_predictions import restore_predictions

    raw_root, _, _, cropped_dir, case_info = _make_synthetic_case_data(tmp_path)
    if mutation == "missing":
        (cropped_dir / "caseB.nii.gz").unlink()
    else:
        _write(NiftiVolume(np.zeros((2, 4, 5), dtype=np.uint8), (1.7, 0.8, 2.0), (11.0, 20.0, 30.0), IDENTITY), cropped_dir / "extra.nii.gz")
    with pytest.raises(ValueError, match="missing|extra|exact"):
        restore_predictions(
            manifest=Path(case_info["__manifest__"][0]),
            cropped_predictions=cropped_dir,
            dataset501_raw=raw_root,
            output_dir=tmp_path / "restored",
        )


def test_compare_predictions_writes_full_volume_csv_and_json(tmp_path):
    from coarse_to_fine_dwi.cli.restore_predictions import restore_predictions
    from coarse_to_fine_dwi.evaluate import compare_full_volume_predictions

    raw_root, labels_dir, stage1_dir, cropped_dir, case_info = _make_synthetic_case_data(tmp_path)
    restored_dir = restore_predictions(
        manifest=Path(case_info["__manifest__"][0]),
        cropped_predictions=cropped_dir,
        dataset501_raw=raw_root,
        output_dir=tmp_path / "restored",
    )
    csv_path, json_path = compare_full_volume_predictions(
        labels_dir=labels_dir,
        stage1_dir=stage1_dir,
        stage2_restored_dir=restored_dir,
        output_dir=tmp_path / "evaluation",
        expected_case_count=2,
    )

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["case_id"] for row in rows] == ["caseA", "caseB"]
    assert rows[0]["stage2_dice"] == "0.6666666666666666"
    assert rows[0]["dice_delta"] == "-0.33333333333333337"

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    assert summary["protocol"] == {
        "space": "original_full_volume",
        "gt_source": "Dataset501_labelsTr",
        "case_aggregation": "equal_case_macro",
    }
    assert summary["case_count"] == 2
    assert summary["case_ids"] == ["caseA", "caseB"]
    assert summary["formal_eligible"] is False
    assert summary["stage2_minus_stage1"]["dice"] == pytest.approx(-1 / 6)
    assert summary["global"]["stage1"]["tp"] == 2
    assert summary["global"]["stage2"]["fp"] == 2


def test_compare_predictions_rejects_cropped_stage1_prediction(tmp_path):
    from coarse_to_fine_dwi.evaluate import compare_full_volume_predictions

    raw_root, labels_dir, stage1_dir, _, _ = _make_synthetic_case_data(tmp_path)
    cropped_stage1 = tmp_path / "cropped_stage1"
    cropped_stage1.mkdir()
    _write(NiftiVolume(np.zeros((2, 3, 3), dtype=np.uint8), (0.7, 0.8, 2.0), (10.7, 20.8, 30.0), IDENTITY), cropped_stage1 / "caseA.nii.gz")
    _write(NiftiVolume(np.zeros((2, 4, 5), dtype=np.uint8), (1.7, 0.8, 2.0), (11.0, 20.0, 30.0), IDENTITY), cropped_stage1 / "caseB.nii.gz")
    with pytest.raises(ValueError, match="shape|metadata|compatible"):
        compare_full_volume_predictions(
            labels_dir=labels_dir,
            stage1_dir=cropped_stage1,
            stage2_restored_dir=stage1_dir,
            output_dir=tmp_path / "evaluation",
            expected_case_count=2,
        )


@pytest.mark.parametrize("overlap_dir", ["labels", "stage1", "stage2"])
def test_compare_predictions_rejects_output_dir_overlapping_input(tmp_path, overlap_dir):
    from coarse_to_fine_dwi.evaluate import compare_full_volume_predictions

    _, labels_dir, stage1_dir, _, case_info = _make_synthetic_case_data(tmp_path)
    restored_dir = tmp_path / "restored"
    from coarse_to_fine_dwi.cli.restore_predictions import restore_predictions

    restored_dir = restore_predictions(
        manifest=Path(case_info["__manifest__"][0]),
        cropped_predictions=tmp_path / "stage2_cropped",
        dataset501_raw=tmp_path / "dataset501",
        output_dir=restored_dir,
    )
    input_dirs = {
        "labels": labels_dir,
        "stage1": stage1_dir,
        "stage2": restored_dir,
    }
    with pytest.raises(ValueError, match="output_dir.*overlap"):
        compare_full_volume_predictions(
            labels_dir=labels_dir,
            stage1_dir=stage1_dir,
            stage2_restored_dir=restored_dir,
            output_dir=input_dirs[overlap_dir],
            expected_case_count=2,
        )


def test_compare_predictions_rejects_existing_nonempty_output_dir(tmp_path):
    from coarse_to_fine_dwi.cli.restore_predictions import restore_predictions
    from coarse_to_fine_dwi.evaluate import compare_full_volume_predictions

    _, labels_dir, stage1_dir, cropped_dir, case_info = _make_synthetic_case_data(tmp_path)
    restored_dir = restore_predictions(
        manifest=Path(case_info["__manifest__"][0]),
        cropped_predictions=cropped_dir,
        dataset501_raw=tmp_path / "dataset501",
        output_dir=tmp_path / "restored",
    )
    output_dir = tmp_path / "evaluation"
    output_dir.mkdir()
    (output_dir / "sentinel.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        compare_full_volume_predictions(
            labels_dir=labels_dir,
            stage1_dir=stage1_dir,
            stage2_restored_dir=restored_dir,
            output_dir=output_dir,
            expected_case_count=2,
        )
