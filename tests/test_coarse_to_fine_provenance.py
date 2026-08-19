import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from coarse_to_fine_dwi.nifti import NiftiVolume


REFERENCE_SPLITS = (
    Path(__file__).parents[1]
    / "standalone_nnunet2d"
    / "reference"
    / "splits_final.json"
)


def _volume(value: int) -> NiftiVolume:
    return NiftiVolume(
        array=np.full((1, 2, 2), value, dtype=np.uint8),
        spacing_xyz=(1.5, 1.0, 2.0),
        origin_xyz=(10.0, 20.0, 30.0),
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )


def _make_inputs(tmp_path: Path) -> tuple[Path, Path, dict[int, Path]]:
    pytest.importorskip("SimpleITK")
    splits = json.loads(REFERENCE_SPLITS.read_text(encoding="utf-8"))
    case_ids = sorted({case_id for fold in splits for case_id in fold["val"]})
    raw = tmp_path / "Dataset501"
    images = raw / "imagesTr"
    labels = raw / "labelsTr"
    result_root = tmp_path / "nnUNetTrainer__nnUNetPlans__2d"
    combined = result_root / "crossval_results_folds_0_1_2_3_4"
    images.mkdir(parents=True)
    labels.mkdir()
    combined.mkdir(parents=True)
    fold_dirs = {}
    case_values = {case_id: index for index, case_id in enumerate(case_ids)}
    for fold_index, fold in enumerate(splits):
        fold_dir = result_root / f"fold_{fold_index}" / "validation"
        fold_dir.mkdir(parents=True)
        fold_dirs[fold_index] = fold_dir
        for case_id in fold["val"]:
            _volume(case_values[case_id]).write(fold_dir / f"{case_id}.nii.gz")
    for case_id in case_ids:
        _volume(case_values[case_id]).write(combined / f"{case_id}.nii.gz")
        _volume(case_values[case_id]).write(images / f"{case_id}_0000.nii.gz")
        _volume(case_values[case_id]).write(labels / f"{case_id}.nii.gz")
    return raw, combined, fold_dirs


def _build_args(raw: Path, combined: Path, folds: dict[int, Path], output: Path) -> list[str]:
    args = [
        "--dataset501-raw",
        str(raw),
        "--splits",
        str(REFERENCE_SPLITS),
        "--stage1-oof-dir",
        str(combined),
    ]
    for fold_index in range(5):
        args.extend([f"--fold-{fold_index}-validation", str(folds[fold_index])])
    args.extend(["--output", str(output)])
    return args


def test_build_cli_generates_verified_provenance_from_complete_five_fold_oof(tmp_path):
    from coarse_to_fine_dwi.cli import build_stage1_provenance

    raw, combined, folds = _make_inputs(tmp_path)
    output = tmp_path / "stage1_provenance.json"

    assert build_stage1_provenance.main(_build_args(raw, combined, folds, output)) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["verified"] is True
    assert payload["case_count"] == 95
    assert payload["num_folds"] == 5
    assert len(payload["folds"]) == 5
    assert len(payload["cases"]) == 95
    assert all(Path(payload["folds"][i]["validation_dir"]).is_absolute() for i in range(5))
    from coarse_to_fine_dwi.provenance import validate_stage1_provenance

    assert validate_stage1_provenance(output) == payload


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_provenance_rejects_fold_missing_or_extra_case(tmp_path, mutation):
    from coarse_to_fine_dwi.provenance import build_stage1_provenance

    raw, combined, folds = _make_inputs(tmp_path)
    target = folds[0]
    target_case_id = json.loads(REFERENCE_SPLITS.read_text(encoding="utf-8"))[0]["val"][0]
    if mutation == "missing":
        (target / f"{target_case_id}.nii.gz").unlink()
    else:
        _volume(1).write(target / "case-extra.nii.gz")

    with pytest.raises(ValueError, match="fold 0"):
        build_stage1_provenance(
            raw,
            REFERENCE_SPLITS,
            combined,
            folds,
            tmp_path / "stage1_provenance.json",
        )


def test_provenance_rejects_fold_to_combined_content_mismatch(tmp_path):
    from coarse_to_fine_dwi.provenance import build_stage1_provenance

    raw, combined, folds = _make_inputs(tmp_path)
    target_case_id = json.loads(REFERENCE_SPLITS.read_text(encoding="utf-8"))[0]["val"][0]
    _volume(99).write(folds[0] / f"{target_case_id}.nii.gz")

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        build_stage1_provenance(
            raw,
            REFERENCE_SPLITS,
            combined,
            folds,
            tmp_path / "stage1_provenance.json",
        )


def test_provenance_rejects_combined_oof_from_unrelated_result_root(tmp_path):
    from coarse_to_fine_dwi.provenance import build_stage1_provenance

    raw, combined, folds = _make_inputs(tmp_path)
    unrelated_root = tmp_path / "unrelated" / "nnUNetTrainer__nnUNetPlans__2d"
    unrelated_combined = unrelated_root / "crossval_results_folds_0_1_2_3_4"
    unrelated_combined.mkdir(parents=True)
    for source in combined.iterdir():
        shutil.copy2(source, unrelated_combined / source.name)

    with pytest.raises(ValueError, match="same Stage-1 result root"):
        build_stage1_provenance(
            raw,
            REFERENCE_SPLITS,
            unrelated_combined,
            folds,
            tmp_path / "stage1_provenance.json",
        )


def test_provenance_rejects_dataset501_case_id_mismatch(tmp_path):
    from coarse_to_fine_dwi.provenance import build_stage1_provenance

    raw, combined, folds = _make_inputs(tmp_path)
    (raw / "labelsTr" / "case001.nii.gz").unlink()
    _volume(1).write(raw / "labelsTr" / "case-extra.nii.gz")

    with pytest.raises(ValueError, match="labelsTr"):
        build_stage1_provenance(
            raw,
            REFERENCE_SPLITS,
            combined,
            folds,
            tmp_path / "stage1_provenance.json",
        )


def test_static_verified_provenance_is_rejected(tmp_path):
    from coarse_to_fine_dwi.provenance import validate_stage1_provenance

    path = tmp_path / "static.json"
    path.write_text(
        json.dumps(
            {
                "verified": True,
                "stage1_trainer": "nnUNetTrainer",
                "stage1_prediction_source": "complete_5_fold_oof",
                "roi_source": "stage1_prediction_only",
                "split_policy": "fixed_5_fold_patient_level",
                "num_folds": 5,
                "case_count": 95,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="evidence"):
        validate_stage1_provenance(path)
