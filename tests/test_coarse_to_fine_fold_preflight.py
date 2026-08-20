import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from coarse_to_fine_dwi.dataset import build_dataset504
from coarse_to_fine_dwi.nifti import NiftiVolume
from coarse_to_fine_dwi.provenance import build_stage1_provenance


REFERENCE_SPLITS = (
    Path(__file__).parents[1]
    / "standalone_nnunet2d"
    / "reference"
    / "splits_final.json"
)


def _volume(array: np.ndarray) -> NiftiVolume:
    return NiftiVolume(
        array=array,
        spacing_xyz=(1.5, 1.0, 2.0),
        origin_xyz=(10.0, 20.0, 30.0),
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )


def _fixed_splits() -> list[dict[str, list[str]]]:
    return json.loads(REFERENCE_SPLITS.read_text(encoding="utf-8"))


def _fixture(tmp_path: Path) -> dict[str, Path | list[str]]:
    pytest.importorskip("SimpleITK")
    splits = _fixed_splits()
    fold_ids = splits[0]["val"]
    case_ids = sorted({case_id for fold in splits for case_id in fold["val"]})

    raw_root = tmp_path / "dataset501"
    images_root = raw_root / "imagesTr"
    labels_root = raw_root / "labelsTr"
    stage1_result_root = tmp_path / "nnUNetTrainer__nnUNetPlans__2d"
    stage1_oof_dir = stage1_result_root / "crossval_results_folds_0_1_2_3_4"
    images_root.mkdir(parents=True)
    labels_root.mkdir()
    stage1_oof_dir.mkdir(parents=True)

    for index, case_id in enumerate(case_ids):
        image = np.full((2, 8, 8), index, dtype=np.float32)
        label = np.zeros((2, 8, 8), dtype=np.uint8)
        label[:, 2:4, 3:5] = 1
        prediction = label.copy()
        _volume(image).write(images_root / f"{case_id}_0000.nii.gz")
        _volume(label).write(labels_root / f"{case_id}.nii.gz")
        _volume(prediction).write(stage1_oof_dir / f"{case_id}.nii.gz")

    fold_dirs: dict[int, Path] = {}
    for fold_index, fold in enumerate(splits):
        validation_dir = stage1_result_root / f"fold_{fold_index}" / "validation"
        validation_dir.mkdir(parents=True)
        fold_dirs[fold_index] = validation_dir
        for case_id in fold["val"]:
            shutil.copy2(
                stage1_oof_dir / f"{case_id}.nii.gz",
                validation_dir / f"{case_id}.nii.gz",
            )

    provenance_path = tmp_path / "stage1_provenance.json"
    build_stage1_provenance(
        raw_root,
        REFERENCE_SPLITS,
        stage1_oof_dir,
        fold_dirs,
        provenance_path,
    )

    dataset504_root = tmp_path / "Dataset504"
    build_dataset504(
        raw_root,
        stage1_oof_dir,
        dataset504_root,
        splits_path=REFERENCE_SPLITS,
        margin=0,
        min_width=4,
        min_height=2,
    )
    manifest_path = dataset504_root / "manifest.json"

    stage2_root = tmp_path / "stage2_cropped_predictions"
    stage2_root.mkdir()
    for case_id in fold_ids:
        shutil.copy2(
            dataset504_root / "labelsTr" / f"{case_id}.nii.gz",
            stage2_root / f"{case_id}.nii.gz",
        )

    return {
        "raw": raw_root,
        "splits": REFERENCE_SPLITS,
        "stage1_oof": stage1_oof_dir,
        "provenance": provenance_path,
        "manifest": manifest_path,
        "stage2": stage2_root,
        "fold_ids": fold_ids,
    }


def _cli_args(inputs: dict[str, Path | list[str]], tmp_path: Path, *, fold: int = 0) -> list[str]:
    return [
        "--manifest",
        str(inputs["manifest"]),
        "--stage2-cropped-predictions",
        str(inputs["stage2"]),
        "--dataset501-raw",
        str(inputs["raw"]),
        "--splits",
        str(inputs["splits"]),
        "--fold",
        str(fold),
        "--stage1-oof-dir",
        str(inputs["stage1_oof"]),
        "--stage1-provenance",
        str(inputs["provenance"]),
        "--restored-output-dir",
        str(tmp_path / "restored"),
        "--evaluation-output-dir",
        str(tmp_path / "evaluation"),
    ]


def test_fold_preflight_restores_and_compares_exact_fixed_fold(tmp_path):
    from coarse_to_fine_dwi.cli import evaluate_fold_preflight

    inputs = _fixture(tmp_path)
    assert evaluate_fold_preflight.main(_cli_args(inputs, tmp_path)) == 0

    fold_ids = sorted(inputs["fold_ids"])
    restored_files = sorted((tmp_path / "restored").glob("*.nii.gz"))
    assert len(restored_files) == 19
    assert [path.name[:-7] for path in restored_files] == fold_ids

    summary = json.loads(
        (tmp_path / "evaluation" / "stage1_vs_stage2_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["protocol"]["gt_source"] == "Dataset501_labelsTr"
    assert summary["formal_eligible"] is False
    assert summary["case_count"] == 19
    assert summary["case_ids"] == fold_ids

    metadata = json.loads(
        (tmp_path / "evaluation" / "fold_preflight_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["run_kind"] == "fold_preflight"
    assert metadata["formal_eligible"] is False
    assert metadata["fold"] == 0
    assert metadata["case_count"] == 19
    assert metadata["case_ids"] == fold_ids
    for field in (
        "manifest",
        "splits",
        "stage1_oof_dir",
        "stage1_provenance",
        "stage2_cropped_predictions",
        "restored_output_dir",
        "evaluation_output_dir",
    ):
        assert Path(metadata[field]).is_absolute()


def test_fold_preflight_rejects_wrong_fold(tmp_path):
    from coarse_to_fine_dwi.cli.evaluate_fold_preflight import evaluate_fold_preflight

    inputs = _fixture(tmp_path)
    with pytest.raises(ValueError, match="fold must be between 0 and 4"):
        evaluate_fold_preflight(
            manifest=inputs["manifest"],
            stage2_cropped_predictions=inputs["stage2"],
            dataset501_raw=inputs["raw"],
            splits=inputs["splits"],
            fold=5,
            stage1_oof_dir=inputs["stage1_oof"],
            stage1_provenance=inputs["provenance"],
            restored_output_dir=tmp_path / "restored",
            evaluation_output_dir=tmp_path / "evaluation",
        )


def test_fold_preflight_rejects_mismatched_provenance_input(tmp_path):
    from coarse_to_fine_dwi.cli.evaluate_fold_preflight import evaluate_fold_preflight

    inputs = _fixture(tmp_path)
    other_oof = tmp_path / "other_oof"
    other_oof.mkdir()
    with pytest.raises(ValueError, match="stage1_oof_dir does not match verified provenance"):
        evaluate_fold_preflight(
            manifest=inputs["manifest"],
            stage2_cropped_predictions=inputs["stage2"],
            dataset501_raw=inputs["raw"],
            splits=inputs["splits"],
            fold=0,
            stage1_oof_dir=other_oof,
            stage1_provenance=inputs["provenance"],
            restored_output_dir=tmp_path / "restored",
            evaluation_output_dir=tmp_path / "evaluation",
        )


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_fold_preflight_rejects_missing_or_extra_stage2_prediction_ids(tmp_path, mutation):
    from coarse_to_fine_dwi.cli.evaluate_fold_preflight import evaluate_fold_preflight

    inputs = _fixture(tmp_path)
    stage2 = Path(inputs["stage2"])
    if mutation == "missing":
        (stage2 / f"{inputs['fold_ids'][0]}.nii.gz").unlink()
    else:
        shutil.copy2(
            stage2 / f"{inputs['fold_ids'][0]}.nii.gz",
            stage2 / "case-extra.nii.gz",
        )
    with pytest.raises(ValueError, match="cropped predictions IDs mismatch"):
        evaluate_fold_preflight(
            manifest=inputs["manifest"],
            stage2_cropped_predictions=stage2,
            dataset501_raw=inputs["raw"],
            splits=inputs["splits"],
            fold=0,
            stage1_oof_dir=inputs["stage1_oof"],
            stage1_provenance=inputs["provenance"],
            restored_output_dir=tmp_path / "restored",
            evaluation_output_dir=tmp_path / "evaluation",
        )


def test_fold_preflight_rejects_incorrect_manifest_fold_metadata(tmp_path):
    from coarse_to_fine_dwi.cli.evaluate_fold_preflight import evaluate_fold_preflight

    inputs = _fixture(tmp_path)
    manifest = json.loads(Path(inputs["manifest"]).read_text(encoding="utf-8"))
    case_id = inputs["fold_ids"][0]
    manifest["cases"][case_id]["fold"] = 1
    Path(inputs["manifest"]).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest fold metadata mismatch"):
        evaluate_fold_preflight(
            manifest=inputs["manifest"],
            stage2_cropped_predictions=inputs["stage2"],
            dataset501_raw=inputs["raw"],
            splits=inputs["splits"],
            fold=0,
            stage1_oof_dir=inputs["stage1_oof"],
            stage1_provenance=inputs["provenance"],
            restored_output_dir=tmp_path / "restored",
            evaluation_output_dir=tmp_path / "evaluation",
        )


def test_fold_preflight_rejects_nonfixed_splits(tmp_path):
    from coarse_to_fine_dwi.cli.evaluate_fold_preflight import evaluate_fold_preflight

    inputs = _fixture(tmp_path)
    bad_splits = tmp_path / "bad_splits.json"
    splits = _fixed_splits()
    splits[0]["val"][0] = "case-not-fixed"
    bad_splits.write_text(json.dumps(splits), encoding="utf-8")
    with pytest.raises(ValueError, match="established fixed 5-fold Dataset501 split"):
        evaluate_fold_preflight(
            manifest=inputs["manifest"],
            stage2_cropped_predictions=inputs["stage2"],
            dataset501_raw=inputs["raw"],
            splits=bad_splits,
            fold=0,
            stage1_oof_dir=inputs["stage1_oof"],
            stage1_provenance=inputs["provenance"],
            restored_output_dir=tmp_path / "restored",
            evaluation_output_dir=tmp_path / "evaluation",
        )


def test_fold_preflight_rejects_nested_output_directories_before_writing(tmp_path):
    from coarse_to_fine_dwi.cli.evaluate_fold_preflight import evaluate_fold_preflight

    inputs = _fixture(tmp_path)
    restored = tmp_path / "outputs"
    evaluation = restored / "evaluation"
    with pytest.raises(ValueError, match="restored and evaluation output directories overlap"):
        evaluate_fold_preflight(
            manifest=inputs["manifest"],
            stage2_cropped_predictions=inputs["stage2"],
            dataset501_raw=inputs["raw"],
            splits=inputs["splits"],
            fold=0,
            stage1_oof_dir=inputs["stage1_oof"],
            stage1_provenance=inputs["provenance"],
            restored_output_dir=restored,
            evaluation_output_dir=evaluation,
        )
    assert not restored.exists()
