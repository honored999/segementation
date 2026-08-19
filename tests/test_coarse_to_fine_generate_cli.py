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


def _fixed_case_ids() -> list[str]:
    folds = json.loads(REFERENCE_SPLITS.read_text(encoding="utf-8"))
    return sorted({case_id for fold in folds for case_id in fold["val"]})


def _volume(array: np.ndarray) -> NiftiVolume:
    return NiftiVolume(
        array=array,
        spacing_xyz=(1.5, 1.0, 2.0),
        origin_xyz=(10.0, 20.0, 30.0),
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )


def _make_synthetic_inputs(tmp_path: Path) -> tuple[Path, Path]:
    pytest.importorskip("SimpleITK")
    raw_root = tmp_path / "dataset501"
    images_root = raw_root / "imagesTr"
    labels_root = raw_root / "labelsTr"
    result_root = tmp_path / "nnUNetTrainer__nnUNetPlans__2d"
    prediction_root = result_root / "crossval_results_folds_0_1_2_3_4"
    images_root.mkdir(parents=True)
    labels_root.mkdir()
    prediction_root.mkdir(parents=True)

    for index, case_id in enumerate(_fixed_case_ids()):
        image = np.full((2, 6, 8), index, dtype=np.float32)
        label = np.zeros((2, 6, 8), dtype=np.uint8)
        label[:, 4:6, 6:8] = 1
        prediction = np.zeros((2, 6, 8), dtype=np.uint8)
        if case_id != "case001":
            prediction[:, 1, 2] = 1
        _volume(image).write(images_root / f"{case_id}_0000.nii.gz")
        _volume(label).write(labels_root / f"{case_id}.nii.gz")
        _volume(prediction).write(prediction_root / f"{case_id}.nii.gz")
    return raw_root, prediction_root


def _write_provenance(path: Path, *, verified: bool = False) -> Path:
    path.write_text(
        json.dumps(
            {
                "verified": verified,
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
    return path


def _build_valid_provenance(tmp_path: Path) -> tuple[Path, Path, Path]:
    from coarse_to_fine_dwi.cli import build_stage1_provenance

    raw_root, prediction_root = _make_synthetic_inputs(tmp_path)
    splits = json.loads(REFERENCE_SPLITS.read_text(encoding="utf-8"))
    fold_dirs = {}
    for fold_index, fold in enumerate(splits):
        fold_dir = prediction_root.parent / f"fold_{fold_index}" / "validation"
        fold_dir.mkdir(parents=True)
        fold_dirs[fold_index] = fold_dir
        for case_id in fold["val"]:
            shutil.copy2(
                prediction_root / f"{case_id}.nii.gz",
                fold_dir / f"{case_id}.nii.gz",
            )
    provenance = tmp_path / "stage1_provenance.json"
    args = [
        "--dataset501-raw", str(raw_root),
        "--splits", str(REFERENCE_SPLITS),
        "--stage1-oof-dir", str(prediction_root),
    ]
    for fold_index in range(5):
        args.extend([f"--fold-{fold_index}-validation", str(fold_dirs[fold_index])])
    args.extend(["--output", str(provenance)])
    assert build_stage1_provenance.main(args) == 0
    return raw_root, prediction_root, provenance


def test_cli_generates_dataset504_and_records_roi_manifest(tmp_path):
    from coarse_to_fine_dwi.cli import generate_dataset

    raw_root, prediction_root, provenance = _build_valid_provenance(tmp_path)
    output_root = tmp_path / "derived" / "Dataset504"

    roi_manifest = generate_dataset.generate_dataset504(
        dataset501_raw=raw_root,
        stage1_oof_dir=prediction_root,
        output_root=output_root,
        splits=REFERENCE_SPLITS,
        margin_px=1,
        min_roi_size=(2, 3),
        stage1_provenance=provenance,
    )

    assert roi_manifest == output_root / "roi_manifest.json"
    assert len(list((output_root / "imagesTr").glob("*.nii.gz"))) == 95
    assert len(list((output_root / "labelsTr").glob("*.nii.gz"))) == 95
    manifest = json.loads(roi_manifest.read_text(encoding="utf-8"))
    assert manifest["formal_eligible"] is True
    assert manifest["cases"]["case001"]["fallback"] is True


def test_cli_reports_missing_oof_prediction(tmp_path):
    from coarse_to_fine_dwi.cli import generate_dataset

    raw_root, prediction_root, provenance = _build_valid_provenance(tmp_path)
    case_id = _fixed_case_ids()[-1]
    (prediction_root / f"{case_id}.nii.gz").unlink()

    with pytest.raises(ValueError, match="combined OOF predictions IDs mismatch"):
        generate_dataset.generate_dataset504(
            dataset501_raw=raw_root,
            stage1_oof_dir=prediction_root,
            output_root=tmp_path / "derived",
            splits=REFERENCE_SPLITS,
            margin_px=0,
            min_roi_size=(1, 1),
            stage1_provenance=provenance,
        )


def test_cli_reports_fixed_split_case_id_mismatch(tmp_path):
    from coarse_to_fine_dwi.cli import generate_dataset

    raw_root, prediction_root, provenance = _build_valid_provenance(tmp_path)
    bad_splits = tmp_path / "bad_splits.json"
    splits = json.loads(REFERENCE_SPLITS.read_text(encoding="utf-8"))
    splits[0]["val"][0] = "case-not-in-fixed-split"
    bad_splits.write_text(json.dumps(splits), encoding="utf-8")
    with pytest.raises(ValueError, match="splits does not match verified provenance"):
        generate_dataset.generate_dataset504(
            dataset501_raw=raw_root,
            stage1_oof_dir=prediction_root,
            output_root=tmp_path / "derived",
            splits=bad_splits,
            margin_px=0,
            min_roi_size=(1, 1),
            stage1_provenance=provenance,
        )


def test_cli_forwards_explicit_arguments_to_builder(tmp_path, monkeypatch):
    from coarse_to_fine_dwi.cli import generate_dataset

    raw_root, oof_root, provenance = _build_valid_provenance(tmp_path)
    captured = {}

    def fake_builder(raw, oof, output, *, splits_path, margin, min_width, min_height):
        captured.update(
            raw=raw,
            oof=oof,
            output=output,
            splits_path=splits_path,
            margin=margin,
            min_width=min_width,
            min_height=min_height,
        )
        output.mkdir(parents=True)
        (output / "manifest.json").write_text(
            json.dumps({"roi_source": "stage1_prediction_only"}), encoding="utf-8"
        )
        return output

    monkeypatch.setattr(generate_dataset, "build_dataset504", fake_builder)
    output_root = tmp_path / "output"

    result = generate_dataset.main(
        [
            "--dataset501-raw",
            str(raw_root),
            "--stage1-oof-dir",
            str(oof_root),
            "--output-root",
            str(output_root),
            "--splits",
            str(REFERENCE_SPLITS),
            "--margin-px",
            "7",
            "--min-roi-size",
            "5",
            "9",
            "--stage1-provenance",
            str(provenance),
        ]
    )

    assert result == 0
    assert captured == {
        "raw": raw_root,
        "oof": oof_root,
        "output": output_root,
        "splits_path": REFERENCE_SPLITS,
        "margin": 7,
        "min_width": 5,
        "min_height": 9,
    }
    manifest = json.loads((output_root / "roi_manifest.json").read_text(encoding="utf-8"))
    assert manifest["formal_eligible"] is True


def test_cli_forwards_preferred_roi_arguments_to_builder(tmp_path, monkeypatch):
    from coarse_to_fine_dwi.cli import generate_dataset

    raw_root, oof_root, provenance = _build_valid_provenance(tmp_path)
    captured = {}

    def fake_builder(raw, oof, output, *, splits_path, margin, min_width, min_height):
        captured.update(margin=margin, min_width=min_width, min_height=min_height)
        output.mkdir(parents=True)
        (output / "manifest.json").write_text(
            json.dumps({"roi_source": "stage1_prediction_only"}), encoding="utf-8"
        )
        return output

    monkeypatch.setattr(generate_dataset, "build_dataset504", fake_builder)

    result = generate_dataset.main(
        [
            "--dataset501-raw",
            str(raw_root),
            "--stage1-oof-dir",
            str(oof_root),
            "--output-root",
            str(tmp_path / "output"),
            "--splits",
            str(REFERENCE_SPLITS),
            "--roi-margin",
            "11",
            "--min-roi-width",
            "17",
            "--min-roi-height",
            "19",
            "--stage1-provenance",
            str(provenance),
        ]
    )

    assert result == 0
    assert captured == {"margin": 11, "min_width": 17, "min_height": 19}


def test_cli_defaults_to_contextual_roi_policy(tmp_path, monkeypatch):
    from coarse_to_fine_dwi.cli import generate_dataset

    raw_root, oof_root, provenance = _build_valid_provenance(tmp_path)
    captured = {}

    def fake_builder(raw, oof, output, *, splits_path, margin, min_width, min_height):
        captured.update(margin=margin, min_width=min_width, min_height=min_height)
        output.mkdir(parents=True)
        (output / "manifest.json").write_text("{}", encoding="utf-8")
        return output

    monkeypatch.setattr(generate_dataset, "build_dataset504", fake_builder)

    assert generate_dataset.main(
        [
            "--dataset501-raw",
            str(raw_root),
            "--stage1-oof-dir",
            str(oof_root),
            "--output-root",
            str(tmp_path / "output"),
            "--splits",
            str(REFERENCE_SPLITS),
            "--stage1-provenance",
            str(provenance),
        ]
    ) == 0
    assert captured == {"margin": 32, "min_width": 128, "min_height": 128}


def test_cli_rejects_conflicting_preferred_and_legacy_roi_arguments(tmp_path):
    from coarse_to_fine_dwi.cli import generate_dataset

    raw_root, oof_root, provenance = _build_valid_provenance(tmp_path)

    with pytest.raises(ValueError, match="cannot be combined"):
        generate_dataset.generate_dataset504(
            dataset501_raw=raw_root,
            stage1_oof_dir=oof_root,
            output_root=tmp_path / "derived",
            splits=REFERENCE_SPLITS,
            stage1_provenance=provenance,
            roi_margin=11,
            margin_px=7,
        )


def test_tampered_generated_provenance_is_rejected_before_builder(tmp_path, monkeypatch):
    from coarse_to_fine_dwi.cli import generate_dataset

    raw_root, prediction_root, provenance = _build_valid_provenance(tmp_path)
    payload = json.loads(provenance.read_text(encoding="utf-8"))
    payload["cases"][_fixed_case_ids()[0]]["fold"] = 4
    provenance.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        generate_dataset,
        "build_dataset504",
        lambda *args, **kwargs: pytest.fail("builder must not run for tampered provenance"),
    )

    with pytest.raises(ValueError, match="does not match current inputs"):
        generate_dataset.generate_dataset504(
            dataset501_raw=raw_root,
            stage1_oof_dir=prediction_root,
            output_root=tmp_path / "derived",
            splits=REFERENCE_SPLITS,
            margin_px=0,
            min_roi_size=(1, 1),
            stage1_provenance=provenance,
        )


@pytest.mark.parametrize(
    ("mismatched_argument", "message"),
    [
        ("dataset501_raw", "dataset501_raw"),
        ("stage1_oof_dir", "stage1_oof_dir"),
        ("splits", "splits"),
    ],
)
def test_verified_provenance_rejects_mismatched_builder_inputs(
    tmp_path, monkeypatch, mismatched_argument, message
):
    from coarse_to_fine_dwi.cli import generate_dataset

    raw_root, prediction_root, provenance = _build_valid_provenance(tmp_path)
    inputs = {
        "dataset501_raw": raw_root,
        "stage1_oof_dir": prediction_root,
        "splits": REFERENCE_SPLITS,
    }
    inputs[mismatched_argument] = tmp_path / f"mismatched-{mismatched_argument}"
    monkeypatch.setattr(
        generate_dataset,
        "build_dataset504",
        lambda *args, **kwargs: pytest.fail("builder must not run for mismatched provenance inputs"),
    )

    with pytest.raises(ValueError, match=message):
        generate_dataset.generate_dataset504(
            dataset501_raw=inputs["dataset501_raw"],
            stage1_oof_dir=inputs["stage1_oof_dir"],
            output_root=tmp_path / "derived",
            splits=inputs["splits"],
            margin_px=0,
            min_roi_size=(1, 1),
            stage1_provenance=provenance,
        )


def test_unverified_provenance_cannot_be_marked_formal(tmp_path, monkeypatch):
    from coarse_to_fine_dwi.cli import generate_dataset

    provenance = _write_provenance(tmp_path / "stage1_provenance.json", verified=False)

    monkeypatch.setattr(
        generate_dataset,
        "build_dataset504",
        lambda *args, **kwargs: pytest.fail("builder must not run"),
    )

    with pytest.raises(ValueError, match="evidence"):
        generate_dataset.generate_dataset504(
            dataset501_raw=tmp_path / "raw",
            stage1_oof_dir=tmp_path / "oof",
            output_root=tmp_path / "output",
            splits=REFERENCE_SPLITS,
            margin_px=0,
            min_roi_size=(1, 1),
            stage1_provenance=provenance,
        )
