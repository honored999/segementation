import json
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
    prediction_root = tmp_path / "stage1_oof"
    images_root.mkdir(parents=True)
    labels_root.mkdir()
    prediction_root.mkdir()

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


def test_cli_generates_dataset504_and_records_roi_manifest(tmp_path):
    from coarse_to_fine_dwi.cli import generate_dataset

    raw_root, prediction_root = _make_synthetic_inputs(tmp_path)
    provenance = _write_provenance(tmp_path / "stage1_provenance.json")
    output_root = tmp_path / "derived" / "Dataset504"

    result = generate_dataset.main(
        [
            "--dataset501-raw",
            str(raw_root),
            "--stage1-oof-dir",
            str(prediction_root),
            "--output-root",
            str(output_root),
            "--splits",
            str(REFERENCE_SPLITS),
            "--margin-px",
            "1",
            "--min-roi-size",
            "2",
            "3",
            "--stage1-provenance",
            str(provenance),
        ]
    )

    assert result == 0
    roi_manifest = json.loads((output_root / "roi_manifest.json").read_text(encoding="utf-8"))
    assert roi_manifest["stage1_provenance"]["verified"] is False
    assert roi_manifest["formal_eligible"] is False
    assert roi_manifest["roi_source"] == "stage1_prediction_only"
    assert (output_root / "imagesTr" / "case001_0000.nii.gz").is_file()


def test_cli_reports_missing_oof_prediction(tmp_path):
    from coarse_to_fine_dwi.cli import generate_dataset

    raw_root, prediction_root = _make_synthetic_inputs(tmp_path)
    (prediction_root / "case095.nii.gz").unlink()
    provenance = _write_provenance(tmp_path / "stage1_provenance.json")

    with pytest.raises(ValueError, match="exactly the fixed 95-case OOF IDs"):
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

    raw_root, prediction_root = _make_synthetic_inputs(tmp_path)
    bad_splits = tmp_path / "bad_splits.json"
    splits = json.loads(REFERENCE_SPLITS.read_text(encoding="utf-8"))
    splits[0]["val"][0] = "case-not-in-fixed-split"
    bad_splits.write_text(json.dumps(splits), encoding="utf-8")
    provenance = _write_provenance(tmp_path / "stage1_provenance.json")

    with pytest.raises(ValueError, match="established fixed 5-fold Dataset501 split"):
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

    provenance = _write_provenance(tmp_path / "stage1_provenance.json")
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
    raw_root = tmp_path / "raw"
    oof_root = tmp_path / "oof"
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
    assert manifest["formal_eligible"] is False


def test_unverified_provenance_cannot_be_marked_formal(tmp_path, monkeypatch):
    from coarse_to_fine_dwi.cli import generate_dataset

    provenance = _write_provenance(tmp_path / "stage1_provenance.json", verified=False)

    def fake_builder(raw, oof, output, **kwargs):
        output.mkdir(parents=True)
        (output / "manifest.json").write_text(
            json.dumps({
                "stage1_prediction_source": "complete_5_fold_oof",
                "roi_source": "stage1_prediction_only",
            }),
            encoding="utf-8",
        )
        return output

    monkeypatch.setattr(generate_dataset, "build_dataset504", fake_builder)

    result = generate_dataset.main(
        [
            "--dataset501-raw",
            str(tmp_path / "raw"),
            "--stage1-oof-dir",
            str(tmp_path / "oof"),
            "--output-root",
            str(tmp_path / "output"),
            "--splits",
            str(REFERENCE_SPLITS),
            "--margin-px",
            "0",
            "--min-roi-size",
            "1",
            "1",
            "--stage1-provenance",
            str(provenance),
        ]
    )

    assert result == 0
    manifest = json.loads(
        (tmp_path / "output" / "roi_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["formal_eligible"] is False
