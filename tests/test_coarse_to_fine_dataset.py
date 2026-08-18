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


def _make_synthetic_inputs(
    tmp_path: Path,
    *,
    target_case: str = "case005",
    empty_prediction_case: str = "case001",
) -> tuple[Path, Path]:
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
        if case_id == target_case:
            prediction[:, 1, 2] = 1
        if case_id == empty_prediction_case:
            prediction.fill(0)
        else:
            prediction[:, 1, 2] = 1
        _volume(image).write(images_root / f"{case_id}_0000.nii.gz")
        _volume(label).write(labels_root / f"{case_id}.nii.gz")
        _volume(prediction).write(prediction_root / f"{case_id}.nii.gz")

    return raw_root, prediction_root


def test_build_dataset504_uses_exact_fixed_folds_and_writes_prediction_guided_manifest(tmp_path):
    from coarse_to_fine_dwi.dataset import build_dataset504

    raw_root, prediction_root = _make_synthetic_inputs(tmp_path)
    output_root = tmp_path / "generated" / "Dataset504_StrokeLesion_CoarseToFine"

    result = build_dataset504(
        raw_root,
        prediction_root,
        output_root,
        splits_path=REFERENCE_SPLITS,
        margin=1,
    )

    assert result == output_root
    assert sorted(path.name for path in (output_root / "imagesTr").glob("*.nii.gz")) == [
        f"{case_id}_0000.nii.gz" for case_id in _fixed_case_ids()
    ]
    assert sorted(path.name for path in (output_root / "labelsTr").glob("*.nii.gz")) == [
        f"{case_id}.nii.gz" for case_id in _fixed_case_ids()
    ]

    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["num_cases"] == 95
    assert manifest["num_folds"] == 5
    assert manifest["case_ids"] == _fixed_case_ids()
    assert manifest["cases"]["case005"]["roi"] == [1, 0, 4, 3]
    assert manifest["cases"]["case005"]["fallback"] is False
    assert manifest["cases"]["case001"]["fallback"] is True
    assert json.loads((output_root / "splits_final.json").read_text(encoding="utf-8")) == json.loads(
        REFERENCE_SPLITS.read_text(encoding="utf-8")
    )

    dataset_json = json.loads((output_root / "dataset.json").read_text(encoding="utf-8"))
    assert dataset_json["channel_names"] == {"0": "DWI"}
    assert dataset_json["labels"] == {"background": 0, "lesion": 1}
    assert dataset_json["numTraining"] == 95

    empty_case = NiftiVolume.read(output_root / "imagesTr" / "case001_0000.nii.gz")
    assert empty_case.shape_zyx == (2, 6, 8)


def test_builder_derives_roi_from_prediction_before_reading_gt(tmp_path, monkeypatch):
    import coarse_to_fine_dwi.dataset as dataset_module

    raw_root, prediction_root = _make_synthetic_inputs(tmp_path)
    output_root = tmp_path / "generated"
    events: list[str] = []
    original_read = dataset_module.NiftiVolume.read
    original_roi = dataset_module.compute_prediction_roi

    def read_with_event(path):
        if Path(path).parent.name == "labelsTr":
            assert "roi" in events
            events.append("gt")
        else:
            events.append("input")
        return original_read(path)

    def roi_with_event(prediction, **kwargs):
        events.append("roi")
        return original_roi(prediction, **kwargs)

    monkeypatch.setattr(dataset_module.NiftiVolume, "read", staticmethod(read_with_event))
    monkeypatch.setattr(dataset_module, "compute_prediction_roi", roi_with_event)

    dataset_module.build_dataset504(
        raw_root,
        prediction_root,
        output_root,
        splits_path=REFERENCE_SPLITS,
    )

    assert events.index("roi") < events.index("gt")


@pytest.mark.parametrize("bad_prediction_case", ["missing", "extra"])
def test_builder_requires_exact_95_oof_prediction_ids(tmp_path, bad_prediction_case):
    from coarse_to_fine_dwi.dataset import build_dataset504

    raw_root, prediction_root = _make_synthetic_inputs(tmp_path)
    if bad_prediction_case == "missing":
        (prediction_root / "case095.nii.gz").unlink()
    else:
        NiftiVolume.write(
            _volume(np.zeros((2, 6, 8), dtype=np.uint8)),
            prediction_root / "case-extra.nii.gz",
        )

    with pytest.raises(ValueError, match="exactly the fixed 95-case OOF IDs"):
        build_dataset504(
            raw_root,
            prediction_root,
            tmp_path / "generated",
            splits_path=REFERENCE_SPLITS,
        )


def test_builder_rejects_output_inside_raw_or_prediction_roots(tmp_path):
    from coarse_to_fine_dwi.dataset import build_dataset504

    raw_root, prediction_root = _make_synthetic_inputs(tmp_path)

    with pytest.raises(ValueError, match="isolated"):
        build_dataset504(
            raw_root,
            prediction_root,
            raw_root / "generated504",
            splits_path=REFERENCE_SPLITS,
        )
    with pytest.raises(ValueError, match="isolated"):
        build_dataset504(
            raw_root,
            prediction_root,
            prediction_root / "generated504",
            splits_path=REFERENCE_SPLITS,
        )
