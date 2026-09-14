from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from torch.utils.data import DataLoader

from standalone_nnunet2d import formal_train
from standalone_nnunet2d.data.dataset import StrokeSliceDataset, load_fold_cases
from standalone_nnunet2d.data.nifti_io import NiftiVolume, write_nifti
from standalone_nnunet2d.training.official_config import OfficialTrainerSchedule


def _save_b2nd(path: Path, array: np.ndarray) -> None:
    blosc2 = pytest.importorskip("blosc2")
    encoded = blosc2.asarray(array, chunks=(1, 1, array.shape[2], array.shape[3]))
    blosc2.save(encoded, str(path))


def _write_raw_case(raw_root: Path, case_id: str, image: np.ndarray, label: np.ndarray) -> None:
    volume = NiftiVolume(
        image,
        (0.4892368018627167, 0.4892368018627167, 3.0),
        (0.0, 0.0, 0.0),
    )
    write_nifti(raw_root / "imagesTr" / f"{case_id}_0000.nii.gz", volume)
    write_nifti(
        raw_root / "labelsTr" / f"{case_id}.nii.gz",
        NiftiVolume(label, volume.spacing_xyz, volume.origin_xyz),
    )


def _write_preprocessed_case(
    preprocessed_root: Path, case_id: str, image: np.ndarray, label: np.ndarray
) -> None:
    _save_b2nd(preprocessed_root / f"{case_id}.b2nd", image[None].astype(np.float32))
    _save_b2nd(preprocessed_root / f"{case_id}_seg.b2nd", label[None].astype(np.int16))
    (preprocessed_root / f"{case_id}.pkl").write_bytes(b"not parsed")


def test_formal_preprocessed_dataset_reads_only_selected_image_slice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blosc2 = pytest.importorskip("blosc2")
    case_id = load_fold_cases(0, "train")[0]
    preprocessed_root = tmp_path / "nnUNetPlans_2d"
    preprocessed_root.mkdir()
    image = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)
    label = np.zeros((3, 8, 8), dtype=np.int16)
    label[2, 2, 3] = 1
    _write_preprocessed_case(preprocessed_root, case_id, image, label)

    calls: list[tuple[object, object]] = []
    original_open = blosc2.open

    class RecordingStore:
        def __init__(self, store: object, path: str) -> None:
            self._store = store
            self.path = path
            self.shape = store.shape  # type: ignore[attr-defined]
            self.dtype = store.dtype  # type: ignore[attr-defined]

        def __getitem__(self, key: object) -> object:
            calls.append((self.path, key))
            return self._store[key]  # type: ignore[index]

    def recording_open(*, urlpath: str, mode: str) -> RecordingStore:
        return RecordingStore(original_open(urlpath=urlpath, mode=mode), urlpath)

    monkeypatch.setattr(blosc2, "open", recording_open)

    from standalone_nnunet2d.data.data_source import NNUNET_PREPROCESSED_B2ND
    from standalone_nnunet2d.training.formal_dataset import FormalPatchDataset

    dataset = FormalPatchDataset(
        preprocessed_root,
        fold=0,
        split="train",
        case_ids=(case_id,),
        data_source=NNUNET_PREPROCESSED_B2ND,
        patch_size=(8, 8),
        oversample_foreground_percent=1.0,
        rng=np.random.default_rng(0),
        augment=False,
    )

    image_tensor, target_tensor = dataset[0]

    assert tuple(image_tensor.shape) == (1, 8, 8)
    assert tuple(target_tensor.shape) == (8, 8)
    # crop_or_pad centers the sampled foreground coordinate in the output
    # patch, so source (2, 3) maps to (4, 4) after the required edge padding.
    assert target_tensor[4, 4].item() == 1
    data_calls = [key for path, key in calls if path.endswith(f"{case_id}.b2nd")]
    assert data_calls == [(0, 2, slice(None), slice(None))]
    assert all(key != (slice(None),) for key in data_calls)


def test_preprocessed_source_normalizes_minus_one_before_binary_sampling(
    tmp_path: Path,
) -> None:
    pytest.importorskip("blosc2")
    case_id = load_fold_cases(0, "train")[0]
    preprocessed_root = tmp_path / "preprocessed"
    preprocessed_root.mkdir()
    image = np.zeros((3, 4, 4), dtype=np.float32)
    label = np.zeros((3, 4, 4), dtype=np.int16)
    label[0, 0, 0] = -1
    label[2, 1, 2] = 1
    _write_preprocessed_case(preprocessed_root, case_id, image, label)

    from standalone_nnunet2d.data.data_source import PreprocessedB2ndCaseSource

    prepared = PreprocessedB2ndCaseSource(
        preprocessed_root, fold=0, split="train", case_ids=(case_id,)
    ).prepare_case(case_id)

    np.testing.assert_array_equal(prepared.label, np.where(label == -1, 0, label))
    assert set(np.unique(prepared.label).tolist()) <= {0, 1}
    assert prepared.raw_label_unique == (-1, 0, 1)
    assert prepared.raw_label_counts == {
        -1: 1,
        0: int(np.count_nonzero(label == 0)),
        1: 1,
    }


def test_preprocessed_formal_sampling_uses_only_label_one(tmp_path: Path) -> None:
    pytest.importorskip("blosc2")
    case_id = load_fold_cases(0, "train")[0]
    preprocessed_root = tmp_path / "preprocessed"
    preprocessed_root.mkdir()
    image = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
    label = np.full((3, 4, 4), -1, dtype=np.int16)
    label[2, 1, 2] = 1
    _write_preprocessed_case(preprocessed_root, case_id, image, label)

    from standalone_nnunet2d.data.data_source import NNUNET_PREPROCESSED_B2ND
    from standalone_nnunet2d.training.formal_dataset import FormalPatchDataset

    dataset = FormalPatchDataset(
        preprocessed_root,
        fold=0,
        split="train",
        case_ids=(case_id,),
        data_source=NNUNET_PREPROCESSED_B2ND,
        patch_size=(4, 4),
        oversample_foreground_percent=1.0,
        rng=np.random.default_rng(0),
        augment=False,
    )

    _, target = dataset[0]

    assert target.sum().item() == 1
    assert target[2, 2].item() == 1
    assert set(target.unique().tolist()) <= {0, 1}


def test_formal_preprocessed_dataset_supports_len_and_real_dataloader(
    tmp_path: Path,
) -> None:
    pytest.importorskip("blosc2")
    case_id = load_fold_cases(0, "train")[0]
    preprocessed_root = tmp_path / "preprocessed"
    preprocessed_root.mkdir()
    image = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)
    label = np.zeros((3, 8, 8), dtype=np.int16)
    label[1, 2, 3] = 1
    _write_preprocessed_case(preprocessed_root, case_id, image, label)

    from standalone_nnunet2d.data.data_source import NNUNET_PREPROCESSED_B2ND
    from standalone_nnunet2d.training.formal_dataset import FormalPatchDataset

    dataset = FormalPatchDataset(
        preprocessed_root,
        fold=0,
        split="train",
        case_ids=(case_id,),
        data_source=NNUNET_PREPROCESSED_B2ND,
        patch_size=(8, 8),
        rng=np.random.default_rng(0),
        augment=False,
    )

    assert len(dataset) == 1
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    image_batch, target_batch = next(iter(loader))
    assert tuple(image_batch.shape) == (1, 1, 8, 8)
    assert tuple(target_batch.shape) == (1, 8, 8)


def test_formal_preprocessed_and_raw_paths_preserve_sampling_output(tmp_path: Path) -> None:
    pytest.importorskip("blosc2")
    case_id = load_fold_cases(0, "train")[0]
    raw_root = tmp_path / "raw"
    preprocessed_root = tmp_path / "preprocessed"
    (raw_root / "imagesTr").mkdir(parents=True)
    (raw_root / "labelsTr").mkdir()
    preprocessed_root.mkdir()
    image = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)
    label = np.zeros((3, 8, 8), dtype=np.int16)
    label[2, 1:4, 2:5] = 1
    _write_raw_case(raw_root, case_id, image, label)

    raw_dataset = StrokeSliceDataset(
        raw_root, fold=0, split="train", case_ids=(case_id,)
    )
    processed_image, processed_label = raw_dataset.load_case(case_id)
    _write_preprocessed_case(preprocessed_root, case_id, processed_image, processed_label)

    from standalone_nnunet2d.data.data_source import NNUNET_PREPROCESSED_B2ND
    from standalone_nnunet2d.training.formal_dataset import FormalPatchDataset

    raw_formal = FormalPatchDataset(
        raw_root,
        fold=0,
        split="train",
        case_ids=(case_id,),
        patch_size=(4, 4),
        oversample_foreground_percent=1.0,
        rng=np.random.default_rng(17),
        augment=False,
    )
    preprocessed_formal = FormalPatchDataset(
        preprocessed_root,
        fold=0,
        split="train",
        case_ids=(case_id,),
        data_source=NNUNET_PREPROCESSED_B2ND,
        patch_size=(4, 4),
        oversample_foreground_percent=1.0,
        rng=np.random.default_rng(17),
        augment=False,
    )

    raw_sample = raw_formal[0]
    preprocessed_sample = preprocessed_formal[0]
    np.testing.assert_allclose(raw_sample[0].numpy(), preprocessed_sample[0].numpy())
    np.testing.assert_array_equal(raw_sample[1].numpy(), preprocessed_sample[1].numpy())


def test_formal_preprocessed_source_rejects_shape_and_label_contract_violations(
    tmp_path: Path,
) -> None:
    pytest.importorskip("blosc2")
    case_id = load_fold_cases(0, "val")[0]
    from standalone_nnunet2d.data.data_source import NNUNET_PREPROCESSED_B2ND
    from standalone_nnunet2d.training.formal_dataset import FormalPatchDataset

    for index, (data, seg, message) in enumerate(
        (
        (
            np.zeros((1, 2, 4, 4), dtype=np.float32),
            np.zeros((1, 3, 4, 4), dtype=np.int16),
            "shapes",
        ),
        (
            np.zeros((1, 2, 4, 4), dtype=np.float32),
            np.full((1, 2, 4, 4), 2, dtype=np.int16),
            "labels",
        ),
        (
            np.zeros((1, 2, 4, 4), dtype=np.float32),
            np.full((1, 2, 4, 4), -2, dtype=np.int16),
            "labels",
        ),
        )
    ):
        root = tmp_path / f"invalid-{index}-{message}"
        root.mkdir()
        _save_b2nd(root / f"{case_id}.b2nd", data)
        _save_b2nd(root / f"{case_id}_seg.b2nd", seg)
        dataset = FormalPatchDataset(
            root,
            fold=0,
            split="val",
            case_ids=(case_id,),
            data_source=NNUNET_PREPROCESSED_B2ND,
            patch_size=(4, 4),
            augment=False,
        )
        with pytest.raises(ValueError, match=message):
            dataset[0]


def test_preprocessed_source_is_explicitly_lazy_about_blosc2_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from standalone_nnunet2d.data import data_source

    original_import_module = data_source.importlib.import_module
    imported: list[str] = []

    def recording_import_module(name: str) -> object:
        if name == "blosc2":
            imported.append(name)
        return original_import_module(name)

    monkeypatch.setattr(
        data_source.importlib, "import_module", recording_import_module
    )
    assert data_source.RAW_NIFTI_ONLINE
    assert "blosc2" not in imported
    root = tmp_path / "preprocessed"
    root.mkdir()
    case_id = load_fold_cases(0, "val")[0]
    data_source.PreprocessedB2ndCaseSource(
        root, fold=0, split="val", case_ids=(case_id,)
    )
    assert imported == ["blosc2"]


def test_formal_config_distinguishes_data_source_but_not_machine_root(tmp_path: Path) -> None:
    from standalone_nnunet2d.data.data_source import (
        NNUNET_PREPROCESSED_B2ND,
        RAW_NIFTI_ONLINE,
    )

    schedule = OfficialTrainerSchedule(num_iterations_per_epoch=1, num_val_iterations_per_epoch=1)
    raw = formal_train.build_formal_config(
        fold=0,
        epochs=1,
        schedule=schedule,
        data_source=RAW_NIFTI_ONLINE,
        data_root=tmp_path / "raw-a",
    )
    preprocessed = formal_train.build_formal_config(
        fold=0,
        epochs=1,
        schedule=schedule,
        data_source=NNUNET_PREPROCESSED_B2ND,
        data_root=tmp_path / "preprocessed",
    )
    preprocessed_other_root = formal_train.build_formal_config(
        fold=0,
        epochs=1,
        schedule=schedule,
        data_source=NNUNET_PREPROCESSED_B2ND,
        data_root=tmp_path / "other-machine-root",
    )

    assert raw["data_source"] == {
        "type": RAW_NIFTI_ONLINE,
        "root": str((tmp_path / "raw-a").resolve()),
    }
    assert preprocessed["data_source"]["type"] == NNUNET_PREPROCESSED_B2ND
    assert raw["plan_hash"] != preprocessed["plan_hash"]
    assert preprocessed["plan_hash"] == preprocessed_other_root["plan_hash"]


def test_formal_parser_requires_exactly_one_data_root(tmp_path: Path) -> None:
    parser = formal_train.build_parser()
    common = [
        "--output-root", str(tmp_path / "output"),
        "--plans", str(tmp_path / "plans.json"),
    ]
    with pytest.raises(SystemExit):
        parser.parse_args(common)
    with pytest.raises(SystemExit):
        parser.parse_args(common + ["--raw-root", "raw", "--preprocessed-root", "pre"])
    arguments = parser.parse_args(common + ["--preprocessed-root", "pre"])
    assert arguments.raw_root is None
    assert arguments.preprocessed_root == Path("pre")


def test_data_source_parity_audit_writes_split_and_statistics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("blosc2")
    from standalone_nnunet2d.tools.audit_data_source_parity import main

    raw_root = tmp_path / "raw"
    preprocessed_root = tmp_path / "preprocessed"
    (raw_root / "imagesTr").mkdir(parents=True)
    (raw_root / "labelsTr").mkdir()
    preprocessed_root.mkdir()
    case_ids = load_fold_cases(0, "val")[:3]
    for offset, case_id in enumerate(case_ids):
        image = (np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8) + offset)
        label = np.zeros((3, 8, 8), dtype=np.int16)
        label[offset % 3, 1, 2] = 1
        _write_raw_case(raw_root, case_id, image, label)
        raw_dataset = StrokeSliceDataset(
            raw_root, fold=0, split="val", case_ids=(case_id,)
        )
        processed_image, processed_label = raw_dataset.load_case(case_id)
        if offset == 0:
            processed_image = processed_image + np.float32(5e-7)
        raw_preprocessed_label = processed_label.copy()
        raw_preprocessed_label[0, 0, 0] = -1
        _write_preprocessed_case(
            preprocessed_root, case_id, processed_image, raw_preprocessed_label
        )

    output_path = tmp_path / "audit.json"
    assert main(
        [
            "--raw-root", str(raw_root),
            "--preprocessed-root", str(preprocessed_root),
            "--fold", "0",
            "--case-id", case_ids[0],
            "--case-id", case_ids[1],
            "--case-id", case_ids[2],
            "--output", str(output_path),
        ]
    ) == 0

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["synthetic_or_real"] == "unknown_input_provenance"
    assert [record["split"] for record in report["cases"]] == ["val"] * 3
    assert all(record["shape_match"] for record in report["cases"])
    assert all(record["image_close"] for record in report["cases"])
    assert all(record["label_equal"] for record in report["cases"])
    assert report["cases"][0]["preprocessed_raw_label_unique"] == [-1, 0, 1]
    assert report["cases"][0]["preprocessed_raw_label_counts"] == {
        "-1": 1,
        "0": 190,
        "1": 1,
    }
    assert report["image_tolerance"] == {"rtol": 1e-5, "atol": 1e-6}
    assert report["cases"][0]["image_difference"]["max_abs"] > 0.0
    assert report["cases"][0]["image_difference"]["rmse"] > 0.0
    assert "image_difference" in report["cases"][0]
    assert json.loads(capsys.readouterr().out)["audit"] == "completed"


@pytest.mark.parametrize("mismatch", ("value", "axis"))
def test_data_source_parity_audit_rejects_image_mismatch(
    tmp_path: Path, mismatch: str
) -> None:
    pytest.importorskip("blosc2")
    from standalone_nnunet2d.tools.audit_data_source_parity import main

    raw_root = tmp_path / "raw"
    preprocessed_root = tmp_path / "preprocessed"
    (raw_root / "imagesTr").mkdir(parents=True)
    (raw_root / "labelsTr").mkdir()
    preprocessed_root.mkdir()
    case_ids = load_fold_cases(0, "val")[:3]
    for offset, case_id in enumerate(case_ids):
        image = (np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8) + offset)
        label = np.zeros((3, 8, 8), dtype=np.int16)
        label[offset % 3, 1, 2] = 1
        _write_raw_case(raw_root, case_id, image, label)
        raw_dataset = StrokeSliceDataset(
            raw_root, fold=0, split="val", case_ids=(case_id,)
        )
        processed_image, processed_label = raw_dataset.load_case(case_id)
        if offset == 0:
            processed_image = (
                processed_image + 1.0
                if mismatch == "value"
                else processed_image[:, :, ::-1].copy()
            )
        _write_preprocessed_case(
            preprocessed_root, case_id, processed_image, processed_label
        )

    with pytest.raises(SystemExit):
        main(
            [
                "--raw-root", str(raw_root),
                "--preprocessed-root", str(preprocessed_root),
                "--fold", "0",
                "--case-id", case_ids[0],
                "--case-id", case_ids[1],
                "--case-id", case_ids[2],
                "--output", str(tmp_path / "audit.json"),
            ]
        )


def test_data_source_parity_audit_rejects_label_mismatch_and_input_output_overlap(
    tmp_path: Path,
) -> None:
    pytest.importorskip("blosc2")
    from standalone_nnunet2d.tools.audit_data_source_parity import main

    raw_root = tmp_path / "raw"
    preprocessed_root = tmp_path / "preprocessed"
    (raw_root / "imagesTr").mkdir(parents=True)
    (raw_root / "labelsTr").mkdir()
    preprocessed_root.mkdir()
    case_id = load_fold_cases(0, "val")[0]
    image = np.ones((2, 8, 8), dtype=np.float32)
    label = np.zeros((2, 8, 8), dtype=np.int16)
    label[1, 2, 3] = 1
    _write_raw_case(raw_root, case_id, image, label)
    raw_dataset = StrokeSliceDataset(raw_root, fold=0, split="val", case_ids=(case_id,))
    processed_image, processed_label = raw_dataset.load_case(case_id)
    processed_label = processed_label.copy()
    processed_label[0, 0, 0] = 1
    _write_preprocessed_case(preprocessed_root, case_id, processed_image, processed_label)

    with pytest.raises(SystemExit):
        main(
            [
                "--raw-root", str(raw_root),
                "--preprocessed-root", str(preprocessed_root),
                "--fold", "0",
                "--case-id", case_id,
                "--case-id", load_fold_cases(0, "val")[1],
                "--case-id", load_fold_cases(0, "val")[2],
                "--output", str(tmp_path / "audit.json"),
            ]
        )

    with pytest.raises(SystemExit):
        main(
            [
                "--raw-root", str(raw_root),
                "--preprocessed-root", str(preprocessed_root),
                "--fold", "0",
                "--case-id", case_id,
                "--case-id", case_id,
                "--case-id", load_fold_cases(0, "val")[1],
                "--output", str(raw_root / "nested" / "audit.json"),
            ]
        )
