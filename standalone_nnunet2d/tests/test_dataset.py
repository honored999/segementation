from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

from standalone_nnunet2d import overfit_one_case
from standalone_nnunet2d.data import dataset as dataset_module
from standalone_nnunet2d.data.dataset import StrokeSliceDataset, load_fold_cases
from standalone_nnunet2d.data.augmentation import AugmentationConfig
from standalone_nnunet2d.data.nifti_io import NiftiVolume, write_nifti
from standalone_nnunet2d.data.sampling import select_axial_slice, select_slice_index


def test_fold_cases_are_loaded_from_the_supplied_fixed_splits() -> None:
    train_cases = load_fold_cases(fold=0, split="train")
    validation_cases = load_fold_cases(fold=0, split="val")

    assert len(train_cases) == 76
    assert len(validation_cases) == 19
    assert set(train_cases).isdisjoint(validation_cases)


def test_select_axial_slice_checks_bounds() -> None:
    volume = np.zeros((3, 8, 8), dtype=np.float32)

    with pytest.raises(IndexError, match="slice index"):
        select_axial_slice(volume, 3)


def test_foreground_sampler_chooses_foreground_slice_when_probability_is_one() -> None:
    labels = np.zeros((3, 2, 2), dtype=np.int16)
    labels[2, 0, 0] = 1

    assert select_slice_index(labels, np.random.default_rng(7), foreground_probability=1.0) == 2


def test_foreground_sampler_falls_back_to_valid_index_without_foreground() -> None:
    result = select_slice_index(np.zeros((3, 2, 2), dtype=np.int16), np.random.default_rng(7), foreground_probability=1.0)

    assert 0 <= result < 3


def test_dataset_loads_one_requested_fixed_fold_case_on_demand(tmp_path: Path) -> None:
    case_id = load_fold_cases(fold=0, split="val")[0]
    image = NiftiVolume(np.arange(24, dtype=np.float32).reshape(2, 3, 4), (1.0, 1.0, 5.0), (0.0, 0.0, 0.0))
    label = NiftiVolume(np.zeros((2, 3, 4), dtype=np.int16), (1.0, 1.0, 5.0), (0.0, 0.0, 0.0))
    write_nifti(tmp_path / "imagesTr" / f"{case_id}_0000.nii.gz", image)
    write_nifti(tmp_path / "labelsTr" / f"{case_id}.nii.gz", label)

    dataset = StrokeSliceDataset(tmp_path, fold=0, split="val", case_ids=(case_id,), target_spacing_xy=(1.0, 1.0))
    normalized_image, _ = dataset.load_case(case_id)
    image_slice, label_slice = dataset[0]

    assert tuple(image_slice.shape) == (1, 3, 4)
    assert tuple(label_slice.shape) == (3, 4)
    assert normalized_image.mean() == pytest.approx(0.0, abs=1e-6)
    assert len(dataset) == 1


def test_dataset_can_opt_into_foreground_sampling(tmp_path: Path) -> None:
    case_id = load_fold_cases(fold=0, split="val")[0]
    image = NiftiVolume(np.arange(24, dtype=np.float32).reshape(2, 3, 4), (1.0, 1.0, 5.0), (0.0, 0.0, 0.0))
    label_array = np.zeros((2, 3, 4), dtype=np.int16)
    label_array[1, 0, 0] = 1
    label = NiftiVolume(label_array, (1.0, 1.0, 5.0), (0.0, 0.0, 0.0))
    write_nifti(tmp_path / "imagesTr" / f"{case_id}_0000.nii.gz", image)
    write_nifti(tmp_path / "labelsTr" / f"{case_id}.nii.gz", label)

    dataset = StrokeSliceDataset(
        tmp_path, fold=0, split="val", case_ids=(case_id,), target_spacing_xy=(1.0, 1.0),
        rng=np.random.default_rng(3), foreground_probability=1.0, augmentation_config=AugmentationConfig(),
    )
    _, label_slice = dataset[0]

    assert label_slice[0, 0].item() == 1


def test_dataset_uses_strict_declared_channel_order_and_normalizes_each_channel_independently(
    tmp_path: Path,
) -> None:
    case_id = load_fold_cases(fold=0, split="val")[0]
    (tmp_path / "dataset.json").write_text(
        '{"channel_names": {"0": "DWI", "1": "ADC"}}', encoding="utf-8"
    )
    first = NiftiVolume(
        np.arange(24, dtype=np.float32).reshape(2, 3, 4), (1.0, 1.0, 5.0), (0.0, 0.0, 0.0)
    )
    second_array = (100.0 + np.array(
        [0, 3, 1, 7, 2, 9, 4, 6, 8, 5, 11, 10, 14, 13, 15, 12, 18, 17, 16, 20, 23, 19, 22, 21],
        dtype=np.float32,
    )).reshape(2, 3, 4)
    second = NiftiVolume(second_array, (1.0, 1.0, 5.0), (0.0, 0.0, 0.0))
    label = NiftiVolume(np.zeros((2, 3, 4), dtype=np.int16), (1.0, 1.0, 5.0), (0.0, 0.0, 0.0))
    write_nifti(tmp_path / "imagesTr" / f"{case_id}_0000.nii.gz", first)
    write_nifti(tmp_path / "imagesTr" / f"{case_id}_0001.nii.gz", second)
    write_nifti(tmp_path / "labelsTr" / f"{case_id}.nii.gz", label)

    assert dataset_module.resolve_channel_specs(tmp_path) == ((0, "DWI"), (1, "ADC"))
    dataset = StrokeSliceDataset(
        tmp_path, fold=0, split="val", case_ids=(case_id,), target_spacing_xy=(1.0, 1.0)
    )
    normalized, _ = dataset.load_case(case_id)

    assert normalized.shape == (2, 2, 3, 4)
    np.testing.assert_allclose(
        normalized[0], (first.array - first.array.mean()) / first.array.std(), atol=1e-6
    )
    np.testing.assert_allclose(
        normalized[1], (second_array - second_array.mean()) / second_array.std(), atol=1e-6
    )
    np.testing.assert_allclose(normalized[0].mean(), 0.0, atol=1e-6)
    np.testing.assert_allclose(normalized[1].mean(), 0.0, atol=1e-6)
    np.testing.assert_allclose(normalized[0].std(), 1.0, atol=1e-6)
    np.testing.assert_allclose(normalized[1].std(), 1.0, atol=1e-6)
    image_slice, label_slice = dataset[0]
    assert tuple(image_slice.shape) == (2, 3, 4)
    assert tuple(label_slice.shape) == (3, 4)


def test_dataset_resolves_numeric_channel_names_independent_of_json_insertion_order(tmp_path: Path) -> None:
    (tmp_path / "dataset.json").write_text(
        '{"channel_names": {"1": "ADC", "0": "DWI"}}', encoding="utf-8"
    )

    assert dataset_module.resolve_channel_specs(tmp_path) == ((0, "DWI"), (1, "ADC"))


def test_dataset_rejects_nonconsecutive_declared_channels(tmp_path: Path) -> None:
    (tmp_path / "dataset.json").write_text(
        '{"channel_names": {"0": "DWI", "2": "FLAIR"}}', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="channel_names.*0.*1.*2"):
        dataset_module.resolve_channel_specs(tmp_path)


def test_dataset_reports_missing_declared_channel_with_case_and_reason(tmp_path: Path) -> None:
    case_id = load_fold_cases(fold=0, split="val")[0]
    (tmp_path / "dataset.json").write_text(
        '{"channel_names": {"0": "DWI", "1": "ADC"}}', encoding="utf-8"
    )
    image = NiftiVolume(np.zeros((1, 2, 2), dtype=np.float32), (1, 1, 1), (0, 0, 0))
    label = NiftiVolume(np.zeros((1, 2, 2), dtype=np.int16), (1, 1, 1), (0, 0, 0))
    write_nifti(tmp_path / "imagesTr" / f"{case_id}_0000.nii.gz", image)
    write_nifti(tmp_path / "labelsTr" / f"{case_id}.nii.gz", label)

    dataset = StrokeSliceDataset(tmp_path, fold=0, split="val", case_ids=(case_id,))
    with pytest.raises(FileNotFoundError, match=f"case {case_id}.*channel 1.*ADC.*missing"):
        dataset.load_case(case_id)


def test_dataset_reports_channel_geometry_mismatch_with_reason(tmp_path: Path) -> None:
    case_id = load_fold_cases(fold=0, split="val")[0]
    (tmp_path / "dataset.json").write_text(
        '{"channel_names": {"0": "DWI", "1": "ADC"}}', encoding="utf-8"
    )
    channel0 = NiftiVolume(np.zeros((1, 2, 2), dtype=np.float32), (1, 1, 1), (0, 0, 0))
    channel1 = NiftiVolume(np.zeros((1, 2, 2), dtype=np.float32), (2, 1, 1), (0, 0, 0))
    label = NiftiVolume(np.zeros((1, 2, 2), dtype=np.int16), (1, 1, 1), (0, 0, 0))
    write_nifti(tmp_path / "imagesTr" / f"{case_id}_0000.nii.gz", channel0)
    write_nifti(tmp_path / "imagesTr" / f"{case_id}_0001.nii.gz", channel1)
    write_nifti(tmp_path / "labelsTr" / f"{case_id}.nii.gz", label)

    dataset = StrokeSliceDataset(tmp_path, fold=0, split="val", case_ids=(case_id,))
    with pytest.raises(ValueError, match=f"case {case_id}.*channel 1.*spacing"):
        dataset.load_case(case_id)


def test_overfit_one_case_builds_bchw_batches_and_uses_3d_prediction_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_id = load_fold_cases(fold=0, split="train")[0]
    image = NiftiVolume(
        np.arange(24, dtype=np.float32).reshape(2, 3, 4), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)
    )
    label_array = np.zeros((2, 3, 4), dtype=np.int16)
    label_array[1, 1, 2] = 1
    label = NiftiVolume(label_array, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    write_nifti(tmp_path / "imagesTr" / f"{case_id}_0000.nii.gz", image)
    write_nifti(tmp_path / "labelsTr" / f"{case_id}.nii.gz", label)

    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    references: list[NiftiVolume] = []

    class DummyModel(torch.nn.Module):
        def __init__(self, config) -> None:
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.zeros(1))

    monkeypatch.setattr(overfit_one_case, "PlainConvUNet2D", DummyModel)
    monkeypatch.setattr(
        overfit_one_case,
        "train_step",
        lambda model, batch, loss_fn, optimizer, device: (
            batches.append(batch) or SimpleNamespace(loss=0.0, output_shapes=())
        ),
    )

    def fake_predict_volume(model, reference, device):
        references.append(reference)
        return np.zeros(reference.array.shape, dtype=np.uint8)

    monkeypatch.setattr(overfit_one_case, "predict_volume", fake_predict_volume)
    monkeypatch.setattr(overfit_one_case, "save_and_validate_prediction", lambda *args, **kwargs: {})
    monkeypatch.setattr(overfit_one_case, "write_overlay", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", [
        "overfit_one_case.py",
        "--raw-root", str(tmp_path),
        "--output-root", str(tmp_path / "outputs"),
        "--case-id", case_id,
        "--device", "cpu",
        "--iterations", "1",
    ])

    assert overfit_one_case.main() == 0
    assert batches
    assert tuple(batches[0][0].shape) == (1, 1, 3, 4)
    assert tuple(batches[0][1].shape) == (1, 3, 4)
    assert references
    assert all(reference.array.ndim == 3 for reference in references)
