from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

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
