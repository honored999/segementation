from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from optical_deeplab2d.datasets.dataset_2d import (
    DwiSliceDataset,
    SampleRecord,
    fit_percentile_normalizer,
    load_sample,
)
from optical_deeplab2d.datasets.split import build_patient_folds


def test_load_sample_returns_binary_single_channel_tensors(tmp_path: Path) -> None:
    image = tmp_path / "image.png"; mask = tmp_path / "mask.png"
    Image.fromarray(np.full((4, 5), 128, dtype=np.uint8)).save(image)
    Image.fromarray(np.array([[0, 2, 0, 0, 0]] * 4, dtype=np.uint8)).save(mask)
    loaded_image, loaded_mask = load_sample(SampleRecord("p", "D1", image, mask, 1))
    assert tuple(loaded_image.shape) == (1, 4, 5)
    assert set(loaded_mask.unique().tolist()) <= {0.0, 1.0}


def test_patient_folds_keep_all_timepoints_together() -> None:
    rows = [SampleRecord(patient, time, Path("a"), Path("b"), 0)
            for patient in ["p1", "p2", "p3", "p4", "p5"] for time in ["D1", "D2"]]
    folds = build_patient_folds(rows, seed=2026, n_splits=5)
    assert len(folds) == 5
    for fold in folds:
        assert not set(fold.train_patients) & set(fold.val_patients)


def test_percentile_normalizer_uses_training_records_and_clips_validation_values(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train.npy"
    validation_path = tmp_path / "validation.npy"
    mask_path = tmp_path / "mask.npy"
    np.save(train_path, np.array([[0, 10], [20, 30]], dtype=np.uint16))
    np.save(validation_path, np.array([[0, 15], [30, 60]], dtype=np.uint16))
    np.save(mask_path, np.zeros((2, 2), dtype=np.uint8))
    train_record = SampleRecord("train", "D1", train_path, mask_path, 0)
    validation_record = SampleRecord("validation", "D1", validation_path, mask_path, 0)

    normalizer = fit_percentile_normalizer(
        [train_record], lower_percentile=0, upper_percentile=100
    )
    image, _, _ = DwiSliceDataset([validation_record], normalizer=normalizer)[0]

    assert normalizer.lower == 0
    assert normalizer.upper == 30
    assert torch.allclose(image[0], torch.tensor([[0.0, 0.5], [1.0, 1.0]]))
