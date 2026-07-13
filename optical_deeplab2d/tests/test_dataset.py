from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from optical_deeplab2d.datasets.dataset_2d import SampleRecord, load_sample
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

