from __future__ import annotations

import numpy as np
import pytest

from standalone_nnunet2d.data.nifti_io import NiftiVolume
from standalone_nnunet2d.data.preprocessing import resample_inplane, z_score_normalize


def test_z_score_normalize_returns_zero_mean_unit_variance() -> None:
    normalized = z_score_normalize(np.array([1.0, 2.0, 3.0], dtype=np.float32))

    assert normalized.mean() == pytest.approx(0.0, abs=1e-6)
    assert normalized.std() == pytest.approx(1.0, abs=1e-6)


def test_inplane_resampling_preserves_discrete_segmentation_labels() -> None:
    segmentation = NiftiVolume(
        array=np.array([[[0, 1], [1, 0]]], dtype=np.int16),
        spacing_xyz=(1.0, 1.0, 5.0),
        origin_xyz=(0.0, 0.0, 0.0),
    )

    result = resample_inplane(segmentation, target_spacing_xy=(0.5, 0.5), is_segmentation=True)

    assert result.array.shape == (1, 4, 4)
    assert set(np.unique(result.array)).issubset({0, 1})
    assert result.array.dtype == np.int16
