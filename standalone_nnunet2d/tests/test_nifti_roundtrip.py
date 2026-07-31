from __future__ import annotations

from pathlib import Path

import numpy as np

from standalone_nnunet2d.data.nifti_io import NiftiVolume, read_nifti, write_nifti


def test_nifti_round_trip_preserves_array_and_metadata(tmp_path: Path) -> None:
    source = NiftiVolume(
        array=np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        spacing_xyz=(0.5, 0.6, 5.0),
        origin_xyz=(1.0, 2.0, 3.0),
    )
    path = tmp_path / "case001_0000.nii.gz"

    write_nifti(path, source)
    restored = read_nifti(path)

    np.testing.assert_array_equal(restored.array, source.array)
    np.testing.assert_allclose(restored.spacing_xyz, source.spacing_xyz, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(restored.origin_xyz, source.origin_xyz, rtol=0.0, atol=1e-6)
