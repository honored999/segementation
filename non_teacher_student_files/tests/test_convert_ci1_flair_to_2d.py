import unittest
from pathlib import Path

import numpy as np

from convert_ci1_flair_to_2d import (
    extract_view_slice,
    find_flair_segmentation_path,
    mask_area_by_view,
)


class ConvertCI1FlairTo2DTest(unittest.TestCase):
    def test_finds_direct_flair_mask_before_flair_adc(self):
        existing = {
            Path("CI-1/patient/patient-D3-DWI.nii.gz"),
            Path("CI-1/patient/patient-D3-FLAIR.nii.gz"),
            Path("CI-1/patient/patient-D3-FLAIR-ADC.nii.gz"),
        }

        self.assertEqual(
            find_flair_segmentation_path(
                Path("CI-1/patient/patient-D3-DWI.nii.gz"),
                path_exists=existing.__contains__,
                glob_paths=lambda _pattern: list(existing),
            ),
            Path("CI-1/patient/patient-D3-FLAIR.nii.gz"),
        )

    def test_extracts_coronal_slice_from_zyx_volume(self):
        volume = np.arange(2 * 3 * 4).reshape(2, 3, 4)

        np.testing.assert_array_equal(
            extract_view_slice(volume, view="coronal", index=1),
            volume[:, 1, :],
        )

    def test_mask_area_by_coronal_view(self):
        mask = np.zeros((2, 3, 4), dtype=bool)
        mask[:, 1, 2:4] = True
        mask[0, 2, 0] = True

        np.testing.assert_array_equal(mask_area_by_view(mask, "coronal"), [0, 4, 1])


if __name__ == "__main__":
    unittest.main()
