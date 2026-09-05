import unittest
from pathlib import Path

import numpy as np

from convert_ci1_adc_to_2d import (
    extract_view_slice,
    find_adc_segmentation_path,
    mask_area_by_view,
)


class ConvertCI1AdcTo2DTest(unittest.TestCase):
    def test_finds_direct_adc_mask_from_dwi_mask_path(self):
        existing = {
            Path("CI-1/patient/patient-D3-DWI.nii.gz"),
            Path("CI-1/patient/patient-D3-ADC.nii.gz"),
        }

        self.assertEqual(
            find_adc_segmentation_path(
                Path("CI-1/patient/patient-D3-DWI.nii.gz"),
                path_exists=existing.__contains__,
                glob_paths=lambda _pattern: list(existing),
            ),
            Path("CI-1/patient/patient-D3-ADC.nii.gz"),
        )

    def test_extracts_axial_slice_from_zyx_volume(self):
        volume = np.arange(2 * 3 * 4).reshape(2, 3, 4)

        np.testing.assert_array_equal(
            extract_view_slice(volume, view="axial", index=1),
            volume[1, :, :],
        )

    def test_mask_area_by_axial_view(self):
        mask = np.zeros((2, 3, 4), dtype=bool)
        mask[0, 1, 2:4] = True
        mask[1, :, 0] = True

        np.testing.assert_array_equal(mask_area_by_view(mask, "axial"), [2, 3])


if __name__ == "__main__":
    unittest.main()
