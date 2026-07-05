import unittest
from pathlib import Path

import numpy as np

from audit_ci1_dwi_adc_masks import (
    extract_view_slice,
    find_adc_segmentation_path,
    image_extent,
    is_adc_series_description,
    is_dwi_series_description,
    is_flair_series_description,
    positive_slice_indices,
)


class CI1DwiAdcMaskAuditTest(unittest.TestCase):
    def test_identifies_dwi_and_adc_series_descriptions(self):
        self.assertTrue(is_dwi_series_description("Axi FSE-DWI"))
        self.assertFalse(is_dwi_series_description("ADC from 2"))

        self.assertTrue(is_adc_series_description("ADC from 2"))
        self.assertFalse(is_adc_series_description("Axi FSE-DWI"))

        self.assertTrue(is_flair_series_description("Axi FLAIR_PS T2"))
        self.assertFalse(is_flair_series_description("T1-Flair-short"))

    def test_finds_adc_segmentation_next_to_dwi_segmentation(self):
        existing = {
            Path("CI-1/patient/patient-D7-DWI.nii.gz"),
            Path("CI-1/patient/patient-D7-ADC.nii.gz"),
        }

        self.assertEqual(
            find_adc_segmentation_path(
                Path("CI-1/patient/patient-D7-DWI.nii.gz"),
                path_exists=existing.__contains__,
            ),
            Path("CI-1/patient/patient-D7-ADC.nii.gz"),
        )

    def test_extracts_coronal_and_sagittal_slices(self):
        volume = np.arange(2 * 3 * 4).reshape(2, 3, 4)

        np.testing.assert_array_equal(extract_view_slice(volume, "axial", 1), volume[1, :, :])
        np.testing.assert_array_equal(extract_view_slice(volume, "coronal", 2), volume[:, 2, :])
        np.testing.assert_array_equal(extract_view_slice(volume, "sagittal", 3), volume[:, :, 3])

    def test_selects_positive_slices_for_requested_view(self):
        dwi_mask = np.zeros((2, 3, 4), dtype=bool)
        adc_mask = np.zeros((2, 3, 4), dtype=bool)
        dwi_mask[:, 2, 1:3] = True
        adc_mask[:, 1, 3] = True

        self.assertEqual(positive_slice_indices(dwi_mask, adc_mask, max_slices=4, view="coronal"), [1, 2])

    def test_image_extent_uses_physical_spacing(self):
        self.assertEqual(image_extent((2, 4), (0.5, 3.0)), [0.0, 2.0, 6.0, 0.0])


if __name__ == "__main__":
    unittest.main()
