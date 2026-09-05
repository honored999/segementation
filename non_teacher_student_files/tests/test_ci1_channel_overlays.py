import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from audit_ci1_channel_overlays import (
    channel_view_output_path,
    find_modality_segmentation_path,
    positive_slice_indices,
    select_diverse_index_rows,
)


class CI1ChannelOverlaysTest(unittest.TestCase):
    def test_output_path_uses_channel_and_view_subdirectories(self):
        path = channel_view_output_path(
            output_dir=Path("results/out"),
            channel="dwi",
            view="axial",
            patient="patient/a",
            timepoint="D1",
            index=3,
        )

        self.assertEqual(
            path,
            Path("results/out/dwi/axial/overlay_003_patient_a_D1.png"),
        )

    def test_finds_flair_segmentation_without_using_flair_adc(self):
        existing = {
            Path("CI-1/patient/patient-D7-DWI.nii.gz"),
            Path("CI-1/patient/patient-D7-FLAIR.nii.gz"),
            Path("CI-1/patient/patient-D7-FLAIR-ADC.nii.gz"),
        }

        self.assertEqual(
            find_modality_segmentation_path(
                Path("CI-1/patient/patient-D7-DWI.nii.gz"),
                "flair",
                path_exists=existing.__contains__,
                glob_paths=lambda _pattern: list(existing),
            ),
            Path("CI-1/patient/patient-D7-FLAIR.nii.gz"),
        )

    def test_selects_largest_positive_slices_for_view(self):
        mask = np.zeros((3, 4, 5), dtype=bool)
        mask[0, 1:3, 1:3] = True
        mask[2, 1:4, 1:4] = True

        self.assertEqual(positive_slice_indices(mask, view="axial", max_slices=1), [2])

    def test_selects_different_patients_before_repeating_patient(self):
        rows = [
            SimpleNamespace(patient="A", timepoint="D1"),
            SimpleNamespace(patient="A", timepoint="D2"),
            SimpleNamespace(patient="B", timepoint="D1"),
            SimpleNamespace(patient="C", timepoint="D1"),
            SimpleNamespace(patient="B", timepoint="D2"),
        ]

        selected = select_diverse_index_rows(rows, max_cases=4)

        self.assertEqual(
            [(row.patient, row.timepoint) for row in selected],
            [("A", "D1"), ("B", "D1"), ("C", "D1"), ("A", "D2")],
        )


if __name__ == "__main__":
    unittest.main()
