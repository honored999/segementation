import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from audit_ci1_channel_pair_differences import (
    build_difference_overlay,
    channel_pair_output_path,
    select_diverse_ranked_samples,
)


class CI1ChannelPairDifferencesTest(unittest.TestCase):
    def test_output_path_uses_pair_view_kind_subdirectories(self):
        path = channel_pair_output_path(
            output_dir=Path("results/root"),
            pair_name="adcvsdwi",
            view="coronal",
            kind="mask",
            index=7,
            patient="patient/a",
            timepoint="D3",
            slice_index=12,
        )

        self.assertEqual(
            path,
            Path("results/root/adcvsdwi/coronal/mask/compare_007_patient_a_D3_s12.png"),
        )

    def test_difference_overlay_marks_left_and_right_only_regions(self):
        left = np.zeros((4, 4), dtype=bool)
        right = np.zeros((4, 4), dtype=bool)
        left[1, 1] = True
        right[2, 2] = True

        overlay, diff = build_difference_overlay(
            base=np.zeros((4, 4), dtype=np.float32),
            left_mask=left,
            right_mask=right,
        )

        self.assertTrue(diff[1, 1])
        self.assertTrue(diff[2, 2])
        self.assertGreater(overlay[1, 1, 0], overlay[1, 1, 1])
        self.assertGreater(overlay[2, 2, 1], overlay[2, 2, 0])

    def test_selects_different_patients_before_repeating_patient(self):
        samples = [
            SimpleNamespace(patient="A", timepoint="D1", score=10),
            SimpleNamespace(patient="A", timepoint="D2", score=90),
            SimpleNamespace(patient="B", timepoint="D1", score=50),
            SimpleNamespace(patient="C", timepoint="D1", score=40),
            SimpleNamespace(patient="B", timepoint="D2", score=80),
        ]

        selected = select_diverse_ranked_samples(samples, max_samples=4)

        self.assertEqual(
            [(sample.patient, sample.timepoint) for sample in selected],
            [("A", "D2"), ("B", "D2"), ("C", "D1"), ("A", "D1")],
        )


if __name__ == "__main__":
    unittest.main()
