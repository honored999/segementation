import unittest

from convert_ci1_dwi_to_2d import DicomSliceInfo, select_unique_slice_files


class DicomSliceSelectionTest(unittest.TestCase):
    def test_select_unique_slice_files_keeps_one_file_per_position(self):
        slices = [
            DicomSliceInfo("z0_b0.dcm", 0.0, 1),
            DicomSliceInfo("z0_b1000.dcm", 0.0, 2),
            DicomSliceInfo("z1_b0.dcm", 8.0, 3),
            DicomSliceInfo("z1_b1000.dcm", 8.0, 4),
        ]

        selected = select_unique_slice_files(slices)

        self.assertEqual(selected, ["z0_b0.dcm", "z1_b0.dcm"])

    def test_select_unique_slice_files_sorts_by_position_before_selecting(self):
        slices = [
            DicomSliceInfo("z2.dcm", 16.0, 3),
            DicomSliceInfo("z0.dcm", 0.0, 1),
            DicomSliceInfo("z1.dcm", 8.0, 2),
        ]

        selected = select_unique_slice_files(slices)

        self.assertEqual(selected, ["z0.dcm", "z1.dcm", "z2.dcm"])


if __name__ == "__main__":
    unittest.main()
