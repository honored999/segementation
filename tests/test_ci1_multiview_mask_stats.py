import unittest

import numpy as np

from analyze_ci1_multiview_mask_stats import summarize_mask_volume


class CI1MultiviewMaskStatsTest(unittest.TestCase):
    def test_summarize_mask_volume_counts_three_views(self):
        mask = np.zeros((3, 4, 5), dtype=bool)
        mask[1, 1:3, 2:5] = True
        mask[2, 2, 3] = True

        stats = summarize_mask_volume(mask)

        self.assertEqual(stats["axial"].positive_slices, 2)
        self.assertEqual(stats["axial"].max_mask_area, 6)
        self.assertEqual(stats["coronal"].positive_slices, 2)
        self.assertEqual(stats["sagittal"].positive_slices, 3)
        self.assertEqual(stats["sagittal"].max_mask_area, 3)


if __name__ == "__main__":
    unittest.main()
