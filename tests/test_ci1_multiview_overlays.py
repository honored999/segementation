import unittest

import numpy as np

from make_ci1_multiview_overlays import IMAGE_ASPECT, select_largest_mask_slice


class CI1MultiviewOverlaysTest(unittest.TestCase):
    def test_uses_true_image_aspect_by_default(self):
        self.assertEqual(IMAGE_ASPECT, "equal")

    def test_select_largest_mask_slice_for_each_view(self):
        axial_mask = np.zeros((3, 4, 5), dtype=bool)
        axial_mask[2, 1:4, 1:4] = True

        coronal_mask = np.zeros((3, 4, 5), dtype=bool)
        coronal_mask[:, 3, 1:4] = True

        sagittal_mask = np.zeros((3, 4, 5), dtype=bool)
        sagittal_mask[:, 1:4, 4] = True

        self.assertEqual(select_largest_mask_slice(axial_mask, "axial"), 2)
        self.assertEqual(select_largest_mask_slice(coronal_mask, "coronal"), 3)
        self.assertEqual(select_largest_mask_slice(sagittal_mask, "sagittal"), 4)


if __name__ == "__main__":
    unittest.main()
