import unittest

import torch

from train_ci1_dwi_unet_teacher import CI1DwiUNetTeacher


class CI1DwiUNetTeacherTest(unittest.TestCase):
    def test_teacher_output_matches_input_spatial_size(self):
        model = CI1DwiUNetTeacher(in_channels=1, out_channels=1, base_channels=8)
        images = torch.randn(2, 1, 64, 80)

        logits = model(images)

        self.assertEqual(tuple(logits.shape), (2, 1, 64, 80))


if __name__ == "__main__":
    unittest.main()
