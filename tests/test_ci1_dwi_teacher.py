import unittest

import torch
import torch.nn as nn

from train_ci1_dwi_unet_teacher import CI1DwiUNetTeacher, UpBlock


class CI1DwiUNetTeacherTest(unittest.TestCase):
    def test_teacher_output_matches_input_spatial_size(self):
        model = CI1DwiUNetTeacher(in_channels=1, out_channels=1, base_channels=8)
        images = torch.randn(2, 1, 64, 80)

        logits = model(images)

        self.assertEqual(tuple(logits.shape), (2, 1, 64, 80))

    def test_teacher_upblock_uses_upsample_conv_decoder(self):
        block = UpBlock(in_channels=32, skip_channels=16, out_channels=16)

        self.assertIsInstance(block.up, nn.Sequential)
        self.assertFalse(any(isinstance(layer, nn.ConvTranspose2d) for layer in block.modules()))
        self.assertTrue(any(isinstance(layer, nn.Upsample) for layer in block.up.modules()))
        self.assertTrue(any(isinstance(layer, nn.Conv2d) for layer in block.up.modules()))


if __name__ == "__main__":
    unittest.main()
