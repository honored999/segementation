"""Stage 2 nnU-Net Trainer using the fixed small 2D U-Net architecture."""

from __future__ import annotations

from small_unet_2d import SmallUNet2D
from nnUNetTrainerStage2Base import nnUNetTrainerStage2Base


class nnUNetTrainerStage2SmallUNet(nnUNetTrainerStage2Base):
    network_class = SmallUNet2D
