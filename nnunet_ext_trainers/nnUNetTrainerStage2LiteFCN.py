"""Stage 2 nnU-Net Trainer using the fixed 2D LiteFCN architecture."""

from __future__ import annotations

from lite_fcn_2d import LiteFCN2D
from nnUNetTrainerStage2Base import nnUNetTrainerStage2Base


class nnUNetTrainerStage2LiteFCN(nnUNetTrainerStage2Base):
    network_class = LiteFCN2D
