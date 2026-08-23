"""Stage 2 nnU-Net Trainer using the shallow residual FCN architecture."""

from __future__ import annotations

from shallow_res_fcn_2d import ShallowResFCN2D
from nnUNetTrainerStage2Base import nnUNetTrainerStage2Base


class nnUNetTrainerStage2ShallowResFCN(nnUNetTrainerStage2Base):
    network_class = ShallowResFCN2D
