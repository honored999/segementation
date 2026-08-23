"""Stage 2 nnU-Net Trainer using the fixed 2D Lightweight ASPP architecture."""

from __future__ import annotations

from lightweight_aspp_2d import LightweightASPP2D
from nnUNetTrainerStage2Base import nnUNetTrainerStage2Base


class nnUNetTrainerStage2LightweightASPP(nnUNetTrainerStage2Base):
    network_class = LightweightASPP2D
