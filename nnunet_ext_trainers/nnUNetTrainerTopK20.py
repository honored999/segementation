"""Official nnU-Net Trainer variant with Dice plus TopK20 cross-entropy."""

from __future__ import annotations

from nnUNetTrainerTopK10 import nnUNetTrainerTopK10


class nnUNetTrainerTopK20(nnUNetTrainerTopK10):
    """Use the TopK10 experiment contract with the hardest 20% of pixels."""

    TOPK_PERCENT = 20
