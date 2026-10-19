"""Compose the existing TopK10 loss and early-stopping lifecycle."""

from nnUNetTrainerEarlyStopping import nnUNetTrainerEarlyStopping
from nnUNetTrainerTopK10 import nnUNetTrainerTopK10


class nnUNetTrainerTopK10EarlyStopping(
    nnUNetTrainerEarlyStopping, nnUNetTrainerTopK10
):
    """Keep ordinary deep supervision and both parent contracts unchanged."""
