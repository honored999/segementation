"""Compose the existing TopK10 loss with EMA early stopping."""

from nnUNetTrainerEarlyStopping import nnUNetTrainerEarlyStopping
from nnUNetTrainerTopK10 import nnUNetTrainerTopK10


class nnUNetTrainerTopK10EarlyStopping(nnUNetTrainerEarlyStopping, nnUNetTrainerTopK10):
    """Original TopK10 Trainer with inherited persistent early-stopping control."""
