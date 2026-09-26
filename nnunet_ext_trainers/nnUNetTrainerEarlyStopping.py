"""Official nnU-Net 2.8.1 Trainer with EMA pseudo-Dice early stopping."""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from nnUNetTrainerMixins import EarlyStoppingMixin


class nnUNetTrainerEarlyStopping(EarlyStoppingMixin, nnUNetTrainer):
    """Keep the official Trainer and add only cooperative early stopping."""
