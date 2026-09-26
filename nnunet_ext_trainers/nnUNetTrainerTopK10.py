"""Official nnU-Net Trainer variant with Dice plus TopK cross-entropy."""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from nnUNetTrainerMixins import TopK10LossMixin


class nnUNetTrainerTopK10(TopK10LossMixin, nnUNetTrainer):
    """Use official Dice plus TopK CE with the hardest 10 percent of pixels."""
