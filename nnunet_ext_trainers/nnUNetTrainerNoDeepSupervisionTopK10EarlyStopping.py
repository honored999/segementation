"""Official PlainConvUNet single-output TopK10 Trainer with early stopping."""

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import (
    nnUNetTrainerNoDeepSupervision,
)

from nnUNetTrainerMixins import EarlyStoppingMixin, TopK10LossMixin


class nnUNetTrainerNoDeepSupervisionTopK10EarlyStopping(
    EarlyStoppingMixin,
    TopK10LossMixin,
    nnUNetTrainerNoDeepSupervision,
):
    """Official decoder, Dice/TopK10, single-output, early stopping."""
