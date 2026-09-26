"""Official PlainConvUNet single-output Trainer with early stopping."""

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import (
    nnUNetTrainerNoDeepSupervision,
)

from nnUNetTrainerMixins import EarlyStoppingMixin


class nnUNetTrainerNoDeepSupervisionEarlyStopping(EarlyStoppingMixin, nnUNetTrainerNoDeepSupervision):
    """Official decoder, official Dice/CE loss, single-output, early stopping."""
