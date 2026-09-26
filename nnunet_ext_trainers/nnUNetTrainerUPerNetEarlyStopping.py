"""Official PlainConvUNet encoder with UPerNet and early stopping."""

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import (
    nnUNetTrainerNoDeepSupervision,
)

from nnUNetTrainerMixins import EarlyStoppingMixin, UPerNetArchitectureMixin


class nnUNetTrainerUPerNetEarlyStopping(
    EarlyStoppingMixin,
    UPerNetArchitectureMixin,
    nnUNetTrainerNoDeepSupervision,
):
    """Official PlainConvUNet encoder, UPerNet decoder, Dice/CE, early stopping."""
