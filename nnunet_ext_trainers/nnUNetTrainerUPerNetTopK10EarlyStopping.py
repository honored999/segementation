"""Official PlainConvUNet encoder with UPerNet, TopK10 and early stopping."""

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import (
    nnUNetTrainerNoDeepSupervision,
)

from nnUNetTrainerMixins import EarlyStoppingMixin, TopK10LossMixin, UPerNetArchitectureMixin


class nnUNetTrainerUPerNetTopK10EarlyStopping(
    EarlyStoppingMixin,
    TopK10LossMixin,
    UPerNetArchitectureMixin,
    nnUNetTrainerNoDeepSupervision,
):
    """Official PlainConvUNet encoder, UPerNet, Dice/TopK10, early stopping."""
