"""Plain nnU-Net with configurable Tversky + CE and EMA early stopping."""

from nnUNetTrainerEarlyStopping import nnUNetTrainerEarlyStopping
from tversky_ce import TverskyCELossMixin


class nnUNetTrainerTverskyCEEarlyStopping(TverskyCELossMixin, nnUNetTrainerEarlyStopping):
    """Compose the weighted loss with the standard network and early stopping."""
