"""Stage 2 Trainer with foreground oversampling and FN-weighted Tversky loss."""

from __future__ import annotations

from nnUNetTrainerTverskyForeground50 import nnUNetTrainerTverskyForeground50


class nnUNetTrainerStage2TverskyForeground50(nnUNetTrainerTverskyForeground50):
    """Foreground50 sampling with a Stage 2 FN-weighted Tversky loss."""

    OVERSAMPLE_FOREGROUND_PERCENT = 0.50
    FP_ALPHA = 0.4
    FN_BETA = 0.6
