"""Official nnU-Net Trainer variant with 50% foreground-patch oversampling."""

from __future__ import annotations

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerForeground50(nnUNetTrainer):
    """Keep official training behavior while raising foreground oversampling."""

    OVERSAMPLE_FOREGROUND_PERCENT = 0.50

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.oversample_foreground_percent = self.OVERSAMPLE_FOREGROUND_PERCENT
