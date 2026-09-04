"""Official nnU-Net Trainer variant with Dice plus TopK cross-entropy."""

from __future__ import annotations

import numpy as np
import torch

from nnunetv2.training.loss.compound_losses import DC_and_topk_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerTopK10(nnUNetTrainer):
    """Use official Dice plus TopK CE with the hardest 10 percent of pixels."""

    TOPK_PERCENT = 10

    def _build_loss(self):
        if self.label_manager.has_regions:
            return super()._build_loss()

        loss = DC_and_topk_loss(
            {
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5,
                "do_bg": False,
                "ddp": self.is_ddp,
            },
            {"k": self.TOPK_PERCENT},
            weight_ce=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
        )
        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            weights = np.array(
                [1 / (2**index) for index in range(len(self._get_deep_supervision_scales()))]
            )
            weights[-1] = 1e-6 if self.is_ddp and not self._do_i_compile() else 0
            loss = DeepSupervisionWrapper(loss, weights / weights.sum())
        return loss
