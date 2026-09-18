"""Official nnU-Net Trainer variant with binary grouped TopK10 CE."""

from __future__ import annotations

import numpy as np
import torch

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

from grouped_topk_loss import DC_and_grouped_topk_loss
from nnUNetTrainerTopK10 import nnUNetTrainerTopK10


class nnUNetTrainerGroupedTopK10(nnUNetTrainerTopK10):
    """Use per-image, equally weighted foreground/background TopK10 CE."""

    def _build_loss(self):
        if self.label_manager.has_regions:
            raise RuntimeError(
                "nnUNetTrainerGroupedTopK10 does not support region-based training"
            )
        if self.label_manager.all_labels != [0, 1]:
            raise RuntimeError(
                "nnUNetTrainerGroupedTopK10 supports only binary labels [0, 1]; "
                f"got {self.label_manager.all_labels}"
            )

        loss = DC_and_grouped_topk_loss(
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
