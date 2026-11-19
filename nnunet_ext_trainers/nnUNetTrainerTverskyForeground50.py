"""Stage 1 Trainer with foreground oversampling and FN-weighted Tversky loss."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import AllGatherGrad, get_tp_fp_fn_tn

from nnUNetTrainerForeground50 import nnUNetTrainerForeground50


FP_ALPHA = 0.3
FN_BETA = 0.7


class TverskyLoss(nn.Module):
    """Soft Tversky loss with explicit FP alpha and FN beta semantics.

    The component returns ``-mean(T)`` to match nnU-Net's SoftDiceLoss sign
    convention. It is gradient-equivalent to ``1 - mean(T)`` while keeping
    the same loss scale and composition behavior as the official Dice loss.
    """

    def __init__(
        self,
        batch_dice: bool = False,
        do_bg: bool = True,
        smooth: float = 1.0,
        ddp: bool = False,
        apply_nonlin=None,
        alpha: float = FP_ALPHA,
        beta: float = FN_BETA,
    ) -> None:
        super().__init__()
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.smooth = smooth
        self.ddp = ddp
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        axes = [0] + list(range(2, len(x.shape))) if self.batch_dice else list(range(2, len(x.shape)))
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        if y.ndim == x.ndim - 1:
            y = y[:, None]
        y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
        y_onehot.scatter_(1, y.long(), 1)

        if not self.do_bg:
            x = x[:, 1:]
            y_onehot = y_onehot[:, 1:]

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y_onehot, axes, loss_mask)
        if self.ddp and self.batch_dice:
            tp = AllGatherGrad.apply(tp).sum(0, dtype=torch.float32)
            fp = AllGatherGrad.apply(fp).sum(0, dtype=torch.float32)
            fn = AllGatherGrad.apply(fn).sum(0, dtype=torch.float32)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return -tversky.mean()


class nnUNetTrainerTverskyForeground50(nnUNetTrainerForeground50):
    """Foreground50 sampling with Tversky emphasis on false negatives."""

    FP_ALPHA = FP_ALPHA
    FN_BETA = FN_BETA

    def _build_loss(self):
        if self.label_manager.has_regions:
            return super()._build_loss()

        loss = DC_and_CE_loss(
            {
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5,
                "do_bg": False,
                "ddp": self.is_ddp,
                "alpha": self.FP_ALPHA,
                "beta": self.FN_BETA,
            },
            {},
            weight_ce=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=TverskyLoss,
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
