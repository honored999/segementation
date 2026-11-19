"""Configurable foreground Tversky overlap for the official nnU-Net CE loss."""

from __future__ import annotations

from math import isfinite

import numpy as np
import torch
from torch import Tensor

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import AllGatherGrad, MemoryEfficientSoftDiceLoss


class TverskyLoss(MemoryEfficientSoftDiceLoss):
    """Use nnU-Net's Dice reductions with separately weighted FP and FN."""

    def __init__(self, *, alpha: float, beta: float, **dice_options) -> None:
        super().__init__(**dice_options)
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: Tensor, y: Tensor, loss_mask: Tensor | None = None) -> Tensor:
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        axes = tuple(range(2, x.ndim))
        with torch.no_grad():
            if y.ndim == x.ndim - 1:
                y = y.unsqueeze(1)
            if y.shape == x.shape:
                target = y.to(dtype=torch.float32)
            else:
                target = torch.zeros_like(x, dtype=torch.float32)
                target.scatter_(1, y.long(), 1)

            if not self.do_bg:
                target = target[:, 1:]
            if loss_mask is not None:
                target = target * loss_mask
            target_sum = target.sum(axes, dtype=torch.float32)

        if not self.do_bg:
            x = x[:, 1:]
        true_positive = (x * target).sum(axes, dtype=torch.float32)
        prediction = x if loss_mask is None else x * loss_mask
        prediction_sum = prediction.sum(axes, dtype=torch.float32)

        if self.batch_dice:
            if self.ddp:
                true_positive = AllGatherGrad.apply(true_positive).sum(0, dtype=torch.float32)
                prediction_sum = AllGatherGrad.apply(prediction_sum).sum(0, dtype=torch.float32)
                target_sum = AllGatherGrad.apply(target_sum).sum(0, dtype=torch.float32)
            true_positive = true_positive.sum(0, dtype=torch.float32)
            prediction_sum = prediction_sum.sum(0, dtype=torch.float32)
            target_sum = target_sum.sum(0, dtype=torch.float32)

        false_positive = (prediction_sum - true_positive).clamp_min(0)
        false_negative = (target_sum - true_positive).clamp_min(0)
        # Divide the official Dice smoothing by two to match Dice at alpha=beta=0.5.
        smooth = self.smooth * 0.5
        denominator = true_positive + self.alpha * false_positive + self.beta * false_negative + smooth
        return -((true_positive + smooth) / denominator).mean()


class _TverskyAndCE(DC_and_CE_loss):
    def forward(self, net_output: Tensor, target: Tensor) -> Tensor:
        result = super().forward(net_output, target)
        # nnU-Net returns a Python zero for CE-only batches containing only ignore labels.
        return result if isinstance(result, Tensor) else net_output.sum(dtype=torch.float32) * 0.0


class TverskyCELossMixin:
    """Configure the Tversky overlap and CE weights through the active plans."""

    def _build_loss(self):
        if self.label_manager.has_regions:
            raise ValueError("tversky_ce requires class labels")

        weights = {"alpha": 0.4, "beta": 0.6, "weight_tversky": 1.0, "weight_ce": 1.0}
        overrides = self.configuration_manager.configuration.get("tversky_ce", {})
        if not isinstance(overrides, dict) or overrides.keys() - weights.keys():
            raise ValueError("tversky_ce must contain only alpha, beta, weight_tversky and weight_ce")
        weights.update(overrides)
        for name, value in weights.items():
            if type(value) not in (int, float) or not 0 <= value or not isfinite(value):
                raise ValueError(f"tversky_ce.{name} must be a finite non-negative number")
        if not (weights["alpha"] or weights["beta"]):
            raise ValueError("tversky_ce.alpha and beta cannot both be zero")
        if not (weights["weight_tversky"] or weights["weight_ce"]):
            raise ValueError("tversky_ce.weight_tversky and weight_ce cannot both be zero")

        loss = _TverskyAndCE(
            {
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5,
                "do_bg": False,
                "ddp": self.is_ddp,
                "alpha": weights["alpha"],
                "beta": weights["beta"],
            },
            {},
            weight_dice=weights["weight_tversky"],
            weight_ce=weights["weight_ce"],
            ignore_label=self.label_manager.ignore_label,
            dice_class=TverskyLoss,
        )
        compile_loss = self._do_i_compile()
        if compile_loss:
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            scales = self._get_deep_supervision_scales()
            ds_weights = np.array([1 / (2**index) for index in range(len(scales))])
            if len(ds_weights) > 1:
                ds_weights[-1] = 1e-6 if self.is_ddp and not compile_loss else 0
            loss = DeepSupervisionWrapper(loss, ds_weights / ds_weights.sum())
        return loss
