"""Batch Dice loss for 2D segmentation logits."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _validate_logits_and_target(logits: Tensor, target: Tensor) -> None:
    """Check the dense segmentation tensor contract before one-hot encoding."""
    if logits.ndim != 4 or target.ndim != 3:
        raise ValueError("logits must be (B, C, H, W) and target must be (B, H, W)")
    if logits.shape[0] != target.shape[0] or logits.shape[2:] != target.shape[1:]:
        raise ValueError("target shape must match logits batch and spatial dimensions")
    if logits.shape[1] < 2:
        raise ValueError("logits require at least two classes")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must contain only finite values")
    if target.device != logits.device:
        raise ValueError("target and logits must be on the same device")
    if target.is_floating_point():
        raise ValueError("target must contain integer class labels")
    if target.numel() and (target.min() < 0 or target.max() >= logits.shape[1]):
        raise ValueError("target labels must be in [0, C)")


class SoftDiceLoss(nn.Module):
    """Soft Dice over the whole batch when ``batch_dice`` is enabled."""

    def __init__(
        self,
        *,
        smooth: float = 1e-5,
        batch_dice: bool = True,
        include_background: bool = False,
    ) -> None:
        super().__init__()
        if smooth <= 0:
            raise ValueError("smooth must be positive")
        self.smooth = smooth
        self.batch_dice = batch_dice
        self.include_background = include_background

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        _validate_logits_and_target(logits, target)
        probabilities = torch.softmax(logits, dim=1)
        one_hot_target = F.one_hot(target.long(), num_classes=logits.shape[1]).movedim(-1, 1).to(probabilities.dtype)
        reduction_dims = (0, 2, 3) if self.batch_dice else (2, 3)
        intersection = (probabilities * one_hot_target).sum(reduction_dims)
        denominator = probabilities.sum(reduction_dims) + one_hot_target.sum(reduction_dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        if not self.include_background:
            dice = dice[1:] if self.batch_dice else dice[:, 1:]
        return 1.0 - dice.mean()
