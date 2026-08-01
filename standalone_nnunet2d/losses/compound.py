"""Compound Dice and categorical cross-entropy loss."""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn
from torch.nn import functional as F

from standalone_nnunet2d.losses.dice import SoftDiceLoss


class DiceCrossEntropyLoss(nn.Module):
    """Add configurable Dice and cross-entropy terms for segmentation logits."""

    def __init__(
        self,
        *,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        dice_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if dice_weight < 0 or ce_weight < 0 or dice_weight + ce_weight == 0:
            raise ValueError("at least one of dice_weight and ce_weight must be positive")
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice = SoftDiceLoss(**(dice_kwargs or {}))

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        dice_loss = self.dice(logits, target)
        cross_entropy_loss = F.cross_entropy(logits, target.long())
        return self.dice_weight * dice_loss + self.ce_weight * cross_entropy_loss
