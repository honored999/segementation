from __future__ import annotations
import torch
from torch import nn

class CombinedBCEDiceLoss(nn.Module):
    """Half BCE-with-logits and half per-sample soft Dice loss."""
    def __init__(self, pos_weight: float = 1.0, smooth: float = 1e-5) -> None:
        super().__init__(); self.register_buffer("pos_weight", torch.tensor([pos_weight])); self.smooth = smooth
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight)
        probability = logits.sigmoid().flatten(1); target = target.flatten(1)
        dice = (2 * (probability * target).sum(1) + self.smooth) / (probability.sum(1) + target.sum(1) + self.smooth)
        return 0.5 * bce + 0.5 * (1 - dice).mean()

