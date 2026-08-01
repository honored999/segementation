"""Explicit-weight aggregation for ordered deep-supervision logits."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn
from torch.nn import functional as F


def resize_target_nearest(target: Tensor, spatial_shape: tuple[int, int]) -> Tensor:
    """Align integer labels to one output level without creating fractional IDs."""
    if target.ndim != 3:
        raise ValueError(f"target must be (B, H, W), got {tuple(target.shape)}")
    if target.shape[1:] == spatial_shape:
        return target.long()
    return F.interpolate(target.unsqueeze(1).float(), size=spatial_shape, mode="nearest").squeeze(1).long()


class DeepSupervisionLoss(nn.Module):
    """Normalize caller-supplied positive weights over multi-scale outputs."""

    def __init__(self, base_loss: nn.Module, *, weights: Sequence[float]) -> None:
        super().__init__()
        supplied_weights = tuple(float(weight) for weight in weights)
        if not supplied_weights or any(weight <= 0 for weight in supplied_weights):
            raise ValueError("deep-supervision weights must all be positive")
        total_weight = sum(supplied_weights)
        self.base_loss = base_loss
        self.weights = tuple(weight / total_weight for weight in supplied_weights)

    def forward(self, outputs: Tensor | Sequence[Tensor], target: Tensor) -> Tensor:
        levels = (outputs,) if isinstance(outputs, Tensor) else tuple(outputs)
        if not levels:
            raise ValueError("deep-supervision outputs must not be empty")
        if len(levels) != len(self.weights):
            raise ValueError("number of deep-supervision weights must match output count")
        total_loss: Tensor | None = None
        for logits, weight in zip(levels, self.weights, strict=True):
            if logits.ndim != 4:
                raise ValueError(f"deep-supervision logits must be (B, C, H, W), got {tuple(logits.shape)}")
            level_loss = self.base_loss(logits, resize_target_nearest(target, tuple(logits.shape[2:])))
            total_loss = weight * level_loss if total_loss is None else total_loss + weight * level_loss
        assert total_loss is not None
        return total_loss
