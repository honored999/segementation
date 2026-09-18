"""Binary per-image, foreground/background grouped TopK cross-entropy."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss


class GroupedTopKCrossEntropyLoss(nn.Module):
    """Average equally weighted foreground/background TopK losses per image."""

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        ignore_index: int = -100,
        k: float = 10,
        label_smoothing: float = 0,
    ) -> None:
        super().__init__()
        if not 0 < k <= 100:
            raise ValueError(f"k must be in (0, 100], got {k}")
        self.register_buffer("weight", weight)
        self.ignore_index = ignore_index
        self.k = k
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim < 3 or logits.shape[1] != 2:
            raise ValueError(
                "GroupedTopKCrossEntropyLoss supports binary segmentation with "
                f"exactly two logit channels, got shape {tuple(logits.shape)}"
            )
        if target.ndim == logits.ndim:
            if target.shape[1] != 1:
                raise ValueError(
                    "target with the same rank as logits must have one singleton channel"
                )
            target = target[:, 0]
        if target.ndim != logits.ndim - 1 or target.shape != logits.shape[:1] + logits.shape[2:]:
            raise ValueError(
                f"target shape {tuple(target.shape)} is incompatible with logits shape {tuple(logits.shape)}"
            )

        target = target.long()
        valid = target != self.ignore_index
        invalid_labels = valid & (target != 0) & (target != 1)
        if torch.any(invalid_labels):
            labels = torch.unique(target[invalid_labels]).detach().cpu().tolist()
            raise ValueError(
                "GroupedTopKCrossEntropyLoss supports only labels 0 and 1 "
                f"(plus ignore_index={self.ignore_index}); got {labels}"
            )

        voxel_ce = F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        image_losses = []
        for image_ce, image_target, image_valid in zip(voxel_ce, target, valid):
            group_losses = []
            for label in (1, 0):
                values = image_ce[image_valid & (image_target == label)]
                if values.numel() > 0:
                    count = max(1, math.ceil(values.numel() * self.k / 100))
                    group_losses.append(
                        torch.topk(values, count, sorted=False).values.mean()
                    )
            if group_losses:
                image_losses.append(torch.stack(group_losses).mean())

        if not image_losses:
            return logits.sum() * 0.0
        return torch.stack(image_losses).mean()


class DC_and_grouped_topk_loss(DC_and_CE_loss):
    """Official Dice+CE composition with grouped TopK replacing only CE."""

    def __init__(
        self,
        soft_dice_kwargs: dict,
        ce_kwargs: dict,
        weight_ce: float = 1,
        weight_dice: float = 1,
        ignore_label: int | None = None,
    ) -> None:
        grouped_ce_kwargs = dict(ce_kwargs)
        if ignore_label is not None:
            grouped_ce_kwargs["ignore_index"] = ignore_label
        super().__init__(
            soft_dice_kwargs,
            {},
            weight_ce=weight_ce,
            weight_dice=weight_dice,
            ignore_label=ignore_label,
        )
        self.ce = GroupedTopKCrossEntropyLoss(**grouped_ce_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.ignore_label is not None:
            if target.shape[1] != 1:
                raise ValueError(
                    "ignore labels require a target with one singleton channel"
                )
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
        else:
            mask = None
            target_dice = target

        dc_loss = (
            self.dc(net_output, target_dice, loss_mask=mask)
            if self.weight_dice != 0
            else net_output.sum() * 0.0
        )
        ce_loss = (
            self.ce(net_output, target)
            if self.weight_ce != 0
            else net_output.sum() * 0.0
        )
        return self.weight_ce * ce_loss + self.weight_dice * dc_loss
