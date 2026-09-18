"""Focused synthetic contracts for grouped per-image TopK10 cross-entropy."""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
if str(EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTENSION_ROOT))

from grouped_topk_loss import DC_and_grouped_topk_loss, GroupedTopKCrossEntropyLoss
from nnUNetTrainerGroupedTopK10 import nnUNetTrainerGroupedTopK10


def _manual_grouped_topk(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    voxel_ce = F.cross_entropy(logits, target, reduction="none")
    image_losses = []
    for image_ce, image_target in zip(voxel_ce, target):
        group_losses = []
        for label in (1, 0):
            values = image_ce[image_target == label]
            if values.numel():
                count = math.ceil(0.1 * values.numel())
                group_losses.append(torch.topk(values, count, sorted=False).values.mean())
        image_losses.append(torch.stack(group_losses).mean())
    return torch.stack(image_losses).mean()


def test_grouped_topk10_matches_per_image_equal_group_definition_and_backward() -> None:
    logits = torch.randn(2, 2, 1, 20, requires_grad=True)
    target = torch.tensor(
        [
            [[[0] * 9 + [1] * 11]],
            [[[0] * 12 + [1] * 8]],
        ]
    )

    loss = GroupedTopKCrossEntropyLoss(k=10)(logits, target)
    expected = _manual_grouped_topk(logits, target[:, 0])

    first_ce = F.cross_entropy(logits[0:1], target[0], reduction="none")[0]
    expected_first = 0.5 * (
        torch.topk(first_ce[target[0, 0] == 1], 2).values.mean()
        + torch.topk(first_ce[target[0, 0] == 0], 1).values.mean()
    )

    assert torch.allclose(loss, expected)
    assert torch.allclose(_manual_grouped_topk(logits[:1], target[:1, 0]), expected_first)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_grouped_topk10_handles_empty_foreground_ignore_and_all_ignore() -> None:
    logits = torch.randn(3, 2, 1, 5, requires_grad=True)
    target = torch.tensor(
        [
            [[[0, 0, 0, 0, 0]]],
            [[[0, 1, -1, -1, -1]]],
            [[[-1, -1, -1, -1, -1]]],
        ]
    )
    criterion = GroupedTopKCrossEntropyLoss(k=10, ignore_index=-1)

    loss = criterion(logits, target)
    # Recompute the second image without ignored voxels; ignored entries must not affect counts or ranking.
    ce0 = F.cross_entropy(logits[0:1], target[0, 0].unsqueeze(0), reduction="none").max()
    ce1 = F.cross_entropy(logits[1:2, :, :, :2], target[1, 0, :, :2].unsqueeze(0), reduction="none")
    expected = torch.stack((ce0, ce1[0, 0].mean())).mean()

    assert torch.allclose(loss, expected)
    assert torch.isfinite(loss)

    all_ignore_logits = torch.randn(2, 2, 2, 2, requires_grad=True)
    all_ignore_target = torch.full((2, 1, 2, 2), -1)
    zero = criterion(all_ignore_logits, all_ignore_target)
    assert zero.item() == 0
    assert zero.requires_grad
    zero.backward()
    assert all_ignore_logits.grad is not None
    assert torch.count_nonzero(all_ignore_logits.grad) == 0


def test_grouped_topk10_rejects_nonbinary_logits_and_labels() -> None:
    criterion = GroupedTopKCrossEntropyLoss(k=10)
    with pytest.raises(ValueError, match="exactly two logit channels"):
        criterion(torch.randn(1, 3, 2, 2), torch.zeros(1, 1, 2, 2, dtype=torch.long))
    with pytest.raises(ValueError, match="labels 0 and 1"):
        criterion(torch.randn(1, 2, 2, 2), torch.tensor([[[[0, 2], [0, 1]]]]))


def test_compound_grouped_ce_keeps_all_ignore_zero_connected() -> None:
    logits = torch.randn(1, 2, 2, 2, requires_grad=True)
    target = torch.full((1, 1, 2, 2), -1)
    loss = DC_and_grouped_topk_loss(
        {"batch_dice": False, "smooth": 1e-5, "do_bg": False, "ddp": False},
        {"k": 10},
        weight_ce=1,
        weight_dice=0,
        ignore_label=-1,
    )(logits, target)

    assert isinstance(loss, torch.Tensor)
    assert loss.item() == 0
    assert loss.requires_grad
    loss.backward()
    assert logits.grad is not None


def test_grouped_topk10_trainer_contract_and_external_discovery() -> None:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name

    assert issubclass(nnUNetTrainerGroupedTopK10, nnUNetTrainer)
    assert issubclass(nnUNetTrainerGroupedTopK10, __import__("nnUNetTrainerTopK10").nnUNetTrainerTopK10)
    assert nnUNetTrainerGroupedTopK10.TOPK_PERCENT == 10

    old_path = os.environ.get("nnUNet_extTrainer")
    os.environ["nnUNet_extTrainer"] = str(EXTENSION_ROOT)
    try:
        resolved = recursive_find_trainer_class_by_name("nnUNetTrainerGroupedTopK10")
    finally:
        if old_path is None:
            os.environ.pop("nnUNet_extTrainer", None)
        else:
            os.environ["nnUNet_extTrainer"] = old_path
    assert resolved is nnUNetTrainerGroupedTopK10
