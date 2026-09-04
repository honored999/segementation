from __future__ import annotations

import pytest
import torch

from standalone_nnunet2d.losses.dice import SoftDiceLoss
from standalone_nnunet2d.losses.compound import DiceCrossEntropyLoss
from standalone_nnunet2d.losses.deep_supervision import DeepSupervisionLoss


def test_batch_dice_is_near_zero_for_confident_correct_foreground() -> None:
    logits = torch.tensor([[[[10.0, -10.0]], [[-10.0, 10.0]]]], requires_grad=True)
    target = torch.tensor([[[0, 1]]])

    loss = SoftDiceLoss(batch_dice=True, include_background=False)(logits, target)

    assert loss.item() < 1e-4


def test_batch_dice_rejects_wrong_target_shape() -> None:
    with pytest.raises(ValueError, match="target"):
        SoftDiceLoss()(torch.randn(1, 2, 4, 4), torch.zeros(1, 1, 4, 4, dtype=torch.long))


def test_compound_loss_is_finite_and_backpropagates() -> None:
    logits = torch.randn(2, 2, 8, 8, requires_grad=True)
    target = torch.randint(0, 2, (2, 8, 8))

    loss = DiceCrossEntropyLoss()(logits, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_deep_supervision_normalizes_explicit_weights_and_resizes_targets() -> None:
    outputs = (
        torch.randn(1, 2, 8, 8, requires_grad=True),
        torch.randn(1, 2, 4, 4, requires_grad=True),
    )
    target = torch.randint(0, 2, (1, 8, 8))
    loss_fn = DeepSupervisionLoss(DiceCrossEntropyLoss(), weights=(2.0, 1.0))

    loss = loss_fn(outputs, target)
    loss.backward()

    assert loss_fn.weights == pytest.approx((2.0 / 3.0, 1.0 / 3.0))
    assert torch.isfinite(loss)
    assert outputs[0].grad is not None and outputs[1].grad is not None


def test_deep_supervision_rejects_mismatched_weight_count() -> None:
    with pytest.raises(ValueError, match="weights"):
        DeepSupervisionLoss(DiceCrossEntropyLoss(), weights=(1.0,))(
            (torch.randn(1, 2, 8, 8), torch.randn(1, 2, 4, 4)),
            torch.zeros(1, 8, 8, dtype=torch.long),
        )
