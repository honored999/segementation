"""Tests for the Stage 1 Tversky high-recall Trainer variant."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
if str(EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTENSION_ROOT))


nnunetv2 = pytest.importorskip("nnunetv2")

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnUNetTrainerForeground50 import nnUNetTrainerForeground50
from nnUNetTrainerTverskyForeground50 import (
    FN_BETA,
    FP_ALPHA,
    TverskyLoss,
    nnUNetTrainerTverskyForeground50,
)


def test_trainer_inherits_foreground50_and_declares_fn_weight() -> None:
    assert issubclass(nnUNetTrainerTverskyForeground50, nnUNetTrainerForeground50)
    assert issubclass(nnUNetTrainerTverskyForeground50, nnUNetTrainer)
    assert FP_ALPHA == 0.3
    assert FN_BETA == 0.7


def test_equal_count_false_negatives_are_penalized_more_than_false_positives() -> None:
    loss = TverskyLoss(
        batch_dice=False,
        do_bg=False,
        smooth=1e-5,
        ddp=False,
        apply_nonlin=lambda logits: torch.softmax(logits, dim=1),
    )

    # One foreground target pixel is fixed. Each example has one error, but
    # the error is either an FN or an FP. The configured beta must make FN
    # produce the larger loss.
    fn_logits = torch.tensor([[[[8.0, 8.0]], [[-8.0, -8.0]]]])
    fn_target = torch.tensor([[[[1, 0]]]])
    fp_logits = torch.tensor([[[[-8.0, 8.0]], [[8.0, -8.0]]]])
    fp_target = torch.tensor([[[[0, 0]]]])

    fn_loss = loss(fn_logits, fn_target)
    fp_loss = loss(fp_logits, fp_target)

    assert fn_loss > fp_loss


def test_tversky_accepts_dc_ce_activation_and_uses_it() -> None:
    calls = []

    def activation(logits: torch.Tensor) -> torch.Tensor:
        calls.append(tuple(logits.shape))
        return torch.softmax(logits, dim=1)

    combined = DC_and_CE_loss(
        {
            "batch_dice": False,
            "smooth": 1e-5,
            "do_bg": False,
            "ddp": False,
        },
        {},
        weight_ce=1,
        weight_dice=1,
        dice_class=TverskyLoss,
    )
    assert isinstance(combined.dc, TverskyLoss)
    assert combined.dc.apply_nonlin is not None

    loss = TverskyLoss(do_bg=False, smooth=1e-5, apply_nonlin=activation)
    logits = torch.zeros((1, 2, 1, 2))
    target = torch.tensor([[[[1, 0]]]])
    loss(logits, target)

    assert calls == [(1, 2, 1, 2)]


def _contains_tversky(loss: object) -> bool:
    if isinstance(loss, TverskyLoss):
        return True
    if isinstance(loss, DeepSupervisionWrapper):
        return _contains_tversky(loss.loss)
    return any(_contains_tversky(child) for child in getattr(loss, "children", lambda: ())())


def smoke_construct_and_build_loss() -> None:
    """Construct and initialize the Trainer without starting an epoch."""
    reference = EXTENSION_ROOT.parent / "standalone_nnunet2d" / "reference"
    plans_path = reference / "nnUNetPlans.json"
    dataset_path = reference / "dataset.json"
    if not plans_path.is_file() or not dataset_path.is_file():
        pytest.skip("reference plans/dataset are unavailable")

    plans = json.loads(plans_path.read_text(encoding="utf-8"))
    # Match nnU-Net's official entry-point transient flag for a fresh run.
    plans["continue_training"] = False
    dataset_json = json.loads(dataset_path.read_text(encoding="utf-8"))
    trainer = nnUNetTrainerTverskyForeground50(
        plans=plans,
        configuration="2d",
        fold=0,
        dataset_json=dataset_json,
        device=torch.device("cpu"),
    )
    trainer.initialize()
    loss = trainer._build_loss()

    assert trainer.oversample_foreground_percent == 0.50
    assert _contains_tversky(loss)


def test_trainer_can_construct_and_build_loss_without_training() -> None:
    smoke_construct_and_build_loss()


if __name__ == "__main__":
    test_trainer_inherits_foreground50_and_declares_fn_weight()
    test_equal_count_false_negatives_are_penalized_more_than_false_positives()
