"""Tests for the Stage 2 FN-weighted Tversky Trainer variant."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
if str(EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTENSION_ROOT))


pytest.importorskip("nnunetv2")

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss  # noqa: E402
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper  # noqa: E402
from nnUNetTrainerStage2TverskyForeground50 import (  # noqa: E402
    nnUNetTrainerStage2TverskyForeground50,
)
from nnUNetTrainerTverskyForeground50 import (  # noqa: E402
    TverskyLoss,
    nnUNetTrainerTverskyForeground50,
)


def _unwrap_dc_and_ce_loss(loss: object) -> DC_and_CE_loss:
    if isinstance(loss, DeepSupervisionWrapper):
        loss = loss.loss
    assert isinstance(loss, DC_and_CE_loss)
    return loss


def test_stage2_inherits_tversky_foreground50_and_declares_weights() -> None:
    assert issubclass(
        nnUNetTrainerStage2TverskyForeground50,
        nnUNetTrainerTverskyForeground50,
    )
    assert nnUNetTrainerStage2TverskyForeground50.OVERSAMPLE_FOREGROUND_PERCENT == 0.50
    assert nnUNetTrainerStage2TverskyForeground50.FP_ALPHA == 0.4
    assert nnUNetTrainerStage2TverskyForeground50.FN_BETA == 0.6
    assert (
        nnUNetTrainerStage2TverskyForeground50.FN_BETA
        > nnUNetTrainerStage2TverskyForeground50.FP_ALPHA
    )


def test_stage2_reuses_parent_loss_builder() -> None:
    assert "_build_loss" not in nnUNetTrainerStage2TverskyForeground50.__dict__
    assert (
        nnUNetTrainerStage2TverskyForeground50._build_loss
        is nnUNetTrainerTverskyForeground50._build_loss
    )


def test_stage2_build_loss_uses_stage2_tversky_weights() -> None:
    reference = EXTENSION_ROOT.parent / "standalone_nnunet2d" / "reference"
    plans_path = reference / "nnUNetPlans.json"
    dataset_path = reference / "dataset.json"
    if not plans_path.is_file() or not dataset_path.is_file():
        pytest.skip("reference plans/dataset are unavailable")

    plans = json.loads(plans_path.read_text(encoding="utf-8"))
    plans["continue_training"] = False
    dataset_json = json.loads(dataset_path.read_text(encoding="utf-8"))
    trainer = nnUNetTrainerStage2TverskyForeground50(
        plans=plans,
        configuration="2d",
        fold=0,
        dataset_json=dataset_json,
        device=torch.device("cpu"),
    )
    trainer.initialize()
    loss = _unwrap_dc_and_ce_loss(trainer._build_loss())

    assert isinstance(loss.dc, TverskyLoss)
    assert loss.dc.alpha == pytest.approx(0.4)
    assert loss.dc.beta == pytest.approx(0.6)
    assert loss.dc.smooth == pytest.approx(1e-5)
    assert loss.weight_ce == 1
    assert isinstance(loss.ce, torch.nn.Module)


def test_stage2_tversky_forward_backward_is_finite() -> None:
    loss_fn = TverskyLoss(
        alpha=nnUNetTrainerStage2TverskyForeground50.FP_ALPHA,
        beta=nnUNetTrainerStage2TverskyForeground50.FN_BETA,
        do_bg=False,
        smooth=1e-5,
        apply_nonlin=lambda logits: torch.softmax(logits, dim=1),
    )
    logits = torch.tensor(
        [[[[2.0, -1.0], [0.5, 1.0]], [[-2.0, 1.0], [-0.5, -1.0]]]],
        requires_grad=True,
    )
    target = torch.tensor([[[1, 0], [0, 1]]])

    loss = loss_fn(logits, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_equal_count_false_negatives_have_larger_loss_than_false_positives() -> None:
    loss_fn = TverskyLoss(
        alpha=nnUNetTrainerStage2TverskyForeground50.FP_ALPHA,
        beta=nnUNetTrainerStage2TverskyForeground50.FN_BETA,
        do_bg=False,
        smooth=1e-5,
        apply_nonlin=lambda logits: torch.softmax(logits, dim=1),
    )

    fn_logits = torch.tensor([[[[8.0, 8.0]], [[-8.0, -8.0]]]])
    fn_target = torch.tensor([[[1, 0]]])
    fp_logits = torch.tensor([[[[-8.0, 8.0]], [[8.0, -8.0]]]])
    fp_target = torch.tensor([[[0, 0]]])

    fn_loss = loss_fn(fn_logits, fn_target)
    fp_loss = loss_fn(fp_logits, fp_target)

    assert fn_loss > fp_loss
