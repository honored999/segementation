from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
if str(EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTENSION_ROOT))

from nnUNetTrainerTverskyCEEarlyStopping import nnUNetTrainerTverskyCEEarlyStopping
from tversky_ce import TverskyLoss


def _build_loss(settings, *, deep_supervision=False, ignore_label=None):
    trainer = object.__new__(nnUNetTrainerTverskyCEEarlyStopping)
    trainer.label_manager = SimpleNamespace(has_regions=False, ignore_label=ignore_label)
    trainer.configuration_manager = SimpleNamespace(
        batch_dice=True, configuration={"tversky_ce": settings}
    )
    trainer.enable_deep_supervision = deep_supervision
    trainer.is_ddp = False
    trainer._get_deep_supervision_scales = lambda: [[1, 1], [0.5, 0.5]]
    trainer._do_i_compile = lambda: False
    return trainer._build_loss()


@pytest.mark.parametrize("batch_dice", [False, True])
def test_symmetric_tversky_matches_official_dice_value_and_gradient(batch_dice):
    kwargs = dict(
        apply_nonlin=lambda x: x.softmax(dim=1),
        batch_dice=batch_dice,
        do_bg=False,
        ddp=False,
        smooth=1e-5,
    )
    logits = torch.linspace(-2, 2, 16).reshape(2, 2, 2, 2).requires_grad_()
    target = torch.tensor([[[[1, 0], [0, 1]]], [[[0, 0], [0, 0]]]])
    mask = torch.tensor([[[[True, True], [False, True]]], [[[True, True], [True, True]]]])
    actual = TverskyLoss(**kwargs, alpha=0.5, beta=0.5)(logits, target, mask)
    expected = MemoryEfficientSoftDiceLoss(**kwargs)(logits, target, mask)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, logits)[0], torch.autograd.grad(expected, logits)[0]
    )


def test_larger_beta_penalizes_false_negatives_more():
    loss = TverskyLoss(alpha=0.4, beta=0.6)
    with_fn = loss(torch.tensor([[[[1.0, 0.0, 0.0]]]]), torch.tensor([[[[1, 1, 0]]]]))
    with_fp = loss(torch.tensor([[[[1.0, 1.0, 0.0]]]]), torch.tensor([[[[1, 0, 0]]]]))
    assert with_fn > with_fp


def test_external_trainer_applies_independent_compound_weights_with_deep_supervision(monkeypatch):
    monkeypatch.setenv("nnUNet_extTrainer", str(EXTENSION_ROOT))
    trainer = recursive_find_trainer_class_by_name("nnUNetTrainerTverskyCEEarlyStopping")
    assert trainer is nnUNetTrainerTverskyCEEarlyStopping
    loss = _build_loss(
        {"alpha": 0.3, "beta": 0.7, "weight_tversky": 1.5, "weight_ce": 0.5},
        deep_supervision=True,
    )
    logits = torch.linspace(-2, 2, 8).reshape(1, 2, 2, 2).requires_grad_()
    target = torch.tensor([[[[1, 0], [0, 1]]]])
    actual = loss([logits, logits[:, :, :1, :1]], [target, target[:, :, :1, :1]])
    assert (loss.loss.dc.alpha, loss.loss.dc.beta) == (0.3, 0.7)
    expected = 1.5 * loss.loss.dc(logits, target) + 0.5 * loss.loss.ce(logits, target[:, 0])
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(logits.grad).all()


def test_ce_only_all_ignored_has_a_zero_gradient():
    loss = _build_loss({"weight_tversky": 0, "weight_ce": 1}, ignore_label=2)
    logits = torch.zeros((1, 2, 2, 2), requires_grad=True)
    target = torch.full((1, 1, 2, 2), 2)
    result = loss(logits, target)
    assert result.item() == 0
    result.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


@pytest.mark.parametrize(
    "settings",
    [
        {"weight_ce": -1},
        {"weight_tversky": 0, "weight_ce": 0},
        {"alpha": float("nan")},
        {"alpha": 0, "beta": 0},
        {"weight_tversy": 1},
    ],
)
def test_invalid_plans_weights_fail_at_loss_construction(settings):
    with pytest.raises(ValueError, match="tversky_ce"):
        _build_loss(settings)
