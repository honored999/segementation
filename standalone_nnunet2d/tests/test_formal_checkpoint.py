from __future__ import annotations
import json
import random
from uuid import uuid4
import numpy as np
import torch
from torch import nn
from standalone_nnunet2d.engine.checkpoint import PROJECT_OUTPUTS_DIRECTORY
from standalone_nnunet2d.training.formal_checkpoint import (
    FormalTrainerState,
    load_formal_checkpoint,
    save_formal_checkpoint,
)
from standalone_nnunet2d.training.official_config import PolyLRScheduler


def _checkpoint_path() -> object:
    return PROJECT_OUTPUTS_DIRECTORY / f"pytest-formal-{uuid4().hex}.pth"


def test_formal_checkpoint_restores_scheduler_and_rng_state() -> None:
    torch.manual_seed(3)
    model = nn.Conv2d(1, 2, 1)
    optimizer = torch.optim.SGD(model.parameters(), .01, momentum=.9)
    scheduler = PolyLRScheduler(optimizer, .01, 1000)
    scheduler.step(17)
    state = FormalTrainerState(epoch=18, global_step=4500, best_validation_dice=.4, fold=0)
    config = {"seed": 0, "resolved": True}
    policies = {"scheduler": {"name": "poly", "exponent": .9}, "sampling": {"foreground": .33}}

    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    path = _checkpoint_path()
    save_formal_checkpoint(
        model,
        optimizer,
        scheduler,
        path,
        state,
        config,
        plan_hash="plan-sha256",
        policies=policies,
    )
    expected_rng = (random.random(), float(np.random.random()), torch.rand(3))
    random.seed(91)
    np.random.seed(91)
    torch.manual_seed(91)

    restored_model = nn.Conv2d(1, 2, 1)
    restored_optimizer = torch.optim.SGD(restored_model.parameters(), .01, momentum=.9)
    restored_scheduler = PolyLRScheduler(restored_optimizer, .01, 1000)
    restored = load_formal_checkpoint(
        restored_model,
        restored_optimizer,
        restored_scheduler,
        path,
        fold=0,
        plan_hash="plan-sha256",
        policies=policies,
    )

    assert restored.state == state
    assert restored.scheduler_step == 17
    assert restored.config == config
    assert restored.policies == policies
    assert restored.run_state == "official_alignment_pending"
    assert random.random() == expected_rng[0]
    assert np.random.random() == expected_rng[1]
    assert torch.equal(torch.rand(3), expected_rng[2])

    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    assert metadata["scheduler_state"]["step"] == 17
    assert metadata["plan_hash"] == "plan-sha256"
    assert metadata["policies"] == policies
    assert metadata["run_state"] == "official_alignment_pending"
    assert metadata["resolved_config"] == config


def test_formal_checkpoint_rejects_official_aligned_local_state() -> None:
    model = nn.Conv2d(1, 2, 1)
    optimizer = torch.optim.SGD(model.parameters(), .01)
    scheduler = PolyLRScheduler(optimizer, .01, 1000)
    state = FormalTrainerState(epoch=1, global_step=1, best_validation_dice=.1, fold=0)

    try:
        save_formal_checkpoint(
            model,
            optimizer,
            scheduler,
            _checkpoint_path(),
            state,
            {"run_state": "official_aligned"},
            run_state="official_aligned",
        )
    except ValueError as error:
        assert "official_alignment_pending" in str(error)
    else:
        raise AssertionError("official_aligned must not be persisted locally")
