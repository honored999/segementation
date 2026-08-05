from __future__ import annotations

import pytest
import torch
from torch import nn

from standalone_nnunet2d.engine.trainer import run_train_epoch, train_step
from standalone_nnunet2d.losses.compound import DiceCrossEntropyLoss


def _tiny_model_and_optimizer() -> tuple[nn.Conv2d, torch.optim.Optimizer]:
    model = nn.Conv2d(1, 2, kernel_size=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return model, optimizer


def test_train_step_updates_tiny_model_parameters() -> None:
    model, optimizer = _tiny_model_and_optimizer()
    before = model.weight.detach().clone()
    batch = (torch.randn(2, 1, 4, 4), torch.randint(0, 2, (2, 4, 4)))

    result = train_step(model, batch, DiceCrossEntropyLoss(), optimizer, torch.device("cpu"))

    assert result.loss >= 0
    assert result.output_shapes == ((2, 2, 4, 4),)
    assert not torch.equal(before, model.weight.detach())


def test_run_train_epoch_updates_model_and_averages_batch_losses() -> None:
    torch.manual_seed(7)
    model, optimizer = _tiny_model_and_optimizer()
    before = model.weight.detach().clone()
    batches = [
        (torch.randn(2, 1, 4, 4), torch.randint(0, 2, (2, 4, 4))),
        (torch.randn(2, 1, 4, 4), torch.randint(0, 2, (2, 4, 4))),
    ]
    reference, reference_optimizer = _tiny_model_and_optimizer()
    reference.load_state_dict(model.state_dict())
    reference_first = train_step(
        reference, batches[0], DiceCrossEntropyLoss(), reference_optimizer, torch.device("cpu")
    )
    reference_second = train_step(
        reference, batches[1], DiceCrossEntropyLoss(), reference_optimizer, torch.device("cpu")
    )

    result = run_train_epoch(model, batches, DiceCrossEntropyLoss(), optimizer, torch.device("cpu"))

    assert result.batch_count == 2
    assert result.mean_loss == pytest.approx((reference_first.loss + reference_second.loss) / 2)
    assert result.output_shapes == ((2, 2, 4, 4),)
    assert not torch.equal(before, model.weight.detach())


def test_run_train_epoch_rejects_empty_batches() -> None:
    model, optimizer = _tiny_model_and_optimizer()

    with pytest.raises(ValueError, match="empty"):
        run_train_epoch(model, [], DiceCrossEntropyLoss(), optimizer, torch.device("cpu"))
