from __future__ import annotations

import pytest
import torch
from torch import nn

from standalone_nnunet2d.engine.validator import run_validation_epoch
from standalone_nnunet2d.losses.compound import DiceCrossEntropyLoss


class _PerfectBinaryModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack((1 - x[:, 0], x[:, 0]), dim=1) * 20


def test_run_validation_epoch_reports_perfect_metrics_and_restores_train_mode() -> None:
    model = _PerfectBinaryModel()
    model.train()
    image = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    target = torch.tensor([[[0, 1], [1, 0]]])

    result = run_validation_epoch(
        model,
        [(image, target)],
        DiceCrossEntropyLoss(),
        torch.device("cpu"),
    )

    assert result.batch_count == 1
    assert result.dice == 1.0
    assert result.iou == 1.0
    assert model.training is True


def test_run_validation_epoch_rejects_empty_batches() -> None:
    model = _PerfectBinaryModel()

    with pytest.raises(ValueError, match="empty"):
        run_validation_epoch(model, [], DiceCrossEntropyLoss(), torch.device("cpu"))
