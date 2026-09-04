from __future__ import annotations

import torch
from torch import nn
from uuid import uuid4

from standalone_nnunet2d.engine.checkpoint import PROJECT_OUTPUTS_DIRECTORY
from standalone_nnunet2d.engine.smoke_runner import run_smoke_epoch
from standalone_nnunet2d.losses.compound import DiceCrossEntropyLoss


def test_run_smoke_epoch_runs_one_train_and_validation_batch(tmp_path) -> None:
    model = nn.Conv2d(1, 2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = (torch.randn(1, 1, 4, 4), torch.randint(0, 2, (1, 4, 4)))

    result = run_smoke_epoch(
        model, [batch], [batch], DiceCrossEntropyLoss(), optimizer, torch.device("cpu"),
        PROJECT_OUTPUTS_DIRECTORY / f"pytest-smoke-{uuid4().hex}.pt",
    )

    assert result["train"]["batch_count"] == 1
    assert result["validation"]["batch_count"] == 1
