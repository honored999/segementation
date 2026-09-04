from __future__ import annotations

import torch
from torch import nn
from uuid import uuid4

from standalone_nnunet2d.engine.checkpoint import PROJECT_OUTPUTS_DIRECTORY, load_checkpoint, save_checkpoint


def test_checkpoint_restore_preserves_argmax_mask() -> None:
    model = nn.Conv2d(1, 2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    image = torch.randn(1, 1, 4, 4)
    before = model(image).argmax(dim=1)
    path = PROJECT_OUTPUTS_DIRECTORY / f"pytest-consistency-{uuid4().hex}.pth"
    save_checkpoint(model, optimizer, path, {"smoke_run_only": True, "epoch": 1})
    restored = nn.Conv2d(1, 2, 1)
    restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.01)
    load_checkpoint(restored, restored_optimizer, path, {"smoke_run_only": True})
    assert torch.equal(before, restored(image).argmax(dim=1))
