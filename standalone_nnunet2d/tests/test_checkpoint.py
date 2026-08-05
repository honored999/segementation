from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import standalone_nnunet2d.engine.checkpoint as checkpoint


def _checkpoint_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str) -> Path:
    monkeypatch.setattr(checkpoint, "PROJECT_OUTPUTS_DIRECTORY", tmp_path.resolve())
    return tmp_path / name


def _prime_optimizer_state(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    model(torch.ones(1, 1, 2, 2)).sum().backward()
    optimizer.step()


def test_checkpoint_round_trip_restores_model_optimizer_and_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = nn.Conv2d(1, 2, kernel_size=1)
    source_optimizer = torch.optim.SGD(source.parameters(), lr=0.1, momentum=0.9)
    _prime_optimizer_state(source, source_optimizer)
    checkpoint_path = _checkpoint_path(monkeypatch, tmp_path, "round-trip.pt")

    checkpoint.save_checkpoint(source, source_optimizer, checkpoint_path, {"fold": 0})

    target = nn.Conv2d(1, 2, kernel_size=1)
    target_optimizer = torch.optim.SGD(target.parameters(), lr=0.1, momentum=0.9)
    metadata = checkpoint.load_checkpoint(target, target_optimizer, checkpoint_path, {"fold": 0})

    assert metadata == {"fold": 0}
    assert torch.equal(source.weight, target.weight)
    assert target_optimizer.state_dict()["state"]


def test_save_checkpoint_rejects_path_outside_project_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _checkpoint_path(monkeypatch, tmp_path, "inside.pt")

    with pytest.raises(ValueError, match="outputs"):
        checkpoint.save_checkpoint(nn.Conv2d(1, 2, 1), None, tmp_path.parent / "outside.pt")


def test_load_checkpoint_rejects_metadata_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint_path = _checkpoint_path(monkeypatch, tmp_path, "metadata.pt")
    checkpoint.save_checkpoint(nn.Conv2d(1, 2, 1), None, checkpoint_path, {"fold": 0})

    with pytest.raises(ValueError, match="metadata"):
        checkpoint.load_checkpoint(nn.Conv2d(1, 2, 1), None, checkpoint_path, {"fold": 1})
