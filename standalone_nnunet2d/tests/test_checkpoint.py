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


def test_save_checkpoint_allows_path_under_explicit_root_and_rejects_escapes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    model = nn.Conv2d(1, 2, 1)

    resolved = checkpoint.save_checkpoint(
        model,
        None,
        root / "nested" / "checkpoint.pth",
        allowed_root=root,
    )

    assert resolved == (root / "nested" / "checkpoint.pth").resolve()
    assert resolved.exists()

    with pytest.raises(ValueError, match="allowed root"):
        checkpoint.save_checkpoint(model, None, tmp_path / "run_sibling" / "checkpoint.pth", allowed_root=root)

    with pytest.raises(ValueError, match="allowed root"):
        checkpoint.save_checkpoint(model, None, root / "nested" / ".." / ".." / "escape.pth", allowed_root=root)


def test_load_checkpoint_allows_path_under_explicit_root(tmp_path: Path) -> None:
    root = tmp_path / "run"
    checkpoint_path = root / "checkpoint.pth"
    source = nn.Conv2d(1, 2, 1)
    checkpoint.save_checkpoint(source, None, checkpoint_path, {"fold": 0}, allowed_root=root)

    target = nn.Conv2d(1, 2, 1)
    metadata = checkpoint.load_checkpoint(target, None, checkpoint_path, {"fold": 0}, allowed_root=root)

    assert metadata == {"fold": 0}
    assert torch.equal(source.weight, target.weight)


def test_load_checkpoint_rejects_outside_explicit_root_before_read_or_state_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "run"
    outside_path = tmp_path / "other" / "checkpoint.pth"
    load_calls: list[tuple[object, ...]] = []
    state_load_calls: list[object] = []

    monkeypatch.setattr(checkpoint.torch, "load", lambda *args, **kwargs: load_calls.append(args) or {})
    model = nn.Conv2d(1, 2, 1)
    model.load_state_dict = lambda *args, **kwargs: state_load_calls.append(args)  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="allowed root"):
        checkpoint.load_checkpoint(model, None, outside_path, allowed_root=root)

    assert load_calls == []
    assert state_load_calls == []


def test_load_checkpoint_rejects_metadata_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint_path = _checkpoint_path(monkeypatch, tmp_path, "metadata.pt")
    checkpoint.save_checkpoint(nn.Conv2d(1, 2, 1), None, checkpoint_path, {"fold": 0})

    with pytest.raises(ValueError, match="metadata"):
        checkpoint.load_checkpoint(nn.Conv2d(1, 2, 1), None, checkpoint_path, {"fold": 1})
