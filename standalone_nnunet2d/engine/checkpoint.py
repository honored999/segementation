"""Explicit, local checkpoint persistence without a training loop."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


CHECKPOINT_FORMAT_VERSION = 1
PROJECT_OUTPUTS_DIRECTORY = Path(__file__).resolve().parents[1] / "outputs"


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer | None,
    path: str | Path,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Persist explicitly supplied state below the standalone outputs directory."""
    resolved_path = _resolve_output_path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
            "metadata": dict(metadata or {}),
        },
        resolved_path,
    )
    return resolved_path


def load_checkpoint(
    model: nn.Module,
    optimizer: Optimizer | None,
    path: str | Path,
    expected_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Restore an explicit local checkpoint after format and metadata checks."""
    payload = torch.load(_resolve_output_path(path), map_location="cpu", weights_only=False)
    _validate_payload(payload, expected_metadata)
    model.load_state_dict(payload["model_state_dict"])
    optimizer_state = payload["optimizer_state_dict"]
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    return dict(payload["metadata"])


def _resolve_output_path(path: str | Path) -> Path:
    candidate = Path(path).resolve()
    try:
        candidate.relative_to(PROJECT_OUTPUTS_DIRECTORY.resolve())
    except ValueError as error:
        raise ValueError("checkpoint path must be under standalone_nnunet2d/outputs") from error
    return candidate


def _validate_payload(payload: object, expected_metadata: Mapping[str, Any] | None) -> None:
    if not isinstance(payload, dict) or payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported checkpoint format")
    required_keys = {"model_state_dict", "optimizer_state_dict", "metadata"}
    if not required_keys.issubset(payload):
        raise ValueError("checkpoint payload is missing required fields")
    metadata = payload["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata must be a dictionary")
    for key, expected_value in (expected_metadata or {}).items():
        if metadata.get(key) != expected_value:
            raise ValueError("checkpoint metadata does not match expectations")
