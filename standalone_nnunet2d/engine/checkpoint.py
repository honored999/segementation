"""Explicit, local checkpoint persistence without a training loop."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from standalone_nnunet2d.models.factory import resolve_checkpoint_model_identity


CHECKPOINT_FORMAT_VERSION = 1
PROJECT_OUTPUTS_DIRECTORY = Path(__file__).resolve().parents[1] / "outputs"


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer | None,
    path: str | Path,
    metadata: Mapping[str, Any] | None = None,
    *,
    allowed_root: str | Path | None = None,
) -> Path:
    """Persist explicitly supplied state below the configured checkpoint root."""
    resolved_path = _resolve_output_path(path, allowed_root=allowed_root)
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
    *,
    allowed_root: str | Path | None = None,
) -> dict[str, Any]:
    """Restore an explicit local checkpoint after format and metadata checks."""
    payload = torch.load(
        _resolve_output_path(path, allowed_root=allowed_root),
        map_location="cpu",
        weights_only=False,
    )
    _validate_payload(payload, expected_metadata)
    validate_model_state_compatibility(model, payload["model_state_dict"])
    model.load_state_dict(payload["model_state_dict"])
    optimizer_state = payload["optimizer_state_dict"]
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    return dict(payload["metadata"])


def validate_model_state_compatibility(model: nn.Module, state_dict: object) -> None:
    """Reject missing, unexpected or differently shaped weights before target loading."""
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint model_state_dict must be a mapping")
    target = model.state_dict()
    if set(state_dict) != set(target):
        raise ValueError("checkpoint model_state_dict keys do not match target model")
    for key, expected in target.items():
        value = state_dict[key]
        if not isinstance(value, torch.Tensor) or value.shape != expected.shape:
            raise ValueError(f"checkpoint model_state_dict shape/type mismatch for {key!r}")


def _resolve_output_path(path: str | Path, *, allowed_root: str | Path | None = None) -> Path:
    candidate = Path(path).resolve()
    root = (PROJECT_OUTPUTS_DIRECTORY if allowed_root is None else Path(allowed_root)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        if allowed_root is None:
            raise ValueError("checkpoint path must be under standalone_nnunet2d/outputs") from error
        raise ValueError(f"checkpoint path must be under allowed root: {root}") from error
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
    expected_items = dict(expected_metadata or {})
    checkpoint_model_name, checkpoint_supervision_mode = resolve_checkpoint_model_identity(metadata)
    expected_model_keys = {"model_name", "supervision_mode"}.intersection(expected_items)
    if expected_model_keys and expected_model_keys != {"model_name", "supervision_mode"}:
        raise ValueError("expected checkpoint model identity must contain model_name and supervision_mode")
    if expected_model_keys:
        if (
            checkpoint_model_name != expected_items["model_name"]
            or checkpoint_supervision_mode != expected_items["supervision_mode"]
        ):
            raise ValueError(
                "checkpoint model_name/supervision_mode identity does not match expectations"
            )
    for key, expected_value in expected_items.items():
        if key in {"model_name", "supervision_mode"}:
            continue
        if metadata.get(key) != expected_value:
            raise ValueError(f"checkpoint metadata key {key!r} does not match expectations")
