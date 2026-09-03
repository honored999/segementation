"""Reproducible checkpoint persistence for the formal trainer."""
from __future__ import annotations

import hashlib
import json
import random
from copy import deepcopy
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from standalone_nnunet2d.engine.checkpoint import load_checkpoint, save_checkpoint
from standalone_nnunet2d.data.input_mode import InputMode, input_spec
from standalone_nnunet2d.training.official_config import DEFAULT_RUN_STATE
from standalone_nnunet2d.alignment_evidence import OFFICIAL_ALIGNED, validate_alignment_evidence_record


@dataclass(frozen=True)
class FormalTrainerState:
    epoch: int
    global_step: int
    best_validation_dice: float
    fold: int


@dataclass(frozen=True)
class FormalCheckpointRestore:
    state: FormalTrainerState
    scheduler_step: int
    config: dict[str, Any]
    plan_hash: str
    policies: dict[str, Any]
    run_state: str
    alignment_evidence: dict[str, Any] | None

    @property
    def epoch(self) -> int:
        return self.state.epoch

    @property
    def global_step(self) -> int:
        return self.state.global_step

    @property
    def best_validation_dice(self) -> float:
        return self.state.best_validation_dice

    @property
    def fold(self) -> int:
        return self.state.fold

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FormalTrainerState):
            return self.state == other
        if isinstance(other, FormalCheckpointRestore):
            return (
                self.state == other.state
                and self.scheduler_step == other.scheduler_step
                and self.config == other.config
                and self.plan_hash == other.plan_hash
                and self.policies == other.policies
                and self.run_state == other.run_state
                and self.alignment_evidence == other.alignment_evidence
            )
        return NotImplemented


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "values": value.tolist()}
    if isinstance(value, (np.generic, torch.Tensor)):
        return value.item() if isinstance(value, np.generic) else value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def compute_plan_hash(plan: Mapping[str, Any]) -> str:
    canonical = json.dumps(_jsonable(plan), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def checkpoint_input_channels(metadata: Mapping[str, Any]) -> int:
    """Read effective model input channels, including legacy bilateral C=2."""
    if not isinstance(metadata, Mapping):
        raise TypeError("checkpoint metadata must be a mapping")

    sources: list[tuple[str, Mapping[str, Any]]] = [("metadata", metadata)]
    for key in ("resolved_config", "config"):
        if key not in metadata:
            continue
        value = metadata[key]
        if not isinstance(value, Mapping):
            raise ValueError(f"checkpoint {key} must be a mapping")
        sources.append((key, value))

    def consistent_value(key: str) -> Any:
        declared = [(name, source[key]) for name, source in sources if key in source]
        if not declared:
            return None
        if key in ("physical_input_channels", "effective_model_input_channels"):
            for name, value in declared:
                if type(value) is not int:
                    raise ValueError(
                        f"checkpoint {key} must be an integer, got {value!r} in {name}"
                    )
        first_name, first_value = declared[0]
        if any(value != first_value for _, value in declared[1:]):
            details = ", ".join(f"{name}={value!r}" for name, value in declared)
            raise ValueError(f"checkpoint {key} declarations conflict: {details}")
        return first_value

    input_mode = consistent_value("input_mode")
    legacy_flag = consistent_value("bilateral_asymmetry_channel")
    if legacy_flag is not None and not isinstance(legacy_flag, bool):
        raise ValueError(
            "checkpoint bilateral_asymmetry_channel must be a boolean, "
            f"got {legacy_flag!r}"
        )
    consistent_value("physical_input_channels")
    effective_channels = consistent_value("effective_model_input_channels")
    value = consistent_value("input_channels")
    if value is None:
        value = effective_channels
    if value is None and legacy_flag is True and input_mode is None:
        value = 2
    if value is None:
        value = 1
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"checkpoint input_channels must be a positive integer, got {value!r}")
    return value


def checkpoint_input_mode(metadata: Mapping[str, Any]) -> InputMode:
    """Resolve and validate the checkpoint's physical-to-model input contract."""
    if not isinstance(metadata, Mapping):
        raise TypeError("checkpoint metadata must be a mapping")

    sources: list[tuple[str, Mapping[str, Any]]] = [("metadata", metadata)]
    for key in ("resolved_config", "config"):
        if key not in metadata:
            continue
        value = metadata[key]
        if not isinstance(value, Mapping):
            raise ValueError(f"checkpoint {key} must be a mapping")
        sources.append((key, value))

    def consistent_value(key: str) -> Any:
        declared = [(name, source[key]) for name, source in sources if key in source]
        if not declared:
            return None
        if key in ("physical_input_channels", "effective_model_input_channels"):
            for name, value in declared:
                if type(value) is not int:
                    raise ValueError(
                        f"checkpoint {key} must be an integer, got {value!r} in {name}"
                    )
        first_name, first_value = declared[0]
        if any(value != first_value for _, value in declared[1:]):
            details = ", ".join(f"{name}={value!r}" for name, value in declared)
            raise ValueError(f"checkpoint {key} declarations conflict: {details}")
        return first_value

    mode_value = consistent_value("input_mode")
    resolved_mode: InputMode | None = None
    if mode_value is not None:
        try:
            resolved_mode = mode_value if isinstance(mode_value, InputMode) else InputMode(mode_value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"unsupported checkpoint input_mode: {mode_value!r}") from error

    legacy_value = consistent_value("bilateral_asymmetry_channel")
    if legacy_value is not None and not isinstance(legacy_value, bool):
        raise ValueError(
            "checkpoint bilateral_asymmetry_channel must be a boolean, "
            f"got {legacy_value!r}"
        )
    if legacy_value is True:
        if resolved_mode is not None and resolved_mode is not InputMode.DWI_BILATERAL:
            raise ValueError(
                "checkpoint input_mode conflicts with legacy bilateral_asymmetry_channel=True"
            )
        resolved_mode = InputMode.DWI_BILATERAL
    elif legacy_value is False and resolved_mode is InputMode.DWI_BILATERAL:
        raise ValueError(
            "checkpoint input_mode=dwi_bilateral conflicts with "
            "legacy bilateral_asymmetry_channel=False"
        )

    input_channels = consistent_value("input_channels")
    effective_channels = consistent_value("effective_model_input_channels")
    physical_channels = consistent_value("physical_input_channels")
    if legacy_value is True and mode_value is None:
        if input_channels is None:
            input_channels = effective_channels if effective_channels is not None else 2
        if effective_channels is None:
            effective_channels = 2
        if physical_channels is None:
            physical_channels = 1
    else:
        if input_channels is None:
            input_channels = effective_channels
        if input_channels is None:
            input_channels = 1
    if isinstance(input_channels, bool) or not isinstance(input_channels, int) or input_channels < 1:
        raise ValueError(
            "checkpoint input_channels must be a positive integer, "
            f"got {input_channels!r}"
        )

    if resolved_mode is None:
        if physical_channels is not None and effective_channels is not None:
            inferred = {
                (1, 1): InputMode.DWI,
                (2, 2): InputMode.DWI_ADC,
                (1, 2): InputMode.DWI_BILATERAL,
                (2, 4): InputMode.DWI_ADC_BILATERAL,
            }.get((physical_channels, effective_channels))
            if inferred is None:
                raise ValueError(
                    "checkpoint physical/effective input channel counts do not identify "
                    f"a supported input mode: physical={physical_channels}, "
                    f"effective={effective_channels}"
                )
            resolved_mode = inferred
        elif input_channels == 1:
            resolved_mode = InputMode.DWI
        elif input_channels == 2:
            resolved_mode = InputMode.DWI_ADC
        else:
            raise ValueError(
                "checkpoint input_mode is required for input_channels="
                f"{input_channels}"
            )

    spec = input_spec(resolved_mode)
    expected_physical = spec.physical_input_channels
    expected_effective = spec.effective_input_channels
    if input_channels != expected_effective:
        raise ValueError(
            f"checkpoint input_mode={resolved_mode.value} requires input_channels="
            f"{expected_effective}, got {input_channels}"
        )
    if effective_channels is not None and effective_channels != expected_effective:
        raise ValueError(
            f"checkpoint input_mode={resolved_mode.value} requires "
            f"effective_model_input_channels={expected_effective}, got {effective_channels}"
        )
    if physical_channels is not None and physical_channels != expected_physical:
        raise ValueError(
            f"checkpoint input_mode={resolved_mode.value} requires "
            f"physical_input_channels={expected_physical}, got {physical_channels}"
        )
    if resolved_mode in (InputMode.DWI_BILATERAL, InputMode.DWI_ADC_BILATERAL):
        if physical_channels is None or effective_channels is None:
            raise ValueError(
                f"checkpoint input_mode={resolved_mode.value} must declare physical_input_channels="
                f"{expected_physical} and effective_model_input_channels={expected_effective}"
            )
    return resolved_mode


def checkpoint_bilateral_asymmetry_channel(metadata: Mapping[str, Any]) -> bool:
    """Resolve and validate the explicit bilateral-derived input provenance."""
    config = metadata.get("resolved_config", metadata.get("config", {}))
    if not isinstance(config, Mapping) or not config.get("bilateral_asymmetry_channel", False):
        return False
    physical_channels = config.get("physical_input_channels")
    effective_channels = config.get("effective_model_input_channels")
    checkpoint_channels = checkpoint_input_channels(metadata)
    if physical_channels != 1 or effective_channels != 2 or checkpoint_channels != 2:
        raise ValueError(
            "bilateral_asymmetry_channel checkpoint must declare physical_input_channels=1, "
            "effective_model_input_channels=2, and input_channels=2"
        )
    return True


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(rng_state: Mapping[str, Any]) -> None:
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch_cpu"])
    cuda_state = rng_state.get("torch_cuda", [])
    if torch.cuda.is_available() and cuda_state:
        torch.cuda.set_rng_state_all(cuda_state)


def _scheduler_step(scheduler: Any, state: FormalTrainerState) -> int:
    recorded = getattr(scheduler, "_formal_last_step", None)
    return int(recorded) if recorded is not None else max(0, state.epoch - 1)


def _capture_scheduler_state(scheduler: Any, state: FormalTrainerState) -> dict[str, Any]:
    step = _scheduler_step(scheduler, state)
    last_lr = list(scheduler.get_last_lr()) if hasattr(scheduler, "get_last_lr") else []
    return {
        "step": step,
        "ctr": step + 1,
        "last_lr": last_lr,
        "initial_lr": getattr(scheduler, "initial_lr", None),
        "max_steps": getattr(scheduler, "max_steps", None),
        "exponent": getattr(scheduler, "exponent", None),
    }


def _restore_scheduler_state(scheduler: Any, scheduler_state: Mapping[str, Any]) -> int:
    step = int(scheduler_state["step"])
    scheduler.step(step)
    last_lr = scheduler_state.get("last_lr", [])
    if last_lr:
        for group, lr in zip(scheduler.optimizer.param_groups, last_lr):
            group["lr"] = float(lr)
    if hasattr(scheduler, "ctr"):
        scheduler.ctr = int(scheduler_state.get("ctr", step + 1))
    scheduler._formal_last_step = step
    return step


def _normalise_save_arguments(
    scheduler_or_path: Any,
    path_or_state: Any,
    state_or_config: Any,
    config: dict[str, Any] | None,
) -> tuple[Any | None, Path, FormalTrainerState, dict[str, Any]]:
    if isinstance(scheduler_or_path, (str, Path)):
        scheduler = None
        path = Path(scheduler_or_path)
        state = path_or_state
        resolved_config = state_or_config
    else:
        scheduler = scheduler_or_path
        path = Path(path_or_state)
        state = state_or_config
        resolved_config = config
    if not isinstance(state, FormalTrainerState):
        raise TypeError("state must be a FormalTrainerState")
    if not isinstance(resolved_config, dict):
        raise TypeError("config must be a dictionary")
    return scheduler, path, state, dict(resolved_config)


def _resolve_contract(
    *,
    config: Mapping[str, Any],
    run_state: str,
    alignment_evidence: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    config_run_type = config.get("run_type", DEFAULT_RUN_STATE)
    config_run_state = config.get("run_state", config_run_type)
    if config_run_type != config_run_state or config_run_state != run_state:
        raise ValueError(
            "formal checkpoint config and run_state do not match; "
            f"pending state is {DEFAULT_RUN_STATE}"
        )
    config_evidence = config.get("alignment_evidence")
    if run_state == DEFAULT_RUN_STATE:
        if alignment_evidence is not None or config_evidence is not None:
            raise ValueError("pending formal checkpoint cannot carry alignment evidence")
        return None
    if run_state != OFFICIAL_ALIGNED:
        raise ValueError(f"unsupported formal checkpoint run_state: {run_state}")
    if alignment_evidence is None or config_evidence is None:
        raise ValueError("official_aligned formal checkpoint requires alignment evidence")
    validated = validate_alignment_evidence_record(alignment_evidence)
    if config_evidence != validated:
        raise ValueError("formal checkpoint config alignment evidence does not match")
    return validated


def save_formal_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler_or_path: Any,
    path_or_state: Any,
    state_or_config: Any,
    config: dict[str, Any] | None = None,
    *,
    rng_state: Mapping[str, Any] | None = None,
    plan_hash: str | None = None,
    policies: Mapping[str, Any] | None = None,
    run_state: str = DEFAULT_RUN_STATE,
    alignment_evidence: Mapping[str, Any] | None = None,
) -> Path:
    scheduler, path, state, resolved_config = _normalise_save_arguments(
        scheduler_or_path, path_or_state, state_or_config, config
    )
    validated_evidence = _resolve_contract(
        config=resolved_config,
        run_state=run_state,
        alignment_evidence=alignment_evidence,
    )
    resolved_plan_hash = plan_hash or str(resolved_config.get("plan_hash") or compute_plan_hash(resolved_config))
    resolved_policies = dict(policies or resolved_config.get("policies", {}))
    input_channels = checkpoint_input_channels(resolved_config)
    metadata: dict[str, Any] = {
        "run_type": run_state,
        "run_state": run_state,
        "alignment_evidence": deepcopy(validated_evidence),
        "epoch": state.epoch,
        "global_step": state.global_step,
        "best_validation_dice": state.best_validation_dice,
        "fold": state.fold,
        "input_channels": input_channels,
        "config": resolved_config,
        "resolved_config": resolved_config,
        "plan_hash": resolved_plan_hash,
        "policies": resolved_policies,
        "rng_state": dict(capture_rng_state() if rng_state is None else rng_state),
        "scheduler_state": None if scheduler is None else _capture_scheduler_state(scheduler, state),
    }
    return save_checkpoint(model, optimizer, path, metadata)


def load_formal_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler_or_path: Any,
    path: Path | None = None,
    *,
    fold: int,
    plan_hash: str | None = None,
    policies: Mapping[str, Any] | None = None,
    run_state: str = DEFAULT_RUN_STATE,
    alignment_evidence: Mapping[str, Any] | None = None,
) -> FormalCheckpointRestore:
    if path is None:
        scheduler = None
        checkpoint_path = Path(scheduler_or_path)
    else:
        scheduler = scheduler_or_path
        checkpoint_path = Path(path)
    expected_evidence = _resolve_contract(
        config={"run_type": run_state, "run_state": run_state, "alignment_evidence": alignment_evidence},
        run_state=run_state,
        alignment_evidence=alignment_evidence,
    )
    expected: dict[str, Any] = {
        "run_type": run_state,
        "run_state": run_state,
        "alignment_evidence": expected_evidence,
        "fold": fold,
    }
    if plan_hash is not None:
        expected["plan_hash"] = plan_hash
    if policies is not None:
        expected["policies"] = dict(policies)
    metadata = load_checkpoint(model, optimizer, checkpoint_path, expected)
    actual_run_state = str(metadata.get("run_state", metadata.get("run_type", "")))
    actual_evidence = _resolve_contract(
        config=dict(metadata.get("resolved_config", metadata.get("config", {}))),
        run_state=actual_run_state,
        alignment_evidence=metadata.get("alignment_evidence"),
    )
    state = FormalTrainerState(
        int(metadata["epoch"]),
        int(metadata["global_step"]),
        float(metadata["best_validation_dice"]),
        int(metadata["fold"]),
    )
    scheduler_state = metadata.get("scheduler_state")
    if scheduler is not None and scheduler_state is None:
        raise ValueError("formal checkpoint is missing scheduler_state")
    scheduler_step = (
        _restore_scheduler_state(scheduler, scheduler_state) if scheduler is not None else max(0, state.epoch - 1)
    )
    rng_state = metadata.get("rng_state")
    if rng_state is not None:
        _restore_rng_state(rng_state)
    return FormalCheckpointRestore(
        state=state,
        scheduler_step=scheduler_step,
        config=dict(metadata.get("resolved_config", metadata.get("config", {}))),
        plan_hash=str(metadata.get("plan_hash", "")),
        policies=dict(metadata.get("policies", {})),
        run_state=actual_run_state,
        alignment_evidence=deepcopy(actual_evidence),
    )
