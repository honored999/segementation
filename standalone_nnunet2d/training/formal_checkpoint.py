"""Reproducible checkpoint persistence for the formal trainer."""
from __future__ import annotations

import io
import hashlib
import json
import random
import math
import re
from copy import copy, deepcopy
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from standalone_nnunet2d.engine.checkpoint import save_checkpoint, _validate_payload, _resolve_output_path
from standalone_nnunet2d.tools.convert_official_checkpoint import _numpy_weights_only_safe_globals
from standalone_nnunet2d.training.official_config import DEFAULT_RUN_STATE
from standalone_nnunet2d.alignment_evidence import OFFICIAL_ALIGNED, validate_alignment_evidence_record
from standalone_nnunet2d.models.factory import get_model_contract


@dataclass(frozen=True)
class FormalTrainerState:
    epoch: int
    global_step: int
    best_validation_dice: float
    fold: int
    best_selection_dice: float = -1.0
    best_selection_epoch: int = 0
    early_stop_reference_dice: float = -1.0
    checks_without_improvement: int = 0


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
    model_name: str | None = None,
    supervision_mode: str | None = None,
    checkpoint_root: str | Path | None = None,
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
    nested_model = resolved_config.get("model")
    if nested_model is not None:
        if not isinstance(nested_model, Mapping):
            raise ValueError("formal checkpoint config model must be a mapping")
        config_model_name = nested_model.get("name")
        config_supervision_mode = nested_model.get("supervision_mode")
        if config_model_name is None or config_supervision_mode is None:
            raise ValueError("formal checkpoint config model must contain name and supervision_mode")
        contract = get_model_contract(str(config_model_name), supervision_mode=str(config_supervision_mode))
        if model_name is not None and model_name != contract.name:
            raise ValueError("formal checkpoint model_name does not match config model")
        if supervision_mode is not None and supervision_mode != contract.supervision_mode:
            raise ValueError("formal checkpoint supervision_mode does not match config model")
        model_name = contract.name
        supervision_mode = contract.supervision_mode
    elif (model_name is None) != (supervision_mode is None):
        raise ValueError("formal checkpoint model identity must contain model_name and supervision_mode")
    elif model_name is not None and supervision_mode is not None:
        contract = get_model_contract(model_name, supervision_mode=supervision_mode)
        model_name = contract.name
        supervision_mode = contract.supervision_mode
    metadata: dict[str, Any] = {
        "run_type": run_state,
        "run_state": run_state,
        "alignment_evidence": deepcopy(validated_evidence),
        "epoch": state.epoch,
        "global_step": state.global_step,
        "best_validation_dice": state.best_validation_dice,
        "best_selection_dice": state.best_selection_dice,
        "best_selection_epoch": state.best_selection_epoch,
        "early_stop_reference_dice": state.early_stop_reference_dice,
        "checks_without_improvement": state.checks_without_improvement,
        "fold": state.fold,
        "config": resolved_config,
        "resolved_config": resolved_config,
        "plan_hash": resolved_plan_hash,
        "policies": resolved_policies,
        "rng_state": dict(capture_rng_state() if rng_state is None else rng_state),
        "scheduler_state": None if scheduler is None else _capture_scheduler_state(scheduler, state),
    }
    if model_name is not None and supervision_mode is not None:
        metadata["model_name"] = model_name
        metadata["supervision_mode"] = supervision_mode
    return save_checkpoint(model, optimizer, path, metadata, allowed_root=checkpoint_root)


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
    model_name: str | None = None,
    supervision_mode: str | None = None,
    checkpoint_root: str | Path | None = None,
    source_identity: Mapping[str, Any] | None = None,
    initialization_provenance: Mapping[str, Any] | None = None,
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
    if (model_name is None) != (supervision_mode is None):
        raise ValueError("expected model identity must contain model_name and supervision_mode")
    if model_name is not None and supervision_mode is not None:
        contract = get_model_contract(model_name, supervision_mode=supervision_mode)
        expected["model_name"] = contract.name
        expected["supervision_mode"] = contract.supervision_mode
    payload, _ = read_formal_payload(_resolve_output_path(checkpoint_path, allowed_root=checkpoint_root))
    _validate_payload(payload, expected)
    metadata = dict(payload["metadata"])
    stored_config = metadata.get("resolved_config", metadata.get("config", {}))
    if source_identity is not None:
        validate_source_identity(stored_config.get("source_identity"), source_identity)
        validate_initialization_provenance(stored_config.get("initialization_provenance"))
        validate_initialization_provenance(initialization_provenance)
        if stored_config.get("initialization_provenance") != initialization_provenance:
            raise ValueError("checkpoint initialization provenance does not match")
    elif stored_config.get("source_identity") is not None:
        raise ValueError("marked checkpoint requires explicit stage/source identity")
    validate_model_tensors(model, payload["model_state_dict"])
    actual_run_state = str(metadata.get("run_state", metadata.get("run_type", "")))
    actual_evidence = _resolve_contract(
        config=dict(metadata.get("resolved_config", metadata.get("config", {}))),
        run_state=actual_run_state,
        alignment_evidence=metadata.get("alignment_evidence"),
    )
    if not isinstance(stored_config, Mapping) or not isinstance(metadata.get("policies", {}), Mapping):
        raise ValueError("formal checkpoint config/policies must be mappings")
    state = FormalTrainerState(
        int(metadata["epoch"]),
        int(metadata["global_step"]),
        float(metadata["best_validation_dice"]),
        int(metadata["fold"]),
        float(metadata.get("best_selection_dice", -1.0)),
        int(metadata.get("best_selection_epoch", 0)),
        float(metadata.get("early_stop_reference_dice", -1.0)),
        int(metadata.get("checks_without_improvement", 0)),
    )
    scheduler_state = metadata.get("scheduler_state")
    if scheduler is not None and scheduler_state is None:
        raise ValueError("formal checkpoint is missing scheduler_state")
    _preflight_restore(optimizer, scheduler, metadata, payload["optimizer_state_dict"])
    restored = FormalCheckpointRestore(
        state=state,
        scheduler_step=int(scheduler_state["step"]) if scheduler is not None else max(0, state.epoch - 1),
        config=dict(stored_config), plan_hash=str(metadata.get("plan_hash", "")),
        policies=dict(metadata.get("policies", {})), run_state=actual_run_state,
        alignment_evidence=deepcopy(actual_evidence),
    )
    model_before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    optimizer_before = copy(optimizer.state)
    for parameter, value in optimizer.state.items():
        optimizer_before[parameter] = deepcopy(value)
    groups_before = [{**deepcopy({k: v for k, v in g.items() if k != "params"}), "params": list(g["params"])} for g in optimizer.param_groups]
    scheduler_before = None if scheduler is None else deepcopy({k: v for k, v in scheduler.__dict__.items() if k != "optimizer"})
    rng_before = capture_rng_state()
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
        if payload["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler_step = (_restore_scheduler_state(scheduler, scheduler_state) if scheduler is not None else max(0, state.epoch - 1))
        if metadata.get("rng_state") is not None:
            _restore_rng_state(metadata["rng_state"])
    except BaseException as error:
        try:
            # Direct restoration bypasses model/optimizer/scheduler loading hooks.
            with torch.no_grad():
                for k, v in model.state_dict().items(): v.copy_(model_before[k])
            optimizer.state = optimizer_before
            optimizer.param_groups = groups_before
            if scheduler is not None:
                scheduler.__dict__.clear()
                scheduler.__dict__.update(scheduler_before, optimizer=optimizer)
            random.setstate(rng_before["python"])
            np.random.set_state(rng_before["numpy"])
            torch.set_rng_state(rng_before["torch_cpu"])
            if rng_before["torch_cuda"]: torch.cuda.set_rng_state_all(rng_before["torch_cuda"])
        except BaseException as rollback_error:
            error.add_note(f"formal checkpoint rollback failed: {rollback_error!r}")
            raise error from rollback_error
        raise
    return restored


def read_formal_payload(path: Path) -> tuple[dict[str, Any], str]:
    """Restricted deserialization of the existing schema, including NumPy RNG."""
    raw = Path(path).resolve().read_bytes()
    numpy_core = getattr(np, "_core", np.core)
    globals_ = (*_numpy_weights_only_safe_globals(), numpy_core.multiarray._reconstruct,
                np.ndarray, type(np.dtype(np.uint32)))
    with torch.serialization.safe_globals(globals_):
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    _validate_payload(payload, None)
    metadata = payload["metadata"]
    if "config" in metadata and "resolved_config" in metadata and metadata["config"] != metadata["resolved_config"]:
        raise ValueError("checkpoint config aliases conflict")
    return payload, hashlib.sha256(raw).hexdigest()


def validate_source_schema(source: Any) -> None:
    required = {"stage", "dataset_id", "source_version", "split_sha256", "cases_sha256", "plans_sha256", "dataset_metadata_sha256", "patient_map_sha256", "type", "root", "channels", "labels", "patch_size"}
    if not isinstance(source, Mapping) or not required.issubset(source):
        raise ValueError("source identity schema is incomplete")
    for key in ("dataset_id", "source_version", "root"):
        if not isinstance(source[key], str) or not source[key].strip() or "\x00" in source[key]:
            raise ValueError(f"source identity invalid {key}")
    for key in ("split_sha256", "cases_sha256", "plans_sha256", "dataset_metadata_sha256", "patient_map_sha256"):
        if key == "patient_map_sha256" and source[key] is None: continue
        if not isinstance(source[key], str) or re.fullmatch(r"[0-9a-f]{64}", source[key]) is None:
            raise ValueError(f"source identity invalid {key}")
    if source["stage"] not in ("pretrain", "finetune") or source["type"] not in ("raw_nifti_online", "nnunet_preprocessed_b2nd"):
        raise ValueError("source identity invalid stage/type")
    if (type(source["channels"]) is not int or source["channels"] != 1
        or source["labels"] != {"background": 0, "lesion": 1}
        or any(type(v) is not int for v in source["labels"].values())
        or source["patch_size"] != [512, 512] or any(type(v) is not int for v in source["patch_size"])):
        raise ValueError("source identity invalid channel/label/patch contract")
    dataset501 = source["dataset_id"].lower().startswith("dataset501") or source["dataset_id"] == "501"
    if dataset501 != (source["stage"] == "finetune"):
        raise ValueError("source identity stage/dataset contract mismatch")


def validate_source_identity(actual: Mapping[str, Any] | None, expected: Mapping[str, Any]) -> None:
    if actual is None: raise ValueError("historical checkpoint cannot resume an explicit stage")
    validate_source_schema(actual)
    validate_source_schema(expected)
    # Only the data root may relocate; no source filesystem access is required.
    if {k: v for k, v in actual.items() if k != "root"} != {k: v for k, v in expected.items() if k != "root"}:
        raise ValueError("checkpoint source identity does not match")


def validate_pretrained_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    _validate_payload(payload, {"model_name": "h2former", "supervision_mode": "single_output"})
    config = payload["metadata"].get("resolved_config", {})
    if not isinstance(config, Mapping) or config.get("model") != get_model_contract("h2former").as_dict():
        raise ValueError("pretrained model contract is not base H2Former")
    validate_source_schema(config.get("source_identity"))
    if config["source_identity"]["stage"] != "pretrain":
        raise ValueError("pretrained checkpoint requires external pretrain source contract")
    _resolve_contract(config=config, run_state=payload["metadata"].get("run_state"), alignment_evidence=payload["metadata"].get("alignment_evidence"))
    return config


def validate_initialization_provenance(provenance: Any) -> None:
    if provenance is None: return
    try:
        if not isinstance(provenance, Mapping): raise ValueError("must be a mapping")
        if not isinstance(provenance.get("path"), str) or not provenance["path"].strip() or "\x00" in provenance["path"]: raise ValueError("invalid path")
        if not isinstance(provenance.get("sha256"), str) or re.fullmatch(r"[0-9a-f]{64}", provenance["sha256"]) is None: raise ValueError("invalid SHA256")
        if provenance.get("model") != get_model_contract("h2former").as_dict(): raise ValueError("invalid model")
        validate_source_schema(provenance.get("source_identity"))
        if provenance["source_identity"]["stage"] != "pretrain": raise ValueError("invalid source stage")
    except ValueError as error:
        raise ValueError(f"initialization provenance contract: {error}") from error


def _validate_optimizer_schema(optimizer: Optimizer, saved: Any) -> None:
    """Validate the project's AdamW/SGD contract before native state casting.

    Parameter IDs are positional, as in Optimizer.load_state_dict. Optional
    historical flags use the same defaults as Adam/SGD.__setstate__; required
    arithmetic fields must be present. Never compare a decayed LR to initial LR.
    """
    adamw = type(optimizer) is torch.optim.AdamW
    if not adamw and type(optimizer) is not torch.optim.SGD:
        raise ValueError("formal checkpoint unsupported optimizer; expected AdamW or SGD")
    if not isinstance(saved, Mapping) or set(saved) != {"state", "param_groups"}:
        raise ValueError("formal checkpoint invalid optimizer schema")
    states, groups = saved["state"], saved["param_groups"]
    if not isinstance(states, Mapping) or not isinstance(groups, list) or len(groups) != len(optimizer.param_groups):
        raise ValueError("formal checkpoint invalid optimizer state/groups")

    def number(value: Any, name: str, *, upper: float | None = None) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (upper is not None and value >= upper):
            raise ValueError(f"formal checkpoint invalid optimizer {name}")
        return float(value)

    def moment(value: Any, parameter: torch.Tensor, name: str) -> None:
        if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape
            or value.layout != torch.strided or value.dtype != parameter.dtype
            or not torch.isfinite(value).all().item()):
            raise ValueError(f"formal checkpoint incompatible optimizer {name}")
        # In-place moments require distinct element addresses, not contiguity.
        # Ignore singleton axes (and empty tensors): they cannot alias elements.
        axes = sorted((stride, size) for size, stride in zip(value.shape, value.stride()) if size > 1)
        if value.numel() > 1:
            span = 1
            for stride, size in axes:
                if stride == 0:
                    raise ValueError(f"formal checkpoint optimizer {name} internal overlap")
                if stride < span:
                    # The sorted-stride span proof is inconclusive, just like
                    # PyTorch's TooHard result. Resolve actual element offsets;
                    # neither reject valid interleaving nor assume it is safe.
                    offsets = {0}
                    for axis_stride, axis_size in axes:
                        expanded = set()
                        for offset in offsets:
                            for index in range(axis_size):
                                address = offset + index * axis_stride
                                if address in expanded:
                                    raise ValueError(f"formal checkpoint optimizer {name} internal overlap")
                                expanded.add(address)
                        offsets = expanded
                    break
                span += (size - 1) * stride
        if name in ("exp_avg_sq", "max_exp_avg_sq") and (value < 0).any().item():
            raise ValueError(f"formal checkpoint invalid optimizer {name}")

    id_map: dict[int, tuple[torch.Tensor, Mapping[str, Any]]] = {}
    target_ids: set[int] = set()
    for group, target in zip(groups, optimizer.param_groups):
        if not isinstance(group, Mapping) or not isinstance(group.get("params"), list) or len(group["params"]) != len(target["params"]):
            raise ValueError("formal checkpoint incompatible optimizer parameter group")
        required = {"lr", "weight_decay", "betas", "eps"} if adamw else {"lr", "weight_decay", "momentum", "dampening"}
        if not required.issubset(group):
            raise ValueError("formal checkpoint missing optimizer group field")
        for key in ("lr", "weight_decay", "initial_lr"):
            if key in group: number(group[key], key)
        flags = {"maximize": False, "differentiable": False, "foreach": None, "fused": None if adamw else False}
        flags.update({"amsgrad": False, "capturable": False, "decoupled_weight_decay": True} if adamw else {"nesterov": False})
        options = {}
        for key, default in flags.items():
            value = group.get(key, default)
            if type(value) is not bool and not (key in ("foreach", "fused") and value is None):
                raise ValueError(f"formal checkpoint invalid optimizer {key}")
            options[key] = value
        if options["fused"] and (options["foreach"] or options["differentiable"]):
            raise ValueError("formal checkpoint invalid optimizer fused combination")
        # These execution modes are not enabled by make_official_optimizer.
        if options["differentiable"] or (adamw and options["capturable"]):
            raise ValueError("formal checkpoint unsupported optimizer execution mode")
        if adamw:
            betas = group["betas"]
            if not isinstance(betas, (tuple, list)) or len(betas) != 2:
                raise ValueError("formal checkpoint invalid optimizer betas")
            for beta in betas: number(beta, "betas", upper=1.)
            number(group["eps"], "eps")
        else:
            momentum = number(group["momentum"], "momentum")
            dampening = number(group["dampening"], "dampening")
            if options["nesterov"] and (momentum <= 0 or dampening != 0):
                raise ValueError("formal checkpoint invalid optimizer nesterov combination")
        if "param_names" in group and (not isinstance(group["param_names"], list) or len(group["param_names"]) != len(group["params"]) or any(not isinstance(n, str) for n in group["param_names"])):
            raise ValueError("formal checkpoint invalid optimizer param_names")
        for key, parameter in zip(group["params"], target["params"]):
            if type(key) is not int or key < 0 or key in id_map or id(parameter) in target_ids:
                raise ValueError("formal checkpoint duplicate/invalid optimizer parameter reference")
            id_map[key] = (parameter, options)
            target_ids.add(id(parameter))
    for key, values in states.items():
        if type(key) is not int or key not in id_map:
            raise ValueError("formal checkpoint unknown optimizer state reference")
        parameter, options = id_map[key]
        if not isinstance(values, Mapping):
            raise ValueError("formal checkpoint invalid optimizer parameter state")
        if not values: continue  # Native lazy initialization, including SGD without momentum.
        if adamw:
            required = {"step", "exp_avg", "exp_avg_sq"}
            if options["amsgrad"]: required.add("max_exp_avg_sq")
            if not required.issubset(values) or not set(values).issubset({"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"}):
                raise ValueError("formal checkpoint incomplete/unknown optimizer AdamW state")
            step = values["step"]
            if isinstance(step, torch.Tensor):
                if step.ndim != 0 or step.layout != torch.strided or step.dtype not in (torch.float32, torch.float64) or step.requires_grad:
                    raise ValueError("formal checkpoint invalid optimizer step")
                step = step.item()
            # Older Adam checkpoints stored step as a Python number.
            step = number(step, "step")
            if not step.is_integer(): raise ValueError("formal checkpoint invalid optimizer step")
            for name in set(values) - {"step"}: moment(values[name], parameter, name)
        else:
            if set(values) != {"momentum_buffer"}:
                raise ValueError("formal checkpoint invalid optimizer SGD state")
            moment(values["momentum_buffer"], parameter, "momentum_buffer")

def _preflight_restore(optimizer: Optimizer, scheduler: Any, metadata: Mapping[str, Any], optimizer_state: Any) -> None:
    for key in ("epoch", "global_step", "fold", "best_selection_epoch", "checks_without_improvement"):
        value = metadata.get(key, 0) if key in ("best_selection_epoch", "checks_without_improvement") else metadata.get(key)
        if type(value) is not int or value < 0: raise ValueError(f"formal checkpoint invalid {key}")
    for key in ("best_validation_dice", "best_selection_dice", "early_stop_reference_dice"):
        value = metadata.get(key, -1.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value): raise ValueError(f"formal checkpoint invalid {key}")
    if optimizer_state is not None:
        _validate_optimizer_schema(optimizer, optimizer_state)
        probe = copy(optimizer)
        probe.__dict__ = dict(optimizer.__dict__)
        probe.param_groups = [{**{k: deepcopy(v) for k, v in g.items() if k != "params"}, "params": list(g["params"])} for g in optimizer.param_groups]
        probe.state = copy(optimizer.state)
        probe._optimizer_load_state_dict_pre_hooks = {}
        probe._optimizer_load_state_dict_post_hooks = {}
        probe.load_state_dict(optimizer_state)
        for parameter, values in probe.state.items():
            if not isinstance(values, Mapping):
                raise ValueError("formal checkpoint invalid optimizer state")
            for value in values.values():
                if isinstance(value, torch.Tensor):
                    if (value.ndim != 0 and value.shape != parameter.shape) or not torch.isfinite(value).all().item():
                        raise ValueError("formal checkpoint incompatible optimizer tensor")
        del probe
    if scheduler is not None:
        saved = metadata.get("scheduler_state")
        if not isinstance(saved, Mapping): raise ValueError("formal checkpoint is missing scheduler_state")
        for key in ("step", "ctr"):
            value = saved.get(key) if key == "step" else saved.get(key, 0)
            if type(value) is not int or value < 0: raise ValueError(f"formal checkpoint invalid scheduler {key}")
        lrs = saved.get("last_lr", [])
        if not isinstance(lrs, list) or (lrs and len(lrs) != len(optimizer.param_groups)) or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in lrs): raise ValueError("formal checkpoint invalid scheduler last_lr")
    rng = metadata.get("rng_state")
    if rng is not None:
        try:
            random.Random().setstate(rng["python"])
            np.random.RandomState().set_state(rng["numpy"])
            torch.Generator(device="cpu").set_state(rng["torch_cpu"])
            cuda = rng.get("torch_cuda", [])
            if not isinstance(cuda, list): raise ValueError("CUDA RNG must be a list")
            if torch.cuda.is_available() and cuda and len(cuda) != torch.cuda.device_count(): raise ValueError("CUDA RNG device count mismatch")
            for value in cuda:
                if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1: raise ValueError("invalid CUDA RNG tensor")
        except (KeyError, IndexError, TypeError, ValueError, RuntimeError) as error:
            raise ValueError(f"formal checkpoint invalid RNG state: {error}") from error


def validate_model_tensors(model: nn.Module, weights: Any) -> None:
    target = model.state_dict()
    if not isinstance(weights, Mapping) or set(weights) != set(target):
        raise ValueError("model state keys do not match")
    for key, value in weights.items():
        if not isinstance(value, torch.Tensor) or value.shape != target[key].shape:
            raise ValueError(f"model state shape mismatch: {key}")
        if value.dtype != target[key].dtype or value.layout != torch.strided:
            raise ValueError(f"model state dtype/layout mismatch: {key}")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"model state non-finite tensor: {key}")


def initialize_pretrained_model(model: nn.Module, path: Path) -> dict[str, Any]:
    """Business weights-only transfer: all parameters/buffers, no training state."""
    payload, digest = read_formal_payload(path)
    config = validate_pretrained_payload(payload)
    source = config["source_identity"]
    validate_model_tensors(model, payload["model_state_dict"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return {"path": str(Path(path).resolve()), "sha256": digest,
            "model": deepcopy(config["model"]), "source_identity": deepcopy(source)}
