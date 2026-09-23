"""Synthetic engineering fit of active physical-mode H2Former phase-only PSFs.

This module fits only the per-route phase maps stored in an already-created
physical initialization artifact. It never trains the segmentation model.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from standalone_nnunet2d.engine.optoelectronic_initialization import (
    CHECKPOINT_FORMAT_VERSION,
    _canonical_json,
    _sha256_bytes,
    _sha256_file,
    _validate_artifact_payload,
    _write_artifacts_atomically,
    load_optoelectronic_initialization,
)
from standalone_nnunet2d.models.optoelectronic_frontend import (
    ActiveRoute,
    OptoelectronicConfig,
    PhaseOnlyPSF,
    PhasePSFConfig,
    SupportSweepResult,
    crop_centered,
    physical_psf_from_digital_lobe,
    phase_to_full_psf,
    support_sweep,
)
from standalone_nnunet2d.models.optoelectronic_h2former import (
    ENTRY_GROUPS,
    OptoelectronicH2Former,
    OptoelectronicPCAConv2d,
)

ARTIFACT_TYPE = "optoelectronic_h2former_psf_fit"
ARTIFACT_VERSION = 1
REPORT_SCHEMA = "optoelectronic_h2former_psf_fit_report"
REPORT_VERSION = 1
FIT_FILENAME = "optoelectronic_psf_fit.pth"
FIT_REPORT_FILENAME = "optoelectronic_psf_fit_report.json"

_ARTIFACT_KEYS = {
    "format_version",
    "artifact_type",
    "artifact_version",
    "model_state_dict",
    "optimizer_state_dict",
    "metadata",
}
_METADATA_KEYS = {
    "model_name",
    "supervision_mode",
    "fit_only",
    "initialization_only",
    "resume_eligible",
    "evidence_class",
    "source_initialization_sha256",
    "source_checkpoint_sha256",
    "creation_identity_digest",
    "initialization_metadata_sha256",
    "resolved_optoelectronic_config",
    "fit_config",
    "seed",
    "device",
    "dtype",
    "group_order",
    "route_order",
    "route_count",
    "theta_digests",
    "optimizer_parameter_names",
    "non_theta_state_sha256",
    "global_ledger_identity",
    "physical_assumptions",
    "fit_status",
    "route_metrics",
    "aggregate_metrics",
    "claims",
    "metadata_sha256",
}
_REPORT_KEYS = {
    "report_schema",
    "report_version",
    "artifact_type",
    "artifact_version",
    "model_name",
    "supervision_mode",
    "fit_only",
    "resume_eligible",
    "evidence_class",
    "fit_status",
    "source_initialization_sha256",
    "source_checkpoint_sha256",
    "creation_identity_digest",
    "initialization_metadata_sha256",
    "resolved_optoelectronic_config",
    "fit_config",
    "group_order",
    "route_order",
    "route_count",
    "routes",
    "aggregate_metrics",
    "theta_digests",
    "optimizer_parameter_names",
    "non_theta_state_sha256",
    "global_ledger_identity",
    "physical_assumptions",
    "claims",
    "metadata_sha256",
    "limitations",
}


@dataclass(frozen=True)
class OptoelectronicPSFFitConfig:
    """Strict fixed-step optimizer and quality-report configuration."""

    steps: int = 100
    learning_rate: float = 0.01
    seed: int = 0
    phase_init_std: float = 0.0
    shape_weight: float = 1.0
    throughput_weight: float = 0.0
    gain_weight: float = 0.0
    max_grad_norm: float | None = None
    device: str = "cpu"
    map_location: str = "cpu"
    max_normalized_mse: float | None = None
    min_cosine_similarity: float | None = None
    quality_threshold_mode: str = "report_only"

    def __post_init__(self) -> None:
        if type(self.steps) is not int or self.steps <= 0:
            raise ValueError("steps must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        for name in ("learning_rate", "phase_init_std", "shape_weight", "throughput_weight", "gain_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, float(value))
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.phase_init_std < 0:
            raise ValueError("phase_init_std must be non-negative")
        if min(self.shape_weight, self.throughput_weight, self.gain_weight) < 0:
            raise ValueError("loss weights must be non-negative")
        if self.shape_weight + self.throughput_weight + self.gain_weight <= 0:
            raise ValueError("at least one loss weight must be greater than zero")
        if self.max_grad_norm is not None:
            value = self.max_grad_norm
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
                raise ValueError("max_grad_norm must be finite and positive or None")
            object.__setattr__(self, "max_grad_norm", float(value))
        if self.max_normalized_mse is not None:
            value = self.max_normalized_mse
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
                raise ValueError("max_normalized_mse must be finite and non-negative or None")
            object.__setattr__(self, "max_normalized_mse", float(value))
        if self.min_cosine_similarity is not None:
            value = self.min_cosine_similarity
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not -1 <= value <= 1:
                raise ValueError("min_cosine_similarity must be finite and in [-1, 1] or None")
            object.__setattr__(self, "min_cosine_similarity", float(value))
        if type(self.device) is not str or not self.device:
            raise ValueError("device must be a non-empty torch device string")
        if type(self.map_location) is not str or not self.map_location:
            raise ValueError("map_location must be a non-empty torch device string")
        try:
            device = torch.device(self.device)
            map_device = torch.device(self.map_location)
        except (TypeError, RuntimeError) as error:
            raise ValueError("device and map_location must be valid torch device strings") from error
        if device.type == "meta" or map_device.type == "meta":
            raise ValueError("meta devices are not valid fit or artifact map locations")
        if self.quality_threshold_mode not in {"report_only", "fail_closed"}:
            raise ValueError("quality_threshold_mode must be report_only or fail_closed")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OptoelectronicPSFFitConfig":
        if not isinstance(value, Mapping):
            raise ValueError("fit config must be a mapping")
        known = {field.name for field in fields(cls)}
        required = {"steps", "learning_rate", "seed", "phase_init_std", "shape_weight", "throughput_weight", "gain_weight"}
        supplied = set(value)
        missing = required - supplied
        unexpected = supplied - known
        if missing or unexpected:
            raise ValueError(
                "fit config keys do not match schema: "
                f"missing={sorted(missing)!r}, unexpected={sorted(unexpected)!r}"
            )
        resolved = cls().as_dict()
        resolved.update(dict(value))
        return cls(**resolved)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OptoelectronicPSFFitResult:
    artifact_path: Path
    report_path: Path
    report: dict[str, Any]


def _strict_json_clone(value: Any) -> Any:
    value_type = type(value)
    if value is None:
        return None
    if value_type is bool or value_type is int or value_type is str:
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError("value is not a finite JSON tree: non-finite float")
        return value
    if value_type is list:
        return [_strict_json_clone(item) for item in value]
    if value_type is dict:
        cloned: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("value is not a finite JSON tree: object keys must be exact built-in strings")
            cloned[key] = _strict_json_clone(item)
        return cloned
    raise ValueError("value is not a finite JSON tree: unsupported or non-exact JSON type")


def _json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _metadata_digest(metadata: Mapping[str, Any]) -> str:
    view = copy.deepcopy(dict(metadata))
    view.pop("metadata_sha256", None)
    return _json_sha256(view)


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _tensor_sha256(tensor: Tensor) -> str:
    if type(tensor) is not torch.Tensor:
        raise ValueError("digest input must be an exact plain Tensor")
    if tensor.is_meta or tensor.is_quantized or tensor.layout is not torch.strided:
        raise ValueError("digest input must be a materialized strided non-quantized Tensor")
    if not torch.isfinite(tensor).all():
        raise ValueError("digest input must be finite")
    raw = tensor.detach().to(device="cpu").contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _is_theta_key(name: str) -> bool:
    return ".phase_psfs." in name and name.endswith(".theta")


def _clone_state_to_cpu(state: Mapping[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for name, value in state.items():
        if type(name) is not str:
            raise ValueError("state dictionary keys must be exact strings")
        if isinstance(value, Tensor):
            if value.is_meta or value.is_quantized or value.layout is not torch.strided:
                raise ValueError(f"state entry {name!r} has unsupported tensor storage")
            cloned[name] = value.detach().to(device="cpu").clone()
        else:
            cloned[name] = copy.deepcopy(value)
    return cloned


def _state_value_equal(expected: Any, actual: Any) -> bool:
    if isinstance(expected, Tensor):
        if (
            type(expected) is not torch.Tensor
            or type(actual) is not torch.Tensor
            or expected.shape != actual.shape
            or expected.dtype != actual.dtype
        ):
            return False
        if expected.is_meta or expected.is_quantized or expected.layout is not torch.strided:
            return False
        if actual.is_meta or actual.is_quantized or actual.layout is not torch.strided:
            return False
        return torch.equal(expected.detach().to("cpu"), actual.detach().to("cpu"))
    try:
        expected_tree = _strict_json_clone(expected)
        actual_tree = _strict_json_clone(actual)
        return _canonical_json({"value": expected_tree}) == _canonical_json({"value": actual_tree})
    except ValueError:
        return False


def _state_digest(state: Mapping[str, Any], *, exclude_theta: bool) -> str:
    entries: list[dict[str, Any]] = []
    for name in sorted(state):
        if exclude_theta and _is_theta_key(name):
            continue
        value = state[name]
        if isinstance(value, Tensor):
            entries.append({
                "name": name,
                "kind": "tensor",
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": _tensor_sha256(value),
            })
        else:
            entries.append({"name": name, "kind": "json", "value": _strict_json_clone(value)})
    return _json_sha256({"entries": entries})


@contextmanager
def _preserve_torch_rng_state() -> Iterator[None]:
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() and torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


@contextmanager
def _rng_guard() -> Iterator[None]:
    with _preserve_torch_rng_state():
        yield


def _device_for_fit(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(f"requested device {device} is unavailable")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"requested device {device} does not exist")
    return device


def _physical_target_for_route(route: ActiveRoute) -> Tensor:
    """Flip the digital correlation lobe once, then normalize target energy."""

    physical = physical_psf_from_digital_lobe(route.lobe)
    while physical.ndim > 2 and physical.shape[0] == 1:
        physical = physical.squeeze(0)
    if physical.ndim != 2 or not torch.isfinite(physical).all() or torch.any(physical < 0):
        raise ValueError(f"route {route.route_id!r} produced an invalid physical target")
    energy = physical.sum()
    if not torch.isfinite(energy) or float(energy.detach().item()) <= 0:
        raise ValueError(f"route {route.route_id!r} physical target has no energy")
    return physical / energy


def _differentiable_route_objective(
    full_psf: Tensor,
    physical_target: Tensor,
    *,
    support: int,
    route: ActiveRoute,
    rho: float,
    beta: float,
    physical_config: OptoelectronicConfig,
    fit_config: OptoelectronicPSFFitConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return the differentiable shape/throughput/gain surrogate objective."""

    if full_psf.ndim != 2 or not torch.is_floating_point(full_psf) or not torch.isfinite(full_psf).all():
        raise ValueError("full_psf must be a finite floating-point [H,W] tensor")
    if physical_target.ndim != 2 or not torch.is_floating_point(physical_target) or not torch.isfinite(physical_target).all():
        raise ValueError("physical_target must be a finite floating-point [H,W] tensor")
    if tuple(physical_target.shape) != (support, support):
        raise ValueError("physical target shape must equal the cropped convolution support")
    if not torch.isfinite(torch.tensor([rho, beta, route.alpha])).all() or rho <= 0 or beta <= 0 or route.alpha <= 0:
        raise ValueError("rho, beta, and route alpha must be finite and positive")
    raw_crop = crop_centered(full_psf, support)
    tau = raw_crop.sum()
    if not torch.isfinite(tau) or float(tau.detach().item()) <= 0:
        raise ValueError("raw cropped PSF throughput must be finite and positive")
    normalized_crop = raw_crop / tau
    target = physical_target.to(device=full_psf.device, dtype=full_psf.dtype)
    shape_loss = (normalized_crop - target).square().mean()
    throughput_loss = torch.relu(full_psf.new_tensor(physical_config.min_throughput) - tau).square()
    gain = full_psf.new_tensor(float(route.alpha) / (float(rho) * float(beta))) / tau
    gain_loss = torch.relu(gain.abs() - full_psf.new_tensor(physical_config.max_electronic_gain)).square()
    target_mean_square = target.square().mean().clamp_min(torch.finfo(target.dtype).tiny)
    normalized_mse = shape_loss / target_mean_square
    cosine = F.cosine_similarity(normalized_crop.reshape(1, -1), target.reshape(1, -1), dim=1).squeeze(0)
    objective = (
        fit_config.shape_weight * shape_loss
        + fit_config.throughput_weight * throughput_loss
        + fit_config.gain_weight * gain_loss
    )
    values = {
        "shape": shape_loss,
        "throughput": throughput_loss,
        "gain_penalty": gain_loss,
        "tau": tau,
        "gain": gain,
        "normalized_mse": normalized_mse,
        "cosine_similarity": cosine,
    }
    if not torch.isfinite(objective) or any(not torch.isfinite(value) for value in values.values()):
        raise ValueError(f"route {route.route_id!r} objective contains NaN or infinity")
    return objective, values


def _support_sizes(support: int, phase_grid_size: int) -> tuple[int, ...]:
    if type(support) is not int or support <= 0:
        raise ValueError("support must be a positive integer")
    if type(phase_grid_size) is not int or phase_grid_size < support:
        raise ValueError("phase grid must be at least as large as the convolution support")
    return tuple(dict.fromkeys((support, min(2 * support, phase_grid_size), phase_grid_size)))


def _support_sweep_for_psf(
    full_psf: Tensor,
    support: int,
    *,
    seed: int,
    physical_config: OptoelectronicConfig,
) -> SupportSweepResult:
    if full_psf.ndim != 2:
        raise ValueError("full_psf must be two-dimensional")
    grid = int(full_psf.shape[-1])
    if full_psf.shape[-2] != grid:
        raise ValueError("full_psf must be square")
    supports = _support_sizes(support, grid)
    return support_sweep(
        full_psf.detach(),
        supports,
        seed=seed,
        max_support_response_error=physical_config.max_support_response_error,
        mode=physical_config.normalized_support_error_mode,
    )


def _check_quality_thresholds(
    metrics: Mapping[str, float], config: OptoelectronicPSFFitConfig
) -> list[str]:
    violations: list[str] = []
    if config.max_normalized_mse is not None and metrics["normalized_mse"] > config.max_normalized_mse:
        violations.append(
            f"normalized_mse {metrics['normalized_mse']:.9g} exceeds {config.max_normalized_mse:.9g}"
        )
    if config.min_cosine_similarity is not None and metrics["cosine_similarity"] < config.min_cosine_similarity:
        violations.append(
            f"cosine_similarity {metrics['cosine_similarity']:.9g} is below {config.min_cosine_similarity:.9g}"
        )
    if violations and config.quality_threshold_mode == "fail_closed":
        raise ValueError("fit quality threshold failure: " + "; ".join(violations))
    return violations


def _check_final_physical_limits(
    metrics: Mapping[str, float], physical_config: OptoelectronicConfig
) -> None:
    tau = metrics.get("tau")
    gain = metrics.get("gain")
    if not isinstance(tau, (int, float)) or not math.isfinite(float(tau)):
        raise ValueError("final physical throughput is not finite")
    if not isinstance(gain, (int, float)) or not math.isfinite(float(gain)):
        raise ValueError("final physical gain is not finite")
    if float(tau) < physical_config.min_throughput:
        raise ValueError(
            f"final route throughput {float(tau):.9g} is below min_throughput "
            f"{physical_config.min_throughput:.9g}"
        )
    if abs(float(gain)) > physical_config.max_electronic_gain:
        raise ValueError(
            f"final route gain {float(gain):.9g} exceeds max_electronic_gain "
            f"{physical_config.max_electronic_gain:.9g}"
        )


def _perturb_initial_theta(theta: Tensor, *, std: float, seed: int) -> Tensor:
    if not math.isfinite(float(std)) or std < 0 or type(seed) is not int or seed < 0:
        raise ValueError("phase perturbation std and seed are invalid")
    base = theta.detach()
    if std == 0:
        return base.clone()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    noise = torch.randn(
        base.shape,
        dtype=base.dtype,
        device="cpu",
        generator=generator,
    ).to(device=base.device)
    result = base + float(std) * noise
    if not torch.isfinite(result).all():
        raise ValueError("phase initialization perturbation is not finite")
    return result


def _route_seed(seed: int, route_id: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{route_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _physical_metrics_from_theta(
    theta: Tensor,
    phase_psf: PhaseOnlyPSF,
    target: Tensor,
    *,
    support: int,
    route: ActiveRoute,
    rho: float,
    beta: float,
    physical_config: OptoelectronicConfig,
    fit_config: OptoelectronicPSFFitConfig,
) -> tuple[dict[str, float], Tensor]:
    cpu_theta = theta.detach().to(device="cpu")
    aperture = phase_psf.aperture.detach().to(device="cpu", dtype=cpu_theta.dtype)
    full_psf = phase_to_full_psf(cpu_theta, config=phase_psf.config, aperture=aperture)
    if not torch.isfinite(full_psf).all() or torch.any(full_psf < 0):
        raise ValueError(f"route {route.route_id!r} generated an invalid full PSF")
    full_sum = full_psf.sum()
    if not torch.isfinite(full_sum) or not torch.allclose(
        full_sum, full_sum.new_tensor(1.0), rtol=1e-6, atol=1e-6
    ):
        raise ValueError(f"route {route.route_id!r} full PSF does not have unit energy")
    objective, values = _differentiable_route_objective(
        full_psf,
        target.to(device="cpu", dtype=full_psf.dtype),
        support=support,
        route=route,
        rho=rho,
        beta=beta,
        physical_config=physical_config,
        fit_config=fit_config,
    )
    metrics = {
        "objective": float(objective.item()),
        "loss_shape": float(values["shape"].item()),
        "loss_throughput": float(values["throughput"].item()),
        "loss_gain": float(values["gain_penalty"].item()),
        "normalized_mse": float(values["normalized_mse"].item()),
        "cosine_similarity": float(values["cosine_similarity"].item()),
        "tau": float(values["tau"].item()),
        "leakage": float(1.0 - values["tau"].item()),
        "gain": float(values["gain"].item()),
        "full_psf_sum": float(full_sum.item()),
        "full_psf_min": float(full_psf.min().item()),
        "full_psf_max": float(full_psf.max().item()),
    }
    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError(f"route {route.route_id!r} metrics contain NaN or infinity")
    return metrics, full_psf


def _active_route_records(
    model: OptoelectronicH2Former,
) -> list[dict[str, Any]]:
    if model.opto_config.mode != "physical":
        raise ValueError("PSF fitting accepts only physical-mode initialization artifacts")
    if model.exposure_ledger is None:
        raise ValueError("physical initialization has no ExposureLedger")
    model.exposure_ledger.validate()
    entries = model._entry_modules()
    if tuple(module.group_id for module in entries) != tuple(ENTRY_GROUPS):
        raise ValueError("physical initialization does not cover the five canonical entry groups")
    entry_by_group = {module.group_id: module for module in entries}
    prefixes = {
        group: ("conv1" if index == 0 else f"patch_embed.projs.{index - 1}")
        for index, group in enumerate(ENTRY_GROUPS)
    }
    records: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for route in model.exposure_ledger.canonical_routes:
        module = entry_by_group.get(route.group_id)
        if module is None or module.mode != "physical":
            raise ValueError(f"route {route.route_id!r} is not in a physical entry group")
        registered = module._routes_by_component.get(route.component_index, {}).get(route.sign)
        if registered is not route:
            raise ValueError(f"route {route.route_id!r} is not the canonical active route object")
        phase_key = module._phase_key(route.component_index, route.sign)
        if phase_key not in module.phase_psfs:
            raise ValueError(f"route {route.route_id!r} is missing its registered PhaseOnlyPSF")
        phase_psf = module.phase_psfs[phase_key]
        if not isinstance(phase_psf, PhaseOnlyPSF):
            raise ValueError(f"route {route.route_id!r} has no PhaseOnlyPSF")
        state_key = f"{prefixes[route.group_id]}.phase_psfs.{phase_key}.theta"
        if state_key in seen_keys:
            raise ValueError(f"active routes share theta state key {state_key!r}")
        seen_keys.add(state_key)
        records.append({
            "route": route,
            "module": module,
            "phase_psf": phase_psf,
            "state_key": state_key,
            "support": int(module.kernel_size[0]),
        })
    covered_groups = {record["route"].group_id for record in records}
    if covered_groups != set(ENTRY_GROUPS):
        raise ValueError(
            "physical initialization must have active routes in all five entry groups; "
            f"covered={sorted(covered_groups)!r}"
        )
    actual_theta_keys = {
        name for name, _ in model.named_parameters() if _is_theta_key(name)
    }
    if actual_theta_keys != seen_keys:
        raise ValueError("active route theta keys do not exactly match model theta parameters")
    if not records:
        raise ValueError("physical initialization has no active routes to fit")
    return records


def _aggregate_route_metrics(routes: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "objective",
        "loss_shape",
        "loss_throughput",
        "loss_gain",
        "normalized_mse",
        "cosine_similarity",
        "tau",
        "leakage",
        "gain",
    )
    values: dict[str, float] = {}
    for name in metric_names:
        observations = [float(route["final"][name]) for route in routes.values()]
        values[f"mean_{name}"] = float(sum(observations) / len(observations))
        values[f"max_{name}"] = float(max(observations))
        values[f"min_{name}"] = float(min(observations))
    values["max_support_response_error"] = max(
        float(report["response_error"])
        for route in routes.values()
        for report in route["support_sweep"]["reports"]
    )
    values["route_count"] = len(routes)
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("aggregate fit metrics contain NaN or infinity")
    return values


def _physical_assumptions() -> dict[str, Any]:
    return {
        "evidence_class": "synthetic_engineering_psf_fit",
        "propagation_model": "synthetic monochromatic spatially incoherent ideal 4f phase-only model",
        "digital_target_conversion": "physical_psf_from_digital_lobe applies one 180-degree spatial flip",
        "cropped_detector_kernel": "raw centered crop h_K; normalized q_K is used only for shape loss and reporting",
        "crop_semantics": "K crop is a numerical approximation; energy outside support remains in the full PSF",
        "support_reference": "complete full-plane PSF is the support-sweep reference",
        "larger_support_semantics": "K, min(2K,N), and N are numerical reviews and do not change network convolution support",
        "gain_penalty_semantics": "fit-only differentiable surrogate through tau; runtime detector remains calibrated_detached",
        "normalized_mse_definition": "mean((q_K-p)^2) divided by mean(p^2)",
        "physical_validation": "not hardware, manufacturability, or measured propagation validation",
    }


def _claims() -> dict[str, bool]:
    return {
        "synthetic_engineering_fit": True,
        "hardware_validated": False,
        "manufacturability_validated": False,
        "measured_propagation_validated": False,
        "medical_performance": False,
        "gpu_acceleration_claim": False,
        "segmentation_training_checkpoint": False,
        "resume_eligible": False,
    }


def _build_report(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return _strict_json_clone({
        "report_schema": REPORT_SCHEMA,
        "report_version": REPORT_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "artifact_version": ARTIFACT_VERSION,
        "model_name": metadata["model_name"],
        "supervision_mode": metadata["supervision_mode"],
        "fit_only": metadata["fit_only"],
        "resume_eligible": metadata["resume_eligible"],
        "evidence_class": metadata["evidence_class"],
        "fit_status": metadata["fit_status"],
        "source_initialization_sha256": metadata["source_initialization_sha256"],
        "source_checkpoint_sha256": metadata["source_checkpoint_sha256"],
        "creation_identity_digest": metadata["creation_identity_digest"],
        "initialization_metadata_sha256": metadata["initialization_metadata_sha256"],
        "resolved_optoelectronic_config": metadata["resolved_optoelectronic_config"],
        "fit_config": metadata["fit_config"],
        "group_order": metadata["group_order"],
        "route_order": metadata["route_order"],
        "route_count": metadata["route_count"],
        "routes": metadata["route_metrics"],
        "aggregate_metrics": metadata["aggregate_metrics"],
        "theta_digests": metadata["theta_digests"],
        "optimizer_parameter_names": metadata["optimizer_parameter_names"],
        "non_theta_state_sha256": metadata["non_theta_state_sha256"],
        "global_ledger_identity": metadata["global_ledger_identity"],
        "physical_assumptions": metadata["physical_assumptions"],
        "claims": metadata["claims"],
        "metadata_sha256": metadata["metadata_sha256"],
        "limitations": [
            "K crop is a numerical approximation; support-external energy remains in the full PSF.",
            "Full support is the numerical reference; larger supports do not alter network convolution support or model identity.",
            "Five entry groups were fitted sequentially; they are not five metasurfaces.",
            "This is a synthetic engineering fit and does not validate hardware, manufacturability, medical performance, or GPU acceleration.",
            "A later segmentation joint fine-tuning experiment is separate from this fit artifact.",
        ],
    })


def _prepare_fit_output_paths(output_root: str | Path) -> tuple[Path, Path, Path]:
    root = Path(output_root).expanduser().resolve()
    if root.exists() and not root.is_dir():
        raise ValueError(f"output-root is not a directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    artifact_path = root / FIT_FILENAME
    report_path = root / FIT_REPORT_FILENAME
    existing = [str(path) for path in (artifact_path, report_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing fit output(s): {existing!r}")
    return root, artifact_path, report_path


def _publish_fit_artifacts(
    artifact_payload: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    output_root: str | Path,
) -> tuple[Path, Path]:
    root, artifact_path, report_path = _prepare_fit_output_paths(output_root)
    # Reuse the initialization artifact's hard-link-only no-replace publisher,
    # including its two-file rollback and temporary-file cleanup behavior.
    _write_artifacts_atomically(
        artifact_payload,
        report,
        output_root=root,
        artifact_path=artifact_path,
        report_path=report_path,
    )
    return artifact_path, report_path


def _read_initialization_metadata(path: Path, *, map_location: str) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as error:
        raise ValueError(f"failed to read initialization artifact: {path}") from error
    _, metadata = _validate_artifact_payload(payload)
    return dict(metadata)


def _fit_optoelectronic_h2former_psf_impl(
    initialization_artifact: str | Path,
    source_checkpoint: str | Path,
    fit_config: OptoelectronicPSFFitConfig,
    output_root: str | Path,
) -> OptoelectronicPSFFitResult:
    if not isinstance(fit_config, OptoelectronicPSFFitConfig):
        raise ValueError("fit_config must be an OptoelectronicPSFFitConfig")
    output_root_path, artifact_path, report_path = _prepare_fit_output_paths(output_root)
    initialization_path = Path(initialization_artifact).expanduser().resolve()
    source_path = Path(source_checkpoint).expanduser().resolve()
    if not initialization_path.is_file():
        raise FileNotFoundError(f"initialization artifact does not exist: {initialization_path}")
    if not source_path.is_file():
        raise FileNotFoundError(f"source checkpoint does not exist: {source_path}")
    initialization_sha256 = _sha256_file(initialization_path)
    initialization_metadata = _read_initialization_metadata(
        initialization_path, map_location=fit_config.map_location
    )
    source_sha256 = _sha256_file(source_path)
    if source_sha256 != initialization_metadata["source_checkpoint"]["sha256"]:
        raise ValueError("source checkpoint SHA256 does not match the initialization artifact")
    model = load_optoelectronic_initialization(
        initialization_path,
        source_path,
        map_location=fit_config.map_location,
    )
    if model.opto_config.mode != "physical":
        raise ValueError("PSF fitting accepts only physical-mode initialization artifacts")
    device = _device_for_fit(fit_config.device)
    records = _active_route_records(model)
    physical_identity = model.identity_metadata()
    creation_identity_digest = initialization_metadata["identity_digest"]
    if physical_identity["identity_snapshot_sha256"] != creation_identity_digest:
        raise ValueError("reconstructed model creation identity differs from initialization metadata")
    if initialization_metadata["physical_fit_status"] != "unfitted_synthetic_initialization":
        raise ValueError("initialization artifact is not an unfitted physical initialization")

    original_state = _clone_state_to_cpu(model.state_dict())
    initial_non_theta_digest = _state_digest(original_state, exclude_theta=True)
    identity_before = physical_identity["identity_snapshot_sha256"]
    all_parameters = list(model.named_parameters())
    original_requires_grad = {id(parameter): parameter.requires_grad for _, parameter in all_parameters}
    original_gradients = {
        id(parameter): None if parameter.grad is None else parameter.grad.detach().clone()
        for _, parameter in all_parameters
    }
    active_parameters = [record["phase_psf"].theta for record in records]
    active_parameter_ids = {id(parameter) for parameter in active_parameters}
    if len(active_parameter_ids) != len(active_parameters):
        raise ValueError("active routes do not have unique theta parameters")
    parameter_names = {id(parameter): name for name, parameter in all_parameters}
    theta_state_keys = {record["state_key"] for record in records}
    if {parameter_names.get(id(parameter)) for parameter in active_parameters} != theta_state_keys:
        raise ValueError("active theta parameter names do not match canonical route state keys")
    original_devices = {id(parameter): parameter.device for parameter in active_parameters}
    route_metrics: dict[str, Any] = {}
    theta_digests: dict[str, Any] = {}

    try:
        for _, parameter in all_parameters:
            parameter.requires_grad_(id(parameter) in active_parameter_ids)
        for parameter in active_parameters:
            if parameter.device != device:
                parameter.data = parameter.data.to(device=device)
            parameter.requires_grad_(True)
        optimizer = torch.optim.Adam(active_parameters, lr=fit_config.learning_rate)
        optimizer_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        if optimizer_ids != active_parameter_ids:
            raise RuntimeError("optimizer parameters are not exactly the active theta parameters")

        for record in records:
            route: ActiveRoute = record["route"]
            phase_psf: PhaseOnlyPSF = record["phase_psf"]
            state_key: str = record["state_key"]
            support: int = record["support"]
            parameter = phase_psf.theta
            physical_config = record["module"].physical_config
            target_cpu = _physical_target_for_route(route)
            target = target_cpu.to(device=device, dtype=parameter.dtype)
            rho = float(physical_config.rho)
            beta = float(model.exposure_ledger.resolve_dual_rail_beta(route.route_id))
            if beta <= 0 or not math.isfinite(beta):
                raise ValueError(f"route {route.route_id!r} has invalid ledger beta")
            source_theta_sha256 = _tensor_sha256(original_state[state_key])
            if fit_config.phase_init_std:
                perturbed = _perturb_initial_theta(
                    parameter,
                    std=fit_config.phase_init_std,
                    seed=_route_seed(fit_config.seed, route.route_id),
                )
                with torch.no_grad():
                    parameter.copy_(perturbed)
            initial_metrics, _ = _physical_metrics_from_theta(
                parameter,
                phase_psf,
                target,
                support=support,
                route=route,
                rho=rho,
                beta=beta,
                physical_config=physical_config,
                fit_config=fit_config,
            )
            phase_init_sha256 = _tensor_sha256(parameter.detach())
            best_theta = parameter.detach().clone()
            best_metrics = dict(initial_metrics)
            last_metrics = dict(initial_metrics)

            for _step in range(fit_config.steps):
                optimizer.zero_grad(set_to_none=True)
                full_psf = phase_psf()
                objective, _terms = _differentiable_route_objective(
                    full_psf,
                    target,
                    support=support,
                    route=route,
                    rho=rho,
                    beta=beta,
                    physical_config=physical_config,
                    fit_config=fit_config,
                )
                if not objective.requires_grad:
                    raise ValueError(f"route {route.route_id!r} objective is not connected to theta")
                objective.backward()
                gradient = parameter.grad
                if gradient is None:
                    raise ValueError(f"route {route.route_id!r} theta gradient is missing")
                if not torch.isfinite(gradient).all():
                    raise ValueError(f"route {route.route_id!r} theta gradient is not finite")
                if fit_config.max_grad_norm is not None:
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        [parameter], fit_config.max_grad_norm
                    )
                    if not torch.isfinite(gradient_norm):
                        raise ValueError(f"route {route.route_id!r} clipped gradient norm is not finite")
                optimizer.step()
                if not torch.isfinite(parameter).all():
                    raise ValueError(f"route {route.route_id!r} theta became non-finite")
                last_metrics, _ = _physical_metrics_from_theta(
                    parameter,
                    phase_psf,
                    target,
                    support=support,
                    route=route,
                    rho=rho,
                    beta=beta,
                    physical_config=physical_config,
                    fit_config=fit_config,
                )
                if last_metrics["objective"] < best_metrics["objective"]:
                    best_theta = parameter.detach().clone()
                    best_metrics = dict(last_metrics)

            with torch.no_grad():
                parameter.copy_(best_theta)
            final_metrics, selected_full_psf = _physical_metrics_from_theta(
                parameter,
                phase_psf,
                target,
                support=support,
                route=route,
                rho=rho,
                beta=beta,
                physical_config=physical_config,
                fit_config=fit_config,
            )
            _check_final_physical_limits(final_metrics, physical_config)
            quality_violations = _check_quality_thresholds(final_metrics, fit_config)
            support_report = _support_sweep_for_psf(
                selected_full_psf,
                support,
                seed=_route_seed(fit_config.seed, route.route_id),
                physical_config=physical_config,
            ).as_dict()
            best_sha256 = _tensor_sha256(best_theta)
            final_sha256 = _tensor_sha256(parameter.detach())
            if best_sha256 != final_sha256:
                raise RuntimeError(f"route {route.route_id!r} selected theta differs from best finite objective")
            theta_digests[state_key] = {
                "dtype": str(parameter.dtype),
                "shape": list(parameter.shape),
                "sha256": final_sha256,
                "initial_sha256": source_theta_sha256,
                "phase_init_sha256": phase_init_sha256,
                "best_sha256": best_sha256,
                "final_sha256": final_sha256,
            }
            route_metrics[route.route_id] = {
                "group_id": route.group_id,
                "component_index": route.component_index,
                "sign": route.sign,
                "theta_state_key": state_key,
                "theta_dtype": str(parameter.dtype),
                "theta_shape": list(parameter.shape),
                "source_initial_theta_sha256": source_theta_sha256,
                "phase_init_theta_sha256": phase_init_sha256,
                "best_theta_sha256": best_sha256,
                "final_theta_sha256": final_sha256,
                "support": support,
                "alpha": float(route.alpha),
                "rho": rho,
                "beta": beta,
                "target_conversion": "digital_correlation_lobe_to_physical_psf_once",
                "target_sum": float(target_cpu.sum().item()),
                "initial": initial_metrics,
                "best": best_metrics,
                "final": final_metrics,
                "quality_threshold_violations": quality_violations,
                "support_sweep": support_report,
            }

    finally:
        for parameter in active_parameters:
            original_device = original_devices[id(parameter)]
            if parameter.device != original_device:
                parameter.data = parameter.data.to(device=original_device)
        for _, parameter in all_parameters:
            parameter.requires_grad_(original_requires_grad[id(parameter)])
            original_gradient = original_gradients[id(parameter)]
            parameter.grad = None if original_gradient is None else original_gradient.to(parameter.device)

    final_state = _clone_state_to_cpu(model.state_dict())
    if set(original_state) != set(final_state):
        raise RuntimeError("fitting changed the model state key set")
    for key, initial_value in original_state.items():
        if _is_theta_key(key):
            continue
        if not _state_value_equal(initial_value, final_state[key]):
            raise RuntimeError(f"fitting changed frozen non-theta state {key!r}")
    non_theta_digest = _state_digest(final_state, exclude_theta=True)
    if non_theta_digest != initial_non_theta_digest:
        raise RuntimeError("fitting changed non-theta state identity")
    identity_after = model.identity_metadata()["identity_snapshot_sha256"]
    if identity_after != identity_before or identity_after != creation_identity_digest:
        raise RuntimeError("fitting changed the model's creation identity")

    route_order = [record["route"].route_id for record in records]
    group_order = list(ENTRY_GROUPS)
    if list(route_metrics) != route_order:
        raise RuntimeError("route fit report does not preserve canonical route order")
    aggregate_metrics = _aggregate_route_metrics(route_metrics)
    ledger_data = model.exposure_ledger.as_dict()
    # Persist tuple shapes as JSON arrays, matching the existing artifact schema.
    for route in ledger_data["canonical_routes"]:
        route["lobe_shape"] = list(route["lobe_shape"])
    for route_identity in ledger_data["canonical_route_identities"]:
        route_identity["lobe_shape"] = list(route_identity["lobe_shape"])
    ledger_identity = {
        "sha256": _json_sha256({"ledger": ledger_data}),
        "policy": ledger_data["policy"],
        "active_group_count": ledger_data["active_group_count"],
        "active_route_count": ledger_data["active_route_count"],
        "exposure_count": ledger_data["exposure_count"],
        "identity": ledger_data,
    }
    dtype_names = sorted({value["dtype"] for value in theta_digests.values()})
    metadata: dict[str, Any] = {
        "model_name": "optoelectronic_h2former",
        "supervision_mode": "single_output",
        "fit_only": True,
        "initialization_only": False,
        "resume_eligible": False,
        "evidence_class": "synthetic_engineering_psf_fit",
        "source_initialization_sha256": initialization_sha256,
        "source_checkpoint_sha256": source_sha256,
        "creation_identity_digest": creation_identity_digest,
        "initialization_metadata_sha256": initialization_metadata["metadata_sha256"],
        "resolved_optoelectronic_config": _strict_json_clone(
            initialization_metadata["resolved_optoelectronic_config"]
        ),
        "fit_config": fit_config.as_dict(),
        "seed": fit_config.seed,
        "device": str(device),
        "dtype": dtype_names[0] if len(dtype_names) == 1 else dtype_names,
        "group_order": group_order,
        "route_order": route_order,
        "route_count": len(route_order),
        "theta_digests": theta_digests,
        "optimizer_parameter_names": sorted(theta_state_keys),
        "non_theta_state_sha256": non_theta_digest,
        "global_ledger_identity": ledger_identity,
        "physical_assumptions": _physical_assumptions(),
        "fit_status": "synthetic_engineering_psf_fit",
        "route_metrics": route_metrics,
        "aggregate_metrics": aggregate_metrics,
        "claims": _claims(),
        "metadata_sha256": "",
    }
    metadata["metadata_sha256"] = _metadata_digest(metadata)
    metadata = _strict_json_clone(metadata)
    report = _build_report(metadata)
    artifact_payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "artifact_version": ARTIFACT_VERSION,
        "model_state_dict": final_state,
        "optimizer_state_dict": None,
        "metadata": metadata,
    }
    _write_artifacts_atomically(
        artifact_payload,
        report,
        output_root=output_root_path,
        artifact_path=artifact_path,
        report_path=report_path,
    )
    return OptoelectronicPSFFitResult(
        artifact_path=artifact_path,
        report_path=report_path,
        report=report,
    )


def fit_optoelectronic_h2former_psf(
    initialization_artifact: str | Path,
    source_checkpoint: str | Path,
    fit_config: OptoelectronicPSFFitConfig | Mapping[str, Any],
    output_root: str | Path,
) -> OptoelectronicPSFFitResult:
    """Fit all active route phase maps and atomically publish an independent artifact."""

    config = (
        fit_config
        if isinstance(fit_config, OptoelectronicPSFFitConfig)
        else OptoelectronicPSFFitConfig.from_mapping(fit_config)
    )
    with _rng_guard():
        return _fit_optoelectronic_h2former_psf_impl(
            initialization_artifact,
            source_checkpoint,
            config,
            output_root,
        )




def _validate_fit_metadata(value: object) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError("PSF fit metadata must be an exact built-in JSON object")
    metadata = _strict_json_clone(value)
    if set(metadata) != _METADATA_KEYS:
        missing = _METADATA_KEYS - set(metadata)
        extra = set(metadata) - _METADATA_KEYS
        raise ValueError(
            "PSF fit metadata has missing or unexpected keys: "
            f"missing={sorted(missing)!r}, unexpected={sorted(extra)!r}"
        )
    if metadata["model_name"] != "optoelectronic_h2former" or metadata["supervision_mode"] != "single_output":
        raise ValueError("PSF fit artifact model identity is invalid")
    if metadata["fit_only"] is not True or metadata["initialization_only"] is not False or metadata["resume_eligible"] is not False:
        raise ValueError("PSF fit artifact training/resume contract is invalid")
    if metadata["evidence_class"] != "synthetic_engineering_psf_fit" or metadata["fit_status"] != "synthetic_engineering_psf_fit":
        raise ValueError("PSF fit artifact evidence class or status is invalid")
    for name in (
        "source_initialization_sha256",
        "source_checkpoint_sha256",
        "creation_identity_digest",
        "initialization_metadata_sha256",
        "non_theta_state_sha256",
    ):
        if not _valid_sha256(metadata[name]):
            raise ValueError(f"PSF fit metadata field {name!r} is not a SHA256 digest")
    if type(metadata["seed"]) is not int or metadata["seed"] < 0:
        raise ValueError("PSF fit seed is invalid")
    if type(metadata["device"]) is not str or not metadata["device"]:
        raise ValueError("PSF fit device is invalid")
    if type(metadata["dtype"]) is str:
        if not metadata["dtype"]:
            raise ValueError("PSF fit dtype is invalid")
    elif type(metadata["dtype"]) is list and (not metadata["dtype"] or any(type(item) is not str for item in metadata["dtype"])):
        raise ValueError("PSF fit dtype list is invalid")
    elif type(metadata["dtype"]) is not list:
        raise ValueError("PSF fit dtype is invalid")

    fit_config = OptoelectronicPSFFitConfig.from_mapping(metadata["fit_config"])
    if fit_config.as_dict() != metadata["fit_config"] or metadata["seed"] != fit_config.seed:
        raise ValueError("PSF fit metadata configuration does not match its strict schema")
    if type(metadata["group_order"]) is not list or metadata["group_order"] != list(ENTRY_GROUPS):
        raise ValueError("PSF fit group order is invalid")
    route_order = metadata["route_order"]
    if (
        type(route_order) is not list
        or not route_order
        or any(type(route_id) is not str or not route_id for route_id in route_order)
        or len(set(route_order)) != len(route_order)
    ):
        raise ValueError("PSF fit canonical route order is invalid")
    if type(metadata["route_count"]) is not int or metadata["route_count"] != len(route_order):
        raise ValueError("PSF fit route count does not match route order")
    routes = metadata["route_metrics"]
    if type(routes) is not dict or list(routes) != route_order:
        raise ValueError("PSF fit route metrics do not preserve canonical route order")
    theta_digests = metadata["theta_digests"]
    if type(theta_digests) is not dict or not theta_digests:
        raise ValueError("PSF fit theta digest mapping is invalid")
    if set(metadata["optimizer_parameter_names"]) != set(theta_digests) or len(metadata["optimizer_parameter_names"]) != len(theta_digests):
        raise ValueError("PSF fit optimizer names do not match active theta keys")
    if any(type(name) is not str or not _is_theta_key(name) for name in theta_digests):
        raise ValueError("PSF fit theta digest keys are invalid")
    expected_theta_entry_keys = {
        "dtype", "shape", "sha256", "initial_sha256", "phase_init_sha256", "best_sha256", "final_sha256"
    }
    for name, digest in theta_digests.items():
        if type(digest) is not dict or set(digest) != expected_theta_entry_keys:
            raise ValueError(f"PSF fit theta digest entry {name!r} has an invalid schema")
        if type(digest["dtype"]) is not str or type(digest["shape"]) is not list:
            raise ValueError(f"PSF fit theta digest entry {name!r} dtype or shape is invalid")
        if any(type(size) is not int or size <= 0 for size in digest["shape"]):
            raise ValueError(f"PSF fit theta digest entry {name!r} shape is invalid")
        if any(not _valid_sha256(digest[key]) for key in ("sha256", "initial_sha256", "phase_init_sha256", "best_sha256", "final_sha256")):
            raise ValueError(f"PSF fit theta digest entry {name!r} contains an invalid digest")
        if digest["sha256"] != digest["best_sha256"] or digest["sha256"] != digest["final_sha256"]:
            raise ValueError(f"PSF fit theta digest entry {name!r} does not describe the selected best theta")

    ledger = metadata["global_ledger_identity"]
    ledger_keys = {"sha256", "policy", "active_group_count", "active_route_count", "exposure_count", "identity"}
    if type(ledger) is not dict or set(ledger) != ledger_keys or not _valid_sha256(ledger["sha256"]):
        raise ValueError("PSF fit global ledger identity schema is invalid")
    if ledger["sha256"] != _json_sha256({"ledger": ledger["identity"]}):
        raise ValueError("PSF fit global ledger identity digest mismatch")
    if ledger["active_route_count"] != metadata["route_count"] or ledger["active_group_count"] != len(ENTRY_GROUPS):
        raise ValueError("PSF fit global ledger counts do not match fitted routes")
    if ledger["exposure_count"] <= 0 or type(ledger["policy"]) is not str:
        raise ValueError("PSF fit global ledger fields are invalid")
    if metadata["physical_assumptions"] != _physical_assumptions():
        raise ValueError("PSF fit physical assumptions do not match the supported synthetic model")
    if metadata["claims"] != _claims():
        raise ValueError("PSF fit claims are invalid")
    if type(metadata["aggregate_metrics"]) is not dict or type(metadata["resolved_optoelectronic_config"]) is not dict:
        raise ValueError("PSF fit aggregate metrics or resolved model config is invalid")
    if metadata["metadata_sha256"] != _metadata_digest(metadata):
        raise ValueError("PSF fit metadata integrity digest mismatch")
    return metadata


def _load_fit_payload(path: Path, *, map_location: str | torch.device) -> tuple[Mapping[str, Any], dict[str, Any]]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as error:
        raise ValueError(f"failed to read PSF fit artifact: {path}") from error
    if not isinstance(payload, Mapping) or set(payload) != _ARTIFACT_KEYS:
        raise ValueError("PSF fit artifact has missing or unexpected top-level keys")
    if type(payload["format_version"]) is not int or payload["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("PSF fit artifact format version is unsupported")
    if type(payload["artifact_type"]) is not str or payload["artifact_type"] != ARTIFACT_TYPE:
        raise ValueError("PSF fit artifact type is unsupported")
    if type(payload["artifact_version"]) is not int or payload["artifact_version"] != ARTIFACT_VERSION:
        raise ValueError("PSF fit artifact version is unsupported")
    if payload["optimizer_state_dict"] is not None:
        raise ValueError("PSF fit artifact must not contain optimizer state")
    state = payload["model_state_dict"]
    if not isinstance(state, Mapping) or any(type(key) is not str for key in state):
        raise ValueError("PSF fit model_state_dict must have exact string keys")
    metadata = _validate_fit_metadata(payload["metadata"])
    return state, metadata


def _validate_fit_state_dict(
    fit_state: Mapping[str, Any],
    initialization_state: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    theta_keys: set[str],
) -> None:
    fit_keys = set(fit_state)
    init_keys = set(initialization_state)
    if fit_keys != init_keys:
        raise ValueError(
            "PSF fit state keys do not match initialization: "
            f"missing={sorted(init_keys - fit_keys)!r}, unexpected={sorted(fit_keys - init_keys)!r}"
        )
    actual_theta_keys = {key for key in fit_keys if _is_theta_key(key)}
    if actual_theta_keys != theta_keys or set(metadata["theta_digests"]) != theta_keys:
        raise ValueError("PSF fit theta key set does not exactly match active routes")
    for key in sorted(fit_keys - theta_keys):
        if not _state_value_equal(initialization_state[key], fit_state[key]):
            raise ValueError(f"PSF fit non-theta state entry {key!r} differs from initialization")
    if _state_digest(fit_state, exclude_theta=True) != metadata["non_theta_state_sha256"]:
        raise ValueError("PSF fit non-theta state digest does not match metadata")
    for key in sorted(theta_keys):
        actual = fit_state[key]
        expected = initialization_state[key]
        digest = metadata["theta_digests"][key]
        if type(actual) is not torch.Tensor:
            raise ValueError(f"PSF fit theta {key!r} must be an exact plain Tensor")
        if actual.is_meta:
            raise ValueError(f"PSF fit theta {key!r} is a meta tensor")
        if actual.is_quantized:
            raise ValueError(f"PSF fit theta {key!r} is quantized")
        if actual.layout is not torch.strided:
            raise ValueError(f"PSF fit theta {key!r} is not strided dense storage")
        if tuple(actual.shape) != tuple(expected.shape) or tuple(actual.shape) != tuple(digest["shape"]):
            raise ValueError(f"PSF fit theta {key!r} has an invalid shape")
        if actual.dtype != expected.dtype or str(actual.dtype) != digest["dtype"]:
            raise ValueError(f"PSF fit theta {key!r} has an invalid dtype")
        if not torch.isfinite(actual).all():
            raise ValueError(f"PSF fit theta {key!r} contains NaN or infinity")
        if _tensor_sha256(actual) != digest["sha256"]:
            raise ValueError(f"PSF fit theta {key!r} digest mismatch")
        if _tensor_sha256(expected) != digest["initial_sha256"]:
            raise ValueError(f"PSF fit theta {key!r} initial digest does not match initialization")


def _read_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"PSF fit report does not exist: {path}")

    def no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"PSF fit report contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        with path.open("r", encoding="utf-8") as handle:
            report = json.load(handle, object_pairs_hook=no_duplicate_keys, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid JSON constant {value}")))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"failed to read strict PSF fit report: {path}") from error
    if type(report) is not dict or set(report) != _REPORT_KEYS:
        raise ValueError("PSF fit report has missing or unexpected keys")
    return _strict_json_clone(report)


def _json_values_equal(actual: Any, expected: Any, *, name: str) -> None:
    if _canonical_json({"value": actual}) != _canonical_json({"value": expected}):
        raise ValueError(f"PSF fit {name} does not match recomputed artifact values")


def _recompute_route_metrics(
    model: OptoelectronicH2Former,
    initialization_state: Mapping[str, Any],
    fit_state: Mapping[str, Any],
    fit_config: OptoelectronicPSFFitConfig,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    records = _active_route_records(model)
    route_order = [record["route"].route_id for record in records]
    if route_order != metadata["route_order"]:
        raise ValueError("PSF fit route order differs from reconstructed canonical ledger")
    recomputed: dict[str, Any] = {}
    for record in records:
        route: ActiveRoute = record["route"]
        phase_psf: PhaseOnlyPSF = record["phase_psf"]
        state_key: str = record["state_key"]
        support: int = record["support"]
        physical_config = record["module"].physical_config
        target = _physical_target_for_route(route)
        rho = float(physical_config.rho)
        beta = float(model.exposure_ledger.resolve_dual_rail_beta(route.route_id))
        init_theta = initialization_state[state_key]
        seeded_theta = _perturb_initial_theta(
            init_theta,
            std=fit_config.phase_init_std,
            seed=_route_seed(fit_config.seed, route.route_id),
        )
        initial_metrics, _ = _physical_metrics_from_theta(
            seeded_theta,
            phase_psf,
            target,
            support=support,
            route=route,
            rho=rho,
            beta=beta,
            physical_config=physical_config,
            fit_config=fit_config,
        )
        fitted_theta = fit_state[state_key]
        final_metrics, full_psf = _physical_metrics_from_theta(
            fitted_theta,
            phase_psf,
            target,
            support=support,
            route=route,
            rho=rho,
            beta=beta,
            physical_config=physical_config,
            fit_config=fit_config,
        )
        _check_final_physical_limits(final_metrics, physical_config)
        quality_violations = _check_quality_thresholds(final_metrics, fit_config)
        sweep = _support_sweep_for_psf(
            full_psf,
            support,
            seed=_route_seed(fit_config.seed, route.route_id),
            physical_config=physical_config,
        ).as_dict()
        source_digest = _tensor_sha256(init_theta)
        phase_init_digest = _tensor_sha256(seeded_theta)
        selected_digest = _tensor_sha256(fitted_theta)
        stored_digest = metadata["theta_digests"][state_key]
        if phase_init_digest != stored_digest["phase_init_sha256"]:
            raise ValueError(f"PSF fit route {route.route_id!r} phase initialization digest mismatch")
        if source_digest != stored_digest["initial_sha256"]:
            raise ValueError(f"PSF fit route {route.route_id!r} initialization theta digest mismatch")
        if selected_digest != stored_digest["best_sha256"] or selected_digest != stored_digest["final_sha256"]:
            raise ValueError(f"PSF fit route {route.route_id!r} best theta digest mismatch")
        recomputed[route.route_id] = {
            "group_id": route.group_id,
            "component_index": route.component_index,
            "sign": route.sign,
            "theta_state_key": state_key,
            "theta_dtype": str(fitted_theta.dtype),
            "theta_shape": list(fitted_theta.shape),
            "source_initial_theta_sha256": source_digest,
            "phase_init_theta_sha256": phase_init_digest,
            "best_theta_sha256": selected_digest,
            "final_theta_sha256": selected_digest,
            "support": support,
            "alpha": float(route.alpha),
            "rho": rho,
            "beta": beta,
            "target_conversion": "digital_correlation_lobe_to_physical_psf_once",
            "target_sum": float(target.sum().item()),
            "initial": initial_metrics,
            "best": final_metrics,
            "final": final_metrics,
            "quality_threshold_violations": quality_violations,
            "support_sweep": sweep,
        }
    return recomputed


def _load_optoelectronic_psf_fit_impl(
    fit_artifact: str | Path,
    initialization_artifact: str | Path,
    source_checkpoint: str | Path,
    *,
    map_location: str | torch.device,
    target_model: OptoelectronicH2Former | None,
    report_path: str | Path | None,
) -> OptoelectronicH2Former:
    fit_path = Path(fit_artifact).expanduser().resolve()
    init_path = Path(initialization_artifact).expanduser().resolve()
    source_path = Path(source_checkpoint).expanduser().resolve()
    map_device = torch.device(map_location)
    if map_device.type == "meta":
        raise ValueError("meta map_location is not supported")
    fit_state, metadata = _load_fit_payload(fit_path, map_location=map_location)

    initialization_sha256 = _sha256_file(init_path) if init_path.is_file() else None
    if initialization_sha256 != metadata["source_initialization_sha256"]:
        raise ValueError(
            "initialization artifact SHA256 does not match the PSF fit artifact: "
            f"expected {metadata['source_initialization_sha256']}, got {initialization_sha256}"
        )
    if not source_path.is_file():
        raise FileNotFoundError(f"source checkpoint does not exist: {source_path}")
    model = load_optoelectronic_initialization(
        init_path,
        source_path,
        map_location=str(map_location),
    )
    source_sha256 = _sha256_file(source_path)
    if source_sha256 != metadata["source_checkpoint_sha256"]:
        raise ValueError("source checkpoint SHA256 does not match the PSF fit artifact")
    initialization_metadata = _read_initialization_metadata(init_path, map_location="cpu")
    if initialization_metadata["source_checkpoint"]["sha256"] != source_sha256:
        raise ValueError("source checkpoint SHA256 does not match the initialization artifact")
    if initialization_metadata["identity_digest"] != metadata["creation_identity_digest"]:
        raise ValueError("creation identity digest does not match the initialization artifact")
    if initialization_metadata["metadata_sha256"] != metadata["initialization_metadata_sha256"]:
        raise ValueError("initialization metadata digest does not match the PSF fit artifact")
    if initialization_metadata["resolved_optoelectronic_config"] != metadata["resolved_optoelectronic_config"]:
        raise ValueError("resolved optoelectronic config does not match initialization artifact")
    if model.identity_metadata()["identity_snapshot_sha256"] != metadata["creation_identity_digest"]:
        raise ValueError("reconstructed model creation identity does not match PSF fit metadata")
    if model.opto_config.mode != "physical":
        raise ValueError("PSF fit loader accepts only physical-mode initialization artifacts")
    if initialization_metadata["physical_fit_status"] != "unfitted_synthetic_initialization":
        raise ValueError("source initialization artifact is not in the unfitted physical state")
    initialization_state = _clone_state_to_cpu(model.state_dict())
    records = _active_route_records(model)
    theta_keys = {record["state_key"] for record in records}
    _validate_fit_state_dict(fit_state, initialization_state, metadata, theta_keys=theta_keys)

    ledger_data = model.exposure_ledger.as_dict()
    expected_ledger = {
        "sha256": _json_sha256({"ledger": ledger_data}),
        "policy": ledger_data["policy"],
        "active_group_count": ledger_data["active_group_count"],
        "active_route_count": ledger_data["active_route_count"],
        "exposure_count": ledger_data["exposure_count"],
        "identity": ledger_data,
    }
    _json_values_equal(metadata["global_ledger_identity"], expected_ledger, name="global ledger identity")
    if metadata["route_count"] != len(records) or set(metadata["route_order"]) != {record["route"].route_id for record in records}:
        raise ValueError("PSF fit route count or IDs do not match reconstructed physical routes")
    recomputed_routes = _recompute_route_metrics(
        model,
        initialization_state,
        fit_state,
        OptoelectronicPSFFitConfig.from_mapping(metadata["fit_config"]),
        metadata,
    )
    _json_values_equal(metadata["route_metrics"], recomputed_routes, name="route metrics")
    recomputed_aggregate = _aggregate_route_metrics(recomputed_routes)
    _json_values_equal(metadata["aggregate_metrics"], recomputed_aggregate, name="aggregate metrics")

    resolved_report_path = (
        Path(report_path).expanduser().resolve()
        if report_path is not None
        else fit_path.parent / FIT_REPORT_FILENAME
    )
    report = _read_report(resolved_report_path)
    _json_values_equal(report, _build_report(metadata), name="JSON report")

    destination = target_model if target_model is not None else model
    if not isinstance(destination, OptoelectronicH2Former):
        raise ValueError("target_model must be an OptoelectronicH2Former")
    if destination is not model:
        if destination.identity_metadata()["identity_snapshot_sha256"] != model.identity_metadata()["identity_snapshot_sha256"]:
            raise ValueError("target model creation identity differs from initialization model")
        destination_state = _clone_state_to_cpu(destination.state_dict())
        if set(destination_state) != set(initialization_state) or any(
            not _state_value_equal(initialization_state[key], destination_state[key])
            for key in initialization_state
        ):
            raise ValueError("target model is not the unchanged initialization state")
    try:
        destination.load_state_dict(fit_state, strict=True, assign=False)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"PSF fit state failed strict loading: {error}") from error
    return destination


def load_optoelectronic_psf_fit(
    fit_artifact: str | Path,
    initialization_artifact: str | Path,
    source_checkpoint: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    target_model: OptoelectronicH2Former | None = None,
    report_path: str | Path | None = None,
) -> OptoelectronicH2Former:
    """Strictly validate lineage, state, theta digests, and recomputed PSF metrics."""

    with _rng_guard():
        return _load_optoelectronic_psf_fit_impl(
            fit_artifact,
            initialization_artifact,
            source_checkpoint,
            map_location=map_location,
            target_model=target_model,
            report_path=report_path,
        )


__all__ = [
    "ARTIFACT_TYPE",
    "ARTIFACT_VERSION",
    "FIT_FILENAME",
    "FIT_REPORT_FILENAME",
    "REPORT_SCHEMA",
    "REPORT_VERSION",
    "OptoelectronicPSFFitConfig",
    "OptoelectronicPSFFitResult",
    "fit_optoelectronic_h2former_psf",
    "load_optoelectronic_psf_fit",
]
