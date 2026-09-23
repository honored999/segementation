"""Portable initialization artifacts for the isolated optoelectronic H2Former.

This module deliberately owns only initialization-artifact creation and loading.
It does not modify the H2Former implementation, connect the optical model to
training or prediction, or read any medical data.  The source checkpoint and
the optical initialization artifact remain separate lineage objects: the
artifact stores the source content digest and identity, but not a second copy
of the source model.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from standalone_nnunet2d.models.factory import resolve_checkpoint_model_identity
from standalone_nnunet2d.models.h2former import H2Former
from standalone_nnunet2d.models.optoelectronic_frontend import (
    OptoelectronicConfig,
    PhasePSFConfig,
)
from standalone_nnunet2d.models.optoelectronic_h2former import (
    ENTRY_GROUPS,
    OptoelectronicH2Former,
    OptoelectronicH2FormerConfig,
)


ARTIFACT_TYPE = "optoelectronic_h2former_initialization"
ARTIFACT_VERSION = 1
REPORT_SCHEMA = "optoelectronic_h2former_initialization_report"
REPORT_VERSION = 1
INITIALIZATION_FILENAME = "optoelectronic_initialization.pth"
REPORT_FILENAME = "optoelectronic_initialization_report.json"
CHECKPOINT_FORMAT_VERSION = 1

_SOURCE_IDENTITY = {"model_name": "h2former", "supervision_mode": "single_output"}
_ENTRY_RANK_UPPER_BOUNDS = {
    "stem7": 49,
    "patch2": 4,
    "patch4": 15,
    "patch8": 7,
    "patch16": 7,
}
_EXPOSURE_POLICIES = {
    "shared_input_global_parallel_equal_split",
    "global_parallel",
    "group_sequential",
    "route_sequential",
    "bounded_parallel",
}
_MODES = {"ideal", "physical"}

_ROOT_CONFIG_KEYS = {
    "mode",
    "rank_by_group",
    "variance_threshold",
    "trainable_phase",
    "trainable_mixing",
    "trainable_bias",
    "trainable_electronic_backend",
    "physical_config",
    "exposure_policy",
    "route_batch_capacity",
    # Present only in the JSON representation emitted by Config.as_dict().
    "model_class",
}
_PHYSICAL_CONFIG_KEYS = {
    "phase",
    "dual_rail_mode",
    "gain_update_mode",
    "rho",
    "min_throughput",
    "max_electronic_gain",
    "min_split_fraction",
    "max_support_response_error",
    "support_error_mode",
    # Derived field emitted by OptoelectronicConfig.as_dict().
    "assumption_status",
}
_PHASE_CONFIG_KEYS = {
    "wavelength_m",
    "phase_grid_size",
    "phase_pitch_m",
    "focal_length_m",
    "aperture_diameter_m",
    "magnification",
    # Derived fields emitted by PhasePSFConfig.as_dict().
    "detector_pitch_m",
    "assumption_status",
}

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
    "initialization_only",
    "resume_eligible",
    "initialization_semantics",
    "source_model_identity",
    "source_checkpoint",
    "resolved_optoelectronic_config",
    "pca_rank_selection",
    "creation_identity",
    "identity_metadata",
    "identity_digest",
    "resolved_ranks",
    "pca_reports",
    "global_exposure",
    "trainability_contract",
    "physical_fit_status",
    "evidence_class",
    "global_physics",
    "claims",
    "metadata_sha256",
}
_SOURCE_CHECKPOINT_KEYS = {
    "sha256",
    "model_name",
    "supervision_mode",
    "path",
    "path_identity_role",
}


@dataclass(frozen=True)
class InitializationResult:
    """Paths and report returned after both final files are published."""

    artifact_path: Path
    report_path: Path
    source_checkpoint_sha256: str
    report: dict[str, Any]


@dataclass(frozen=True)
class _ConfigResolution:
    config: OptoelectronicH2FormerConfig
    rank_selection: dict[str, Any]


@dataclass(frozen=True)
class _LoadedSource:
    path: Path
    sha256: str
    metadata: dict[str, Any]
    model: H2Former


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON-safe metadata keys must be strings")
            result[key] = _json_safe(item)
        return result
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON-safe metadata cannot contain NaN or infinity")
        return value
    if isinstance(value, Path):
        return str(value)
    raise ValueError(f"value is not JSON-safe: {type(value).__name__}")


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            _json_safe(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(f"value is not JSON-safe: {error}") from error


def _sha256_bytes(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings")
    return value


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"{name} contains unknown key(s): {unknown!r}")


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _require_finite(value: object, name: str) -> int | float:
    if not _is_finite_number(value):
        raise ValueError(f"{name} must be a finite number")
    return value  # type: ignore[return-value]


def _derived_float_matches(actual: object, expected: float, name: str) -> None:
    if not _is_finite_number(actual) or not math.isclose(
        float(actual), expected, rel_tol=0.0, abs_tol=max(1e-18, abs(expected) * 1e-12)
    ):
        raise ValueError(f"{name} does not match its resolved value")


def _build_phase_config(value: object) -> PhasePSFConfig:
    phase_data = _require_mapping(value, "physical_config.phase")
    _reject_unknown(phase_data, _PHASE_CONFIG_KEYS, "physical_config.phase")
    kwargs: dict[str, Any] = {}
    for name in (
        "wavelength_m",
        "phase_pitch_m",
        "focal_length_m",
        "aperture_diameter_m",
        "magnification",
    ):
        if name in phase_data:
            kwargs[name] = _require_finite(phase_data[name], f"physical_config.phase.{name}")
    if "phase_grid_size" in phase_data:
        value = phase_data["phase_grid_size"]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("physical_config.phase.phase_grid_size must be an integer")
        kwargs["phase_grid_size"] = value
    phase = PhasePSFConfig(**kwargs)
    if "detector_pitch_m" in phase_data:
        _derived_float_matches(
            phase_data["detector_pitch_m"], phase.detector_pitch_m, "detector_pitch_m"
        )
    if "assumption_status" in phase_data and phase_data["assumption_status"] != phase.assumption_status:
        raise ValueError("physical_config.phase.assumption_status does not match")
    return phase


def _build_physical_config(value: object) -> OptoelectronicConfig:
    physical_data = _require_mapping(value, "physical_config")
    _reject_unknown(physical_data, _PHYSICAL_CONFIG_KEYS, "physical_config")
    if "phase" not in physical_data:
        raise ValueError("physical_config must contain the nested phase mapping")
    phase = _build_phase_config(physical_data["phase"])
    kwargs: dict[str, Any] = {"phase": phase}
    for name in (
        "dual_rail_mode",
        "gain_update_mode",
        "support_error_mode",
    ):
        if name in physical_data:
            kwargs[name] = _require_string(physical_data[name], f"physical_config.{name}")
    for name in (
        "rho",
        "min_throughput",
        "max_electronic_gain",
        "min_split_fraction",
        "max_support_response_error",
    ):
        if name in physical_data:
            kwargs[name] = _require_finite(physical_data[name], f"physical_config.{name}")
    config = OptoelectronicConfig(**kwargs)
    if "assumption_status" in physical_data and physical_data["assumption_status"] != config.phase.assumption_status:
        raise ValueError("physical_config.assumption_status does not match")
    return config


def _normalise_mode(value: object) -> str:
    mode = _require_string(value, "mode").lower().replace("-", "_")
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {sorted(_MODES)!r}")
    return mode


def _normalise_exposure_policy(value: object) -> str:
    policy = _require_string(value, "exposure_policy").lower().replace("-", "_")
    if policy not in _EXPOSURE_POLICIES:
        raise ValueError(f"unsupported exposure_policy {value!r}")
    return policy


def _build_config_resolution(config_value: object) -> _ConfigResolution:
    data = _require_mapping(config_value, "optoelectronic config")
    _reject_unknown(data, _ROOT_CONFIG_KEYS, "optoelectronic config")
    if "model_class" in data and data["model_class"] != "OptoelectronicH2Former":
        raise ValueError("model_class must be OptoelectronicH2Former")
    if "physical_config" not in data:
        raise ValueError("optoelectronic config must contain physical_config")

    mode = _normalise_mode(data.get("mode", "ideal"))
    physical_config = _build_physical_config(data["physical_config"])
    rank_value = data.get("rank_by_group")
    threshold_present = "variance_threshold" in data
    threshold_value = data.get("variance_threshold")
    if rank_value is not None:
        ranks = _require_mapping(rank_value, "rank_by_group")
        if set(ranks) != set(ENTRY_GROUPS):
            raise ValueError(
                "rank_by_group must contain exactly the five entrance groups "
                f"{ENTRY_GROUPS!r}"
            )
        normalized_ranks: dict[str, int] = {}
        for group_id in ENTRY_GROUPS:
            rank = ranks[group_id]
            if isinstance(rank, bool) or not isinstance(rank, int):
                raise ValueError(f"rank for group {group_id!r} must be an integer")
            upper_bound = _ENTRY_RANK_UPPER_BOUNDS[group_id]
            if rank < 0 or rank > upper_bound:
                raise ValueError(
                    f"rank for group {group_id!r} must satisfy 0 <= rank <= {upper_bound}, got {rank}"
                )
            normalized_ranks[group_id] = rank
        if threshold_present and threshold_value is not None:
            raise ValueError("rank_by_group and variance_threshold are mutually exclusive")
        rank_by_group: dict[str, int] | None = normalized_ranks
        resolved_threshold: float | None = None
        rank_selection = {
            "rule": "explicit_rank_by_group",
            "source": "explicit",
            "variance_threshold": None,
            "performance_guarantee": False,
        }
    else:
        rank_by_group = None
        if threshold_value is None:
            resolved_threshold = 0.99
            threshold_source = "project_default"
        else:
            resolved_threshold = float(_require_finite(threshold_value, "variance_threshold"))
            if not 0.0 <= resolved_threshold <= 1.0:
                raise ValueError("variance_threshold must be in [0, 1]")
            threshold_source = "explicit"
        rank_selection = {
            "rule": "centered_pca_variance_threshold",
            "source": threshold_source,
            "variance_threshold": resolved_threshold,
            "performance_guarantee": False,
        }

    trainable_defaults = {
        "trainable_phase": True,
        "trainable_mixing": False,
        "trainable_bias": False,
        "trainable_electronic_backend": True,
    }
    trainable_values: dict[str, bool] = {}
    for name, default in trainable_defaults.items():
        value = data.get(name, default)
        trainable_values[name] = _require_bool(value, name)

    exposure_policy = _normalise_exposure_policy(
        data.get("exposure_policy", "shared_input_global_parallel_equal_split")
    )
    route_batch_capacity = data.get("route_batch_capacity")
    if route_batch_capacity is not None:
        if isinstance(route_batch_capacity, bool) or not isinstance(route_batch_capacity, int) or route_batch_capacity <= 0:
            raise ValueError("route_batch_capacity must be a positive integer or null")
    config = OptoelectronicH2FormerConfig(
        mode=mode,
        rank_by_group=rank_by_group,
        variance_threshold=resolved_threshold,
        physical_config=physical_config,
        exposure_policy=exposure_policy,
        route_batch_capacity=route_batch_capacity,
        **trainable_values,
    )
    return _ConfigResolution(config=config, rank_selection=rank_selection)


def build_optoelectronic_config(config: Mapping[str, Any]) -> OptoelectronicH2FormerConfig:
    """Strictly resolve a JSON-style mapping into the model config dataclass."""

    return _build_config_resolution(config).config


def _config_resolution(config: object) -> _ConfigResolution:
    if isinstance(config, OptoelectronicH2FormerConfig):
        resolved = config
        if resolved.rank_by_group is None:
            source = "explicit"
            rule = "centered_pca_variance_threshold"
        else:
            source = "explicit"
            rule = "explicit_rank_by_group"
        return _ConfigResolution(
            config=resolved,
            rank_selection={
                "rule": rule,
                "source": source,
                "variance_threshold": resolved.resolved_variance_threshold,
                "performance_guarantee": False,
            },
        )
    return _build_config_resolution(config)


def _validate_source_payload(payload: object) -> tuple[Mapping[str, Any], dict[str, Any]]:
    if not isinstance(payload, Mapping):
        raise ValueError("source checkpoint payload must be a mapping")
    if set(payload) != {"format_version", "model_state_dict", "optimizer_state_dict", "metadata"}:
        raise ValueError("source checkpoint payload has missing or unexpected keys")
    if type(payload["format_version"]) is not int or payload["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported source checkpoint format_version")
    state = payload["model_state_dict"]
    if not isinstance(state, Mapping):
        raise ValueError("source checkpoint model_state_dict must be a mapping")
    optimizer_state = payload["optimizer_state_dict"]
    if optimizer_state is not None and not isinstance(optimizer_state, Mapping):
        raise ValueError("source checkpoint optimizer_state_dict must be null or a mapping")
    metadata_value = payload["metadata"]
    if not isinstance(metadata_value, Mapping):
        raise ValueError("source checkpoint metadata must be a mapping")
    metadata = dict(metadata_value)
    if type(metadata.get("model_name")) is not str or type(metadata.get("supervision_mode")) is not str:
        raise ValueError("source checkpoint metadata must contain model_name and supervision_mode")
    try:
        identity = resolve_checkpoint_model_identity(metadata)
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError("source checkpoint model identity is malformed or conflicting") from error
    if identity != (_SOURCE_IDENTITY["model_name"], _SOURCE_IDENTITY["supervision_mode"]):
        raise ValueError(
            "source checkpoint must identify model_name=h2former and supervision_mode=single_output"
        )
    if metadata["model_name"] != _SOURCE_IDENTITY["model_name"] or metadata["supervision_mode"] != _SOURCE_IDENTITY["supervision_mode"]:
        raise ValueError("source checkpoint has an invalid H2Former identity")
    return state, metadata


def _load_source_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> _LoadedSource:
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"source checkpoint does not exist: {source_path}")
    source_sha256 = _sha256_file(source_path)
    if expected_sha256 is not None and source_sha256 != expected_sha256:
        raise ValueError(
            "source checkpoint SHA256 does not match the initialization artifact: "
            f"expected {expected_sha256}, got {source_sha256}"
        )
    try:
        payload = torch.load(source_path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError(f"failed to read source checkpoint: {source_path}") from error
    state, metadata = _validate_source_payload(payload)

    # Model construction is intentionally after payload format and identity
    # validation.  No optimizer or scheduler state is read or restored.
    source_model = H2Former(in_channels=1, num_classes=2, image_size=512)
    try:
        source_model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError("source H2Former state_dict failed strict loading") from error
    source_model.eval()
    return _LoadedSource(
        path=source_path,
        sha256=source_sha256,
        metadata=metadata,
        model=source_model,
    )


def _entry_report(model: OptoelectronicH2Former) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for module in model._entry_modules():
        target = module.pca_target_weight
        pca = module.pca_report.as_dict()
        shape = [int(size) for size in target.shape]
        sample_shape = [shape[0], int(torch.tensor(shape[1:]).prod().item())]
        reports.append(
            {
                "group_id": module.group_id,
                "original_conv_shape": shape,
                "stride": [int(value) for value in module.stride],
                "padding": [int(value) for value in module.padding],
                "bias": module.original_bias is not None,
                "pca_sample_matrix_shape": sample_shape,
                "legal_maximum_rank": int(module.structural_rank_upper_bound),
                "resolved_rank": int(module.resolved_rank),
                "explained_variance_ratio": float(pca["explained_variance_ratio"]),
                "frobenius_reconstruction_error": float(pca["absolute_frobenius_error"]),
                "relative_reconstruction_error": float(pca["relative_frobenius_error"]),
                "maximum_absolute_reconstruction_error": float(pca["maximum_absolute_error"]),
                # Preserve the names used by the reusable PCA operator as
                # aliases in the standalone initialization report.
                "absolute_frobenius_error": float(pca["absolute_frobenius_error"]),
                "relative_frobenius_error": float(pca["relative_frobenius_error"]),
                "maximum_absolute_error": float(pca["maximum_absolute_error"]),
                "centered_mean_contribution": {
                    "retained": True,
                    "component_bank_index": 0,
                    "relation": "W = 1 * centered_mean + centered_components",
                },
                "original_bias_retention": {
                    "present": module.original_bias is not None,
                    "retained": True,
                    "storage": "original_bias buffer and bias initialization; added once by the entry operator",
                },
            }
        )
    if tuple(report["group_id"] for report in reports) != tuple(ENTRY_GROUPS):
        raise RuntimeError("unexpected optoelectronic entry group order")
    return reports


def _global_physics_report(
    model: OptoelectronicH2Former,
    config: OptoelectronicH2FormerConfig,
) -> dict[str, Any]:
    ledger = model.exposure_ledger
    exposures = ledger.as_dict()["exposures"]
    return {
        "total_entry_kernel_count": 128,
        "active_route_count": int(ledger.active_route_count),
        "exposure_count": int(ledger.exposure_count),
        "exposure_policy": ledger.policy,
        "global_beam_split_budget": {
            "scope": "all five entry groups and all active signed PCA component routes",
            "allocation_is_global_across_groups": True,
            "dual_rail_policy": config.physical_config.dual_rail_mode,
            "exposure_count": int(ledger.exposure_count),
            "sum_beta_by_exposure": [float(exposure["sum_beta"]) for exposure in exposures],
            "unused_budget_by_exposure": [float(exposure["unused_budget"]) for exposure in exposures],
            "maximum_sum_beta": max(float(exposure["sum_beta"]) for exposure in exposures),
        },
        "detector_gain_convention": {
            "rho": float(config.physical_config.rho),
            "gain_update_mode": config.physical_config.gain_update_mode,
            "response_equation": "D = rho * beta * convolution(input_rail, h_support)",
            "gain_is_fitted_here": False,
            "convention_class": "synthetic project assumption",
        },
        "support_truncation": {
            "policy": "center_crop_full_psf_to_entry_kernel_support_during_physical_forward",
            "physical_truncation_claim": False,
            "outside_support_energy_is_not_assumed_to_vanish": True,
            "support_error_evaluation": "not performed by initialization artifact creation",
            "support_error_mode": config.physical_config.normalized_support_error_mode,
            "max_support_response_error": float(config.physical_config.max_support_response_error),
        },
        "physical_assumptions": config.physical_config.as_dict(),
        "assumption_statement": (
            "wavelength, sampling, aperture, propagation and detector constants are "
            "synthetic project assumptions, not paper-original hardware validation"
        ),
    }


def _claims(config: OptoelectronicH2FormerConfig) -> dict[str, Any]:
    return {
        "is_medical_result": False,
        "is_speed_result": False,
        "is_hardware_manufacturability_result": False,
        "physical_psf_fit_completed": False,
        "physical_mode_status": (
            "unfitted_synthetic_initialization"
            if config.mode == "physical"
            else "not_applicable"
        ),
        "not_paper_original_h2former_claim": True,
        "not_paper_original_hardware_claim": True,
        "not_training_resume_checkpoint": True,
    }


def _source_metadata(
    source: _LoadedSource,
) -> dict[str, Any]:
    return {
        "sha256": source.sha256,
        "model_name": source.metadata["model_name"],
        "supervision_mode": source.metadata["supervision_mode"],
        "path": str(source.path),
        "path_identity_role": "audit_only_not_portable_identity",
    }


def _metadata_integrity_view(metadata: Mapping[str, Any]) -> dict[str, Any]:
    view = copy.deepcopy(dict(metadata))
    source = view.get("source_checkpoint")
    if isinstance(source, dict) and "path" in source:
        source["path"] = "<source-checkpoint-path-excluded-from-integrity-digest>"
    view.pop("metadata_sha256", None)
    return view


def _build_artifact_metadata(
    source: _LoadedSource,
    config_resolution: _ConfigResolution,
    model: OptoelectronicH2Former,
    *,
    rank_selection: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = config_resolution.config
    identity_metadata = _json_safe(model.identity_metadata())
    group_reports = _entry_report(model)
    global_physics = _global_physics_report(model, config)
    claims = _claims(config)
    selected_rank_rule = _json_safe(
        config_resolution.rank_selection if rank_selection is None else dict(rank_selection)
    )
    metadata: dict[str, Any] = {
        "model_name": "optoelectronic_h2former",
        "supervision_mode": "single_output",
        "initialization_only": True,
        "resume_eligible": False,
        "initialization_semantics": "new_architecture_initialized_from_h2former_weights_not_resume",
        "source_model_identity": copy.deepcopy(_SOURCE_IDENTITY),
        "source_checkpoint": _source_metadata(source),
        "resolved_optoelectronic_config": _json_safe(config.as_dict()),
        "pca_rank_selection": selected_rank_rule,
        "creation_identity": copy.deepcopy(identity_metadata["identity_snapshot"]),
        "identity_metadata": identity_metadata,
        "identity_digest": identity_metadata["identity_snapshot_sha256"],
        "resolved_ranks": copy.deepcopy(identity_metadata["resolved_ranks"]),
        "pca_reports": {report["group_id"]: copy.deepcopy(report) for report in group_reports},
        "global_exposure": copy.deepcopy(identity_metadata["global_exposure"]),
        "trainability_contract": copy.deepcopy(identity_metadata["trainable"]),
        "physical_fit_status": identity_metadata["physical_fit_status"],
        "evidence_class": "synthetic_engineering_initialization",
        "global_physics": global_physics,
        "claims": claims,
        "metadata_sha256": "",
    }
    metadata["metadata_sha256"] = _sha256_bytes(_metadata_integrity_view(metadata))
    metadata = _json_safe(metadata)
    report = {
        "report_schema": REPORT_SCHEMA,
        "report_version": REPORT_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "artifact_version": ARTIFACT_VERSION,
        "evidence_class": "synthetic_engineering_initialization",
        "source_checkpoint": copy.deepcopy(metadata["source_checkpoint"]),
        "resolved_config": copy.deepcopy(metadata["resolved_optoelectronic_config"]),
        "pca_rank_selection": copy.deepcopy(metadata["pca_rank_selection"]),
        "groups": group_reports,
        "global": global_physics,
        "physical_fit_status": metadata["physical_fit_status"],
        "claims": copy.deepcopy(claims),
        "creation_identity_digest": metadata["identity_digest"],
        "artifact_metadata_sha256": metadata["metadata_sha256"],
    }
    return metadata, _json_safe(report)


def _check_output_paths(output_root: str | Path) -> tuple[Path, Path, Path]:
    root = Path(output_root).expanduser().resolve()
    if root.exists() and not root.is_dir():
        raise ValueError(f"output-root is not a directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    artifact_path = root / INITIALIZATION_FILENAME
    report_path = root / REPORT_FILENAME
    existing = [str(path) for path in (artifact_path, report_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing output(s): {existing!r}")
    return root, artifact_path, report_path


def _publish_without_overwrite(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite existing output: {destination}") from error

    try:
        source.unlink()
    except Exception:
        try:
            destination.unlink()
        except FileNotFoundError:
            pass
        raise


def _write_artifacts_atomically(
    artifact_payload: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    output_root: Path,
    artifact_path: Path,
    report_path: Path,
) -> None:
    temporary_paths: list[Path] = []
    published_paths: list[Path] = []
    try:
        artifact_fd, artifact_tmp_name = tempfile.mkstemp(
            prefix=".optoelectronic-initialization-", suffix=".tmp", dir=output_root
        )
        try:
            os.close(artifact_fd)
        finally:
            artifact_tmp = Path(artifact_tmp_name)
            temporary_paths.append(artifact_tmp)
        report_fd, report_tmp_name = tempfile.mkstemp(
            prefix=".optoelectronic-report-", suffix=".tmp", dir=output_root
        )
        try:
            os.close(report_fd)
        finally:
            report_tmp = Path(report_tmp_name)
            temporary_paths.append(report_tmp)
        torch.save(dict(artifact_payload), artifact_tmp)
        with report_tmp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(_json_safe(report), handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _publish_without_overwrite(artifact_tmp, artifact_path)
        published_paths.append(artifact_path)
        _publish_without_overwrite(report_tmp, report_path)
        published_paths.append(report_path)
    except Exception:
        for path in published_paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for path in temporary_paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def create_optoelectronic_initialization(
    source_checkpoint_path: str | Path,
    config: Mapping[str, Any] | OptoelectronicH2FormerConfig,
    output_root: str | Path,
) -> InitializationResult:
    """Create a new initialization artifact and JSON report without overwriting."""

    output_root_path, artifact_path, report_path = _check_output_paths(output_root)
    config_resolution = _config_resolution(config)
    source = _load_source_checkpoint(source_checkpoint_path)
    optical_model = OptoelectronicH2Former.from_h2former(
        source.model,
        config_resolution.config,
    )
    metadata, report = _build_artifact_metadata(source, config_resolution, optical_model)
    artifact_payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "artifact_version": ARTIFACT_VERSION,
        "model_state_dict": optical_model.state_dict(),
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
    return InitializationResult(
        artifact_path=artifact_path,
        report_path=report_path,
        source_checkpoint_sha256=source.sha256,
        report=report,
    )


def _validate_artifact_metadata_shape(metadata: object) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise ValueError("initialization artifact metadata must be a mapping")
    normalized = _json_safe(metadata)
    if set(normalized) != _METADATA_KEYS:
        raise ValueError(
            "initialization artifact metadata has missing or unexpected keys: "
            f"missing={sorted(_METADATA_KEYS - set(normalized))!r}, "
            f"unexpected={sorted(set(normalized) - _METADATA_KEYS)!r}"
        )
    if normalized["model_name"] != "optoelectronic_h2former" or normalized["supervision_mode"] != "single_output":
        raise ValueError("initialization artifact has an invalid model identity")
    if normalized["initialization_only"] is not True or normalized["resume_eligible"] is not False:
        raise ValueError("initialization artifact resume contract is invalid")
    if normalized["evidence_class"] != "synthetic_engineering_initialization":
        raise ValueError("initialization artifact evidence_class is invalid")
    source = _require_mapping(normalized["source_checkpoint"], "artifact source_checkpoint")
    if set(source) != _SOURCE_CHECKPOINT_KEYS:
        raise ValueError("artifact source_checkpoint has missing or unexpected keys")
    if (
        not isinstance(source["sha256"], str)
        or len(source["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in source["sha256"])
    ):
        raise ValueError("artifact source_checkpoint sha256 is invalid")
    if source["model_name"] != "h2former" or source["supervision_mode"] != "single_output":
        raise ValueError("artifact source model identity is invalid")
    if not isinstance(source["path"], str) or not source["path"]:
        raise ValueError("artifact source checkpoint path must be an audit string")
    if source["path_identity_role"] != "audit_only_not_portable_identity":
        raise ValueError("artifact source checkpoint path identity role is invalid")
    digest = normalized["identity_digest"]
    if not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("artifact identity_digest is invalid")
    metadata_digest = normalized["metadata_sha256"]
    if not isinstance(metadata_digest, str) or len(metadata_digest) != 64 or any(character not in "0123456789abcdef" for character in metadata_digest):
        raise ValueError("artifact metadata_sha256 is invalid")
    if _sha256_bytes(_metadata_integrity_view(normalized)) != metadata_digest:
        raise ValueError("initialization artifact metadata integrity digest mismatch")
    if not isinstance(normalized["creation_identity"], Mapping) or not isinstance(normalized["identity_metadata"], Mapping):
        raise ValueError("artifact creation identity metadata must be mappings")
    if not isinstance(normalized["resolved_optoelectronic_config"], Mapping):
        raise ValueError("artifact resolved_optoelectronic_config must be a mapping")
    if not isinstance(normalized["resolved_ranks"], Mapping) or set(normalized["resolved_ranks"]) != set(ENTRY_GROUPS):
        raise ValueError("artifact resolved_ranks must cover exactly the five entry groups")
    for group_id, rank in normalized["resolved_ranks"].items():
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0 or rank > _ENTRY_RANK_UPPER_BOUNDS[group_id]:
            raise ValueError(f"artifact resolved rank for {group_id!r} is invalid")
    if not isinstance(normalized["pca_reports"], Mapping) or set(normalized["pca_reports"]) != set(ENTRY_GROUPS):
        raise ValueError("artifact pca_reports must cover exactly the five entry groups")
    if normalized["physical_fit_status"] not in {"not_applicable", "unfitted_synthetic_initialization"}:
        raise ValueError("artifact physical_fit_status is invalid")
    for name in ("source_model_identity", "pca_rank_selection", "global_exposure", "trainability_contract", "global_physics", "claims"):
        if not isinstance(normalized[name], Mapping):
            raise ValueError(f"artifact metadata field {name!r} must be a mapping")
    return normalized


def _validate_artifact_payload(payload: object) -> tuple[Mapping[str, Any], dict[str, Any]]:
    if not isinstance(payload, Mapping):
        raise ValueError("initialization artifact payload must be a mapping")
    if set(payload) != _ARTIFACT_KEYS:
        raise ValueError("initialization artifact has missing or unexpected top-level keys")
    if type(payload["format_version"]) is not int or payload["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("initialization artifact format_version is unsupported")
    if payload["artifact_type"] != ARTIFACT_TYPE or payload["artifact_version"] != ARTIFACT_VERSION:
        raise ValueError("initialization artifact type or version is unsupported")
    state = payload["model_state_dict"]
    if not isinstance(state, Mapping):
        raise ValueError("initialization artifact model_state_dict must be a mapping")
    if payload["optimizer_state_dict"] is not None:
        raise ValueError("initialization artifact optimizer_state_dict must be null")
    return state, _validate_artifact_metadata_shape(payload["metadata"])


def _validate_rank_selection_against_config(
    selection: Mapping[str, Any], config: OptoelectronicH2FormerConfig
) -> None:
    if set(selection) != {"rule", "source", "variance_threshold", "performance_guarantee"}:
        raise ValueError("artifact pca_rank_selection has unexpected keys")
    if selection["performance_guarantee"] is not False:
        raise ValueError("PCA variance is an initialization rule, not a performance guarantee")
    if config.rank_by_group is not None:
        if selection["rule"] != "explicit_rank_by_group" or selection["source"] != "explicit" or selection["variance_threshold"] is not None:
            raise ValueError("artifact PCA rank selection conflicts with explicit ranks")
        return
    if selection["rule"] != "centered_pca_variance_threshold":
        raise ValueError("artifact PCA rank selection rule conflicts with resolved config")
    if selection["source"] not in {"project_default", "explicit"}:
        raise ValueError("artifact PCA rank selection source is invalid")
    if not _is_finite_number(selection["variance_threshold"]) or not math.isclose(
        float(selection["variance_threshold"]), float(config.resolved_variance_threshold), rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("artifact PCA rank selection threshold conflicts with resolved config")


def _compare_metadata_to_model(
    metadata: Mapping[str, Any],
    source: _LoadedSource,
    config_resolution: _ConfigResolution,
    model: OptoelectronicH2Former,
) -> None:
    config = config_resolution.config
    _validate_rank_selection_against_config(metadata["pca_rank_selection"], config)
    expected, _ = _build_artifact_metadata(
        source,
        _ConfigResolution(config=config, rank_selection=dict(metadata["pca_rank_selection"])),
        model,
        rank_selection=metadata["pca_rank_selection"],
    )
    stored = copy.deepcopy(dict(metadata))
    expected_copy = copy.deepcopy(expected)
    stored["source_checkpoint"]["path"] = "<source-checkpoint-path-excluded-from-comparison>"
    expected_copy["source_checkpoint"]["path"] = "<source-checkpoint-path-excluded-from-comparison>"
    if stored != expected_copy:
        raise ValueError("initialization artifact metadata does not match resolved model identity")



def _strict_json_tree_bytes(value: Any) -> bytes:
    """Return canonical JSON bytes after accepting only exact built-in JSON types."""

    def normalize(current: Any) -> Any:
        current_type = type(current)
        if current is None:
            return None
        if current_type is bool or current_type is int or current_type is str:
            return current
        if current_type is float:
            if not math.isfinite(current):
                raise ValueError("strict JSON state contains a non-finite float")
            return current
        if current_type is list:
            return [normalize(item) for item in current]
        if current_type is dict:
            normalized: dict[str, Any] = {}
            for key, item in current.items():
                if type(key) is not str:
                    raise ValueError("strict JSON state dictionary keys must be exact built-in strings")
                normalized[key] = normalize(item)
            return normalized
        raise ValueError(
            "strict JSON state contains a type outside the exact built-in JSON types"
        )

    return _canonical_json(normalize(value))


def _validate_initialization_state_matches_reconstruction(
    artifact_state: Mapping[str, Any],
    reconstructed_state: Mapping[str, Any],
) -> None:
    artifact_key_sequence = tuple(artifact_state.keys())
    if any(type(key) is not str for key in artifact_key_sequence):
        raise ValueError("initialization artifact state keys must be exact built-in strings")
    artifact_keys = set(artifact_key_sequence)
    reconstructed_keys = set(reconstructed_state)
    if artifact_keys != reconstructed_keys:
        raise ValueError(
            "initialization artifact state keys do not match the reconstructed initialization: "
            f"missing={sorted(reconstructed_keys - artifact_keys)!r}, "
            f"unexpected={sorted(artifact_keys - reconstructed_keys)!r}"
        )

    for key, expected in reconstructed_state.items():
        actual = artifact_state[key]
        if isinstance(expected, torch.Tensor):
            if type(actual) is not torch.Tensor:
                raise ValueError(f"initialization artifact state entry {key!r} is not a plain Tensor")
            if actual.is_meta:
                raise ValueError(f"initialization artifact state entry {key!r} is a meta tensor")
            if actual.is_quantized:
                raise ValueError(f"initialization artifact state entry {key!r} is quantized")
            if actual.layout is not torch.strided:
                raise ValueError(f"initialization artifact state entry {key!r} is not strided")
            if actual.shape != expected.shape:
                raise ValueError(f"initialization artifact state entry {key!r} has a mismatched shape")
            if actual.dtype != expected.dtype:
                raise ValueError(f"initialization artifact state entry {key!r} has a mismatched dtype")
            try:
                matches = torch.equal(
                    actual.detach().to(device="cpu"),
                    expected.detach().to(device="cpu"),
                )
            except Exception as error:
                raise ValueError(
                    f"initialization artifact state entry {key!r} could not be compared"
                ) from error
            if not matches:
                raise ValueError(
                    f"initialization artifact state entry {key!r} does not match the reconstructed initialization"
                )
        else:
            try:
                actual_bytes = _strict_json_tree_bytes(actual)
                expected_bytes = _strict_json_tree_bytes(expected)
            except ValueError as error:
                raise ValueError(
                    f"initialization artifact non-tensor state entry {key!r} is not a strict JSON tree: {error}"
                ) from error
            if actual_bytes != expected_bytes:
                raise ValueError(
                    f"initialization artifact non-tensor state entry {key!r} does not match the reconstructed initialization"
                )

def load_optoelectronic_initialization(

    artifact_path: str | Path,
    source_checkpoint_path: str | Path,
    *,
    map_location: Any = "cpu",
) -> OptoelectronicH2Former:
    """Reload an initialization artifact only with its matching source content."""

    artifact_file = Path(artifact_path).expanduser().resolve()
    if not artifact_file.is_file():
        raise FileNotFoundError(f"initialization artifact does not exist: {artifact_file}")
    try:
        payload = torch.load(artifact_file, map_location=map_location, weights_only=False)
    except Exception as error:
        raise ValueError(f"failed to read initialization artifact: {artifact_file}") from error
    artifact_state, metadata = _validate_artifact_payload(payload)
    source_file = Path(source_checkpoint_path).expanduser().resolve()
    source_sha256 = _sha256_file(source_file) if source_file.is_file() else None
    expected_source_sha256 = metadata["source_checkpoint"]["sha256"]
    if source_sha256 != expected_source_sha256:
        raise ValueError(
            "source checkpoint SHA256 does not match the initialization artifact: "
            f"expected {expected_source_sha256}, got {source_sha256}"
        )
    config_resolution = _build_config_resolution(metadata["resolved_optoelectronic_config"])
    source = _load_source_checkpoint(source_file, expected_sha256=expected_source_sha256)
    if source.metadata["model_name"] != metadata["source_model_identity"]["model_name"] or source.metadata["supervision_mode"] != metadata["source_model_identity"]["supervision_mode"]:
        raise ValueError("source checkpoint identity does not match initialization artifact")
    optical_model = OptoelectronicH2Former.from_h2former(
        source.model,
        config_resolution.config,
    )
    _compare_metadata_to_model(metadata, source, config_resolution, optical_model)
    _validate_initialization_state_matches_reconstruction(artifact_state, optical_model.state_dict())
    try:
        optical_model.load_state_dict(artifact_state, strict=True)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(
            f"initialization artifact optical state failed strict loading: {error}"
        ) from error
    return optical_model


# A descriptive alias for callers that prefer the artifact-oriented name.
create_optoelectronic_initialization_artifact = create_optoelectronic_initialization
initialize_optoelectronic_h2former = create_optoelectronic_initialization


__all__ = [
    "ARTIFACT_TYPE",
    "ARTIFACT_VERSION",
    "INITIALIZATION_FILENAME",
    "REPORT_FILENAME",
    "InitializationResult",
    "build_optoelectronic_config",
    "create_optoelectronic_initialization",
    "create_optoelectronic_initialization_artifact",
    "initialize_optoelectronic_h2former",
    "load_optoelectronic_initialization",
]
