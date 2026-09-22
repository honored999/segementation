"""Optoelectronic entrance replacement for the standalone H2Former.

The wrapper keeps :class:`H2Former`'s electronic encoder, fusion points, and
decoder unchanged.  Only its five input convolutions are replaced by PCA
component banks followed by either the explicit signed-dual-rail ideal
operator or the synthetic physical propagation operator from
``optoelectronic_frontend``.

This module is an initialization bridge, not a transparent checkpoint-resume
path.  ``from_h2former`` copies an existing H2Former state into a new model
identity and then replaces the five entrance convolutions.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from standalone_nnunet2d.models.h2former import H2Former
from standalone_nnunet2d.models.optoelectronic_frontend import (
    ActiveRoute,
    CenteredPCAResult,
    ExposureLedger,
    OptoelectronicConfig,
    PhaseOnlyPSF,
    build_exposure_ledger,
    decompose_conv2d,
    ideal_signed_conv2d,
    physical_dual_rail_forward,
)


ENTRY_GROUPS = ("stem7", "patch2", "patch4", "patch8", "patch16")
_IDENTITY_SCHEMA = "optoelectronic_h2former.creation_identity"
_IDENTITY_VERSION = 1
_IMMUTABLE_PCA_TENSOR_NAMES = (
    "mean_kernel",
    "basis",
    "coefficients",
    "component_bank",
    "pca_target_weight",
    "original_bias",
)


@dataclass(frozen=True)
class _EntrySpec:
    group_id: str
    path: str
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    bias: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "path": self.path,
            "in_channels": 1,
            "out_channels": self.out_channels,
            "kernel_size": (self.kernel_size, self.kernel_size),
            "stride": (self.stride, self.stride),
            "padding": (self.padding, self.padding),
            "bias": self.bias,
        }


_ENTRY_SPECS = (
    _EntrySpec("stem7", "conv1", 64, 7, 1, 3, False),
    _EntrySpec("patch2", "patch_embed.projs.0", 32, 2, 2, 0, True),
    _EntrySpec("patch4", "patch_embed.projs.1", 16, 4, 2, 1, True),
    _EntrySpec("patch8", "patch_embed.projs.2", 8, 8, 2, 3, True),
    _EntrySpec("patch16", "patch_embed.projs.3", 8, 16, 2, 7, True),
)


def _normalize_mode(mode: str) -> str:
    if not isinstance(mode, str):
        raise ValueError(f"mode must be 'ideal' or 'physical', got {mode!r}")
    normalized = mode.replace("-", "_")
    if normalized not in {"ideal", "physical"}:
        raise ValueError(f"mode must be 'ideal' or 'physical', got {mode!r}")
    return normalized


def _as_pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    return tuple(value)


def _validate_bool(value: object, name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean, got {value!r}")


def _runtime_scalar_snapshot(value: Any) -> tuple[str, Any]:
    if type(value) in {bool, int, float, str, type(None)}:
        return type(value).__name__, value
    return _type_name(value), id(value)


def _physical_config_runtime_snapshot(config: OptoelectronicConfig) -> tuple[Any, ...]:
    phase = config.phase
    return (
        _runtime_scalar_snapshot(config.dual_rail_mode),
        _runtime_scalar_snapshot(config.gain_update_mode),
        _runtime_scalar_snapshot(config.rho),
        _runtime_scalar_snapshot(config.min_throughput),
        _runtime_scalar_snapshot(config.max_electronic_gain),
        _runtime_scalar_snapshot(config.min_split_fraction),
        _runtime_scalar_snapshot(config.max_support_response_error),
        _runtime_scalar_snapshot(config.support_error_mode),
        _runtime_scalar_snapshot(phase.wavelength_m),
        _runtime_scalar_snapshot(phase.phase_grid_size),
        _runtime_scalar_snapshot(phase.phase_pitch_m),
        _runtime_scalar_snapshot(phase.focal_length_m),
        _runtime_scalar_snapshot(phase.aperture_diameter_m),
        _runtime_scalar_snapshot(phase.magnification),
    )


def _opto_config_runtime_snapshot(config: OptoelectronicH2FormerConfig) -> tuple[Any, ...]:
    if config.rank_by_group is None:
        ranks: tuple[tuple[str, tuple[str, Any]], ...] | None = None
    else:
        ranks = tuple(
            sorted(
                (str(group_id), _runtime_scalar_snapshot(rank))
                for group_id, rank in config.rank_by_group.items()
            )
        )
    return (
        _runtime_scalar_snapshot(config.mode),
        ranks,
        _runtime_scalar_snapshot(config.variance_threshold),
        _runtime_scalar_snapshot(config.trainable_phase),
        _runtime_scalar_snapshot(config.trainable_mixing),
        _runtime_scalar_snapshot(config.trainable_bias),
        _runtime_scalar_snapshot(config.trainable_electronic_backend),
        _physical_config_runtime_snapshot(config.physical_config),
        _runtime_scalar_snapshot(config.exposure_policy),
        _runtime_scalar_snapshot(config.route_batch_capacity),
    )


class _ImmutableRankMapping(Mapping[str, int]):
    """Tuple-backed read-only ranks that remain easy to copy and serialize."""

    __slots__ = ("_items", "_values")

    def __init__(self, values: Mapping[str, int]) -> None:
        self._items = tuple(sorted((str(key), int(value)) for key, value in values.items()))
        self._values = dict(self._items)

    def __getitem__(self, key: str) -> int:
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return f"_ImmutableRankMapping({dict(self._items)!r})"

    def __reduce__(self):
        return type(self), (dict(self._items),)

    def __deepcopy__(self, memo: dict[int, Any]):
        return self


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("identity metadata cannot contain non-finite floats")
        return value
    raise ValueError(f"identity metadata contains a non-JSON-safe value: {type(value).__name__}")


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _json_safe(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(f"identity metadata is not JSON serializable: {error}") from error


def _tensor_digest(tensor: Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    if not isinstance(tensor, Tensor):
        raise ValueError("immutable identity tensors must be torch tensors or None")
    contiguous_cpu = tensor.detach().to(device="cpu").contiguous()
    raw_bytes = contiguous_cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {
        "dtype": str(tensor.dtype),
        "shape": [int(size) for size in tensor.shape],
        "num_bytes": len(raw_bytes),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }


def _tensor_version(tensor: Tensor) -> int:
    version = getattr(tensor, "_version", None)
    if isinstance(version, bool) or not isinstance(version, int):
        raise RuntimeError("immutable identity tensor has no valid version counter")
    return int(version)


def _snapshot_digest(snapshot: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _type_name(value: object) -> str:
    value_type = value if isinstance(value, type) else type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


@dataclass(frozen=True)
class OptoelectronicH2FormerConfig:
    """Configuration for the isolated optoelectronic H2Former identity.

    ``rank_by_group`` and ``variance_threshold`` are mutually exclusive.  If
    no explicit rank map is supplied, ``variance_threshold=None`` resolves to
    the default 0.99 centered-PCA rule.  An explicit rank map must cover all
    five stable entrance group IDs, so no group silently falls back to a
    different selection rule.
    """

    mode: str = "ideal"
    rank_by_group: Mapping[str, int] | None = None
    variance_threshold: float | None = None
    trainable_phase: bool = True
    trainable_mixing: bool = False
    trainable_bias: bool = False
    trainable_electronic_backend: bool = True
    physical_config: OptoelectronicConfig = field(default_factory=OptoelectronicConfig)
    exposure_policy: str = "shared_input_global_parallel_equal_split"
    route_batch_capacity: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", _normalize_mode(self.mode))
        for name in (
            "trainable_phase",
            "trainable_mixing",
            "trainable_bias",
            "trainable_electronic_backend",
        ):
            _validate_bool(getattr(self, name), name)
        if not isinstance(self.physical_config, OptoelectronicConfig):
            raise ValueError("physical_config must be an OptoelectronicConfig")
        object.__setattr__(self, "physical_config", copy.deepcopy(self.physical_config))
        if not isinstance(self.exposure_policy, str) or not self.exposure_policy:
            raise ValueError("exposure_policy must be a non-empty string")
        if self.route_batch_capacity is not None and (
            isinstance(self.route_batch_capacity, bool)
            or not isinstance(self.route_batch_capacity, int)
            or self.route_batch_capacity <= 0
        ):
            raise ValueError("route_batch_capacity must be a positive integer or None")

        if self.rank_by_group is not None:
            if not isinstance(self.rank_by_group, Mapping):
                raise ValueError("rank_by_group must be a mapping or None")
            if self.variance_threshold is not None:
                raise ValueError("specify exactly one of rank_by_group or variance_threshold")
            normalized_ranks = {str(key): value for key, value in self.rank_by_group.items()}
            if set(normalized_ranks) != set(ENTRY_GROUPS):
                raise ValueError(
                    "rank_by_group must contain exactly the five entrance groups "
                    f"{ENTRY_GROUPS}, got {tuple(sorted(normalized_ranks))}"
                )
            for group_id, rank in normalized_ranks.items():
                if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
                    raise ValueError(f"rank for group {group_id!r} must be a non-negative integer")
            object.__setattr__(self, "rank_by_group", _ImmutableRankMapping(normalized_ranks))
        else:
            threshold = 0.99 if self.variance_threshold is None else self.variance_threshold
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, (int, float))
                or not math.isfinite(float(threshold))
                or not 0.0 <= float(threshold) <= 1.0
            ):
                raise ValueError("variance_threshold must be finite and in [0, 1]")
            object.__setattr__(self, "variance_threshold", float(threshold))

    @property
    def uses_explicit_ranks(self) -> bool:
        return self.rank_by_group is not None

    @property
    def resolved_variance_threshold(self) -> float | None:
        return None if self.uses_explicit_ranks else self.variance_threshold

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_class": "OptoelectronicH2Former",
            "mode": self.mode,
            "rank_by_group": None
            if self.rank_by_group is None
            else dict(self.rank_by_group),
            "variance_threshold": self.resolved_variance_threshold,
            "trainable_phase": self.trainable_phase,
            "trainable_mixing": self.trainable_mixing,
            "trainable_bias": self.trainable_bias,
            "trainable_electronic_backend": self.trainable_electronic_backend,
            "physical_config": self.physical_config.as_dict(),
            "exposure_policy": self.exposure_policy,
            "route_batch_capacity": self.route_batch_capacity,
        }


def _validate_source_entry(convolution: nn.Module, spec: _EntrySpec) -> nn.Conv2d:
    if not isinstance(convolution, nn.Conv2d):
        raise ValueError(f"{spec.path} must be an nn.Conv2d")
    if convolution.in_channels != 1:
        raise ValueError(f"{spec.path} must have in_channels=1, got {convolution.in_channels}")
    if convolution.out_channels != spec.out_channels:
        raise ValueError(
            f"{spec.path} must have out_channels={spec.out_channels}, got {convolution.out_channels}"
        )
    if _as_pair(convolution.kernel_size) != (spec.kernel_size, spec.kernel_size):
        raise ValueError(f"{spec.path} has an unsupported kernel_size {convolution.kernel_size}")
    if _as_pair(convolution.stride) != (spec.stride, spec.stride):
        raise ValueError(f"{spec.path} has an unsupported stride {convolution.stride}")
    if _as_pair(convolution.padding) != (spec.padding, spec.padding):
        raise ValueError(f"{spec.path} has an unsupported padding {convolution.padding}")
    if convolution.groups != 1 or _as_pair(convolution.dilation) != (1, 1):
        raise ValueError(f"{spec.path} must use groups=1 and dilation=1")
    if (convolution.bias is not None) is not spec.bias:
        raise ValueError(
            f"{spec.path} bias contract mismatch: expected {spec.bias}, "
            f"got {convolution.bias is not None}"
        )
    return convolution


class OptoelectronicPCAConv2d(nn.Module):
    """One entrance convolution represented by a frozen PCA component bank.

    In ideal mode, every component is evaluated by ``ideal_signed_conv2d``;
    the component responses are then reconstructed by one electronic 1x1
    mixing operation and the original bias is added exactly once.  Physical
    mode uses one ``PhaseOnlyPSF`` per active signed component route and the
    shared validated exposure ledger supplied by the wrapper model.
    """

    def __init__(
        self,
        convolution_or_decomposition: nn.Conv2d | CenteredPCAResult,
        *,
        group_id: str,
        mode: str = "ideal",
        rank: int | None = None,
        variance_threshold: float | None = None,
        physical_config: OptoelectronicConfig | None = None,
        trainable_phase: bool = True,
        trainable_mixing: bool = False,
        trainable_bias: bool = False,
        exposure_ledger: ExposureLedger | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("group_id must be a non-empty string")
        self.group_id = group_id
        self.mode = _normalize_mode(mode)
        _validate_bool(trainable_phase, "trainable_phase")
        _validate_bool(trainable_mixing, "trainable_mixing")
        _validate_bool(trainable_bias, "trainable_bias")

        if isinstance(convolution_or_decomposition, CenteredPCAResult):
            if rank is not None or variance_threshold is not None:
                raise ValueError(
                    "rank and variance_threshold are not accepted with a precomputed decomposition"
                )
            decomposition = convolution_or_decomposition
        elif isinstance(convolution_or_decomposition, nn.Conv2d):
            decomposition = decompose_conv2d(
                convolution_or_decomposition,
                rank=rank,
                variance_threshold=variance_threshold,
            )
        else:
            raise ValueError("expected an nn.Conv2d or CenteredPCAResult")

        self.in_channels = int(decomposition.original_weight.shape[1])
        self.out_channels = int(decomposition.original_weight.shape[0])
        self.kernel_size = tuple(int(value) for value in decomposition.kernel_size)
        self.stride = _as_pair(decomposition.stride)
        self.padding = _as_pair(decomposition.padding)
        self.dilation = (1, 1)
        self.groups = 1
        self.resolved_rank = decomposition.resolved_rank
        self.structural_rank_upper_bound = decomposition.structural_rank_upper_bound
        self.pca_report = decomposition.report
        self.physical_config = physical_config or OptoelectronicConfig()
        if not isinstance(self.physical_config, OptoelectronicConfig):
            raise ValueError("physical_config must be an OptoelectronicConfig")
        self._creation_physical_config = self.physical_config
        self._creation_physical_config_runtime_snapshot = _physical_config_runtime_snapshot(
            self.physical_config
        )
        self._creation_physical_config_snapshot = copy.deepcopy(self.physical_config.as_dict())
        self._creation_exposure_ledger: ExposureLedger | None = None
        self._creation_route_topology: tuple[tuple[int, tuple[tuple[str, ActiveRoute], ...]], ...] = ()
        self._creation_phase_keys: tuple[str, ...] = ()
        self._creation_phase_psf_refs: tuple[tuple[str, PhaseOnlyPSF], ...] = ()
        if self.mode == "physical" and self.physical_config.phase.phase_grid_size < max(self.kernel_size):
            raise ValueError(
                "physical phase_grid_size must be at least the entrance kernel size "
                f"{max(self.kernel_size)}, got {self.physical_config.phase.phase_grid_size}"
            )

        self.register_buffer("mean_kernel", decomposition.mean_kernel.detach().clone())
        self.register_buffer("basis", decomposition.basis.detach().clone())
        self.register_buffer("coefficients", decomposition.coefficients.detach().clone())
        self.register_buffer("component_bank", decomposition.component_bank.detach().clone())
        self.register_buffer(
            "pca_target_weight", decomposition.original_weight.detach().clone()
        )
        self.register_buffer(
            "original_bias",
            None
            if decomposition.original_bias is None
            else decomposition.original_bias.detach().clone(),
        )

        mixing = decomposition.default_mixing().detach().clone()
        if trainable_mixing:
            self.mixing = nn.Parameter(mixing)
        else:
            self.register_buffer("mixing", mixing)
        if decomposition.original_bias is None:
            self.register_parameter("bias", None)
        elif trainable_bias:
            self.bias = nn.Parameter(decomposition.original_bias.detach().clone())
        else:
            self.register_buffer("bias", decomposition.original_bias.detach().clone())
        self.mixing_trainable = bool(trainable_mixing)
        self.bias_trainable = bool(trainable_bias and decomposition.original_bias is not None)
        self.phase_trainable = bool(trainable_phase)

        self.phase_psfs = nn.ModuleDict()
        self.exposure_ledger: ExposureLedger | None = None
        self._routes_by_component: dict[int, dict[str, ActiveRoute]] = {}
        self.physical_fit_status = (
            "unfitted_synthetic_initialization" if self.mode == "physical" else "not_applicable"
        )
        if exposure_ledger is None:
            if self.mode == "physical":
                exposure_ledger = build_exposure_ledger(
                    {self.group_id: self.component_bank.detach()},
                    policy="shared_input_global_parallel_equal_split",
                )
            else:
                self.exposure_ledger = None
        if exposure_ledger is not None:
            self.bind_exposure_ledger(exposure_ledger)

    @property
    def component_count(self) -> int:
        return int(self.component_bank.shape[0])

    def bind_exposure_ledger(self, ledger: ExposureLedger) -> None:
        if not isinstance(ledger, ExposureLedger):
            raise ValueError("exposure_ledger must be an ExposureLedger")
        if self._creation_exposure_ledger is not None:
            raise RuntimeError("entry exposure ledger binding is immutable after construction")
        ledger.validate()
        self.exposure_ledger = ledger
        self._routes_by_component = {}
        if self.mode != "physical":
            self._creation_exposure_ledger = ledger
            self._creation_route_topology = ()
            self._creation_phase_keys = ()
            return
        self.phase_psfs = nn.ModuleDict()
        for route in ledger.canonical_routes:
            if route.group_id != self.group_id:
                continue
            self._routes_by_component.setdefault(route.component_index, {})[route.sign] = route
            key = self._phase_key(route.component_index, route.sign)
            phase_psf = PhaseOnlyPSF(
                self.physical_config.phase,
                trainable=self.phase_trainable,
            ).to(device=self.component_bank.device, dtype=self.component_bank.dtype)
            self.phase_psfs[key] = phase_psf
        self._creation_exposure_ledger = ledger
        self._creation_route_topology = tuple(
            (
                component_index,
                tuple(sorted(routes.items(), key=lambda item: item[0])),
            )
            for component_index, routes in sorted(self._routes_by_component.items())
        )
        self._creation_phase_keys = tuple(sorted(self.phase_psfs.keys()))
        self._creation_phase_psf_refs = tuple(sorted(self.phase_psfs.items()))

    @staticmethod
    def _phase_key(component_index: int, sign: str) -> str:
        return f"component{component_index}_{sign}"

    def reconstruct_weight(self) -> Tensor:
        flat = self.mixing @ self.component_bank.flatten(1)
        return flat.view(self.out_channels, self.in_channels, *self.kernel_size)

    def _assert_runtime_binding(
        self,
        *,
        expected_config: OptoelectronicConfig | None = None,
        expected_ledger: ExposureLedger | None = None,
        full: bool,
        validate_ledger: bool = True,
    ) -> None:
        if self.physical_config is not self._creation_physical_config:
            raise RuntimeError("entry binding drift: physical_config object was replaced")
        try:
            current_config_runtime_snapshot = _physical_config_runtime_snapshot(self.physical_config)
        except (AttributeError, TypeError, ValueError) as error:
            raise RuntimeError("entry binding drift: physical_config is malformed") from error
        if current_config_runtime_snapshot != self._creation_physical_config_runtime_snapshot:
            raise RuntimeError("entry binding drift: physical_config values changed")
        if expected_config is not None and self.physical_config is not expected_config:
            raise RuntimeError("entry binding drift: physical_config is not the root creation config")
        if full:
            try:
                current_config_snapshot = _json_safe(self.physical_config.as_dict())
            except (AttributeError, TypeError, ValueError) as error:
                raise RuntimeError("entry binding drift: physical_config serialization failed") from error
            if current_config_snapshot != self._creation_physical_config_snapshot:
                raise RuntimeError("entry binding drift: physical_config snapshot changed")

        if self.exposure_ledger is not self._creation_exposure_ledger:
            raise RuntimeError("entry binding drift: exposure ledger object was replaced")
        if expected_ledger is not None and self.exposure_ledger is not expected_ledger:
            raise RuntimeError("entry binding drift: exposure ledger is not the root creation ledger")
        if self.exposure_ledger is not None:
            if validate_ledger:
                self.exposure_ledger.validate_runtime()
                if full:
                    self.exposure_ledger.validate()
            elif not full:
                self.exposure_ledger.validate_runtime()

        if tuple(sorted(self.phase_psfs.keys())) != self._creation_phase_keys:
            raise RuntimeError("entry binding drift: phase route topology changed")
        for phase_key, expected_phase_psf in self._creation_phase_psf_refs:
            if self.phase_psfs[phase_key] is not expected_phase_psf:
                raise RuntimeError(
                    "entry binding drift: PhaseOnlyPSF module object was replaced"
                )
        if len(self._routes_by_component) != len(self._creation_route_topology):
            raise RuntimeError("entry binding drift: route topology changed")
        for component_index, expected_routes in self._creation_route_topology:
            current_routes = self._routes_by_component.get(component_index)
            if not isinstance(current_routes, dict):
                raise RuntimeError("entry binding drift: component route topology changed")
            if tuple(sorted(current_routes.keys())) != tuple(sign for sign, _ in expected_routes):
                raise RuntimeError("entry binding drift: route sign topology changed")
            for sign, expected_route in expected_routes:
                if current_routes.get(sign) is not expected_route:
                    raise RuntimeError("entry binding drift: canonical route object was replaced")

    def state_dict(self, *args: Any, **kwargs: Any):
        self._assert_runtime_binding(full=True)
        return super().state_dict(*args, **kwargs)

    def pca_metadata(self) -> dict[str, Any]:
        self._assert_runtime_binding(full=True)
        return {
            "group_id": self.group_id,
            "resolved_rank": self.resolved_rank,
            "structural_rank_upper_bound": self.structural_rank_upper_bound,
            "report": self.pca_report.as_dict(),
            "pca_target_trainable": False,
            "mean_and_basis_trainable": False,
            "component_count": self.component_count,
        }

    def _ideal_components(self, x: Tensor) -> Tensor:
        responses: list[Tensor] = []
        for component in self.component_bank:
            responses.append(
                ideal_signed_conv2d(
                    x,
                    component.unsqueeze(0),
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                )
            )
        return torch.cat(responses, dim=1)

    def _physical_components(self, x: Tensor) -> Tensor:
        if self.exposure_ledger is None:
            raise RuntimeError("physical mode requires a bound ExposureLedger")
        responses: list[Tensor] = []
        for component_index in range(self.component_count):
            routes = self._routes_by_component.get(component_index, {})
            positive_route = routes.get("positive")
            negative_route = routes.get("negative")
            positive_psf = (
                self.phase_psfs[self._phase_key(component_index, "positive")]()
                if positive_route is not None
                else None
            )
            negative_psf = (
                self.phase_psfs[self._phase_key(component_index, "negative")]()
                if negative_route is not None
                else None
            )
            result = physical_dual_rail_forward(
                x,
                positive_psf=positive_psf,
                negative_psf=negative_psf,
                positive_route=positive_route,
                negative_route=negative_route,
                support=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=None,
                config=self.physical_config,
                ledger=self.exposure_ledger,
            )
            responses.append(result.output)
        return torch.cat(responses, dim=1)

    def forward(self, x: Tensor) -> Tensor:
        self._assert_runtime_binding(full=False)
        if x.ndim != 4:
            raise ValueError(f"OptoelectronicPCAConv2d expects BCHW input, got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"OptoelectronicPCAConv2d expected {self.in_channels} input channels, got {x.shape[1]}"
            )
        components = (
            self._ideal_components(x)
            if self.mode == "ideal"
            else self._physical_components(x)
        )
        output = F.conv2d(
            components,
            self.mixing.view(self.out_channels, self.component_count, 1, 1),
            bias=None,
        )
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        return output


class OptoelectronicH2Former(H2Former):
    """H2Former with only its five entrance convolutions optoelectronically replaced."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        image_size: int = 512,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes, image_size=image_size)

    @staticmethod
    def _source_entries(source: H2Former) -> tuple[nn.Conv2d, ...]:
        if not isinstance(source, H2Former):
            raise ValueError("source must be an H2Former instance")
        if source.in_channels != 1:
            raise ValueError(
                f"OptoelectronicH2Former supports only single-channel H2Former, got {source.in_channels}"
            )
        if source.image_size != 512:
            raise ValueError(
                f"OptoelectronicH2Former supports only image_size=512, got {source.image_size}"
            )
        if len(source.patch_embed.projs) != 4:
            raise ValueError("source H2Former must contain exactly four PatchEmbed projections")
        entries: list[nn.Conv2d] = []
        for spec in _ENTRY_SPECS:
            if spec.path == "conv1":
                module = source.conv1
            else:
                module = source.patch_embed.projs[int(spec.path.rsplit(".", 1)[1])]
            entries.append(_validate_source_entry(module, spec))
        return tuple(entries)

    @classmethod
    def from_h2former(
        cls,
        source: H2Former,
        config: OptoelectronicH2FormerConfig | None = None,
    ) -> "OptoelectronicH2Former":
        """Create a new architecture initialized from ``source`` weights.

        The source is validated and read only.  This method intentionally does
        not claim checkpoint-resume identity with the original H2Former.
        """

        config = config or OptoelectronicH2FormerConfig()
        if not isinstance(config, OptoelectronicH2FormerConfig):
            raise ValueError("config must be an OptoelectronicH2FormerConfig")
        source_entries = cls._source_entries(source)
        source_state = source.state_dict()
        source_state_identity = cls._source_state_identity(source_state)
        source_parameter = next(source.parameters())
        model = cls(
            in_channels=source.in_channels,
            num_classes=source.num_classes,
            image_size=source.image_size,
        ).to(device=source_parameter.device, dtype=source_parameter.dtype)
        cls._copy_source_state(source, model)
        model.train(source.training)

        decompositions: dict[str, CenteredPCAResult] = {}
        for spec, convolution in zip(_ENTRY_SPECS, source_entries):
            if config.rank_by_group is None:
                decomposition = decompose_conv2d(
                    convolution,
                    variance_threshold=config.resolved_variance_threshold,
                )
            else:
                decomposition = decompose_conv2d(
                    convolution,
                    rank=config.rank_by_group[spec.group_id],
                    variance_threshold=None,
                )
            decompositions[spec.group_id] = decomposition

        ledger = build_exposure_ledger(
            {group_id: decomposition.component_bank for group_id, decomposition in decompositions.items()},
            policy=config.exposure_policy,
            route_batch_capacity=config.route_batch_capacity,
        )

        def make_entry(spec: _EntrySpec) -> OptoelectronicPCAConv2d:
            return OptoelectronicPCAConv2d(
                decompositions[spec.group_id],
                group_id=spec.group_id,
                mode=config.mode,
                physical_config=config.physical_config,
                trainable_phase=config.trainable_phase,
                trainable_mixing=config.trainable_mixing,
                trainable_bias=config.trainable_bias,
                exposure_ledger=ledger,
            ).to(device=source_parameter.device, dtype=source_parameter.dtype)

        model.conv1 = make_entry(_ENTRY_SPECS[0])
        model.patch_embed.projs = nn.ModuleList(
            [make_entry(spec) for spec in _ENTRY_SPECS[1:]]
        )
        model.exposure_ledger = ledger
        model.opto_config = config
        model._set_trainability(config)
        model._initialize_creation_identity(config, source_state_identity)
        return model

    def _entry_modules(self) -> tuple[OptoelectronicPCAConv2d, ...]:
        return (self.conv1, *tuple(self.patch_embed.projs))  # type: ignore[return-value]

    @staticmethod
    def _source_state_identity(source_state: Mapping[str, Any]) -> dict[str, Any]:
        identity: dict[str, Any] = {}
        for name, value in source_state.items():
            if name == "_extra_state" or name.startswith("conv1.") or name.startswith("patch_embed.projs."):
                continue
            if not isinstance(value, Tensor):
                raise ValueError(f"source state entry {name!r} is not a tensor")
            identity[name] = _tensor_digest(value)
        return _json_safe(identity)

    @staticmethod
    def _copy_source_state(source: H2Former, target: "OptoelectronicH2Former") -> None:
        source_parameters = dict(source.named_parameters())
        target_parameters = dict(target.named_parameters())
        if set(source_parameters) != set(target_parameters):
            raise ValueError("source and target H2Former parameter identities do not match")
        source_buffers = dict(source.named_buffers())
        target_buffers = dict(target.named_buffers())
        if set(source_buffers) != set(target_buffers):
            raise ValueError("source and target H2Former buffer identities do not match")
        with torch.no_grad():
            for name, target_parameter in target_parameters.items():
                source_parameter = source_parameters[name]
                target_parameter.copy_(
                    source_parameter.detach().to(
                        device=target_parameter.device,
                        dtype=target_parameter.dtype,
                    )
                )
            for name, target_buffer in target_buffers.items():
                source_buffer = source_buffers[name]
                target_buffer.copy_(
                    source_buffer.detach().to(
                        device=target_buffer.device,
                        dtype=target_buffer.dtype,
                    )
                )

    @staticmethod
    def _entry_state_prefix(spec: _EntrySpec) -> str:
        if spec.path == "conv1":
            return "conv1"
        return f"patch_embed.projs.{spec.path.rsplit('.', 1)[1]}"

    @staticmethod
    def _compact_ledger_identity(ledger: ExposureLedger) -> dict[str, Any]:
        serialized = ledger.as_dict()
        return _json_safe(
            {
                "policy": serialized["policy"],
                "active_group_count": serialized["active_group_count"],
                "active_route_count": serialized["active_route_count"],
                "exposure_count": serialized["exposure_count"],
                "canonical_route_identity_digests": serialized["canonical_route_identity_digests"],
                "canonical_route_identities": serialized["canonical_route_identities"],
                "exposures": serialized["exposures"],
            }
        )

    @staticmethod
    def _parameter_contract(model: "OptoelectronicH2Former") -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "shape": [int(size) for size in parameter.shape],
                "dtype": str(parameter.dtype),
                "requires_grad": bool(parameter.requires_grad),
            }
            for name, parameter in sorted(model.named_parameters())
        ]

    def _initialize_creation_identity(
        self,
        config: OptoelectronicH2FormerConfig,
        source_state_identity: Mapping[str, Any],
    ) -> None:
        entries = self._entry_modules()
        ledger_identity = self._compact_ledger_identity(self.exposure_ledger)
        route_identities = ledger_identity["canonical_route_identities"]
        immutable_state_identity: dict[str, Any] = {}
        immutable_tensor_refs: dict[str, tuple[nn.Module, str, Tensor, int]] = {}
        groups: list[dict[str, Any]] = []

        def register_immutable_tensor(
            state_key: str,
            owner: nn.Module,
            attribute_name: str,
            tensor: Tensor,
        ) -> dict[str, Any]:
            digest = _tensor_digest(tensor)
            if digest is None:
                raise RuntimeError(f"immutable identity tensor {state_key!r} is missing")
            immutable_state_identity[state_key] = copy.deepcopy(digest)
            immutable_tensor_refs[state_key] = (
                owner,
                attribute_name,
                tensor,
                _tensor_version(tensor),
            )
            return digest

        for spec, module in zip(_ENTRY_SPECS, entries):
            prefix = self._entry_state_prefix(spec)
            tensor_identities: dict[str, Any] = {}
            for tensor_name in _IMMUTABLE_PCA_TENSOR_NAMES:
                tensor = getattr(module, tensor_name)
                digest = _tensor_digest(tensor)
                tensor_identities[tensor_name] = digest
                if tensor is not None:
                    register_immutable_tensor(f"{prefix}.{tensor_name}", module, tensor_name, tensor)

            module_routes = [
                route
                for route in route_identities
                if route["group_id"] == module.group_id
            ]
            phase_apertures: list[dict[str, Any]] = []
            if module.mode == "physical":
                expected_phase_keys = {
                    f"component{route['component_index']}_{route['sign']}"
                    for route in module_routes
                }
                if set(module.phase_psfs.keys()) != expected_phase_keys:
                    raise RuntimeError(
                        f"physical phase route topology is incomplete for {module.group_id!r}"
                    )
                for route in module_routes:
                    phase_key = f"component{route['component_index']}_{route['sign']}"
                    phase_psf = module.phase_psfs[phase_key]
                    aperture = getattr(phase_psf, "aperture", None)
                    if not isinstance(aperture, Tensor):
                        raise RuntimeError(
                            f"phase route {phase_key!r} has no tensor aperture"
                        )
                    state_key = f"{prefix}.phase_psfs.{phase_key}.aperture"
                    aperture_digest = register_immutable_tensor(
                        state_key,
                        phase_psf,
                        "aperture",
                        aperture,
                    )
                    phase_apertures.append(
                        {
                            "phase_key": phase_key,
                            "state_key": state_key,
                            "group_id": module.group_id,
                            "component_index": route["component_index"],
                            "sign": route["sign"],
                            "route_id": route["route_id"],
                            "route_identity_digest": route["digest"],
                            **copy.deepcopy(aperture_digest),
                        }
                    )
            groups.append(
                {
                    "group_id": module.group_id,
                    "kernel_contract": {
                        "in_channels": module.in_channels,
                        "out_channels": module.out_channels,
                        "kernel_size": list(module.kernel_size),
                        "stride": list(module.stride),
                        "padding": list(module.padding),
                        "dilation": list(module.dilation),
                        "groups": module.groups,
                        "bias": module.bias is not None,
                    },
                    "resolved_rank": module.resolved_rank,
                    "structural_rank_upper_bound": module.structural_rank_upper_bound,
                    "pca_report": module.pca_report.as_dict(),
                    "immutable_tensors": tensor_identities,
                    "initialization": {
                        "mixing": {
                            "digest": _tensor_digest(module.mixing.detach()),
                            "relation": "mixing[:,0]=1 and mixing[:,1:]=coefficients",
                        },
                        "bias": {
                            "present": module.bias is not None,
                            "initial_digest": _tensor_digest(
                                None if module.bias is None else module.bias.detach()
                            ),
                        },
                    },
                    "routes": module_routes,
                    "phase_apertures": phase_apertures,
                }
            )

        trainable = {
            "phase": config.trainable_phase,
            "mixing": config.trainable_mixing,
            "bias": config.trainable_bias,
            "electronic_backend": config.trainable_electronic_backend,
            "pca_target_weight": False,
            "pca_mean_and_basis": False,
        }
        snapshot = _json_safe(
            {
                "schema": _IDENTITY_SCHEMA,
                "version": _IDENTITY_VERSION,
                "model_class": type(self).__name__,
                "architecture_identity": "optoelectronic_h2former_from_h2former",
                "source_architecture": "H2Former",
                "initialization_semantics": (
                    "new_architecture_initialized_from_h2former_weights_not_resume"
                ),
                "mode": config.mode,
                "in_channels": self.in_channels,
                "num_classes": self.num_classes,
                "image_size": self.image_size,
                "supervision_mode": "single_output",
                "normalization_config": {
                    "encoder_norm_layer": _type_name(self._norm_layer),
                    "patch_embedding_norm_layer": _type_name(self.patch_embed.norm),
                },
                "entry_convolutions": [spec.as_dict() for spec in _ENTRY_SPECS],
                "groups": groups,
                "resolved_ranks": {
                    group_id: next(
                        group["resolved_rank"] for group in groups if group["group_id"] == group_id
                    )
                    for group_id in ENTRY_GROUPS
                },
                "exposure_policy": config.exposure_policy,
                "route_batch_capacity": config.route_batch_capacity,
                "exposure_ledger": ledger_identity,
                "routes": route_identities,
                "synthetic_physical_assumptions": config.physical_config.as_dict(),
                "config": config.as_dict(),
                "trainable": trainable,
                "parameter_contract": self._parameter_contract(self),
                "source_h2former_state": copy.deepcopy(dict(source_state_identity)),
            }
        )
        snapshot["snapshot_sha256"] = _snapshot_digest(snapshot)
        self._creation_identity_snapshot = snapshot
        self._creation_identity_digest = snapshot["snapshot_sha256"]
        self._creation_immutable_state_identity = immutable_state_identity
        self._creation_immutable_tensor_refs = immutable_tensor_refs
        self._creation_entry_module_refs = tuple(entries)
        self._creation_ledger = self.exposure_ledger
        self._creation_config = config
        self._creation_config_runtime_snapshot = _opto_config_runtime_snapshot(config)

    def _require_creation_identity(self) -> None:
        if not hasattr(self, "_creation_identity_snapshot"):
            raise RuntimeError(
                "OptoelectronicH2Former creation identity is unavailable; "
                "initialize with from_h2former first"
            )

    def _assert_live_identity_intact(self, *, full: bool) -> None:
        self._require_creation_identity()
        if self.opto_config is not self._creation_config:
            raise RuntimeError("creation identity drift: configuration object was replaced")
        if self.exposure_ledger is not self._creation_ledger:
            raise RuntimeError("creation identity drift: exposure ledger was replaced")
        self.exposure_ledger.validate_runtime()
        if full:
            self.exposure_ledger.validate()

        entries = self._entry_modules()
        if tuple(id(module) for module in entries) != tuple(
            id(module) for module in self._creation_entry_module_refs
        ):
            raise RuntimeError("creation identity drift: entry module topology changed")
        try:
            current_config_runtime_snapshot = _opto_config_runtime_snapshot(self.opto_config)
        except (AttributeError, TypeError, ValueError) as error:
            raise RuntimeError("creation identity drift: configuration is malformed") from error
        if current_config_runtime_snapshot != self._creation_config_runtime_snapshot:
            raise RuntimeError("creation identity drift: configuration values changed")
        if full:
            try:
                current_config_snapshot = _json_safe(self.opto_config.as_dict())
            except (AttributeError, TypeError, ValueError) as error:
                raise RuntimeError("creation identity drift: configuration serialization failed") from error
            if current_config_snapshot != self._creation_identity_snapshot["config"]:
                raise RuntimeError("creation identity drift: configuration snapshot changed")
        for module in entries:
            module._assert_runtime_binding(
                expected_config=self._creation_config.physical_config,
                expected_ledger=self._creation_ledger,
                full=full,
                validate_ledger=False,
            )
        if self._parameter_contract(self) != self._creation_identity_snapshot["parameter_contract"]:
            raise RuntimeError("creation identity drift: trainable/frozen parameter strategy changed")

        for expected_group, module in zip(self._creation_identity_snapshot["groups"], entries):
            if module.group_id != expected_group["group_id"]:
                raise RuntimeError("creation identity drift: group identity changed")
            if module.resolved_rank != expected_group["resolved_rank"]:
                raise RuntimeError("creation identity drift: resolved rank changed")
            if _json_safe(module.pca_report.as_dict()) != expected_group["pca_report"]:
                raise RuntimeError("creation identity drift: PCA report changed")
            expected_route_ids = sorted(route["route_id"] for route in expected_group["routes"])
            if module.mode == "physical":
                current_route_ids = sorted(
                    route.route_id
                    for routes in module._routes_by_component.values()
                    for route in routes.values()
                )
                if current_route_ids != expected_route_ids:
                    raise RuntimeError("creation identity drift: route topology changed")
            elif module._routes_by_component:
                raise RuntimeError("creation identity drift: ideal route topology changed")
            expected_phase_keys = (
                {
                    f"component{route['component_index']}_{route['sign']}"
                    for route in expected_group["routes"]
                }
                if module.mode == "physical"
                else set()
            )
            if set(module.phase_psfs.keys()) != expected_phase_keys:
                raise RuntimeError("creation identity drift: phase route topology changed")

            expected_aperture_keys = {
                aperture["phase_key"] for aperture in expected_group["phase_apertures"]
            }
            if expected_aperture_keys != expected_phase_keys:
                raise RuntimeError("creation identity drift: aperture identity topology changed")
            current_aperture_keys = {
                phase_key
                for phase_key, phase_psf in module.phase_psfs.items()
                if isinstance(getattr(phase_psf, "aperture", None), Tensor)
            }
            if current_aperture_keys != expected_aperture_keys:
                raise RuntimeError("creation identity drift: aperture topology changed")

        for state_key, (owner, attribute_name, creation_tensor, baseline_version) in self._creation_immutable_tensor_refs.items():
            current_tensor = getattr(owner, attribute_name, None)
            if not isinstance(current_tensor, Tensor):
                raise RuntimeError(f"immutable identity tensor {state_key!r} is missing")
            if id(current_tensor) != id(creation_tensor):
                raise RuntimeError(f"immutable identity tensor {state_key!r} was replaced")
            if _tensor_version(current_tensor) != baseline_version:
                raise RuntimeError(f"immutable identity tensor {state_key!r} was mutated in place")
            if full and _tensor_digest(current_tensor) != self._creation_immutable_state_identity[state_key]:
                raise RuntimeError(f"immutable identity tensor {state_key!r} content changed")

        if full and self._compact_ledger_identity(self.exposure_ledger) != self._creation_identity_snapshot[
            "exposure_ledger"
        ]:
            raise RuntimeError("creation identity drift: exposure ledger identity changed")

    @staticmethod
    def _validate_extra_state_payload(state: Any) -> dict[str, Any]:
        if not isinstance(state, Mapping):
            raise ValueError("state_dict is missing a mapping-valued identity extra state")
        expected_keys = {"schema", "version", "snapshot", "snapshot_sha256"}
        actual_keys = set(state.keys())
        if actual_keys != expected_keys:
            raise ValueError(
                "identity extra state wrapper keys must be exact: "
                f"missing={sorted(expected_keys - actual_keys)!r}, "
                f"unexpected={sorted(actual_keys - expected_keys)!r}"
            )
        if state.get("schema") != _IDENTITY_SCHEMA:
            raise ValueError("identity extra state has an invalid schema")
        if state.get("version") != _IDENTITY_VERSION:
            raise ValueError("identity extra state has an unsupported version")
        snapshot = state.get("snapshot")
        if not isinstance(snapshot, Mapping):
            raise ValueError("identity extra state is missing its snapshot")
        try:
            normalized_snapshot = _json_safe(snapshot)
        except ValueError as error:
            raise ValueError(f"identity snapshot is not JSON-safe: {error}") from error
        if normalized_snapshot.get("schema") != _IDENTITY_SCHEMA:
            raise ValueError("identity snapshot is missing the expected schema")
        if normalized_snapshot.get("version") != _IDENTITY_VERSION:
            raise ValueError("identity snapshot is missing the expected version")
        expected_digest = normalized_snapshot.get("snapshot_sha256")
        supplied_digest = state.get("snapshot_sha256")
        if not isinstance(expected_digest, str) or not isinstance(supplied_digest, str):
            raise ValueError("identity snapshot is missing its digest")
        computed_digest = _snapshot_digest(normalized_snapshot)
        if expected_digest != computed_digest or supplied_digest != computed_digest:
            raise ValueError("identity snapshot digest is invalid")
        return normalized_snapshot

    def get_extra_state(self) -> dict[str, Any]:
        self._assert_live_identity_intact(full=True)
        return {
            "schema": _IDENTITY_SCHEMA,
            "version": _IDENTITY_VERSION,
            "snapshot": copy.deepcopy(self._creation_identity_snapshot),
            "snapshot_sha256": self._creation_identity_digest,
        }

    def set_extra_state(self, state: Any) -> None:
        if not getattr(self, "_load_state_dict_in_progress", False):
            self._assert_live_identity_intact(full=True)
        incoming_snapshot = self._validate_extra_state_payload(state)
        if incoming_snapshot != self._creation_identity_snapshot:
            raise ValueError(
                "incoming creation identity does not exactly match the target model identity"
            )

    def _validate_incoming_pca_state(
        self,
        state_dict: Mapping[str, Any],
        incoming_snapshot: Mapping[str, Any],
    ) -> None:
        for state_key, expected_digest in self._creation_immutable_state_identity.items():
            if state_key not in state_dict:
                raise ValueError(f"state_dict is missing immutable identity tensor {state_key!r}")
            value = state_dict[state_key]
            if not isinstance(value, Tensor):
                raise ValueError(f"state_dict identity tensor {state_key!r} is not a tensor")
            if _tensor_digest(value) != expected_digest:
                raise ValueError(f"state_dict immutable tensor digest mismatch for {state_key!r}")

        for spec, module in zip(_ENTRY_SPECS, self._entry_modules()):
            prefix = self._entry_state_prefix(spec)
            values = {
                name: state_dict[f"{prefix}.{name}"]
                for name in _IMMUTABLE_PCA_TENSOR_NAMES
                if f"{prefix}.{name}" in state_dict
            }
            expected_shapes = {
                "mean_kernel": (1, module.in_channels, *module.kernel_size),
                "basis": (module.resolved_rank, module.in_channels, *module.kernel_size),
                "coefficients": (module.out_channels, module.resolved_rank),
                "component_bank": (
                    module.resolved_rank + 1,
                    module.in_channels,
                    *module.kernel_size,
                ),
                "pca_target_weight": (
                    module.out_channels,
                    module.in_channels,
                    *module.kernel_size,
                ),
            }
            for name, shape in expected_shapes.items():
                value = values.get(name)
                if not isinstance(value, Tensor) or tuple(value.shape) != shape:
                    raise ValueError(f"incoming PCA buffer {prefix}.{name} has an invalid shape")
                if value.dtype != module.component_bank.dtype:
                    raise ValueError(f"incoming PCA buffer {prefix}.{name} has an invalid dtype")
            original_bias = values.get("original_bias")
            if module.original_bias is not None:
                if not isinstance(original_bias, Tensor) or tuple(original_bias.shape) != (
                    module.out_channels,
                ):
                    raise ValueError(f"incoming PCA buffer {prefix}.original_bias has an invalid shape")
            if not torch.equal(
                values["component_bank"][:1].detach().cpu(),
                values["mean_kernel"].detach().cpu(),
            ):
                raise ValueError(f"incoming component_bank[:1] does not match {prefix}.mean_kernel")
            if not torch.equal(
                values["component_bank"][1:].detach().cpu(),
                values["basis"].detach().cpu(),
            ):
                raise ValueError(f"incoming component_bank[1:] does not match {prefix}.basis")

        incoming_components = {
            module.group_id: state_dict[f"{self._entry_state_prefix(spec)}.component_bank"]
            for spec, module in zip(_ENTRY_SPECS, self._entry_modules())
        }
        expected_ledger = build_exposure_ledger(
            incoming_components,
            policy=incoming_snapshot["exposure_policy"],
            route_batch_capacity=incoming_snapshot["route_batch_capacity"],
        )
        if self._compact_ledger_identity(expected_ledger) != incoming_snapshot["exposure_ledger"]:
            raise ValueError("incoming ledger identity does not match PCA component banks")

    @staticmethod
    def _validate_incoming_tensor_layout(
        state_dict: Mapping[str, Any],
        current_state: Mapping[str, Any],
    ) -> None:
        for name, current_value in current_state.items():
            if name not in state_dict or not isinstance(current_value, Tensor):
                continue
            incoming_value = state_dict[name]
            if type(incoming_value) is not Tensor:
                raise ValueError(f"state_dict entry {name!r} is not a plain tensor")
            if incoming_value.is_meta:
                raise ValueError(f"state_dict entry {name!r} uses the meta device")
            if incoming_value.is_quantized:
                raise ValueError(f"state_dict entry {name!r} is quantized")
            if incoming_value.layout != torch.strided:
                raise ValueError(
                    f"state_dict entry {name!r} has unsupported layout {incoming_value.layout}"
                )
            if tuple(incoming_value.shape) != tuple(current_value.shape):
                raise ValueError(
                    f"state_dict entry {name!r} has shape {tuple(incoming_value.shape)}, "
                    f"expected {tuple(current_value.shape)}"
                )
            if incoming_value.dtype != current_value.dtype:
                raise ValueError(
                    f"state_dict entry {name!r} has dtype {incoming_value.dtype}, "
                    f"expected {current_value.dtype}"
                )

    def _refresh_immutable_tensor_baseline_versions(self) -> None:
        refreshed: dict[str, tuple[nn.Module, str, Tensor, int]] = {}
        for state_key, (owner, attribute_name, creation_tensor, _) in self._creation_immutable_tensor_refs.items():
            current_tensor = getattr(owner, attribute_name, None)
            if current_tensor is not creation_tensor:
                raise RuntimeError(
                    f"immutable identity tensor {state_key!r} changed object during load"
                )
            if not isinstance(current_tensor, Tensor):
                raise RuntimeError(f"immutable identity tensor {state_key!r} is missing after load")
            refreshed[state_key] = (
                owner,
                attribute_name,
                creation_tensor,
                _tensor_version(current_tensor),
            )
        self._creation_immutable_tensor_refs = refreshed

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        if assign is not False:
            raise ValueError("assign=True is not supported; use assign=False for identity-safe loading")
        if strict is not True:
            raise ValueError("strict=False is not supported; strict identity restoration is required")
        self._assert_live_identity_intact(full=True)
        if not isinstance(state_dict, Mapping):
            raise TypeError("state_dict must be a mapping")
        if "_extra_state" not in state_dict:
            raise ValueError("state_dict is missing the required identity extra state")
        incoming_snapshot = self._validate_extra_state_payload(state_dict["_extra_state"])
        if incoming_snapshot != self._creation_identity_snapshot:
            raise ValueError(
                "incoming creation identity does not exactly match the target model identity"
            )

        current_state = nn.Module.state_dict(self)
        missing_keys = sorted(set(current_state) - set(state_dict))
        unexpected_keys = sorted(set(state_dict) - set(current_state))
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                "state_dict preflight rejected incompatible keys: "
                f"missing={missing_keys!r}, unexpected={unexpected_keys!r}"
            )
        self._validate_incoming_tensor_layout(state_dict, current_state)
        self._validate_incoming_pca_state(state_dict, incoming_snapshot)

        self._load_state_dict_in_progress = True
        try:
            result = super().load_state_dict(state_dict, strict=True, assign=False)
            self._refresh_immutable_tensor_baseline_versions()
        finally:
            if getattr(self, "_load_state_dict_in_progress", False):
                del self._load_state_dict_in_progress
        self._assert_live_identity_intact(full=False)
        self._assert_live_identity_intact(full=True)
        return result

    def _set_trainability(self, config: OptoelectronicH2FormerConfig) -> None:
        entry_module_ids = {id(module) for module in self._entry_modules()}
        for module in self.modules():
            if id(module) in entry_module_ids:
                continue
            for parameter in module.parameters(recurse=False):
                parameter.requires_grad_(config.trainable_electronic_backend)
        for module in self._entry_modules():
            if isinstance(module.mixing, nn.Parameter):
                module.mixing.requires_grad_(config.trainable_mixing)
            if isinstance(module.bias, nn.Parameter):
                module.bias.requires_grad_(config.trainable_bias)
            for phase_psf in module.phase_psfs.values():
                phase_psf.theta.requires_grad_(config.trainable_phase)

    def forward(self, x: Tensor) -> Tensor:
        self._assert_live_identity_intact(full=False)
        return super().forward(x)

    def identity_metadata(self) -> dict[str, Any]:
        self._assert_live_identity_intact(full=True)
        config = self.opto_config
        entries = self._entry_modules()
        pca = {module.group_id: module.pca_metadata() for module in entries}
        global_exposure = self.exposure_ledger.as_dict()
        metadata = {
            "model_class": type(self).__name__,
            "architecture_identity": "optoelectronic_h2former_from_h2former",
            "source_architecture": "H2Former",
            "initialization_semantics": "new_architecture_initialized_from_h2former_weights_not_resume",
            "mode": config.mode,
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "image_size": self.image_size,
            "supervision_mode": "single_output",
            "entry_convolutions": [spec.as_dict() for spec in _ENTRY_SPECS],
            "resolved_ranks": {group_id: pca[group_id]["resolved_rank"] for group_id in ENTRY_GROUPS},
            "pca": pca,
            "trainable": {
                "phase": config.trainable_phase,
                "mixing": config.trainable_mixing,
                "bias": config.trainable_bias,
                "electronic_backend": config.trainable_electronic_backend,
                "pca_target_weight": False,
                "pca_mean_and_basis": False,
            },
            "global_exposure": global_exposure,
            "active_route_count": self.exposure_ledger.active_route_count,
            "exposure_count": self.exposure_ledger.exposure_count,
            "physical_fit_status": (
                "unfitted_synthetic_initialization"
                if config.mode == "physical"
                else "not_applicable"
            ),
            "synthetic_physical_assumptions": config.physical_config.as_dict(),
            "config": config.as_dict(),
            "identity_schema": _IDENTITY_SCHEMA,
            "identity_version": _IDENTITY_VERSION,
            "identity_snapshot_sha256": self._creation_identity_digest,
            "identity_snapshot": copy.deepcopy(self._creation_identity_snapshot),
        }
        return _json_safe(metadata)

    def as_dict(self) -> dict[str, Any]:
        return self.identity_metadata()


__all__ = [
    "ENTRY_GROUPS",
    "OptoelectronicH2Former",
    "OptoelectronicH2FormerConfig",
    "OptoelectronicPCAConv2d",
]
