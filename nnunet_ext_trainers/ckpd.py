"""Bounded Conv2d CKPD helpers for the Dataset501 transfer experiment.

The decomposition follows the equations and ASR rule in CKPD paper v2. The
Conv2d activation statistic and its per-case spatial normalization are explicit
adaptations for nnU-Net's 2D encoder; this module does not implement FSCIL.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
import random
import tempfile
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize
import torch.nn.functional as F


CKPD_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Conv2dDecomposition:
    """An exactly reconstructing frozen base plus trainable low-rank factors."""

    base_weight: torch.Tensor
    A: torch.Tensor
    B: torch.Tensor
    singular_values: torch.Tensor
    asr: float
    damping: float


def _pair(value) -> tuple[int, int]:
    if isinstance(value, tuple):
        return int(value[0]), int(value[1])
    if isinstance(value, list):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def sample_conv2d_patches(
    x: torch.Tensor,
    conv: nn.Conv2d,
    max_positions: int,
    seed: int = 0,
) -> torch.Tensor:
    """Return at most ``max_positions`` input patches per item, without full unfold.

    Samples output-grid positions once and uses those positions for each batch
    item. Flattening order is the same as ``torch.nn.functional.unfold``. Only
    zero padding and ungrouped Conv2d are accepted because those match the
    supported nnU-Net PlainConvUNet encoder.
    """
    if x.ndim != 4:
        raise ValueError("input must have shape [batch, channels, height, width]")
    if conv.groups != 1:
        raise ValueError("grouped Conv2d is not supported by CKPD calibration")
    if conv.padding_mode != "zeros":
        raise ValueError("only zero-padded Conv2d is supported")
    if isinstance(conv.padding, str):
        raise ValueError("string Conv2d padding is not supported")
    if max_positions < 1:
        raise ValueError("max_positions must be positive")

    kh, kw = _pair(conv.kernel_size)
    sh, sw = _pair(conv.stride)
    dh, dw = _pair(conv.dilation)
    ph, pw = _pair(conv.padding)
    height, width = x.shape[-2:]
    out_h = (height + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    out_w = (width + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    if out_h < 1 or out_w < 1:
        raise ValueError("Conv2d produces an empty output grid")

    total = out_h * out_w
    count = min(int(max_positions), total)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)
    positions = torch.randperm(total, generator=generator)[:count].to(x.device)
    output_y = torch.div(positions, out_w, rounding_mode="floor")
    output_x = positions.remainder(out_w)

    padded = F.pad(x, (pw, pw, ph, ph)) if ph or pw else x
    y_indices = output_y[:, None] * sh + torch.arange(kh, device=x.device)[None, :] * dh
    x_indices = output_x[:, None] * sw + torch.arange(kw, device=x.device)[None, :] * dw
    columns = []
    for channel in range(x.shape[1]):
        for ky in range(kh):
            ys = y_indices[:, ky]
            for kx in range(kw):
                xs = x_indices[:, kx]
                columns.append(padded[:, channel, ys, xs])
    # [B, positions, flattened channel/kernel coordinates] then [B*P, D].
    return torch.stack(columns, dim=-1).reshape(x.shape[0] * count, -1)


def decompose_conv2d(
    weight: torch.Tensor,
    covariance: torch.Tensor,
    rank: int,
    damping_relative: float = 0.01,
) -> Conv2dDecomposition:
    """Build CKPD v2 factors from the flattened Conv2d weight and second moment.

    ``covariance`` is the uncentered input-patch second moment. A small diagonal
    damping term is added as an explicit numerical-stability assumption; it is
    not specified by paper v2. The returned ``base_weight`` is defined as
    ``weight - B @ A`` so the initialized effective kernel reconstructs the
    original kernel even for a truncated rank.
    """
    if weight.ndim != 4:
        raise ValueError("weight must have shape [out, in, kernel_height, kernel_width]")
    if not torch.is_floating_point(weight) or not bool(torch.isfinite(weight).all()):
        raise ValueError("weight must be finite floating-point values")
    if not torch.is_floating_point(covariance) or not bool(torch.isfinite(covariance).all()):
        raise ValueError("covariance must contain finite floating-point values")
    flat = weight.detach().reshape(weight.shape[0], -1).to(dtype=torch.float64, device="cpu")
    cov = covariance.detach().to(dtype=torch.float64, device="cpu")
    dimension = flat.shape[1]
    if cov.ndim != 2 or tuple(cov.shape) != (dimension, dimension):
        raise ValueError(f"covariance shape must be {(dimension, dimension)}")
    max_rank = min(flat.shape)
    if not isinstance(rank, int) or rank < 1 or rank > max_rank:
        raise ValueError(f"rank must be in [1, {max_rank}]")
    if not math.isfinite(damping_relative) or damping_relative < 0:
        raise ValueError("damping_relative must be finite and nonnegative")

    cov = (cov + cov.T) * 0.5
    scale = torch.diagonal(cov).mean()
    if not bool(torch.isfinite(scale)) or float(scale) <= torch.finfo(torch.float64).eps:
        raise ValueError("covariance must have positive finite mean diagonal")
    damping = float(scale) * float(damping_relative)
    regularized = cov + torch.eye(dimension, dtype=torch.float64) * damping
    chol, info = torch.linalg.cholesky_ex(regularized, check_errors=False)
    if int(info) != 0:
        raise ValueError("regularized covariance is not positive definite")

    weighted = flat @ regularized
    try:
        left, singular, right_h = torch.linalg.svd(weighted, full_matrices=False)
    except RuntimeError as exc:
        raise ValueError(f"SVD(WC) failed: {exc}") from exc
    smallest = singular[-1]
    if not bool(torch.isfinite(singular).all()) or float(smallest) <= torch.finfo(torch.float64).tiny:
        raise ValueError("SVD(WC) has a non-finite or zero smallest singular value")

    tail_singular = singular[-rank:]
    root = tail_singular.sqrt()
    # A = diag(sqrt(S_tail)) Vh_tail C^-1; solve rather than materializing C^-1.
    right_scaled = right_h[-rank:, :] * root[:, None]
    A64 = torch.cholesky_solve(right_scaled.T, chol, upper=False).T
    B64 = left[:, -rank:] * root[None, :]
    base64 = flat - B64 @ A64
    if not all(bool(torch.isfinite(t).all()) for t in (base64, A64, B64)):
        raise ValueError("CKPD factorization produced non-finite factors")
    asr = float(singular[-rank] / singular[-1])
    return Conv2dDecomposition(
        base_weight=base64.reshape_as(weight).to(dtype=weight.dtype, device=weight.device),
        A=A64.to(dtype=weight.dtype, device=weight.device),
        B=B64.to(dtype=weight.dtype, device=weight.device),
        singular_values=singular.to(dtype=torch.float32),
        asr=asr,
        damping=damping,
    )


def select_smallest_asr(
    singular_values_by_name: Mapping[str, torch.Tensor], rank: int, count: int
) -> list[str]:
    """Select the K smallest paper-v2 ASR values, with name-stable ties."""
    if not isinstance(rank, int) or rank < 1:
        raise ValueError("rank must be positive")
    if not isinstance(count, int) or count < 1 or count > len(singular_values_by_name):
        raise ValueError("count must be between one and the number of eligible layers")
    scored = []
    for name, values in singular_values_by_name.items():
        singular = torch.as_tensor(values, dtype=torch.float64, device="cpu").flatten()
        if rank > singular.numel():
            raise ValueError(f"rank exceeds available singular values for {name}")
        if not bool(torch.isfinite(singular).all()) or float(singular[-1]) <= 0:
            raise ValueError(f"invalid singular values for {name}")
        score = float(singular[-rank] / singular[-1])
        scored.append((score, str(name)))
    return [name for _, name in sorted(scored, key=lambda item: (item[0], item[1]))[:count]]


class CKPDWeightParametrization(nn.Module):
    """Add a trainable low-rank residual to one frozen Conv2d weight."""

    def __init__(self, A: torch.Tensor, B: torch.Tensor):
        super().__init__()
        self.A = nn.Parameter(A.detach().clone())
        self.B = nn.Parameter(B.detach().clone())

    def forward(self, original: torch.Tensor) -> torch.Tensor:
        product_dtype = (
            torch.float64
            if torch.float64 in (original.dtype, self.A.dtype, self.B.dtype)
            else torch.float32
        )
        with torch.autocast(device_type=original.device.type, enabled=False):
            residual = (self.B.to(dtype=product_dtype) @ self.A.to(dtype=product_dtype)).reshape_as(original)
        return original + residual.to(dtype=original.dtype)


def encoder_conv_layers(network: nn.Module) -> dict[str, nn.Conv2d]:
    encoder = getattr(network, "encoder", None)
    if not isinstance(encoder, nn.Module):
        raise ValueError("network does not expose an encoder module")
    return {
        "encoder." + name: module
        for name, module in encoder.named_modules()
        if isinstance(module, nn.Conv2d) and module.groups == 1
    }


def _all_modules_by_name(network: nn.Module) -> dict[str, nn.Module]:
    return dict(network.named_modules(remove_duplicate=False))


def apply_ckpd_adapters(
    network: nn.Module, factors: Mapping[str, Conv2dDecomposition]
) -> None:
    """Install selected adapters and freeze every other encoder parameter.

    Parametrization wraps the Conv2d object itself, preserving all registered
    aliases in ConvDropoutNormReLU and the shared ``decoder.encoder`` reference.
    Decoder parameters are enabled by identity after encoder parameters freeze,
    so the decoder's registered encoder alias cannot accidentally unfreeze it.
    """
    modules = _all_modules_by_name(network)
    encoder = getattr(network, "encoder", None)
    if not isinstance(encoder, nn.Module):
        raise ValueError("network does not expose an encoder module")
    if not factors:
        raise ValueError("at least one CKPD layer is required")
    installed: list[CKPDWeightParametrization] = []
    encoder_layers = encoder_conv_layers(network)
    for name, decomposition in factors.items():
        if name not in encoder_layers or name not in modules:
            raise ValueError(f"selected layer is not an ungrouped encoder Conv2d: {name}")
        conv = modules[name]
        if parametrize.is_parametrized(conv, "weight"):
            raise ValueError(f"Conv2d is already parametrized: {name}")
        if tuple(decomposition.base_weight.shape) != tuple(conv.weight.shape):
            raise ValueError(f"base weight shape mismatch for {name}")
        if tuple(decomposition.A.shape[1:]) != (conv.weight.numel() // conv.out_channels,):
            raise ValueError(f"A factor shape mismatch for {name}")
        if decomposition.B.shape != (conv.out_channels, decomposition.A.shape[0]):
            raise ValueError(f"B factor shape mismatch for {name}")
        with torch.no_grad():
            conv.weight.copy_(decomposition.base_weight.to(device=conv.weight.device, dtype=conv.weight.dtype))
        parametrization = CKPDWeightParametrization(
            decomposition.A.to(device=conv.weight.device, dtype=conv.weight.dtype),
            decomposition.B.to(device=conv.weight.device, dtype=conv.weight.dtype),
        )
        parametrize.register_parametrization(conv, "weight", parametrization)
        installed.append(parametrization)

    encoder_ids = {id(parameter) for parameter in encoder.parameters()}
    for parameter in network.parameters():
        parameter.requires_grad_(id(parameter) not in encoder_ids)
    for parametrization in installed:
        parametrization.A.requires_grad_(True)
        parametrization.B.requires_grad_(True)
    encoder.eval()


def merged_state_dict(network: nn.Module) -> dict[str, torch.Tensor]:
    """Export ordinary baseline-network keys and merged weights for inference."""
    original_state = network.state_dict()
    merged = {
        key: value.detach().clone()
        for key, value in original_state.items()
        if ".parametrizations.weight." not in key
    }
    seen = set()
    aliases = network.named_modules(remove_duplicate=False)
    for prefix, module in aliases:
        if not isinstance(module, nn.Conv2d) or id(module) in seen:
            # Revisit once below to assign all aliases of this module.
            continue
        if not parametrize.is_parametrized(module, "weight"):
            continue
        if not isinstance(module.parametrizations.weight[0], CKPDWeightParametrization):
            continue
        seen.add(id(module))
        weight = module.weight.detach().clone()
        for alias_prefix, alias_module in network.named_modules(remove_duplicate=False):
            if alias_module is module:
                merged[f"{alias_prefix}.weight" if alias_prefix else "weight"] = weight.clone()
    if not merged:
        raise ValueError("network state is empty")
    return merged


_SKIP_SEMANTIC_VALUE = object()


def _semantic_value(value):
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if callable(value):
        return f"{value.__module__}.{getattr(value, '__qualname__', type(value).__qualname__)}"
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            converted = _semantic_value(item)
            if converted is _SKIP_SEMANTIC_VALUE:
                return _SKIP_SEMANTIC_VALUE
            normalized[str(key)] = converted
        return normalized
    if isinstance(value, (tuple, list)):
        converted = [_semantic_value(item) for item in value]
        return _SKIP_SEMANTIC_VALUE if any(item is _SKIP_SEMANTIC_VALUE for item in converted) else converted
    return _SKIP_SEMANTIC_VALUE


def encoder_semantic_signature(network: nn.Module) -> str:
    """Hash resolved encoder operations, their order, and forward properties."""
    encoder = getattr(network, "encoder", None)
    if not isinstance(encoder, nn.Module):
        raise ValueError("network does not expose an encoder module")
    modules = []
    for path, module in encoder.named_modules(remove_duplicate=False):
        attributes = {}
        for name, value in vars(module).items():
            if name.startswith("_") or name == "training":
                continue
            converted = _semantic_value(value)
            if converted is not _SKIP_SEMANTIC_VALUE:
                attributes[name] = converted
        children = [
            [name, None if child is None else f"{type(child).__module__}.{type(child).__qualname__}"]
            for name, child in module._modules.items()
        ]
        modules.append({
            "path": path,
            "type": f"{type(module).__module__}.{type(module).__qualname__}",
            "attributes": attributes,
            "children_in_order": children,
        })
    encoded = json.dumps(modules, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def module_fingerprint(module: nn.Module) -> str:
    """Hash module tensor names, dtypes, shapes, and exact values."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def hash_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_identifier(identifier: str) -> str:
    return hashlib.sha256(str(identifier).encode("utf-8")).hexdigest()


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": [state.cpu() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() and torch.cuda.is_initialized()
        else None,
    }


def validate_rng_state(state: Mapping) -> None:
    """Validate saved RNG payloads with isolated generators and no global mutation."""
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if not isinstance(state, Mapping) or not required.issubset(state):
        raise ValueError("checkpoint RNG state is incomplete")
    try:
        random.Random().setstate(state["python"])
    except Exception as exc:
        raise ValueError("checkpoint Python RNG state is invalid") from exc
    try:
        numpy_rng = np.random.RandomState()
        numpy_rng.set_state(state["numpy"])
    except Exception as exc:
        raise ValueError("checkpoint NumPy RNG state is invalid") from exc

    cpu_state = state["torch_cpu"]
    if not isinstance(cpu_state, torch.Tensor) or cpu_state.dtype != torch.uint8 or cpu_state.ndim != 1:
        raise ValueError("checkpoint CPU Torch RNG state is invalid")
    try:
        torch.Generator(device="cpu").set_state(cpu_state.detach().cpu())
    except Exception as exc:
        raise ValueError("checkpoint CPU Torch RNG state is invalid") from exc

    cuda_states = state["torch_cuda"]
    if cuda_states is not None:
        if (not isinstance(cuda_states, (tuple, list)) or not torch.cuda.is_available()
                or len(cuda_states) != torch.cuda.device_count()):
            raise ValueError("saved CUDA RNG state does not match the available CUDA devices")
        for index, cuda_state in enumerate(cuda_states):
            if not isinstance(cuda_state, torch.Tensor) or cuda_state.dtype != torch.uint8 or cuda_state.ndim != 1:
                raise ValueError(f"checkpoint CUDA RNG state {index} is invalid")
            try:
                generator = torch.Generator(device=torch.device(f"cuda:{index}"))
                generator.set_state(cuda_state.detach().cpu())
            except Exception as exc:
                raise ValueError(f"checkpoint CUDA RNG state {index} is invalid") from exc


def restore_rng_state(state: Mapping) -> None:
    validate_rng_state(state)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].detach().cpu())
    cuda_states = state["torch_cuda"]
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)


def atomic_torch_save(payload: object, path: str | os.PathLike[str]) -> None:
    """Write a checkpoint beside its destination, then atomically replace it."""
    destination = os.path.abspath(os.fspath(path))
    directory = os.path.dirname(destination)
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".ckpd-", suffix=".tmp", dir=directory)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
