"""Convert official nnU-Net network weights into a standalone checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from standalone_nnunet2d.config import load_model_config
from standalone_nnunet2d.models import PlainConvUNet2D


EXPECTED_TENSOR_COUNT = 148
OFFICIAL_ALIGNMENT_PENDING = "official_alignment_pending"


def _numpy_weights_only_safe_globals() -> tuple[object, ...]:
    """Return only NumPy globals needed for nnU-Net scalar metadata."""
    numpy_core = getattr(np, "_core", None)
    if numpy_core is None:
        numpy_core = np.core
    return (
        numpy_core.multiarray.scalar,
        np.dtype,
        type(np.dtype(np.float64)),
        type(np.dtype(np.float32)),
    )

_ENCODER_TARGET = re.compile(
    r"^encoder_stages\.(?P<stage>\d+)\.blocks\.(?P<block>\d+)\."
    r"(?P<position>[01])\.(?P<param>weight|bias|param)$"
)
_DECODER_TARGET = re.compile(
    r"^decoder_stages\.(?P<stage>\d+)\.blocks\.(?P<block>\d+)\."
    r"(?P<position>[01])\.(?P<param>weight|bias|param)$"
)
_TRANSPOSED_TARGET = re.compile(
    r"^transposed_convolutions\.(?P<stage>\d+)\.(?P<param>weight|bias|param)$"
)
_HEAD_TARGET = re.compile(r"^segmentation_heads\.(?P<stage>\d+)\.(?P<param>weight|bias|param)$")


def target_to_source_key(target_key: str) -> str:
    """Map one standalone target key to its exact official semantic key."""
    match = _ENCODER_TARGET.fullmatch(target_key)
    if match is not None:
        module_name = "conv" if match["position"] == "0" else "norm"
        return (
            f"encoder.stages.{match['stage']}.0.convs.{match['block']}."
            f"{module_name}.{match['param']}"
        )

    match = _DECODER_TARGET.fullmatch(target_key)
    if match is not None:
        module_name = "conv" if match["position"] == "0" else "norm"
        return f"decoder.stages.{match['stage']}.convs.{match['block']}.{module_name}.{match['param']}"

    match = _TRANSPOSED_TARGET.fullmatch(target_key)
    if match is not None:
        return f"decoder.transpconvs.{match['stage']}.{match['param']}"

    match = _HEAD_TARGET.fullmatch(target_key)
    if match is not None:
        return f"decoder.seg_layers.{match['stage']}.{match['param']}"

    raise ValueError(f"unsupported target key: {target_key}")


def get_target_state_dict() -> Mapping[str, Tensor]:
    """Return the state-dict contract of the non-deep-supervised target model."""
    model = PlainConvUNet2D(load_model_config(), deep_supervision=False)
    return model.state_dict()


def map_official_weights(
    network_weights: Mapping[str, Tensor],
    target_state_dict: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Map official tensors by semantic names, with exact shape validation."""
    if not isinstance(network_weights, Mapping):
        raise TypeError("official network_weights must be a mapping")

    mapped: dict[str, Tensor] = {}
    source_keys: set[str] = set()
    for target_key, target_value in target_state_dict.items():
        source_key = target_to_source_key(target_key)
        if source_key in source_keys:
            raise ValueError(
                "target semantic mappings must reference unique source keys: "
                f"{source_key}"
            )
        source_keys.add(source_key)
        if source_key not in network_weights:
            raise KeyError(f"missing source key: {source_key}")

        source_value = network_weights[source_key]
        if not isinstance(source_value, Tensor):
            raise TypeError(f"source value is not a tensor: {source_key}")
        if not isinstance(target_value, Tensor):
            raise TypeError(f"target contract value is not a tensor: {target_key}")
        if tuple(source_value.shape) != tuple(target_value.shape):
            raise ValueError(
                f"shape mismatch for {target_key} <- {source_key}: "
                f"source {tuple(source_value.shape)} != target {tuple(target_value.shape)}"
            )

        mapped[target_key] = source_value.detach().cpu().clone()

    if set(mapped) != set(target_state_dict):
        raise ValueError("mapped state_dict does not cover every target key exactly once")
    return mapped


def validate_mapped_state_dict(mapped_state_dict: Mapping[str, Tensor]) -> None:
    """Strict-load mapped weights into the same target model contract."""
    model = PlainConvUNet2D(load_model_config(), deep_supervision=False)
    model.load_state_dict(mapped_state_dict, strict=True)


def _read_network_weights(path: Path) -> Mapping[str, Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"official checkpoint does not exist: {path}")

    with torch.serialization.safe_globals(_numpy_weights_only_safe_globals()):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("official checkpoint must be a dict")
    if "network_weights" not in checkpoint:
        raise ValueError("official checkpoint is missing network_weights")

    network_weights = checkpoint["network_weights"]
    if not isinstance(network_weights, Mapping):
        raise ValueError("official network_weights must be a mapping")
    return network_weights


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_checkpoint(official_checkpoint: Path, output: Path, *, fold: int = 0) -> dict[str, Any]:
    """Convert one official checkpoint and write only the pending checkpoint payload."""
    official_path = Path(official_checkpoint).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if official_path == output_path:
        raise ValueError("official checkpoint and output must not resolve to the same path")
    network_weights = _read_network_weights(official_path)

    target_state_dict = get_target_state_dict()
    if len(target_state_dict) != EXPECTED_TENSOR_COUNT:
        raise ValueError(
            f"target state_dict must contain exactly {EXPECTED_TENSOR_COUNT} tensors, "
            f"got {len(target_state_dict)}"
        )

    mapped_state_dict = map_official_weights(network_weights, target_state_dict)
    validate_mapped_state_dict(mapped_state_dict)

    payload = {
        "format_version": 1,
        "model_state_dict": mapped_state_dict,
        "metadata": {
            "run_type": OFFICIAL_ALIGNMENT_PENDING,
            "run_state": OFFICIAL_ALIGNMENT_PENDING,
            "fold": int(fold),
            "source_checkpoint": str(official_path),
            "source_sha256": _sha256(official_path),
            "mapping_policy": "semantic_name_v1",
            "source_format": "nnunetv2_network_weights",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)

    return {
        "output": str(output_path),
        "mapped_count": len(mapped_state_dict),
        "run_state": OFFICIAL_ALIGNMENT_PENDING,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert official nnU-Net weights to standalone format")
    parser.add_argument("--official-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fold", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = convert_checkpoint(arguments.official_checkpoint, arguments.output, fold=arguments.fold)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
