"""Typed configuration read directly from the supplied nnU-Net plans JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "reference"


@dataclass(frozen=True)
class ModelConfig:
    """Architecture values required by :class:`PlainConvUNet2D`."""

    input_channels: int
    output_channels: int
    n_stages: int
    features_per_stage: tuple[int, ...]
    kernel_sizes: tuple[tuple[int, int], ...]
    strides: tuple[tuple[int, int], ...]
    n_conv_per_stage: tuple[int, ...]
    n_conv_per_stage_decoder: tuple[int, ...]
    conv_bias: bool
    norm_eps: float
    norm_affine: bool
    leaky_relu_negative_slope: float
    leaky_relu_inplace: bool
    batch_dice: bool


def _as_pairs(values: list[list[int]]) -> tuple[tuple[int, int], ...]:
    return tuple(tuple(int(axis) for axis in value) for value in values)  # type: ignore[return-value]


def load_model_config(reference_dir: Path | None = None) -> ModelConfig:
    """Parse the 2D architecture without importing nnunetv2 at runtime."""
    directory = reference_dir or DEFAULT_REFERENCE_DIR
    plans_path = directory / "nnUNetPlans.json"
    with plans_path.open(encoding="utf-8") as handle:
        plans: dict[str, Any] = json.load(handle)

    configuration = plans["configurations"]["2d"]
    architecture = configuration["architecture"]["arch_kwargs"]
    if architecture["n_stages"] != len(architecture["features_per_stage"]):
        raise ValueError("2D plans have inconsistent stage and feature counts")
    if architecture["dropout_op"] is not None:
        raise ValueError("this reproduction intentionally supports plans without dropout only")

    # PyTorch documents nn.LeakyReLU's default negative_slope as 0.01. The
    # supplied plans omit it, so it is explicit here rather than guessed.
    return ModelConfig(
        input_channels=1,
        output_channels=2,
        n_stages=int(architecture["n_stages"]),
        features_per_stage=tuple(int(value) for value in architecture["features_per_stage"]),
        kernel_sizes=_as_pairs(architecture["kernel_sizes"]),
        strides=_as_pairs(architecture["strides"]),
        n_conv_per_stage=tuple(int(value) for value in architecture["n_conv_per_stage"]),
        n_conv_per_stage_decoder=tuple(int(value) for value in architecture["n_conv_per_stage_decoder"]),
        conv_bias=bool(architecture["conv_bias"]),
        norm_eps=float(architecture["norm_op_kwargs"]["eps"]),
        norm_affine=bool(architecture["norm_op_kwargs"]["affine"]),
        leaky_relu_negative_slope=0.01,
        leaky_relu_inplace=bool(architecture["nonlin_kwargs"]["inplace"]),
        batch_dice=bool(configuration["batch_dice"]),
    )
