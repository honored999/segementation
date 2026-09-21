"""Report synthetic fixed-input parameter and profiler-supported FLOP counts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile

from standalone_nnunet2d.models.factory import (
    H2FORMER,
    H2FORMER_LITE_UPERNET,
    PLAIN_CONV_UNET,
    PLAIN_CONV_UNET_LITE_UPERNET,
    SINGLE_OUTPUT,
    build_model,
)


PROFILE_MODELS = (
    H2FORMER,
    H2FORMER_LITE_UPERNET,
    PLAIN_CONV_UNET,
    PLAIN_CONV_UNET_LITE_UPERNET,
)


def _parameter_count(modules: Sequence[nn.Module]) -> int:
    return sum(parameter.numel() for module in modules for parameter in module.parameters())


def _decoder_parameter_count(model_name: str, model: nn.Module) -> int:
    if model_name == H2FORMER:
        return _parameter_count((model.decode4, model.decode3, model.decode2, model.decode0))
    if model_name == PLAIN_CONV_UNET:
        return _parameter_count(
            (model.transposed_convolutions, model.decoder_stages, model.segmentation_heads)
        )
    if model_name in {H2FORMER_LITE_UPERNET, PLAIN_CONV_UNET_LITE_UPERNET}:
        return _parameter_count((model.decoder,))
    raise ValueError(f"unsupported profile model {model_name!r}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile one standalone 2D decoder variant")
    parser.add_argument("--model", required=True, choices=PROFILE_MODELS)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    device = torch.device(arguments.device)
    supervision_mode = SINGLE_OUTPUT if arguments.model == PLAIN_CONV_UNET else None
    model = build_model(
        arguments.model,
        supervision_mode=supervision_mode,
        inference=True,
    ).to(device).eval()
    image = torch.zeros((1, 1, 512, 512), device=device)
    with torch.inference_mode():
        model(image)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with torch.inference_mode(), profile(activities=activities, with_flops=True) as profiler:
        output = model(image)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = {
        "model": arguments.model,
        "supervision_mode": SINGLE_OUTPUT,
        "device": str(device),
        "input_shape": list(image.shape),
        "output_shape": list(output.shape),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "decoder_parameters": _decoder_parameter_count(arguments.model, model),
        "profiler_flops": sum(int(event.flops or 0) for event in profiler.key_averages()),
        "flop_scope": "torch.profiler supported operators only",
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
