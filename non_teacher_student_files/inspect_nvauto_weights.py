"""Inspect DeepISLES NVAUTO TorchScript weights for SegResNet compatibility."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


FORWARD_INPUT_SHAPES = (
    (1, 2, 16, 128, 128),
    (1, 2, 32, 96, 96),
    (1, 2, 96, 96, 16),
)


def find_torchscript_files(weight_dir: Path) -> list[Path]:
    """Find NVAUTO TorchScript weights, including the conventional ts subdirectory."""
    return sorted(weight_dir.rglob("*.ts"))


def require_segresnet():
    try:
        from monai.networks.nets import SegResNet
    except ImportError as exc:
        raise ImportError("MONAI is required for model compatibility checks: pip install monai") from exc
    return SegResNet


def inspect_torchscript(path: Path) -> dict[str, torch.Tensor] | None:
    """Load one TorchScript file and print the information needed for inspection."""
    module = torch.jit.load(str(path), map_location="cpu")
    module.eval()
    print(f"script_module_type: {type(module)}")

    try:
        state = module.state_dict()
    except (AttributeError, RuntimeError) as exc:
        print(f"state_dict: unavailable ({exc})")
        state = None
    else:
        print(f"state_dict keys: {len(state)}")
        print("first 30 tensors:")
        for index, (key, value) in enumerate(state.items()):
            if index >= 30:
                break
            shape = tuple(value.shape) if torch.is_tensor(value) else "non-tensor"
            print(f"  {key}: {shape}")

    with torch.inference_mode():
        for shape in FORWARD_INPUT_SHAPES:
            input_tensor = torch.zeros(shape, dtype=torch.float32)
            try:
                output = module(input_tensor)
            except Exception as exc:  # TorchScript errors vary by exported architecture.
                print(f"forward {list(shape)}: failed ({type(exc).__name__}: {exc})")
            else:
                if torch.is_tensor(output):
                    print(f"forward {list(shape)}: output {tuple(output.shape)}")
                elif isinstance(output, (tuple, list)):
                    output_shapes = [tuple(value.shape) for value in output if torch.is_tensor(value)]
                    print(f"forward {list(shape)}: output sequence {output_shapes}")
                else:
                    print(f"forward {list(shape)}: output type {type(output)}")
    return dict(state) if state is not None else None


def report_segresnet_matches(state: dict[str, torch.Tensor] | None, model: torch.nn.Module) -> None:
    """Report only exact tensor matches; this command never loads weights into SegResNet."""
    if state is None:
        print("fine_tune: state_dict unavailable; keep the from-scratch SegResNet baseline route.")
        return

    model_state = model.state_dict()
    matched = [
        key
        for key, value in state.items()
        if key in model_state and torch.is_tensor(value) and tuple(value.shape) == tuple(model_state[key].shape)
    ]
    print(f"SegResNet exact shape matches: {len(matched)}/{len(model_state)}")
    print(f"matched preview: {matched[:20]}")
    if len(matched) < 10 or len(matched) / max(len(model_state), 1) < 0.1:
        print("fine_tune: too few compatible tensors; keep the from-scratch SegResNet baseline route.")
    else:
        print("fine_tune: compatible tensors exist, but inspect the architecture before any transfer experiment.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect NVAUTO TorchScript compatibility.")
    parser.add_argument("--deepisles-root", type=Path, default=Path(r"C:\lijialin\models3d\DeepIsles"))
    parser.add_argument("--out-channels", type=int, choices=(1, 2), default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    SegResNet = require_segresnet()
    weight_dir = args.deepisles_root / "weights" / "NVAUTO"
    paths = find_torchscript_files(weight_dir)
    print(f"Found TorchScript files: {len(paths)}")
    model = SegResNet(spatial_dims=3, in_channels=2, out_channels=args.out_channels, init_filters=16)
    for path in paths:
        print(f"\n[TORCHSCRIPT] {path}")
        try:
            state = inspect_torchscript(path)
        except RuntimeError as exc:
            print(f"load: failed ({type(exc).__name__}: {exc})")
            continue
        report_segresnet_matches(state, model)


if __name__ == "__main__":
    main()
