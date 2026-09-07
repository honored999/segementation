"""Create read-only QC views for one canonicalized DWI and ADN checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from standalone_nnunet2d.brain_alignment.adn_transform import left_right_flip
from standalone_nnunet2d.brain_alignment.diagnostic import (
    ensure_diagnostic_output_dir,
    load_diagnostic_checkpoint,
    normalize_volume,
)
from standalone_nnunet2d.brain_alignment.nifti_adapter import canonicalize_nifti
from standalone_nnunet2d.data.nifti_io import read_nifti


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--case-id")
    return parser


def _case_id(path: Path) -> str:
    name = path.name
    suffix = "_0000.nii.gz"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def _slice_indices(depth: int) -> list[dict[str, int]]:
    return [
        {"percent": percent, "index": int(round((depth - 1) * percent / 100.0))}
        for percent in (25, 50, 75)
    ]


def _save_panel(path: Path, arrays: list[np.ndarray], titles: list[str]) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    for axis, array, title in zip(axes, arrays, titles, strict=True):
        axis.imshow(array, cmap="gray")
        axis.set_title(title)
        axis.axis("off")
    figure.savefig(path, dpi=120)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    image_path = arguments.image.resolve()
    checkpoint_path = arguments.checkpoint.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"NIfTI image does not exist: {image_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    output = ensure_diagnostic_output_dir(arguments.output_dir)
    device = torch.device(arguments.device)
    canonical = canonicalize_nifti(read_nifti(image_path))
    original = torch.from_numpy(normalize_volume(canonical.array))[None, None].to(device)
    loaded = load_diagnostic_checkpoint(checkpoint_path, device=device)
    loaded.model.eval()
    with torch.no_grad():
        result = loaded.model(original)
        mirrored = left_right_flip(result.aligned)
        difference = torch.abs(result.aligned - mirrored)
        flip_before = F.l1_loss(original, left_right_flip(original))
        flip_after = F.l1_loss(result.aligned, mirrored)
    original_np = original[0, 0].cpu().numpy()
    aligned_np = result.aligned[0, 0].cpu().numpy()
    mirrored_np = mirrored[0, 0].cpu().numpy()
    difference_np = difference[0, 0].cpu().numpy()
    slices = _slice_indices(original_np.shape[0])
    for item in slices:
        index = item["index"]
        _save_panel(
            output / f"slice_{item['percent']}.png",
            [original_np[index], aligned_np[index], mirrored_np[index], difference_np[index]],
            ["original", "aligned", "mirrored aligned", "abs(aligned - mirrored)"],
        )
    raw_params = result.raw_params[0].detach().cpu().tolist()
    scaled_params = result.scaled_params[0].detach().cpu().tolist()
    summary = {
        "case_id": arguments.case_id or _case_id(image_path),
        "image": str(image_path),
        "checkpoint": str(checkpoint_path),
        "raw_params": raw_params,
        "scaled_params": scaled_params,
        "rz_degree": math.degrees(float(scaled_params[2])),
        "tx": float(scaled_params[3]),
        "flip_loss_before": float(flip_before.cpu()),
        "flip_loss_after": float(flip_after.cpu()),
        "raw_geometry": {
            "spacing_xyz": list(canonical.spacing_xyz),
            "origin_xyz": list(canonical.origin_xyz),
            "direction": list(canonical.direction),
            "array_order": canonical.original_array_order,
        },
        "orientation_canonicalization": canonical.provenance,
        "slice_indices": slices,
        "display_space": "canonical_per_volume_zscore",
        "diagnostic_only": True,
        "input_modified": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
