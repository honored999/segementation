"""Scan fixed ADN transform candidates without loading or training a model."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from standalone_nnunet2d.brain_alignment.adn_transform import (
    alignment_losses,
    build_transform_matrices,
    warp_volume,
)
from standalone_nnunet2d.brain_alignment.diagnostic import (
    ensure_diagnostic_output_dir,
    normalize_volume,
    pad_model_input_depth,
)
from standalone_nnunet2d.brain_alignment.nifti_adapter import canonicalize_nifti
from standalone_nnunet2d.data.nifti_io import read_nifti


RZ_DEGREES = (-15, -10, -5, 0, 5, 10, 15)
TX_PIXELS = (-32, -16, 0, 16, 32)
CSV_FIELDS = (
    "case_id",
    "rz_degree",
    "tx_pixels",
    "tx_normalized",
    "flip_loss",
    "reconstruction_loss",
    "total_loss",
)
_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class GridPoint:
    rz_degree: int
    tx_pixels: int
    tx_normalized: float


def grid_points(width: int) -> tuple[GridPoint, ...]:
    if width < 2:
        raise ValueError("model-input width must be at least 2")
    return tuple(
        GridPoint(rz, tx, 2.0 * tx / width)
        for rz in RZ_DEGREES
        for tx in TX_PIXELS
    )


def summarize_landscape(case_id: str, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    identity = next(
        (row for row in rows if row["rz_degree"] == 0 and row["tx_pixels"] == 0),
        None,
    )
    if identity is None:
        raise ValueError("landscape does not contain the identity grid point")
    if not rows:
        raise ValueError("landscape is empty")
    best = min(rows, key=lambda row: float(row["total_loss"]))
    identity_total = float(identity["total_loss"])
    best_total = float(best["total_loss"])
    ratio = None if identity_total == 0.0 else best_total / identity_total
    improvement = None if identity_total == 0.0 else 100.0 * (identity_total - best_total) / identity_total
    return {
        "case_id": case_id,
        "identity": {
            "flip_loss": float(identity["flip_loss"]),
            "reconstruction_loss": float(identity["reconstruction_loss"]),
            "total_loss": identity_total,
        },
        "global_minimum": {
            "rz_degree": int(best["rz_degree"]),
            "tx_pixels": int(best["tx_pixels"]),
            "tx_normalized": float(best["tx_normalized"]),
            "flip_loss": float(best["flip_loss"]),
            "reconstruction_loss": float(best["reconstruction_loss"]),
            "total_loss": best_total,
        },
        "best_to_identity_ratio": ratio,
        "improvement_percent_relative_to_identity": improvement,
        "minimum_on_grid_boundary": (
            int(best["rz_degree"]) in (RZ_DEGREES[0], RZ_DEGREES[-1])
            or int(best["tx_pixels"]) in (TX_PIXELS[0], TX_PIXELS[-1])
        ),
        "grid_shape": {"rz": len(RZ_DEGREES), "tx": len(TX_PIXELS), "points": len(rows)},
        "tx_semantics": "normalized output-to-input sampling-matrix coordinate; sign is not interpreted physically",
    }


def _save_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _save_heatmap(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    values = np.asarray([float(row["total_loss"]) for row in rows], dtype=np.float64).reshape(
        len(RZ_DEGREES), len(TX_PIXELS)
    )
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    image = axis.imshow(values, origin="lower", aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(TX_PIXELS)), labels=TX_PIXELS)
    axis.set_yticks(range(len(RZ_DEGREES)), labels=RZ_DEGREES)
    axis.set_xlabel("tx candidate (pixels; output-to-input sampling parameter)")
    axis.set_ylabel("rz (degrees)")
    axis.set_title("ADN total alignment loss")
    figure.colorbar(image, ax=axis, label="total loss")
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _scan_case(case_id: str, image_path: Path, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    canonical = canonicalize_nifti(read_nifti(image_path))
    original = torch.from_numpy(normalize_volume(canonical.array))[None, None].to(device)
    model_input, padding = pad_model_input_depth(original)
    spatial_shape = tuple(int(size) for size in model_input.shape[-3:])
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for point in grid_points(spatial_shape[-1]):
            params = model_input.new_tensor(
                [[0.0, 0.0, math.radians(point.rz_degree), point.tx_normalized, 0.0, 0.0]]
            )
            matrix, inverse = build_transform_matrices(params, spatial_shape=spatial_shape)
            aligned = warp_volume(model_input, matrix)
            losses = alignment_losses(model_input, aligned, inverse)
            row = {
                "case_id": case_id,
                "rz_degree": point.rz_degree,
                "tx_pixels": point.tx_pixels,
                "tx_normalized": point.tx_normalized,
                "flip_loss": float(losses.flip_loss.detach().cpu()),
                "reconstruction_loss": float(losses.reconstruction_loss.detach().cpu()),
                "total_loss": float(losses.total_loss.detach().cpu()),
            }
            if not all(math.isfinite(float(row[field])) for field in CSV_FIELDS[3:]):
                raise ValueError(f"non-finite loss-landscape value for {case_id}")
            rows.append(row)
    return rows, padding


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    dataset = arguments.dataset_dir.resolve()
    if len(set(arguments.cases)) != len(arguments.cases):
        raise ValueError("case ids must be unique")
    if any(_CASE_ID.fullmatch(case_id) is None for case_id in arguments.cases):
        raise ValueError("case ids may contain only letters, digits, underscores, and hyphens")
    image_paths = [(dataset / "imagesTr" / f"{case_id}_0000.nii.gz").resolve() for case_id in arguments.cases]
    for image_path in image_paths:
        if not image_path.is_file():
            raise FileNotFoundError(f"NIfTI image does not exist: {image_path}")
    output = ensure_diagnostic_output_dir(arguments.output_dir, dataset_dir=dataset)
    device = torch.device(arguments.device)
    for case_id, image_path in zip(arguments.cases, image_paths, strict=True):
        rows, padding = _scan_case(case_id, image_path, device)
        case_output = output / case_id
        case_output.mkdir()
        _save_csv(case_output / "landscape.csv", rows)
        summary = summarize_landscape(case_id, rows)
        summary["image"] = str(image_path)
        summary["device"] = str(device)
        summary["model_input_depth_padding"] = padding
        (case_output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        _save_heatmap(case_output / "total_loss_heatmap.png", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
