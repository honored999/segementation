"""Create in-memory quasi-symmetric alignment QC panels for one NIfTI image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as pyplot
import numpy as np

from standalone_nnunet2d.data.nifti_io import NiftiVolume, read_nifti
from standalone_nnunet2d.data.preprocessing import resample_inplane
from standalone_nnunet2d.data.symmetry_alignment import (
    AlignmentResult,
    align_case,
    align_case_result,
    build_alignment_qc,
)


DEFAULT_TARGET_SPACING_XY = (0.4892368018627167, 0.4892368018627167)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without reading or writing any data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--label", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target-spacing-xy",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=DEFAULT_TARGET_SPACING_XY,
    )
    return parser


def choose_slice_indices(depth: int) -> tuple[int, int, int]:
    """Choose deterministic axial slices nearest to 25, 50, and 75 percent."""
    if depth < 3:
        raise ValueError("need three distinct axial slice indices, but depth is smaller than 3")
    indices = tuple(int(round((depth - 1) * fraction)) for fraction in (0.25, 0.5, 0.75))
    if len(set(indices)) != 3:
        raise ValueError("need three distinct axial slice indices at 25%, 50%, and 75%")
    return indices


def _geometry(volume: NiftiVolume) -> dict[str, list[int] | list[float]]:
    return {
        "shape_zyx": [int(value) for value in volume.array.shape],
        "spacing_xyz": [float(value) for value in volume.spacing_xyz],
        "origin_xyz": [float(value) for value in volume.origin_xyz],
        "direction": [float(value) for value in volume.direction],
    }


def _same_geometry(image: NiftiVolume, label: NiftiVolume) -> bool:
    return (
        image.array.shape == label.array.shape
        and np.allclose(image.spacing_xyz, label.spacing_xyz, rtol=0.0, atol=1e-6)
        and np.allclose(image.origin_xyz, label.origin_xyz, rtol=0.0, atol=1e-6)
        and np.allclose(image.direction, label.direction, rtol=0.0, atol=1e-6)
    )


def _add_contour(axis: object, label: NiftiVolume | None, slice_index: int) -> None:
    if label is None:
        return
    mask = np.asarray(label.array[slice_index]) != 0
    if np.any(mask):
        axis.contour(mask, levels=(0.5,), colors=("lime",), linewidths=0.75)


def _write_panel(
    path: Path,
    *,
    original: NiftiVolume,
    result: AlignmentResult,
    original_label: NiftiVolume | None,
    slice_index: int,
) -> None:
    qc = build_alignment_qc(original, result=result, slice_index=slice_index)
    figure, axes = pyplot.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    panels = (
        ("original", qc.original),
        ("aligned", qc.aligned),
        ("mirrored", qc.mirrored),
        ("abs(aligned-mirrored)", qc.absolute_difference),
    )
    for axis, (title, panel) in zip(axes, panels):
        axis.imshow(panel, cmap="gray")
        axis.set_title(title)
        axis.set_axis_off()
    _add_contour(axes[0], original_label, slice_index)
    _add_contour(axes[1], result.label, slice_index)
    figure.savefig(path, dpi=150)
    pyplot.close(figure)


def _success_summary(
    *,
    original_image: NiftiVolume,
    resampled_image: NiftiVolume,
    result: AlignmentResult,
    slice_indices: tuple[int, int, int],
    original_label: NiftiVolume | None,
    resampled_label: NiftiVolume | None,
) -> dict[str, object]:
    estimate = result.estimate
    center = [float(value) for value in estimate.center_xyz]
    reference_center = [float(value) for value in estimate.reference_center_xyz]
    summary: dict[str, object] = {
        "status": "aligned",
        "original_geometry": _geometry(original_image),
        "resampled_geometry": _geometry(resampled_image),
        "aligned_geometry": _geometry(result.image),
        "orientation_handling": "supported axis-aligned axial direction preserved during in-memory alignment",
        "center_xyz": center,
        "reference_center_xyz": reference_center,
        "center_to_reference_xyz": [
            float(reference - centre) for centre, reference in zip(center, reference_center)
        ],
        "output_to_input_translation_xyz": [
            float(value) for value in estimate.output_to_input_translation_xyz
        ],
        "rotation_angle_radians": float(estimate.rotation_angle_radians),
        "foreground_voxel_count": int(estimate.foreground_voxel_count),
        "selected_slice_indices": [int(value) for value in slice_indices],
        "chosen_slice_indices": [int(value) for value in slice_indices],
    }
    if original_label is not None and resampled_label is not None and result.label is not None:
        summary["label_geometry"] = {
            "original": _geometry(original_label),
            "resampled": _geometry(resampled_label),
            "aligned": _geometry(result.label),
        }
    return summary


def _write_summary(output_dir: Path, summary: dict[str, object]) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run one image-only alignment estimate and write compact QC evidence."""
    arguments = build_parser().parse_args(argv)
    output_dir = arguments.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir()

    try:
        original_image = read_nifti(arguments.image)
        original_label = read_nifti(arguments.label) if arguments.label is not None else None
        if original_label is not None and not _same_geometry(original_image, original_label):
            raise ValueError("image and label geometry mismatch before resampling")

        target_spacing_xy = tuple(float(value) for value in arguments.target_spacing_xy)
        resampled_image = resample_inplane(
            original_image, target_spacing_xy, is_segmentation=False
        )
        resampled_label = (
            resample_inplane(original_label, target_spacing_xy, is_segmentation=True)
            if original_label is not None
            else None
        )
        if resampled_label is None:
            result = align_case_result(resampled_image)
        else:
            aligned_image, aligned_label, estimate = align_case(resampled_image, resampled_label)
            result = AlignmentResult(image=aligned_image, label=aligned_label, estimate=estimate)

        slice_indices = choose_slice_indices(resampled_image.array.shape[0])
        for order, slice_index in enumerate(slice_indices):
            _write_panel(
                output_dir / f"slice_{order:02d}_z{slice_index:03d}.png",
                original=resampled_image,
                result=result,
                original_label=resampled_label,
                slice_index=slice_index,
            )
        _write_summary(
            output_dir,
            _success_summary(
                original_image=original_image,
                resampled_image=resampled_image,
                result=result,
                slice_indices=slice_indices,
                original_label=original_label,
                resampled_label=resampled_label,
            ),
        )
        return 0
    except Exception as error:
        _write_summary(output_dir, {"status": "failed", "error": str(error)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
