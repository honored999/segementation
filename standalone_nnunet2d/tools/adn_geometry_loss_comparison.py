"""Compare a simple image-geometry correction with the existing ADN loss.

This is an image-only, diagnostic-only CLI.  It does not train a model or
construct a lesion mask.  The measured ellipse pose is reported separately
from the applied correction: the correction aligns the measured undirected
axis to the canonical H/y vertical axis (±90°), and its inverse is the
sampling transform passed to the existing ``warp_volume`` implementation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage
import torch
from torch import Tensor

from standalone_nnunet2d.brain_alignment.adn_transform import alignment_losses, warp_volume
from standalone_nnunet2d.brain_alignment.diagnostic import (
    ensure_diagnostic_output_dir,
    load_diagnostic_checkpoint,
    normalize_volume,
    pad_model_input_depth,
    unpad_model_input_depth,
)
from standalone_nnunet2d.brain_alignment.nifti_adapter import canonicalize_nifti
from standalone_nnunet2d.data.nifti_io import read_nifti


GEOMETRY_SLICE_PERCENTS = (25, 50, 75)
CANONICAL_VERTICAL_AXIS_DEGREE = 90.0
_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

CSV_FIELDS = (
    "case_id",
    "state",
    "status",
    "available",
    "estimated_content_rotation_degree",
    "estimated_content_lr_translation_pixels",
    "applied_content_rotation_degree",
    "applied_content_lr_translation_pixels",
    "applied_content_transform",
    "sampling_matrix",
    "inverse_sampling_matrix",
    "flip_loss",
    "reconstruction_loss",
    "total_loss",
    "centroid_x_before",
    "centroid_y_before",
    "centroid_x_after",
    "centroid_y_after",
    "principal_axis_angle_before_degree",
    "principal_axis_angle_after_degree",
    "absolute_centroid_offset_before_pixels",
    "absolute_centroid_offset_after_pixels",
    "absolute_principal_axis_tilt_before_degree",
    "absolute_principal_axis_tilt_after_degree",
)


@dataclass(frozen=True)
class SliceIndex:
    """One acquisition-depth slice selected for the geometry estimate."""

    percent: int
    index: int


@dataclass(frozen=True)
class SliceGeometry:
    """Binary-component equivalent-ellipse measurements for one slice."""

    percent: int
    index: int
    foreground_pixel_count: int
    foreground_threshold: float
    centroid_x: float
    centroid_y: float
    lr_centroid_offset_pixels: float
    principal_axis_angle_degree: float
    equivalent_major_radius_pixels: float
    equivalent_minor_radius_pixels: float


@dataclass(frozen=True)
class GeometryEstimate:
    """Median geometry measurements across the three requested slices."""

    slices: tuple[SliceGeometry, ...]
    estimated_content_rotation_degree: float
    estimated_content_lr_translation_pixels: float
    estimated_centroid_x: float
    estimated_centroid_y: float


@dataclass(frozen=True)
class StateComparison:
    """One state in the identity/ADN/geometry comparison."""

    state: str
    status: str
    available: bool
    aligned: Tensor | None
    forward_content_transform: Tensor | None
    sampling_matrix: Tensor | None
    inverse_sampling_matrix: Tensor | None
    flip_loss: float | None
    reconstruction_loss: float | None
    total_loss: float | None
    geometry: GeometryEstimate | None
    raw_params: tuple[float, ...] | None = None
    scaled_params: tuple[float, ...] | None = None


def slice_indices(depth: int) -> tuple[SliceIndex, ...]:
    """Select 25/50/75 percent acquisition-plane indices deterministically."""
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1:
        raise ValueError(f"depth must be a positive integer, got {depth!r}")
    return tuple(
        SliceIndex(percent=percent, index=int(round((depth - 1) * percent / 100.0)))
        for percent in GEOMETRY_SLICE_PERCENTS
    )


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Return the first largest 8-connected component in row-major order."""
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("connected-component input must be a 2-D boolean array")
    visited = np.zeros(mask.shape, dtype=bool)
    best_coordinates: list[tuple[int, int]] = []
    height, width = mask.shape
    for start_y, start_x in np.argwhere(mask):
        start = (int(start_y), int(start_x))
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        coordinates: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            coordinates.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    neighbor_y = y + dy
                    neighbor_x = x + dx
                    if (
                        0 <= neighbor_y < height
                        and 0 <= neighbor_x < width
                        and mask[neighbor_y, neighbor_x]
                        and not visited[neighbor_y, neighbor_x]
                    ):
                        visited[neighbor_y, neighbor_x] = True
                        stack.append((neighbor_y, neighbor_x))
        if len(coordinates) > len(best_coordinates):
            best_coordinates = coordinates
    result = np.zeros(mask.shape, dtype=bool)
    if best_coordinates:
        coordinates = np.asarray(best_coordinates, dtype=np.int64)
        result[coordinates[:, 0], coordinates[:, 1]] = True
    return result


def deterministic_foreground_component(slice_array: np.ndarray) -> tuple[np.ndarray, float]:
    """Extract a binary whole-head outer shape for diagnostic geometry.

    The image border is treated as a background reference only through its
    robust median and MAD; background is not assumed to be zero.  For nonzero
    border MAD, a low threshold is clipped to half of the robust p90 contrast.
    For zero or near-zero border MAD, the threshold uses half of the 10th
    percentile of values above an interior background noise floor. The negative
    tail relative to the border median estimates Gaussian background sigma
    (half-normal median / 0.67448975); three sigma excludes background from
    foreground candidates and bounds the threshold below. This assumes a
    brighter head and approximately symmetric background noise, and requires
    dim head signal above that noise floor. Bright high-tail structures do not
    set the noise estimate. The largest 8-connected component removes
    separate bright structures, then hole filling and one 3x3 closing produce
    the binary outer shape used by centroid/PCA.
    This is an image-only diagnostic heuristic, not a lesion or brain mask.
    """
    values = np.asarray(slice_array, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"slice must be 2-D, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("slice contains non-finite values")
    border = np.concatenate(
        (values[0, :], values[-1, :], values[1:-1, 0], values[1:-1, -1])
    )
    background = float(np.median(border))
    mad = float(np.median(np.abs(border - background)))
    robust_scale = 1.4826 * mad
    epsilon = float(np.finfo(np.float32).eps)
    if robust_scale <= epsilon:
        negative_tail = background - values[values < background - epsilon]
        interior_sigma = (
            float(np.median(negative_tail)) / 0.67448975
            if negative_tail.size else 0.0
        )
        noise_floor = background + 3.0 * interior_sigma
        positive_foreground = values[values > noise_floor + epsilon]
        if positive_foreground.size:
            low_foreground = float(np.percentile(positive_foreground, 10.0))
            threshold = max(noise_floor, background + 0.5 * (low_foreground - background))
        else:
            threshold = noise_floor
    else:
        upper = float(np.percentile(values, 90.0))
        contrast = max(upper - background, 0.0)
        threshold = background + min(2.5 * robust_scale, 0.5 * contrast)
    if not math.isfinite(threshold):
        raise ValueError("foreground threshold is non-finite")
    mask = values > threshold
    component = _largest_connected_component(mask)
    component = ndimage.binary_fill_holes(component)
    component = ndimage.binary_closing(
        component, structure=np.ones((3, 3), dtype=bool), iterations=1
    )
    if int(component.sum()) < 3:
        raise ValueError("whole-head foreground component is too small")
    return component, threshold


def _normalize_axis_angle(angle_degree: float) -> float:
    """Normalize an undirected principal-axis angle to [-90, 90)."""
    normalized = (float(angle_degree) + 90.0) % 180.0 - 90.0
    if normalized >= 90.0:
        normalized -= 180.0
    return normalized


def _axial_median_degree(angles: Sequence[float]) -> float:
    """Return a deterministic modulo-180 median in [-90, 90)."""
    normalized = tuple(_normalize_axis_angle(angle) for angle in angles)
    if not normalized:
        raise ValueError("axial median requires at least one angle")
    anchor_index = min(
        range(len(normalized)),
        key=lambda index: (
            sum(
                abs(_normalize_axis_angle(angle - normalized[index]))
                for angle in normalized
            ),
            index,
        ),
    )
    anchor = normalized[anchor_index]
    unwrapped = [
        anchor + _normalize_axis_angle(angle - anchor)
        for angle in normalized
    ]
    return _normalize_axis_angle(float(np.median(unwrapped)))


def _vertical_axis_correction_degree(measured_axis_degree: float) -> float:
    """Return the shortest modulo-180 rotation from an axis to canonical H/y."""
    return _normalize_axis_angle(
        CANONICAL_VERTICAL_AXIS_DEGREE - float(measured_axis_degree)
    )


def _vertical_axis_distance_degree(angle_degree: float) -> float:
    """Return an undirected axis angle's distance from canonical H/y."""
    return abs(_vertical_axis_correction_degree(float(angle_degree)))


def estimate_slice_geometry(slice_array: np.ndarray, *, slice_index: int, percent: int) -> SliceGeometry:
    """Estimate centroid and principal axis from one foreground component."""
    component, threshold = deterministic_foreground_component(slice_array)
    coordinates_yx = np.argwhere(component).astype(np.float64)
    coordinates_xy = coordinates_yx[:, ::-1]
    centroid = coordinates_xy.mean(axis=0)
    centered = coordinates_xy - centroid
    covariance = (centered.T @ centered) / float(len(coordinates_xy))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal_index = int(np.argmax(eigenvalues))
    principal_vector = eigenvectors[:, principal_index]
    if float(eigenvalues[-1] - eigenvalues[0]) <= np.finfo(np.float64).eps:
        angle_degree = 0.0
    else:
        angle_degree = _normalize_axis_angle(
            math.degrees(math.atan2(float(principal_vector[1]), float(principal_vector[0])))
        )
    height, width = component.shape
    image_center_x = (width - 1) / 2.0
    return SliceGeometry(
        percent=int(percent),
        index=int(slice_index),
        foreground_pixel_count=int(len(coordinates_xy)),
        foreground_threshold=float(threshold),
        centroid_x=float(centroid[0]),
        centroid_y=float(centroid[1]),
        lr_centroid_offset_pixels=float(centroid[0] - image_center_x),
        principal_axis_angle_degree=float(angle_degree),
        equivalent_major_radius_pixels=float(2.0 * math.sqrt(max(float(eigenvalues[-1]), 0.0))),
        equivalent_minor_radius_pixels=float(2.0 * math.sqrt(max(float(eigenvalues[0]), 0.0))),
    )


def estimate_case_geometry(volume: np.ndarray) -> GeometryEstimate:
    """Estimate geometry on 25/50/75 acquisition-plane slices and take medians."""
    values = np.asarray(volume, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"volume must be 3-D, got {values.shape}")
    selected = tuple(
        estimate_slice_geometry(values[item.index], slice_index=item.index, percent=item.percent)
        for item in slice_indices(int(values.shape[0]))
    )
    return GeometryEstimate(
        slices=selected,
        estimated_content_rotation_degree=_axial_median_degree(
            [item.principal_axis_angle_degree for item in selected]
        ),
        estimated_content_lr_translation_pixels=float(
            np.median([item.lr_centroid_offset_pixels for item in selected])
        ),
        estimated_centroid_x=float(np.median([item.centroid_x for item in selected])),
        estimated_centroid_y=float(np.median([item.centroid_y for item in selected])),
    )


def geometry_pose_plausibility(estimate: GeometryEstimate) -> dict[str, float | bool]:
    """Report a diagnostic warning for large poses from canonical H/y vertical."""
    estimated_abs_rotation_degree = _vertical_axis_distance_degree(
        estimate.estimated_content_rotation_degree
    )
    return {
        "estimated_abs_rotation_degree": estimated_abs_rotation_degree,
        "large_rotation_warning": estimated_abs_rotation_degree > 30.0,
    }


def _validate_spatial_shape(spatial_shape: tuple[int, int, int]) -> None:
    if len(spatial_shape) != 3 or any(
        not isinstance(size, int) or isinstance(size, bool) or size < 2 for size in spatial_shape
    ):
        raise ValueError("spatial_shape must be a (D, H, W) tuple with each size at least 2")


def build_forward_content_transform(
    *,
    rotation_degree: float,
    lr_translation_pixels: float,
    spatial_shape: tuple[int, int, int],
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> Tensor:
    """Build a voxel-space forward content transform ``F``.

    Vectors use ``(x, y, z) = (W, H, D)`` voxel coordinates.  The rotation is
    around the image center and the LR translation is applied in output voxel
    coordinates, so positive translation moves image content toward increasing
    canonical W/x.  The returned shape is ``[1, 4, 4]``.
    """
    _validate_spatial_shape(spatial_shape)
    if not math.isfinite(float(rotation_degree)) or not math.isfinite(float(lr_translation_pixels)):
        raise ValueError("content transform parameters must be finite")
    depth, height, width = spatial_shape
    if float(rotation_degree) == 0.0 and float(lr_translation_pixels) == 0.0:
        return torch.eye(4, dtype=dtype, device=device).unsqueeze(0)
    angle = math.radians(float(rotation_degree))
    cosine, sine = math.cos(angle), math.sin(angle)
    center = torch.tensor(
        ((width - 1) / 2.0, (height - 1) / 2.0, (depth - 1) / 2.0),
        dtype=dtype,
        device=device,
    )
    rotation = torch.eye(4, dtype=dtype, device=device)
    rotation[:3, :3] = torch.tensor(
        (
            (cosine, -sine, 0.0),
            (sine, cosine, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=dtype,
        device=device,
    )
    rotation[:3, 3] = center - rotation[:3, :3] @ center
    translation = torch.eye(4, dtype=dtype, device=device)
    translation[0, 3] = float(lr_translation_pixels)
    # F maps input content to output content.  This order represents the
    # standard pose p_out = R_about_center(p_in) + t.
    return (translation @ rotation).unsqueeze(0)


def build_geometry_correction_transform(
    estimate: GeometryEstimate,
    *,
    spatial_shape: tuple[int, int, int],
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> Tensor:
    """Return the forward transform that aligns the measured axis to canonical H/y.

    The measured angle is relative to the W/x horizontal axis and is
    undirected modulo 180 degrees.  The applied rotation is the shortest
    modulo-180 correction from that axis to canonical H/y vertical (±90°).
    The reference pose stores the negated correction before taking its explicit
    inverse, preserving the existing centroid translation order and logic.
    """
    correction_degree = _vertical_axis_correction_degree(
        estimate.estimated_content_rotation_degree
    )
    measured_pose = build_forward_content_transform(
        rotation_degree=-correction_degree,
        lr_translation_pixels=estimate.estimated_content_lr_translation_pixels,
        spatial_shape=spatial_shape,
        dtype=dtype,
        device=device,
    )
    return torch.linalg.inv(measured_pose)


def _normalized_voxel_matrices(
    spatial_shape: tuple[int, int, int], *, batch_size: int, dtype: torch.dtype, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Return align_corners=False normalized-to-voxel and inverse matrices."""
    _validate_spatial_shape(spatial_shape)
    depth, height, width = spatial_shape
    normalized_to_voxel = torch.eye(4, dtype=dtype, device=device).expand(batch_size, -1, -1).clone()
    normalized_to_voxel[:, :3, :3] = torch.diag(
        torch.tensor((width / 2.0, height / 2.0, depth / 2.0), dtype=dtype, device=device)
    )
    normalized_to_voxel[:, :3, 3] = torch.tensor(
        ((width - 1) / 2.0, (height - 1) / 2.0, (depth - 1) / 2.0),
        dtype=dtype,
        device=device,
    )
    return normalized_to_voxel, torch.linalg.inv(normalized_to_voxel)


def _as_batch_transform(transform: Tensor, *, name: str) -> Tensor:
    if not isinstance(transform, Tensor):
        raise ValueError(f"{name} must be a torch tensor")
    if transform.ndim == 2:
        transform = transform.unsqueeze(0)
    if transform.ndim != 3 or transform.shape[1:] != (4, 4):
        raise ValueError(f"{name} must have shape [B, 4, 4] or [4, 4], got {tuple(transform.shape)}")
    if not torch.isfinite(transform).all():
        raise ValueError(f"{name} contains non-finite values")
    return transform


def content_transform_to_sampling_matrices(
    forward_content_transform: Tensor, *, spatial_shape: tuple[int, int, int]
) -> tuple[Tensor, Tensor]:
    """Convert voxel-space ``F`` and ``inverse(F)`` to normalized sampling matrices.

    If ``F`` maps input content voxels to output content voxels, the sampler
    must use ``inverse(F)``.  For ``align_corners=False`` and
    ``A: normalized -> voxel``, the matrix passed to ``warp_volume`` is
    ``A^-1 @ inverse(F) @ A``.  The second return value is its exact inverse
    and is suitable for the existing ``alignment_losses`` reconstruction term.
    """
    forward = _as_batch_transform(forward_content_transform, name="forward_content_transform")
    identity = torch.eye(4, dtype=forward.dtype, device=forward.device).expand(
        forward.shape[0], -1, -1
    )
    if torch.equal(forward, identity):
        return identity.clone(), identity.clone()
    normalized_to_voxel, voxel_to_normalized = _normalized_voxel_matrices(
        spatial_shape,
        batch_size=int(forward.shape[0]),
        dtype=forward.dtype,
        device=forward.device,
    )
    inverse_forward = torch.linalg.inv(forward)
    sampling = voxel_to_normalized @ inverse_forward @ normalized_to_voxel
    inverse_sampling = voxel_to_normalized @ forward @ normalized_to_voxel
    return sampling, inverse_sampling


def content_transform_to_sampling_matrix(
    forward_content_transform: Tensor, *, spatial_shape: tuple[int, int, int]
) -> Tensor:
    """Return only the normalized output-to-input matrix for ``warp_volume``."""
    return content_transform_to_sampling_matrices(
        forward_content_transform, spatial_shape=spatial_shape
    )[0]


def sampling_to_forward_content_transform(sampling_matrix: Tensor, *, spatial_shape: tuple[int, int, int]) -> Tensor:
    """Express an output-to-input normalized sampler as a voxel-space content map."""
    sampling = _as_batch_transform(sampling_matrix, name="sampling_matrix")
    normalized_to_voxel, voxel_to_normalized = _normalized_voxel_matrices(
        spatial_shape,
        batch_size=int(sampling.shape[0]),
        dtype=sampling.dtype,
        device=sampling.device,
    )
    return normalized_to_voxel @ torch.linalg.inv(sampling) @ voxel_to_normalized


def _finite_loss_values(
    losses: Any, *, case_id: str, state: str
) -> tuple[float, float, float]:
    values = tuple(float(getattr(losses, name).detach().cpu()) for name in (
        "flip_loss", "reconstruction_loss", "total_loss"
    ))
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"non-finite {state} loss for {case_id}")
    return values


def _geometry_from_aligned(aligned: Tensor, padding: dict[str, Any]) -> GeometryEstimate:
    unpadded = unpad_model_input_depth(aligned, padding)
    return estimate_case_geometry(unpadded[0, 0].detach().cpu().numpy())


def _matrix_list(matrix: Tensor | None) -> list[list[float]] | None:
    if matrix is None:
        return None
    batch = _as_batch_transform(matrix, name="matrix")
    return [[float(value) for value in row] for row in batch[0].detach().cpu().tolist()]


def _tensor_tuple(values: Tensor | None) -> tuple[float, ...] | None:
    if values is None:
        return None
    return tuple(float(value) for value in values[0].detach().cpu().tolist())


def _extract_content_parameters(
    forward: Tensor | None, *, spatial_shape: tuple[int, int, int]
) -> tuple[float, float] | tuple[None, None]:
    if forward is None:
        return None, None
    matrix = _as_batch_transform(forward, name="forward_content_transform")[0].detach().cpu().numpy()
    rotation_degree = _normalize_axis_angle(math.degrees(math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))))
    depth, height, width = spatial_shape
    center = np.asarray(((width - 1) / 2.0, (height - 1) / 2.0, (depth - 1) / 2.0))
    rotation_part = matrix[:3, :3]
    translation = matrix[:3, 3] - (center - rotation_part @ center)
    return float(rotation_degree), float(translation[0])


def _compare_state_geometry(
    state: StateComparison, *, before: GeometryEstimate
) -> dict[str, float | None]:
    after = state.geometry
    return {
        "centroid_x_before": before.estimated_centroid_x,
        "centroid_y_before": before.estimated_centroid_y,
        "centroid_x_after": None if after is None else after.estimated_centroid_x,
        "centroid_y_after": None if after is None else after.estimated_centroid_y,
        "principal_axis_angle_before_degree": before.estimated_content_rotation_degree,
        "principal_axis_angle_after_degree": None if after is None else after.estimated_content_rotation_degree,
        "absolute_centroid_offset_before_pixels": abs(before.estimated_content_lr_translation_pixels),
        "absolute_centroid_offset_after_pixels": (
            None if after is None else abs(after.estimated_content_lr_translation_pixels)
        ),
        "absolute_principal_axis_tilt_before_degree": _vertical_axis_distance_degree(
            before.estimated_content_rotation_degree
        ),
        "absolute_principal_axis_tilt_after_degree": (
            None if after is None else _vertical_axis_distance_degree(
                after.estimated_content_rotation_degree
            )
        ),
    }


def _state_record(
    state: StateComparison,
    *,
    case_id: str,
    before: GeometryEstimate,
    spatial_shape: tuple[int, int, int],
) -> dict[str, Any]:
    geometry_fields = _compare_state_geometry(state, before=before)
    applied_rotation, applied_translation = _extract_content_parameters(
        state.forward_content_transform, spatial_shape=spatial_shape
    )
    return {
        "case_id": case_id,
        "state": state.state,
        "status": state.status,
        "available": state.available,
        "estimated_content_rotation_degree": before.estimated_content_rotation_degree,
        "estimated_content_lr_translation_pixels": before.estimated_content_lr_translation_pixels,
        "applied_content_rotation_degree": applied_rotation,
        "applied_content_lr_translation_pixels": applied_translation,
        "applied_content_transform": _matrix_list(state.forward_content_transform),
        "sampling_matrix": _matrix_list(state.sampling_matrix),
        "inverse_sampling_matrix": _matrix_list(state.inverse_sampling_matrix),
        "flip_loss": state.flip_loss,
        "reconstruction_loss": state.reconstruction_loss,
        "total_loss": state.total_loss,
        **geometry_fields,
        "raw_params": None if state.raw_params is None else list(state.raw_params),
        "scaled_params": None if state.scaled_params is None else list(state.scaled_params),
    }


def _csv_row(record: dict[str, Any]) -> dict[str, Any]:
    row = {field: record.get(field) for field in CSV_FIELDS}
    for field in ("applied_content_transform", "sampling_matrix", "inverse_sampling_matrix"):
        value = row[field]
        row[field] = "" if value is None else json.dumps(value, separators=(",", ":"))
    return row


def _save_comparison_csv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_csv_row(record) for record in records)


def _state_volume(state: StateComparison, *, original: np.ndarray, padding: dict[str, Any]) -> np.ndarray:
    if state.aligned is None:
        return original
    return unpad_model_input_depth(state.aligned, padding)[0, 0].detach().cpu().numpy()


def _draw_image(
    axis: Any,
    image: np.ndarray,
    *,
    title: str,
    geometry: SliceGeometry | None,
    contour_mask: np.ndarray | None = None,
    cmap: str = "gray",
) -> None:
    axis.imshow(image, cmap=cmap, interpolation="nearest")
    axis.set_title(title, fontsize=7)
    axis.axis("off")
    if contour_mask is not None:
        axis.contour(contour_mask.astype(np.float32), levels=(0.5,), colors="cyan", linewidths=0.7)
    if geometry is not None:
        axis.scatter([geometry.centroid_x], [geometry.centroid_y], s=10, c="red")
        angle = math.radians(geometry.principal_axis_angle_degree)
        length = max(4.0, min(image.shape) * 0.20)
        dx = length * math.cos(angle)
        dy = length * math.sin(angle)
        axis.plot(
            [geometry.centroid_x - dx, geometry.centroid_x + dx],
            [geometry.centroid_y - dy, geometry.centroid_y + dy],
            color="yellow",
            linewidth=1.0,
        )


def _slice_geometry(estimate: GeometryEstimate | None, index: int) -> SliceGeometry | None:
    if estimate is None:
        return None
    for item in estimate.slices:
        if item.index == index:
            return item
    return None


def _geometry_text(label: str, geometry: GeometryEstimate | None) -> str:
    if geometry is None:
        return f"{label}: unavailable"
    return (
        f"{label}: centroid=({geometry.estimated_centroid_x:.1f}, "
        f"{geometry.estimated_centroid_y:.1f}), "
        f"axis angle (W/x reference)={geometry.estimated_content_rotation_degree:.1f} deg"
    )


def _save_geometry_qc(
    path: Path,
    *,
    original: np.ndarray,
    adn: np.ndarray,
    geometry_aligned: np.ndarray,
    before: GeometryEstimate,
    adn_geometry: GeometryEstimate | None,
    geometry_geometry: GeometryEstimate,
) -> None:
    selected = slice_indices(int(original.shape[0]))
    figure, axes = plt.subplots(
        len(selected),
        11,
        figsize=(27.5, 4.0 * len(selected)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, selected_slice in enumerate(selected):
        index = selected_slice.index
        original_slice = original[index]
        adn_slice = adn[index]
        geometry_slice = geometry_aligned[index]
        original_mirror = np.flip(original_slice, axis=1)
        adn_mirror = np.flip(adn_slice, axis=1)
        geometry_mirror = np.flip(geometry_slice, axis=1)
        original_geometry = _slice_geometry(before, index)
        adn_slice_geometry = _slice_geometry(adn_geometry, index)
        geometry_slice_geometry = _slice_geometry(geometry_geometry, index)
        whole_head_mask, _ = deterministic_foreground_component(original_slice)
        panels = (
            (original_slice, f"original {selected_slice.percent}%", original_geometry, None, "gray"),
            (whole_head_mask, "whole-head mask/contour", original_geometry, whole_head_mask, "gray"),
            (original_mirror, "original LR mirror", None, None, "gray"),
            (np.abs(original_slice - original_mirror), "original abs diff", None, None, "magma"),
            (
                adn_slice,
                "ADN aligned" if adn_geometry is not None else "ADN aligned (not requested)",
                adn_slice_geometry,
                None,
                "gray",
            ),
            (adn_mirror, "ADN LR mirror", None, None, "gray"),
            (np.abs(adn_slice - adn_mirror), "ADN abs diff", None, None, "magma"),
            (geometry_slice, "geometry aligned", geometry_slice_geometry, None, "gray"),
            (geometry_mirror, "geometry LR mirror", None, None, "gray"),
            (np.abs(geometry_slice - geometry_mirror), "geometry abs diff", None, None, "magma"),
        )
        for column_index, (image, title, slice_geometry, contour_mask, cmap) in enumerate(panels):
            _draw_image(
                axes[row_index, column_index],
                image,
                title=title,
                geometry=slice_geometry,
                contour_mask=contour_mask,
                cmap=cmap,
            )
        text_axis = axes[row_index, 10]
        text_axis.axis("off")
        text_axis.text(
            0.0,
            1.0,
            "\n".join(
                (
                    f"slice {selected_slice.percent}% (index {index})",
                    _geometry_text("before", before),
                    _geometry_text("ADN after", adn_geometry),
                    _geometry_text("geometry after", geometry_geometry),
                )
            ),
            va="top",
            fontsize=8,
        )
    figure.suptitle(
        "Geometry-vs-loss QC: red=centroid, yellow=principal axis; "
        "geometry target axis = canonical H/y vertical (±90°); "
        "centroid and principal-axis angle are shown before/after",
        fontsize=11,
    )
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _loss_change_percent(new_value: float, reference_value: float) -> float | None:
    if reference_value == 0.0:
        return None
    return 100.0 * (new_value - reference_value) / abs(reference_value)


def _run_case(
    *,
    case_id: str,
    image_path: Path,
    checkpoint: Any | None,
    checkpoint_path: Path | None,
    device: torch.device,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    # Keep this order explicit: image read -> acquisition-preserving canonical
    # array -> per-volume normalization -> model-only D padding.
    canonical = canonicalize_nifti(read_nifti(image_path))
    normalized_array = normalize_volume(canonical.array)
    original = torch.from_numpy(normalized_array)[None, None].to(device)
    model_input, padding = pad_model_input_depth(original)
    spatial_shape = tuple(int(size) for size in model_input.shape[-3:])
    before_geometry = estimate_case_geometry(normalized_array)

    identity_forward = build_forward_content_transform(
        rotation_degree=0.0,
        lr_translation_pixels=0.0,
        spatial_shape=spatial_shape,
        dtype=model_input.dtype,
        device=model_input.device,
    )
    identity_sampling, identity_inverse_sampling = content_transform_to_sampling_matrices(
        identity_forward, spatial_shape=spatial_shape
    )
    identity_losses = alignment_losses(model_input, model_input, identity_inverse_sampling)
    identity_flip, identity_reconstruction, identity_total = _finite_loss_values(
        identity_losses, case_id=case_id, state="identity"
    )
    identity = StateComparison(
        state="identity",
        status="ok",
        available=True,
        aligned=model_input,
        forward_content_transform=identity_forward,
        sampling_matrix=identity_sampling,
        inverse_sampling_matrix=identity_inverse_sampling,
        flip_loss=identity_flip,
        reconstruction_loss=identity_reconstruction,
        total_loss=identity_total,
        geometry=before_geometry,
    )

    geometry_forward = build_geometry_correction_transform(
        before_geometry,
        spatial_shape=spatial_shape,
        dtype=model_input.dtype,
        device=model_input.device,
    )
    geometry_sampling, geometry_inverse_sampling = content_transform_to_sampling_matrices(
        geometry_forward, spatial_shape=spatial_shape
    )
    with torch.no_grad():
        geometry_aligned_tensor = warp_volume(model_input, geometry_sampling)
        geometry_losses = alignment_losses(
            model_input, geometry_aligned_tensor, geometry_inverse_sampling
        )
    geometry_flip, geometry_reconstruction, geometry_total = _finite_loss_values(
        geometry_losses, case_id=case_id, state="geometry_estimate"
    )
    geometry = StateComparison(
        state="geometry_estimate",
        status="ok",
        available=True,
        aligned=geometry_aligned_tensor,
        forward_content_transform=geometry_forward,
        sampling_matrix=geometry_sampling,
        inverse_sampling_matrix=geometry_inverse_sampling,
        flip_loss=geometry_flip,
        reconstruction_loss=geometry_reconstruction,
        total_loss=geometry_total,
        geometry=_geometry_from_aligned(geometry_aligned_tensor, padding),
    )

    if checkpoint is None:
        adn = StateComparison(
            state="adn_prediction",
            status="not_requested",
            available=False,
            aligned=None,
            forward_content_transform=None,
            sampling_matrix=None,
            inverse_sampling_matrix=None,
            flip_loss=None,
            reconstruction_loss=None,
            total_loss=None,
            geometry=None,
        )
    else:
        with torch.no_grad():
            result = checkpoint.model(model_input)
            adn_losses = alignment_losses(
                model_input, result.aligned, result.inverse_sampling_matrix
            )
        adn_flip, adn_reconstruction, adn_total = _finite_loss_values(
            adn_losses, case_id=case_id, state="adn_prediction"
        )
        adn = StateComparison(
            state="adn_prediction",
            status="ok",
            available=True,
            aligned=result.aligned,
            forward_content_transform=sampling_to_forward_content_transform(
                result.sampling_matrix, spatial_shape=spatial_shape
            ),
            sampling_matrix=result.sampling_matrix,
            inverse_sampling_matrix=result.inverse_sampling_matrix,
            flip_loss=adn_flip,
            reconstruction_loss=adn_reconstruction,
            total_loss=adn_total,
            geometry=_geometry_from_aligned(result.aligned, padding),
            raw_params=_tensor_tuple(result.raw_params),
            scaled_params=_tensor_tuple(result.scaled_params),
        )

    records = [
        _state_record(
            state,
            case_id=case_id,
            before=before_geometry,
            spatial_shape=spatial_shape,
        )
        for state in (identity, adn, geometry)
    ]
    case_output = output_dir / case_id
    case_output.mkdir()
    _save_comparison_csv(case_output / "comparison.csv", records)

    adn_volume = _state_volume(adn, original=normalized_array, padding=padding)
    geometry_volume = _state_volume(geometry, original=normalized_array, padding=padding)
    _save_geometry_qc(
        case_output / "geometry_qc.png",
        original=normalized_array,
        adn=adn_volume,
        geometry_aligned=geometry_volume,
        before=before_geometry,
        adn_geometry=adn.geometry,
        geometry_geometry=geometry.geometry,
    )

    geometry_after = geometry.geometry
    assert geometry_after is not None
    summary = {
        "case_id": case_id,
        "image": str(image_path),
        "checkpoint": None if checkpoint_path is None else str(checkpoint_path.resolve()),
        "device": str(device),
        "diagnostic_only": True,
        "labels_accessed": False,
        "training_called": False,
        "adn_available": adn.available,
        "slice_indices": [asdict(item) for item in slice_indices(int(normalized_array.shape[0]))],
        "estimated_content_rotation_degree": before_geometry.estimated_content_rotation_degree,
        "estimated_content_lr_translation_pixels": before_geometry.estimated_content_lr_translation_pixels,
        "geometry_angle_degree": before_geometry.estimated_content_rotation_degree,
        "geometry_translation_pixels": before_geometry.estimated_content_lr_translation_pixels,
        "geometry_pose_plausibility": geometry_pose_plausibility(before_geometry),
        "geometry_applied_content_rotation_degree": records[2]["applied_content_rotation_degree"],
        "geometry_applied_content_lr_translation_pixels": records[2]["applied_content_lr_translation_pixels"],
        "identity_total_loss": identity.total_loss,
        "adn_total_loss": adn.total_loss,
        "geometry_aligned_total_loss": geometry.total_loss,
        "geometry_vs_identity_loss_change_percent": _loss_change_percent(
            float(geometry.total_loss), float(identity.total_loss)
        ),
        "geometry_reduced_absolute_centroid_offset": (
            abs(geometry_after.estimated_content_lr_translation_pixels)
            < abs(before_geometry.estimated_content_lr_translation_pixels)
        ),
        "geometry_reduced_absolute_principal_axis_tilt": (
            _vertical_axis_distance_degree(geometry_after.estimated_content_rotation_degree)
            < _vertical_axis_distance_degree(before_geometry.estimated_content_rotation_degree)
        ),
        "geometry_before": asdict(before_geometry),
        "geometry_after": asdict(geometry_after),
        "states": records,
        "model_input_depth_padding": padding,
        "orientation_canonicalization": getattr(canonical, "provenance", {}),
        "display_space": "canonical_per_volume_zscore",
        "geometry_mask_assumptions": {
            "background_reference": "image-border median and MAD; background is not assumed to be zero",
            "threshold": "if 1.4826 * border_MAD <= float32_epsilon: sigma = median(border_median - values[values < border_median - float32_epsilon]) / 0.67448975 (zero if empty); noise_floor = border_median + 3 * sigma; use max(noise_floor, border_median + 0.5 * (p10(values > noise_floor + float32_epsilon) - border_median)), or noise_floor when empty; otherwise use border_median + min(2.5 * 1.4826 * border_MAD, 0.5 * max(p90 - border_median, 0)); strict greater-than",
            "postprocessing": "largest 8-connected component, binary hole fill, one 3x3 binary closing",
            "purpose": "diagnostic whole-head outer shape for centroid/PCA; not a lesion or brain segmentation",
        },
        "transform_semantics": {
            "content_coordinates": "voxel_(x=W,y=H,z=D)",
            "geometry_estimate": "observed_content_pose_from_binary_whole_head_centroid_and_PCA_angle_relative_to_W/x_horizontal",
            "geometry_forward_transform": "inverse_of_vertical_reference_pose; maps input content to canonical H/y vertical output content",
            "sampling_matrix": "align_corners_false_output_to_input_normalized_inverse(F)",
            "adn_sampling_matrix": "existing_checkpoint_output_to_input_normalized_sampling_matrix",
            "physical_3d_rigid_registration": False,
        },
    }
    return summary, records, case_output


def _case_id(path: Path) -> str:
    name = path.name
    suffix = "_0000.nii.gz"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path)
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
    image_paths = [
        (dataset / "imagesTr" / f"{case_id}_0000.nii.gz").resolve()
        for case_id in arguments.cases
    ]
    for image_path in image_paths:
        if not image_path.is_file():
            raise FileNotFoundError(f"NIfTI image does not exist: {image_path}")
    checkpoint_path = None if arguments.checkpoint is None else arguments.checkpoint.resolve()
    if checkpoint_path is not None and not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    output = ensure_diagnostic_output_dir(arguments.output_dir, dataset_dir=dataset)
    device = torch.device(arguments.device)
    loaded_checkpoint = None
    if checkpoint_path is not None:
        loaded_checkpoint = load_diagnostic_checkpoint(checkpoint_path, device=device)
        loaded_checkpoint.model.eval()

    for case_id, image_path in zip(arguments.cases, image_paths, strict=True):
        summary, records, case_output = _run_case(
            case_id=case_id,
            image_path=image_path,
            checkpoint=loaded_checkpoint,
            checkpoint_path=checkpoint_path,
            device=device,
            output_dir=output,
        )
        (case_output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
