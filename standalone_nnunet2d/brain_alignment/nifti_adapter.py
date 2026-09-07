"""Array-only acquisition-preserving NIfTI adapter for ADN model space.

SimpleITK direction columns map voxel ``(x, y, z)`` axes into LPS physical
coordinates.  Canonical ``D`` always remains source voxel z, canonical ``W``
is the safely identified in-plane anatomical LR axis, and canonical ``H`` is
the remaining acquisition in-plane axis.  Only transpose and flip operations
are used; source geometry is retained as provenance and is never rewritten.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from standalone_nnunet2d.data.nifti_io import NiftiVolume


ORTHOGONALITY_TOLERANCE = 1e-4
LR_MARGIN_MINIMUM = 0.20
_SCORE_TOLERANCE = 1e-12
_VOXEL_AXIS_NAMES = ("x", "y", "z")
_ARRAY_AXIS_FOR_VOXEL = (2, 1, 0)
_LPS_AXIS_NAMES = ("X", "Y", "Z")


class OrientationError(ValueError):
    """Raised when voxel-axis LR anatomy cannot be established safely."""

    def __init__(self, reason: str, details: dict[str, Any]) -> None:
        self.reason = reason
        self.details = details
        super().__init__(f"{reason}: {details}")


def _components(vector: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(_LPS_AXIS_NAMES, vector, strict=True)}


def _axis_obliquity(vector: np.ndarray, world_axis: int) -> float:
    """Return unsigned angular deviation from a named LPS axis for provenance."""
    component = float(np.clip(abs(vector[world_axis]), 0.0, 1.0))
    return float(math.degrees(math.acos(component)))


@dataclass(frozen=True)
class OrientationMapping:
    """Acquisition-preserving model-axis mapping and reversible flip record."""

    permutation: tuple[int, int, int]
    flips: tuple[bool, bool, bool]
    direction_labels: tuple[str, str, str]
    source_voxel_axis_for_dhw: tuple[int, int, int]
    lr_source_voxel_axis: int
    lr_signed_component: float
    lr_absolute_component: float
    lr_second_best_component: float
    lr_margin: float
    orthogonality_error: float
    source_direction_vectors: tuple[tuple[float, float, float], ...]
    d_lps_components: tuple[float, float, float]
    h_lps_components: tuple[float, float, float]
    handedness_before_h_flip: float
    handedness_after_flips: float

    def to_provenance(self) -> dict[str, Any]:
        source_axes = [_VOXEL_AXIS_NAMES[axis] for axis in self.source_voxel_axis_for_dhw]
        return {
            "canonical_contract": "acquisition_preserving_lr",
            "canonical_axes": ["D", "H", "W"],
            "model_axis_semantics": {
                "D": "acquisition_through_plane",
                "H": "acquisition_in_plane_non_lr",
                "W": "anatomical_lr",
            },
            "source_voxel_axis_for_dhw": source_axes,
            "lr_source_voxel_axis": _VOXEL_AXIS_NAMES[self.lr_source_voxel_axis],
            "lr_signed_component": self.lr_signed_component,
            "lr_absolute_component": self.lr_absolute_component,
            "lr_second_best_component": self.lr_second_best_component,
            "lr_margin": self.lr_margin,
            "applied_permutation": list(self.permutation),
            "applied_flips": list(self.flips),
            "permutation": list(self.permutation),
            "flips": list(self.flips),
            "direction_labels": list(self.direction_labels),
            "source_direction_vectors": [
                {"voxel_axis": _VOXEL_AXIS_NAMES[axis], "lps_components": _components(np.asarray(vector))}
                for axis, vector in enumerate(self.source_direction_vectors)
            ],
            "d_lps_components": _components(np.asarray(self.d_lps_components)),
            "h_lps_components": _components(np.asarray(self.h_lps_components)),
            "d_obliquity_from_lps_si_degrees": _axis_obliquity(np.asarray(self.d_lps_components), 2),
            "h_obliquity_from_lps_ap_degrees": _axis_obliquity(np.asarray(self.h_lps_components), 1),
            "handedness_before_h_flip": self.handedness_before_h_flip,
            "handedness_after_flips": self.handedness_after_flips,
            "flip_rules": {
                "D": "source_voxel_z_to_lps_positive_z",
                "H": "right_handed_w_h_d",
                "W": "lr_to_lps_positive_x",
            },
            "orthogonality_error": self.orthogonality_error,
            "thresholds": {
                "orthogonality_error_max": ORTHOGONALITY_TOLERANCE,
                "lr_best_minus_second_best_min": LR_MARGIN_MINIMUM,
            },
        }


@dataclass(frozen=True)
class CanonicalNiftiArray:
    """Canonical model array plus unchanged source geometry and inverse map."""

    array: np.ndarray
    permutation: tuple[int, int, int]
    flips: tuple[bool, bool, bool]
    direction_labels: tuple[str, str, str]
    spacing_xyz: tuple[float, float, float]
    origin_xyz: tuple[float, float, float]
    direction: tuple[float, ...]
    original_array_order: str
    provenance: dict[str, Any]

    def restore_array(self, array: np.ndarray | None = None) -> np.ndarray:
        """Exactly undo canonical flips/permutation; no interpolation occurs."""
        restored = self.array if array is None else np.asarray(array)
        if restored.ndim != 3:
            raise ValueError(f"canonical array must be 3D, got {restored.shape}")
        for axis, flip in enumerate(self.flips):
            if flip:
                restored = np.flip(restored, axis=axis)
        inverse = tuple(int(axis) for axis in np.argsort(self.permutation))
        return np.transpose(restored, inverse)


def _reject(reason: str, **details: Any) -> None:
    raise OrientationError(reason, details)


def compute_orientation_mapping(direction: tuple[float, ...]) -> OrientationMapping:
    """Keep voxel z as depth and resolve only a safely separated LR axis."""
    matrix = np.asarray(direction, dtype=np.float64)
    if matrix.shape != (9,):
        _reject("invalid_direction_length", expected=9, actual=int(matrix.size))
    if not np.isfinite(matrix).all():
        _reject("nonfinite_direction", direction=matrix.tolist())
    matrix = matrix.reshape(3, 3)
    orthogonality_error = float(np.max(np.abs(matrix.T @ matrix - np.eye(3))))
    if orthogonality_error > ORTHOGONALITY_TOLERANCE:
        _reject("nonorthogonal_direction", orthogonality_error=orthogonality_error, maximum=ORTHOGONALITY_TOLERANCE)

    lr_absolute = np.abs(matrix[0, :])
    best_abs = float(np.max(lr_absolute))
    best_axes = np.flatnonzero(np.abs(lr_absolute - best_abs) <= _SCORE_TOLERANCE)
    if len(best_axes) != 1:
        _reject("lr_ambiguous", best_axes=[_VOXEL_AXIS_NAMES[int(axis)] for axis in best_axes], best_abs=best_abs)
    lr_axis = int(best_axes[0])
    sorted_abs = sorted((float(value) for value in lr_absolute), reverse=True)
    second_best = sorted_abs[1]
    margin = best_abs - second_best
    if lr_axis == 2:
        _reject("lr_axis_is_voxel_z", lr_absolute_components=lr_absolute.tolist(), lr_margin=margin)
    if margin < LR_MARGIN_MINIMUM:
        _reject("lr_ambiguous", lr_absolute_components=lr_absolute.tolist(), lr_margin=margin, minimum=LR_MARGIN_MINIMUM)

    h_axis = 1 - lr_axis
    d_axis = 2
    if matrix[2, d_axis] == 0.0:
        _reject("d_lps_z_zero", d_lps_z_component=0.0)
    d_flip = bool(matrix[2, d_axis] < 0.0)
    w_flip = bool(matrix[0, lr_axis] < 0.0)
    d_vector = matrix[:, d_axis] * (-1.0 if d_flip else 1.0)
    w_vector = matrix[:, lr_axis] * (-1.0 if w_flip else 1.0)
    h_vector = matrix[:, h_axis]
    handedness_before = float(np.dot(np.cross(w_vector, h_vector), d_vector))
    h_flip = handedness_before < 0.0
    if h_flip:
        h_vector = -h_vector
    handedness_after = float(np.dot(np.cross(w_vector, h_vector), d_vector))

    source_axes = (d_axis, h_axis, lr_axis)
    permutation = tuple(_ARRAY_AXIS_FOR_VOXEL[axis] for axis in source_axes)
    flips = (d_flip, h_flip, w_flip)
    vectors = tuple(tuple(float(value) for value in matrix[:, axis]) for axis in range(3))
    return OrientationMapping(
        permutation=permutation,
        flips=flips,
        direction_labels=("D+Z", "H+right_handed", "W+X"),
        source_voxel_axis_for_dhw=source_axes,
        lr_source_voxel_axis=lr_axis,
        lr_signed_component=float(matrix[0, lr_axis]),
        lr_absolute_component=best_abs,
        lr_second_best_component=second_best,
        lr_margin=margin,
        orthogonality_error=orthogonality_error,
        source_direction_vectors=vectors,
        d_lps_components=tuple(float(value) for value in d_vector),
        h_lps_components=tuple(float(value) for value in h_vector),
        handedness_before_h_flip=handedness_before,
        handedness_after_flips=handedness_after,
    )


def canonicalize_nifti(volume: NiftiVolume) -> CanonicalNiftiArray:
    """Create an acquisition-preserving ADN tensor while retaining geometry."""
    mapping = compute_orientation_mapping(volume.direction)
    original = np.asarray(volume.array)
    canonical = np.transpose(original, mapping.permutation)
    for axis, flip in enumerate(mapping.flips):
        if flip:
            canonical = np.flip(canonical, axis=axis)
    canonical = np.ascontiguousarray(canonical)
    provenance = mapping.to_provenance()
    inverse_permutation = tuple(int(axis) for axis in np.argsort(mapping.permutation))
    provenance.update(
        {
            "original_array_order": "zyx",
            "original_shape": list(original.shape),
            "canonical_shape": list(canonical.shape),
            "source_geometry": {
                "spacing_xyz": list(volume.spacing_xyz),
                "origin_xyz": list(volume.origin_xyz),
                "direction": list(volume.direction),
            },
            "operations": ["transpose", "flip"],
            "inverse_operations": {
                "flip_canonical_axes": [axis for axis, flip in enumerate(mapping.flips) if flip],
                "transpose_axes": list(inverse_permutation),
            },
            "interpolation_performed": False,
            "geometry_modified": False,
            "reversible": True,
        }
    )
    return CanonicalNiftiArray(
        array=canonical,
        permutation=mapping.permutation,
        flips=mapping.flips,
        direction_labels=mapping.direction_labels,
        spacing_xyz=volume.spacing_xyz,
        origin_xyz=volume.origin_xyz,
        direction=volume.direction,
        original_array_order="zyx",
        provenance=provenance,
    )
