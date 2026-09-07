"""Array-only NIfTI orientation adapter for canonical ADN model space.

SimpleITK directions map voxel ``(x, y, z)`` axes into LPS physical axes.
This module establishes ``[D,H,W] = [SI,AP,LR]`` with positive directions
``LPS +Z,+Y,+X`` using only permutation and flips.  It never constructs a
replacement NIfTI geometry and never resamples voxel values.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import math
from typing import Any

import numpy as np

from standalone_nnunet2d.data.nifti_io import NiftiVolume


ORTHOGONALITY_TOLERANCE = 1e-4
DOMINANT_ABS_MINIMUM = math.cos(math.radians(20.0))
DOMINANCE_MARGIN_MINIMUM = 0.20
_SCORE_TOLERANCE = 1e-12
_CANONICAL_WORLD_AXES = (2, 1, 0)  # D=SI(+Z), H=AP(+Y), W=LR(+X)
_CANONICAL_AXIS_NAMES = ("SI", "AP", "LR")


class OrientationError(ValueError):
    """Raised when voxel-axis anatomy cannot be established safely."""

    def __init__(self, reason: str, details: dict[str, Any]) -> None:
        self.reason = reason
        self.details = details
        super().__init__(f"{reason}: {details}")


@dataclass(frozen=True)
class OrientationMapping:
    """A safe, global canonical-axis assignment."""

    permutation: tuple[int, int, int]
    flips: tuple[bool, bool, bool]
    direction_labels: tuple[str, str, str]
    voxel_axis_for_world_xyz: tuple[int, int, int]
    assignment_score: float
    assignment_count: int
    orthogonality_error: float
    matched_abs: tuple[float, float, float]
    dominance_margins: tuple[float, float, float]

    def to_provenance(self) -> dict[str, Any]:
        return {
            "canonical_axes": list(_CANONICAL_AXIS_NAMES),
            "canonical_positive_lps": ["+Z", "+Y", "+X"],
            "permutation": list(self.permutation),
            "flips": list(self.flips),
            "direction_labels": list(self.direction_labels),
            "voxel_axis_for_world_xyz": list(self.voxel_axis_for_world_xyz),
            "assignment_method": "exhaustive_global_one_to_one_permutation",
            "assignment_score": self.assignment_score,
            "assignment_count": self.assignment_count,
            "orthogonality_error": self.orthogonality_error,
            "matched_abs": list(self.matched_abs),
            "dominance_margins": list(self.dominance_margins),
            "thresholds": {
                "orthogonality_error_max": ORTHOGONALITY_TOLERANCE,
                "dominant_abs_min": DOMINANT_ABS_MINIMUM,
                "dominant_minus_second_best_min": DOMINANCE_MARGIN_MINIMUM,
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
    """Resolve voxel axes with one exhaustive global assignment."""
    matrix = np.asarray(direction, dtype=np.float64)
    if matrix.shape != (9,):
        _reject("invalid_direction_length", expected=9, actual=int(matrix.size))
    if not np.isfinite(matrix).all():
        _reject("nonfinite_direction", direction=matrix.tolist())
    matrix = matrix.reshape(3, 3)
    orthogonality_error = float(np.max(np.abs(matrix.T @ matrix - np.eye(3))))
    if orthogonality_error > ORTHOGONALITY_TOLERANCE:
        _reject(
            "nonorthogonal_direction",
            orthogonality_error=orthogonality_error,
            maximum=ORTHOGONALITY_TOLERANCE,
        )

    absolute = np.abs(matrix)
    candidates: list[tuple[float, tuple[int, int, int]]] = []
    # candidate[world_axis] = voxel axis, hence every candidate is one-to-one.
    for assignment in permutations(range(3)):
        score = float(sum(absolute[world, assignment[world]] for world in range(3)))
        candidates.append((score, assignment))
    best_score = max(score for score, _ in candidates)
    best = [item for item in candidates if abs(item[0] - best_score) <= _SCORE_TOLERANCE]
    if len(best) != 1:
        _reject(
            "ambiguous_assignment",
            assignment_count=len(best),
            best_score=best_score,
            assignments=[list(item[1]) for item in best],
        )
    assignment = best[0][1]

    matched_by_world = tuple(float(absolute[world, assignment[world]]) for world in range(3))
    margins_by_world: list[float] = []
    for world, voxel in enumerate(assignment):
        second_best = max(float(absolute[other_world, voxel]) for other_world in range(3) if other_world != world)
        margins_by_world.append(float(absolute[world, voxel]) - second_best)
    if min(margins_by_world) < DOMINANCE_MARGIN_MINIMUM:
        _reject(
            "dominance_margin",
            margins=margins_by_world,
            minimum=DOMINANCE_MARGIN_MINIMUM,
        )
    if min(matched_by_world) < DOMINANT_ABS_MINIMUM:
        _reject(
            "dominance_threshold",
            matched_abs=list(matched_by_world),
            minimum=DOMINANT_ABS_MINIMUM,
        )

    permutation: list[int] = []
    flips: list[bool] = []
    labels: list[str] = []
    matched_canonical: list[float] = []
    margins_canonical: list[float] = []
    for name, world_axis in zip(_CANONICAL_AXIS_NAMES, _CANONICAL_WORLD_AXES, strict=True):
        voxel_axis = assignment[world_axis]
        permutation.append(2 - voxel_axis)  # SITK array order is z,y,x.
        negative = bool(matrix[world_axis, voxel_axis] < 0)
        flips.append(negative)
        labels.append(f"{name}{'-' if negative else '+'}")
        matched_canonical.append(matched_by_world[world_axis])
        margins_canonical.append(margins_by_world[world_axis])
    return OrientationMapping(
        permutation=tuple(permutation),
        flips=tuple(flips),
        direction_labels=tuple(labels),
        voxel_axis_for_world_xyz=tuple(assignment),
        assignment_score=best_score,
        assignment_count=1,
        orthogonality_error=orthogonality_error,
        matched_abs=tuple(matched_canonical),
        dominance_margins=tuple(margins_canonical),
    )


def canonicalize_nifti(volume: NiftiVolume) -> CanonicalNiftiArray:
    """Create an ADN canonical tensor array while retaining source geometry."""
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
