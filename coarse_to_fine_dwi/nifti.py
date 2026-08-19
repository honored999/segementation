"""Small, explicit NIfTI volume and reversible XY crop primitives."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import TypeAlias

import numpy as np

XYBBox: TypeAlias = tuple[int, int, int, int]
_METADATA_TOLERANCE = 1e-6
# Crop origins accumulate direction/offset arithmetic before NIfTI header
# serialization; keep this tolerance local to restore-time origin checking.
_CROP_ORIGIN_SERIALIZATION_TOLERANCE = 1e-5


def _vector3(values: object, name: str) -> tuple[float, float, float]:
    vector = tuple(float(value) for value in values)  # type: ignore[union-attr]
    if len(vector) != 3 or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain exactly three finite values")
    return vector  # type: ignore[return-value]


def _direction9(values: object) -> tuple[float, ...]:
    direction = tuple(float(value) for value in values)  # type: ignore[union-attr]
    if len(direction) != 9 or not np.isfinite(direction).all():
        raise ValueError("direction must contain exactly nine finite values")
    return direction


@dataclass(frozen=True)
class NiftiVolume:
    """A 3-D array in ``(z, y, x)`` order with SimpleITK spatial metadata."""

    array: np.ndarray
    spacing_xyz: tuple[float, float, float]
    origin_xyz: tuple[float, float, float]
    direction: tuple[float, ...]

    def __post_init__(self) -> None:
        array = np.asarray(self.array)
        if array.ndim != 3:
            raise ValueError("array must be 3-dimensional in (z, y, x) order")
        object.__setattr__(self, "array", array)
        object.__setattr__(self, "spacing_xyz", _vector3(self.spacing_xyz, "spacing_xyz"))
        object.__setattr__(self, "origin_xyz", _vector3(self.origin_xyz, "origin_xyz"))
        object.__setattr__(self, "direction", _direction9(self.direction))

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.array.shape)  # type: ignore[return-value]

    @property
    def direction_matrix(self) -> np.ndarray:
        return np.asarray(self.direction, dtype=float).reshape(3, 3)

    @classmethod
    def read(cls, path: str | PathLike[str]) -> "NiftiVolume":
        import SimpleITK as sitk

        image = sitk.ReadImage(str(Path(path)))
        return cls(
            array=np.asarray(sitk.GetArrayFromImage(image)),
            spacing_xyz=tuple(image.GetSpacing()),
            origin_xyz=tuple(image.GetOrigin()),
            direction=tuple(image.GetDirection()),
        )

    def write(self, path: str | PathLike[str]) -> None:
        import SimpleITK as sitk

        image = sitk.GetImageFromArray(self.array)
        image.SetSpacing(self.spacing_xyz)
        image.SetOrigin(self.origin_xyz)
        image.SetDirection(self.direction)
        sitk.WriteImage(image, str(Path(path)))


def assert_compatible(reference: NiftiVolume, candidate: NiftiVolume) -> None:
    """Raise when two volumes do not describe the same voxel space."""
    if reference.shape_zyx != candidate.shape_zyx:
        raise ValueError(
            f"shape mismatch: reference {reference.shape_zyx}, candidate {candidate.shape_zyx}"
        )
    if not np.allclose(
        reference.spacing_xyz,
        candidate.spacing_xyz,
        atol=_METADATA_TOLERANCE,
        rtol=0.0,
    ) or not np.allclose(
        reference.origin_xyz,
        candidate.origin_xyz,
        atol=_METADATA_TOLERANCE,
        rtol=0.0,
    ) or not np.allclose(
        reference.direction,
        candidate.direction,
        atol=_METADATA_TOLERANCE,
        rtol=0.0,
    ):
        raise ValueError("metadata mismatch between volumes")


def _validate_bbox(bbox: XYBBox, width: int, height: int) -> XYBBox:
    if len(bbox) != 4 or any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in bbox):
        raise ValueError("bbox must contain four integer half-open coordinates")
    x0, y0, x1, y1 = (int(value) for value in bbox)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("bbox must be a non-empty in-bounds half-open XY box")
    return x0, y0, x1, y1


def crop_xy(volume: NiftiVolume, bbox: XYBBox) -> NiftiVolume:
    """Crop an XY half-open box while preserving all z slices and space."""
    _, height, width = volume.shape_zyx
    x0, y0, x1, y1 = _validate_bbox(bbox, width, height)
    offset_xyz = np.array([x0 * volume.spacing_xyz[0], y0 * volume.spacing_xyz[1], 0.0])
    shifted_origin = tuple(volume.origin_xyz + volume.direction_matrix @ offset_xyz)
    return NiftiVolume(
        array=volume.array[:, y0:y1, x0:x1].copy(),
        spacing_xyz=volume.spacing_xyz,
        origin_xyz=shifted_origin,
        direction=volume.direction,
    )


def restore_xy(cropped: NiftiVolume, reference: NiftiVolume, bbox: XYBBox) -> NiftiVolume:
    """Restore a validated XY crop into the reference volume's full space."""
    _, height, width = reference.shape_zyx
    x0, y0, x1, y1 = _validate_bbox(bbox, width, height)
    expected_shape = (reference.shape_zyx[0], y1 - y0, x1 - x0)
    if cropped.shape_zyx != expected_shape:
        raise ValueError(
            f"crop shape mismatch: expected {expected_shape}, got {cropped.shape_zyx}"
        )
    expected_origin = tuple(
        reference.origin_xyz
        + reference.direction_matrix
        @ np.array([x0 * reference.spacing_xyz[0], y0 * reference.spacing_xyz[1], 0.0])
    )
    if not np.allclose(cropped.spacing_xyz, reference.spacing_xyz, atol=_METADATA_TOLERANCE, rtol=0.0):
        raise ValueError("crop metadata mismatch: spacing")
    if not np.allclose(
        cropped.origin_xyz,
        expected_origin,
        atol=_CROP_ORIGIN_SERIALIZATION_TOLERANCE,
        rtol=0.0,
    ):
        raise ValueError("crop metadata mismatch: origin")
    if not np.allclose(
        cropped.direction,
        reference.direction,
        atol=_METADATA_TOLERANCE,
        rtol=0.0,
    ):
        raise ValueError("crop metadata mismatch: direction")

    restored = np.zeros_like(reference.array)
    restored[:, y0:y1, x0:x1] = cropped.array
    return NiftiVolume(
        array=restored,
        spacing_xyz=reference.spacing_xyz,
        origin_xyz=reference.origin_xyz,
        direction=reference.direction,
    )


crop_volume_xy = crop_xy
restore_volume_xy = restore_xy
