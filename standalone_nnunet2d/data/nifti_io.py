"""Bounded SimpleITK I/O for one requested NIfTI volume at a time."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk


@dataclass(frozen=True)
class NiftiVolume:
    """Array and spatial metadata, with arrays ordered as ``(z, y, x)``."""

    array: np.ndarray
    spacing_xyz: tuple[float, float, float]
    origin_xyz: tuple[float, float, float]
    direction: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        if self.array.ndim != 3:
            raise ValueError(f"NIfTI volumes must be 3D, got array shape {self.array.shape}")
        if len(self.spacing_xyz) != 3 or any(value <= 0 for value in self.spacing_xyz):
            raise ValueError(f"spacing must contain three positive values, got {self.spacing_xyz}")
        if len(self.origin_xyz) != 3 or len(self.direction) != 9:
            raise ValueError("origin must have 3 values and direction must have 9 values")


def to_sitk(volume: NiftiVolume) -> sitk.Image:
    """Convert a volume to SimpleITK without writing it to disk."""
    image = sitk.GetImageFromArray(volume.array)
    image.SetSpacing(volume.spacing_xyz)
    image.SetOrigin(volume.origin_xyz)
    image.SetDirection(volume.direction)
    return image


def from_sitk(image: sitk.Image) -> NiftiVolume:
    """Convert a SimpleITK image to a channel-free ``(z, y, x)`` volume."""
    return NiftiVolume(
        array=sitk.GetArrayFromImage(image),
        spacing_xyz=tuple(float(value) for value in image.GetSpacing()),
        origin_xyz=tuple(float(value) for value in image.GetOrigin()),
        direction=tuple(float(value) for value in image.GetDirection()),
    )


def read_nifti(path: Path) -> NiftiVolume:
    """Read exactly one existing NIfTI file; no directory scanning occurs."""
    if not path.is_file():
        raise FileNotFoundError(f"NIfTI file does not exist: {path}")
    return from_sitk(sitk.ReadImage(str(path)))


def write_nifti(path: Path, volume: NiftiVolume) -> None:
    """Write a caller-selected path, intended only for synthetic test fixtures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(to_sitk(volume), str(path))
