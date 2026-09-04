"""Small, deterministic slice-selection helpers without foreground oversampling."""

from __future__ import annotations

import numpy as np


def select_axial_slice(volume: np.ndarray, index: int) -> np.ndarray:
    """Return one ``(y, x)`` axial slice from a ``(z, y, x)`` volume."""
    if volume.ndim != 3:
        raise ValueError(f"expected a (z, y, x) volume, got shape {volume.shape}")
    if not 0 <= index < volume.shape[0]:
        raise IndexError(f"slice index {index} is outside [0, {volume.shape[0]})")
    return volume[index]


def central_slice_index(depth: int) -> int:
    """Choose the lower central index deterministically for the current phase."""
    if depth < 1:
        raise ValueError("volume depth must be positive")
    return (depth - 1) // 2


def select_slice_index(
    labels: np.ndarray,
    rng: np.random.Generator,
    *,
    foreground_probability: float = 0.0,
) -> int:
    """Sample a valid slice, optionally preferring non-empty label slices."""
    if labels.ndim != 3:
        raise ValueError(f"labels must be (z, y, x), got shape {labels.shape}")
    if labels.shape[0] < 1:
        raise ValueError("labels must contain at least one slice")
    if not 0.0 <= foreground_probability <= 1.0:
        raise ValueError("foreground_probability must be in [0, 1]")
    foreground_indices = np.flatnonzero(np.any(labels != 0, axis=(1, 2)))
    if foreground_indices.size and rng.random() < foreground_probability:
        return int(rng.choice(foreground_indices))
    return int(rng.integers(labels.shape[0]))
