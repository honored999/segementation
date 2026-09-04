"""Opt-in, label-safe paired 2D augmentations with explicit local settings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AugmentationConfig:
    horizontal_flip_probability: float = 0.0
    vertical_flip_probability: float = 0.0
    intensity_scale_range: tuple[float, float] = (1.0, 1.0)

    def __post_init__(self) -> None:
        if not 0.0 <= self.horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal_flip_probability must be in [0, 1]")
        if not 0.0 <= self.vertical_flip_probability <= 1.0:
            raise ValueError("vertical_flip_probability must be in [0, 1]")
        low, high = self.intensity_scale_range
        if low <= 0 or high < low:
            raise ValueError("intensity_scale_range must be positive and ordered")


def augment_slice(
    image: np.ndarray,
    label: np.ndarray,
    rng: np.random.Generator,
    config: AugmentationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same geometry to image/label and intensity only to image."""
    if image.ndim != 2 or label.ndim != 2 or image.shape != label.shape:
        raise ValueError("image and label must be matched 2D arrays")
    augmented_image, augmented_label = image.copy(), label.copy()
    if rng.random() < config.horizontal_flip_probability:
        augmented_image, augmented_label = augmented_image[:, ::-1], augmented_label[:, ::-1]
    if rng.random() < config.vertical_flip_probability:
        augmented_image, augmented_label = augmented_image[::-1, :], augmented_label[::-1, :]
    low, high = config.intensity_scale_range
    augmented_image = (augmented_image * rng.uniform(low, high)).astype(image.dtype, copy=False)
    return np.ascontiguousarray(augmented_image), np.ascontiguousarray(augmented_label)
