"""Small, deterministic paired transforms used by the formal 2D pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter, rotate, zoom

from standalone_nnunet2d.training.patch_sampler import crop_or_pad


@dataclass(frozen=True)
class FormalTransformResult:
    image: np.ndarray
    label: np.ndarray


@dataclass(frozen=True)
class SpatialResult(FormalTransformResult):
    pass


def _validate_pair(image: np.ndarray, label: np.ndarray) -> None:
    if image.ndim != 2 or label.ndim != 2 or image.shape != label.shape:
        raise ValueError("image and label must be matched 2D arrays")


def _validate_probability(probability: float) -> None:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")


def _validate_range(values: tuple[float, float], name: str) -> tuple[float, float]:
    if len(values) != 2 or not np.isfinite(values).all() or values[0] > values[1]:
        raise ValueError(f"{name} must be an ordered finite pair")
    return float(values[0]), float(values[1])


def _centre(shape: tuple[int, int]) -> tuple[int, int]:
    return shape[0] // 2, shape[1] // 2


def _apply_probability(rng: np.random.Generator, probability: float) -> bool:
    _validate_probability(probability)
    return bool(rng.random() < probability)


def _rotate_pair(
    image: np.ndarray,
    label: np.ndarray,
    angle_radians: float,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        rotate(image, np.degrees(angle_radians), reshape=False, order=3, mode="constant", cval=0.0),
        rotate(label, np.degrees(angle_radians), reshape=False, order=0, mode="constant", cval=-1),
    )


def _scale_pair(
    image: np.ndarray,
    label: np.ndarray,
    scale: float,
    patch_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    scaled_image = zoom(image, scale, order=3)
    scaled_label = zoom(label, scale, order=0)
    return crop_or_pad(
        scaled_image,
        scaled_label,
        _centre(scaled_image.shape),
        patch_size,
    )


def apply_formal_spatial_transform(
    image: np.ndarray,
    label: np.ndarray,
    *,
    rng: np.random.Generator,
    initial_patch_size: tuple[int, int],
    patch_size: tuple[int, int],
    rotation_probability: float = 0.2,
    rotation_radians: tuple[float, float] = (-np.pi, np.pi),
    scale_probability: float = 0.2,
    scale_range: tuple[float, float] = (0.7, 1.4),
) -> SpatialResult:
    """Apply geometry to an initial patch and crop/pad to the final patch."""
    _validate_pair(image, label)
    rotation_radians = _validate_range(rotation_radians, "rotation_radians")
    scale_range = _validate_range(scale_range, "scale_range")
    image, label = crop_or_pad(image, label, _centre(image.shape), initial_patch_size)

    if _apply_probability(rng, rotation_probability):
        image, label = _rotate_pair(image, label, rng.uniform(*rotation_radians))
    if _apply_probability(rng, scale_probability):
        image, label = _scale_pair(image, label, rng.uniform(*scale_range), patch_size)

    image, label = crop_or_pad(image, label, _centre(image.shape), patch_size)
    return SpatialResult(
        image=np.asarray(image),
        label=np.asarray(label).astype(label.dtype, copy=False),
    )


def _as_float_image(image: np.ndarray) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError("image must be a 2D array")
    return image.astype(np.float32, copy=True)


def apply_noise(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    probability: float = 0.1,
    noise_std_range: tuple[float, float] = (0.0, 0.1),
) -> np.ndarray:
    """Add zero-mean Gaussian noise to an image with a caller-owned RNG."""
    if not _apply_probability(rng, probability):
        return np.array(image, copy=True)
    low, high = _validate_range(noise_std_range, "noise_std_range")
    if low < 0.0:
        raise ValueError("noise standard deviation must be non-negative")
    result = _as_float_image(image)
    return result + rng.normal(0.0, rng.uniform(low, high), result.shape).astype(np.float32)


def apply_blur(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    probability: float = 0.2,
    sigma_range: tuple[float, float] = (0.5, 1.0),
) -> np.ndarray:
    if not _apply_probability(rng, probability):
        return np.array(image, copy=True)
    sigma = _validate_range(sigma_range, "sigma_range")
    if sigma[0] < 0.0:
        raise ValueError("blur sigma must be non-negative")
    return gaussian_filter(_as_float_image(image), rng.uniform(*sigma)).astype(np.float32, copy=False)


def apply_brightness(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    probability: float = 0.15,
    brightness_range: tuple[float, float] = (0.75, 1.25),
) -> np.ndarray:
    if not _apply_probability(rng, probability):
        return np.array(image, copy=True)
    brightness = _validate_range(brightness_range, "brightness_range")
    return (_as_float_image(image) * rng.uniform(*brightness)).astype(np.float32, copy=False)


def apply_contrast(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    probability: float = 0.15,
    contrast_range: tuple[float, float] = (0.75, 1.25),
) -> np.ndarray:
    if not _apply_probability(rng, probability):
        return np.array(image, copy=True)
    contrast = _validate_range(contrast_range, "contrast_range")
    result = _as_float_image(image)
    mean = result.mean()
    return ((result - mean) * rng.uniform(*contrast) + mean).astype(np.float32, copy=False)


def _fit_to_shape(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    result = np.zeros(shape, dtype=np.float32)
    source_y = max((image.shape[0] - shape[0]) // 2, 0)
    source_x = max((image.shape[1] - shape[1]) // 2, 0)
    target_y = max((shape[0] - image.shape[0]) // 2, 0)
    target_x = max((shape[1] - image.shape[1]) // 2, 0)
    height = min(image.shape[0], shape[0])
    width = min(image.shape[1], shape[1])
    result[target_y : target_y + height, target_x : target_x + width] = image[
        source_y : source_y + height, source_x : source_x + width
    ]
    return result


def apply_low_resolution(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    probability: float = 0.25,
    scale_range: tuple[float, float] = (0.5, 1.0),
) -> np.ndarray:
    if not _apply_probability(rng, probability):
        return np.array(image, copy=True)
    scale = _validate_range(scale_range, "scale_range")
    if scale[0] <= 0.0 or scale[1] > 1.0:
        raise ValueError("low-resolution scale must be in (0, 1]")
    result = _as_float_image(image)
    low_shape = tuple(max(1, int(round(size * rng.uniform(*scale)))) for size in result.shape)
    low = zoom(result, (low_shape[0] / result.shape[0], low_shape[1] / result.shape[1]), order=1)
    restored = zoom(low, (result.shape[0] / low.shape[0], result.shape[1] / low.shape[1]), order=1)
    return _fit_to_shape(restored, result.shape)


def _gamma_image(image: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    result = _as_float_image(image)
    lower, upper = float(result.min()), float(result.max())
    if upper <= lower:
        return result
    normalized = (result - lower) / (upper - lower)
    return (np.power(normalized, gamma) * (upper - lower) + lower).astype(np.float32, copy=False)


def apply_gamma(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    probability: float = 0.3,
    gamma_range: tuple[float, float] = (0.7, 1.5),
) -> np.ndarray:
    if not _apply_probability(rng, probability):
        return np.array(image, copy=True)
    gamma = _validate_range(gamma_range, "gamma_range")
    return _gamma_image(image, rng.uniform(*gamma))


def apply_gamma_inversion(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    probability: float = 0.1,
    gamma_range: tuple[float, float] = (0.7, 1.5),
) -> np.ndarray:
    if not _apply_probability(rng, probability):
        return np.array(image, copy=True)
    gamma = _validate_range(gamma_range, "gamma_range")
    result = _as_float_image(image)
    return _gamma_image(-result, rng.uniform(*gamma))


def apply_mirroring(
    image: np.ndarray,
    label: np.ndarray,
    rng: np.random.Generator,
    *,
    probability: float = 0.5,
    axes: tuple[int, ...] = (0, 1),
) -> tuple[np.ndarray, np.ndarray]:
    _validate_pair(image, label)
    if not _apply_probability(rng, probability):
        return np.array(image, copy=True), np.array(label, copy=True)
    if any(axis not in (0, 1) for axis in axes):
        raise ValueError("mirroring axes must be 0 or 1")
    return (
        np.ascontiguousarray(np.flip(image, axis=axes)),
        np.ascontiguousarray(np.flip(label, axis=axes)),
    )


def make_deep_supervision_targets(
    label: np.ndarray,
    *,
    scales: Sequence[tuple[float, float]],
) -> tuple[np.ndarray, ...]:
    """Downsample discrete targets with nearest-neighbour interpolation."""
    if label.ndim != 2:
        raise ValueError("label must be a 2D array")
    targets: list[np.ndarray] = []
    for scale in scales:
        scale = _validate_range(scale, "scale")
        if scale[0] <= 0.0 or scale[1] <= 0.0:
            raise ValueError("deep-supervision scales must be positive")
        target = zoom(label, scale, order=0)
        targets.append(np.asarray(target).astype(label.dtype, copy=False))
    return tuple(targets)


def remove_label_values(
    label: np.ndarray,
    *,
    values: tuple[int, ...] = (-1,),
    replacement: int = 0,
) -> np.ndarray:
    """Replace ignored label values while preserving all other labels."""
    if label.ndim != 2:
        raise ValueError("label must be a 2D array")
    result = np.array(label, copy=True)
    result[np.isin(result, values)] = replacement
    return result
