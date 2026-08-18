"""Prediction-guided XY ROI calculation with explicit safety checks."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Iterable

import numpy as np

from .nifti import NiftiVolume, XYBBox


@dataclass(frozen=True)
class PredictionROI:
    x0: int
    y0: int
    x1: int
    y1: int
    fallback: bool = False

    @property
    def bbox(self) -> XYBBox:
        return self.x0, self.y0, self.x1, self.y1

    def __iter__(self) -> Iterable[object]:
        yield self.bbox
        yield self.fallback


def validate_binary_prediction(prediction: object) -> np.ndarray:
    """Validate a prediction as finite 3-D binary data and return uint8."""
    array = np.asarray(prediction)
    if array.ndim != 3:
        raise ValueError("prediction must be 3-dimensional")
    try:
        finite = np.isfinite(array).all()
    except TypeError as error:
        raise ValueError("prediction must contain finite numeric values") from error
    if not finite:
        raise ValueError("prediction must contain only finite values")
    if not np.isin(array, (0, 1)).all():
        raise ValueError("prediction must contain only 0 and 1")
    return array.astype(np.uint8, copy=True)


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _margin_xy(margin: object) -> tuple[int, int]:
    if isinstance(margin, Integral) and not isinstance(margin, bool):
        margin_value = int(margin)
        if margin_value < 0:
            raise ValueError("margin must be non-negative")
        return margin_value, margin_value
    if isinstance(margin, (tuple, list)) and len(margin) == 2:
        values = tuple(margin)
        if any(isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0 for value in values):
            raise ValueError("margin values must be non-negative integers")
        return int(values[0]), int(values[1])
    raise ValueError("margin must be a non-negative integer or an (x, y) pair")


def _expand_interval(low: int, high: int, minimum: int, limit: int) -> tuple[int, int]:
    if high - low >= minimum:
        return low, high
    deficit = minimum - (high - low)
    low = max(0, low - deficit // 2)
    high = min(limit, high + deficit - deficit // 2)
    if high - low < minimum:
        if low == 0:
            high = minimum
        else:
            low = limit - minimum
    return low, high


def compute_prediction_roi(
    prediction: np.ndarray | NiftiVolume,
    *,
    margin: int | tuple[int, int] = 0,
    min_width: int = 1,
    min_height: int = 1,
) -> PredictionROI:
    """Return one XY box from the foreground union across every z slice."""
    array = prediction.array if isinstance(prediction, NiftiVolume) else prediction
    binary = validate_binary_prediction(array)
    _, height, width = binary.shape
    margin_x, margin_y = _margin_xy(margin)
    minimum_width = _positive_integer(min_width, "min_width")
    minimum_height = _positive_integer(min_height, "min_height")
    if minimum_width > width or minimum_height > height:
        raise ValueError("minimum ROI size cannot exceed prediction dimensions")

    foreground_xy = np.any(binary == 1, axis=0)
    if not foreground_xy.any():
        return PredictionROI(0, 0, width, height, fallback=True)

    ys, xs = np.where(foreground_xy)
    x0 = max(0, int(xs.min()) - margin_x)
    y0 = max(0, int(ys.min()) - margin_y)
    x1 = min(width, int(xs.max()) + 1 + margin_x)
    y1 = min(height, int(ys.max()) + 1 + margin_y)
    x0, x1 = _expand_interval(x0, x1, minimum_width, width)
    y0, y1 = _expand_interval(y0, y1, minimum_height, height)
    return PredictionROI(x0, y0, x1, y1, fallback=False)
