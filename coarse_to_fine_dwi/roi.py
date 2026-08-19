"""Prediction-guided XY ROI calculation with explicit safety checks."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Iterable

import numpy as np

from .nifti import NiftiVolume, XYBBox


DEFAULT_ROI_MARGIN = 32
DEFAULT_MIN_ROI_WIDTH = 128
DEFAULT_MIN_ROI_HEIGHT = 128


@dataclass(frozen=True)
class PredictionROI:
    x0: int
    y0: int
    x1: int
    y1: int
    fallback: bool = False
    raw_prediction_bbox: XYBBox | None = None
    roi_margin: int | tuple[int, int] = DEFAULT_ROI_MARGIN
    min_roi_width: int = DEFAULT_MIN_ROI_WIDTH
    min_roi_height: int = DEFAULT_MIN_ROI_HEIGHT

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


def _fit_interval(low: int, high: int, minimum: int, limit: int) -> tuple[int, int]:
    requested = min(minimum, limit)
    if high - low >= requested:
        return low, high
    low = low - (requested - (high - low)) // 2
    high = low + requested
    if low < 0:
        low = 0
        high = requested
    if high > limit:
        high = limit
        low = limit - requested
    return low, high


def compute_prediction_roi(
    prediction: np.ndarray | NiftiVolume,
    *,
    margin: int | tuple[int, int] = DEFAULT_ROI_MARGIN,
    min_width: int = DEFAULT_MIN_ROI_WIDTH,
    min_height: int = DEFAULT_MIN_ROI_HEIGHT,
) -> PredictionROI:
    """Return one XY box from the foreground union across every z slice."""
    array = prediction.array if isinstance(prediction, NiftiVolume) else prediction
    binary = validate_binary_prediction(array)
    _, height, width = binary.shape
    margin_x, margin_y = _margin_xy(margin)
    minimum_width = _positive_integer(min_width, "min_width")
    minimum_height = _positive_integer(min_height, "min_height")
    normalized_margin: int | tuple[int, int]
    if isinstance(margin, Integral) and not isinstance(margin, bool):
        normalized_margin = int(margin)
    else:
        normalized_margin = (margin_x, margin_y)

    foreground_xy = np.any(binary == 1, axis=0)
    if not foreground_xy.any():
        return PredictionROI(
            0,
            0,
            width,
            height,
            fallback=True,
            raw_prediction_bbox=None,
            roi_margin=normalized_margin,
            min_roi_width=minimum_width,
            min_roi_height=minimum_height,
        )

    ys, xs = np.where(foreground_xy)
    raw_bbox: XYBBox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    x0 = max(0, raw_bbox[0] - margin_x)
    y0 = max(0, raw_bbox[1] - margin_y)
    x1 = min(width, raw_bbox[2] + margin_x)
    y1 = min(height, raw_bbox[3] + margin_y)
    x0, x1 = _fit_interval(x0, x1, minimum_width, width)
    y0, y1 = _fit_interval(y0, y1, minimum_height, height)
    return PredictionROI(
        x0,
        y0,
        x1,
        y1,
        fallback=False,
        raw_prediction_bbox=raw_bbox,
        roi_margin=normalized_margin,
        min_roi_width=minimum_width,
        min_roi_height=minimum_height,
    )
