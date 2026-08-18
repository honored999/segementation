"""Primitives for the prediction-guided coarse-to-fine DWI pipeline."""

from .nifti import (
    NiftiVolume,
    XYBBox,
    assert_compatible,
    crop_volume_xy,
    crop_xy,
    restore_volume_xy,
    restore_xy,
)
from .roi import PredictionROI, compute_prediction_roi, validate_binary_prediction

__all__ = [
    "NiftiVolume",
    "PredictionROI",
    "XYBBox",
    "assert_compatible",
    "compute_prediction_roi",
    "crop_volume_xy",
    "crop_xy",
    "restore_volume_xy",
    "restore_xy",
    "validate_binary_prediction",
]
