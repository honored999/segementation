"""Plan-driven, in-memory preprocessing for a single requested volume."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from standalone_nnunet2d.data.nifti_io import NiftiVolume, from_sitk, to_sitk


def z_score_normalize(image: np.ndarray) -> np.ndarray:
    """Apply full-image Z-score normalization required by ``use_mask_for_norm=false``."""
    if not np.isfinite(image).all():
        raise ValueError("image contains non-finite values")
    image_float = image.astype(np.float32, copy=False)
    standard_deviation = float(image_float.std())
    if standard_deviation == 0.0:
        return np.zeros_like(image_float)
    return ((image_float - image_float.mean()) / standard_deviation).astype(np.float32, copy=False)


def resample_inplane(
    volume: NiftiVolume,
    target_spacing_xy: tuple[float, float],
    *,
    is_segmentation: bool,
) -> NiftiVolume:
    """Resample x/y only, preserving the z spacing and number of slices.

    Images use the plan's cubic data interpolation. Segmentations use its
    linear order and are rounded to recover discrete integer label values.
    """
    if len(target_spacing_xy) != 2 or any(value <= 0 for value in target_spacing_xy):
        raise ValueError(f"target in-plane spacing must contain two positive values, got {target_spacing_xy}")
    if np.allclose(volume.spacing_xyz[:2], target_spacing_xy, rtol=0.0, atol=1e-7):
        return volume
    source = to_sitk(volume)
    old_size = source.GetSize()
    old_spacing = source.GetSpacing()
    target_size = (
        max(1, int(round(old_size[0] * old_spacing[0] / target_spacing_xy[0]))),
        max(1, int(round(old_size[1] * old_spacing[1] / target_spacing_xy[1]))),
        old_size[2],
    )
    interpolation = sitk.sitkNearestNeighbor if is_segmentation else sitk.sitkBSpline
    resampled = sitk.Resample(
        source,
        target_size,
        sitk.Transform(),
        interpolation,
        source.GetOrigin(),
        (target_spacing_xy[0], target_spacing_xy[1], old_spacing[2]),
        source.GetDirection(),
        0.0,
        source.GetPixelID(),
    )
    result = from_sitk(resampled)
    if is_segmentation:
        return NiftiVolume(
            array=result.array.astype(np.int16),
            spacing_xyz=result.spacing_xyz,
            origin_xyz=result.origin_xyz,
            direction=result.direction,
        )
    return NiftiVolume(
        array=result.array.astype(np.float32),
        spacing_xyz=result.spacing_xyz,
        origin_xyz=result.origin_xyz,
        direction=result.direction,
    )
