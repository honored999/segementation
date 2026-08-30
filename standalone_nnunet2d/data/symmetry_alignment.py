"""In-memory quasi-symmetric in-plane alignment for axial NIfTI volumes.

The estimator accepts only axis-aligned axial LPS directions.  It derives a
foreground centroid and principal axis from image intensities, then constructs
one physical output-to-input transform for SimpleITK resampling.  Labels are
never used to estimate that transform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import SimpleITK as sitk

if TYPE_CHECKING:
    from standalone_nnunet2d.data.nifti_io import NiftiVolume


_DIRECTION_TOLERANCE = 1e-6
_MIN_FOREGROUND_VOXELS = 16
_MIN_AXIS_ANISOTROPY = 0.08


class QuasiSymmetricAlignmentError(ValueError):
    """Raised when a volume cannot be aligned under the supported contract."""


@dataclass(frozen=True)
class AlignmentEstimate:
    """Image-derived physical correction and its output-to-input affine map."""

    center_xyz: tuple[float, float, float]
    reference_center_xyz: tuple[float, float, float]
    rotation_angle_radians: float
    output_to_input_matrix: tuple[float, ...]
    output_to_input_translation_xyz: tuple[float, float, float]
    foreground_voxel_count: int

    @property
    def centre_xyz(self) -> tuple[float, float, float]:
        """British-English alias for callers reporting QC values."""
        return self.center_xyz

    @property
    def angle_radians(self) -> float:
        """Short alias for the image-derived in-plane principal-axis angle."""
        return self.rotation_angle_radians


@dataclass(frozen=True)
class AlignmentResult:
    """Already-resampled data retained for in-memory alignment QC."""

    image: "NiftiVolume"
    label: "NiftiVolume | None"
    estimate: AlignmentEstimate


@dataclass(frozen=True)
class AlignmentQC:
    """Representative in-memory slices only; this helper never writes files."""

    original: np.ndarray
    aligned: np.ndarray
    mirrored: np.ndarray
    absolute_difference: np.ndarray
    center_xyz: tuple[float, float, float]
    rotation_angle_radians: float
    slice_index: int


def _direction_matrix(volume: "NiftiVolume") -> np.ndarray:
    direction = np.asarray(volume.direction, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(direction).all():
        raise QuasiSymmetricAlignmentError("unsupported direction: contains non-finite values")
    if not np.allclose(direction.T @ direction, np.eye(3), rtol=0.0, atol=_DIRECTION_TOLERANCE):
        raise QuasiSymmetricAlignmentError("unsupported direction: must be orthonormal")
    rounded = np.rint(direction)
    if not np.allclose(direction, rounded, rtol=0.0, atol=_DIRECTION_TOLERANCE):
        raise QuasiSymmetricAlignmentError(
            "unsupported direction: only axis-aligned axial LPS directions are supported; oblique orientation is rejected"
        )
    if not np.all(np.isin(rounded, (-1.0, 0.0, 1.0))):
        raise QuasiSymmetricAlignmentError("unsupported direction: expected signed axis-aligned directions")
    if not (
        np.allclose(rounded[:2, 2], 0.0, rtol=0.0, atol=_DIRECTION_TOLERANCE)
        and np.allclose(rounded[2, :2], 0.0, rtol=0.0, atol=_DIRECTION_TOLERANCE)
        and abs(rounded[2, 2]) == 1.0
    ):
        raise QuasiSymmetricAlignmentError(
            "unsupported direction: z-mixing orientation cannot preserve the axial z plane"
        )
    return rounded


def _physical_points(volume: "NiftiVolume", indices_zyx: np.ndarray, direction: np.ndarray) -> np.ndarray:
    indices_xyz = indices_zyx[:, ::-1].astype(np.float64, copy=False)
    scaled_indices = indices_xyz * np.asarray(volume.spacing_xyz, dtype=np.float64)
    return np.asarray(volume.origin_xyz, dtype=np.float64) + scaled_indices @ direction.T


def _reference_center(volume: "NiftiVolume", direction: np.ndarray) -> np.ndarray:
    size_xyz = np.asarray(volume.array.shape[::-1], dtype=np.float64)
    center_index_xyz = (size_xyz - 1.0) / 2.0
    return np.asarray(volume.origin_xyz, dtype=np.float64) + direction @ (
        center_index_xyz * np.asarray(volume.spacing_xyz, dtype=np.float64)
    )


def _boundary_values(array: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (
            array[0].ravel(),
            array[-1].ravel(),
            array[:, 0, :].ravel(),
            array[:, -1, :].ravel(),
            array[:, :, 0].ravel(),
            array[:, :, -1].ravel(),
        )
    )


def _foreground_mask(image: "NiftiVolume") -> np.ndarray:
    array = np.asarray(image.array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise QuasiSymmetricAlignmentError("image contains non-finite values")
    boundary = _boundary_values(array)
    background = float(np.median(boundary))
    dynamic_range = float(np.percentile(array, 99.0) - np.percentile(array, 1.0))
    background_mad = float(np.median(np.abs(boundary - background)))
    if not np.isfinite(dynamic_range) or dynamic_range <= np.finfo(np.float64).eps:
        raise QuasiSymmetricAlignmentError("insufficient foreground contrast for quasi-symmetric alignment")
    threshold = max(5.0 * 1.4826 * background_mad, 0.05 * dynamic_range, np.finfo(np.float64).eps)
    foreground = np.abs(array - background) > threshold
    if int(foreground.sum()) < _MIN_FOREGROUND_VOXELS:
        raise QuasiSymmetricAlignmentError("insufficient foreground voxels for quasi-symmetric alignment")
    return foreground


def estimate_quasi_symmetric_alignment(image: "NiftiVolume") -> AlignmentEstimate:
    """Estimate a rigid XY correction from the image alone.

    The returned affine follows SimpleITK's resampler convention: it maps an
    output physical point to the corresponding input physical point.
    """
    direction = _direction_matrix(image)
    foreground = _foreground_mask(image)
    points = _physical_points(image, np.argwhere(foreground), direction)
    center = points.mean(axis=0)
    if not np.isfinite(center).all():
        raise QuasiSymmetricAlignmentError("non-finite foreground centre")

    covariance = np.cov(points[:, :2], rowvar=False, bias=True)
    if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
        raise QuasiSymmetricAlignmentError("non-finite principal axis covariance")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major = float(eigenvalues[-1])
    minor = float(eigenvalues[0])
    if major <= np.finfo(np.float64).eps or (major - minor) / major < _MIN_AXIS_ANISOTROPY:
        raise QuasiSymmetricAlignmentError("degenerate principal axis for quasi-symmetric alignment")
    axis = eigenvectors[:, -1]
    angle = float(np.arctan2(axis[1], axis[0]))
    angle = float(0.5 * np.arctan2(np.sin(2.0 * angle), np.cos(2.0 * angle)))
    if not np.isfinite(angle) or abs(angle) > np.pi / 2.0 + _DIRECTION_TOLERANCE:
        raise QuasiSymmetricAlignmentError("non-finite or anomalous in-plane rotation angle")

    reference = _reference_center(image, direction)
    correction_angle = -angle
    correction = np.array(
        (
            (np.cos(correction_angle), -np.sin(correction_angle), 0.0),
            (np.sin(correction_angle), np.cos(correction_angle), 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    forward_translation = reference - correction @ center
    output_to_input = correction.T
    output_to_input_translation = -output_to_input @ forward_translation
    if not (np.isfinite(output_to_input).all() and np.isfinite(output_to_input_translation).all()):
        raise QuasiSymmetricAlignmentError("non-finite physical alignment transform")
    return AlignmentEstimate(
        center_xyz=tuple(float(value) for value in center),
        reference_center_xyz=tuple(float(value) for value in reference),
        rotation_angle_radians=angle,
        output_to_input_matrix=tuple(float(value) for value in output_to_input.ravel()),
        output_to_input_translation_xyz=tuple(float(value) for value in output_to_input_translation),
        foreground_voxel_count=int(foreground.sum()),
    )


def _to_sitk(volume: "NiftiVolume") -> sitk.Image:
    image = sitk.GetImageFromArray(volume.array)
    image.SetSpacing(volume.spacing_xyz)
    image.SetOrigin(volume.origin_xyz)
    image.SetDirection(volume.direction)
    return image


def _new_volume_like(volume: "NiftiVolume", array: np.ndarray) -> "NiftiVolume":
    return type(volume)(
        array=array,
        spacing_xyz=volume.spacing_xyz,
        origin_xyz=volume.origin_xyz,
        direction=volume.direction,
    )


def apply_quasi_symmetric_alignment(
    volume: "NiftiVolume",
    estimate: AlignmentEstimate,
    *,
    is_segmentation: bool,
) -> "NiftiVolume":
    """Apply exactly one final physical resample on the original reference grid."""
    _direction_matrix(volume)
    source = _to_sitk(volume)
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(estimate.output_to_input_matrix)
    transform.SetTranslation(estimate.output_to_input_translation_xyz)
    interpolator = sitk.sitkNearestNeighbor if is_segmentation else sitk.sitkLinear
    resampled = sitk.Resample(
        source,
        source.GetSize(),
        transform,
        interpolator,
        source.GetOrigin(),
        source.GetSpacing(),
        source.GetDirection(),
        0.0,
        source.GetPixelID(),
    )
    array = sitk.GetArrayFromImage(resampled)
    if is_segmentation:
        array = array.astype(volume.array.dtype, copy=False)
    else:
        array = array.astype(np.float32, copy=False)
    return _new_volume_like(volume, array)


def _same_geometry(first: "NiftiVolume", second: "NiftiVolume") -> bool:
    return (
        first.array.shape == second.array.shape
        and np.allclose(first.spacing_xyz, second.spacing_xyz, rtol=0.0, atol=1e-6)
        and np.allclose(first.origin_xyz, second.origin_xyz, rtol=0.0, atol=1e-6)
        and np.allclose(first.direction, second.direction, rtol=0.0, atol=1e-6)
    )


def align_case(image: "NiftiVolume", label: "NiftiVolume") -> tuple["NiftiVolume", "NiftiVolume", AlignmentEstimate]:
    """Estimate from ``image`` and apply the identical XY transform to image and label."""
    if not _same_geometry(image, label):
        raise QuasiSymmetricAlignmentError("image and label geometry mismatch before alignment")
    estimate = estimate_quasi_symmetric_alignment(image)
    return (
        apply_quasi_symmetric_alignment(image, estimate, is_segmentation=False),
        apply_quasi_symmetric_alignment(label, estimate, is_segmentation=True),
        estimate,
    )


def align_case_result(image: "NiftiVolume", label: "NiftiVolume | None" = None) -> AlignmentResult:
    """Return aligned data and its reusable estimate for callers that need QC."""
    if label is None:
        estimate = estimate_quasi_symmetric_alignment(image)
        return AlignmentResult(
            image=apply_quasi_symmetric_alignment(image, estimate, is_segmentation=False),
            label=None,
            estimate=estimate,
        )
    aligned_image, aligned_label, estimate = align_case(image, label)
    return AlignmentResult(image=aligned_image, label=aligned_label, estimate=estimate)


def _physical_left_right_mirror(slice_yx: np.ndarray, direction: tuple[float, ...]) -> np.ndarray:
    matrix = np.asarray(direction, dtype=np.float64).reshape(3, 3)
    physical_x_index_axis = int(np.argmax(np.abs(matrix[0, :2])))
    return np.flip(slice_yx, axis=1 if physical_x_index_axis == 0 else 0)


def build_alignment_qc(
    original: "NiftiVolume",
    *,
    result: AlignmentResult,
    slice_index: int | None = None,
) -> AlignmentQC:
    """Return representative alignment data without exporting a file.

    ``result`` is deliberately required so QC reuses an existing alignment
    result and never re-estimates from a label or performs extra resampling.
    """
    if original.array.shape != result.image.array.shape:
        raise QuasiSymmetricAlignmentError("QC original and aligned image shapes differ")
    index = original.array.shape[0] // 2 if slice_index is None else slice_index
    if not isinstance(index, (int, np.integer)) or not 0 <= int(index) < original.array.shape[0]:
        raise IndexError("slice index is outside the volume bounds")
    original_slice = np.asarray(original.array[int(index)]).copy()
    aligned_slice = np.asarray(result.image.array[int(index)]).copy()
    mirrored = _physical_left_right_mirror(aligned_slice, original.direction).copy()
    return AlignmentQC(
        original=original_slice,
        aligned=aligned_slice,
        mirrored=mirrored,
        absolute_difference=np.abs(aligned_slice - mirrored),
        center_xyz=result.estimate.center_xyz,
        rotation_angle_radians=result.estimate.rotation_angle_radians,
        slice_index=int(index),
    )
