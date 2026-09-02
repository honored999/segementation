"""Inference-side reconstruction of the bilateral training input contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from standalone_nnunet2d.data.dataset import (
    build_input_channels,
    prepare_dwi_adc_bilateral_images,
    read_case_images,
    resolve_channel_specs,
)
from standalone_nnunet2d.data.input_mode import InputMode
from standalone_nnunet2d.data.nifti_io import NiftiVolume, from_sitk, to_sitk
from standalone_nnunet2d.data.preprocessing import resample_inplane, z_score_normalize
from standalone_nnunet2d.data.symmetry_alignment import (
    AlignmentEstimate,
    align_case_result,
    build_bilateral_asymmetry_channels,
)


DEFAULT_TARGET_SPACING_XY = (0.4892368018627167, 0.4892368018627167)


@dataclass(frozen=True)
class BilateralAsymmetryInferenceCase:
    """Prepared C=2 model input plus the geometry required for source restoration."""

    source_image: NiftiVolume
    resampled_source_image: NiftiVolume
    aligned_image: NiftiVolume
    alignment_estimate: AlignmentEstimate
    model_input: np.ndarray
    model_volumes: tuple[NiftiVolume, NiftiVolume]
    physical_input_channels: int = 1
    effective_input_channels: int = 2


@dataclass(frozen=True)
class DwiAdcBilateralInferenceCase:
    """Prepared C=4 DWI+ADC model input and DWI restoration geometry."""

    source_image: NiftiVolume
    resampled_source_image: NiftiVolume
    aligned_image: NiftiVolume
    alignment_estimate: AlignmentEstimate
    model_input: np.ndarray
    model_volumes: tuple[NiftiVolume, ...]
    physical_input_channels: int = 2
    effective_input_channels: int = 4


def prepare_bilateral_asymmetry_volume(
    source_image: NiftiVolume,
    *,
    target_spacing_xy: tuple[float, float] = DEFAULT_TARGET_SPACING_XY,
) -> BilateralAsymmetryInferenceCase:
    """Build the training-equivalent C=2 input from one physical DWI volume.

    The sequence is image-only: XY resample, quasi-symmetric alignment, DWI
    Z-score normalization, then anatomical LR difference.
    """
    resampled = resample_inplane(source_image, target_spacing_xy, is_segmentation=False)
    result = align_case_result(resampled)
    normalized = NiftiVolume(
        z_score_normalize(result.image.array),
        result.image.spacing_xyz,
        result.image.origin_xyz,
        result.image.direction,
    )
    model_input = build_bilateral_asymmetry_channels(normalized)
    model_volumes = (
        normalized,
        NiftiVolume(model_input[1], normalized.spacing_xyz, normalized.origin_xyz, normalized.direction),
    )
    return BilateralAsymmetryInferenceCase(
        source_image=source_image,
        resampled_source_image=resampled,
        aligned_image=result.image,
        alignment_estimate=result.estimate,
        model_input=model_input,
        model_volumes=model_volumes,
    )


def prepare_bilateral_asymmetry_case(
    raw_root: Path,
    case_id: str,
    *,
    target_spacing_xy: tuple[float, float] = DEFAULT_TARGET_SPACING_XY,
) -> BilateralAsymmetryInferenceCase:
    """Read exactly physical DWI channel 0 and prepare its derived C=2 input."""
    channel_specs = resolve_channel_specs(raw_root)
    if len(channel_specs) != 1:
        raise ValueError(
            "bilateral_asymmetry_channel requires exactly one physical DWI channel; "
            f"found {len(channel_specs)} declared channels"
        )
    source_image, = read_case_images(raw_root, case_id)
    return prepare_bilateral_asymmetry_volume(source_image, target_spacing_xy=target_spacing_xy)


def prepare_dwi_adc_bilateral_case(
    raw_root: Path,
    case_id: str,
    *,
    target_spacing_xy: tuple[float, float] = DEFAULT_TARGET_SPACING_XY,
) -> DwiAdcBilateralInferenceCase:
    """Prepare the training-equivalent C=4 DWI+ADC bilateral input.

    DWI and ADC are resampled to the same target spacing, one alignment is
    estimated from the resampled DWI, and that image-only transform is applied
    with linear interpolation to both modalities before separate normalization.
    """
    channel_specs = resolve_channel_specs(raw_root)
    expected_specs = ((0, "DWI"), (1, "ADC"))
    if channel_specs != expected_specs:
        raise ValueError(
            "dwi_adc_bilateral input requires exactly declared channels "
            f"{expected_specs}; found {channel_specs}"
        )

    dwi, adc = read_case_images(raw_root, case_id)
    prepared = prepare_dwi_adc_bilateral_images(
        dwi, adc, target_spacing_xy=target_spacing_xy
    )
    normalized_modalities = {
        "DWI": prepared.normalized_dwi,
        "ADC": prepared.normalized_adc,
    }
    model_input = build_input_channels(
        normalized_modalities, InputMode.DWI_ADC_BILATERAL
    )
    model_volumes = tuple(
        NiftiVolume(
            model_input[index],
            prepared.aligned_dwi.spacing_xyz,
            prepared.aligned_dwi.origin_xyz,
            prepared.aligned_dwi.direction,
        )
        for index in range(model_input.shape[0])
    )
    return DwiAdcBilateralInferenceCase(
        source_image=dwi,
        resampled_source_image=prepared.resampled_dwi,
        aligned_image=prepared.aligned_dwi,
        alignment_estimate=prepared.alignment_estimate,
        model_input=model_input,
        model_volumes=model_volumes,
    )


def _alignment_inverse(estimate: AlignmentEstimate) -> sitk.Transform:
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(estimate.output_to_input_matrix)
    transform.SetTranslation(estimate.output_to_input_translation_xyz)
    return transform.GetInverse()


def _resample_to_reference(source: NiftiVolume, reference: NiftiVolume, transform: sitk.Transform) -> NiftiVolume:
    restored = sitk.Resample(
        to_sitk(source),
        to_sitk(reference),
        transform,
        sitk.sitkNearestNeighbor,
        0.0,
        sitk.sitkUnknown,
    )
    return from_sitk(restored)


def restore_bilateral_asymmetry_prediction(
    prepared: BilateralAsymmetryInferenceCase | DwiAdcBilateralInferenceCase,
    prediction: np.ndarray,
) -> np.ndarray:
    """Map aligned/resampled binary labels back to the original source grid."""
    prediction_array = np.asarray(prediction)
    if prediction_array.shape != prepared.aligned_image.array.shape:
        raise ValueError(
            "bilateral prediction shape must match aligned model input: "
            f"expected {prepared.aligned_image.array.shape}, got {prediction_array.shape}"
        )
    aligned_prediction = NiftiVolume(
        prediction_array.astype(np.uint8, copy=False),
        prepared.aligned_image.spacing_xyz,
        prepared.aligned_image.origin_xyz,
        prepared.aligned_image.direction,
    )
    resampled_prediction = _resample_to_reference(
        aligned_prediction,
        prepared.resampled_source_image,
        _alignment_inverse(prepared.alignment_estimate),
    )
    source_prediction = _resample_to_reference(
        resampled_prediction,
        prepared.source_image,
        sitk.Transform(3, sitk.sitkIdentity),
    )
    return source_prediction.array.astype(np.uint8, copy=False)
