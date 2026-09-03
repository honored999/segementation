"""Read-only, on-demand Dataset501 fixed-fold 2D slice dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from standalone_nnunet2d.data.augmentation import AugmentationConfig, augment_slice
from standalone_nnunet2d.data.input_mode import InputMode, InputSpec, input_spec
from standalone_nnunet2d.data.nifti_io import NiftiVolume, read_nifti
from standalone_nnunet2d.data.preprocessing import resample_inplane, z_score_normalize
from standalone_nnunet2d.data.sampling import central_slice_index, select_axial_slice, select_slice_index
from standalone_nnunet2d.data.symmetry_alignment import (
    AlignmentEstimate,
    align_case,
    apply_quasi_symmetric_alignment,
    bilateral_difference,
    build_bilateral_asymmetry_channels,
    estimate_quasi_symmetric_alignment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS_PATH = PROJECT_ROOT / "reference" / "splits_final.json"
SplitName = Literal["train", "val"]
ChannelSpec = tuple[int, str]


@dataclass(frozen=True)
class DwiAdcBilateralImagePreparation:
    """Shared image-only preparation for the DWI+ADC C=4 input contract."""

    resampled_dwi: NiftiVolume
    resampled_adc: NiftiVolume
    aligned_dwi: NiftiVolume
    aligned_adc: NiftiVolume
    normalized_dwi: NiftiVolume
    normalized_adc: NiftiVolume
    alignment_estimate: AlignmentEstimate


def prepare_dwi_adc_bilateral_images(
    dwi: NiftiVolume,
    adc: NiftiVolume,
    *,
    target_spacing_xy: tuple[float, float],
) -> DwiAdcBilateralImagePreparation:
    """Prepare DWI and ADC once for both dataset and inference callers.

    This image-only primitive owns the common XY resample, DWI-derived
    alignment, shared transform application, and separate modality
    normalization.  A caller may apply the returned estimate to a label, but
    labels are never used by this function.
    """
    geometry_mismatch = _geometry_mismatch_reason(dwi, adc)
    if geometry_mismatch is not None:
        raise ValueError(
            "DWI and ADC geometry mismatch before resampling: "
            f"{geometry_mismatch}"
        )
    resampled_dwi = resample_inplane(dwi, target_spacing_xy, is_segmentation=False)
    resampled_adc = resample_inplane(adc, target_spacing_xy, is_segmentation=False)
    if resampled_adc.array.shape != resampled_dwi.array.shape:
        raise ValueError(
            "DWI and ADC resampled shapes must match for bilateral input: "
            f"DWI {resampled_dwi.array.shape}, ADC {resampled_adc.array.shape}"
        )

    estimate = estimate_quasi_symmetric_alignment(resampled_dwi)
    aligned_dwi = apply_quasi_symmetric_alignment(
        resampled_dwi, estimate, is_segmentation=False
    )
    aligned_adc = apply_quasi_symmetric_alignment(
        resampled_adc, estimate, is_segmentation=False
    )
    normalized_dwi = NiftiVolume(
        z_score_normalize(aligned_dwi.array),
        aligned_dwi.spacing_xyz,
        aligned_dwi.origin_xyz,
        aligned_dwi.direction,
    )
    normalized_adc = NiftiVolume(
        z_score_normalize(aligned_adc.array),
        aligned_adc.spacing_xyz,
        aligned_adc.origin_xyz,
        aligned_adc.direction,
    )
    return DwiAdcBilateralImagePreparation(
        resampled_dwi=resampled_dwi,
        resampled_adc=resampled_adc,
        aligned_dwi=aligned_dwi,
        aligned_adc=aligned_adc,
        normalized_dwi=normalized_dwi,
        normalized_adc=normalized_adc,
        alignment_estimate=estimate,
    )


def resolve_channel_specs(raw_root: Path) -> tuple[ChannelSpec, ...]:
    """Resolve the declared raw-data channels, with a legacy C=1 fallback."""
    dataset_path = raw_root.resolve() / "dataset.json"
    if not dataset_path.is_file():
        return ((0, "legacy_single_channel"),)
    with dataset_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if "channel_names" not in metadata:
        return ((0, "legacy_single_channel"),)
    channel_names = metadata["channel_names"]
    if not isinstance(channel_names, dict) or not channel_names:
        raise ValueError("dataset.json channel_names must be a non-empty object")
    try:
        channel_indices = [int(key) for key in channel_names]
    except (TypeError, ValueError) as error:
        raise ValueError("dataset.json channel_names keys must be numeric") from error
    expected_indices = list(range(len(channel_indices)))
    if sorted(channel_indices) != expected_indices:
        raise ValueError(
            "dataset.json channel_names keys must be exactly consecutive "
            f"0..{len(channel_indices) - 1}; expected {expected_indices}, "
            f"found {sorted(channel_indices)}"
        )
    if not all(isinstance(name, str) and name for name in channel_names.values()):
        raise ValueError("dataset.json channel_names values must be non-empty strings")
    names_by_index = {int(key): name for key, name in channel_names.items()}
    return tuple((index, names_by_index[index]) for index in expected_indices)


def resolve_input_channels(raw_root: Path, *, bilateral_asymmetry_channel: bool = False) -> int:
    """Return physical plus explicitly requested derived input channels."""
    physical_input_channels = len(resolve_channel_specs(raw_root))
    if bilateral_asymmetry_channel:
        if physical_input_channels != 1:
            raise ValueError(
                "bilateral_asymmetry_channel requires exactly one physical DWI channel; "
                f"found {physical_input_channels} declared channels"
            )
        return 2
    return physical_input_channels


def build_input_channels(
    normalized_modalities: Mapping[str, NiftiVolume],
    mode: InputMode | str,
) -> np.ndarray:
    """Build an ordered model-input stack from already normalized modalities."""
    spec = input_spec(mode)
    missing = [name for name in spec.physical_modalities if name not in normalized_modalities]
    if missing:
        raise ValueError(f"input mode {mode!r} is missing normalized modalities: {missing}")
    reference = normalized_modalities[spec.physical_modalities[0]]
    channels: list[np.ndarray] = []
    for recipe in spec.channel_recipes:
        if recipe == "DWI":
            channels.append(np.asarray(normalized_modalities["DWI"].array, dtype=np.float32))
        elif recipe == "ADC":
            channels.append(np.asarray(normalized_modalities["ADC"].array, dtype=np.float32))
        elif recipe == "DWI_LR_ABS_DIFF":
            channels.append(bilateral_difference(normalized_modalities["DWI"], mode="absolute"))
        elif recipe == "DWI_LR_SIGNED_DIFF":
            channels.append(bilateral_difference(normalized_modalities["DWI"], mode="signed"))
        elif recipe == "ADC_LR_SIGNED_DIFF":
            channels.append(bilateral_difference(normalized_modalities["ADC"], mode="signed"))
        else:
            raise ValueError(f"unsupported input channel recipe: {recipe!r}")
    if any(channel.shape != reference.array.shape for channel in channels):
        raise ValueError("normalized input channel geometry is inconsistent")
    return np.stack(channels, axis=0).astype(np.float32, copy=False)


def _mode_for_dataset(
    raw_root: Path,
    *,
    input_mode: InputMode | str | None,
    bilateral_asymmetry_channel: bool,
) -> tuple[InputMode | None, InputSpec | None]:
    if input_mode is None:
        if not bilateral_asymmetry_channel:
            return None, None
        resolved = InputMode.DWI_BILATERAL
    else:
        try:
            resolved = input_mode if isinstance(input_mode, InputMode) else InputMode(input_mode)
        except (TypeError, ValueError) as error:
            raise ValueError(f"unsupported input mode: {input_mode!r}") from error
        if bilateral_asymmetry_channel and resolved is not InputMode.DWI_BILATERAL:
            raise ValueError(
                "bilateral_asymmetry_channel conflicts with input_mode; "
                "use input_mode='dwi_bilateral'"
            )
    spec = input_spec(resolved)
    declared = resolve_channel_specs(raw_root)
    expected = tuple(enumerate(spec.physical_modalities))
    if input_mode is not None and declared != expected:
        raise ValueError(
            f"input mode {resolved.value} requires exact channel declaration {expected}; "
            f"found {declared}"
        )
    return resolved, spec


def _channel_path(raw_root: Path, case_id: str, channel_index: int) -> Path:
    return raw_root.resolve() / "imagesTr" / f"{case_id}_{channel_index:04d}.nii.gz"


def read_case_images(raw_root: Path, case_id: str) -> tuple[NiftiVolume, ...]:
    """Read every declared modality for one case in declaration order."""
    images: list[NiftiVolume] = []
    for channel_index, channel_name in resolve_channel_specs(raw_root):
        path = _channel_path(raw_root, case_id, channel_index)
        if not path.is_file():
            raise FileNotFoundError(
                f"case {case_id} channel {channel_index} ({channel_name}) is missing: "
                f"expected declared channel file {path}"
            )
        images.append(read_nifti(path))
    return tuple(images)


def load_fold_cases(fold: int, split: SplitName) -> tuple[str, ...]:
    """Read case IDs from the supplied split file without random re-splitting."""
    if not 0 <= fold < 5:
        raise ValueError(f"fold must be in [0, 5), got {fold}")
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    with SPLITS_PATH.open(encoding="utf-8") as handle:
        folds = json.load(handle)
    if len(folds) != 5:
        raise ValueError(f"expected 5 supplied folds, found {len(folds)}")
    cases = folds[fold][split]
    if not isinstance(cases, list) or not all(isinstance(case_id, str) for case_id in cases):
        raise ValueError(f"fold {fold} has invalid {split} case IDs")
    return tuple(cases)


def validate_raw_root(raw_root: Path) -> Path:
    """Verify only the two required child directories; never create them."""
    root = raw_root.resolve()
    missing = [name for name in ("imagesTr", "labelsTr") if not (root / name).is_dir()]
    if missing:
        raise FileNotFoundError(f"raw Dataset501 root is missing required directories: {', '.join(missing)} under {root}")
    return root


def _same_geometry(image: NiftiVolume, label: NiftiVolume) -> bool:
    return (
        image.array.shape == label.array.shape
        and np.allclose(image.spacing_xyz, label.spacing_xyz, rtol=0.0, atol=1e-6)
        and np.allclose(image.origin_xyz, label.origin_xyz, rtol=0.0, atol=1e-6)
        and np.allclose(image.direction, label.direction, rtol=0.0, atol=1e-6)
    )


def _geometry_mismatch_reason(expected: NiftiVolume, actual: NiftiVolume) -> str | None:
    if expected.array.shape != actual.array.shape:
        return f"shape expected {expected.array.shape}, got {actual.array.shape}"
    if not np.allclose(expected.spacing_xyz, actual.spacing_xyz, rtol=0.0, atol=1e-6):
        return f"spacing expected {expected.spacing_xyz}, got {actual.spacing_xyz}"
    if not np.allclose(expected.origin_xyz, actual.origin_xyz, rtol=0.0, atol=1e-6):
        return f"origin expected {expected.origin_xyz}, got {actual.origin_xyz}"
    if not np.allclose(expected.direction, actual.direction, rtol=0.0, atol=1e-6):
        return "direction does not match the reference geometry"
    return None


class StrokeSliceDataset(Dataset[tuple[Tensor, Tensor]]):
    """One deterministic axial slice per requested case, loaded only on indexing."""

    def __init__(
        self,
        raw_root: Path,
        *,
        fold: int,
        split: SplitName,
        case_ids: tuple[str, ...] | None = None,
        target_spacing_xy: tuple[float, float] = (0.4892368018627167, 0.4892368018627167),
        rng: np.random.Generator | None = None,
        foreground_probability: float = 0.0,
        augmentation_config: AugmentationConfig | None = None,
        bilateral_asymmetry_channel: bool = False,
        input_mode: InputMode | str | None = None,
    ) -> None:
        self.raw_root = validate_raw_root(raw_root)
        self.channel_specs = resolve_channel_specs(self.raw_root)
        self.physical_input_channels = len(self.channel_specs)
        self.bilateral_asymmetry_channel = bilateral_asymmetry_channel
        self.input_mode, self.input_spec = _mode_for_dataset(
            self.raw_root,
            input_mode=input_mode,
            bilateral_asymmetry_channel=bilateral_asymmetry_channel,
        )
        if self.input_mode is None:
            self.derived_input_channels = 0
            self.input_channels = self.physical_input_channels
        else:
            assert self.input_spec is not None
            self.derived_input_channels = self.input_spec.effective_input_channels - self.physical_input_channels
            self.input_channels = self.input_spec.effective_input_channels
        if input_mode is None and bilateral_asymmetry_channel:
            self.input_channels = resolve_input_channels(
                self.raw_root,
                bilateral_asymmetry_channel=True,
            )
        allowed_case_ids = load_fold_cases(fold, split)
        self.case_ids = case_ids if case_ids is not None else allowed_case_ids
        if not self.case_ids:
            raise ValueError("case_ids must not be empty")
        outside_split = set(self.case_ids) - set(allowed_case_ids)
        if outside_split:
            raise ValueError(f"case IDs are not members of fold {fold} {split}: {sorted(outside_split)}")
        self.fold = fold
        self.split = split
        self.target_spacing_xy = target_spacing_xy
        if not 0.0 <= foreground_probability <= 1.0:
            raise ValueError("foreground_probability must be in [0, 1]")
        self.rng = rng or np.random.default_rng()
        self.foreground_probability = foreground_probability
        self.augmentation_config = augmentation_config or AugmentationConfig()

    def __len__(self) -> int:
        return len(self.case_ids)

    def load_case(self, case_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Read and process one image/label pair without retaining a cache."""
        if case_id not in self.case_ids:
            raise ValueError(f"case {case_id!r} is not available in this dataset instance")
        label = read_nifti(self.raw_root / "labelsTr" / f"{case_id}.nii.gz")
        images = read_case_images(self.raw_root, case_id)
        if self.input_mode is InputMode.DWI_ADC_BILATERAL:
            processed_label = resample_inplane(label, self.target_spacing_xy, is_segmentation=True)
            for (channel_index, channel_name), modality_name in zip(self.channel_specs, ("DWI", "ADC")):
                image = images[channel_index]
                reason = _geometry_mismatch_reason(label, image)
                if reason is not None:
                    raise ValueError(
                        f"case {case_id} channel {channel_index} ({channel_name}) geometry "
                        f"mismatch against label: {reason}"
                    )
            prepared = prepare_dwi_adc_bilateral_images(
                images[0], images[1], target_spacing_xy=self.target_spacing_xy
            )
            for (channel_index, channel_name), processed_image in zip(
                self.channel_specs,
                (prepared.resampled_dwi, prepared.resampled_adc),
            ):
                if processed_image.array.shape != processed_label.array.shape:
                    raise ValueError(
                        f"case {case_id} channel {channel_index} ({channel_name}) "
                        "resampled shape mismatch against label: "
                        f"image {processed_image.array.shape}, label {processed_label.array.shape}"
                    )
            aligned_label = apply_quasi_symmetric_alignment(
                processed_label, prepared.alignment_estimate, is_segmentation=True
            )
            normalized_modalities = {
                "DWI": prepared.normalized_dwi,
                "ADC": prepared.normalized_adc,
            }
            return build_input_channels(normalized_modalities, self.input_mode), aligned_label.array
        if self.input_mode is InputMode.DWI_BILATERAL:
            channel_index, channel_name = self.channel_specs[0]
            image = images[0]
            reason = _geometry_mismatch_reason(label, image)
            if reason is not None:
                raise ValueError(
                    f"case {case_id} channel {channel_index} ({channel_name}) geometry "
                    f"mismatch against label: {reason}"
                )
            processed_image = resample_inplane(image, self.target_spacing_xy, is_segmentation=False)
            processed_label = resample_inplane(label, self.target_spacing_xy, is_segmentation=True)
            if processed_image.array.shape != processed_label.array.shape:
                raise ValueError(
                    f"case {case_id} channel {channel_index} ({channel_name}) "
                    "resampled shape mismatch against label: "
                    f"image {processed_image.array.shape}, label {processed_label.array.shape}"
                )
            aligned_image, aligned_label, _ = align_case(processed_image, processed_label)
            normalized_dwi = z_score_normalize(aligned_image.array)
            normalized_volume = NiftiVolume(
                normalized_dwi,
                aligned_image.spacing_xyz,
                aligned_image.origin_xyz,
                aligned_image.direction,
            )
            return build_bilateral_asymmetry_channels(normalized_volume), aligned_label.array
        processed_label = resample_inplane(label, self.target_spacing_xy, is_segmentation=True)
        processed_images: list[np.ndarray] = []
        for channel_index, channel_name in self.channel_specs:
            image = images[channel_index]
            reason = _geometry_mismatch_reason(label, image)
            if reason is not None:
                raise ValueError(
                    f"case {case_id} channel {channel_index} ({channel_name}) geometry "
                    f"mismatch against label: {reason}"
                )
            processed_image = resample_inplane(image, self.target_spacing_xy, is_segmentation=False)
            if processed_image.array.shape != processed_label.array.shape:
                raise ValueError(
                    f"case {case_id} channel {channel_index} ({channel_name}) "
                    "resampled shape mismatch against label: "
                    f"image {processed_image.array.shape}, label {processed_label.array.shape}"
                )
            processed_images.append(z_score_normalize(processed_image.array))
        return np.stack(processed_images, axis=0), processed_label.array

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        case_id = self.case_ids[index]
        image, label = self.load_case(case_id)
        slice_index = (
            central_slice_index(image.shape[1])
            if self.foreground_probability == 0.0
            else select_slice_index(label, self.rng, foreground_probability=self.foreground_probability)
        )
        image_slice = image[:, slice_index]
        label_slice = label[slice_index]
        if self.input_channels == 1:
            image_slice, label_slice = augment_slice(
                image_slice[0],
                label_slice,
                self.rng,
                self.augmentation_config,
            )
            image_slice = image_slice[None]
        else:
            image_slice, label_slice = _augment_multichannel_slice(
                image_slice,
                label_slice,
                self.rng,
                self.augmentation_config,
            )
        return torch.from_numpy(image_slice).float(), torch.from_numpy(label_slice).long()


def _augment_multichannel_slice(
    image: np.ndarray,
    label: np.ndarray,
    rng: np.random.Generator,
    config: AugmentationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if image.ndim != 3 or label.ndim != 2 or image.shape[1:] != label.shape:
        raise ValueError("multichannel image and label must have shapes (C, H, W) and (H, W)")
    augmented_image, augmented_label = image.copy(), label.copy()
    if rng.random() < config.horizontal_flip_probability:
        augmented_image, augmented_label = augmented_image[:, :, ::-1], augmented_label[:, ::-1]
    if rng.random() < config.vertical_flip_probability:
        augmented_image, augmented_label = augmented_image[:, ::-1, :], augmented_label[::-1, :]
    low, high = config.intensity_scale_range
    scales = rng.uniform(low, high, size=augmented_image.shape[0]).astype(np.float32)
    augmented_image = augmented_image * scales[:, None, None]
    return np.ascontiguousarray(augmented_image), np.ascontiguousarray(augmented_label)
