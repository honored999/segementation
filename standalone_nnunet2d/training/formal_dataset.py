"""Fixed-fold 2D patch dataset for the separately configured formal Trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from standalone_nnunet2d.data.dataset import SplitName, StrokeSliceDataset
from standalone_nnunet2d.data.input_mode import InputMode
from standalone_nnunet2d.training.batch_sampler import PatchRequest
from standalone_nnunet2d.training.official_augmentation import apply_official_2d_batchgeneratorsv2
from standalone_nnunet2d.training.patch_sampler import crop_or_pad, sample_patch_center


def resolve_input_mode(
    input_mode: InputMode | str | None,
    *,
    bilateral_asymmetry_channel: bool = False,
) -> InputMode | None:
    """Resolve the explicit mode and retain the legacy flag as a compatibility alias."""
    if input_mode is None:
        return InputMode.DWI_BILATERAL if bilateral_asymmetry_channel else None
    try:
        resolved = input_mode if isinstance(input_mode, InputMode) else InputMode(input_mode)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unsupported input mode: {input_mode!r}") from error
    if bilateral_asymmetry_channel and resolved is not InputMode.DWI_BILATERAL:
        raise ValueError(
            "bilateral_asymmetry_channel conflicts with input_mode; "
            "use input_mode='dwi_bilateral'"
        )
    return resolved


class FormalPatchDataset(StrokeSliceDataset):
    """Read one fixed-fold case on demand and return one random 2D patch."""

    def __init__(
        self,
        raw_root: Path,
        *,
        fold: int,
        split: SplitName,
        case_ids: tuple[str, ...] | None = None,
        patch_size: tuple[int, int] = (512, 512),
        use_mask_for_norm: tuple[bool, ...] = (False,),
        oversample_foreground_percent: float = 0.33,
        rng: np.random.Generator | None = None,
        augment: bool = True,
        patch_request: PatchRequest | None = None,
        input_mode: InputMode | str | None = None,
        bilateral_asymmetry_channel: bool = False,
    ) -> None:
        if len(patch_size) != 2 or any(size <= 0 for size in patch_size):
            raise ValueError("patch_size must contain two positive values")
        if not 0.0 <= oversample_foreground_percent <= 1.0:
            raise ValueError("oversample_foreground_percent must be in [0, 1]")
        self.patch_size = patch_size
        self.oversample_foreground_percent = oversample_foreground_percent
        self.patch_rng = rng or np.random.default_rng()
        self.augment = augment
        if patch_request is not None and case_ids is None:
            case_ids = (patch_request.case_id,)
        self.patch_request = patch_request
        resolved_input_mode = resolve_input_mode(
            input_mode,
            bilateral_asymmetry_channel=bilateral_asymmetry_channel,
        )
        legacy_bilateral = resolved_input_mode is InputMode.DWI_BILATERAL
        has_dataset_metadata = (Path(raw_root).resolve() / "dataset.json").is_file()
        super().__init__(
            raw_root,
            fold=fold,
            split=split,
            case_ids=case_ids,
            rng=self.patch_rng,
            foreground_probability=0.0,
            bilateral_asymmetry_channel=legacy_bilateral,
            input_mode=(
                None
                if legacy_bilateral and not has_dataset_metadata
                else resolved_input_mode
            ),
        )
        self.use_mask_for_norm = tuple(use_mask_for_norm)
        if len(self.use_mask_for_norm) == 1:
            self.use_mask_for_norm *= self.input_channels
        elif len(self.use_mask_for_norm) != self.input_channels:
            raise ValueError(
                "use_mask_for_norm must contain one value per input channel: "
                f"expected {self.input_channels}, got {len(self.use_mask_for_norm)}"
            )
        if self.patch_request is not None and self.patch_request.case_id not in self.case_ids:
            raise ValueError(f"patch request case {self.patch_request.case_id!r} is not in this dataset")

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image, label = self.load_case(self.case_ids[index])
        if self.patch_request is None:
            z_index, force_foreground = self._select_z_index(label)
            center = None
        else:
            if self.patch_request.case_id != self.case_ids[index]:
                raise ValueError("patch request case does not match the requested dataset item")
            z_index = self.patch_request.z_index
            if not 0 <= z_index < label.shape[0]:
                raise ValueError("patch request z_index is outside the loaded label volume")
            force_foreground = self.patch_request.force_foreground
            center = self.patch_request.center_yx
        image_slice, label_slice = image[:, z_index], label[z_index]
        if center is None:
            center = sample_patch_center(
                label_slice,
                self.patch_rng,
                oversample_foreground_percent=1.0 if force_foreground else 0.0,
            )
        image_patch, label_patch = _crop_or_pad_channels(image_slice, label_slice, center, self.patch_size)
        if self.augment:
            seed = int(self.patch_rng.integers(0, 2**32 - 1))
            image_for_augmentation = image_patch[0] if self.input_channels == 1 else image_patch
            image_for_augmentation, label_patch = apply_official_2d_batchgeneratorsv2(
                image_for_augmentation,
                label_patch,
                patch_size=self.patch_size,
                use_mask_for_norm=self.use_mask_for_norm,
                seed=seed,
            )
            image_patch = (
                image_for_augmentation[None]
                if image_for_augmentation.ndim == 2
                else image_for_augmentation
            )
        label_patch = np.where(label_patch < 0, 0, label_patch).astype(np.int64, copy=False)
        if not np.isin(label_patch, (0, 1)).all():
            raise ValueError("formal patch labels must contain only 0 and 1")
        return torch.from_numpy(image_patch).float(), torch.from_numpy(label_patch).long()

    def _select_z_index(self, label: np.ndarray) -> tuple[int, bool]:
        foreground_z = np.flatnonzero(label.sum(axis=(1, 2)) > 0)
        force_foreground = len(foreground_z) > 0 and self.patch_rng.random() < self.oversample_foreground_percent
        z_index = int(self.patch_rng.choice(foreground_z)) if force_foreground else int(self.patch_rng.integers(label.shape[0]))
        return z_index, force_foreground


def _crop_or_pad_channels(
    image: np.ndarray,
    label: np.ndarray,
    center: tuple[int, int],
    patch_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    if image.ndim != 3 or label.ndim != 2 or image.shape[1:] != label.shape:
        raise ValueError("image and label must have shapes (C, H, W) and (H, W)")
    first_patch, label_patch = crop_or_pad(image[0], label, center, patch_size)
    image_patches = [first_patch]
    image_patches.extend(crop_or_pad(channel, label, center, patch_size)[0] for channel in image[1:])
    return np.stack(image_patches, axis=0), label_patch
