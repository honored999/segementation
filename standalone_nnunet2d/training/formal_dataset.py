"""Fixed-fold 2D patch dataset for the separately configured formal Trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from standalone_nnunet2d.data.dataset import SplitName, StrokeSliceDataset
from standalone_nnunet2d.training.batch_sampler import PatchRequest
from standalone_nnunet2d.training.official_augmentation import apply_official_2d_batchgeneratorsv2
from standalone_nnunet2d.training.patch_sampler import crop_or_pad, sample_patch_center


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
        symmetry_alignment: bool = False,
    ) -> None:
        if len(patch_size) != 2 or any(size <= 0 for size in patch_size):
            raise ValueError("patch_size must contain two positive values")
        if not 0.0 <= oversample_foreground_percent <= 1.0:
            raise ValueError("oversample_foreground_percent must be in [0, 1]")
        self.patch_size = patch_size
        self.use_mask_for_norm = tuple(use_mask_for_norm)
        self.oversample_foreground_percent = oversample_foreground_percent
        self.patch_rng = rng or np.random.default_rng()
        self.augment = augment
        if patch_request is not None and case_ids is None:
            case_ids = (patch_request.case_id,)
        self.patch_request = patch_request
        super().__init__(
            raw_root,
            fold=fold,
            split=split,
            case_ids=case_ids,
            rng=self.patch_rng,
            foreground_probability=0.0,
            symmetry_alignment=symmetry_alignment,
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
        image_slice, label_slice = image[z_index], label[z_index]
        if center is None:
            center = sample_patch_center(
                label_slice,
                self.patch_rng,
                oversample_foreground_percent=1.0 if force_foreground else 0.0,
            )
        image_patch, label_patch = crop_or_pad(image_slice, label_slice, center, self.patch_size)
        if self.augment:
            seed = int(self.patch_rng.integers(0, 2**32 - 1))
            image_patch, label_patch = apply_official_2d_batchgeneratorsv2(
                image_patch,
                label_patch,
                patch_size=self.patch_size,
                use_mask_for_norm=self.use_mask_for_norm,
                seed=seed,
            )
        label_patch = np.where(label_patch < 0, 0, label_patch).astype(np.int64, copy=False)
        if not np.isin(label_patch, (0, 1)).all():
            raise ValueError("formal patch labels must contain only 0 and 1")
        return torch.from_numpy(image_patch).unsqueeze(0).float(), torch.from_numpy(label_patch).long()

    def _select_z_index(self, label: np.ndarray) -> tuple[int, bool]:
        foreground_z = np.flatnonzero(label.sum(axis=(1, 2)) > 0)
        force_foreground = len(foreground_z) > 0 and self.patch_rng.random() < self.oversample_foreground_percent
        z_index = int(self.patch_rng.choice(foreground_z)) if force_foreground else int(self.patch_rng.integers(label.shape[0]))
        return z_index, force_foreground
