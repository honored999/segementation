"""Read-only, on-demand Dataset501 fixed-fold 2D slice dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from standalone_nnunet2d.data.augmentation import AugmentationConfig, augment_slice
from standalone_nnunet2d.data.nifti_io import NiftiVolume, read_nifti
from standalone_nnunet2d.data.preprocessing import resample_inplane, z_score_normalize
from standalone_nnunet2d.data.sampling import central_slice_index, select_axial_slice, select_slice_index
from standalone_nnunet2d.data.symmetry_alignment import align_case


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS_PATH = PROJECT_ROOT / "reference" / "splits_final.json"
SplitName = Literal["train", "val"]


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
        symmetry_alignment: bool = False,
    ) -> None:
        self.raw_root = validate_raw_root(raw_root)
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
        self.symmetry_alignment = symmetry_alignment

    def __len__(self) -> int:
        return len(self.case_ids)

    def load_case(self, case_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Read and process one image/label pair without retaining a cache."""
        if case_id not in self.case_ids:
            raise ValueError(f"case {case_id!r} is not available in this dataset instance")
        image = read_nifti(self.raw_root / "imagesTr" / f"{case_id}_0000.nii.gz")
        label = read_nifti(self.raw_root / "labelsTr" / f"{case_id}.nii.gz")
        if not _same_geometry(image, label):
            raise ValueError(f"image and label geometry mismatch for case {case_id}")
        processed_image = resample_inplane(image, self.target_spacing_xy, is_segmentation=False)
        processed_label = resample_inplane(label, self.target_spacing_xy, is_segmentation=True)
        if processed_image.array.shape != processed_label.array.shape:
            raise ValueError(f"resampling produced mismatched shapes for case {case_id}")
        if self.symmetry_alignment:
            processed_image, processed_label, _ = align_case(processed_image, processed_label)
        return z_score_normalize(processed_image.array), processed_label.array

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        case_id = self.case_ids[index]
        image, label = self.load_case(case_id)
        slice_index = (
            central_slice_index(image.shape[0])
            if self.foreground_probability == 0.0
            else select_slice_index(label, self.rng, foreground_probability=self.foreground_probability)
        )
        image_slice, label_slice = augment_slice(
            select_axial_slice(image, slice_index),
            select_axial_slice(label, slice_index),
            self.rng,
            self.augmentation_config,
        )
        return torch.from_numpy(image_slice).unsqueeze(0).float(), torch.from_numpy(label_slice).long()
