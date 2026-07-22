"""Validated grayscale image/mask reading for 2D DWI samples."""
from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

SUPPORTED_SUFFIXES = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".npy"}

@dataclass(frozen=True)
class SampleRecord:
    patient: str
    timepoint: str
    image_path: Path
    mask_path: Path
    has_mask: int


@dataclass(frozen=True)
class PercentileNormalizer:
    lower: float
    upper: float

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if self.upper <= self.lower:
            raise ValueError("Percentile normalization requires upper > lower.")
        return np.clip((image.astype(np.float32) - self.lower) / (self.upper - self.lower), 0.0, 1.0)


@dataclass(frozen=True)
class PixelClassBalance:
    foreground_pixels: int
    background_pixels: int
    raw_pos_weight: float
    pos_weight: float

def _read_gray(path: Path) -> np.ndarray:
    if path.suffix.lower() not in SUPPORTED_SUFFIXES: raise ValueError(f"Unsupported image type: {path}")
    array = np.load(path) if path.suffix.lower() == ".npy" else np.asarray(Image.open(path))
    if array.ndim == 3: array = array[..., 0]
    if array.ndim != 2 or not np.isfinite(array).all(): raise ValueError(f"Invalid non-finite/non-2D image: {path}")
    return array

def load_sample(
    record: SampleRecord, normalizer: PercentileNormalizer | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read one manifest pair as `[1,H,W]` float tensors, raising on bad data."""
    image, mask = _read_gray(record.image_path), _read_gray(record.mask_path)
    if image.shape != mask.shape: raise ValueError(f"Image/mask size mismatch: {record.image_path} vs {record.mask_path}")
    if image.size == 0: raise ValueError(f"Empty image: {record.image_path}")
    image_dtype = image.dtype
    image = normalizer(image) if normalizer else image.astype(np.float32)
    if normalizer is None and np.issubdtype(image_dtype, np.integer): image /= float(np.iinfo(image_dtype).max)
    return torch.from_numpy(image[None]), torch.from_numpy((mask > 0).astype(np.float32)[None])


def fit_percentile_normalizer(
    records: list[SampleRecord], lower_percentile: float = 1.0, upper_percentile: float = 99.0
) -> PercentileNormalizer:
    if not records:
        raise ValueError("Cannot fit normalization without training records.")
    bounds = np.asarray([
        np.percentile(_read_gray(record.image_path), (lower_percentile, upper_percentile))
        for record in records
    ])
    lower, upper = (float(value) for value in np.median(bounds, axis=0))
    return PercentileNormalizer(lower, upper)


def calculate_pixel_class_balance(
    records: list[SampleRecord], max_pos_weight: float = 20.0
) -> PixelClassBalance:
    foreground_pixels = sum(int((_read_gray(record.mask_path) > 0).sum()) for record in records)
    total_pixels = sum(int(_read_gray(record.mask_path).size) for record in records)
    background_pixels = total_pixels - foreground_pixels
    if foreground_pixels == 0:
        raise ValueError("Training records contain no foreground pixels.")
    raw_pos_weight = background_pixels / foreground_pixels
    return PixelClassBalance(
        foreground_pixels=foreground_pixels,
        background_pixels=background_pixels,
        raw_pos_weight=raw_pos_weight,
        pos_weight=min(max_pos_weight, max(1.0, raw_pos_weight)),
    )

def read_manifest(data_root: Path) -> list[SampleRecord]:
    """Load pairs from the authoritative manifest, never infer patient ID from filenames."""
    manifest = data_root / "manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as handle: rows = list(csv.DictReader(handle))
    required = {"patient", "timepoint", "image_path", "mask_path", "has_mask"}
    if not rows or not required <= set(rows[0]): raise ValueError(f"Manifest missing required fields: {manifest}")
    seen: set[tuple[str, str]] = set(); records = []
    for row in rows:
        key = (row["image_path"], row["mask_path"])
        if key in seen: raise ValueError(f"Duplicate image/mask pairing: {key}")
        seen.add(key); image = Path(row["image_path"]); mask = Path(row["mask_path"])
        records.append(SampleRecord(row["patient"], row["timepoint"], image, mask, int(row["has_mask"])))
    return records

class DwiSliceDataset(Dataset[tuple[torch.Tensor, torch.Tensor, SampleRecord]]):
    def __init__(self, records: list[SampleRecord], transform: Callable | None = None, normalizer: PercentileNormalizer | None = None) -> None: self.records, self.transform, self.normalizer = records, transform, normalizer
    def __len__(self) -> int: return len(self.records)
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, SampleRecord]:
        image, mask = load_sample(self.records[index], self.normalizer)
        if self.transform:
            pair = self.transform(image=image.numpy()[0], mask=mask.numpy()[0]); image = torch.from_numpy(pair["image"])[None].float(); mask = torch.from_numpy((pair["mask"] > 0).astype(np.float32))[None]
        return image, mask, self.records[index]

def collate_samples(batch: list[tuple[torch.Tensor, torch.Tensor, SampleRecord]]) -> tuple[torch.Tensor, torch.Tensor, list[SampleRecord]]:
    """Stack tensors while preserving metadata records as a Python list."""
    images, masks, records = zip(*batch)
    return torch.stack(images), torch.stack(masks), list(records)
