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

def _read_gray(path: Path) -> np.ndarray:
    if path.suffix.lower() not in SUPPORTED_SUFFIXES: raise ValueError(f"Unsupported image type: {path}")
    array = np.load(path) if path.suffix.lower() == ".npy" else np.asarray(Image.open(path))
    if array.ndim == 3: array = array[..., 0]
    if array.ndim != 2 or not np.isfinite(array).all(): raise ValueError(f"Invalid non-finite/non-2D image: {path}")
    return array

def load_sample(record: SampleRecord) -> tuple[torch.Tensor, torch.Tensor]:
    """Read one manifest pair as `[1,H,W]` float tensors, raising on bad data."""
    image, mask = _read_gray(record.image_path), _read_gray(record.mask_path)
    if image.shape != mask.shape: raise ValueError(f"Image/mask size mismatch: {record.image_path} vs {record.mask_path}")
    if image.size == 0: raise ValueError(f"Empty image: {record.image_path}")
    image = image.astype(np.float32)
    if np.issubdtype(_read_gray(record.image_path).dtype, np.integer): image /= float(np.iinfo(_read_gray(record.image_path).dtype).max)
    return torch.from_numpy(image[None]), torch.from_numpy((mask > 0).astype(np.float32)[None])

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
    def __init__(self, records: list[SampleRecord], transform: Callable | None = None) -> None: self.records, self.transform = records, transform
    def __len__(self) -> int: return len(self.records)
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, SampleRecord]:
        image, mask = load_sample(self.records[index])
        if self.transform:
            pair = self.transform(image=image.numpy()[0], mask=mask.numpy()[0]); image = torch.from_numpy(pair["image"])[None].float(); mask = torch.from_numpy((pair["mask"] > 0).astype(np.float32))[None]
        return image, mask, self.records[index]

def collate_samples(batch: list[tuple[torch.Tensor, torch.Tensor, SampleRecord]]) -> tuple[torch.Tensor, torch.Tensor, list[SampleRecord]]:
    """Stack tensors while preserving metadata records as a Python list."""
    images, masks, records = zip(*batch)
    return torch.stack(images), torch.stack(masks), list(records)
