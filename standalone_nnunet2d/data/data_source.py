"""Formal-training case sources for raw NIfTI and nnU-Net b2nd data."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from standalone_nnunet2d.data.dataset import SplitName, StrokeSliceDataset, load_fold_cases


DataSourceName = Literal["raw_nifti_online", "nnunet_preprocessed_b2nd"]
RAW_NIFTI_ONLINE: DataSourceName = "raw_nifti_online"
NNUNET_PREPROCESSED_B2ND: DataSourceName = "nnunet_preprocessed_b2nd"


def validate_data_source(data_source: str) -> DataSourceName:
    if data_source not in (RAW_NIFTI_ONLINE, NNUNET_PREPROCESSED_B2ND):
        raise ValueError(
            "data_source must be 'raw_nifti_online' or 'nnunet_preprocessed_b2nd'"
        )
    return data_source  # type: ignore[return-value]


def resolve_data_source_root(
    raw_root: Path | None,
    preprocessed_root: Path | None,
    *,
    data_source: str,
) -> tuple[DataSourceName, Path]:
    source = validate_data_source(data_source)
    if (raw_root is None) == (preprocessed_root is None):
        raise ValueError("exactly one of raw_root and preprocessed_root must be supplied")
    if source == RAW_NIFTI_ONLINE:
        if raw_root is None:
            raise ValueError("raw_nifti_online requires raw_root")
        return source, Path(raw_root)
    if preprocessed_root is None:
        raise ValueError("nnunet_preprocessed_b2nd requires preprocessed_root")
    return source, Path(preprocessed_root)


def _validate_case_ids(
    fold: int,
    split: SplitName,
    case_ids: tuple[str, ...] | None,
) -> tuple[str, ...]:
    allowed_case_ids = load_fold_cases(fold, split)
    resolved_case_ids = case_ids if case_ids is not None else allowed_case_ids
    if not resolved_case_ids:
        raise ValueError("case_ids must not be empty")
    outside_split = set(resolved_case_ids) - set(allowed_case_ids)
    if outside_split:
        raise ValueError(
            f"case IDs are not members of fold {fold} {split}: {sorted(outside_split)}"
        )
    return tuple(resolved_case_ids)


@dataclass
class PreparedCase:
    """Validated label volume plus a source-specific image-slice reader."""

    label: np.ndarray
    shape: tuple[int, int, int]
    _image_volume: np.ndarray | None = None
    _image_store: Any | None = None

    def image_slice(self, z_index: int) -> np.ndarray:
        if not 0 <= z_index < self.shape[0]:
            raise IndexError(f"slice index {z_index} is outside [0, {self.shape[0]})")
        if self._image_volume is not None:
            image_slice = self._image_volume[z_index]
        elif self._image_store is not None:
            # Keep this exact channel/z indexing so one b2nd chunk is read.
            image_slice = self._image_store[0, z_index, :, :]
        else:  # pragma: no cover - construction always supplies one backend
            raise RuntimeError("prepared case has no image backend")
        result = np.asarray(image_slice)
        expected_shape = self.shape[1:]
        if result.shape != expected_shape:
            raise ValueError(
                f"image slice shape is {result.shape}, expected {expected_shape}"
            )
        if not np.isfinite(result).all():
            raise ValueError("preprocessed image contains non-finite values")
        return result.astype(np.float32, copy=False)


class FormalCaseSource(Protocol):
    case_ids: tuple[str, ...]

    def prepare_case(self, case_id: str) -> PreparedCase:
        """Load labels and prepare a source-specific image-slice reader."""


class RawNiftiCaseSource:
    """Adapter that delegates raw loading to the existing StrokeSliceDataset."""

    def __init__(
        self,
        raw_root: Path,
        *,
        fold: int,
        split: SplitName,
        case_ids: tuple[str, ...] | None = None,
    ) -> None:
        self.dataset = StrokeSliceDataset(
            raw_root,
            fold=fold,
            split=split,
            case_ids=case_ids,
            foreground_probability=0.0,
        )
        self.case_ids = self.dataset.case_ids

    def prepare_case(self, case_id: str) -> PreparedCase:
        image, label = self.dataset.load_case(case_id)
        if image.ndim != 3 or label.ndim != 3 or image.shape != label.shape:
            raise ValueError(
                f"raw case {case_id} must provide matched non-empty 3D arrays, "
                f"got {image.shape} and {label.shape}"
            )
        if any(size <= 0 for size in image.shape):
            raise ValueError(f"raw case {case_id} has an empty dimension")
        return PreparedCase(
            label=np.asarray(label),
            shape=tuple(int(size) for size in image.shape),
            _image_volume=np.asarray(image),
        )


def _validate_b2nd_store_pair(
    case_id: str,
    data_store: Any,
    seg_store: Any,
) -> tuple[int, int, int]:
    data_shape = tuple(int(size) for size in data_store.shape)
    seg_shape = tuple(int(size) for size in seg_store.shape)
    if data_shape != seg_shape:
        raise ValueError(
            f"preprocessed data/seg shapes differ for {case_id}: "
            f"{data_shape} != {seg_shape}"
        )
    if len(data_shape) != 4 or data_shape[0] != 1 or any(size <= 0 for size in data_shape):
        raise ValueError(
            f"preprocessed data/seg for {case_id} must be non-empty [1,Z,Y,X], "
            f"got {data_shape}"
        )
    if not np.issubdtype(np.dtype(data_store.dtype), np.number):
        raise ValueError(f"preprocessed image dtype for {case_id} must be numeric")
    if not np.issubdtype(np.dtype(seg_store.dtype), np.integer):
        raise ValueError(f"preprocessed label dtype for {case_id} must be integer")
    return data_shape[1], data_shape[2], data_shape[3]


def _validate_b2nd_label(case_id: str, label: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    if label.shape != shape:
        raise ValueError(
            f"preprocessed label shape for {case_id} is {label.shape}, expected {shape}"
        )
    if not np.isin(label, (0, 1)).all():
        raise ValueError(f"preprocessed labels for {case_id} must contain only 0 and 1")
    return label.astype(np.int16, copy=False)


class PreprocessedB2ndCaseSource:
    """Read nnU-Net's root-level ``caseXXX*.b2nd`` pair on demand."""

    def __init__(
        self,
        preprocessed_root: Path,
        *,
        fold: int,
        split: SplitName,
        case_ids: tuple[str, ...] | None = None,
    ) -> None:
        self.root = Path(preprocessed_root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"preprocessed root does not exist: {self.root}")
        self.case_ids = _validate_case_ids(fold, split, case_ids)
        try:
            self._blosc2 = importlib.import_module("blosc2")
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "nnunet_preprocessed_b2nd requires the optional 'blosc2' package"
            ) from error

    def _paths(self, case_id: str) -> tuple[Path, Path]:
        if case_id not in self.case_ids:
            raise ValueError(f"case {case_id!r} is not available in this dataset instance")
        return (
            self.root / f"{case_id}.b2nd",
            self.root / f"{case_id}_seg.b2nd",
        )

    def prepare_case(self, case_id: str) -> PreparedCase:
        data_path, seg_path = self._paths(case_id)
        if not data_path.is_file() or not seg_path.is_file():
            missing = [str(path) for path in (data_path, seg_path) if not path.is_file()]
            raise FileNotFoundError(
                f"preprocessed case {case_id} is missing required b2nd file(s): {', '.join(missing)}"
            )
        data_store = self._blosc2.open(urlpath=str(data_path), mode="r")
        seg_store = self._blosc2.open(urlpath=str(seg_path), mode="r")
        shape = _validate_b2nd_store_pair(case_id, data_store, seg_store)
        # Full seg loading is intentional: current formal z oversampling samples
        # from the same foreground-slice distribution as the raw loader.
        label = np.asarray(seg_store[0, :, :, :])
        label = _validate_b2nd_label(case_id, label, shape)
        return PreparedCase(label=label, shape=shape, _image_store=data_store)


def make_formal_case_source(
    data_root: Path,
    *,
    data_source: str,
    fold: int,
    split: SplitName,
    case_ids: tuple[str, ...] | None = None,
) -> FormalCaseSource:
    source = validate_data_source(data_source)
    if source == RAW_NIFTI_ONLINE:
        return RawNiftiCaseSource(
            data_root, fold=fold, split=split, case_ids=case_ids
        )
    return PreprocessedB2ndCaseSource(
        data_root, fold=fold, split=split, case_ids=case_ids
    )
