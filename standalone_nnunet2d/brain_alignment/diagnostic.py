"""Shared, label-free ADN real-data diagnostic training utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import nn

from standalone_nnunet2d.brain_alignment.adn_transform import (
    ADNTransformAligner,
    TransformRanges,
    alignment_losses,
)
from standalone_nnunet2d.brain_alignment.nifti_adapter import canonicalize_nifti
from standalone_nnunet2d.data.nifti_io import read_nifti


CHECKPOINT_FORMAT_VERSION = 1
CHECKPOINT_KIND = "adn_real_data_diagnostic"
MODEL_INPUT_MINIMUM_DEPTH = 16
MODEL_INPUT_PADDING_POLICY = "symmetric_constant_zero_D_only"


@dataclass(frozen=True)
class LoadedDiagnosticCheckpoint:
    model: nn.Module
    epoch: int
    history: list[dict[str, float | int]]
    best_loss: float
    metadata: dict[str, Any]


def build_alignment_model() -> ADNTransformAligner:
    """Build the existing ADN model with its unchanged default ranges."""
    return ADNTransformAligner(in_channels=1)


def normalize_volume(array: np.ndarray) -> np.ndarray:
    """Finite per-volume z-score normalization without cohort statistics."""
    values = np.asarray(array, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"volume must be 3D, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("volume contains non-finite values")
    mean = float(values.mean(dtype=np.float64))
    standard_deviation = float(values.std(dtype=np.float64))
    if standard_deviation <= np.finfo(np.float32).eps:
        return np.zeros_like(values, dtype=np.float32)
    return np.ascontiguousarray((values - mean) / standard_deviation, dtype=np.float32)


def _model_input_depth_contract() -> dict[str, Any]:
    return {
        "axis": "D",
        "minimum_model_depth": MODEL_INPUT_MINIMUM_DEPTH,
        "padding_policy": MODEL_INPUT_PADDING_POLICY,
        "pad_value": 0.0,
    }


def _validate_model_input_depth_contract(
    contract: object, *, context: str, allow_extra: bool = False
) -> None:
    if not isinstance(contract, Mapping):
        raise ValueError(f"{context} model_input_depth contract must be a mapping")
    expected = _model_input_depth_contract()
    if (set(contract) != set(expected)) if not allow_extra else not set(expected).issubset(contract):
        raise ValueError(f"{context} model_input_depth contract has invalid fields")
    if contract.get("axis") != expected["axis"]:
        raise ValueError(f"{context} model_input_depth contract has invalid axis")
    minimum_depth = contract.get("minimum_model_depth")
    if (
        not isinstance(minimum_depth, int)
        or isinstance(minimum_depth, bool)
        or minimum_depth != MODEL_INPUT_MINIMUM_DEPTH
    ):
        raise ValueError(f"{context} model_input_depth contract has invalid minimum depth")
    if contract.get("padding_policy") != expected["padding_policy"]:
        raise ValueError(f"{context} model_input_depth contract has invalid padding policy")
    pad_value = contract.get("pad_value")
    if isinstance(pad_value, bool) or not isinstance(pad_value, (int, float)) or pad_value != 0.0:
        raise ValueError(f"{context} model_input_depth contract has invalid pad value")


def _validate_model_input_depth_fields(
    metadata: Mapping[str, Any], *, context: str, require_padded_field: bool
) -> bool:
    fields = ("input_depth_before", "input_depth_after", "pad_before", "pad_after")
    values = tuple(metadata.get(field) for field in fields)
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        raise ValueError(f"{context} has invalid depth fields")
    input_depth_before, input_depth_after, pad_before, pad_after = values
    if min(values) < 0:
        raise ValueError(f"{context} is inconsistent with the deterministic padding policy")
    missing = max(0, MODEL_INPUT_MINIMUM_DEPTH - input_depth_before)
    expected = (
        max(MODEL_INPUT_MINIMUM_DEPTH, input_depth_before),
        missing // 2,
        missing - (missing // 2),
        bool(missing),
    )
    if (input_depth_after, pad_before, pad_after) != expected[:3]:
        raise ValueError(f"{context} is inconsistent with the deterministic padding policy")
    padded = metadata.get("padded")
    if padded is not None and (not isinstance(padded, bool) or padded != expected[3]):
        raise ValueError(f"{context} is inconsistent with the deterministic padding policy")
    if require_padded_field and padded is None:
        raise ValueError(f"{context} is missing the padded field")
    return expected[3]


def _validate_model_input_depth_entry(metadata: object, *, context: str) -> None:
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{context} model_input_depth metadata must be a mapping")
    _validate_model_input_depth_contract(metadata, context=context, allow_extra=True)
    _validate_model_input_depth_fields(metadata, context=context, require_padded_field=True)


def _validate_model_input_depth_metadata(metadata: object, *, context: str) -> None:
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{context} model_input_depth metadata must be a mapping")
    _validate_model_input_depth_contract(metadata, context=context, allow_extra=True)
    padded_case_ids = metadata.get("padded_case_ids")
    padded_cases = metadata.get("padded_cases")
    if not isinstance(padded_case_ids, list) or not isinstance(padded_cases, list):
        raise ValueError(f"{context} model_input_depth metadata requires padded_case_ids and padded_cases lists")
    if any(not isinstance(case_id, str) or not case_id for case_id in padded_case_ids):
        raise ValueError(f"{context} model_input_depth padded_case_ids contains an invalid case id")
    if len(set(padded_case_ids)) != len(padded_case_ids) or len(padded_case_ids) != len(padded_cases):
        raise ValueError(f"{context} model_input_depth padded case records are inconsistent")
    case_ids = metadata.get("case_ids")
    if case_ids is not None:
        if (
            not isinstance(case_ids, list)
            or any(not isinstance(case_id, str) or not case_id for case_id in case_ids)
            or len(set(case_ids)) != len(case_ids)
            or any(case_id not in case_ids for case_id in padded_case_ids)
        ):
            raise ValueError(f"{context} model_input_depth padded case ids are inconsistent with case_ids")
    record_fields = {
        "case_id", "input_depth_before", "input_depth_after", "pad_before", "pad_after",
    }
    for case_id, record in zip(padded_case_ids, padded_cases, strict=True):
        if not isinstance(record, Mapping) or set(record) != record_fields or record.get("case_id") != case_id:
            raise ValueError(f"{context} model_input_depth padded case records are inconsistent")
        if not _validate_model_input_depth_fields(
            record, context=f"{context} model_input_depth case {case_id}", require_padded_field=False
        ):
            raise ValueError(f"{context} model_input_depth padded case records are inconsistent")


def pad_model_input_depth(tensor: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    """Pad only the model-input D axis to the minimum depth with constant zeros."""
    if tensor.ndim < 3:
        raise ValueError(f"model input must have at least 3 dimensions, got {tuple(tensor.shape)}")
    input_depth_before = int(tensor.shape[-3])
    missing_depth = max(0, MODEL_INPUT_MINIMUM_DEPTH - input_depth_before)
    pad_before = missing_depth // 2
    pad_after = missing_depth - pad_before
    if missing_depth:
        parts: list[torch.Tensor] = []
        if pad_before:
            shape = list(tensor.shape)
            shape[-3] = pad_before
            parts.append(torch.zeros(tuple(shape), dtype=tensor.dtype, device=tensor.device))
        parts.append(tensor)
        if pad_after:
            shape = list(tensor.shape)
            shape[-3] = pad_after
            parts.append(torch.zeros(tuple(shape), dtype=tensor.dtype, device=tensor.device))
        padded = torch.cat(parts, dim=-3)
    else:
        padded = tensor
    metadata: dict[str, Any] = {
        "axis": "D",
        "minimum_model_depth": MODEL_INPUT_MINIMUM_DEPTH,
        "padding_policy": MODEL_INPUT_PADDING_POLICY,
        "pad_value": 0.0,
        "input_depth_before": input_depth_before,
        "input_depth_after": int(padded.shape[-3]),
        "pad_before": pad_before,
        "pad_after": pad_after,
        "padded": bool(missing_depth),
    }
    _validate_model_input_depth_entry(metadata, context="generated")
    return padded, metadata


def unpad_model_input_depth(tensor: torch.Tensor, metadata: Mapping[str, Any]) -> torch.Tensor:
    """Remove model-only D padding described by ``pad_model_input_depth``."""
    if tensor.ndim < 3:
        raise ValueError(f"model output must have at least 3 dimensions, got {tuple(tensor.shape)}")
    _validate_model_input_depth_entry(metadata, context="model-input depth padding")
    input_depth_after = int(metadata["input_depth_after"])
    pad_before = int(metadata["pad_before"])
    pad_after = int(metadata["pad_after"])
    if int(tensor.shape[-3]) != input_depth_after:
        raise ValueError(
            f"model output depth {int(tensor.shape[-3])} does not match padded depth {input_depth_after}"
        )
    if not pad_before and not pad_after:
        return tensor
    end = input_depth_after - pad_after
    return tensor[..., pad_before:end, :, :]


def load_fold0_train_cases(splits_file: Path) -> tuple[str, ...]:
    """Read exactly fold 0's train list from an explicit split file."""
    path = Path(splits_file).resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        raise ValueError("splits file must contain a non-empty list of fold mappings")
    train = payload[0].get("train")
    if not isinstance(train, list) or not train or any(not isinstance(case, str) or not case for case in train):
        raise ValueError("fold 0 train must be a non-empty list of case ids")
    if len(set(train)) != len(train):
        raise ValueError("fold 0 train contains duplicate case ids")
    return tuple(train)


def fold0_training_image_paths(dataset_dir: Path, case_ids: Iterable[str]) -> tuple[Path, ...]:
    """Construct DWI paths directly; directory enumeration is deliberately absent."""
    images = Path(dataset_dir).resolve() / "imagesTr"
    return tuple((images / f"{case_id}_0000.nii.gz").resolve() for case_id in case_ids)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def ensure_diagnostic_output_dir(output_dir: Path, *, dataset_dir: Path | None = None) -> Path:
    """Create an empty generated-output directory outside source data."""
    output = Path(output_dir).resolve()
    if dataset_dir is not None:
        dataset = Path(dataset_dir).resolve()
        if _is_within(output, dataset) or _is_within(dataset, output):
            raise ValueError("diagnostic output directory must be separate from the dataset directory")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"diagnostic output directory is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _ranges_dict(model: nn.Module) -> dict[str, float]:
    ranges = getattr(model, "ranges", TransformRanges())
    if not isinstance(ranges, TransformRanges):
        ranges = TransformRanges()
    return {key: float(value) for key, value in asdict(ranges).items()}


def save_diagnostic_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    history: list[dict[str, float | int]],
    best_loss: float,
    metadata: dict[str, Any],
) -> None:
    """Persist the ADN diagnostic contract and current training state."""
    metadata_payload = dict(metadata)
    if "model_input_depth_padding" not in metadata_payload:
        metadata_payload["model_input_depth_padding"] = {
            **_model_input_depth_contract(),
            "padded_case_ids": [],
            "padded_cases": [],
        }
    _validate_model_input_depth_metadata(
        metadata_payload["model_input_depth_padding"], context="checkpoint metadata"
    )
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "kind": CHECKPOINT_KIND,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "epoch": int(epoch),
        "history": list(history),
        "best_loss": float(best_loss),
        "model_contract": {
            "class_name": "ADNTransformAligner",
            "in_channels": int(getattr(model, "in_channels", 1)),
            "transform_ranges": _ranges_dict(model),
            "canonical_contract": "acquisition_preserving_lr",
            "model_axis_semantics": {
                "D": "acquisition_through_plane",
                "H": "acquisition_in_plane_non_lr",
                "W": "anatomical_lr",
            },
            "transform_semantics": {
                "tx": "model_space_lr_translation",
                "rz": "acquisition_model_in_plane_rotation",
                "physical_3d_rigid_registration": False,
            },
            "model_input_depth_padding": _model_input_depth_contract(),
        },
        "metadata": metadata_payload,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


def validate_checkpoint_payload(payload: object) -> dict[str, Any]:
    """Validate all non-tensor model contract fields before model creation."""
    if not isinstance(payload, dict):
        raise ValueError("ADN diagnostic checkpoint must be a mapping")
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION or payload.get("kind") != CHECKPOINT_KIND:
        raise ValueError("unrecognized ADN diagnostic checkpoint format")
    required = (
        "model_state_dict",
        "optimizer_state_dict",
        "epoch",
        "history",
        "best_loss",
        "model_contract",
        "metadata",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"ADN diagnostic checkpoint is missing fields: {missing}")
    contract = payload["model_contract"]
    if not isinstance(contract, dict):
        raise ValueError("checkpoint model_contract must be a mapping")
    expected_ranges = asdict(TransformRanges())
    if (
        contract.get("class_name") != "ADNTransformAligner"
        or contract.get("in_channels") != 1
        or contract.get("transform_ranges") != expected_ranges
        or contract.get("canonical_contract") != "acquisition_preserving_lr"
        or contract.get("model_axis_semantics") != {
            "D": "acquisition_through_plane",
            "H": "acquisition_in_plane_non_lr",
            "W": "anatomical_lr",
        }
        or contract.get("transform_semantics") != {
            "tx": "model_space_lr_translation",
            "rz": "acquisition_model_in_plane_rotation",
            "physical_3d_rigid_registration": False,
        }
    ):
        raise ValueError("checkpoint ADN model/orientation contract does not match this workflow")
    _validate_model_input_depth_contract(
        contract.get("model_input_depth_padding"), context="checkpoint model_contract"
    )
    if not isinstance(payload["model_state_dict"], dict) or not isinstance(payload["history"], list):
        raise ValueError("checkpoint state/history fields have invalid types")
    if payload["optimizer_state_dict"] is not None and not isinstance(payload["optimizer_state_dict"], dict):
        raise ValueError("checkpoint optimizer_state_dict must be a mapping or None")
    if not isinstance(payload["metadata"], dict):
        raise ValueError("checkpoint metadata must be a mapping")
    _validate_model_input_depth_metadata(payload["metadata"].get("model_input_depth_padding"), context="checkpoint metadata")
    return payload


def load_diagnostic_checkpoint(path: Path, *, device: torch.device) -> LoadedDiagnosticCheckpoint:
    """Validate metadata first, then construct and load the ADN model."""
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {source}")
    payload = validate_checkpoint_payload(torch.load(source, map_location="cpu", weights_only=False))
    model = build_alignment_model()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    return LoadedDiagnosticCheckpoint(
        model=model,
        epoch=int(payload["epoch"]),
        history=list(payload["history"]),
        best_loss=float(payload["best_loss"]),
        metadata=dict(payload["metadata"]),
    )


def train_fold0(
    *,
    dataset_dir: Path,
    splits_file: Path,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    output_dir: Path,
    batch_size: int = 1,
) -> list[dict[str, float | int]]:
    """Train ADN only on fold-0 DWI images with existing alignment losses."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    if batch_size != 1:
        raise ValueError("batch size must be 1 because volumes are processed independently")
    dataset = Path(dataset_dir).resolve()
    output = ensure_diagnostic_output_dir(output_dir, dataset_dir=dataset)
    case_ids = load_fold0_train_cases(splits_file)
    paths = fold0_training_image_paths(dataset, case_ids)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"fold 0 DWI image does not exist: {missing[0]}")

    model = build_alignment_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    padded_cases: list[dict[str, int | str]] = []
    common_metadata = {
        "dataset_dir": str(dataset),
        "splits_file": str(Path(splits_file).resolve()),
        "fold": 0,
        "split": "train",
        "case_ids": list(case_ids),
        "batch_size": 1,
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "normalization": "finite_per_volume_zscore",
        "labels_accessed": False,
        "experiment_class": "diagnostic",
    }
    for epoch in range(1, epochs + 1):
        model.train()
        totals = {"flip_loss": 0.0, "reconstruction_loss": 0.0, "total_loss": 0.0}
        for case_id, image_path in zip(case_ids, paths, strict=True):
            canonical = canonicalize_nifti(read_nifti(image_path))
            tensor = torch.from_numpy(normalize_volume(canonical.array))[None, None].to(device)
            tensor, padding = pad_model_input_depth(tensor)
            if epoch == 1 and padding["padded"]:
                padded_cases.append({
                    "case_id": case_id,
                    "input_depth_before": int(padding["input_depth_before"]),
                    "input_depth_after": int(padding["input_depth_after"]),
                    "pad_before": int(padding["pad_before"]),
                    "pad_after": int(padding["pad_after"]),
                })
            optimizer.zero_grad(set_to_none=True)
            result = model(tensor)
            losses = alignment_losses(tensor, result.aligned, result.inverse_sampling_matrix)
            losses.total_loss.backward()
            optimizer.step()
            totals["flip_loss"] += float(losses.flip_loss.detach().cpu())
            totals["reconstruction_loss"] += float(losses.reconstruction_loss.detach().cpu())
            totals["total_loss"] += float(losses.total_loss.detach().cpu())
        row: dict[str, float | int] = {"epoch": epoch}
        row.update({key: value / len(paths) for key, value in totals.items()})
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (output / "history.jsonl").write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in history), encoding="utf-8"
        )
        model_input_padding_metadata = {
            "axis": "D",
            "minimum_model_depth": MODEL_INPUT_MINIMUM_DEPTH,
            "padding_policy": MODEL_INPUT_PADDING_POLICY,
            "pad_value": 0.0,
            "padded_case_ids": [item["case_id"] for item in padded_cases],
            "padded_cases": [dict(item) for item in padded_cases],
        }
        latest_metadata = {
            **common_metadata,
            "checkpoint_role": "latest",
            "model_input_depth_padding": model_input_padding_metadata,
        }
        save_diagnostic_checkpoint(
            output / "checkpoint_latest.pth", model, optimizer=optimizer, epoch=epoch,
            history=history, best_loss=min(best_loss, float(row["total_loss"])), metadata=latest_metadata,
        )
        if float(row["total_loss"]) < best_loss:
            best_loss = float(row["total_loss"])
            save_diagnostic_checkpoint(
                output / "checkpoint_best.pth", model, optimizer=optimizer, epoch=epoch,
                history=history, best_loss=best_loss,
                metadata={
                    **common_metadata,
                    "checkpoint_role": "best",
                    "model_input_depth_padding": model_input_padding_metadata,
                },
            )
    return history
