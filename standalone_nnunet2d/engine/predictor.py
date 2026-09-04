"""Full-resolution, case-level 2D inference for arrays ordered as (z, y, x)."""

from __future__ import annotations

from contextlib import nullcontext
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from torch import Tensor, nn

from standalone_nnunet2d.data.nifti_io import NiftiVolume, read_nifti, write_nifti
from standalone_nnunet2d.data.preprocessing import z_score_normalize


DEFAULT_MIRROR_AXES = (0, 1)
DEFAULT_PATCH_SIZE = (512, 512)
DEFAULT_TILE_STEP_SIZE = 0.5


@lru_cache(maxsize=2)
def compute_gaussian(
    tile_size: tuple[int, ...] | list[int],
    sigma_scale: float = 1.0 / 8.0,
    value_scaling_factor: float = 1.0,
    *,
    dtype: torch.dtype = torch.float16,
    device: torch.device | str = torch.device("cpu"),
) -> Tensor:
    """Build the nnU-Net Gaussian importance map for one prediction tile."""
    tile_size = tuple(int(value) for value in tile_size)
    tmp = np.zeros(tile_size)
    center_coords = [value // 2 for value in tile_size]
    sigmas = [value * sigma_scale for value in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(
        tmp,
        sigmas,
        order=0,
        mode="constant",
        cval=0,
    )
    gaussian_importance_map /= np.max(gaussian_importance_map) / value_scaling_factor
    gaussian_importance_map = torch.from_numpy(gaussian_importance_map).to(
        device=torch.device(device),
        dtype=dtype,
    )
    mask = gaussian_importance_map == 0
    gaussian_importance_map[mask] = torch.min(gaussian_importance_map[~mask])
    return gaussian_importance_map


def _autocast_context(device: torch.device):
    """Use the official CUDA autocast policy without enabling CPU autocast."""
    if device.type == "cuda":
        return torch.autocast(device.type, enabled=True)
    return nullcontext()


def _configure_inference_backend(device: torch.device) -> None:
    """Match nnU-Net's CUDA inference constructor backend setting."""
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True


def mirror_combinations(mirror_axes: Iterable[int]) -> tuple[tuple[int, ...], ...]:
    """Return every non-empty spatial mirror combination in stable order."""
    axes = tuple(mirror_axes)
    if any(axis not in (0, 1) for axis in axes) or len(set(axes)) != len(axes):
        raise ValueError(f"mirror axes must be unique values from (0, 1), got {axes}")
    return tuple(
        combination
        for size in range(1, len(axes) + 1)
        for combination in combinations(axes, size)
    )


def _full_resolution_logits(outputs: Tensor | tuple[Tensor, ...] | list[Tensor]) -> Tensor:
    """Select the full-resolution logits from tensor or deep-supervision output."""
    if isinstance(outputs, (tuple, list)):
        if not outputs:
            raise ValueError("model returned no logits")
        outputs = outputs[0]
    if not isinstance(outputs, Tensor) or outputs.ndim != 4:
        raise ValueError("predictor requires four-dimensional full-resolution logits")
    if outputs.shape[1] != 2:
        raise ValueError(f"predictor requires two class logits, got shape {tuple(outputs.shape)}")
    return outputs


def _tile_starts(size: int, patch: int, step: float) -> tuple[int, ...]:
    if size <= patch:
        return (0,)
    num_steps = int(np.ceil((size - patch) / (patch * step))) + 1
    actual_step = (size - patch) / (num_steps - 1)
    return tuple(int(np.round(actual_step * index)) for index in range(num_steps))


def _normalise_patch_size(patch_size: tuple[int, int] | list[int]) -> tuple[int, int]:
    if len(patch_size) != 2 or any(int(value) <= 0 for value in patch_size):
        raise ValueError(f"patch size must contain two positive values, got {patch_size}")
    return int(patch_size[0]), int(patch_size[1])


def _predict_tile_logits(
    model: nn.Module,
    tile: Tensor,
    *,
    mirror_axes: tuple[int, ...],
) -> Tensor:
    """Average base and mirrored logits after undoing each input mirror."""
    logits = _full_resolution_logits(model(tile))
    if logits.shape[0] != tile.shape[0]:
        raise ValueError("model changed batch size between tile input and logits")
    if logits.shape[2:] != tile.shape[2:]:
        raise ValueError(
            "predictor requires full-resolution logits: "
            f"input {tuple(tile.shape[2:])}, output {tuple(logits.shape[2:])}"
        )
    total = logits.clone()
    count = 1
    for axes in mirror_combinations(mirror_axes):
        dims = tuple(axis + 2 for axis in axes)
        mirrored_tile = torch.flip(tile, dims=dims)
        mirrored_logits = _full_resolution_logits(model(mirrored_tile))
        if mirrored_logits.shape != logits.shape:
            raise ValueError("mirrored inference returned inconsistent logits shape")
        total += torch.flip(mirrored_logits, dims=dims)
        count += 1
    return total / count


def predict_logits_2d(
    model: nn.Module,
    image: Tensor,
    device: torch.device,
    *,
    mirror_axes: tuple[int, ...] = DEFAULT_MIRROR_AXES,
    patch_size: tuple[int, int] = DEFAULT_PATCH_SIZE,
    tile_step_size: float = DEFAULT_TILE_STEP_SIZE,
) -> Tensor:
    """Predict a 2D tensor by averaging full-resolution logits over TTA and tiles."""
    if image.ndim != 4 or image.shape[0] < 1 or image.shape[1] != 1:
        raise ValueError(f"image must have shape (B, 1, H, W), got {tuple(image.shape)}")
    if not 0.0 < tile_step_size <= 1.0:
        raise ValueError(f"tile_step_size must be in (0, 1], got {tile_step_size}")
    patch_height, patch_width = _normalise_patch_size(patch_size)
    height, width = image.shape[2:]

    model_device = torch.device(device)
    _configure_inference_backend(model_device)
    tensor = image.to(model_device)
    original_height, original_width = tensor.shape[2:]
    padding_height = max(0, patch_height - original_height)
    padding_width = max(0, patch_width - original_width)
    padding_top = padding_height // 2
    padding_left = padding_width // 2
    padded_height = original_height + padding_height
    padded_width = original_width + padding_width
    if (padded_height, padded_width) != (original_height, original_width):
        padded = torch.zeros(
            (tensor.shape[0], tensor.shape[1], padded_height, padded_width),
            device=model_device,
            dtype=tensor.dtype,
        )
        padded[
            :,
            :,
            padding_top : padding_top + original_height,
            padding_left : padding_left + original_width,
        ] = tensor
        tensor = padded
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode(), _autocast_context(model_device):
            predicted_logits = torch.zeros(
                (tensor.shape[0], 2, padded_height, padded_width),
                device=model_device,
                dtype=torch.half,
            )
            n_predictions = torch.zeros(
                (padded_height, padded_width),
                device=model_device,
                dtype=torch.half,
            )
            gaussian = compute_gaussian(
                (patch_height, patch_width),
                sigma_scale=1.0 / 8.0,
                value_scaling_factor=10.0,
                dtype=torch.half,
                device=model_device,
            )
            y_starts = _tile_starts(padded_height, patch_height, tile_step_size)
            x_starts = _tile_starts(padded_width, patch_width, tile_step_size)
            for y_start in y_starts:
                y_end = y_start + patch_height
                for x_start in x_starts:
                    x_end = x_start + patch_width
                    tile = tensor[:, :, y_start:y_end, x_start:x_end]
                    prediction = _predict_tile_logits(
                        model,
                        tile,
                        mirror_axes=tuple(mirror_axes),
                    ).to(device=model_device)
                    prediction *= gaussian
                    predicted_logits[:, :, y_start:y_end, x_start:x_end] += prediction
                    n_predictions[y_start:y_end, x_start:x_end] += gaussian
            if torch.any(n_predictions == 0):
                raise RuntimeError("tile aggregation did not cover the complete image")
            torch.div(
                predicted_logits,
                n_predictions.view(1, 1, padded_height, padded_width),
                out=predicted_logits,
            )
            return predicted_logits[
                :,
                :,
                padding_top : padding_top + height,
                padding_left : padding_left + width,
            ]
    finally:
        model.train(was_training)


def predict_volume(
    model: nn.Module,
    image: NiftiVolume,
    device: torch.device,
    *,
    mirror_axes: tuple[int, ...] = DEFAULT_MIRROR_AXES,
    patch_size: tuple[int, int] = DEFAULT_PATCH_SIZE,
    tile_step_size: float = DEFAULT_TILE_STEP_SIZE,
    slice_batch_size: int = 1,
) -> np.ndarray:
    """Return one binary uint8 prediction per source-space ``(z, y, x)`` slice."""
    if slice_batch_size <= 0:
        raise ValueError(f"slice_batch_size must be positive, got {slice_batch_size}")
    normalized = z_score_normalize(image.array)
    slice_count, height, width = normalized.shape
    total_logits: Tensor | None = None
    for z_start in range(0, slice_count, slice_batch_size):
        z_end = min(z_start + slice_batch_size, slice_count)
        tensor = torch.from_numpy(normalized[z_start:z_end]).unsqueeze(1)
        logits = predict_logits_2d(
            model,
            tensor,
            device,
            mirror_axes=tuple(mirror_axes),
            patch_size=patch_size,
            tile_step_size=tile_step_size,
        )
        expected_shape = (z_end - z_start, 2, height, width)
        if tuple(logits.shape) != expected_shape:
            raise ValueError(
                "prediction logits shape does not match input slice batch: "
                f"expected {expected_shape}, got {tuple(logits.shape)}"
            )
        if total_logits is None:
            total_logits = torch.zeros(
                (slice_count, 2, height, width),
                device=logits.device,
                dtype=logits.dtype,
            )
        total_logits[z_start:z_end] += logits
    if total_logits is None:
        return np.empty(image.array.shape, dtype=np.uint8)
    return torch.argmax(total_logits, dim=1).cpu().numpy().astype(np.uint8)


def save_and_validate_prediction(
    path: Path,
    prediction: np.ndarray,
    reference: NiftiVolume,
) -> dict[str, object]:
    """Write and read back a binary source-space prediction with metadata checks."""
    if prediction.shape != reference.array.shape:
        raise ValueError("prediction shape must match reference volume")
    if not np.isin(prediction, (0, 1)).all():
        raise ValueError("prediction labels must contain only 0 and 1")
    volume = NiftiVolume(
        prediction.astype(np.uint8, copy=False),
        reference.spacing_xyz,
        reference.origin_xyz,
        reference.direction,
    )
    write_nifti(path, volume)
    restored = read_nifti(path)
    checks = {
        "shape_preserved": restored.array.shape == reference.array.shape,
        "dtype_uint8": restored.array.dtype == np.uint8,
        "binary": bool(np.isin(restored.array, (0, 1)).all()),
        "spacing_preserved": bool(np.allclose(restored.spacing_xyz, reference.spacing_xyz, rtol=0.0, atol=1e-6)),
        "origin_preserved": bool(np.allclose(restored.origin_xyz, reference.origin_xyz, rtol=0.0, atol=1e-6)),
        "direction_preserved": bool(np.allclose(restored.direction, reference.direction, rtol=0.0, atol=1e-6)),
    }
    if not all(checks.values()):
        raise ValueError(f"saved prediction NIfTI validation failed: {checks}")
    return {"passed": True, **checks}
