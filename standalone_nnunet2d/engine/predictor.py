"""Full-resolution, case-level 2D inference for arrays ordered as (z, y, x)."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn

from standalone_nnunet2d.data.nifti_io import NiftiVolume, read_nifti, write_nifti
from standalone_nnunet2d.data.preprocessing import z_score_normalize


DEFAULT_MIRROR_AXES = (0, 1)
DEFAULT_PATCH_SIZE = (512, 512)
DEFAULT_TILE_STEP_SIZE = 0.5


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


def _tile_starts(size: int, patch: int, step: int) -> tuple[int, ...]:
    if size <= patch:
        return (0,)
    starts = list(range(0, size - patch + 1, step))
    final_start = size - patch
    if starts[-1] != final_start:
        starts.append(final_start)
    return tuple(starts)


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
    spatial_dims = tuple(axis + 2 for axis in mirror_axes)
    logits = _full_resolution_logits(model(tile))
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
    step_height = max(1, int(round(patch_height * tile_step_size)))
    step_width = max(1, int(round(patch_width * tile_step_size)))
    y_starts = _tile_starts(height, min(patch_height, height), step_height)
    x_starts = _tile_starts(width, min(patch_width, width), step_width)

    model_device = device
    tensor = image.to(model_device)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            accumulator: Tensor | None = None
            weights = torch.zeros((1, 1, height, width), device=model_device, dtype=torch.float32)
            for y_start in y_starts:
                y_end = min(y_start + patch_height, height)
                for x_start in x_starts:
                    x_end = min(x_start + patch_width, width)
                    tile = tensor[:, :, y_start:y_end, x_start:x_end]
                    tile_logits = _predict_tile_logits(model, tile, mirror_axes=tuple(mirror_axes))
                    if accumulator is None:
                        accumulator = torch.zeros(
                            (tile_logits.shape[0], tile_logits.shape[1], height, width),
                            device=tile_logits.device,
                            dtype=tile_logits.dtype,
                        )
                    if tile_logits.shape[0] != accumulator.shape[0]:
                        raise ValueError("model changed batch size between tiles")
                    accumulator[:, :, y_start:y_end, x_start:x_end] += tile_logits
                    weights[:, :, y_start:y_end, x_start:x_end] += 1.0
            if accumulator is None or torch.any(weights == 0):
                raise RuntimeError("tile aggregation did not cover the complete image")
            return accumulator / weights.to(dtype=accumulator.dtype)
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
) -> np.ndarray:
    """Return one binary uint8 prediction per source-space ``(z, y, x)`` slice."""
    normalized = z_score_normalize(image.array)
    prediction = np.empty(image.array.shape, dtype=np.uint8)
    for z_index in range(normalized.shape[0]):
        tensor = torch.from_numpy(normalized[z_index]).unsqueeze(0).unsqueeze(0)
        logits = predict_logits_2d(
            model,
            tensor,
            device,
            mirror_axes=mirror_axes,
            patch_size=patch_size,
            tile_step_size=tile_step_size,
        )
        mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        if mask.shape != prediction[z_index].shape:
            raise ValueError("prediction slice shape does not match input slice")
        prediction[z_index] = mask
    return prediction


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
