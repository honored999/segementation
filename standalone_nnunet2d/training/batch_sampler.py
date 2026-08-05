"""Deterministic batch-level patch decisions for the formal 2D pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PatchRequest:
    case_id: str
    force_foreground: bool
    z_index: int
    center_yx: tuple[int, int]


class FormalBatchSampler:
    """Create stateless, reproducible patch requests for one batch index."""

    def __init__(
        self,
        *,
        case_ids: tuple[str, ...],
        batch_size: int,
        foreground_slots: tuple[int, ...] = (),
        seed: int = 0,
    ) -> None:
        if not case_ids:
            raise ValueError("case_ids must not be empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if any(slot < 0 or slot >= batch_size for slot in foreground_slots):
            raise ValueError("foreground slots must be valid batch positions")
        self.case_ids = tuple(case_ids)
        self.batch_size = int(batch_size)
        self.foreground_slots = tuple(foreground_slots)
        self.seed = int(seed)

    def batch(self, batch_index: int) -> tuple[PatchRequest, ...]:
        if batch_index < 0:
            raise ValueError("batch_index must be non-negative")
        foreground_slots = set(self.foreground_slots)
        return tuple(
            PatchRequest(
                case_id=self.case_ids[(batch_index * self.batch_size + slot) % len(self.case_ids)],
                force_foreground=slot in foreground_slots,
                z_index=-1,
                center_yx=(-1, -1),
            )
            for slot in range(self.batch_size)
        )


def choose_patch_location(
    labels: np.ndarray,
    rng: np.random.Generator,
    *,
    force_foreground: bool = False,
) -> tuple[int, tuple[int, int]]:
    """Choose a valid voxel, preferring a foreground voxel when requested."""
    if labels.ndim != 3 or any(size <= 0 for size in labels.shape):
        raise ValueError("labels must be a non-empty (z, y, x) array")
    if force_foreground:
        foreground = np.argwhere(labels > 0)
        if len(foreground):
            z_index, y, x = foreground[int(rng.integers(len(foreground)))]
            return int(z_index), (int(y), int(x))
    z_index = int(rng.integers(labels.shape[0]))
    center_y = int(rng.integers(labels.shape[1]))
    center_x = int(rng.integers(labels.shape[2]))
    return z_index, (center_y, center_x)
