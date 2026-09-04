"""Explicit performance profiles and loader settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch.utils.data import DataLoader, Dataset


PerformanceProfile = Literal["alignment", "throughput"]
PinMemoryMode = Literal["auto", "on", "off"]


@dataclass(frozen=True)
class PerformanceConfig:
    profile: PerformanceProfile
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int | None
    non_blocking: bool
    amp: bool = False
    tf32: bool = False
    compile: bool = False

    def loader_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
        }
        if self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs

    def as_dict(self) -> dict[str, object]:
        return {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "prefetch_factor": self.prefetch_factor,
            "non_blocking": self.non_blocking,
        }


def resolve_performance_config(
    profile: str,
    *,
    device: torch.device | str,
    num_workers: int | None = None,
    pin_memory: str = "auto",
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
) -> PerformanceConfig:
    if profile not in ("alignment", "throughput"):
        raise ValueError("performance profile must be 'alignment' or 'throughput'")
    if pin_memory not in ("auto", "on", "off"):
        raise ValueError("pin_memory must be one of: auto, on, off")

    resolved_workers = (0 if profile == "alignment" else 2) if num_workers is None else num_workers
    if resolved_workers < 0:
        raise ValueError("num_workers must be non-negative")

    if persistent_workers is None:
        resolved_persistent = profile == "throughput" and resolved_workers > 0
    else:
        resolved_persistent = persistent_workers
    if resolved_persistent and resolved_workers == 0:
        raise ValueError("persistent_workers requires num_workers > 0")

    if prefetch_factor is None:
        resolved_prefetch = 2 if profile == "throughput" and resolved_workers > 0 else None
    else:
        resolved_prefetch = prefetch_factor
    if resolved_prefetch is not None and resolved_prefetch <= 0:
        raise ValueError("prefetch_factor must be a positive integer")
    if resolved_prefetch is not None and resolved_workers == 0:
        raise ValueError("prefetch_factor requires num_workers > 0")

    device_type = torch.device(device).type
    resolved_pin_memory = (
        profile == "throughput" and device_type == "cuda" if pin_memory == "auto" else pin_memory == "on"
    )
    return PerformanceConfig(
        profile=profile,  # type: ignore[arg-type]
        num_workers=resolved_workers,
        pin_memory=resolved_pin_memory,
        persistent_workers=resolved_persistent,
        prefetch_factor=resolved_prefetch,
        non_blocking=resolved_pin_memory and device_type == "cuda",
    )


def build_formal_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    *,
    performance: PerformanceConfig,
    batch_size: int = 12,
) -> tuple[DataLoader, DataLoader]:
    loader_kwargs = performance.loader_kwargs()
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs),
    )
