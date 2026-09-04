"""Read-only readiness checks for an explicit standalone server dry run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from standalone_nnunet2d.config import DEFAULT_REFERENCE_DIR
from standalone_nnunet2d.tools.inspect_reference import inspect_reference


def inspect_server_readiness(
    raw_root: Path,
    preprocessed_root: Path,
    results_root: Path,
    *,
    device: str | None = None,
) -> dict[str, Any]:
    """Return JSON-safe local readiness facts without scanning dataset contents."""
    normalized_paths = {
        "raw": Path(raw_root).expanduser().resolve(),
        "preprocessed": Path(preprocessed_root).expanduser().resolve(),
        "results": Path(results_root).expanduser().resolve(),
    }
    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    selected_device = torch.device(device or ("cuda" if cuda_available else "cpu"))
    diagnostics = [
        f"{name} directory is missing: {path}"
        for name, path in normalized_paths.items()
        if not path.is_dir()
    ]
    if normalized_paths["raw"].is_dir():
        diagnostics.extend(
            f"raw directory is missing required child: {name}"
            for name in ("imagesTr", "labelsTr")
            if not (normalized_paths["raw"] / name).is_dir()
        )
    diagnostics.extend(_device_diagnostics(selected_device, cuda_available, gpu_count))
    plan = _plan_facts(diagnostics)
    return {
        "ready": not diagnostics,
        "diagnostics": diagnostics,
        "paths": {name: str(path) for name, path in normalized_paths.items()},
        "device": {
            "selected": str(selected_device),
            "cuda_available": cuda_available,
            "gpu_count": gpu_count,
            "gpu_names": [torch.cuda.get_device_name(index) for index in range(gpu_count)],
        },
        "plan": plan,
    }


def _device_diagnostics(device: torch.device, cuda_available: bool, gpu_count: int) -> list[str]:
    if device.type != "cuda":
        return []
    index = 0 if device.index is None else device.index
    if not cuda_available or index < 0 or index >= gpu_count:
        return [f"requested CUDA device is unavailable: {device}"]
    return []


def _plan_facts(diagnostics: list[str]) -> dict[str, Any]:
    try:
        inspection = inspect_reference(DEFAULT_REFERENCE_DIR)
    except (OSError, ValueError, KeyError) as error:
        diagnostics.append(f"reference plan is unreadable: {error}")
        return {}
    return {
        "dataset_name": inspection.dataset_name,
        "patch_size": list(inspection.patch_size),
        "batch_size": inspection.batch_size,
        "stages": inspection.n_stages,
    }
