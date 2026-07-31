"""Report environment and read-only data-path availability without scanning images."""

from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path

import numpy
import torch


RAW_DATASET = Path(r"C:\lijialin\models3d\nnUNet\nnUNet_raw\Dataset501_StrokeLesion")
PREPROCESSED_DATASET = Path(r"C:\lijialin\models3d\nnUNet\nnUNet_preprocessed\Dataset501_StrokeLesion")
RESULTS_DATASET = Path(r"C:\lijialin\models3d\nnUNet\nnUNet_results\Dataset501_StrokeLesion")


def _git(*arguments: str) -> str:
    result = subprocess.run(["git", *arguments], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else f"unavailable ({result.stderr.strip()})"


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def main() -> None:
    cuda_available = torch.cuda.is_available()
    report = {
        "python_version": sys.version.replace("\n", " "),
        "pytorch_version": torch.__version__,
        "cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count() if cuda_available else 0,
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())] if cuda_available else [],
        "working_directory": str(Path.cwd()),
        "git_toplevel": _git("rev-parse", "--show-toplevel"),
        "git_branch": _git("branch", "--show-current"),
        "simpleitk_version": _version("SimpleITK"),
        "numpy_version": numpy.__version__,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", "not set"),
        "raw_dataset_exists": RAW_DATASET.exists(),
        "imagesTr_exists": (RAW_DATASET / "imagesTr").is_dir(),
        "labelsTr_exists": (RAW_DATASET / "labelsTr").is_dir(),
        "preprocessed_dataset_exists": PREPROCESSED_DATASET.exists(),
        "results_dataset_exists": RESULTS_DATASET.exists(),
    }
    for name, value in report.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
