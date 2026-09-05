"""Inspect CI-1 FLAIR-related NIfTI files to distinguish masks from images.

中文说明：
检查 CI-1 中 FLAIR 相关 NIfTI 文件的体素值分布，用于区分它们是原图还是 mask。
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def ascii_nifti_name(path: Path) -> str:
    if path.name.lower().endswith(".nii.gz"):
        return "sample.nii.gz"
    return "sample.nii"


def read_image_with_ascii_copy(path: Path) -> sitk.Image:
    try:
        return sitk.ReadImage(str(path))
    except RuntimeError:
        with tempfile.TemporaryDirectory(prefix="ci1_flair_") as tmp_dir:
            tmp_path = Path(tmp_dir) / ascii_nifti_name(path)
            shutil.copy2(path, tmp_path)
            return sitk.ReadImage(str(tmp_path))


def classify_values(array: np.ndarray) -> str:
    unique_values = np.unique(array)
    if unique_values.size <= 16 and np.all(np.isclose(unique_values, np.round(unique_values))):
        return "mask_like"
    return "image_like"


def inspect_file(path: Path) -> dict[str, str]:
    image = read_image_with_ascii_copy(path)
    array = sitk.GetArrayFromImage(image)
    unique_values = np.unique(array)
    finite = array[np.isfinite(array)]
    return {
        "path": str(path),
        "kind": classify_values(array),
        "size": str(image.GetSize()),
        "shape": str(array.shape),
        "dtype": str(array.dtype),
        "unique_count": str(unique_values.size),
        "min": str(float(finite.min())) if finite.size else "nan",
        "max": str(float(finite.max())) if finite.size else "nan",
        "first_values": str(unique_values[:8].tolist()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect CI-1 FLAIR NIfTI files.")
    parser.add_argument("--ci1-root", type=Path, default=Path("CI-1"))
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--summary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.ci1_root.rglob("*FLAIR*.nii*"), key=lambda item: str(item))
    print(f"FLAIR-related NIfTI files: {len(paths)}")
    if args.summary:
        counts: Counter[str] = Counter()
        failures = 0
        examples: dict[str, str] = {}
        for path in paths:
            try:
                row = inspect_file(path)
            except Exception:
                failures += 1
                continue
            counts[row["kind"]] += 1
            examples.setdefault(row["kind"], row["path"])
        print(f"summary: {dict(counts)}")
        print(f"failures: {failures}")
        print(f"examples: {examples}")
        return

    for path in paths[: args.limit]:
        try:
            row = inspect_file(path)
        except Exception as exc:  # noqa: BLE001 - keep inspection moving.
            print(f"FAILED\t{path}\t{exc}")
            continue
        print(
            "{kind}\tunique={unique_count}\tmin={min}\tmax={max}\t"
            "size={size}\t{path}".format(**row)
        )


if __name__ == "__main__":
    main()
