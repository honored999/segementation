"""Convert CI-1 ADC volumes and ADC masks into a 2D slice dataset.

中文说明：这个脚本基于已有 DWI 索引表中的患者、时间点和 DICOM 目录，
读取 DICOM 中的 ADC 原图序列以及同时间点的 ADC mask，并按
axial/coronal/sagittal 指定视角切成二维分割数据集。它适合快速生成
axial ADC + ADC mask 的训练数据，用来和 DWI、FLAIR 结果做对比。
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image

from audit_ci1_dwi_adc_masks import (
    background_predicate,
    read_mask_on_reference,
    read_series_image,
)
from convert_ci1_dwi_to_2d import (
    IndexRow,
    image_to_array,
    normalize_volume_to_uint8,
    read_index,
    safe_name,
)


VIEWS = ("axial", "coronal", "sagittal")
TIMEPOINT_RE = re.compile(r"(D\d+)", re.IGNORECASE)


def extract_timepoint(path: Path) -> str | None:
    """Extract a D1/D2/D3/D7-style timepoint from a filename."""
    match = TIMEPOINT_RE.search(path.name)
    if match is None:
        return None
    return match.group(1).upper()


def find_adc_segmentation_path(
    dwi_segmentation_path: Path,
    path_exists: Callable[[Path], bool] | None = None,
    glob_paths: Callable[[str], Sequence[Path]] | None = None,
) -> Path | None:
    """Find the ADC mask matching a DWI index row."""
    exists = path_exists or Path.exists
    globber = glob_paths or (lambda pattern: list(dwi_segmentation_path.parent.glob(pattern)))
    name = dwi_segmentation_path.name

    candidates: list[Path] = []
    for old, new in [("DWI", "ADC"), ("dwi", "adc")]:
        if old in name:
            candidates.append(dwi_segmentation_path.with_name(name.replace(old, new, 1)))

    for candidate in candidates:
        if exists(candidate):
            return candidate

    timepoint = extract_timepoint(dwi_segmentation_path)
    if timepoint is None:
        return None

    glob_candidates = [
        path
        for path in globber(f"*{timepoint}*ADC*.nii*")
        if exists(path) and "flair-adc" not in path.name.lower()
    ]
    if not glob_candidates:
        return None
    return sorted(glob_candidates, key=lambda item: (len(item.parts), str(item)))[0]


def extract_view_slice(volume: np.ndarray, view: str, index: int) -> np.ndarray:
    """Extract a 2D slice from a z/y/x volume."""
    if view == "axial":
        return volume[index, :, :]
    if view == "coronal":
        return volume[:, index, :]
    if view == "sagittal":
        return volume[:, :, index]
    raise ValueError(f"Unsupported view: {view}")


def mask_area_by_view(mask: np.ndarray, view: str) -> np.ndarray:
    """Count mask pixels for every slice in the requested view."""
    if view == "axial":
        return mask.reshape(mask.shape[0], -1).sum(axis=1)
    if view == "coronal":
        return mask.sum(axis=(0, 2))
    if view == "sagittal":
        return mask.sum(axis=(0, 1))
    raise ValueError(f"Unsupported view: {view}")


def write_png(array: np.ndarray, path: Path) -> None:
    """Write an array as PNG, creating parent directories when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def convert_case(
    row: IndexRow,
    output_root: Path,
    view: str,
    positive_only: bool,
) -> list[dict[str, str]]:
    """Convert one case into 2D ADC image/mask PNG slices."""
    adc_segmentation_path = find_adc_segmentation_path(row.segmentation_path)
    if adc_segmentation_path is None:
        raise FileNotFoundError(
            f"No ADC mask found for {row.patient} {row.timepoint}: {row.segmentation_path}"
        )

    image, series_description = read_series_image(row.dicom_dir, background_predicate("adc"))
    segmentation_mask = read_mask_on_reference(adc_segmentation_path, image)
    image_array = image_to_array(image)

    if image_array.shape != segmentation_mask.shape:
        raise RuntimeError(
            "Image/mask shape mismatch after resampling: "
            f"{image_array.shape} vs {segmentation_mask.shape} for {adc_segmentation_path}"
        )

    image_uint8 = normalize_volume_to_uint8(image_array)
    mask_uint8 = segmentation_mask.astype(np.uint8) * 255
    areas = mask_area_by_view(segmentation_mask, view)

    case_id = f"{safe_name(row.patient)}_{row.timepoint}_ADC_{view}"
    manifest_rows: list[dict[str, str]] = []
    for slice_index, mask_area in enumerate(areas):
        mask_area_int = int(mask_area)
        if positive_only and mask_area_int == 0:
            continue

        image_slice = extract_view_slice(image_uint8, view, slice_index)
        mask_slice = extract_view_slice(mask_uint8, view, slice_index)
        stem = f"{case_id}_{view}{slice_index:03d}"
        image_path = output_root / "images" / f"{stem}.png"
        mask_path = output_root / "masks" / f"{stem}.png"
        write_png(image_slice, image_path)
        write_png(mask_slice, mask_path)

        manifest_rows.append(
            {
                "patient": row.patient,
                "timepoint": row.timepoint,
                "modality": "ADC",
                "view": view,
                "slice_index": str(slice_index),
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "mask_area": str(mask_area_int),
                "has_mask": "1" if mask_area_int > 0 else "0",
                "source_dicom_dir": str(row.dicom_dir),
                "source_series_description": series_description,
                "source_segmentation_path": str(adc_segmentation_path),
            }
        )

    return manifest_rows


def write_manifest(rows: list[dict[str, str]], output_path: Path) -> None:
    """Write manifest.csv for the generated dataset."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "patient",
        "timepoint",
        "modality",
        "view",
        "slice_index",
        "image_path",
        "mask_path",
        "mask_area",
        "has_mask",
        "source_dicom_dir",
        "source_series_description",
        "source_segmentation_path",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert CI-1 ADC DICOM/mask pairs to 2D PNG slices."
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=Path("data") / "ci1_dwi_index.csv",
        help="DWI index CSV used for patient/timepoint/DICOM directory matching.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data") / "ci1_adc_axial_2d",
    )
    parser.add_argument("--view", choices=VIEWS, default="axial")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--positive-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    rows = read_index(args.index_csv)
    if args.limit > 0:
        rows = rows[: args.limit]

    all_manifest_rows: list[dict[str, str]] = []
    failed_rows: list[tuple[IndexRow, str]] = []
    for case_number, row in enumerate(rows, start=1):
        print(f"[{case_number}/{len(rows)}] {row.patient} {row.timepoint}")
        try:
            all_manifest_rows.extend(
                convert_case(
                    row=row,
                    output_root=args.output_root,
                    view=args.view,
                    positive_only=args.positive_only,
                )
            )
        except Exception as exc:  # noqa: BLE001 - keep batch conversion moving.
            failed_rows.append((row, str(exc)))
            print(f"  [FAILED] {exc}")

    manifest_path = args.output_root / "manifest.csv"
    write_manifest(all_manifest_rows, manifest_path)

    positive_slices = sum(1 for row in all_manifest_rows if row["has_mask"] == "1")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Cases requested: {len(rows)}")
    print(f"Cases failed: {len(failed_rows)}")
    print(f"Slices written: {len(all_manifest_rows)}")
    print(f"Positive mask slices: {positive_slices}")

    if failed_rows:
        failed_path = args.output_root / "failed_cases.csv"
        with failed_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["patient", "timepoint", "segmentation_path", "error"])
            for row, error in failed_rows:
                writer.writerow([row.patient, row.timepoint, row.segmentation_path, error])
        print(f"Wrote failures: {failed_path}")


if __name__ == "__main__":
    main()
