"""Convert indexed CI-1 DWI volumes into a 2D slice segmentation dataset.

Inputs:
    data/ci1_dwi_index.csv from prepare_ci1_dwi_dataset.py

Outputs:
    data/ci1_dwi_2d/
        images/*.png
        masks/*.png
        manifest.csv

The source CI-1 directory is read-only. Segmentations with multiple segments
are collapsed into one binary lesion mask.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image


SAFE_NAME_RE = re.compile(r"[^0-9A-Za-z_.-]+")


@dataclass(frozen=True)
class IndexRow:
    patient: str
    timepoint: str
    modality: str
    segmentation_path: Path
    dicom_dir: Path


def safe_name(text: str) -> str:
    encoded = text.encode("utf-8").hex()[:16]
    readable = SAFE_NAME_RE.sub("_", text).strip("_")
    if readable:
        return readable
    return encoded


def read_index(index_csv: Path) -> list[IndexRow]:
    rows: list[IndexRow] = []
    with index_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("match_status") != "matched":
                continue
            if row.get("modality", "").upper() != "DWI":
                continue
            rows.append(
                IndexRow(
                    patient=row["patient"],
                    timepoint=row["timepoint"],
                    modality=row["modality"],
                    segmentation_path=Path(row["segmentation_path"]),
                    dicom_dir=Path(row["dicom_dir"]),
                )
            )
    return rows


def read_largest_dicom_series(dicom_dir: Path) -> sitk.Image:
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in: {dicom_dir}")

    best_files: list[str] = []
    for series_id in series_ids:
        files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
            str(dicom_dir), series_id
        )
        if len(files) > len(best_files):
            best_files = list(files)

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(best_files)
    return reader.Execute()


def read_segmentation(segmentation_path: Path, reference: sitk.Image) -> sitk.Image:
    try:
        segmentation = sitk.ReadImage(str(segmentation_path))
    except RuntimeError:
        with tempfile.TemporaryDirectory(prefix="ci1_seg_") as tmp_dir:
            tmp_path = Path(tmp_dir) / ascii_nifti_name(segmentation_path)
            shutil.copy2(segmentation_path, tmp_path)
            segmentation = sitk.ReadImage(str(tmp_path))
    if segmentation.GetSize() != reference.GetSize():
        segmentation = sitk.Resample(
            segmentation,
            reference,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            segmentation.GetPixelID(),
        )
    return segmentation


def ascii_nifti_name(path: Path) -> str:
    lower_name = path.name.lower()
    if lower_name.endswith(".nii.gz"):
        return "segmentation.nii.gz"
    if lower_name.endswith(".nii"):
        return "segmentation.nii"
    return "segmentation.nii.gz"


def image_to_array(image: sitk.Image) -> np.ndarray:
    array = sitk.GetArrayFromImage(image)
    if array.ndim == 4:
        if array.shape[-1] <= 16:
            array = np.any(array != 0, axis=-1)
        else:
            array = np.any(array != 0, axis=0)
    return np.asarray(array)


def normalize_volume_to_uint8(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32, copy=False)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros(array.shape, dtype=np.uint8)

    low, high = np.percentile(finite, [1.0, 99.0])
    if high <= low:
        low = float(finite.min())
        high = float(finite.max())
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)

    clipped = np.clip(array, low, high)
    scaled = (clipped - low) / (high - low)
    return (scaled * 255.0).round().astype(np.uint8)


def write_png(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def convert_case(
    row: IndexRow,
    output_root: Path,
    positive_only: bool,
) -> list[dict[str, str]]:
    image = read_largest_dicom_series(row.dicom_dir)
    segmentation = read_segmentation(row.segmentation_path, image)

    image_array = image_to_array(image)
    mask_array = image_to_array(segmentation) != 0

    if image_array.shape != mask_array.shape:
        raise RuntimeError(
            "Image/mask shape mismatch after resampling: "
            f"{image_array.shape} vs {mask_array.shape} for {row.segmentation_path}"
        )

    image_uint8 = normalize_volume_to_uint8(image_array)
    mask_uint8 = (mask_array.astype(np.uint8) * 255)

    case_id = f"{safe_name(row.patient)}_{row.timepoint}_DWI"
    manifest_rows: list[dict[str, str]] = []
    num_slices = image_uint8.shape[0]
    for slice_index in range(num_slices):
        mask_area = int(mask_array[slice_index].sum())
        if positive_only and mask_area == 0:
            continue

        stem = f"{case_id}_z{slice_index:03d}"
        image_path = output_root / "images" / f"{stem}.png"
        mask_path = output_root / "masks" / f"{stem}.png"
        write_png(image_uint8[slice_index], image_path)
        write_png(mask_uint8[slice_index], mask_path)

        manifest_rows.append(
            {
                "patient": row.patient,
                "timepoint": row.timepoint,
                "modality": row.modality,
                "slice_index": str(slice_index),
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "mask_area": str(mask_area),
                "has_mask": "1" if mask_area > 0 else "0",
                "source_dicom_dir": str(row.dicom_dir),
                "source_segmentation_path": str(row.segmentation_path),
            }
        )

    return manifest_rows


def write_manifest(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "patient",
        "timepoint",
        "modality",
        "slice_index",
        "image_path",
        "mask_path",
        "mask_area",
        "has_mask",
        "source_dicom_dir",
        "source_segmentation_path",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CI-1 DWI DICOM/segmentation pairs to 2D PNG slices."
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=Path("data") / "ci1_dwi_index.csv",
        help="CSV generated by prepare_ci1_dwi_dataset.py.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data") / "ci1_dwi_2d",
        help="Directory for generated 2D images, masks, and manifest.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Convert only the first N indexed cases. Use 0 for all cases.",
    )
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Save only slices with non-empty masks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_index(args.index_csv)
    if args.limit > 0:
        rows = rows[: args.limit]

    all_manifest_rows: list[dict[str, str]] = []
    failed_rows: list[tuple[IndexRow, str]] = []
    for case_number, row in enumerate(rows, start=1):
        print(
            f"[{case_number}/{len(rows)}] {row.patient} {row.timepoint} "
            f"{row.segmentation_path}"
        )
        try:
            all_manifest_rows.extend(
                convert_case(row, args.output_root, args.positive_only)
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
                writer.writerow(
                    [row.patient, row.timepoint, row.segmentation_path, error]
                )
        print(f"Wrote failures: {failed_path}")


if __name__ == "__main__":
    main()
