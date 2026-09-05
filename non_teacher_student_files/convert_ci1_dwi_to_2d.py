"""Convert indexed CI-1 DWI volumes into a 2D slice segmentation dataset.

中文说明：
这个脚本根据 DWI 索引表读取 DICOM 原图和 NIfTI 标注，
按 axial 方向切成二维 PNG 图像和 mask，生成可用于 2D 分割训练的数据集。

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
POSITION_TAG = "0020|0032"
ORIENTATION_TAG = "0020|0037"
INSTANCE_NUMBER_TAG = "0020|0013"


@dataclass(frozen=True)
class IndexRow:
    patient: str
    timepoint: str
    modality: str
    segmentation_path: Path
    dicom_dir: Path


@dataclass(frozen=True)
class DicomSliceInfo:
    file_path: str
    slice_position: float
    instance_number: int


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


def read_vector_metadata(path: str, tag: str) -> np.ndarray | None:
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    reader.ReadImageInformation()
    if not reader.HasMetaDataKey(tag):
        return None
    values = [float(item) for item in reader.GetMetaData(tag).split("\\")]
    return np.asarray(values, dtype=np.float64)


def read_int_metadata(path: str, tag: str, default: int) -> int:
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    reader.ReadImageInformation()
    if not reader.HasMetaDataKey(tag):
        return default
    try:
        return int(float(reader.GetMetaData(tag).strip()))
    except ValueError:
        return default


def normal_from_orientation(file_path: str) -> np.ndarray | None:
    orientation = read_vector_metadata(file_path, ORIENTATION_TAG)
    if orientation is None or orientation.size != 6:
        return None
    row_direction = orientation[:3]
    col_direction = orientation[3:]
    normal = np.cross(row_direction, col_direction)
    normal_norm = np.linalg.norm(normal)
    if normal_norm == 0:
        return None
    return normal / normal_norm


def collect_slice_infos(files: list[str]) -> list[DicomSliceInfo]:
    if not files:
        return []

    normal = normal_from_orientation(files[0])
    if normal is None:
        return [
            DicomSliceInfo(file_path=file_path, slice_position=float(index), instance_number=index)
            for index, file_path in enumerate(files)
        ]

    slice_infos: list[DicomSliceInfo] = []
    for index, file_path in enumerate(files):
        position = read_vector_metadata(file_path, POSITION_TAG)
        if position is None or position.size != 3:
            slice_position = float(index)
        else:
            slice_position = float(np.dot(position, normal))
        instance_number = read_int_metadata(file_path, INSTANCE_NUMBER_TAG, index)
        slice_infos.append(
            DicomSliceInfo(
                file_path=file_path,
                slice_position=slice_position,
                instance_number=instance_number,
            )
        )
    return slice_infos


def select_unique_slice_files(
    slices: list[DicomSliceInfo],
    position_tolerance: float = 1e-6,
) -> list[str]:
    selected: list[str] = []
    seen_positions: set[int] = set()
    sorted_slices = sorted(
        slices,
        key=lambda item: (item.slice_position, item.instance_number, item.file_path),
    )

    for slice_info in sorted_slices:
        position_key = int(round(slice_info.slice_position / position_tolerance))
        if position_key in seen_positions:
            continue
        seen_positions.add(position_key)
        selected.append(slice_info.file_path)

    return selected


def get_largest_series_files(dicom_dir: Path) -> list[str]:
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
    return best_files


def read_largest_dicom_series(dicom_dir: Path) -> sitk.Image:
    best_files = get_largest_series_files(dicom_dir)
    selected_files = select_unique_slice_files(collect_slice_infos(best_files))

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(selected_files)
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
