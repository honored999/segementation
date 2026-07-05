"""Create overlay figures comparing CI-1 DWI and ADC segmentation masks.

中文说明：
生成 CI-1 中 DWI mask 和 ADC mask 的对比覆盖图，
用于检查两套标注在不同背景模态和不同视角下的重合关系。
"""

from __future__ import annotations

import argparse
import csv
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from convert_ci1_dwi_to_2d import (
    collect_slice_infos,
    image_to_array,
    normalize_volume_to_uint8,
    select_unique_slice_files,
)


SERIES_DESCRIPTION_TAG = "0008|103e"
VIEWS = ("axial", "coronal", "sagittal")


def is_dwi_series_description(description: str) -> bool:
    """判断 DICOM 序列描述是否是 DWI 原图序列。"""
    lower_description = description.lower()
    return "dwi" in lower_description and "adc" not in lower_description


def is_adc_series_description(description: str) -> bool:
    """判断 DICOM 序列描述是否是 ADC 原图序列。"""
    return "adc" in description.lower()


def is_flair_series_description(description: str) -> bool:
    """判断 DICOM 序列描述是否是 FLAIR 原图序列，并排除 T1-Flair。"""
    lower_description = description.lower()
    return "flair" in lower_description and "t1" not in lower_description


def background_predicate(background: str) -> Callable[[str], bool]:
    """根据命令行选择的背景模态，返回对应的 DICOM 序列筛选函数。"""
    if background.lower() == "adc":
        return is_adc_series_description
    if background.lower() == "dwi":
        return is_dwi_series_description
    if background.lower() == "flair":
        return is_flair_series_description
    raise ValueError(f"Unsupported background: {background}")


def find_adc_segmentation_path(
    dwi_segmentation_path: Path,
    path_exists: Callable[[Path], bool] | None = None,
) -> Path | None:
    """根据 DWI mask 文件名，寻找同一病例和时间点对应的 ADC mask 文件。"""
    exists = path_exists or Path.exists
    name = dwi_segmentation_path.name
    candidates: list[Path] = []
    for old, new in [
        ("DWI", "ADC"),
        ("dwi", "adc"),
        ("DWI", "DWIADC"),
        ("dwi", "dwiadc"),
    ]:
        if old in name:
            candidates.append(dwi_segmentation_path.with_name(name.replace(old, new, 1)))

    for candidate in candidates:
        if exists(candidate):
            return candidate
    return None


def read_index_rows(index_csv: Path) -> list[dict[str, str]]:
    """读取 DWI 索引表，只保留已经成功匹配 DICOM 目录的病例。"""
    with index_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row.get("match_status") == "matched"
        ]


def series_description(dicom_file: str) -> str:
    """读取单个 DICOM 文件里的 Series Description，用来区分 DWI/ADC/FLAIR 等序列。"""
    reader = sitk.ImageFileReader()
    reader.SetFileName(dicom_file)
    reader.ReadImageInformation()
    if reader.HasMetaDataKey(SERIES_DESCRIPTION_TAG):
        return reader.GetMetaData(SERIES_DESCRIPTION_TAG).strip()
    return ""


def find_series_files(
    dicom_dir: Path,
    predicate: Callable[[str], bool],
) -> tuple[list[str], str]:
    """在一个 DICOM 目录里找到符合条件的序列，并去掉重复切片后返回文件列表。"""
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_dir)) or []
    matches: list[tuple[int, list[str], str]] = []
    for series_id in series_ids:
        files = list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_dir), series_id))
        if not files:
            continue
        description = series_description(files[0])
        if predicate(description):
            matches.append((len(files), files, description))

    if not matches:
        raise RuntimeError(f"No matching DICOM series found in: {dicom_dir}")

    _, files, description = max(matches, key=lambda item: item[0])
    selected_files = select_unique_slice_files(collect_slice_infos(files))
    return selected_files, description


def read_series_image(
    dicom_dir: Path,
    predicate: Callable[[str], bool],
) -> tuple[sitk.Image, str]:
    """读取指定模态的 DICOM 序列，返回 SimpleITK 图像和序列描述。"""
    files, description = find_series_files(dicom_dir, predicate)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    return reader.Execute(), description


def ascii_nifti_name(path: Path) -> str:
    """生成临时 ASCII 文件名，绕开 SimpleITK 在中文路径下读取 NIfTI 的问题。"""
    lower_name = path.name.lower()
    if lower_name.endswith(".nii.gz"):
        return "segmentation.nii.gz"
    if lower_name.endswith(".nii"):
        return "segmentation.nii"
    return "segmentation.nii.gz"


def read_mask_on_reference(segmentation_path: Path, reference: sitk.Image) -> np.ndarray:
    """读取 mask，并在尺寸不一致时重采样到背景图像的空间尺寸。"""
    try:
        segmentation = sitk.ReadImage(str(segmentation_path))
    except RuntimeError:
        with tempfile.TemporaryDirectory(prefix="ci1_adc_seg_") as tmp_dir:
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
    return image_to_array(segmentation) != 0


def make_dual_mask_overlay(
    image: np.ndarray,
    dwi_mask: np.ndarray,
    adc_mask: np.ndarray,
) -> np.ndarray:
    """把 DWI mask 和 ADC mask 叠加到灰度背景图上，黄色表示 DWI，红色表示 ADC。"""
    image = image.astype(np.float32)
    if image.max() > image.min():
        image = (image - image.min()) / (image.max() - image.min())
    rgb = np.stack([image, image, image], axis=-1)

    dwi = dwi_mask.astype(bool)
    adc = adc_mask.astype(bool)
    alpha = 0.45
    yellow = np.asarray([1.0, 0.85, 0.0], dtype=np.float32)
    red = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    orange = np.asarray([1.0, 0.45, 0.0], dtype=np.float32)

    dwi_only = dwi & ~adc
    adc_only = adc & ~dwi
    both = dwi & adc
    rgb[dwi_only] = (1.0 - alpha) * rgb[dwi_only] + alpha * yellow
    rgb[adc_only] = (1.0 - alpha) * rgb[adc_only] + alpha * red
    rgb[both] = (1.0 - alpha) * rgb[both] + alpha * orange
    return np.clip(rgb, 0.0, 1.0)


def extract_view_slice(volume: np.ndarray, view: str, index: int) -> np.ndarray:
    """从 3D 体数据中按 axial/coronal/sagittal 方向取一张 2D 切片。"""
    if view == "axial":
        return volume[index, :, :]
    if view == "coronal":
        return volume[:, index, :]
    if view == "sagittal":
        return volume[:, :, index]
    raise ValueError(f"Unsupported view: {view}")


def mask_area_by_view(mask: np.ndarray, view: str) -> np.ndarray:
    """统计指定视角下每张切片的 mask 面积，用于挑选有标注的切片。"""
    if view == "axial":
        return mask.reshape(mask.shape[0], -1).sum(axis=1)
    if view == "coronal":
        return mask.sum(axis=(0, 2))
    if view == "sagittal":
        return mask.sum(axis=(0, 1))
    raise ValueError(f"Unsupported view: {view}")


def view_pixel_spacing(image_spacing: Sequence[float], view: str) -> tuple[float, float]:
    """根据 DICOM spacing 计算某个视角下 2D 图像的横纵向物理像素间距。"""
    x_spacing, y_spacing, z_spacing = image_spacing
    if view == "axial":
        return float(x_spacing), float(y_spacing)
    if view == "coronal":
        return float(x_spacing), float(z_spacing)
    if view == "sagittal":
        return float(y_spacing), float(z_spacing)
    raise ValueError(f"Unsupported view: {view}")


def image_extent(shape: tuple[int, int], pixel_spacing: tuple[float, float]) -> list[float]:
    """把数组尺寸和物理像素间距转换成 matplotlib imshow 使用的显示范围。"""
    row_count, col_count = shape
    col_spacing, row_spacing = pixel_spacing
    return [0.0, float(col_count * col_spacing), float(row_count * row_spacing), 0.0]


def positive_slice_indices(
    dwi_mask: np.ndarray,
    adc_mask: np.ndarray,
    max_slices: int,
    view: str = "axial",
) -> list[int]:
    """选择 DWI 或 ADC 任一 mask 非空的切片；数量过多时取 mask 面积最大的若干张。"""
    union = dwi_mask | adc_mask
    areas = mask_area_by_view(union, view)
    positive = [index for index, area in enumerate(areas) if area > 0]
    if len(positive) <= max_slices:
        return positive
    ranked = sorted(positive, key=lambda index: areas[index], reverse=True)
    return sorted(ranked[:max_slices])


def write_case_overlay_grid(
    image_volume: np.ndarray,
    dwi_mask: np.ndarray,
    adc_mask: np.ndarray,
    output_path: Path,
    title: str,
    max_slices: int,
    view: str,
    pixel_spacing: tuple[float, float],
) -> None:
    """为单个病例写出一张多切片网格图，显示指定视角下的双 mask 覆盖效果。"""
    indices = positive_slice_indices(
        dwi_mask,
        adc_mask,
        max_slices=max_slices,
        view=view,
    )
    if not indices:
        return

    cols = min(4, len(indices))
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes_array = np.asarray(axes).reshape(-1)

    for axis, slice_index in zip(axes_array, indices):
        overlay = make_dual_mask_overlay(
            extract_view_slice(image_volume, view, slice_index),
            extract_view_slice(dwi_mask, view, slice_index),
            extract_view_slice(adc_mask, view, slice_index),
        )
        axis.imshow(
            overlay,
            extent=image_extent(overlay.shape[:2], pixel_spacing),
            aspect="equal",
        )
        axis.set_title(f"{view}={slice_index}")
        axis.axis("off")

    for axis in axes_array[len(indices) :]:
        axis.axis("off")

    fig.suptitle(title, fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_dwi_adc_mask_audit(
    index_csv: Path,
    output_dir: Path,
    max_cases: int,
    max_slices: int,
    background: str,
    views: Sequence[str],
) -> list[Path]:
    """批量生成 DWI/ADC 双 mask 覆盖检查图，可选择 ADC/DWI/FLAIR 作为背景。"""
    rows = read_index_rows(index_csv)
    output_paths: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    completed_cases = 0
    for row in rows:
        if completed_cases >= max_cases:
            break

        dwi_segmentation = Path(row["segmentation_path"])
        adc_segmentation = find_adc_segmentation_path(dwi_segmentation)
        if adc_segmentation is None:
            continue

        dicom_dir = Path(row["dicom_dir"])
        image, description = read_series_image(
            dicom_dir,
            background_predicate(background),
        )
        image_spacing = image.GetSpacing()
        image_volume = normalize_volume_to_uint8(image_to_array(image)).astype(np.float32) / 255.0
        dwi_mask = read_mask_on_reference(dwi_segmentation, image)
        adc_mask = read_mask_on_reference(adc_segmentation, image)

        patient = row["patient"]
        timepoint = row["timepoint"]
        safe_patient = "".join(char if char.isalnum() else "_" for char in patient)
        for view in views:
            output_path = (
                output_dir
                / f"{safe_patient}_{timepoint}_{background.lower()}_{view}_dwi_adc_masks.png"
            )
            title = (
                f"{patient} {timepoint}: yellow=DWI mask, red=ADC mask, "
                f"background={description}, view={view}"
            )
            write_case_overlay_grid(
                image_volume=image_volume,
                dwi_mask=dwi_mask,
                adc_mask=adc_mask,
                output_path=output_path,
                title=title,
                max_slices=max_slices,
                view=view,
                pixel_spacing=view_pixel_spacing(image_spacing, view),
            )
            output_paths.append(output_path)
        completed_cases += 1

    return output_paths


def parse_args() -> argparse.Namespace:
    """解析命令行参数，控制输入索引、输出目录、背景模态和视角。"""
    parser = argparse.ArgumentParser(
        description="Audit DWI and ADC segmentation masks on a DICOM background."
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=Path("data") / "ci1_dwi_index.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "ci1_dwi_adc_mask_audit",
    )
    parser.add_argument("--max-cases", type=int, default=8)
    parser.add_argument("--max-slices", type=int, default=12)
    parser.add_argument(
        "--background",
        choices=["adc", "dwi", "flair"],
        default="adc",
        help="DICOM series used as the grayscale background.",
    )
    parser.add_argument(
        "--views",
        nargs="+",
        choices=VIEWS,
        default=["axial"],
        help="Anatomical views to render.",
    )
    return parser.parse_args()


def main() -> None:
    """命令行入口：读取参数并生成覆盖检查图。"""
    args = parse_args()
    output_paths = build_dwi_adc_mask_audit(
        index_csv=args.index_csv,
        output_dir=args.output_dir,
        max_cases=args.max_cases,
        max_slices=args.max_slices,
        background=args.background,
        views=args.views,
    )
    print(f"Wrote {len(output_paths)} DWI/ADC mask audit figures to {args.output_dir}")


if __name__ == "__main__":
    main()
