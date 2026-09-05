"""Generate per-channel CI-1 image/mask/overlay figures.

中文说明：
这个脚本分别为 DWI、ADC、FLAIR 三个通道生成原图、mask、覆盖图三栏图，
并按照通道和视角输出到分层文件夹中，便于人工检查不同模态的标注质量。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from audit_ci1_dwi_adc_masks import (
    background_predicate,
    extract_view_slice,
    image_extent,
    read_mask_on_reference,
    read_series_image,
    view_pixel_spacing,
)
from convert_ci1_dwi_to_2d import image_to_array, normalize_volume_to_uint8, read_index


CHANNELS = ("dwi", "adc", "flair")
VIEWS = ("axial", "coronal", "sagittal")
TIMEPOINT_RE = re.compile(r"(D\d+)", re.IGNORECASE)


def safe_name(text: str) -> str:
    """把患者名等文本转换成适合作为文件名的字符串。"""
    return "".join(char if char.isalnum() else "_" for char in text).strip("_")


def channel_view_output_path(
    output_dir: Path,
    channel: str,
    view: str,
    patient: str,
    timepoint: str,
    index: int,
) -> Path:
    """生成输出路径：顶层目录/channel/view/overlay_序号_患者_时间点.png。"""
    return (
        output_dir
        / channel
        / view
        / f"overlay_{index:03d}_{safe_name(patient)}_{timepoint}.png"
    )


def extract_timepoint(path: Path) -> str | None:
    """从文件名里提取 D1、D2、D3、D7 这类时间点。"""
    match = TIMEPOINT_RE.search(path.name)
    if match is None:
        return None
    return match.group(1).upper()


def find_modality_segmentation_path(
    dwi_segmentation_path: Path,
    channel: str,
    path_exists: Callable[[Path], bool] | None = None,
    glob_paths: Callable[[str], Sequence[Path]] | None = None,
) -> Path | None:
    """根据 DWI mask 路径查找指定通道对应的 mask 路径。"""
    channel = channel.lower()
    exists = path_exists or Path.exists
    globber = glob_paths or (lambda pattern: list(dwi_segmentation_path.parent.glob(pattern)))

    if channel == "dwi":
        return dwi_segmentation_path if exists(dwi_segmentation_path) else None

    name = dwi_segmentation_path.name
    replacements = {
        "adc": [("DWI", "ADC"), ("dwi", "adc"), ("DWI", "DWIADC"), ("dwi", "dwiadc")],
        "flair": [("DWI", "FLAIR"), ("dwi", "flair")],
    }
    candidates: list[Path] = []
    for old, new in replacements.get(channel, []):
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
        for path in globber(f"*{timepoint}*{channel.upper()}*.nii*")
        if exists(path)
    ]
    if channel == "flair":
        direct_flair = [
            path
            for path in glob_candidates
            if "flair-adc" not in path.name.lower()
        ]
        if direct_flair:
            return sorted(direct_flair, key=lambda item: (len(item.parts), str(item)))[0]
    if glob_candidates:
        return sorted(glob_candidates, key=lambda item: (len(item.parts), str(item)))[0]
    return None


def mask_area_by_view(mask: np.ndarray, view: str) -> np.ndarray:
    """统计指定视角下每张切片的 mask 面积。"""
    if view == "axial":
        return mask.reshape(mask.shape[0], -1).sum(axis=1)
    if view == "coronal":
        return mask.sum(axis=(0, 2))
    if view == "sagittal":
        return mask.sum(axis=(0, 1))
    raise ValueError(f"Unsupported view: {view}")


def positive_slice_indices(mask: np.ndarray, view: str, max_slices: int) -> list[int]:
    """选择指定视角下 mask 非空且面积最大的若干张切片。"""
    areas = mask_area_by_view(mask, view)
    positive = [index for index, area in enumerate(areas) if area > 0]
    ranked = sorted(positive, key=lambda index: areas[index], reverse=True)
    return sorted(ranked[:max_slices])


def select_diverse_index_rows(index_rows: Sequence[object], max_cases: int) -> list[object]:
    """优先从不同患者各选一个时间点，数量不够时再补同一患者的其它时间点。"""
    grouped: dict[str, list[object]] = {}
    for row in index_rows:
        grouped.setdefault(row.patient, []).append(row)

    selected: list[object] = []
    round_index = 0
    patient_names = list(grouped.keys())
    while len(selected) < max_cases:
        added_this_round = False
        for patient in patient_names:
            rows = grouped[patient]
            if round_index >= len(rows):
                continue
            selected.append(rows[round_index])
            added_this_round = True
            if len(selected) >= max_cases:
                break
        if not added_this_round:
            break
        round_index += 1
    return selected


def make_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """把单个 mask 叠加到灰度原图上，红色区域表示 mask。"""
    image = image.astype(np.float32)
    if image.max() > image.min():
        image = (image - image.min()) / (image.max() - image.min())
    rgb = np.stack([image, image, image], axis=-1)
    mask_bool = mask.astype(bool)
    alpha = 0.45
    red = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    rgb[mask_bool] = (1.0 - alpha) * rgb[mask_bool] + alpha * red
    return np.clip(rgb, 0.0, 1.0)


def write_triplet_figure(
    image_slice: np.ndarray,
    mask_slice: np.ndarray,
    output_path: Path,
    title: str,
    pixel_spacing: tuple[float, float],
) -> None:
    """写出一张三栏图：原图、mask、原图加 mask 覆盖。"""
    image_slice = image_slice.astype(np.float32)
    if image_slice.max() > image_slice.min():
        image_slice = (image_slice - image_slice.min()) / (
            image_slice.max() - image_slice.min()
        )
    mask_slice = mask_slice.astype(bool)
    overlay = make_overlay(image_slice, mask_slice)
    extent = image_extent(image_slice.shape[:2], pixel_spacing)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(image_slice, cmap="gray", extent=extent, aspect="equal")
    axes[0].set_title("Image")
    axes[1].imshow(mask_slice, cmap="gray", extent=extent, aspect="equal")
    axes[1].set_title("Mask")
    axes[2].imshow(overlay, extent=extent, aspect="equal")
    axes[2].set_title("Overlay")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(title, fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_channel_case(
    dwi_segmentation_path: Path,
    dicom_dir: Path,
    channel: str,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float], str] | None:
    """读取某个通道的 DICOM 原图和对应 mask，并把 mask 对齐到原图空间。"""
    segmentation_path = find_modality_segmentation_path(dwi_segmentation_path, channel)
    if segmentation_path is None:
        return None

    image, description = read_series_image(dicom_dir, background_predicate(channel))
    image_volume = normalize_volume_to_uint8(image_to_array(image)).astype(np.float32) / 255.0
    mask_volume = read_mask_on_reference(segmentation_path, image)
    return image_volume, mask_volume, image.GetSpacing(), description


def build_channel_overlay_audit(
    index_csv: Path,
    output_dir: Path,
    channels: Sequence[str],
    views: Sequence[str],
    max_cases: int,
    images_per_view: int,
) -> list[Path]:
    """批量生成三通道、三视角的 image/mask/overlay 检查图。"""
    index_rows = select_diverse_index_rows(
        [row for row in read_index(index_csv) if row.dicom_dir],
        max_cases=max_cases,
    )
    output_paths: list[Path] = []
    completed_cases = 0

    for row in index_rows:
        if completed_cases >= max_cases:
            break

        dwi_segmentation_path = row.segmentation_path
        dicom_dir = row.dicom_dir
        wrote_case = False

        for channel in channels:
            loaded = load_channel_case(dwi_segmentation_path, dicom_dir, channel)
            if loaded is None:
                continue
            image_volume, mask_volume, image_spacing, description = loaded

            for view in views:
                indices = positive_slice_indices(
                    mask_volume,
                    view=view,
                    max_slices=images_per_view,
                )
                pixel_spacing = view_pixel_spacing(image_spacing, view)
                for sample_index, slice_index in enumerate(indices, start=1):
                    image_slice = extract_view_slice(image_volume, view, slice_index)
                    mask_slice = extract_view_slice(mask_volume, view, slice_index)
                    output_path = channel_view_output_path(
                        output_dir=output_dir,
                        channel=channel,
                        view=view,
                        patient=row.patient,
                        timepoint=row.timepoint,
                        index=len(output_paths) + 1,
                    )
                    title = (
                        f"{row.patient} {row.timepoint} {channel.upper()} "
                        f"{view}={slice_index} background={description}"
                    )
                    write_triplet_figure(
                        image_slice=image_slice,
                        mask_slice=mask_slice,
                        output_path=output_path,
                        title=title,
                        pixel_spacing=pixel_spacing,
                    )
                    output_paths.append(output_path)
                    wrote_case = True

        if wrote_case:
            completed_cases += 1

    return output_paths


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Create per-channel CI-1 image/mask/overlay audit figures."
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=Path("data") / "ci1_dwi_index.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "ci1_channel_overlay_audit",
    )
    parser.add_argument("--channels", nargs="+", choices=CHANNELS, default=list(CHANNELS))
    parser.add_argument("--views", nargs="+", choices=VIEWS, default=list(VIEWS))
    parser.add_argument("--max-cases", type=int, default=6)
    parser.add_argument("--images-per-view", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""
    args = parse_args()
    output_paths = build_channel_overlay_audit(
        index_csv=args.index_csv,
        output_dir=args.output_dir,
        channels=args.channels,
        views=args.views,
        max_cases=args.max_cases,
        images_per_view=args.images_per_view,
    )
    print(f"Wrote {len(output_paths)} channel overlay figures to {args.output_dir}")


if __name__ == "__main__":
    main()
