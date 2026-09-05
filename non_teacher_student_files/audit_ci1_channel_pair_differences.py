"""Generate paired CI-1 channel difference audit figures.

中文说明：这个脚本在已有三通道 overlay 结果目录下，额外生成 ADC/DWI、
DWI/FLAIR、ADC/FLAIR 两两对比图。每个组合按 axial、coronal、sagittal
三个视角输出 image 和 mask 两类图，优先选择不同病人、不同日期且 mask
占比较大的切片，便于检查不同模态原图和标注之间的差异。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from audit_ci1_channel_overlays import (
    CHANNELS,
    VIEWS,
    load_channel_case,
    mask_area_by_view,
    safe_name,
)
from audit_ci1_dwi_adc_masks import extract_view_slice, image_extent, view_pixel_spacing
from convert_ci1_dwi_to_2d import read_index


CHANNEL_PAIRS = (("adc", "dwi"), ("dwi", "flair"), ("adc", "flair"))


@dataclass(frozen=True)
class PairSample:
    patient: str
    timepoint: str
    slice_index: int
    score: float
    left_image: np.ndarray
    right_image: np.ndarray
    left_mask: np.ndarray
    right_mask: np.ndarray
    left_spacing: tuple[float, float]
    right_spacing: tuple[float, float]
    left_channel: str
    right_channel: str
    view: str


def pair_name(left_channel: str, right_channel: str) -> str:
    """Return the folder name for a channel pair."""
    return f"{left_channel.lower()}vs{right_channel.lower()}"


def channel_pair_output_path(
    output_dir: Path,
    pair_name: str,
    view: str,
    kind: str,
    index: int,
    patient: str,
    timepoint: str,
    slice_index: int,
) -> Path:
    """Build output path: pair/view/image-or-mask/compare_index_patient_day_slice.png."""
    return (
        output_dir
        / pair_name
        / view
        / kind
        / f"compare_{index:03d}_{safe_name(patient)}_{timepoint}_s{slice_index}.png"
    )


def normalize_slice(image: np.ndarray) -> np.ndarray:
    """Normalize a single 2D image to float range 0..1."""
    image = image.astype(np.float32)
    if image.max() > image.min():
        return (image - image.min()) / (image.max() - image.min())
    return np.zeros_like(image, dtype=np.float32)


def resize_nearest(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a 2D array with nearest-neighbor indexing."""
    if image.shape == shape:
        return image
    rows = np.linspace(0, image.shape[0] - 1, shape[0]).round().astype(int)
    cols = np.linspace(0, image.shape[1] - 1, shape[1]).round().astype(int)
    return image[np.ix_(rows, cols)]


def image_difference_mask(left_image: np.ndarray, right_image: np.ndarray) -> np.ndarray:
    """Find visibly different image regions using a robust percentile threshold."""
    left = normalize_slice(left_image)
    right = normalize_slice(resize_nearest(right_image, left.shape))
    diff = np.abs(left - right)
    positive = diff[diff > 0]
    if positive.size == 0:
        return np.zeros_like(diff, dtype=bool)
    threshold = max(float(np.percentile(positive, 85)), 0.12)
    return diff >= threshold


def build_difference_overlay(
    base: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Overlay left-only, right-only, and shared regions on a grayscale base image."""
    base = normalize_slice(base)
    left_mask = resize_nearest(left_mask.astype(bool), base.shape).astype(bool)
    right_mask = resize_nearest(right_mask.astype(bool), base.shape).astype(bool)

    rgb = np.stack([base, base, base], axis=-1)
    left_only = left_mask & ~right_mask
    right_only = right_mask & ~left_mask
    shared = left_mask & right_mask
    diff = left_only | right_only

    alpha = 0.55
    rgb[left_only] = (1.0 - alpha) * rgb[left_only] + alpha * np.array([1.0, 0.0, 0.0])
    rgb[right_only] = (1.0 - alpha) * rgb[right_only] + alpha * np.array([0.0, 0.9, 1.0])
    rgb[shared] = (1.0 - alpha) * rgb[shared] + alpha * np.array([1.0, 1.0, 0.0])
    return np.clip(rgb, 0.0, 1.0), diff


def make_diff_only(diff: np.ndarray) -> np.ndarray:
    """Convert a boolean difference mask into a displayable grayscale image."""
    return diff.astype(np.float32)


def write_pair_comparison_figure(
    left: np.ndarray,
    right: np.ndarray,
    overlay: np.ndarray,
    diff_only: np.ndarray,
    output_path: Path,
    title: str,
    left_title: str,
    right_title: str,
    left_spacing: tuple[float, float],
    right_spacing: tuple[float, float],
) -> None:
    """Write a four-panel comparison figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    left = normalize_slice(left)
    right = normalize_slice(right)
    diff_only = make_diff_only(diff_only)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    axes[0].imshow(
        left,
        cmap="gray",
        extent=image_extent(left.shape[:2], left_spacing),
        aspect="equal",
    )
    axes[0].set_title(left_title)
    axes[1].imshow(
        right,
        cmap="gray",
        extent=image_extent(right.shape[:2], right_spacing),
        aspect="equal",
    )
    axes[1].set_title(right_title)
    axes[2].imshow(
        overlay,
        extent=image_extent(overlay.shape[:2], left_spacing),
        aspect="equal",
    )
    axes[2].set_title("Difference overlay")
    axes[3].imshow(
        diff_only,
        cmap="gray",
        extent=image_extent(diff_only.shape[:2], left_spacing),
        aspect="equal",
    )
    axes[3].set_title("Difference only")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def select_diverse_ranked_samples(
    samples: Sequence[PairSample],
    max_samples: int,
) -> list[PairSample]:
    """Select high-score samples while prioritizing different patients first."""
    grouped: dict[str, list[PairSample]] = {}
    for sample in sorted(samples, key=lambda item: item.score, reverse=True):
        grouped.setdefault(sample.patient, []).append(sample)

    selected: list[PairSample] = []
    round_index = 0
    patient_names = list(grouped.keys())
    while len(selected) < max_samples:
        added = False
        for patient in patient_names:
            patient_samples = grouped[patient]
            if round_index >= len(patient_samples):
                continue
            selected.append(patient_samples[round_index])
            added = True
            if len(selected) >= max_samples:
                break
        if not added:
            break
        round_index += 1
    return selected


def build_pair_samples(
    index_csv: Path,
    left_channel: str,
    right_channel: str,
    view: str,
    max_cases: int,
    slices_per_case: int,
) -> list[PairSample]:
    """Collect candidate paired-channel slices for one pair and one view."""
    samples: list[PairSample] = []
    rows = [row for row in read_index(index_csv) if row.dicom_dir]

    for row in rows:
        left_loaded = load_channel_case(row.segmentation_path, row.dicom_dir, left_channel)
        right_loaded = load_channel_case(row.segmentation_path, row.dicom_dir, right_channel)
        if left_loaded is None or right_loaded is None:
            continue

        left_image, left_mask, left_spacing_3d, _left_description = left_loaded
        right_image, right_mask, right_spacing_3d, _right_description = right_loaded
        left_areas = mask_area_by_view(left_mask, view)
        right_areas = mask_area_by_view(right_mask, view)
        area_len = min(len(left_areas), len(right_areas))
        if area_len == 0:
            continue

        combined_areas = left_areas[:area_len] + right_areas[:area_len]
        ranked_indices = [
            index
            for index in np.argsort(combined_areas)[::-1].tolist()
            if combined_areas[index] > 0
        ][:slices_per_case]

        left_spacing = view_pixel_spacing(left_spacing_3d, view)
        right_spacing = view_pixel_spacing(right_spacing_3d, view)
        for slice_index in ranked_indices:
            left_image_slice = extract_view_slice(left_image, view, slice_index)
            right_image_slice = extract_view_slice(right_image, view, slice_index)
            left_mask_slice = extract_view_slice(left_mask, view, slice_index).astype(bool)
            right_mask_slice = extract_view_slice(right_mask, view, slice_index).astype(bool)
            pixels = max(left_mask_slice.size, right_mask_slice.size)
            score = float(combined_areas[slice_index]) / float(pixels)
            samples.append(
                PairSample(
                    patient=row.patient,
                    timepoint=row.timepoint,
                    slice_index=slice_index,
                    score=score,
                    left_image=left_image_slice,
                    right_image=right_image_slice,
                    left_mask=left_mask_slice,
                    right_mask=right_mask_slice,
                    left_spacing=left_spacing,
                    right_spacing=right_spacing,
                    left_channel=left_channel,
                    right_channel=right_channel,
                    view=view,
                )
            )
        if len({sample.patient for sample in samples}) >= max_cases * 2:
            continue

    return samples


def write_pair_sample(
    sample: PairSample,
    output_dir: Path,
    pair_folder: str,
    index: int,
) -> list[Path]:
    """Write image and mask comparison figures for one selected sample."""
    output_paths: list[Path] = []
    image_diff = image_difference_mask(sample.left_image, sample.right_image)
    image_overlay, image_diff = build_difference_overlay(
        base=sample.left_image,
        left_mask=image_diff,
        right_mask=np.zeros_like(image_diff, dtype=bool),
    )
    mask_overlay, mask_diff = build_difference_overlay(
        base=sample.left_image,
        left_mask=sample.left_mask,
        right_mask=sample.right_mask,
    )

    image_path = channel_pair_output_path(
        output_dir=output_dir,
        pair_name=pair_folder,
        view=sample.view,
        kind="image",
        index=index,
        patient=sample.patient,
        timepoint=sample.timepoint,
        slice_index=sample.slice_index,
    )
    mask_path = channel_pair_output_path(
        output_dir=output_dir,
        pair_name=pair_folder,
        view=sample.view,
        kind="mask",
        index=index,
        patient=sample.patient,
        timepoint=sample.timepoint,
        slice_index=sample.slice_index,
    )
    title = (
        f"{sample.patient} {sample.timepoint} {pair_folder.upper()} "
        f"{sample.view}={sample.slice_index} score={sample.score:.4f}"
    )
    write_pair_comparison_figure(
        left=sample.left_image,
        right=sample.right_image,
        overlay=image_overlay,
        diff_only=image_diff,
        output_path=image_path,
        title=title,
        left_title=f"{sample.left_channel.upper()} image",
        right_title=f"{sample.right_channel.upper()} image",
        left_spacing=sample.left_spacing,
        right_spacing=sample.right_spacing,
    )
    write_pair_comparison_figure(
        left=sample.left_mask.astype(np.float32),
        right=sample.right_mask.astype(np.float32),
        overlay=mask_overlay,
        diff_only=mask_diff,
        output_path=mask_path,
        title=title,
        left_title=f"{sample.left_channel.upper()} mask",
        right_title=f"{sample.right_channel.upper()} mask",
        left_spacing=sample.left_spacing,
        right_spacing=sample.right_spacing,
    )
    output_paths.extend([image_path, mask_path])
    return output_paths


def build_channel_pair_difference_audit(
    index_csv: Path,
    output_dir: Path,
    channel_pairs: Sequence[tuple[str, str]],
    views: Sequence[str],
    images_per_view: int,
    max_cases: int,
    slices_per_case: int,
) -> list[Path]:
    """Build all channel-pair difference audit figures."""
    output_paths: list[Path] = []
    for left_channel, right_channel in channel_pairs:
        folder = pair_name(left_channel, right_channel)
        for view in views:
            samples = build_pair_samples(
                index_csv=index_csv,
                left_channel=left_channel,
                right_channel=right_channel,
                view=view,
                max_cases=max_cases,
                slices_per_case=slices_per_case,
            )
            selected = select_diverse_ranked_samples(samples, images_per_view)
            for index, sample in enumerate(selected, start=1):
                output_paths.extend(
                    write_pair_sample(
                        sample=sample,
                        output_dir=output_dir,
                        pair_folder=folder,
                        index=index,
                    )
                )
    return output_paths


def parse_pair(text: str) -> tuple[str, str]:
    """Parse a pair argument such as adcvsdwi."""
    lowered = text.lower()
    if "vs" not in lowered:
        raise argparse.ArgumentTypeError("Channel pair must look like adcvsdwi.")
    left, right = lowered.split("vs", 1)
    if left not in CHANNELS or right not in CHANNELS or left == right:
        raise argparse.ArgumentTypeError(f"Unsupported channel pair: {text}")
    return left, right


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create paired CI-1 channel image/mask difference figures."
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=Path("data") / "ci1_dwi_index.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "ci1_channel_overlay_audit_diverse",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        type=parse_pair,
        default=list(CHANNEL_PAIRS),
    )
    parser.add_argument("--views", nargs="+", choices=VIEWS, default=list(VIEWS))
    parser.add_argument("--images-per-view", type=int, default=10)
    parser.add_argument("--max-cases", type=int, default=30)
    parser.add_argument("--slices-per-case", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    output_paths = build_channel_pair_difference_audit(
        index_csv=args.index_csv,
        output_dir=args.output_dir,
        channel_pairs=args.pairs,
        views=args.views,
        images_per_view=args.images_per_view,
        max_cases=args.max_cases,
        slices_per_case=args.slices_per_case,
    )
    print(f"Wrote {len(output_paths)} channel-pair comparison figures to {args.output_dir}")


if __name__ == "__main__":
    main()
