"""Create axial/coronal/sagittal DWI-mask overlay figures from CI-1 2D slices.

中文说明：
从 CI-1 的 DWI 二维切片重建体数据，并生成三个视角的 DWI/mask 覆盖图。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from audit_ci1_dwi_overlays import make_overlay


VIEW_AXES = {
    "axial": 0,
    "coronal": 1,
    "sagittal": 2,
}
IMAGE_ASPECT = "equal"


def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def group_rows(rows: Sequence[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row.get("patient", "unknown"), row.get("timepoint", ""))
        groups.setdefault(key, []).append(row)
    return groups


def load_volume(rows: Sequence[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    sorted_rows = sorted(rows, key=lambda row: int(row["slice_index"]))
    images = []
    masks = []
    for row in sorted_rows:
        image = np.asarray(Image.open(row["image_path"]).convert("L"), dtype=np.float32) / 255.0
        mask = np.asarray(Image.open(row["mask_path"]).convert("L")) > 127
        images.append(image)
        masks.append(mask)
    if not images:
        raise ValueError("Cannot build a volume from an empty row group.")
    return np.stack(images, axis=0), np.stack(masks, axis=0)


def slice_area_by_view(mask_volume: np.ndarray, view: str) -> np.ndarray:
    axis = VIEW_AXES[view]
    axes_to_sum = tuple(index for index in range(mask_volume.ndim) if index != axis)
    return mask_volume.sum(axis=axes_to_sum)


def select_largest_mask_slice(mask_volume: np.ndarray, view: str) -> int:
    areas = slice_area_by_view(mask_volume, view)
    return int(np.argmax(areas))


def extract_view_slice(volume: np.ndarray, view: str, index: int) -> np.ndarray:
    if view == "axial":
        return volume[index, :, :]
    if view == "coronal":
        return volume[:, index, :]
    if view == "sagittal":
        return volume[:, :, index]
    raise ValueError(f"Unsupported view: {view}")


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.max() > image.min():
        image = (image - image.min()) / (image.max() - image.min())
    return image


def write_overlay_triplet(
    image: np.ndarray,
    mask: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    image = normalize_image(image)
    mask = mask.astype(bool)
    overlay = make_overlay(image, mask)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(image, cmap="gray", aspect=IMAGE_ASPECT)
    axes[0].set_title("DWI")
    axes[1].imshow(mask, cmap="gray", aspect=IMAGE_ASPECT)
    axes[1].set_title("Mask")
    axes[2].imshow(overlay, aspect=IMAGE_ASPECT)
    axes[2].set_title("Overlay")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(title, fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_multiview_grid(
    image_volume: np.ndarray,
    mask_volume: np.ndarray,
    patient: str,
    timepoint: str,
    output_path: Path,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))

    for row_index, view in enumerate(["axial", "coronal", "sagittal"]):
        index = select_largest_mask_slice(mask_volume, view)
        image_slice = normalize_image(extract_view_slice(image_volume, view, index))
        mask_slice = extract_view_slice(mask_volume, view, index).astype(bool)
        overlay = make_overlay(image_slice, mask_slice)
        mask_area = int(mask_slice.sum())
        selected.append(
            {
                "patient": patient,
                "timepoint": timepoint,
                "view": view,
                "slice_index": str(index),
                "mask_area": str(mask_area),
            }
        )

        panels = [
            ("DWI", image_slice, "gray"),
            ("Mask", mask_slice, "gray"),
            ("Overlay", overlay, None),
        ]
        for col_index, (panel_title, panel_image, cmap) in enumerate(panels):
            axis = axes[row_index, col_index]
            if cmap:
                axis.imshow(panel_image, cmap=cmap, aspect=IMAGE_ASPECT)
            else:
                axis.imshow(panel_image, aspect=IMAGE_ASPECT)
            axis.set_title(f"{view} {panel_title}" if col_index == 0 else panel_title)
            axis.axis("off")

    fig.suptitle(f"{patient} {timepoint} largest-mask slices", fontsize=11)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return selected


def create_multiview_overlays(
    manifest_path: Path,
    output_dir: Path,
    patient: str | None,
    timepoint: str | None,
    max_cases: int,
) -> list[Path]:
    rows = read_manifest(manifest_path)
    groups = group_rows(rows)
    selected_groups = []
    for key, group in sorted(groups.items()):
        group_patient, group_timepoint = key
        if patient is not None and group_patient != patient:
            continue
        if timepoint is not None and group_timepoint != timepoint:
            continue
        if any(float(row.get("mask_area", "0") or 0) > 0 for row in group):
            selected_groups.append((key, group))
        if len(selected_groups) >= max_cases:
            break

    output_paths: list[Path] = []
    selected_rows: list[dict[str, str]] = []
    for case_index, ((group_patient, group_timepoint), group) in enumerate(selected_groups, start=1):
        image_volume, mask_volume = load_volume(group)
        safe_patient = "".join(char if char.isalnum() else "_" for char in group_patient)
        case_prefix = f"case_{case_index:03d}_{safe_patient}_{group_timepoint}"
        grid_path = output_dir / f"{case_prefix}_multiview_grid.png"
        selected_rows.extend(
            write_multiview_grid(
                image_volume,
                mask_volume,
                group_patient,
                group_timepoint,
                grid_path,
            )
        )
        output_paths.append(grid_path)

        for view in ["axial", "coronal", "sagittal"]:
            index = select_largest_mask_slice(mask_volume, view)
            image_slice = extract_view_slice(image_volume, view, index)
            mask_slice = extract_view_slice(mask_volume, view, index)
            single_path = output_dir / f"{case_prefix}_{view}_z{index}.png"
            write_overlay_triplet(
                image_slice,
                mask_slice,
                f"{group_patient} {group_timepoint} {view} index={index}",
                single_path,
            )
            output_paths.append(single_path)

    if selected_rows:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "selected_multiview_slices.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(selected_rows[0].keys()))
            writer.writeheader()
            writer.writerows(selected_rows)
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create CI-1 DWI multiview overlay figures.")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data") / "ci1_dwi_2d_dedup" / "manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "ci1_dwi_multiview_overlays",
    )
    parser.add_argument("--patient", default=None)
    parser.add_argument("--timepoint", default=None)
    parser.add_argument("--max-cases", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_paths = create_multiview_overlays(
        manifest_path=args.manifest_path,
        output_dir=args.output_dir,
        patient=args.patient,
        timepoint=args.timepoint,
        max_cases=args.max_cases,
    )
    print(f"Wrote {len(output_paths)} overlay figures to {args.output_dir}")


if __name__ == "__main__":
    main()
