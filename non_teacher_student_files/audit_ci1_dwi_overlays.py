"""Generate DWI/mask overlay images for checking CI-1 label alignment.

中文说明：
生成 DWI 图像和 mask 的覆盖图，用于检查 CI-1 标注是否和原图对齐。
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def read_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def row_has_mask(row: dict[str, str]) -> bool:
    value = row.get("has_mask", "")
    if value == "":
        return True
    try:
        return float(value) > 0
    except ValueError:
        return value.strip().lower() in {"true", "yes", "y"}


def select_positive_rows(
    rows: Sequence[dict[str, str]],
    max_samples: int,
    seed: int,
) -> list[dict[str, str]]:
    positive_rows = [row for row in rows if row_has_mask(row)]
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in positive_rows:
        key = f"{row.get('patient', 'unknown')}::{row.get('timepoint', '')}"
        grouped.setdefault(key, []).append(row)

    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    for key in sorted(grouped):
        candidates = grouped[key]
        rng.shuffle(candidates)
        selected.append(candidates[0])

    rng.shuffle(selected)
    if len(selected) < max_samples:
        seen = {id(row) for row in selected}
        remainder = [row for row in positive_rows if id(row) not in seen]
        rng.shuffle(remainder)
        selected.extend(remainder[: max_samples - len(selected)])

    return selected[:max_samples]


def load_sample(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    if "tensor_path" in row and row["tensor_path"]:
        sample = torch.load(row["tensor_path"], map_location="cpu", weights_only=False)
        image = sample["image"].float().squeeze().numpy()
        mask = sample["mask"].float().squeeze().numpy()
    else:
        image = np.asarray(Image.open(row["image_path"]).convert("L"), dtype=np.float32) / 255.0
        mask = (np.asarray(Image.open(row["mask_path"]).convert("L"), dtype=np.float32) > 127.0).astype(
            np.float32
        )

    image = np.asarray(image, dtype=np.float32)
    mask = (np.asarray(mask, dtype=np.float32) > 0.5).astype(np.float32)
    if image.max() > image.min():
        image = (image - image.min()) / (image.max() - image.min())
    return image, mask


def make_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image_rgb = np.stack([image, image, image], axis=-1)
    overlay = image_rgb.copy()
    mask_bool = mask > 0
    overlay[mask_bool, 0] = 1.0
    overlay[mask_bool, 1] = 0.0
    overlay[mask_bool, 2] = 0.0
    alpha = 0.45
    image_rgb[mask_bool] = (1.0 - alpha) * image_rgb[mask_bool] + alpha * overlay[mask_bool]
    return np.clip(image_rgb, 0.0, 1.0)


def row_label(row: dict[str, str]) -> str:
    patient = row.get("patient", "unknown")
    timepoint = row.get("timepoint", "")
    slice_index = row.get("slice_index", row.get("z_index", ""))
    return f"{patient} {timepoint} z={slice_index}".strip()


def write_single_overlay(row: dict[str, str], output_path: Path) -> None:
    image, mask = load_sample(row)
    overlay = make_overlay(image, mask)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("DWI")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Mask")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(row_label(row), fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_overlay_grid(rows: Sequence[dict[str, str]], output_path: Path) -> None:
    if not rows:
        return

    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)))
    if len(rows) == 1:
        axes = axes.reshape(1, -1)

    for index, row in enumerate(rows):
        image, mask = load_sample(row)
        overlay = make_overlay(image, mask)
        axes[index, 0].imshow(image, cmap="gray")
        axes[index, 0].set_title("DWI")
        axes[index, 1].imshow(mask, cmap="gray")
        axes[index, 1].set_title("Mask")
        axes[index, 2].imshow(overlay)
        axes[index, 2].set_title(row_label(row))
        for axis in axes[index]:
            axis.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_selected_csv(rows: Sequence[dict[str, str]], output_path: Path) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_overlay_audit(
    manifest_path: Path,
    output_dir: Path,
    max_samples: int,
    seed: int,
) -> list[Path]:
    rows = read_manifest_rows(manifest_path)
    selected_rows = select_positive_rows(rows, max_samples=max_samples, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    for index, row in enumerate(selected_rows, start=1):
        patient = "".join(char if char.isalnum() else "_" for char in row.get("patient", "unknown"))
        slice_index = row.get("slice_index", row.get("z_index", str(index)))
        output_path = output_dir / f"overlay_{index:03d}_{patient}_z{slice_index}.png"
        write_single_overlay(row, output_path)
        output_paths.append(output_path)

    write_overlay_grid(selected_rows, output_dir / "overlay_grid.png")
    write_selected_csv(selected_rows, output_dir / "selected_samples.csv")
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit CI-1 DWI/mask alignment with overlay images.")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data") / "ci1_dwi_tensor_cache_256" / "cache_manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "ci1_dwi_overlay_audit",
    )
    parser.add_argument("--max-samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_paths = build_overlay_audit(
        manifest_path=args.manifest_path,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    print(f"Wrote {len(output_paths)} overlay images to {args.output_dir}")
    print(f"Grid: {args.output_dir / 'overlay_grid.png'}")


if __name__ == "__main__":
    main()
