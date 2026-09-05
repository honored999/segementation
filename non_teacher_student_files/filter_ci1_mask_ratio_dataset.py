"""Filter CI-1 2D slices by mask area ratio.

中文说明：
按照 mask 面积占比筛选 CI-1 二维切片，用于保留病灶占比较高的训练样本。
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


def read_manifest(input_manifest: Path) -> list[dict[str, str]]:
    with input_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def mask_area_from_image(mask_path: Path) -> int:
    mask = np.asarray(Image.open(mask_path).convert("L"))
    return int(np.count_nonzero(mask > 127))


def image_area_from_mask(mask_path: Path) -> int:
    with Image.open(mask_path) as image:
        width, height = image.size
    return width * height


def compute_mask_ratio(row: dict[str, str]) -> float:
    mask_path = Path(row["mask_path"])
    image_area = image_area_from_mask(mask_path)
    if image_area <= 0:
        raise ValueError(f"Invalid mask image area for {mask_path}")

    mask_area_text = row.get("mask_area", "")
    if mask_area_text:
        mask_area = int(float(mask_area_text))
    else:
        mask_area = mask_area_from_image(mask_path)

    return mask_area / image_area


def format_ratio(value: float) -> str:
    return f"{value:.8f}".rstrip("0").rstrip(".")


def copy_pair(row: dict[str, str], output_dir: Path, index: int) -> dict[str, str]:
    output_row = dict(row)
    image_path = Path(row["image_path"])
    mask_path = Path(row["mask_path"])
    image_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{index:06d}_{image_path.stem}"
    image_output = image_dir / f"{stem}{image_path.suffix}"
    mask_output = mask_dir / f"{stem}{mask_path.suffix}"
    shutil.copy2(image_path, image_output)
    shutil.copy2(mask_path, mask_output)
    output_row["image_path"] = str(image_output)
    output_row["mask_path"] = str(mask_output)
    return output_row


def write_manifest(rows: Sequence[dict[str, str]], output_manifest: Path) -> None:
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with output_manifest.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return

    fieldnames = list(rows[0].keys())
    with output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def filter_manifest_by_mask_ratio(
    input_manifest: Path,
    output_manifest: Path,
    min_mask_ratio: float,
    copy_files: bool = False,
) -> list[dict[str, str]]:
    if min_mask_ratio < 0 or min_mask_ratio > 1:
        raise ValueError("--min-mask-ratio must be between 0 and 1.")

    rows = read_manifest(input_manifest)
    kept_rows: list[dict[str, str]] = []
    output_dir = output_manifest.parent

    for row in rows:
        mask_ratio = compute_mask_ratio(row)
        if mask_ratio < min_mask_ratio:
            continue

        output_row = copy_pair(row, output_dir, len(kept_rows)) if copy_files else dict(row)
        output_row["mask_ratio"] = format_ratio(mask_ratio)
        kept_rows.append(output_row)

    write_manifest(kept_rows, output_manifest)
    return kept_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter CI-1 2D DWI slices by mask area ratio."
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=Path("data") / "ci1_dwi_2d_dedup" / "manifest.csv",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("data") / "ci1_dwi_2d_dedup_mask_ge10" / "manifest.csv",
    )
    parser.add_argument(
        "--min-mask-ratio",
        type=float,
        default=0.10,
        help="Keep slices whose mask area / image area is at least this value.",
    )
    parser.add_argument(
        "--copy-files",
        action="store_true",
        help="Copy selected images and masks into the output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = filter_manifest_by_mask_ratio(
        input_manifest=args.input_manifest,
        output_manifest=args.output_manifest,
        min_mask_ratio=args.min_mask_ratio,
        copy_files=args.copy_files,
    )
    print(f"Input manifest: {args.input_manifest}")
    print(f"Output manifest: {args.output_manifest}")
    print(f"Minimum mask ratio: {args.min_mask_ratio:g}")
    print(f"Kept rows: {len(rows)}")
    if args.copy_files:
        print(f"Copied selected images and masks under: {args.output_manifest.parent}")


if __name__ == "__main__":
    main()
