"""Build a tensor cache from the CI-1 DWI 2D PNG dataset.

中文说明：
这个脚本把已经生成的 DWI 二维 PNG 图像和 mask 预处理成 PyTorch tensor 缓存，
训练时可以直接读取 `.pt` 文件，减少反复解码 PNG 和 resize 带来的 CPU 开销。

The training script can read PNGs directly, but that repeatedly performs:
    Image.open -> resize -> numpy -> tensor

This cache builder does those CPU-heavy steps once and writes one `.pt` file per
slice plus a `cache_manifest.csv` that train_ci1_dwi_student_noskip_32ch.py can
use directly.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def read_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def convert_png_pair_to_tensors(
    image_path: Path,
    mask_path: Path,
    image_height: int,
    image_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    image = Image.open(image_path).convert("L")
    mask = Image.open(mask_path).convert("L")
    image = image.resize((image_width, image_height), Image.BILINEAR)
    mask = mask.resize((image_width, image_height), Image.NEAREST)

    image_array = np.asarray(image, dtype=np.float32) / 255.0
    mask_array = (np.asarray(mask, dtype=np.float32) > 127.0).astype(np.float32)
    image_tensor = torch.from_numpy(image_array).unsqueeze(0)
    mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
    return image_tensor, mask_tensor


def build_cache(
    manifest_path: Path,
    output_root: Path,
    image_height: int,
    image_width: int,
) -> Path:
    rows = read_rows(manifest_path)
    tensor_dir = output_root / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = output_root / "cache_manifest.csv"

    output_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        image_tensor, mask_tensor = convert_png_pair_to_tensors(
            image_path=Path(row["image_path"]),
            mask_path=Path(row["mask_path"]),
            image_height=image_height,
            image_width=image_width,
        )

        tensor_path = tensor_dir / f"sample_{index:06d}.pt"
        torch.save(
            {
                "image": image_tensor,
                "mask": mask_tensor,
            },
            tensor_path,
        )

        output_row = dict(row)
        output_row["tensor_path"] = str(tensor_path)
        output_row["cache_height"] = str(image_height)
        output_row["cache_width"] = str(image_width)
        output_rows.append(output_row)

    fieldnames = list(output_rows[0].keys()) if output_rows else [
        "tensor_path",
        "cache_height",
        "cache_width",
    ]
    with output_manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    return output_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CI-1 DWI tensor cache.")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data") / "ci1_dwi_2d_dedup" / "manifest.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data") / "ci1_dwi_tensor_cache_256",
    )
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_manifest = build_cache(
        manifest_path=args.manifest_path,
        output_root=args.output_root,
        image_height=args.height,
        image_width=args.width,
    )
    print(f"Wrote cache manifest: {output_manifest}")


if __name__ == "__main__":
    main()
