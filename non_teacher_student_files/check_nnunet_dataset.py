"""Check a converted nnU-Net v2 dataset.

中文说明：这个脚本检查 Dataset501_StrokeLesion 这类 nnU-Net 数据集目录，
确认 imagesTr 和 labelsTr 病例是否一一对应，label 是否只包含 0/1，
以及 image 和 label 的 shape 是否一致，并逐个打印 image、label 路径。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk


DEFAULT_DATASET_ROOT = Path(r"C:\lijialin\models3d\nnUNet\nnUNet_raw\Dataset501_StrokeLesion")


def case_id_from_image_path(path: Path) -> str:
    """Extract case001 from case001_0000.nii.gz."""
    name = path.name
    if not name.endswith("_0000.nii.gz"):
        raise ValueError(f"Unexpected image filename: {path.name}")
    return name[: -len("_0000.nii.gz")]


def case_id_from_label_path(path: Path) -> str:
    """Extract case001 from case001.nii.gz."""
    name = path.name
    if not name.endswith(".nii.gz"):
        raise ValueError(f"Unexpected label filename: {path.name}")
    return name[: -len(".nii.gz")]


def list_training_images(images_tr: Path) -> dict[str, Path]:
    """Map case IDs to training image paths."""
    return {
        case_id_from_image_path(path): path
        for path in sorted(images_tr.glob("*_0000.nii.gz"))
    }


def list_training_labels(labels_tr: Path) -> dict[str, Path]:
    """Map case IDs to training label paths."""
    return {
        case_id_from_label_path(path): path
        for path in sorted(labels_tr.glob("*.nii.gz"))
    }


def label_has_only_zero_one(label_path: Path) -> tuple[bool, list[int]]:
    """Return whether a label contains only 0 and 1."""
    array = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path)))
    unique_values = np.unique(array)
    values = [int(value) for value in unique_values.tolist()]
    return set(values).issubset({0, 1}), values


def check_dataset(dataset_root: Path) -> bool:
    """Run dataset checks and print every case path."""
    images_tr = dataset_root / "imagesTr"
    labels_tr = dataset_root / "labelsTr"
    if not images_tr.is_dir():
        raise NotADirectoryError(f"Missing imagesTr: {images_tr}")
    if not labels_tr.is_dir():
        raise NotADirectoryError(f"Missing labelsTr: {labels_tr}")

    images = list_training_images(images_tr)
    labels = list_training_labels(labels_tr)
    ok = True

    print(f"imagesTr count: {len(images)}")
    print(f"labelsTr count: {len(labels)}")
    if len(images) != len(labels):
        print("[ERROR] imagesTr and labelsTr counts differ")
        ok = False

    image_cases = set(images)
    label_cases = set(labels)
    missing_labels = sorted(image_cases - label_cases)
    missing_images = sorted(label_cases - image_cases)
    if missing_labels:
        print(f"[ERROR] Missing labels for cases: {missing_labels}")
        ok = False
    if missing_images:
        print(f"[ERROR] Missing images for cases: {missing_images}")
        ok = False

    for case_id in sorted(image_cases & label_cases):
        image_path = images[case_id]
        label_path = labels[case_id]
        print(f"{case_id}")
        print(f"  image: {image_path}")
        print(f"  label: {label_path}")

        image = sitk.ReadImage(str(image_path))
        label = sitk.ReadImage(str(label_path))
        if image.GetSize() != label.GetSize():
            print(
                f"  [ERROR] shape mismatch image={image.GetSize()} "
                f"label={label.GetSize()}"
            )
            ok = False

        label_ok, values = label_has_only_zero_one(label_path)
        print(f"  label unique values: {values}")
        if not label_ok:
            print("  [ERROR] label contains values outside {0, 1}")
            ok = False

    if ok:
        print("[OK] nnU-Net dataset checks passed")
    else:
        print("[FAILED] nnU-Net dataset checks found problems")
    return ok


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Check a converted nnU-Net dataset.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    if not check_dataset(args.dataset_root):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
