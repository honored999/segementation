"""Report Dataset501 path status and optionally inspect one named case."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from standalone_nnunet2d.data.dataset import StrokeSliceDataset


def inspect_dataset(
    raw_root: Path,
    *,
    case_id: str | None = None,
    fold: int = 0,
    split: str = "val",
) -> dict[str, Any]:
    """Inspect paths, loading at most the explicitly requested one case."""
    root = raw_root.resolve()
    report: dict[str, Any] = {
        "raw_root": str(root),
        "raw_root_exists": root.is_dir(),
        "imagesTr_exists": (root / "imagesTr").is_dir(),
        "labelsTr_exists": (root / "labelsTr").is_dir(),
    }
    if case_id is None:
        return report
    dataset = StrokeSliceDataset(root, fold=fold, split=split, case_ids=(case_id,))
    image, label = dataset.load_case(case_id)
    report.update(
        {
            "case_id": case_id,
            "image_shape_zyx": tuple(image.shape),
            "label_shape_zyx": tuple(label.shape),
            "image_dtype": str(image.dtype),
            "label_dtype": str(label.dtype),
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--case-id")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    arguments = parser.parse_args()
    for name, value in inspect_dataset(
        arguments.raw_root,
        case_id=arguments.case_id,
        fold=arguments.fold,
        split=arguments.split,
    ).items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
