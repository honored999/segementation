from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

import compare_segmentation_errors as comparison


def _write_nifti(
    path: Path,
    array: np.ndarray,
    *,
    spacing: tuple[float, float, float] = (1.5, 1.5, 8.0),
    direction: tuple[float, ...] = (
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ),
) -> None:
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    image.SetOrigin((10.0, 20.0, 30.0))
    image.SetDirection(direction)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))


def _make_case(root: Path, *, grouped_spacing=(1.5, 1.5, 8.0)) -> dict[str, Path]:
    directories = {
        name: root / name
        for name in ("images", "labels", "topk10", "foreground50", "grouped")
    }
    image = np.linspace(0.0, 200.0, 4 * 16 * 20, dtype=np.float32).reshape(4, 16, 20)
    gt = np.zeros_like(image, dtype=np.uint8)
    gt[1, 5:9, 7:12] = 1

    topk10 = gt.copy()
    topk10[1, 5:7, 7:9] = 0
    foreground50 = gt.copy()
    foreground50[1, 8:10, 11:13] = 1
    grouped = gt.copy()
    grouped[1, 5:8, 7:10] = 0
    grouped[2, 2:5, 14:18] = 1  # FP on a slice with no GT.

    _write_nifti(directories["images"] / "case001_0000.nii.gz", image)
    _write_nifti(directories["labels"] / "case001.nii.gz", gt)
    _write_nifti(directories["topk10"] / "case001.nii.gz", topk10)
    _write_nifti(directories["foreground50"] / "case001.nii.gz", foreground50)
    _write_nifti(
        directories["grouped"] / "case001.nii.gz",
        grouped,
        spacing=grouped_spacing,
    )
    return directories


def _arguments(directories: dict[str, Path], output_dir: Path) -> list[str]:
    return [
        "--images-dir",
        str(directories["images"]),
        "--labels-dir",
        str(directories["labels"]),
        "--topk10-dir",
        str(directories["topk10"]),
        "--foreground50-dir",
        str(directories["foreground50"]),
        "--grouped-topk-dir",
        str(directories["grouped"]),
        "--output-dir",
        str(output_dir),
        "--case-ids",
        "case001",
        "--slices-per-case",
        "4",
    ]


def test_end_to_end_writes_metrics_selection_png_and_readme(tmp_path: Path) -> None:
    directories = _make_case(tmp_path / "inputs")
    output_dir = tmp_path / "report"

    assert comparison.main(_arguments(directories, output_dir)) == 0

    with (output_dir / "case_comparison.csv").open(newline="", encoding="utf-8") as stream:
        case_rows = list(csv.DictReader(stream))
    assert len(case_rows) == 1
    assert case_rows[0]["case_id"] == "case001"
    assert int(case_rows[0]["grouped_topk10_fp"]) == 12
    assert int(case_rows[0]["grouped_topk10_fn"]) == 9
    assert float(case_rows[0]["grouped_topk10_dice_minus_topk10_dice"]) < 0

    with (output_dir / "slice_selection.csv").open(newline="", encoding="utf-8") as stream:
        slice_rows = list(csv.DictReader(stream))
    no_gt_fp = [row for row in slice_rows if int(row["slice_index_lps_z"]) == 2]
    assert no_gt_fp
    assert "grouped_topk10_fp_max" in no_gt_fp[0]["selection_reasons"]
    assert (output_dir / no_gt_fp[0]["png_file"]).is_file()
    assert (output_dir / "README.txt").is_file()


def test_geometry_mismatch_is_rejected_without_creating_output(tmp_path: Path) -> None:
    directories = _make_case(
        tmp_path / "inputs",
        grouped_spacing=(1.5, 1.5, 8.0001),
    )
    output_dir = tmp_path / "report"

    with pytest.raises(ValueError, match="spacing mismatch"):
        comparison.main(_arguments(directories, output_dir))

    assert not output_dir.exists()


def test_duplicate_case_id_is_rejected(tmp_path: Path) -> None:
    mask_dir = tmp_path / "masks"
    mask = np.zeros((2, 3, 4), dtype=np.uint8)
    _write_nifti(mask_dir / "case001.nii.gz", mask)
    _write_nifti(mask_dir / "case001.nii", mask)

    with pytest.raises(ValueError, match="duplicate case ID"):
        comparison.discover_mask_files(mask_dir, "GT")


def test_oblique_direction_is_rejected() -> None:
    angle = np.deg2rad(5.0)
    oblique = (
        float(np.cos(angle)),
        float(-np.sin(angle)),
        0.0,
        float(np.sin(angle)),
        float(np.cos(angle)),
        0.0,
        0.0,
        0.0,
        1.0,
    )

    with pytest.raises(ValueError, match="axis-aligned"):
        comparison.validate_axis_aligned_direction(oblique, "case001 DWI")


def test_distinct_case_ids_keep_distinct_safe_png_names() -> None:
    assert comparison._safe_case_filename("case 1") != comparison._safe_case_filename("case_1")
