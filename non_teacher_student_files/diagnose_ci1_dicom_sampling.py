"""Diagnose DICOM slice spacing for indexed CI-1 DWI series.

中文说明：
这个脚本检查 CI-1 DWI DICOM 序列的切片间距和重复切片情况，
用于定位 SimpleITK 提示 non-uniform sampling 或 missing slices 的具体病例。

The converter can trigger SimpleITK warnings such as:
    Non uniform sampling or missing slices detected

This script maps that class of warning back to concrete index rows by reading
DICOM metadata for the same series selection used by convert_ci1_dwi_to_2d.py.
It writes a CSV report under data/ and does not read pixels or modify CI-1.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from convert_ci1_dwi_to_2d import (
    collect_slice_infos,
    get_largest_series_files,
    read_index,
    select_unique_slice_files,
)


POSITION_TAG = "0020|0032"
ORIENTATION_TAG = "0020|0037"


def spacing_stats(files: list[str]) -> dict[str, str]:
    slice_infos = collect_slice_infos(files)
    selected_files = select_unique_slice_files(slice_infos)
    if len(files) < 3:
        return {
            "slice_count": str(len(files)),
            "selected_slice_count": str(len(selected_files)),
            "median_spacing": "",
            "max_abs_deviation": "",
            "duplicate_position_steps": "0",
            "nonuniform": "0",
            "reason": "too_few_slices",
        }

    sorted_positions = np.sort(
        np.asarray([item.slice_position for item in slice_infos], dtype=np.float64)
    )
    diffs = np.diff(sorted_positions)
    duplicate_steps = int(np.count_nonzero(np.abs(diffs) <= 1e-6))
    nonzero_diffs = np.abs(diffs[np.abs(diffs) > 1e-6])
    if nonzero_diffs.size == 0:
        return {
            "slice_count": str(len(files)),
            "selected_slice_count": str(len(selected_files)),
            "median_spacing": "0",
            "max_abs_deviation": "0",
            "duplicate_position_steps": str(duplicate_steps),
            "nonuniform": "1",
            "reason": "duplicate_positions",
        }

    median_spacing = float(np.median(nonzero_diffs))
    max_abs_deviation = float(np.max(np.abs(nonzero_diffs - median_spacing)))
    tolerance = max(1e-3, 0.05 * median_spacing)
    nonuniform = max_abs_deviation > tolerance or duplicate_steps > 0

    return {
        "slice_count": str(len(files)),
        "selected_slice_count": str(len(selected_files)),
        "median_spacing": f"{median_spacing:.6g}",
        "max_abs_deviation": f"{max_abs_deviation:.6g}",
        "duplicate_position_steps": str(duplicate_steps),
        "nonuniform": "1" if nonuniform else "0",
        "reason": (
            "duplicate_positions"
            if duplicate_steps > 0
            else "nonuniform_spacing"
            if nonuniform
            else "ok"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map DICOM sampling warnings to CI-1 DWI index rows."
    )
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=Path("data") / "ci1_dwi_index.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data") / "ci1_dicom_sampling_report.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_index(args.index_csv)
    report_rows = []
    for row in rows:
        files = get_largest_series_files(row.dicom_dir)
        stats = spacing_stats(files)
        report_rows.append(
            {
                "patient": row.patient,
                "timepoint": row.timepoint,
                "dicom_dir": str(row.dicom_dir),
                "segmentation_path": str(row.segmentation_path),
                **stats,
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "patient",
        "timepoint",
        "dicom_dir",
        "segmentation_path",
        "slice_count",
        "selected_slice_count",
        "median_spacing",
        "max_abs_deviation",
        "duplicate_position_steps",
        "nonuniform",
        "reason",
    ]
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    affected = [row for row in report_rows if row["nonuniform"] == "1"]
    print(f"Wrote: {args.output_csv}")
    print(f"Rows checked: {len(report_rows)}")
    print(f"Nonuniform rows: {len(affected)}")
    for row in affected[:30]:
        print(
            row["patient"],
            row["timepoint"],
            row["slice_count"],
            row["median_spacing"],
            row["max_abs_deviation"],
            row["duplicate_position_steps"],
            row["dicom_dir"],
        )


if __name__ == "__main__":
    main()
