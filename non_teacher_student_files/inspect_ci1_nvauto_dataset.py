"""Inspect converted CI-1 NVAUTO/MONAI cases."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from ci1_nvauto_common import read_csv, write_csv


FIELDS = [
    "case_id",
    "status",
    "reason",
    "timepoint",
    "dwi_exists",
    "adc_exists",
    "label_exists",
    "shape_match",
    "spacing_match",
    "label_binary",
    "label_voxels",
    "dwi_min",
    "dwi_max",
    "dwi_mean",
    "adc_min",
    "adc_max",
    "adc_mean",
    "has_nan_inf",
]


def stats(array: np.ndarray) -> tuple[float, float, float]:
    return float(np.nanmin(array)), float(np.nanmax(array)), float(np.nanmean(array))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect CI-1 NVAUTO dataset quality.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = args.dataset_root / "conversion_report.csv"
    conversion_rows = read_csv(report_path) if report_path.exists() else []
    timepoint_by_case = {row["case_id"]: row.get("timepoint", "") for row in conversion_rows}
    rows = []
    for case_dir in sorted((args.dataset_root / "cases").glob("case_*")):
        case_id = case_dir.name
        dwi_path = case_dir / "dwi.nii.gz"
        adc_path = case_dir / "adc.nii.gz"
        label_path = case_dir / "label.nii.gz"
        row = {
            "case_id": case_id,
            "timepoint": timepoint_by_case.get(case_id, ""),
            "dwi_exists": dwi_path.exists(),
            "adc_exists": adc_path.exists(),
            "label_exists": label_path.exists(),
        }
        try:
            dwi = sitk.ReadImage(str(dwi_path))
            adc = sitk.ReadImage(str(adc_path))
            label = sitk.ReadImage(str(label_path))
            dwi_array = sitk.GetArrayFromImage(dwi).astype(np.float32)
            adc_array = sitk.GetArrayFromImage(adc).astype(np.float32)
            label_array = sitk.GetArrayFromImage(label)
            label_values = np.unique(label_array)
            shape_match = dwi.GetSize() == adc.GetSize() == label.GetSize()
            spacing_match = dwi.GetSpacing() == adc.GetSpacing() == label.GetSpacing()
            dwi_min, dwi_max, dwi_mean = stats(dwi_array)
            adc_min, adc_max, adc_mean = stats(adc_array)
            has_bad = bool(
                np.any(~np.isfinite(dwi_array))
                or np.any(~np.isfinite(adc_array))
                or np.any(~np.isfinite(label_array.astype(np.float32)))
            )
            voxels = int(np.count_nonzero(label_array))
            status = "ok" if shape_match and spacing_match and set(label_values.tolist()).issubset({0, 1}) and voxels > 0 and not has_bad else "failed"
            row.update(
                {
                    "status": status,
                    "reason": "",
                    "shape_match": shape_match,
                    "spacing_match": spacing_match,
                    "label_binary": set(label_values.tolist()).issubset({0, 1}),
                    "label_voxels": voxels,
                    "dwi_min": dwi_min,
                    "dwi_max": dwi_max,
                    "dwi_mean": dwi_mean,
                    "adc_min": adc_min,
                    "adc_max": adc_max,
                    "adc_mean": adc_mean,
                    "has_nan_inf": has_bad,
                }
            )
        except Exception as exc:
            row.update({"status": "failed", "reason": repr(exc)})
        rows.append(row)

    write_csv(args.dataset_root / "qc_summary.csv", rows, FIELDS)
    ok_count = sum(1 for row in rows if row.get("status") == "ok")
    voxels = [int(row.get("label_voxels") or 0) for row in rows if row.get("label_voxels")]
    print(f"OK cases: {ok_count}")
    print(f"Failed cases: {len(rows) - ok_count}")
    print(f"Timepoints: {dict(Counter(row.get('timepoint', '') for row in rows))}")
    if voxels:
        print(f"Label voxels min/median/max: {min(voxels)}/{float(np.median(voxels))}/{max(voxels)}")


if __name__ == "__main__":
    main()

