"""Export CI-1 DWI, ADC, and FLAIR inputs for official DeepISLES inference."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ci1_nvauto_common import (
    dicom_series_to_sitk,
    read_csv,
    scan_dicom_series,
    select_flair_series,
    write_csv,
    write_image,
)


REPORT_FIELDS = [
    "case_id",
    "patient",
    "timepoint",
    "status",
    "reason",
    "source_timepoint_dir",
    "flair_series_description",
    "flair_series_number",
    "flair_raw_count",
    "flair_selected_count",
    "flair_selection_method",
    "output_case_dir",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export CI-1 cases with FLAIR for official DeepISLES Docker inference."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--max-cases", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def export_case(row: dict[str, str], output_root: Path, overwrite: bool) -> dict[str, object]:
    case_id = row["case_id"]
    output_case_dir = output_root / case_id
    result: dict[str, object] = {
        "case_id": case_id,
        "patient": row.get("patient", ""),
        "timepoint": row.get("timepoint", ""),
        "source_timepoint_dir": row.get("source_timepoint_dir", ""),
        "output_case_dir": str(output_case_dir),
        "status": "",
        "reason": "",
        "flair_series_description": "",
        "flair_series_number": "",
        "flair_raw_count": "",
        "flair_selected_count": "",
        "flair_selection_method": "",
    }
    source_timepoint_dir = Path(row["source_timepoint_dir"])
    source_case_dir = Path(row["output_case_dir"])
    if not source_timepoint_dir.is_dir():
        result.update(status="missing_source", reason="source_timepoint_dir does not exist")
        return result
    if not (source_case_dir / "dwi.nii.gz").is_file() or not (source_case_dir / "adc.nii.gz").is_file():
        result.update(status="missing_input", reason="converted DWI or ADC is missing")
        return result

    try:
        flair_series = select_flair_series(scan_dicom_series(source_timepoint_dir))
        if flair_series is None:
            result.update(status="missing_flair", reason="no conventional FLAIR DICOM series")
            return result
        flair_image, flair_meta = dicom_series_to_sitk(flair_series)
        output_case_dir.mkdir(parents=True, exist_ok=True)
        for name in ("dwi.nii.gz", "adc.nii.gz"):
            source = source_case_dir / name
            destination = output_case_dir / name
            if overwrite or not destination.exists():
                shutil.copy2(source, destination)
        flair_path = output_case_dir / "flair.nii.gz"
        if overwrite or not flair_path.exists():
            write_image(flair_image, flair_path)
        result.update(
            status="ok",
            flair_series_description=flair_series.series_description,
            flair_series_number=flair_series.series_number,
            flair_raw_count=flair_meta["raw_count"],
            flair_selected_count=flair_meta["selected_count"],
            flair_selection_method=flair_meta["selection_method"],
        )
    except Exception as exc:
        result.update(status="export_failed", reason=repr(exc))
    return result


def main() -> None:
    args = parse_args()
    source_report = args.dataset_root / "conversion_report.csv"
    rows = [row for row in read_csv(source_report) if row.get("status") == "ok"]
    if args.mode == "smoke":
        rows = rows[: args.max_cases]
    print(f"Candidate cases: {len(rows)}")
    results = []
    for row in rows:
        print(f"[CASE] {row['case_id']} {row.get('patient', '')} {row.get('timepoint', '')}")
        result = export_case(row, args.output_root, args.overwrite)
        results.append(result)
        print(f"  [{result['status']}] {result['reason']}")
    report_path = args.output_root / "deepisles_export_report.csv"
    write_csv(report_path, results, REPORT_FIELDS)
    ok_count = sum(result["status"] == "ok" for result in results)
    print(f"Wrote export report: {report_path}")
    print(f"DeepISLES-ready cases: {ok_count}")


if __name__ == "__main__":
    main()
