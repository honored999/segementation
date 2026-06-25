"""Build a DWI image/segmentation index for the local CI-1 dataset.

This script does not read image pixels and does not modify the source dataset.
It scans the CI-1 directory, finds DWI segmentation files saved as NIfTI
(`.nii` / `.nii.gz`), and pairs each one with the most likely DICOM directory
for the same patient and time point.

Output:
    data/ci1_dwi_index.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TIMEPOINT_RE = re.compile(r"(?:^|[-_ ])(D\d+)(?:[-_ ]|$)", re.IGNORECASE)


@dataclass(frozen=True)
class DwiSegmentation:
    patient: str
    timepoint: str
    path: Path


@dataclass(frozen=True)
class DicomSeries:
    patient: str
    timepoint: str
    path: Path
    dicom_count: int


def is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def extract_timepoint(text: str) -> str | None:
    match = TIMEPOINT_RE.search(text)
    if match is None:
        return None
    return match.group(1).upper()


def is_dwi_name(name: str) -> bool:
    lower_name = name.lower()
    return "dwi" in lower_name and "adc" not in lower_name


def iter_patient_dirs(ci1_root: Path) -> Iterable[Path]:
    for path in sorted(ci1_root.iterdir(), key=lambda item: item.name):
        if path.is_dir():
            yield path


def find_dwi_segmentations(ci1_root: Path) -> list[DwiSegmentation]:
    grouped: dict[tuple[str, str], DwiSegmentation] = {}
    duplicate_count = 0
    for patient_dir in iter_patient_dirs(ci1_root):
        for path in sorted(patient_dir.rglob("*"), key=lambda item: str(item)):
            if not path.is_file() or not is_nifti(path) or not is_dwi_name(path.name):
                continue
            timepoint = extract_timepoint(path.name)
            if timepoint is None:
                timepoint = extract_timepoint(str(path.relative_to(patient_dir)))
            if timepoint is None:
                continue
            segmentation = DwiSegmentation(
                patient=patient_dir.name,
                timepoint=timepoint,
                path=path,
            )
            key = (segmentation.patient, segmentation.timepoint)
            existing = grouped.get(key)
            if existing is None:
                grouped[key] = segmentation
                continue

            duplicate_count += 1
            existing_depth = len(existing.path.relative_to(patient_dir).parts)
            new_depth = len(segmentation.path.relative_to(patient_dir).parts)
            if new_depth < existing_depth:
                grouped[key] = segmentation

    if duplicate_count:
        print(f"Skipped duplicate DWI segmentations: {duplicate_count}")
    return list(grouped.values())


def find_dicom_series(ci1_root: Path) -> list[DicomSeries]:
    series_list: list[DicomSeries] = []
    for patient_dir in iter_patient_dirs(ci1_root):
        for candidate in sorted(patient_dir.iterdir(), key=lambda item: item.name):
            if not candidate.is_dir():
                continue
            dicom_count = sum(1 for path in candidate.rglob("*.dcm") if path.is_file())
            if dicom_count == 0:
                continue
            timepoint = extract_timepoint(candidate.name)
            if timepoint is None:
                continue
            series_list.append(
                DicomSeries(
                    patient=patient_dir.name,
                    timepoint=timepoint,
                    path=candidate,
                    dicom_count=dicom_count,
                )
            )
    return series_list


def choose_dicom_series(
    segmentation: DwiSegmentation,
    dicom_series: list[DicomSeries],
) -> DicomSeries | None:
    candidates = [
        series
        for series in dicom_series
        if series.patient == segmentation.patient
        and series.timepoint == segmentation.timepoint
    ]
    if not candidates:
        return None

    dwi_candidates = [
        series for series in candidates if "dwi" in series.path.name.lower()
    ]
    if dwi_candidates:
        candidates = dwi_candidates

    return max(candidates, key=lambda series: series.dicom_count)


def write_index(
    rows: list[dict[str, str]],
    output_csv: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "patient",
        "timepoint",
        "modality",
        "segmentation_path",
        "dicom_dir",
        "dicom_count",
        "match_status",
    ]
    with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_index(ci1_root: Path) -> list[dict[str, str]]:
    segmentations = find_dwi_segmentations(ci1_root)
    dicom_series = find_dicom_series(ci1_root)

    rows: list[dict[str, str]] = []
    for segmentation in segmentations:
        matched_series = choose_dicom_series(segmentation, dicom_series)
        rows.append(
            {
                "patient": segmentation.patient,
                "timepoint": segmentation.timepoint,
                "modality": "DWI",
                "segmentation_path": str(segmentation.path),
                "dicom_dir": str(matched_series.path) if matched_series else "",
                "dicom_count": str(matched_series.dicom_count) if matched_series else "0",
                "match_status": "matched" if matched_series else "missing_dicom",
            }
        )

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a CSV index of CI-1 DWI segmentations and DICOM series."
    )
    parser.add_argument(
        "--ci1-root",
        type=Path,
        default=Path("CI-1"),
        help="Path to the CI-1 dataset root.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data") / "ci1_dwi_index.csv",
        help="CSV file to write.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ci1_root = args.ci1_root
    if not ci1_root.exists():
        raise FileNotFoundError(f"CI-1 root does not exist: {ci1_root}")
    if not ci1_root.is_dir():
        raise NotADirectoryError(f"CI-1 root is not a directory: {ci1_root}")

    rows = build_index(ci1_root)
    write_index(rows, args.output_csv)

    matched_count = sum(1 for row in rows if row["match_status"] == "matched")
    missing_count = len(rows) - matched_count
    print(f"Wrote: {args.output_csv}")
    print(f"DWI segmentations: {len(rows)}")
    print(f"Matched DICOM dirs: {matched_count}")
    print(f"Missing DICOM dirs: {missing_count}")


if __name__ == "__main__":
    main()
