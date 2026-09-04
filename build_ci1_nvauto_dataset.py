"""Build a CI-1 DWI+ADC 3D dataset for NVAUTO/MONAI training."""

from __future__ import annotations

import argparse
from pathlib import Path

import SimpleITK as sitk

from ci1_nvauto_common import (
    DicomSeries,
    affine_swap_warning,
    binary_label,
    dicom_series_to_sitk,
    find_label_path,
    image_shape,
    image_spacing,
    iter_patient_dirs,
    iter_timepoint_dirs,
    label_voxels,
    read_nifti,
    resample_to_reference,
    safe_json_dump,
    scan_dicom_series,
    select_adc_series,
    select_dwi_series,
    transpose_xy_label,
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
    "dwi_series_description",
    "dwi_series_number",
    "dwi_raw_count",
    "dwi_selected_count",
    "dwi_b_value",
    "dwi_selection_method",
    "adc_series_description",
    "adc_series_number",
    "adc_count",
    "label_path",
    "dwi_shape",
    "adc_shape",
    "label_shape",
    "dwi_spacing",
    "adc_spacing",
    "label_spacing",
    "label_voxels",
    "affine_warning",
    "output_case_dir",
]


def series_fields(prefix: str, series: DicomSeries | None) -> dict[str, object]:
    if series is None:
        return {
            f"{prefix}_series_description": "",
            f"{prefix}_series_number": "",
        }
    return {
        f"{prefix}_series_description": series.series_description,
        f"{prefix}_series_number": series.series_number,
    }


def fail_row(
    patient: str,
    timepoint: str,
    source_timepoint_dir: Path,
    status: str,
    reason: str,
    case_id: str = "",
    output_case_dir: Path | None = None,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "patient": patient,
        "timepoint": timepoint,
        "status": status,
        "reason": reason,
        "source_timepoint_dir": str(source_timepoint_dir),
        "output_case_dir": str(output_case_dir or ""),
    }


def convert_one_case(
    case_id: str,
    patient: str,
    patient_dir: Path,
    timepoint: str,
    timepoint_dir: Path,
    output_root: Path,
    fix_xy_swap: bool,
) -> dict[str, object]:
    case_dir = output_root / "cases" / case_id
    warnings: list[str] = []
    try:
        series_list = scan_dicom_series(timepoint_dir)
        dwi_series = select_dwi_series(series_list)
        if dwi_series is None:
            return fail_row(patient, timepoint, timepoint_dir, "missing_dwi", "No DWI DICOM series", case_id, case_dir)
        adc_series = select_adc_series(series_list, dwi_series.series_number)
        if adc_series is None:
            return fail_row(patient, timepoint, timepoint_dir, "missing_adc", "No ADC DICOM series", case_id, case_dir)
        label_path = find_label_path(patient_dir, patient, timepoint)
        if label_path is None:
            return fail_row(patient, timepoint, timepoint_dir, "missing_label", "No DWI NIfTI label", case_id, case_dir)

        dwi_image, dwi_meta = dicom_series_to_sitk(dwi_series)
        adc_image, adc_meta = dicom_series_to_sitk(adc_series)
        adc_image = resample_to_reference(adc_image, dwi_image, sitk.sitkLinear)

        label_image = binary_label(resample_to_reference(read_nifti(label_path), dwi_image, sitk.sitkNearestNeighbor))
        voxels = label_voxels(label_image)
        if voxels <= 0:
            return fail_row(patient, timepoint, timepoint_dir, "empty_label", "Label has zero nonzero voxels", case_id, case_dir)

        warning = affine_swap_warning(dwi_image, label_image)
        if warning:
            warnings.append(warning)
            if fix_xy_swap:
                label_image = transpose_xy_label(label_image, dwi_image)
                warnings.append("xy_swap_fixed")

        print(f"  writing {case_dir / 'dwi.nii.gz'}")
        write_image(dwi_image, case_dir / "dwi.nii.gz")
        print(f"  writing {case_dir / 'adc.nii.gz'}")
        write_image(adc_image, case_dir / "adc.nii.gz")
        print(f"  writing {case_dir / 'label.nii.gz'}")
        write_image(label_image, case_dir / "label.nii.gz")

        meta = {
            "case_id": case_id,
            "patient": patient,
            "timepoint": timepoint,
            "source_timepoint_dir": str(timepoint_dir),
            "dwi_series_description": dwi_series.series_description,
            "dwi_series_uid": dwi_series.uid,
            "dwi_series_number": dwi_series.series_number,
            "dwi_raw_count": dwi_series.count,
            "dwi_selected_count": dwi_meta["selected_count"],
            "dwi_b_value": dwi_meta["b_value"],
            "dwi_selection_method": dwi_meta["selection_method"],
            "adc_series_description": adc_series.series_description,
            "adc_series_uid": adc_series.uid,
            "adc_series_number": adc_series.series_number,
            "adc_count": adc_meta["selected_count"],
            "label_source_path": str(label_path),
            "dwi_shape": image_shape(dwi_image),
            "adc_shape": image_shape(adc_image),
            "label_shape": image_shape(label_image),
            "dwi_spacing": image_spacing(dwi_image),
            "label_voxels": voxels,
            "status": "ok",
            "warnings": warnings,
        }
        safe_json_dump(meta, case_dir / "meta.json")
        status = "affine_warning" if warning and not fix_xy_swap else "ok"
        return {
            "case_id": case_id,
            "patient": patient,
            "timepoint": timepoint,
            "status": status,
            "reason": ";".join(warnings),
            "source_timepoint_dir": str(timepoint_dir),
            "dwi_series_description": dwi_series.series_description,
            "dwi_series_number": dwi_series.series_number,
            "dwi_raw_count": dwi_series.count,
            "dwi_selected_count": dwi_meta["selected_count"],
            "dwi_b_value": dwi_meta["b_value"] or "",
            "dwi_selection_method": dwi_meta["selection_method"],
            "adc_series_description": adc_series.series_description,
            "adc_series_number": adc_series.series_number,
            "adc_count": adc_meta["selected_count"],
            "label_path": str(label_path),
            "dwi_shape": image_shape(dwi_image),
            "adc_shape": image_shape(adc_image),
            "label_shape": image_shape(label_image),
            "dwi_spacing": image_spacing(dwi_image),
            "adc_spacing": image_spacing(adc_image),
            "label_spacing": image_spacing(label_image),
            "label_voxels": voxels,
            "affine_warning": warning,
            "output_case_dir": str(case_dir),
        }
    except Exception as exc:
        return fail_row(patient, timepoint, timepoint_dir, "conversion_failed", repr(exc), case_id, case_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CI-1 into NVAUTO/MONAI DWI+ADC cases.")
    parser.add_argument("--ci1-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--max-cases", type=int, default=5)
    parser.add_argument("--fix-xy-swap", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports: list[dict[str, object]] = []
    ok_count = 0
    attempted = 0
    for patient_dir in iter_patient_dirs(args.ci1_root):
        for timepoint, timepoint_dir in iter_timepoint_dirs(patient_dir):
            if args.mode == "smoke" and ok_count >= args.max_cases:
                break
            case_id = f"case_{attempted + 1:04d}"
            print(f"[CASE] {case_id} {patient_dir.name} {timepoint}")
            row = convert_one_case(
                case_id=case_id,
                patient=patient_dir.name,
                patient_dir=patient_dir,
                timepoint=timepoint,
                timepoint_dir=timepoint_dir,
                output_root=args.output_root,
                fix_xy_swap=args.fix_xy_swap,
            )
            reports.append(row)
            attempted += 1
            if row.get("status") == "ok":
                ok_count += 1
                print("  [OK]")
            else:
                print(f"  [{row.get('status')}] {row.get('reason')}")
        if args.mode == "smoke" and ok_count >= args.max_cases:
            break

    write_csv(args.output_root / "conversion_report.csv", reports, REPORT_FIELDS)
    print(f"Wrote conversion report: {args.output_root / 'conversion_report.csv'}")
    print(f"OK cases: {ok_count}")


if __name__ == "__main__":
    main()
