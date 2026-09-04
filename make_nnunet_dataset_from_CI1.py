"""Convert raw CI-1 DWI lesion data to nnU-Net v2 format.

中文说明：这个脚本从 CI-1 原始数据集中读取 DWI DICOM 原图和人工 GT 标注，
生成 nnU-Net v2 所需的 Dataset501_StrokeLesion 目录。第一版只做单通道
DWI，不使用 DeepISLES 预测输出作为训练标签；label 会被二值化为 0/1。
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import SimpleITK as sitk

from audit_ci1_dwi_adc_masks import background_predicate, read_series_image
from prepare_ci1_dwi_dataset import build_index


DEFAULT_OUTPUT_ROOT = Path(r"C:\lijialin\models3d\nnUNet\nnUNet_raw\Dataset501_StrokeLesion")
TIMEPOINT_RE = re.compile(r"(D\d+)", re.IGNORECASE)
DWI_KEYWORDS = ("dwi", "b800")
LABEL_KEYWORDS = ("gt", "label", "mask", "lesion", "seg")
FORBIDDEN_LABEL_KEYWORDS = ("pred", "output", "deepisles", "xy_fixed")


@dataclass(frozen=True)
class CaseCandidate:
    patient: str
    timepoint: str
    dicom_dir: Path
    label_path: Path


@dataclass(frozen=True)
class WrittenCase:
    case_id: str
    patient: str
    timepoint: str
    source_dicom_dir: Path
    source_label: Path
    output_image: Path
    output_label: Path


def is_nifti_path(path: Path) -> bool:
    """Return True for .nii and .nii.gz files."""
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def is_forbidden_label_path(path: Path) -> bool:
    """Reject labels that look like predictions or generated outputs."""
    text = str(path).lower()
    return any(keyword in text for keyword in FORBIDDEN_LABEL_KEYWORDS)


def has_any_keyword(path: Path, keywords: Sequence[str]) -> bool:
    """Check keywords against full path text."""
    text = str(path).lower()
    return any(keyword in text for keyword in keywords)


def is_readable_nifti_path(path: Path) -> bool:
    """Return True when an existing NIfTI file can be opened by SimpleITK."""
    if not path.exists():
        return True
    try:
        read_nifti_image(path)
    except RuntimeError as exc:
        print(f"[SKIP] unreadable NIfTI candidate: {path} ({exc})")
        return False
    return True


def ascii_nifti_name(path: Path) -> str:
    """Return an ASCII fallback filename preserving NIfTI extension."""
    lower_name = path.name.lower()
    if lower_name.endswith(".nii.gz"):
        return "image.nii.gz"
    if lower_name.endswith(".nii"):
        return "image.nii"
    return "image.nii.gz"


def read_nifti_image(path: Path) -> sitk.Image:
    """Read NIfTI robustly, including Windows paths with non-ASCII characters."""
    try:
        return sitk.ReadImage(str(path))
    except RuntimeError:
        with tempfile.TemporaryDirectory(prefix="ci1_nifti_") as tmp_dir:
            tmp_path = Path(tmp_dir) / ascii_nifti_name(path)
            shutil.copy2(path, tmp_path)
            return sitk.ReadImage(str(tmp_path))


def extract_timepoint(path: Path) -> str | None:
    """Extract D1/D2/D3/D7-like timepoint from a path."""
    match = TIMEPOINT_RE.search(str(path))
    if match is None:
        return None
    return match.group(1).upper()


def iter_patient_dirs(ci1_root: Path) -> Iterable[Path]:
    """Yield immediate patient directories under CI-1."""
    for path in sorted(ci1_root.iterdir(), key=lambda item: item.name):
        if path.is_dir():
            yield path


def iter_nifti_files(root: Path) -> list[Path]:
    """Find NIfTI files recursively."""
    return [
        path
        for path in sorted(root.rglob("*"), key=lambda item: str(item))
        if path.is_file() and is_nifti_path(path)
    ]


def image_candidate_score(path: Path) -> tuple[int, int, int, str]:
    """Score possible DWI image files, higher tuple sorts first."""
    text = str(path).lower()
    score = 0
    if "b800" in text:
        score += 40
    if "dwi" in text:
        score += 30
    if any(keyword in text for keyword in LABEL_KEYWORDS):
        score -= 25
    if is_forbidden_label_path(path):
        score -= 50
    if "adc" in text or "flair" in text:
        score -= 20
    suffix_score = 1 if path.name.lower().endswith(".nii.gz") else 0
    size_score = min(path.stat().st_size if path.exists() else 0, 10_000_000)
    return (score, suffix_score, size_score, str(path))


def label_candidate_score(path: Path) -> tuple[int, int, int, str]:
    """Score possible human GT label files, higher tuple sorts first."""
    text = str(path).lower()
    score = 0
    if is_forbidden_label_path(path):
        score -= 1000
    if any(keyword in text for keyword in LABEL_KEYWORDS):
        score += 80
    if "dwi" in text or "b800" in text:
        score += 20
    if "adc" in text or "flair" in text:
        score -= 10
    suffix_score = 1 if path.name.lower().endswith(".nii.gz") else 0
    # GT masks in CI-1 are often small NIfTI files, so prefer smaller files as a tie-breaker.
    inverse_size = -min(path.stat().st_size if path.exists() else 0, 10_000_000)
    return (score, suffix_score, inverse_size, str(path))


def pick_best_dwi_image(candidates: Sequence[Path]) -> Path | None:
    """Pick the best DWI/b800 image candidate."""
    filtered = [
        path
        for path in candidates
        if is_nifti_path(path) and has_any_keyword(path, DWI_KEYWORDS)
        and is_readable_nifti_path(path)
    ]
    if not filtered:
        return None
    return max(filtered, key=image_candidate_score)


def pick_best_label(candidates: Sequence[Path]) -> Path | None:
    """Pick the best label candidate, never using forbidden prediction outputs."""
    filtered = [
        path
        for path in candidates
        if is_nifti_path(path) and not is_forbidden_label_path(path)
        and is_readable_nifti_path(path)
    ]
    if not filtered:
        return None

    keyword_labels = [
        path for path in filtered if has_any_keyword(path, LABEL_KEYWORDS)
    ]
    if keyword_labels:
        return max(keyword_labels, key=label_candidate_score)

    dwi_fallback = [
        path for path in filtered if has_any_keyword(path, DWI_KEYWORDS)
    ]
    if dwi_fallback:
        return max(dwi_fallback, key=label_candidate_score)
    return None


def group_files_by_timepoint(patient_dir: Path) -> dict[str, list[Path]]:
    """Group one patient's NIfTI files by D timepoint."""
    grouped: dict[str, list[Path]] = {}
    for path in iter_nifti_files(patient_dir):
        timepoint = extract_timepoint(path.relative_to(patient_dir))
        if timepoint is None:
            continue
        grouped.setdefault(timepoint, []).append(path)
    return grouped


def find_case_candidates(ci1_root: Path) -> list[CaseCandidate]:
    """Find DWI DICOM directories and DWI GT labels in the raw CI-1 tree."""
    cases: list[CaseCandidate] = []
    for row in build_index(ci1_root):
        if row.get("match_status") != "matched":
            print(
                f"[SKIP] {row.get('patient')} {row.get('timepoint')}: "
                f"no matched DICOM directory"
            )
            continue
        label_path = Path(row["segmentation_path"])
        dicom_dir = Path(row["dicom_dir"])
        if is_forbidden_label_path(label_path):
            print(f"[SKIP] forbidden label path: {label_path}")
            continue
        if not is_readable_nifti_path(label_path):
            continue
        if not dicom_dir.is_dir():
            print(f"[SKIP] missing DICOM directory: {dicom_dir}")
            continue
        cases.append(
            CaseCandidate(
                patient=row["patient"],
                timepoint=row["timepoint"],
                dicom_dir=dicom_dir,
                label_path=label_path,
            )
        )
    return cases


def geometry_close(left: tuple[float, ...], right: tuple[float, ...], tol: float = 1e-5) -> bool:
    """Compare SimpleITK geometry tuples with a small tolerance."""
    if len(left) != len(right):
        return False
    return all(abs(a - b) <= tol for a, b in zip(left, right))


def print_geometry_warnings(case: CaseCandidate, image: sitk.Image, label: sitk.Image) -> None:
    """Warn when spacing/origin/direction do not match."""
    checks = [
        ("spacing", image.GetSpacing(), label.GetSpacing()),
        ("origin", image.GetOrigin(), label.GetOrigin()),
        ("direction", image.GetDirection(), label.GetDirection()),
    ]
    for name, image_value, label_value in checks:
        if not geometry_close(tuple(image_value), tuple(label_value)):
            print(
                f"[WARN] {case.patient} {case.timepoint}: {name} mismatch "
                f"image={image_value} label={label_value}"
            )


def save_binary_label(label_path: Path, output_path: Path) -> None:
    """Read a label NIfTI, convert all nonzero values to 1, and save uint8."""
    label = read_nifti_image(label_path)
    label_array = sitk.GetArrayFromImage(label)
    binary = (label_array != 0).astype(np.uint8)
    output = sitk.GetImageFromArray(binary)
    output.CopyInformation(label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(output, str(output_path))


def save_binary_label_image(label: sitk.Image, output_path: Path) -> None:
    """Convert a label image to uint8 binary values and save it."""
    label_array = sitk.GetArrayFromImage(label)
    binary = (label_array != 0).astype(np.uint8)
    output = sitk.GetImageFromArray(binary)
    output.CopyInformation(label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(output, str(output_path))


def save_dicom_image_as_nii_gz(image: sitk.Image, output_path: Path) -> None:
    """Save a DICOM volume read by SimpleITK as .nii.gz."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(output_path))


def read_dwi_dicom_image(dicom_dir: Path) -> sitk.Image:
    """Read the DWI DICOM series from a CI-1 case directory."""
    image, _description = read_series_image(dicom_dir, background_predicate("dwi"))
    return image


def build_dataset_json(num_training: int) -> dict[str, object]:
    """Return nnU-Net v2 dataset.json content."""
    return {
        "channel_names": {"0": "DWI"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }


def prepare_output_dirs(output_root: Path, clean: bool) -> None:
    """Create nnU-Net output directories, optionally removing old contents."""
    if clean and output_root.exists():
        shutil.rmtree(output_root)
    (output_root / "imagesTr").mkdir(parents=True, exist_ok=True)
    (output_root / "labelsTr").mkdir(parents=True, exist_ok=True)
    (output_root / "imagesTs").mkdir(parents=True, exist_ok=True)


def convert_cases(
    cases: Sequence[CaseCandidate],
    output_root: Path,
    limit: int = 0,
) -> list[WrittenCase]:
    """Convert candidate cases to nnU-Net files."""
    selected_cases = list(cases[:limit]) if limit > 0 else list(cases)
    written: list[WrittenCase] = []
    for candidate in selected_cases:
        print(
            f"[CASE] {candidate.patient} {candidate.timepoint}\n"
            f"  dicom: {candidate.dicom_dir}\n"
            f"  label: {candidate.label_path}"
        )
        try:
            image = read_dwi_dicom_image(candidate.dicom_dir)
            label = read_nifti_image(candidate.label_path)
        except RuntimeError as exc:
            print(f"  [SKIP] unable to read image or label: {exc}")
            continue
        if image.GetSize() != label.GetSize():
            print(
                f"  [SKIP] shape mismatch image={image.GetSize()} "
                f"label={label.GetSize()}"
            )
            continue
        print_geometry_warnings(candidate, image, label)

        case_id = f"case{len(written) + 1:03d}"
        output_image = output_root / "imagesTr" / f"{case_id}_0000.nii.gz"
        output_label = output_root / "labelsTr" / f"{case_id}.nii.gz"
        save_dicom_image_as_nii_gz(image, output_image)
        save_binary_label_image(label, output_label)
        written.append(
            WrittenCase(
                case_id=case_id,
                patient=candidate.patient,
                timepoint=candidate.timepoint,
                source_dicom_dir=candidate.dicom_dir,
                source_label=candidate.label_path,
                output_image=output_image,
                output_label=output_label,
            )
        )
        print(f"  [OK] wrote {case_id}")
    return written


def write_dataset_json(output_root: Path, num_training: int) -> None:
    """Write dataset.json in the output root."""
    dataset_path = output_root / "dataset.json"
    with dataset_path.open("w", encoding="utf-8") as handle:
        json.dump(build_dataset_json(num_training), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_conversion_manifest(output_root: Path, written_cases: Sequence[WrittenCase]) -> None:
    """Write a source-to-output manifest for audit purposes."""
    manifest_path = output_root / "conversion_manifest.tsv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        handle.write(
            "case_id\tpatient\ttimepoint\tsource_dicom_dir\tsource_label\toutput_image\toutput_label\n"
        )
        for case in written_cases:
            handle.write(
                f"{case.case_id}\t{case.patient}\t{case.timepoint}\t"
                f"{case.source_dicom_dir}\t{case.source_label}\t"
                f"{case.output_image}\t{case.output_label}\n"
            )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert raw CI-1 DWI lesion data into nnU-Net v2 format."
    )
    parser.add_argument("--ci1-root", type=Path, default=Path("CI-1"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the output dataset directory before conversion.",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    if not args.ci1_root.exists():
        raise FileNotFoundError(f"CI-1 root does not exist: {args.ci1_root}")
    if not args.ci1_root.is_dir():
        raise NotADirectoryError(f"CI-1 root is not a directory: {args.ci1_root}")

    prepare_output_dirs(args.output_root, clean=args.clean)
    cases = find_case_candidates(args.ci1_root)
    print(f"Found candidate cases: {len(cases)}")
    written_cases = convert_cases(cases, args.output_root, limit=args.limit)
    write_dataset_json(args.output_root, num_training=len(written_cases))
    write_conversion_manifest(args.output_root, written_cases)
    print(f"Wrote nnU-Net dataset: {args.output_root}")
    print(f"Successful training cases: {len(written_cases)}")


if __name__ == "__main__":
    main()
