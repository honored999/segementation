"""Convert raw CI-1 DWI lesion data to nnU-Net v2 format.

中文说明：这个脚本从 CI-1 原始数据集中读取 DWI DICOM 原图和人工 GT 标注，
在显式指定的全新输出目录生成 nnU-Net v2 数据集。只做单通道
DWI，不使用 DeepISLES 预测输出作为训练标签；label 保留既有二值语义：所有非零值转为 1（不推断原始类别含义）。
CLI 必须显式提供 --b-value 和全新的 --output-root；不覆盖旧数据。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import SimpleITK as sitk

from ci1_nvauto_common import diffusion_b_value, require_pydicom
from prepare_ci1_dwi_dataset import FORBIDDEN_LABEL_KEYWORDS, build_index


TIMEPOINT_RE = re.compile(r"(D\d+)", re.IGNORECASE)
DWI_KEYWORDS = ("dwi", "b800")
LABEL_KEYWORDS = ("gt", "label", "mask", "lesion", "seg")


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
    requested_b_value: str = ""
    selected_b_value: float | None = None
    source_series_uid: str = ""
    source_dicom_files: tuple[str, ...] = ()
    selection_reason: str = ""


@dataclass(frozen=True)
class DwiSelection:
    files: tuple[Path, ...]
    requested_b_value: str
    selected_b_value: float
    series_uid: str
    available_b_values: tuple[float, ...]
    selection_reason: str


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


def parse_b_value(value: str) -> str:
    """Accept an explicit high/low choice or a finite nonnegative numeric b."""
    if value in ("high", "low"):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("b value must be high, low, or a nonnegative number") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError("b value must be finite and nonnegative")
    return str(value)


def select_dwi_dicom_files(
    dicom_dir: Path, b_value: str = "high", reference_image: sitk.Image | None = None,
) -> DwiSelection:
    """Select a complete b volume; resolve series by reference geometry/number.

    The helper defaults to high for existing callers; the CLI requires a choice.
    Reference matching uses size/spacing/origin/direction only, never voxel values.
    """
    requested = parse_b_value(b_value)
    pydicom = require_pydicom()
    grouped: dict[str, list[tuple[Path, object]]] = {}
    dwi_uids: set[str] = set()
    for path in sorted(dicom_dir.rglob("*.dcm")):
        try:
            header = pydicom.dcmread(str(path), stop_before_pixels=True)
        except Exception as exc:
            raise RuntimeError(f"Unreadable DICOM header: {path}: {exc}") from exc
        uid = str(getattr(header, "SeriesInstanceUID", ""))
        grouped.setdefault(uid, []).append((path, header))
        description = str(getattr(header, "SeriesDescription", "")).lower()
        if "dwi" in description and "adc" not in description:
            dwi_uids.add(uid)
    if not dwi_uids:
        raise RuntimeError(f"No DWI DICOM series found in {dicom_dir}")
    if "" in dwi_uids:
        raise RuntimeError(f"Missing DWI series UID in {dicom_dir}")
    if len(dwi_uids) == 1:
        uid = next(iter(dwi_uids))
        return _select_series_b_volume(uid, grouped[uid], requested)

    # Validate every candidate with the unchanged whole-series checks. Invalid
    # b/geometry must reject the case, not silently remove a competing series.
    candidates = [
        _select_series_b_volume(uid, grouped[uid], requested)
        for uid in sorted(dwi_uids)
    ]
    matches = []
    if reference_image is not None:
        for candidate in candidates:
            image = read_selected_dwi_image(candidate)
            if image.GetSize() == reference_image.GetSize() and all(
                geometry_close(getattr(image, f"Get{name}")(),
                               getattr(reference_image, f"Get{name}")())
                for name in ("Spacing", "Origin", "Direction")
            ):
                matches.append(candidate)

    def series_number(candidate: DwiSelection) -> int | None:
        values = {str(getattr(header, "SeriesNumber", "")).strip()
                  for _, header in grouped[candidate.series_uid]}
        if len(values) != 1:
            return None
        value = next(iter(values))
        return int(value) if re.fullmatch(r"[+-]?\d+", value) else None

    if len(matches) == 1:
        chosen = matches[0]
        reason = "unique_label_geometry_match"
    else:
        numbers = [series_number(candidate) for candidate in candidates]
        if any(number is None for number in numbers):
            raise RuntimeError("Cannot choose lowest SeriesNumber: missing/noninteger/inconsistent numbering")
        minimum = min(number for number in numbers if number is not None)
        minima = [candidate for candidate, number in zip(candidates, numbers) if number == minimum]
        if len(minima) != 1:
            raise RuntimeError(f"Ambiguous lowest DICOM SeriesNumber={minimum}")
        chosen = minima[0]
        reason = f"lowest_numeric_SeriesNumber; geometry_matches={len(matches)}"
    number = series_number(chosen)
    reason = (f"{chosen.selection_reason}; {reason}; selected_series_uid={chosen.series_uid}; "
              f"selected_SeriesNumber={number if number is not None else 'unknown'}")
    return DwiSelection(chosen.files, chosen.requested_b_value, chosen.selected_b_value,
                        chosen.series_uid, chosen.available_b_values, reason)


def _select_series_b_volume(
    uid: str, headers: Sequence[tuple[Path, object]], requested: str,
) -> DwiSelection:
    """Apply the original b/coverage/position guards independently per series."""
    frames: list[tuple[Path, float, tuple[float, ...], float, tuple[float, ...]]] = []
    reference_geometry = None
    normal = None
    for path, header in headers:
        try:
            frame_count = int(getattr(header, "NumberOfFrames", 1))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid DICOM frame count: {path}") from exc
        if frame_count != 1:
            raise RuntimeError(f"Multiframe DICOM is unsupported: {path}")
        actual_b = diffusion_b_value(header)
        if actual_b is None or not math.isfinite(actual_b) or actual_b < 0:
            raise RuntimeError(f"Missing/unknown/invalid b value in DWI header: {path}")
        try:
            orientation = np.asarray(header.ImageOrientationPatient, dtype=float)
            position = np.asarray(header.ImagePositionPatient, dtype=float)
            spacing = np.asarray(header.PixelSpacing, dtype=float)
            size = (int(header.Rows), int(header.Columns))
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Missing/invalid DICOM geometry: {path}") from exc
        if (orientation.shape != (6,) or position.shape != (3,) or spacing.shape != (2,)
                or not np.isfinite(np.concatenate((orientation, position, spacing))).all()
                or np.any(spacing <= 0) or min(size) <= 0):
            raise RuntimeError(f"Invalid DICOM geometry: {path}")
        if reference_geometry is None:
            reference_geometry = (orientation, spacing, size)
            normal = np.cross(orientation[:3], orientation[3:])
            if (not np.isclose(np.linalg.norm(orientation[:3]), 1, atol=1e-4)
                    or not np.isclose(np.linalg.norm(orientation[3:]), 1, atol=1e-4)
                    or not np.isclose(np.dot(orientation[:3], orientation[3:]), 0, atol=1e-4)):
                raise RuntimeError(f"Invalid DICOM orientation: {path}")
            normal = normal / np.linalg.norm(normal)
        else:
            ref_orientation, ref_spacing, ref_size = reference_geometry
            if (size != ref_size or not np.allclose(orientation, ref_orientation, atol=1e-5, rtol=0)
                    or not np.allclose(spacing, ref_spacing, atol=1e-5, rtol=0)):
                raise RuntimeError(f"Inconsistent DICOM geometry within series {uid}: {path}")
        key = tuple(float(v) for v in np.round(position, 4))
        frames.append((path, actual_b, key, float(np.dot(position, normal)), tuple(float(v) for v in position)))
    available = tuple(sorted({frame[1] for frame in frames}))
    chosen = max(available) if requested == "high" else min(available) if requested == "low" else float(requested)
    if chosen not in available:
        raise RuntimeError(f"Requested b={requested} unavailable in series {uid}; available={available}")
    # First filter the WHOLE series by b. Only then inspect duplicate positions.
    selected = [frame for frame in frames if frame[1] == chosen]
    common_positions = {frame[2] for frame in frames}
    if {frame[2] for frame in selected} != common_positions:
        raise RuntimeError(f"Incomplete b={chosen} coverage in series {uid}; requested={requested}, available={available}")
    if len(selected) != len(common_positions):
        raise RuntimeError(f"Ambiguous duplicate frames at b={chosen} in series {uid}")
    selected.sort(key=lambda frame: frame[3])
    if len(selected) > 1:
        distances = np.diff([frame[3] for frame in selected])
        if np.any(distances <= 1e-4) or not np.allclose(distances, distances[0], atol=1e-4, rtol=1e-4):
            raise RuntimeError(f"Nonuniform/ambiguous slice positions in series {uid}")
        # Use unrounded positions: projection alone cannot detect in-plane drift.
        # A 0.001 mm physical residual allows stored decimal rounding, not shear.
        positions = np.asarray([frame[4] for frame in selected], dtype=float)
        step = (selected[-1][3] - selected[0][3]) / (len(selected) - 1)
        expected = positions[0] + np.arange(len(selected))[:, None] * step * normal
        if np.any(np.linalg.norm(positions - expected, axis=1) > 1e-3):
            raise RuntimeError(f"Unsupported nonorthogonal/nonuniform 3D slice positions in series {uid}")
    reason = f"whole_series_{requested}; full_position_coverage; available_b_values={available}"
    return DwiSelection(tuple(frame[0] for frame in selected), requested, chosen, uid, available, reason)


def read_selected_dwi_image(selection: DwiSelection) -> sitk.Image:
    """Use the exact selected files and SimpleITK's native DICOM geometry."""
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(path) for path in selection.files])
    return reader.Execute()


def read_dwi_dicom_image(dicom_dir: Path, b_value: str = "high") -> sitk.Image:
    """Read one uniform b volume (helper default high; CLI choice is required)."""
    return read_selected_dwi_image(select_dwi_dicom_files(dicom_dir, b_value))


def build_dataset_json(num_training: int) -> dict[str, object]:
    """Return nnU-Net v2 dataset.json content."""
    return {
        "channel_names": {"0": "DWI"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }


def validate_output_separation(output_root: Path, source: Path) -> None:
    """Reject equal, descendant and ancestor paths after resolving links."""
    output, source = output_root.resolve(), source.resolve()
    if output == source or output in source.parents or source in output.parents:
        raise ValueError(f"Output/input overlap is forbidden: output={output}, source={source}")


def prepare_output_dirs(output_root: Path, clean: bool = False) -> None:
    """Create a fresh dataset only; never delete or reuse an existing path."""
    if clean:
        raise ValueError("--clean is disabled; choose a fresh --output-root")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Output root already exists; choose a fresh path: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    for name in ("imagesTr", "labelsTr", "imagesTs"):
        (output_root / name).mkdir()


def convert_cases(
    cases: Sequence[CaseCandidate],
    output_root: Path,
    limit: int = 0,
    b_value: str = "high",
    rejected_cases: list[dict[str, object]] | None = None,
) -> list[WrittenCase]:
    """Convert candidate cases to nnU-Net files."""
    b_value = parse_b_value(b_value)
    for name in ("dataset.json", "conversion_manifest.tsv", "conversion_rejections.tsv"):
        if (output_root / name).exists() or (output_root / name).is_symlink():
            raise FileExistsError(f"Refusing to reuse an existing dataset: {output_root}")
    for candidate in cases:
        validate_output_separation(output_root, candidate.dicom_dir)
        validate_output_separation(output_root, candidate.label_path)
    selected_cases = list(cases[:limit]) if limit > 0 else list(cases)
    written: list[WrittenCase] = []
    for candidate in selected_cases:
        print(
            f"[CASE] {candidate.patient} {candidate.timepoint}\n"
            f"  dicom: {candidate.dicom_dir}\n"
            f"  label: {candidate.label_path}"
        )
        selection = None
        try:
            if is_forbidden_label_path(candidate.label_path):
                raise RuntimeError(f"Forbidden predicted/generated label: {candidate.label_path}")
            label = read_nifti_image(candidate.label_path)
            selection = select_dwi_dicom_files(candidate.dicom_dir, b_value, reference_image=label)
            image = read_selected_dwi_image(selection)
            if image.GetSize() != label.GetSize():
                raise RuntimeError(f"Shape mismatch image={image.GetSize()} label={label.GetSize()}")
            for name in ("Spacing", "Origin", "Direction"):
                if not geometry_close(getattr(image, f"Get{name}")(), getattr(label, f"Get{name}")()):
                    raise RuntimeError(f"Geometry mismatch: {name}; no registration/header repair performed")
        except RuntimeError as exc:
            print(f"  [SKIP] requested b={b_value}: {exc}")
            if rejected_cases is not None:
                rejected_cases.append({
                    "patient": candidate.patient, "timepoint": candidate.timepoint,
                    "source_dicom_dir": candidate.dicom_dir, "source_label": candidate.label_path,
                    "requested_b_value": b_value,
                    "selected_b_value": selection.selected_b_value if selection else "",
                    "source_series_uid": selection.series_uid if selection else "",
                    "selection_reason": selection.selection_reason if selection else "",
                    "source_dicom_files": json.dumps([str(path) for path in selection.files], ensure_ascii=False) if selection else "",
                    "selected_slice_count": len(selection.files) if selection else "",
                    "status": "rejected", "failure_reason": str(exc),
                })
            continue

        case_id = f"case{len(written) + 1:03d}"
        output_image = output_root / "imagesTr" / f"{case_id}_0000.nii.gz"
        output_label = output_root / "labelsTr" / f"{case_id}.nii.gz"
        if any(path.exists() or path.is_symlink() for path in (output_image, output_label)):
            raise FileExistsError(f"Refusing to overwrite existing case output: {case_id}")
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
                requested_b_value=selection.requested_b_value,
                selected_b_value=selection.selected_b_value,
                source_series_uid=selection.series_uid,
                source_dicom_files=tuple(str(path) for path in selection.files),
                selection_reason=selection.selection_reason,
            )
        )
        print(f"  [OK] wrote {case_id}; requested b={b_value}, selected b={selection.selected_b_value}")
    return written


def write_dataset_json(output_root: Path, num_training: int) -> None:
    """Write dataset.json in the output root."""
    dataset_path = output_root / "dataset.json"
    with dataset_path.open("x", encoding="utf-8") as handle:
        json.dump(build_dataset_json(num_training), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_conversion_manifest(
    output_root: Path, written_cases: Sequence[WrittenCase],
    rejected_cases: Sequence[dict[str, object]] = (),
) -> None:
    """Keep the successful-case manifest separate from rejected-case diagnostics."""
    fields = ["case_id", "patient", "timepoint", "source_dicom_dir", "source_label",
              "output_image", "output_label", "requested_b_value", "selected_b_value",
              "source_series_uid", "source_dicom_files", "selected_slice_count",
              "selection_reason", "status", "failure_reason"]
    with (output_root / "conversion_manifest.tsv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for case in written_cases:
            row = dict(vars(case))
            row["source_dicom_files"] = json.dumps(case.source_dicom_files, ensure_ascii=False)
            row["selected_slice_count"] = len(case.source_dicom_files)
            row["status"] = "written"
            writer.writerow(row)
    if rejected_cases:
        with (output_root / "conversion_rejections.tsv").open("x", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(rejected_cases)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert raw CI-1 DWI lesion data into nnU-Net v2 format."
    )
    parser.add_argument("--ci1-root", type=Path, default=Path("CI-1"))
    parser.add_argument("--output-root", type=Path, required=True, help="Fresh dataset path outside the input tree.")
    parser.add_argument("--b-value", type=parse_b_value, required=True,
                        help="Required manual choice: high, low, or a nonnegative numeric b value.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Disabled: existing outputs are never deleted; use a fresh output path.",
    )
    args = parser.parse_args()
    if args.clean:
        parser.error("--clean is disabled; choose a fresh --output-root")
    return args


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    if not args.ci1_root.exists():
        raise FileNotFoundError(f"CI-1 root does not exist: {args.ci1_root}")
    if not args.ci1_root.is_dir():
        raise NotADirectoryError(f"CI-1 root is not a directory: {args.ci1_root}")

    validate_output_separation(args.output_root, args.ci1_root)
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(f"Output root already exists: {args.output_root}")
    cases = find_case_candidates(args.ci1_root)
    prepare_output_dirs(args.output_root, clean=args.clean)
    print(f"Found candidate cases: {len(cases)}")
    rejected_cases: list[dict[str, object]] = []
    written_cases = convert_cases(cases, args.output_root, limit=args.limit,
                                  b_value=args.b_value, rejected_cases=rejected_cases)
    write_dataset_json(args.output_root, num_training=len(written_cases))
    write_conversion_manifest(args.output_root, written_cases, rejected_cases)
    print(f"Wrote nnU-Net dataset: {args.output_root}")
    print(f"Successful training cases: {len(written_cases)}")


if __name__ == "__main__":
    main()
