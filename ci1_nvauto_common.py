"""Shared helpers for converting CI-1 into an NVAUTO/MONAI-style dataset."""

from __future__ import annotations

import csv
import json
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import SimpleITK as sitk


TIMEPOINT_RE = re.compile(r"(D\d+)", re.IGNORECASE)
DWI_INCLUDE = ("dwi", "diff", "diffusion", "axi fse-dwi", "trace", "b800", "b1000")
DWI_EXCLUDE = ("3_plane_loc", "localizer", "scout", "survey", "mra", "mip", "t1")
ADC_INCLUDE = ("adc", "adc from", "apparent diffusion coefficient")
FLAIR_INCLUDE = ("flair", "fluid attenuated inversion recovery")
FLAIR_EXCLUDE = (*DWI_EXCLUDE, "t1", "adc", "dwi", "diff", "diffusion")
LABEL_EXCLUDE = (
    "flair-adc",
    "flair_adc",
    "deepisles",
    "lesion_msk",
    ".png",
    "prediction",
    "pred",
)


def require_pydicom():
    """Import pydicom with a helpful message for server setup."""
    try:
        import pydicom  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pydicom is required for CI-1 DICOM conversion. Install it in the "
            "server environment, for example: pip install pydicom"
        ) from exc
    return pydicom


@dataclass
class DicomSeries:
    uid: str
    files: list[Path]
    series_number: str = ""
    series_description: str = ""
    protocol_name: str = ""
    sequence_name: str = ""
    image_type: str = ""
    rows: str = ""
    columns: str = ""
    pixel_spacing: str = ""
    slice_thickness: str = ""
    spacing_between_slices: str = ""
    has_pixel_data: bool = False
    b_values: list[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.files)

    def text(self) -> str:
        return " ".join(
            [
                self.series_description,
                self.protocol_name,
                self.sequence_name,
                self.image_type,
            ]
        ).lower()


def is_nifti_path(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def extract_timepoint(text_or_path: str | Path) -> str | None:
    match = TIMEPOINT_RE.search(str(text_or_path))
    return match.group(1).upper() if match else None


def iter_patient_dirs(ci1_root: Path) -> Iterable[Path]:
    for path in sorted(ci1_root.iterdir(), key=lambda item: item.name):
        if path.is_dir():
            yield path


def iter_timepoint_dirs(patient_dir: Path) -> Iterable[tuple[str, Path]]:
    for path in sorted(patient_dir.iterdir(), key=lambda item: item.name):
        if not path.is_dir():
            continue
        timepoint = extract_timepoint(path.name)
        if timepoint and any(path.rglob("*.dcm")):
            yield timepoint, path


def safe_json_dump(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_csv(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def dicom_value(ds: object, name: str, default: object = "") -> object:
    value = getattr(ds, name, default)
    if value is None:
        return default
    return value


def stringify(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\\".join(str(item) for item in value)
    return str(value)


def diffusion_b_value(ds: object) -> float | None:
    candidates = []
    if hasattr(ds, "DiffusionBValue"):
        candidates.append(getattr(ds, "DiffusionBValue"))
    try:
        candidates.append(ds[(0x0018, 0x9087)].value)  # type: ignore[index]
    except Exception:
        pass
    for value in candidates:
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def looks_like_pixel_series(ds: object) -> bool:
    """Return True for image-like DICOM headers read without PixelData."""
    try:
        rows = int(float(dicom_value(ds, "Rows", 0) or 0))
        columns = int(float(dicom_value(ds, "Columns", 0) or 0))
    except (TypeError, ValueError):
        return False
    return rows > 0 and columns > 0


def scan_dicom_series(dicom_dir: Path) -> list[DicomSeries]:
    pydicom = require_pydicom()
    grouped: dict[str, list[Path]] = {}
    samples: dict[str, object] = {}
    for file_path in sorted(dicom_dir.rglob("*.dcm"), key=lambda item: str(item)):
        try:
            ds = pydicom.dcmread(str(file_path), stop_before_pixels=True, force=True)
        except Exception:
            continue
        uid = str(dicom_value(ds, "SeriesInstanceUID", "unknown"))
        grouped.setdefault(uid, []).append(file_path)
        samples.setdefault(uid, ds)

    series_list: list[DicomSeries] = []
    for uid, files in grouped.items():
        sample = samples[uid]
        b_values: list[float] = []
        # Headers are read with stop_before_pixels=True for speed, so the
        # PixelData element is intentionally absent here. Rows/Columns are a
        # reliable lightweight signal that the series is image-like; actual
        # pixel decoding is validated later in selected_slice_datasets().
        has_pixel_data = looks_like_pixel_series(sample)
        for file_path in files[: min(len(files), 64)]:
            try:
                ds = pydicom.dcmread(str(file_path), stop_before_pixels=True, force=True)
                b_value = diffusion_b_value(ds)
                if b_value is not None:
                    b_values.append(b_value)
                has_pixel_data = has_pixel_data or looks_like_pixel_series(ds)
            except Exception:
                continue
        series_list.append(
            DicomSeries(
                uid=uid,
                files=files,
                series_number=stringify(dicom_value(sample, "SeriesNumber")),
                series_description=stringify(dicom_value(sample, "SeriesDescription")),
                protocol_name=stringify(dicom_value(sample, "ProtocolName")),
                sequence_name=stringify(dicom_value(sample, "SequenceName")),
                image_type=stringify(dicom_value(sample, "ImageType")),
                rows=stringify(dicom_value(sample, "Rows")),
                columns=stringify(dicom_value(sample, "Columns")),
                pixel_spacing=stringify(dicom_value(sample, "PixelSpacing")),
                slice_thickness=stringify(dicom_value(sample, "SliceThickness")),
                spacing_between_slices=stringify(dicom_value(sample, "SpacingBetweenSlices")),
                has_pixel_data=has_pixel_data,
                b_values=b_values,
            )
        )
    return sorted(series_list, key=lambda item: (-item.count, item.series_number, item.uid))


def dwi_score(series: DicomSeries) -> tuple[int, float, int]:
    text = series.text()
    score = 0
    if any(keyword in text for keyword in DWI_INCLUDE):
        score += 100
    if "dwi" in text:
        score += 40
    if "adc" in text:
        score -= 80
    if "t2" in text and "dwi" not in text:
        score -= 50
    if any(keyword in text for keyword in DWI_EXCLUDE):
        score -= 100
    max_b = max(series.b_values) if series.b_values else -1.0
    return score, max_b, series.count


def adc_score(series: DicomSeries, dwi_series_number: str = "") -> tuple[int, int]:
    text = series.text()
    score = 0
    if any(keyword in text for keyword in ADC_INCLUDE):
        score += 100
    if "adc" in text:
        score += 50
    if dwi_series_number and f"from {dwi_series_number}".lower() in text:
        score += 30
    if "dwi" in text and "adc" not in text:
        score -= 50
    if any(keyword in text for keyword in DWI_EXCLUDE):
        score -= 80
    return score, series.count


def select_dwi_series(series_list: Sequence[DicomSeries]) -> DicomSeries | None:
    candidates = [series for series in series_list if series.has_pixel_data]
    if not candidates:
        return None
    best = max(candidates, key=dwi_score)
    return best if dwi_score(best)[0] > 0 else None


def select_adc_series(series_list: Sequence[DicomSeries], dwi_series_number: str = "") -> DicomSeries | None:
    candidates = [series for series in series_list if series.has_pixel_data]
    if not candidates:
        return None
    best = max(candidates, key=lambda series: adc_score(series, dwi_series_number))
    return best if adc_score(best, dwi_series_number)[0] > 0 else None


def flair_score(series: DicomSeries) -> tuple[int, int]:
    """Score a conventional T2-FLAIR sequence while rejecting lookalikes."""
    text = series.text()
    score = 0
    if any(keyword in text for keyword in FLAIR_INCLUDE):
        score += 100
    if "t2" in text:
        score += 30
    if any(keyword in text for keyword in FLAIR_EXCLUDE):
        score -= 120
    return score, series.count


def select_flair_series(series_list: Sequence[DicomSeries]) -> DicomSeries | None:
    """Return the highest-scoring non-T1 FLAIR image series, when present."""
    candidates = [series for series in series_list if series.has_pixel_data]
    if not candidates:
        return None
    best = max(candidates, key=flair_score)
    return best if flair_score(best)[0] > 0 else None


def slice_position(ds: object, normal: np.ndarray | None, fallback: int) -> float:
    if normal is None or not hasattr(ds, "ImagePositionPatient"):
        return float(fallback)
    try:
        position = np.asarray([float(value) for value in ds.ImagePositionPatient], dtype=np.float64)
        return float(np.dot(position, normal))
    except Exception:
        return float(fallback)


def selected_slice_datasets(series: DicomSeries) -> tuple[list[object], str, float | None]:
    pydicom = require_pydicom()
    datasets: list[object] = []
    for file_path in series.files:
        try:
            ds = pydicom.dcmread(str(file_path), force=True)
            if "PixelData" in ds:
                datasets.append(ds)
        except Exception:
            continue
    if not datasets:
        raise RuntimeError(f"No pixel data in series {series.uid}")

    first = datasets[0]
    normal: np.ndarray | None = None
    if hasattr(first, "ImageOrientationPatient"):
        orientation = np.asarray([float(value) for value in first.ImageOrientationPatient], dtype=np.float64)
        if orientation.size == 6:
            normal = np.cross(orientation[:3], orientation[3:])
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm

    by_position: dict[int, list[object]] = {}
    for index, ds in enumerate(datasets):
        key = int(round(slice_position(ds, normal, index) / 1e-4))
        by_position.setdefault(key, []).append(ds)

    method = "all_slices"
    max_b_value: float | None = None
    selected: list[object] = []
    for group in by_position.values():
        if len(group) == 1:
            selected.append(group[0])
            b_value = diffusion_b_value(group[0])
            if b_value is not None:
                max_b_value = max(max_b_value or b_value, b_value)
            continue
        b_pairs = [(diffusion_b_value(ds), ds) for ds in group]
        known_b = [(value, ds) for value, ds in b_pairs if value is not None]
        if known_b:
            value, ds = max(known_b, key=lambda item: float(item[0]))
            selected.append(ds)
            max_b_value = max(max_b_value or float(value), float(value))
            method = "max_b_value_duplicate_position"
        else:
            selected.append(max(group, key=lambda ds: int(float(dicom_value(ds, "InstanceNumber", 0) or 0))))
            method = "duplicate_position_last_instance"

    selected.sort(key=lambda ds: slice_position(ds, normal, 0))
    return selected, method, max_b_value


def dicom_series_to_sitk(series: DicomSeries) -> tuple[sitk.Image, dict[str, object]]:
    datasets, method, max_b_value = selected_slice_datasets(series)
    arrays = []
    for ds in datasets:
        array = ds.pixel_array.astype(np.float32)
        slope = float(dicom_value(ds, "RescaleSlope", 1) or 1)
        intercept = float(dicom_value(ds, "RescaleIntercept", 0) or 0)
        arrays.append(array * slope + intercept)
    volume = np.stack(arrays, axis=0).astype(np.float32)
    image = sitk.GetImageFromArray(volume)

    first = datasets[0]
    pixel_spacing = [float(v) for v in getattr(first, "PixelSpacing", [1.0, 1.0])]
    row_spacing, col_spacing = pixel_spacing[0], pixel_spacing[1]
    positions = [
        np.asarray([float(value) for value in ds.ImagePositionPatient], dtype=np.float64)
        for ds in datasets
        if hasattr(ds, "ImagePositionPatient")
    ]
    if len(positions) >= 2:
        slice_spacing = float(np.median(np.linalg.norm(np.diff(np.stack(positions), axis=0), axis=1)))
    else:
        slice_spacing = float(dicom_value(first, "SpacingBetweenSlices", dicom_value(first, "SliceThickness", 1.0)) or 1.0)
    image.SetSpacing((col_spacing, row_spacing, slice_spacing))

    if hasattr(first, "ImagePositionPatient"):
        image.SetOrigin(tuple(float(value) for value in first.ImagePositionPatient))
    if hasattr(first, "ImageOrientationPatient"):
        orientation = np.asarray([float(value) for value in first.ImageOrientationPatient], dtype=np.float64)
        if orientation.size == 6:
            row_direction = orientation[:3]
            col_direction = orientation[3:]
            normal = np.cross(row_direction, col_direction)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
                direction = np.column_stack([row_direction, col_direction, normal])
                image.SetDirection(tuple(direction.reshape(-1)))

    meta = {
        "selected_count": len(datasets),
        "selection_method": method,
        "b_value": max_b_value,
        "raw_count": series.count,
    }
    return image, meta


def ascii_nifti_name(path: Path, stem: str = "image") -> str:
    name = path.name.lower()
    if name.endswith(".nii.gz"):
        return f"{stem}.nii.gz"
    if name.endswith(".nii"):
        return f"{stem}.nii"
    return f"{stem}.nii.gz"


def path_has_non_ascii(path: Path) -> bool:
    try:
        str(path).encode("ascii")
    except UnicodeEncodeError:
        return True
    return False


def read_nifti_from_ascii_temp(path: Path) -> sitk.Image:
    with tempfile.TemporaryDirectory(prefix="ci1_nifti_") as tmp_dir:
        tmp_path = Path(tmp_dir) / ascii_nifti_name(path)
        shutil.copy2(path, tmp_path)
        return sitk.ReadImage(str(tmp_path))


def read_nifti(path: Path) -> sitk.Image:
    if path_has_non_ascii(path):
        return read_nifti_from_ascii_temp(path)
    try:
        return sitk.ReadImage(str(path))
    except RuntimeError:
        return read_nifti_from_ascii_temp(path)


def find_label_path(patient_dir: Path, patient: str, timepoint: str) -> Path | None:
    candidates = [
        path
        for path in sorted(patient_dir.rglob("*"), key=lambda item: str(item))
        if path.is_file() and is_nifti_path(path)
    ]
    scored: list[tuple[int, int, str, Path]] = []
    for path in candidates:
        text = str(path).lower()
        name_text = path.name.lower()
        if any(keyword in text for keyword in LABEL_EXCLUDE):
            continue
        if timepoint.lower() not in text:
            continue
        if "dwi" not in text:
            continue
        score = 0
        if patient.lower() in text:
            score += 20
        if timepoint.lower() in name_text:
            score += 30
        if "dwi" in name_text:
            score += 30
        if path.name.lower().endswith(".nii.gz"):
            score += 5
        scored.append((score, -len(path.parts), str(path), path))
    if not scored:
        return None
    return max(scored)[3]


def geometry_matches(left: sitk.Image, right: sitk.Image, tol: float = 1e-4) -> bool:
    if left.GetSize() != right.GetSize():
        return False
    for getter in (sitk.Image.GetSpacing, sitk.Image.GetOrigin, sitk.Image.GetDirection):
        lv = tuple(float(v) for v in getter(left))
        rv = tuple(float(v) for v in getter(right))
        if len(lv) != len(rv) or any(abs(a - b) > tol for a, b in zip(lv, rv)):
            return False
    return True


def resample_to_reference(image: sitk.Image, reference: sitk.Image, interpolator: int, default_value: float = 0.0) -> sitk.Image:
    if geometry_matches(image, reference):
        return image
    return sitk.Resample(image, reference, sitk.Transform(), interpolator, default_value, image.GetPixelID())


def binary_label(image: sitk.Image) -> sitk.Image:
    array = (sitk.GetArrayFromImage(image) != 0).astype(np.uint8)
    output = sitk.GetImageFromArray(array)
    output.CopyInformation(image)
    return output


def write_image(image: sitk.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))


def image_shape(image: sitk.Image) -> list[int]:
    return list(reversed([int(v) for v in image.GetSize()]))


def image_spacing(image: sitk.Image) -> list[float]:
    return [float(v) for v in image.GetSpacing()]


def label_voxels(image: sitk.Image) -> int:
    return int(np.count_nonzero(sitk.GetArrayFromImage(image)))


def affine_swap_warning(image: sitk.Image, label: sitk.Image) -> str:
    if image.GetSize() != label.GetSize():
        return ""
    dwi = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    lab = np.asarray(label.GetDirection(), dtype=np.float64).reshape(3, 3)
    swapped = lab.copy()
    swapped[:, [0, 1]] = swapped[:, [1, 0]]
    if np.allclose(dwi, swapped, atol=1e-3) and not np.allclose(dwi, lab, atol=1e-3):
        return "label_direction_xy_swap_suspected"
    return ""


def transpose_xy_label(label: sitk.Image, reference: sitk.Image) -> sitk.Image:
    array = sitk.GetArrayFromImage(label)
    transposed = np.transpose(array, (0, 2, 1))
    output = sitk.GetImageFromArray(transposed.astype(np.uint8))
    output.CopyInformation(reference)
    return output
