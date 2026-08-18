"""Audit and build DWI+ADC(/FLAIR) nnU-Net datasets from a DWI manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import SimpleITK as sitk


SUPPORTED_MODALITIES = (("DWI", "ADC"), ("DWI", "ADC", "FLAIR"))


def normalize_modalities(modalities: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(modality.upper() for modality in modalities)
    if normalized not in SUPPORTED_MODALITIES:
        raise ValueError(
            "modalities must be exactly DWI ADC or DWI ADC FLAIR, in that order"
        )
    return normalized


def build_dataset_json(modalities: Sequence[str], num_training: int) -> dict[str, object]:
    normalized = normalize_modalities(modalities)
    return {
        "channel_names": {str(index): modality for index, modality in enumerate(normalized)},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }


def _series_description(path: str) -> str:
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    reader.ReadImageInformation()
    return reader.GetMetaData("0008|103e").strip() if reader.HasMetaDataKey("0008|103e") else ""


def _matches_modality(description: str, modality: str) -> bool:
    text = description.lower()
    if modality == "ADC":
        return "adc" in text
    if modality == "FLAIR":
        return "flair" in text and "t1" not in text
    raise ValueError(f"DICOM reading is only required for non-DWI modality: {modality}")


def read_dicom_modality(dicom_dir: Path, modality: str) -> tuple[sitk.Image, str]:
    matches: list[tuple[int, list[str], str]] = []
    for series_id in sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_dir)) or []:
        filenames = list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_dir), series_id))
        if filenames:
            description = _series_description(filenames[0])
            if _matches_modality(description, modality):
                matches.append((len(filenames), filenames, description))
    if not matches:
        raise RuntimeError(f"no {modality} DICOM series in {dicom_dir}")
    _, filenames, description = max(matches, key=lambda item: item[0])
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(filenames)
    return reader.Execute(), description


def resample_to_reference(image: sitk.Image, reference: sitk.Image) -> sitk.Image:
    float_image = sitk.Cast(image, sitk.sitkFloat32)
    return sitk.Resample(
        float_image,
        reference,
        sitk.Transform(),
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )


def geometry_matches(left: sitk.Image, right: sitk.Image, tolerance: float = 1e-5) -> bool:
    return (
        left.GetSize() == right.GetSize()
        and all(abs(a - b) <= tolerance for a, b in zip(left.GetSpacing(), right.GetSpacing()))
        and all(abs(a - b) <= tolerance for a, b in zip(left.GetOrigin(), right.GetOrigin()))
        and all(abs(a - b) <= tolerance for a, b in zip(left.GetDirection(), right.GetDirection()))
    )


@dataclass(frozen=True)
class AuditRow:
    case_id: str
    dwi_path: str
    label_path: str
    source_dicom_dir: str
    modality: str
    series_description: str
    status: str
    detail: str


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    case_ids = [row.get("case_id", "") for row in rows]
    if len(rows) != 95 or len(set(case_ids)) != 95 or any(not case_id for case_id in case_ids):
        raise ValueError("manifest must contain exactly 95 unique nonempty case_id values")
    return rows


def audit_manifest(manifest_path: Path, modalities: Sequence[str]) -> list[AuditRow]:
    normalized = normalize_modalities(modalities)
    results: list[AuditRow] = []
    for row in read_manifest(manifest_path):
        dwi_path = Path(row["output_image"])
        label_path = Path(row["output_label"])
        dicom_dir = Path(row["source_dicom_dir"])
        if not dwi_path.is_file() or not label_path.is_file() or not dicom_dir.is_dir():
            results.append(AuditRow(row["case_id"], str(dwi_path), str(label_path), str(dicom_dir), "SOURCE", "", "failed", "missing source path"))
            continue
        reference = sitk.ReadImage(str(dwi_path))
        for modality in normalized[1:]:
            try:
                image, description = read_dicom_modality(dicom_dir, modality)
                resampled = resample_to_reference(image, reference)
                status = "passed" if geometry_matches(resampled, reference) else "failed"
                detail = "" if status == "passed" else "resampled geometry differs from DWI"
            except RuntimeError as exc:
                description, status, detail = "", "failed", str(exc)
            results.append(AuditRow(row["case_id"], str(dwi_path), str(label_path), str(dicom_dir), modality, description, status, detail))
    return results


def write_audit(rows: Iterable[AuditRow], output_path: Path) -> None:
    values = [asdict(row) for row in rows]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(AuditRow.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset(manifest_path: Path, output_root: Path, modalities: Sequence[str], audit_rows: Sequence[AuditRow]) -> None:
    normalized = normalize_modalities(modalities)
    if output_root.exists():
        raise FileExistsError(f"output root must not already exist: {output_root}")
    if any(row.status != "passed" for row in audit_rows):
        raise RuntimeError("refusing build because multimodal audit did not pass")
    by_case_modality = {(row.case_id, row.modality): row for row in audit_rows}
    manifest_rows = read_manifest(manifest_path)
    for row in manifest_rows:
        if not Path(row["output_image"]).is_file() or not Path(row["output_label"]).is_file():
            raise RuntimeError(f"missing audited DWI or label source for {row['case_id']}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}_building_", dir=output_root.parent)
    )
    try:
        images = temporary_root / "imagesTr"
        labels = temporary_root / "labelsTr"
        images.mkdir()
        labels.mkdir()
        provenance: list[dict[str, str]] = []
        for row in manifest_rows:
            case_id = row["case_id"]
            dwi_path, label_path = Path(row["output_image"]), Path(row["output_label"])
            dwi_target = images / f"{case_id}_0000.nii.gz"
            label_target = labels / f"{case_id}.nii.gz"
            shutil.copy2(dwi_path, dwi_target)
            shutil.copy2(label_path, label_target)
            reference = sitk.ReadImage(str(dwi_path))
            for index, modality in enumerate(normalized[1:], start=1):
                audit = by_case_modality[(case_id, modality)]
                image, _ = read_dicom_modality(Path(audit.source_dicom_dir), modality)
                resampled = resample_to_reference(image, reference)
                if not geometry_matches(resampled, reference):
                    raise RuntimeError(f"geometry changed after approved audit: {case_id} {modality}")
                target = images / f"{case_id}_{index:04d}.nii.gz"
                sitk.WriteImage(resampled, str(target))
                provenance.append({"case_id": case_id, "modality": modality, "source": audit.source_dicom_dir, "output": str(target), "sha256": _sha256(target)})
            provenance.extend([
                {"case_id": case_id, "modality": "DWI", "source": str(dwi_path), "output": str(dwi_target), "sha256": _sha256(dwi_target)},
                {"case_id": case_id, "modality": "LABEL", "source": str(label_path), "output": str(label_target), "sha256": _sha256(label_target)},
            ])
        (temporary_root / "dataset.json").write_text(json.dumps(build_dataset_json(normalized, 95), indent=2) + "\n", encoding="utf-8")
        (temporary_root / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary_root.replace(output_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--modalities", nargs="+", required=True)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    modalities = normalize_modalities(args.modalities)
    rows = audit_manifest(args.manifest, modalities)
    audit_path = args.output_root.parent / f"{args.output_root.name}_audit.tsv"
    write_audit(rows, audit_path)
    failed = sum(row.status != "passed" for row in rows)
    print(json.dumps({"audit": str(audit_path), "rows": len(rows), "failed": failed, "modalities": modalities}))
    if args.build:
        build_dataset(args.manifest, args.output_root, modalities, rows)


if __name__ == "__main__":
    main()
