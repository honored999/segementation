"""Initial contracts for the read-only CI-1 raw-data audit."""

import argparse
import csv
import json
import math
import re
import shutil
import sys
import tempfile
from statistics import median
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping

import numpy as np
from scipy import ndimage

try:
    from . import prepare_ci1_dwi_dataset
except ImportError:  # Support direct module use from non_teacher_student_files.
    import prepare_ci1_dwi_dataset


DEFAULT_TARGET_SPACING_XY_MM = (0.4892368018627167, 0.4892368018627167)
SPACING_TOLERANCE_MM = 1e-5
ORIGIN_TOLERANCE_MM = 1e-4
DIRECTION_TOLERANCE = 1e-6
IPP_UNIFORM_ABS_TOLERANCE_MM = 1e-3
IPP_UNIFORM_REL_TOLERANCE = 1e-3
RESULT_FILES = (
    "dicom_series_statistics.csv",
    "case_statistics.csv",
    "dataset_summary.json",
)


METADATA_FIELDS = (
    "SeriesInstanceUID",
    "SeriesDescription",
    "ProtocolName",
    "SequenceName",
    "ImageType",
    "Modality",
    "Manufacturer",
    "ManufacturerModelName",
    "Rows",
    "Columns",
    "PixelSpacing",
    "SliceThickness",
    "SpacingBetweenSlices",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "RescaleSlope",
    "RescaleIntercept",
    "BitsAllocated",
    "BitsStored",
    "PixelRepresentation",
    "b-value",
)

_METADATA_TAGS = {
    "SeriesInstanceUID": "0020|000e",
    "SeriesDescription": "0008|103e",
    "ProtocolName": "0018|1030",
    "SequenceName": "0018|0024",
    "ImageType": "0008|0008",
    "Modality": "0008|0060",
    "Manufacturer": "0008|0070",
    "ManufacturerModelName": "0008|1090",
    "Rows": "0028|0010",
    "Columns": "0028|0011",
    "PixelSpacing": "0028|0030",
    "SliceThickness": "0018|0050",
    "SpacingBetweenSlices": "0018|0088",
    "ImagePositionPatient": "0020|0032",
    "ImageOrientationPatient": "0020|0037",
    "RescaleSlope": "0028|1053",
    "RescaleIntercept": "0028|1052",
    "BitsAllocated": "0028|0100",
    "BitsStored": "0028|0101",
    "PixelRepresentation": "0028|0103",
    "b-value": "0018|9087",
}

_CLASSIFICATION_FIELDS = (
    ("SeriesDescription", "dicom:SeriesDescription"),
    ("ProtocolName", "dicom:ProtocolName"),
    ("SequenceName", "dicom:SequenceName"),
    ("ImageType", "dicom:ImageType"),
)


CASE_COLUMNS = (
    "case_id,patient_id,timepoint,pairing_status,pairing_source,"
    "build_index_match_status,dwi_source_path,mask_path,dwi_series_uid,"
    "dwi_series_path,audit_status,read_status,error_code,error_message,"
    "geometry_status,geometry_mismatch_fields,dwi_mask_comparison_status,"
    "dwi_size_xyz,mask_size_xyz,dwi_spacing_xyz_mm,mask_spacing_xyz_mm,"
    "dwi_origin_xyz_mm,mask_origin_xyz_mm,dwi_direction_3x3,"
    "mask_direction_3x3,mask_unique_values,empty_mask,mask_voxel_count,"
    "mask_volume_mm3,mask_volume_ml,component_connectivity,"
    "component_count_26,largest_component_voxels_26,"
    "largest_component_volume_mm3_26,smallest_component_voxels_26,"
    "smallest_component_volume_mm3_26,foreground_slice_count,"
    "foreground_slice_ratio,bbox_voxel_size_xyz,bbox_physical_size_mm_xyz,"
    "centroid_voxel_xyz,centroid_physical_xyz_mm,lesion_index_min_xyz,"
    "lesion_index_max_xyz,lesion_physical_min_xyz_mm,"
    "lesion_physical_max_xyz_mm,dwi_finite_voxel_count,"
    "dwi_nonfinite_voxel_count,dwi_all_min,dwi_all_max,dwi_all_mean,"
    "dwi_all_std,dwi_all_median,dwi_all_p05,dwi_all_p95,"
    "lesion_intensity_status,lesion_finite_voxel_count,lesion_min,"
    "lesion_max,lesion_mean,lesion_std,lesion_median,lesion_p5,"
    "lesion_p25,lesion_p75,lesion_p95,xy_target_spacing_xy_mm,"
    "xy_simulated_size_xyz,xy_simulated_spacing_xyz_mm,xy_z_preserved"
).split(",")


SERIES_COLUMNS = (
    "internal_case_patient_id,series_instance_uid,gdcm_series_id,"
    "series_directory,relative_file_count,read_status,uid_status,"
    "metadata_status,error_code,error_message,series_description,"
    "protocol_name,sequence_name,image_type,dicom_modality,manufacturer,"
    "manufacturer_model_name,rows,columns,pixel_spacing,slice_thickness,"
    "spacing_between_slices,image_position_patient,image_orientation_patient,"
    "rescale_slope,rescale_intercept,bits_allocated,bits_stored,"
    "pixel_representation,b_value,metadata_field_status,modality_class,"
    "classification_source,classification_confidence,classification_evidence,"
    "candidate_patient_id,candidate_timepoint,candidate_id_source,"
    "pairing_status,pairing_evidence,confirmed_dwi_case_id,size_xyz,"
    "shape_zyx,spacing_xyz_mm,origin_xyz_mm,direction_3x3,slice_count,"
    "fov_xyz_mm,inplane_spacing_xy_mm,z_spacing_metadata_mm,"
    "z_spacing_metadata_source,z_spacing_ipp_min_mm,z_spacing_ipp_median_mm,"
    "z_spacing_ipp_max_mm,z_spacing_ipp_uniform,"
    "z_spacing_metadata_ipp_discrepancy_mm,pixel_type,voxel_count,"
    "finite_voxel_count,nonfinite_voxel_count,intensity_value_semantics,"
    "raw_value_semantics,reconstructed_value_semantics,rescale_applied,"
    "intensity_metadata_source,all_min,all_p1,all_p5,all_p25,all_median,"
    "all_p75,all_p95,all_p99,all_max,all_mean,all_std,nonzero_min,"
    "nonzero_p1,nonzero_p5,nonzero_p25,nonzero_median,nonzero_p75,"
    "nonzero_p95,nonzero_p99,nonzero_max,nonzero_mean,nonzero_std"
).split(",")


@dataclass(frozen=True)
class SeriesRecord:
    """Immutable result for one independently enumerated GDCM series ID."""

    series_directory: str
    gdcm_series_id: str
    series_instance_uid: str
    relative_file_paths: tuple[str, ...]
    file_count: int
    read_status: str
    uid_status: str
    error_code: str = ""
    error_message: str = ""
    modality_class: str = "unknown"


@dataclass(frozen=True)
class LinkResult:
    """Result of linking one build_index source directory to one series."""

    link_status: str
    series_instance_uid: str = ""
    series_path: str = ""
    record: SeriesRecord | None = None
    error_message: str = ""


@dataclass(frozen=True)
class CaseRecord:
    """Minimal confirmed DWI-to-mask audit record for Task 5."""

    case_id: str
    patient_id: str
    timepoint: str
    pairing_status: str
    pairing_source: str
    build_index_match_status: str
    dwi_source_path: str
    mask_path: str
    link_status: str
    metadata_modality_consistency: str
    dwi_series_uid: str
    dwi_series_path: str
    audit_status: str
    read_status: str
    error_code: str
    error_message: str
    geometry_status: str
    geometry_mismatch_fields: tuple[str, ...]
    dwi_mask_comparison_status: str
    dwi_size_xyz: tuple[int, ...] | None = None
    mask_size_xyz: tuple[int, ...] | None = None
    dwi_spacing_xyz_mm: tuple[float, ...] | None = None
    mask_spacing_xyz_mm: tuple[float, ...] | None = None
    dwi_origin_xyz_mm: tuple[float, ...] | None = None
    mask_origin_xyz_mm: tuple[float, ...] | None = None
    dwi_direction_3x3: tuple[float, ...] | None = None
    mask_direction_3x3: tuple[float, ...] | None = None
    derived_metrics: object | None = None


_INTENSITY_STAT_NAMES = (
    "min",
    "p1",
    "p5",
    "p25",
    "median",
    "p75",
    "p95",
    "p99",
    "max",
    "mean",
    "std",
)

_MASK_METRIC_NAMES = (
    "mask_unique_values",
    "empty_mask",
    "mask_voxel_count",
    "mask_volume_mm3",
    "mask_volume_ml",
    "component_connectivity",
    "component_count_26",
    "largest_component_voxels_26",
    "largest_component_volume_mm3_26",
    "smallest_component_voxels_26",
    "smallest_component_volume_mm3_26",
    "foreground_slice_count",
    "foreground_slice_ratio",
    "bbox_voxel_size_xyz",
    "bbox_physical_size_mm_xyz",
    "centroid_voxel_xyz",
    "centroid_physical_xyz_mm",
    "lesion_index_min_xyz",
    "lesion_index_max_xyz",
    "lesion_physical_min_xyz_mm",
    "lesion_physical_max_xyz_mm",
    "lesion_intensity_status",
    "lesion_finite_voxel_count",
    "lesion_min",
    "lesion_max",
    "lesion_mean",
    "lesion_std",
    "lesion_median",
    "lesion_p5",
    "lesion_p25",
    "lesion_p75",
    "lesion_p95",
)


def _blank_mask_metrics(lesion_intensity_status):
    """Return unavailable case metrics without inventing an aligned mask."""
    metrics = {name: None for name in _MASK_METRIC_NAMES}
    metrics["lesion_intensity_status"] = lesion_intensity_status
    return metrics


def summarize_intensity(values):
    """Summarize a readable image array without changing its values or scale."""
    array = np.asarray(values)
    finite = np.isfinite(array)
    result = {
        "finite_voxel_count": int(np.count_nonzero(finite)),
        "nonfinite_voxel_count": int(array.size - np.count_nonzero(finite)),
    }
    for prefix in ("all", "nonzero"):
        for name in _INTENSITY_STAT_NAMES:
            result[f"{prefix}_{name}"] = None
    if result["nonfinite_voxel_count"]:
        return result

    samples = {
        "all": array.reshape(-1),
        "nonzero": array[array != 0],
    }
    percentiles = {
        "p1": 1,
        "p5": 5,
        "p25": 25,
        "median": 50,
        "p75": 75,
        "p95": 95,
        "p99": 99,
    }
    for prefix, sample in samples.items():
        if not sample.size:
            continue
        result[f"{prefix}_min"] = float(np.min(sample))
        result[f"{prefix}_max"] = float(np.max(sample))
        result[f"{prefix}_mean"] = float(np.mean(sample))
        result[f"{prefix}_std"] = float(np.std(sample))
        for name, percentile in percentiles.items():
            result[f"{prefix}_{name}"] = float(np.percentile(sample, percentile))
    return result


def _lesion_intensity_metrics(dwi_array, foreground):
    """Return finite foreground-DWI summaries without changing either array."""
    result = {
        "lesion_intensity_status": "available",
        "lesion_finite_voxel_count": None,
        "lesion_min": None,
        "lesion_max": None,
        "lesion_mean": None,
        "lesion_std": None,
        "lesion_median": None,
        "lesion_p5": None,
        "lesion_p25": None,
        "lesion_p75": None,
        "lesion_p95": None,
    }
    values = np.asarray(dwi_array)[foreground]
    finite_values = values[np.isfinite(values)]
    result["lesion_finite_voxel_count"] = int(finite_values.size)
    if not finite_values.size:
        result["lesion_intensity_status"] = "no_finite_lesion_voxels"
        return result
    result.update(
        {
            "lesion_min": float(np.min(finite_values)),
            "lesion_max": float(np.max(finite_values)),
            "lesion_mean": float(np.mean(finite_values)),
            "lesion_std": float(np.std(finite_values)),
            "lesion_median": float(np.percentile(finite_values, 50)),
            "lesion_p5": float(np.percentile(finite_values, 5)),
            "lesion_p25": float(np.percentile(finite_values, 25)),
            "lesion_p75": float(np.percentile(finite_values, 75)),
            "lesion_p95": float(np.percentile(finite_values, 95)),
        }
    )
    return result


def mask_metrics(mask_array, mask_image, dwi_array=None):
    """Report descriptive native-mask metrics using mask > 0 and no repair."""
    array = np.asarray(mask_array)
    foreground = array > 0
    spacing = tuple(float(value) for value in mask_image.GetSpacing())
    voxel_volume_mm3 = math.prod(spacing)
    voxel_count = int(np.count_nonzero(foreground))
    result = {
        "mask_unique_values": np.unique(array).tolist(),
        "empty_mask": voxel_count == 0,
        "mask_voxel_count": voxel_count,
        "mask_volume_mm3": float(voxel_count * voxel_volume_mm3),
        "mask_volume_ml": float(voxel_count * voxel_volume_mm3 / 1000.0),
        "component_connectivity": 26,
        "component_count_26": 0,
        "largest_component_voxels_26": None,
        "largest_component_volume_mm3_26": None,
        "smallest_component_voxels_26": None,
        "smallest_component_volume_mm3_26": None,
        "foreground_slice_count": int(np.count_nonzero(np.any(foreground, axis=(1, 2)))),
        "foreground_slice_ratio": float(np.count_nonzero(np.any(foreground, axis=(1, 2))) / array.shape[0]),
        "bbox_voxel_size_xyz": None,
        "bbox_physical_size_mm_xyz": None,
        "centroid_voxel_xyz": None,
        "centroid_physical_xyz_mm": None,
        "lesion_index_min_xyz": None,
        "lesion_index_max_xyz": None,
        "lesion_physical_min_xyz_mm": None,
        "lesion_physical_max_xyz_mm": None,
    }
    if not voxel_count:
        result.update(_lesion_intensity_metrics(np.asarray([]), np.asarray([], dtype=bool)))
        result["lesion_intensity_status"] = "empty_mask"
        result["lesion_finite_voxel_count"] = None
        return result

    labels, component_count = ndimage.label(foreground, structure=np.ones((3, 3, 3), dtype=np.uint8))
    component_sizes = np.bincount(labels.ravel())[1:]
    result.update(
        {
            "component_count_26": int(component_count),
            "largest_component_voxels_26": int(component_sizes.max()),
            "largest_component_volume_mm3_26": float(component_sizes.max() * voxel_volume_mm3),
            "smallest_component_voxels_26": int(component_sizes.min()),
            "smallest_component_volume_mm3_26": float(component_sizes.min() * voxel_volume_mm3),
        }
    )
    indices_zyx = np.argwhere(foreground)
    minimum_xyz = indices_zyx.min(axis=0)[::-1].astype(int).tolist()
    maximum_xyz = indices_zyx.max(axis=0)[::-1].astype(int).tolist()
    size_xyz = [maximum_xyz[i] - minimum_xyz[i] + 1 for i in range(3)]
    centroid_xyz = indices_zyx.mean(axis=0)[::-1].tolist()
    corners = [
        index_xyz_to_physical((x, y, z), mask_image.GetOrigin(), spacing, mask_image.GetDirection())
        for x in (minimum_xyz[0], maximum_xyz[0])
        for y in (minimum_xyz[1], maximum_xyz[1])
        for z in (minimum_xyz[2], maximum_xyz[2])
    ]
    result.update(
        {
            "bbox_voxel_size_xyz": size_xyz,
            "bbox_physical_size_mm_xyz": [float(size_xyz[i] * spacing[i]) for i in range(3)],
            "centroid_voxel_xyz": centroid_xyz,
            "centroid_physical_xyz_mm": index_xyz_to_physical(
                centroid_xyz, mask_image.GetOrigin(), spacing, mask_image.GetDirection()
            ),
            "lesion_index_min_xyz": minimum_xyz,
            "lesion_index_max_xyz": maximum_xyz,
            "lesion_physical_min_xyz_mm": [float(min(point[i] for point in corners)) for i in range(3)],
            "lesion_physical_max_xyz_mm": [float(max(point[i] for point in corners)) for i in range(3)],
        }
    )
    if dwi_array is None:
        import SimpleITK as sitk

        dwi_array = sitk.GetArrayFromImage(mask_image)
    result.update(_lesion_intensity_metrics(dwi_array, foreground))
    return result


def _dwi_metrics(dwi_array):
    """Namespace whole-volume DWI metrics and preserve reader semantics."""
    summary = summarize_intensity(dwi_array)
    result = {f"dwi_{name}": value for name, value in summary.items()}
    result.update(
        {
            "intensity_value_semantics": "reader_output_unverified_rescale",
            "rescale_applied": None,
            "intensity_metadata_source": "public DICOM rescale tags if available",
        }
    )
    return result


def _resolved_is_within(path: Path, directory: Path) -> bool:
    path = path.resolve()
    directory = directory.resolve()
    return path == directory or directory in path.parents


def _source_file_collection(source_dir: Path) -> tuple[Path, ...]:
    """Return the resolved regular files in a source directory."""
    try:
        files = [path.resolve() for path in source_dir.rglob("*") if path.is_file()]
    except OSError as exc:
        raise OSError(f"cannot enumerate source directory: {_concise_error(exc)}") from exc
    return tuple(sorted(files, key=lambda path: path.as_posix()))


def _record_full_paths(record: SeriesRecord) -> tuple[Path, ...] | None:
    """Resolve a record's actual member paths without guessing from names."""
    series_dir = Path(record.series_directory).resolve()
    paths = []
    for relative_path in record.relative_file_paths:
        member = Path(relative_path)
        full_path = member.resolve() if member.is_absolute() else (series_dir / member).resolve()
        if not _resolved_is_within(full_path, series_dir) or not full_path.is_file():
            return None
        paths.append(full_path)
    return tuple(sorted(paths, key=lambda path: path.as_posix()))


def _valid_series_uid(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", str(value).strip()))


def link_build_index_source(dicom_dir, series_records) -> LinkResult:
    """Link a build_index directory to exactly one DWI series record."""
    source_dir = Path(dicom_dir)
    try:
        source_dir = source_dir.resolve(strict=True)
        if not source_dir.is_dir():
            raise NotADirectoryError(str(source_dir))
    except (OSError, RuntimeError) as exc:
        return LinkResult(
            link_status="source_unreadable",
            error_message=_concise_error(exc),
        )

    candidates = []
    for record in series_records:
        record_dir = Path(record.series_directory)
        try:
            if record_dir.resolve(strict=True) != source_dir:
                continue
        except (OSError, RuntimeError):
            continue
        modality_class = str(getattr(record, "modality_class", "unknown") or "unknown").lower()
        if modality_class != "dwi":
            continue
        candidates.append(record)

    if not candidates:
        return LinkResult(link_status="no_matching_series")
    if len(candidates) != 1:
        return LinkResult(
            link_status="multiple_matching_series",
            error_message=f"matching series count: {len(candidates)}",
        )

    record = candidates[0]
    if not _valid_series_uid(record.series_instance_uid):
        return LinkResult(link_status="uid_invalid", error_message="invalid SeriesInstanceUID")
    if record.uid_status != "consistent" or record.read_status != "readable":
        return LinkResult(
            link_status="series_unreadable",
            series_instance_uid=str(record.series_instance_uid),
            series_path=str(source_dir),
            record=record,
            error_message=record.error_message or "series record is not readable and UID-consistent",
        )
    return LinkResult(
        link_status="linked",
        series_instance_uid=str(record.series_instance_uid),
        series_path=str(source_dir),
        record=record,
    )


def compare_geometry(image, mask) -> tuple[str, list[str]]:
    """Compare native SimpleITK geometry using the centralized tolerances."""
    def exceeds_tolerance(left, right, tolerance):
        delta = abs(float(left) - float(right))
        return delta > tolerance and not math.isclose(
            delta, tolerance, rel_tol=1e-9, abs_tol=1e-12
        )

    mismatch_fields = []
    if tuple(image.GetSize()) != tuple(mask.GetSize()):
        mismatch_fields.append("size")
    if any(
        exceeds_tolerance(left, right, SPACING_TOLERANCE_MM)
        for left, right in zip(image.GetSpacing(), mask.GetSpacing())
    ):
        mismatch_fields.append("spacing")
    if any(
        exceeds_tolerance(left, right, ORIGIN_TOLERANCE_MM)
        for left, right in zip(image.GetOrigin(), mask.GetOrigin())
    ):
        mismatch_fields.append("origin")
    if any(
        exceeds_tolerance(left, right, DIRECTION_TOLERANCE)
        for left, right in zip(image.GetDirection(), mask.GetDirection())
    ):
        mismatch_fields.append("direction")
    return ("match" if not mismatch_fields else "mismatch", mismatch_fields)


def _native_geometry(image) -> dict[str, tuple]:
    """Capture only native SimpleITK metadata; never materialize a pixel array."""
    return {
        "size_xyz": tuple(int(value) for value in image.GetSize()),
        "spacing_xyz_mm": tuple(float(value) for value in image.GetSpacing()),
        "origin_xyz_mm": tuple(float(value) for value in image.GetOrigin()),
        "direction_3x3": tuple(float(value) for value in image.GetDirection()),
    }


def _read_dwi_series(record: SeriesRecord):
    import SimpleITK as sitk

    full_paths = _record_full_paths(record)
    if not full_paths:
        raise OSError("linked series has no readable member paths")
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(path) for path in full_paths])
    return reader.Execute()


def _read_raw_mask(mask_path: Path):
    import SimpleITK as sitk
    import tempfile

    try:
        return sitk.ReadImage(str(mask_path))
    except Exception as direct_error:  # noqa: BLE001 - preserve direct reader failure
        name = mask_path.name.lower()
        if not mask_path.is_file() or not (name.endswith(".nii") or name.endswith(".nii.gz")):
            raise

        temp_base = Path(tempfile.gettempdir()).resolve()
        if not str(temp_base).isascii():
            raise direct_error

        try:
            temp_dir = tempfile.TemporaryDirectory(dir=str(temp_base))
        except OSError:
            raise direct_error
        with temp_dir:
            suffix = mask_path.name[-7:] if name.endswith(".nii.gz") else mask_path.name[-4:]
            fallback_path = Path(temp_dir.name) / ("mask" + suffix)
            fallback_path.write_bytes(mask_path.read_bytes())
            return sitk.ReadImage(str(fallback_path))


def _metadata_modality_consistency(record: SeriesRecord) -> str:
    modality = str(getattr(record, "modality_class", "unknown") or "unknown").lower()
    if modality in {"adc", "flair"}:
        return "conflict"
    if modality == "unknown":
        return "unknown"
    return "match"


def _new_case(row: Mapping, link: LinkResult) -> CaseRecord:
    patient = str(row.get("patient", ""))
    timepoint = str(row.get("timepoint", ""))
    record = link.record
    return CaseRecord(
        case_id=f"{patient}_{timepoint}",
        patient_id=patient,
        timepoint=timepoint,
        pairing_status="confirmed",
        pairing_source="prepare_ci1_dwi_dataset.build_index",
        build_index_match_status=str(row.get("match_status", "")),
        dwi_source_path=str(row.get("dicom_dir", "")),
        mask_path=str(row.get("segmentation_path", "")),
        link_status=link.link_status,
        metadata_modality_consistency=_metadata_modality_consistency(record) if record else "unknown",
        dwi_series_uid=link.series_instance_uid,
        dwi_series_path=link.series_path,
        audit_status="failed",
        read_status=link.link_status,
        error_code=link.link_status if link.link_status != "linked" else "",
        error_message=link.error_message,
        geometry_status="not_checked",
        geometry_mismatch_fields=(),
        dwi_mask_comparison_status="failed",
    )


def audit_case(row: Mapping, series_records, read_dwi=None, read_mask=None) -> CaseRecord | None:
    """Audit one matched DWI/mask row without resampling or pixel mutation."""
    if str(row.get("match_status", "")) != "matched":
        return None

    link = link_build_index_source(row.get("dicom_dir", ""), series_records)
    case = _new_case(row, link)
    if link.link_status != "linked":
        return case

    try:
        dwi = (read_dwi or _read_dwi_series)(link.record)
    except Exception as exc:  # noqa: BLE001 - preserve case-level read status
        return dataclass_replace(
            case,
            read_status="dwi_read_failed",
            error_code="dwi_read_failed",
            error_message=_concise_error(exc),
        )
    if dwi.GetDimension() != 3:
        return dataclass_replace(
            case,
            read_status="dwi_not_3d",
            error_code="dwi_not_3d",
            error_message="linked DWI image is not 3D",
        )

    dwi_geometry = _native_geometry(dwi)
    base = dataclass_replace(
        case,
        dwi_size_xyz=dwi_geometry["size_xyz"],
        dwi_spacing_xyz_mm=dwi_geometry["spacing_xyz_mm"],
        dwi_origin_xyz_mm=dwi_geometry["origin_xyz_mm"],
        dwi_direction_3x3=dwi_geometry["direction_3x3"],
        read_status="dwi_readable",
    )
    import SimpleITK as sitk

    dwi_array = sitk.GetArrayFromImage(dwi)
    metrics = _dwi_metrics(dwi_array)
    base = dataclass_replace(base, derived_metrics=metrics)
    mask_path = Path(case.mask_path)
    if not mask_path.exists() or not mask_path.is_file():
        return dataclass_replace(base, read_status="mask_missing", error_code="mask_missing", error_message="mask path does not exist")
    try:
        mask = (read_mask or _read_raw_mask)(mask_path)
    except Exception as exc:  # noqa: BLE001 - preserve case-level read status
        return dataclass_replace(
            base,
            read_status="mask_unreadable",
            error_code="mask_unreadable",
            error_message=_concise_error(exc),
        )
    if mask.GetDimension() != 3:
        return dataclass_replace(
            base,
            read_status="mask_not_3d",
            error_code="mask_not_3d",
            error_message="raw mask image is not 3D",
        )

    mask_geometry = _native_geometry(mask)
    geometry_status, mismatch_fields = compare_geometry(dwi, mask)
    base = dataclass_replace(
        base,
        mask_size_xyz=mask_geometry["size_xyz"],
        mask_spacing_xyz_mm=mask_geometry["spacing_xyz_mm"],
        mask_origin_xyz_mm=mask_geometry["origin_xyz_mm"],
        mask_direction_3x3=mask_geometry["direction_3x3"],
        geometry_status=geometry_status,
        geometry_mismatch_fields=tuple(mismatch_fields),
    )
    if geometry_status != "match":
        metrics = dict(metrics)
        metrics.update(_blank_mask_metrics("skipped_due_to_geometry_mismatch"))
        return dataclass_replace(
            base,
            audit_status="failed",
            read_status="readable",
            error_code="geometry_mismatch",
            error_message="; ".join(mismatch_fields),
            derived_metrics=metrics,
        )
    mask_array = sitk.GetArrayFromImage(mask)
    metrics = dict(metrics)
    metrics.update(mask_metrics(mask_array, mask, dwi_array=dwi_array))
    if metrics["dwi_nonfinite_voxel_count"]:
        for name in (
            "lesion_finite_voxel_count",
            "lesion_min",
            "lesion_max",
            "lesion_mean",
            "lesion_std",
            "lesion_median",
            "lesion_p5",
            "lesion_p25",
            "lesion_p75",
            "lesion_p95",
        ):
            metrics[name] = None
        metrics["lesion_intensity_status"] = "skipped_due_to_nonfinite_dwi"
        return dataclass_replace(
            base,
            audit_status="failed",
            read_status="readable",
            error_code="dwi_nonfinite_intensity",
            error_message="DWI contains non-finite voxel values",
            dwi_mask_comparison_status="match",
            derived_metrics=metrics,
        )
    return dataclass_replace(
        base,
        audit_status="passed",
        read_status="readable",
        error_code="",
        error_message="",
        dwi_mask_comparison_status="match",
        derived_metrics=metrics,
    )


def dataclass_replace(instance, **changes):
    """Small local alias keeping Task 5 independent of output machinery."""
    from dataclasses import replace

    return replace(instance, **changes)


def audit_confirmed_cases(build_index_rows, series_records_for_dir) -> list[CaseRecord]:
    """Create case records only for rows confirmed by build_index."""
    cases = []
    for row in build_index_rows:
        if str(row.get("match_status", "")) != "matched":
            continue
        source = str(row.get("dicom_dir", ""))
        source_key = str(Path(source).resolve())
        if callable(series_records_for_dir):
            records = series_records_for_dir(source)
        else:
            records = series_records_for_dir.get(source_key, series_records_for_dir.get(source, []))
        case = audit_case(row, records)
        if case is not None:
            cases.append(case)
    return cases


def _default_read_uid(file_path: Path) -> str:
    """Read SeriesInstanceUID without loading pixels."""
    import SimpleITK as sitk

    reader = sitk.ImageFileReader()
    reader.SetFileName(str(file_path))
    reader.ReadImageInformation()
    tag = "0020|000e"
    if not reader.HasMetaDataKey(tag):
        return ""
    return reader.GetMetaData(tag).strip()


def discover_dicom_series(
    dicom_dir: Path,
    series_ids=None,
    file_names_for_series=None,
    read_uid=None,
) -> list[SeriesRecord]:
    """Enumerate every GDCM series and validate UID isolation.

    The optional seams accept synthetic series IDs, a mapping/callable returning
    file names, and a UID reader. The production defaults use SimpleITK/GDCM.
    """
    dicom_dir = Path(dicom_dir)
    if series_ids is None:
        import SimpleITK as sitk

        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_dir)) or []
    ordered_series_ids = sorted((str(series_id) for series_id in series_ids))

    if file_names_for_series is None:
        import SimpleITK as sitk

        def get_file_names(series_id):
            return sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
                str(dicom_dir), series_id
            )
    elif callable(file_names_for_series):
        get_file_names = file_names_for_series
    else:
        get_file_names = file_names_for_series.__getitem__

    uid_reader = read_uid or _default_read_uid
    records: list[SeriesRecord] = []
    for gdcm_series_id in ordered_series_ids:
        paths = sorted((Path(path) for path in get_file_names(gdcm_series_id)), key=lambda path: str(path))
        relative_paths = tuple(
            _relative_file_path(path, dicom_dir)
            for path in paths
        )
        if not paths:
            records.append(
                SeriesRecord(
                    series_directory=str(dicom_dir),
                    gdcm_series_id=gdcm_series_id,
                    series_instance_uid="",
                    relative_file_paths=relative_paths,
                    file_count=0,
                    read_status="failed",
                    uid_status="unreadable",
                    error_code="file_list_empty",
                    error_message="empty GDCM file list",
                )
            )
            continue

        uid_values: dict[str, list[str]] = {}
        unreadable: list[str] = []
        for path, relative_path in zip(paths, relative_paths):
            try:
                uid = uid_reader(path)
            except Exception as exc:  # noqa: BLE001 - preserve each read failure
                unreadable.append(f"{relative_path}: {_concise_error(exc)}")
                continue
            normalized_uid = "" if uid is None else str(uid).strip()
            uid_values.setdefault(normalized_uid, []).append(relative_path)

        if unreadable:
            records.append(
                SeriesRecord(
                    series_directory=str(dicom_dir),
                    gdcm_series_id=gdcm_series_id,
                    series_instance_uid="",
                    relative_file_paths=relative_paths,
                    file_count=len(paths),
                    read_status="failed",
                    uid_status="unreadable",
                    error_code="uid_unreadable",
                    error_message="; ".join(sorted(unreadable)),
                )
            )
            continue

        missing = sorted(uid_values.get("", []))
        nonempty_uids = sorted(uid for uid in uid_values if uid)
        if missing:
            records.append(
                SeriesRecord(
                    series_directory=str(dicom_dir),
                    gdcm_series_id=gdcm_series_id,
                    series_instance_uid="",
                    relative_file_paths=relative_paths,
                    file_count=len(paths),
                    read_status="failed",
                    uid_status="missing",
                    error_code="uid_missing",
                    error_message="missing UID: " + ", ".join(missing),
                )
            )
        elif len(nonempty_uids) != 1:
            records.append(
                SeriesRecord(
                    series_directory=str(dicom_dir),
                    gdcm_series_id=gdcm_series_id,
                    series_instance_uid="",
                    relative_file_paths=relative_paths,
                    file_count=len(paths),
                    read_status="failed",
                    uid_status="inconsistent",
                    error_code="uid_inconsistent",
                    error_message="UIDs: " + ", ".join(nonempty_uids),
                )
            )
        else:
            records.append(
                SeriesRecord(
                    series_directory=str(dicom_dir),
                    gdcm_series_id=gdcm_series_id,
                    series_instance_uid=nonempty_uids[0],
                    relative_file_paths=relative_paths,
                    file_count=len(paths),
                    read_status="readable",
                    uid_status="consistent",
                )
            )
    return records


def _relative_file_path(path: Path, dicom_dir: Path) -> str:
    """Return a stable path label without requiring files to exist."""
    try:
        return path.relative_to(dicom_dir).as_posix()
    except ValueError:
        return path.as_posix()


def _concise_error(error: Exception) -> str:
    message = " ".join(str(error).split())
    return message or error.__class__.__name__


def _metadata_value(metadata, field):
    if field in metadata:
        return metadata[field]
    tag = _METADATA_TAGS[field]
    return metadata.get(tag)


def extract_series_metadata(file_paths, read_metadata):
    """Collect public metadata through an injected, pixel-free reader."""
    ordered_paths = tuple(sorted((Path(path) for path in file_paths), key=lambda p: p.as_posix()))
    per_file = {}
    errors = {}
    for path in ordered_paths:
        key = path.as_posix()
        try:
            metadata = read_metadata(path)
            if not isinstance(metadata, Mapping):
                raise TypeError("metadata reader must return a mapping")
            per_file[key] = {
                field: _metadata_value(metadata, field)
                for field in METADATA_FIELDS
                if _metadata_value(metadata, field) is not None
            }
        except Exception as exc:  # noqa: BLE001 - preserve member-level failures
            errors[key] = _concise_error(exc)

    values = {}
    field_status = {}
    per_file_field_values = {field: [] for field in METADATA_FIELDS}
    for metadata in per_file.values():
        for field in METADATA_FIELDS:
            if field in metadata:
                per_file_field_values[field].append(metadata[field])
    per_file_field_values["ImagePositionPatient"] = [
        metadata["ImagePositionPatient"]
        for metadata in per_file.values()
        if "ImagePositionPatient" in metadata
    ]
    for field in METADATA_FIELDS:
        field_values = per_file_field_values[field]
        if not field_values:
            values[field] = None
            field_status[field] = "absent" if per_file else "unreadable"
        elif all(value == field_values[0] for value in field_values[1:]):
            values[field] = field_values[0]
            field_status[field] = "present"
        elif field == "ImagePositionPatient":
            values[field] = field_values[0]
            field_status[field] = "present"
        else:
            values[field] = None
            field_status[field] = "inconsistent"
    return {
        "values": values,
        "field_status": field_status,
        "per_file": per_file,
        "errors": errors,
    }


def _metadata_text(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\\".join(str(item) for item in value).strip().lower()
    return str(value).strip().lower()


def classify_modality(metadata):
    """Classify only from public descriptive DICOM metadata."""
    sources = []
    evidence = []
    texts = []
    for field, source in _CLASSIFICATION_FIELDS:
        value = _metadata_value(metadata, field)
        if value is not None and str(value).strip():
            sources.append(source)
            evidence.append(f"{field}={value}")
            texts.append((field, _metadata_text(value)))
    combined = " ".join(text for _, text in texts)
    hits = set()
    has_adc = bool(re.search(r"\badc\b", combined))
    has_dwi = bool(re.search(r"\bdwi\b", combined))
    if has_adc:
        hits.add("adc")
    if has_dwi:
        hits.add("dwi")
    if re.search(r"\bflair\b", combined) and not re.search(r"\bt1\b", combined):
        hits.add("flair")
    if len(hits) != 1:
        modality_class = "unknown"
        confidence = "none"
    else:
        modality_class = next(iter(hits))
        confidence = "high" if len(texts) >= 2 else "medium"
    return {
        "modality_class": modality_class,
        "classification_source": sources,
        "classification_confidence": confidence,
        "classification_evidence": evidence,
    }


def encode_cell(value):
    """Encode structured CSV-cell values deterministically."""
    if value is None:
        return _sanitize_text("")
    if isinstance(value, (list, tuple, dict)):
        return _sanitize_text(json.dumps(_sanitize_json_value(value), sort_keys=True, separators=(",", ":")))
    return _sanitize_text(str(value))


def _sanitize_text(value: str) -> str:
    """Make surrogate-containing text writable as UTF-8 without changing normal text."""
    return value.encode("utf-8", errors="backslashreplace").decode("utf-8")


def _sanitize_json_value(value):
    """Recursively sanitize strings in JSON-compatible mappings and sequences."""
    if isinstance(value, str):
        return _sanitize_text(value)
    if isinstance(value, dict):
        return {
            _sanitize_json_value(key) if isinstance(key, str) else key: _sanitize_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_json_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_json_value(item) for item in value)
    return value


def parse_target_spacing(x_mm, y_mm):
    """Validate the optional paired XY target spacing arguments."""
    if x_mm is None and y_mm is None:
        return DEFAULT_TARGET_SPACING_XY_MM
    if x_mm is None or y_mm is None:
        raise ValueError("target spacing x and y must be supplied together")
    try:
        values = (float(x_mm), float(y_mm))
    except (TypeError, ValueError) as exc:
        raise ValueError("target spacing values must be finite positive numbers") from exc
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("target spacing values must be finite positive numbers")
    return values


def _finite_vector(values, length):
    try:
        vector = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    if len(vector) != length or not all(math.isfinite(value) for value in vector):
        return None
    return vector


def summarize_image_geometry(image):
    """Summarize SimpleITK geometry without inspecting pixel values."""
    import SimpleITK as sitk

    size_xyz = [int(value) for value in image.GetSize()]
    spacing_xyz = [float(value) for value in image.GetSpacing()]
    origin_xyz = [float(value) for value in image.GetOrigin()]
    direction_3x3 = [float(value) for value in image.GetDirection()]
    shape_zyx = [int(value) for value in sitk.GetArrayFromImage(image).shape]
    return {
        "status": "available",
        "size_xyz": size_xyz,
        "shape_zyx": shape_zyx,
        "spacing_xyz_mm": spacing_xyz,
        "origin_xyz_mm": origin_xyz,
        "direction_3x3": direction_3x3,
        "slice_count": size_xyz[2],
        "inplane_spacing_xy_mm": spacing_xyz[:2],
        "z_spacing_sitk_mm": spacing_xyz[2],
        "fov_xyz_mm": [size_xyz[i] * spacing_xyz[i] for i in range(3)],
    }


def ipp_spacing(positions, orientation):
    """Derive adjacent slice spacing by projecting IPPs onto the slice normal."""
    iop = _finite_vector(orientation, 6)
    if iop is None:
        return {
            "status": "failed",
            "error_code": "orientation_invalid",
            "error_message": "ImageOrientationPatient must contain six finite values",
            "deltas_mm": [],
            "min_mm": None,
            "median_mm": None,
            "max_mm": None,
            "uniform": None,
        }
    row = iop[:3]
    col = iop[3:]
    row_norm = math.sqrt(sum(value * value for value in row))
    col_norm = math.sqrt(sum(value * value for value in col))
    dot = sum(row[i] * col[i] for i in range(3))
    if (
        not math.isclose(row_norm, 1.0, abs_tol=1e-5)
        or not math.isclose(col_norm, 1.0, abs_tol=1e-5)
        or abs(dot) > 1e-5
    ):
        return {
            "status": "failed",
            "error_code": "orientation_invalid",
            "error_message": "row/column direction cosines are not unit and orthogonal",
            "deltas_mm": [],
            "min_mm": None,
            "median_mm": None,
            "max_mm": None,
            "uniform": None,
        }
    normal = [
        row[1] * col[2] - row[2] * col[1],
        row[2] * col[0] - row[0] * col[2],
        row[0] * col[1] - row[1] * col[0],
    ]
    normal_norm = math.sqrt(sum(value * value for value in normal))
    if not math.isfinite(normal_norm) or normal_norm <= 1e-8:
        return {
            "status": "failed",
            "error_code": "orientation_zero_normal",
            "error_message": "row/column cross product has zero norm",
            "deltas_mm": [],
            "min_mm": None,
            "median_mm": None,
            "max_mm": None,
            "uniform": None,
        }
    normal = [value / normal_norm for value in normal]
    projections = []
    invalid_positions = []
    for index, position in enumerate(positions):
        ipp = _finite_vector(position, 3)
        if ipp is None:
            invalid_positions.append(index)
        else:
            projections.append(sum(ipp[i] * normal[i] for i in range(3)))
    if invalid_positions:
        return {
            "status": "unavailable",
            "error_code": "ipp_invalid",
            "error_message": "invalid/missing ImagePositionPatient at indices: " + ",".join(map(str, invalid_positions)),
            "deltas_mm": [],
            "min_mm": None,
            "median_mm": None,
            "max_mm": None,
            "uniform": None,
        }
    if len(projections) < 2:
        return {
            "status": "unavailable",
            "error_code": "ipp_insufficient",
            "error_message": "at least two valid ImagePositionPatient values are required",
            "deltas_mm": [],
            "min_mm": None,
            "median_mm": None,
            "max_mm": None,
            "uniform": None,
        }
    ordered = sorted(projections)
    deltas = [abs(ordered[i + 1] - ordered[i]) for i in range(len(ordered) - 1)]
    middle = float(median(deltas))
    return {
        "status": "available",
        "error_code": "",
        "error_message": "",
        "deltas_mm": deltas,
        "min_mm": min(deltas),
        "median_mm": middle,
        "max_mm": max(deltas),
        "uniform": all(
            math.isclose(
                delta,
                middle,
                rel_tol=IPP_UNIFORM_REL_TOLERANCE,
                abs_tol=IPP_UNIFORM_ABS_TOLERANCE_MM,
            )
            for delta in deltas
        ),
    }


def metadata_z_spacing(metadata):
    """Select metadata z spacing with the documented explicit fallback."""
    for field, source in (
        ("SpacingBetweenSlices", "SpacingBetweenSlices"),
        ("SliceThickness", "SliceThickness"),
    ):
        value = metadata.get(field)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = None
        if value is not None and math.isfinite(value) and value > 0:
            return {"status": "available", "value_mm": value, "source": source}
    return {"status": "unavailable", "value_mm": None, "source": ""}


def z_spacing_discrepancy(metadata_spacing_mm, ipp_median_mm):
    """Return absolute metadata-vs-IPP discrepancy when both values exist."""
    if metadata_spacing_mm is None or ipp_median_mm is None:
        return None
    return abs(float(metadata_spacing_mm) - float(ipp_median_mm))


def index_xyz_to_physical(index_xyz, origin_xyz, spacing_xyz, direction_3x3):
    """Map an XYZ voxel index into physical coordinates."""
    index = _finite_vector(index_xyz, 3)
    origin = _finite_vector(origin_xyz, 3)
    spacing = _finite_vector(spacing_xyz, 3)
    direction = _finite_vector(direction_3x3, 9)
    if None in (index, origin, spacing, direction):
        raise ValueError("physical-coordinate inputs must be finite XYZ values and a 3x3 direction")
    scaled = [index[i] * spacing[i] for i in range(3)]
    return [
        origin[row] + sum(direction[row * 3 + col] * scaled[col] for col in range(3))
        for row in range(3)
    ]


def geometry_failure(error_code, error_message):
    """Create an explicit failed geometry result with no dependent values."""
    return {
        "status": "failed",
        "error_code": str(error_code),
        "error_message": str(error_message),
        "size_xyz": None,
        "shape_zyx": None,
        "spacing_xyz_mm": None,
        "origin_xyz_mm": None,
        "direction_3x3": None,
        "slice_count": None,
        "inplane_spacing_xy_mm": None,
        "z_spacing_sitk_mm": None,
        "fov_xyz_mm": None,
    }


def _read_public_metadata(file_path: Path) -> dict[str, str]:
    """Read only the public DICOM tags used by the audit metadata contract."""
    import SimpleITK as sitk

    reader = sitk.ImageFileReader()
    reader.SetFileName(str(Path(file_path).resolve(strict=True)))
    reader.LoadPrivateTagsOff()
    reader.ReadImageInformation()
    return {
        tag: reader.GetMetaData(tag)
        for tag in _METADATA_TAGS.values()
        if reader.HasMetaDataKey(tag)
    }


def _series_key(record: SeriesRecord) -> tuple[str, str, str]:
    return (
        Path(record.series_directory).resolve().as_posix(),
        str(record.series_instance_uid),
        str(record.gdcm_series_id),
    )


def _candidate_identifiers(series_directory: str, ci1_root: Path) -> dict[str, str]:
    try:
        relative = Path(series_directory).resolve().relative_to(ci1_root)
        parts = relative.parts
    except (OSError, ValueError, RuntimeError):
        parts = ()
    patient = parts[0] if parts else ""
    match = re.search(r"(?:^|[-_ ])(D\d+)(?:[-_ ]|$)", "/".join(parts), re.IGNORECASE)
    timepoint = match.group(1).upper() if match else ""
    source = "path_rule:first_relative_directory;timepoint_token" if parts else ""
    return {
        "internal_case_patient_id": patient,
        "candidate_patient_id": patient,
        "candidate_timepoint": timepoint,
        "candidate_id_source": source,
    }


def _metadata_status(metadata_result: Mapping) -> str:
    per_file = metadata_result.get("per_file", {})
    errors = metadata_result.get("errors", {})
    if errors and per_file:
        return "partial"
    if errors:
        return "failed"
    return "available" if per_file or metadata_result.get("values") else "unavailable"


def _series_row(record, metadata_result, classification, ci1_root, confirmed_case_ids):
    """Render one Task 3 record into the fixed Task 7 series schema."""
    values = metadata_result.get("values", {})
    errors = metadata_result.get("errors", {})
    row = {column: None for column in SERIES_COLUMNS}
    row.update(_candidate_identifiers(record.series_directory, ci1_root))
    row.update(
        {
            "series_instance_uid": record.series_instance_uid,
            "gdcm_series_id": record.gdcm_series_id,
            "series_directory": str(Path(record.series_directory).resolve()),
            "relative_file_count": record.file_count,
            "read_status": record.read_status,
            "uid_status": record.uid_status,
            "metadata_status": _metadata_status(metadata_result),
            "error_code": record.error_code,
            "error_message": record.error_message,
            "metadata_field_status": metadata_result.get("field_status", {}),
            "modality_class": classification.get("modality_class", "unknown"),
            "classification_source": classification.get("classification_source", []),
            "classification_confidence": classification.get("classification_confidence", "none"),
            "classification_evidence": classification.get("classification_evidence", []),
            "pairing_status": "unverified",
            "pairing_evidence": "series_only",
        }
    )
    if errors:
        messages = [f"{path}: {message}" for path, message in sorted(errors.items())]
        row["error_code"] = row["error_code"] or "metadata_unreadable"
        row["error_message"] = "; ".join(
            item for item in (row["error_message"], "; ".join(messages)) if item
        )
    metadata_columns = {
        "SeriesDescription": "series_description",
        "ProtocolName": "protocol_name",
        "SequenceName": "sequence_name",
        "ImageType": "image_type",
        "Modality": "dicom_modality",
        "Manufacturer": "manufacturer",
        "ManufacturerModelName": "manufacturer_model_name",
        "Rows": "rows",
        "Columns": "columns",
        "PixelSpacing": "pixel_spacing",
        "SliceThickness": "slice_thickness",
        "SpacingBetweenSlices": "spacing_between_slices",
        "ImagePositionPatient": "image_position_patient",
        "ImageOrientationPatient": "image_orientation_patient",
        "RescaleSlope": "rescale_slope",
        "RescaleIntercept": "rescale_intercept",
        "BitsAllocated": "bits_allocated",
        "BitsStored": "bits_stored",
        "PixelRepresentation": "pixel_representation",
        "b-value": "b_value",
    }
    for metadata_field, column in metadata_columns.items():
        row[column] = values.get(metadata_field)
    key = (Path(record.series_directory).resolve().as_posix(), str(record.series_instance_uid))
    if key in confirmed_case_ids:
        row.update(
            {
                "pairing_status": "confirmed",
                "pairing_evidence": "prepare_ci1_dwi_dataset.build_index",
                "confirmed_dwi_case_id": confirmed_case_ids[key],
            }
        )
    return row


def _case_row(case: CaseRecord) -> dict[str, object]:
    """Render fixed CaseRecord fields and its derived Task 6 metrics only."""
    metrics = case.derived_metrics if isinstance(case.derived_metrics, Mapping) else {}
    row = {column: None for column in CASE_COLUMNS}
    for column in CASE_COLUMNS:
        if hasattr(case, column):
            row[column] = getattr(case, column)
        elif column in metrics:
            row[column] = metrics[column]
    return row


def _write_csv(path: Path, columns, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: encode_cell(row.get(column)) for column in columns})


def _validate_staged_results(stage_dir: Path, summary, series_count, case_count) -> None:
    if {path.name for path in stage_dir.iterdir()} != set(RESULT_FILES):
        raise OSError("staged result files do not match the fixed result contract")
    for name, columns, expected_count in (
        (RESULT_FILES[0], SERIES_COLUMNS, series_count),
        (RESULT_FILES[1], CASE_COLUMNS, case_count),
    ):
        with (stage_dir / name).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        if not rows or rows[0] != list(columns) or len(rows) - 1 != expected_count:
            raise OSError(f"invalid staged CSV: {name}")
    with (stage_dir / RESULT_FILES[2]).open(encoding="utf-8") as handle:
        staged_summary = json.load(handle)
    if staged_summary != summary or staged_summary.get("result_files") != list(RESULT_FILES):
        raise OSError("invalid staged dataset summary")


def _failure(scope, identifier, error_code, message, state="failed") -> dict[str, str]:
    return {
        "scope": str(scope),
        "identifier": str(identifier),
        "state": str(state),
        "error_code": str(error_code),
        "message": str(message),
    }


def _validate_audit_paths(ci1_root, output_dir) -> tuple[Path, Path, bool]:
    root = Path(ci1_root)
    if not root.exists():
        raise ValueError(f"CI-1 root does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"CI-1 root is not a directory: {root}")
    root = root.resolve(strict=True)
    output = Path(output_dir).resolve()
    if _resolved_is_within(output, root):
        raise ValueError("output directory must be outside the CI-1 input root")
    output_exists = output.exists()
    if output_exists and not output.is_dir():
        raise ValueError(f"output path is not a directory: {output}")
    if output_exists and any(output.iterdir()):
        raise ValueError(f"output directory must be empty: {output}")
    return root, output, output_exists


def audit_dataset(ci1_root, output_dir, target_xy=DEFAULT_TARGET_SPACING_XY_MM):
    """Audit one CI-1 tree into the three fixed, atomically finalized reports."""
    root, output, output_existed = _validate_audit_paths(ci1_root, output_dir)
    target_xy = parse_target_spacing(target_xy[0], target_xy[1])
    output.parent.mkdir(parents=True, exist_ok=True)
    build_index_rows = list(prepare_ci1_dwi_dataset.build_index(root))
    failures = []
    directories = {}
    for row in build_index_rows:
        value = str(row.get("dicom_dir", "")).strip()
        if value:
            path = Path(value).resolve()
            directories[path.as_posix()] = path

    records_by_dir = {}
    enriched_records = []
    metadata_by_record = {}
    classification_by_record = {}
    for directory_key, dicom_dir in sorted(directories.items()):
        try:
            records = discover_dicom_series(dicom_dir)
        except Exception as exc:  # noqa: BLE001 - preserve independent directory failures
            records = []
            failures.append(_failure("dicom_directory", directory_key, "series_discovery_failed", _concise_error(exc)))
        local_records = []
        for record in records:
            full_paths = _record_full_paths(record)
            if full_paths is None:
                metadata_result = {"values": {}, "field_status": {}, "per_file": {}, "errors": {}}
            else:
                metadata_result = extract_series_metadata(full_paths, _read_public_metadata)
            classification = classify_modality(metadata_result.get("values", {}))
            enriched = dataclass_replace(record, modality_class=classification.get("modality_class", "unknown"))
            local_records.append(enriched)
            enriched_records.append(enriched)
            metadata_by_record[_series_key(enriched)] = metadata_result
            classification_by_record[_series_key(enriched)] = classification
        records_by_dir[str(dicom_dir.resolve())] = local_records

    cases = audit_confirmed_cases(build_index_rows, records_by_dir)
    cases = sorted(cases, key=lambda case: (case.case_id, case.patient_id, case.timepoint, case.dwi_source_path))
    confirmed_case_ids = {
        (Path(case.dwi_series_path).resolve().as_posix(), str(case.dwi_series_uid)): case.case_id
        for case in cases
        if case.dwi_series_path and case.dwi_series_uid
    }
    series_rows = [
        _series_row(
            record,
            metadata_by_record[_series_key(record)],
            classification_by_record[_series_key(record)],
            root,
            confirmed_case_ids,
        )
        for record in sorted(enriched_records, key=_series_key)
    ]
    case_rows = [_case_row(case) for case in cases]
    for row in build_index_rows:
        if str(row.get("match_status", "")) != "matched":
            identifier = f"{row.get('patient', '')}_{row.get('timepoint', '')}"
            failures.append(_failure("build_index", identifier, row.get("match_status", ""), "not included in case statistics", "unmatched"))
    for case in cases:
        if case.audit_status != "passed":
            failures.append(_failure("case", case.case_id, case.error_code or "audit_failed", case.error_message, case.audit_status))
    for record in enriched_records:
        if record.read_status != "readable" or record.uid_status != "consistent":
            failures.append(_failure("series", "|".join(_series_key(record)), record.error_code or record.uid_status, record.error_message))
    failures.sort(key=lambda item: (item["scope"], item["identifier"], item["error_code"], item["message"]))
    summary = {
        "schema_version": 1,
        "tool": "ci1_raw_data_audit",
        "protocol": {
            "read_only": True,
            "formal_pairing_source": "prepare_ci1_dwi_dataset.build_index",
            "formal_pairing_status": "match_status=matched",
            "adc_flair_formal_pairing": False,
            "resampling": False,
            "xy_simulation_only": True,
            "connected_component_connectivity": 26,
        },
        "inputs": {
            "ci1_root": str(root),
            "output_dir": str(output),
            "target_spacing_xy_mm": list(target_xy),
            "geometry_tolerances": {
                "spacing_abs_mm": SPACING_TOLERANCE_MM,
                "origin_abs_mm": ORIGIN_TOLERANCE_MM,
                "direction_abs": DIRECTION_TOLERANCE,
            },
        },
        "counts": {
            "build_index_total": len(build_index_rows),
            "build_index_matched": sum(str(row.get("match_status", "")) == "matched" for row in build_index_rows),
            "case_rows": len(case_rows),
            "series_rows": len(series_rows),
        },
        "failures": failures,
        "result_files": list(RESULT_FILES),
    }
    stage_dir = Path(tempfile.mkdtemp(prefix="ci1_audit_", dir=output.parent))
    try:
        _write_csv(stage_dir / RESULT_FILES[0], SERIES_COLUMNS, series_rows)
        _write_csv(stage_dir / RESULT_FILES[1], CASE_COLUMNS, case_rows)
        summary = _sanitize_json_value(summary)
        with (stage_dir / RESULT_FILES[2]).open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
        _validate_staged_results(stage_dir, summary, len(series_rows), len(case_rows))
        if output_existed:
            output.rmdir()
        stage_dir.replace(output)
    except Exception:
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        if output_existed and not output.exists():
            output.mkdir()
        raise
    return summary


def _build_argument_parser():
    parser = argparse.ArgumentParser(description="Write a read-only CI-1 raw-data audit report.")
    parser.add_argument("--ci1-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-spacing-x-mm")
    parser.add_argument("--target-spacing-y-mm")
    return parser


def main(argv=None):
    """Run the dataset audit and return a process status without partial output."""
    parser = _build_argument_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        if exc.code == 0:
            raise
        return 2
    try:
        target_xy = parse_target_spacing(args.target_spacing_x_mm, args.target_spacing_y_mm)
        audit_dataset(args.ci1_root, args.output_dir, target_xy=target_xy)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {_concise_error(exc)}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
