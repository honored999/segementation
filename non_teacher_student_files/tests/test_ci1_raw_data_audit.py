import math
import csv
import json
from dataclasses import replace
from pathlib import Path
import tempfile
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import numpy as np
import SimpleITK as sitk

import audit_ci1_raw_data as audit

from audit_ci1_raw_data import (
    CASE_COLUMNS,
    DEFAULT_TARGET_SPACING_XY_MM,
    DIRECTION_TOLERANCE,
    ORIGIN_TOLERANCE_MM,
    RESULT_FILES,
    SERIES_COLUMNS,
    SPACING_TOLERANCE_MM,
    _write_csv,
    discover_dicom_series,
    encode_cell,
    extract_series_metadata,
    classify_modality,
    summarize_image_geometry,
    ipp_spacing,
    metadata_z_spacing,
    z_spacing_discrepancy,
    index_xyz_to_physical,
    geometry_failure,
    parse_target_spacing,
    _read_raw_mask,
    CaseRecord,
    SeriesRecord,
    audit_case,
    audit_confirmed_cases,
    compare_geometry,
    link_build_index_source,
)


EXPECTED_CASE_COLUMNS = (
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

EXPECTED_SERIES_COLUMNS = (
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


class CI1RawDataAuditContractsTest(unittest.TestCase):
    def test_fixed_result_files_and_tolerances(self):
        self.assertEqual(
            RESULT_FILES,
            ("dicom_series_statistics.csv", "case_statistics.csv", "dataset_summary.json"),
        )
        self.assertEqual(
            DEFAULT_TARGET_SPACING_XY_MM,
            (0.4892368018627167, 0.4892368018627167),
        )
        self.assertEqual(SPACING_TOLERANCE_MM, 1e-5)
        self.assertEqual(ORIGIN_TOLERANCE_MM, 1e-4)
        self.assertEqual(DIRECTION_TOLERANCE, 1e-6)

    def test_case_schema_is_complete_and_has_no_adc_or_flair_fields(self):
        self.assertEqual(tuple(CASE_COLUMNS), tuple(EXPECTED_CASE_COLUMNS))
        required = {
            "pairing_status",
            "pairing_source",
            "build_index_match_status",
            "geometry_status",
            "foreground_slice_count",
            "foreground_slice_ratio",
            "bbox_voxel_size_xyz",
            "bbox_physical_size_mm_xyz",
            "centroid_voxel_xyz",
            "centroid_physical_xyz_mm",
            "component_count_26",
            "largest_component_voxels_26",
            "smallest_component_voxels_26",
            "dwi_all_min",
            "dwi_all_p95",
            "lesion_p5",
            "lesion_p25",
            "lesion_median",
            "lesion_mean",
            "lesion_p75",
            "lesion_p95",
            "xy_target_spacing_xy_mm",
            "xy_simulated_size_xyz",
            "xy_simulated_spacing_xyz_mm",
            "xy_z_preserved",
        }
        self.assertTrue(required.issubset(CASE_COLUMNS))
        self.assertFalse(any(column.startswith(("adc_", "flair_")) for column in CASE_COLUMNS))

    def test_series_schema_is_complete(self):
        self.assertEqual(tuple(SERIES_COLUMNS), tuple(EXPECTED_SERIES_COLUMNS))
        for column in (
            "series_instance_uid",
            "metadata_field_status",
            "modality_class",
            "pairing_evidence",
            "size_xyz",
            "origin_xyz_mm",
            "rescale_applied",
            "all_p5",
            "nonzero_p95",
        ):
            self.assertIn(column, SERIES_COLUMNS)

    def test_encode_cell_is_compact_sorted_and_handles_none(self):
        self.assertEqual(encode_cell(None), "")
        self.assertEqual(encode_cell({"z": 1, "a": [2, 1]}), '{"a":[2,1],"z":1}')
        self.assertEqual(encode_cell([{"b": 2, "a": 1}]), '[{"a":1,"b":2}]')
        self.assertEqual(encode_cell(("x", {"b": 2, "a": 1})), '["x",{"a":1,"b":2}]')

    def test_encode_cell_preserves_utf8_and_escapes_surrogates(self):
        self.assertEqual(encode_cell("中文"), "中文")
        self.assertEqual(encode_cell("ascii"), "ascii")
        encoded = encode_cell("bad\udcff")
        self.assertEqual(encoded, "bad\\udcff")
        encoded_structured = encode_cell({"key": ["bad\udcff"]})
        self.assertEqual(encoded_structured, '{"key":["bad\\\\udcff"]}')

    def test_json_sanitizer_recurses_through_dict_keys_values_and_sequences(self):
        value = {"bad\udcff": ["中文", ("ascii", "bad\udcff")]}
        sanitized = audit._sanitize_json_value(value)
        self.assertEqual(sanitized, {"bad\\udcff": ["中文", ("ascii", "bad\\udcff")]})
        encoded = json.dumps(sanitized, ensure_ascii=False, sort_keys=True)
        self.assertEqual(json.loads(encoded), {"bad\\udcff": ["中文", ["ascii", "bad\\udcff"]]})

    def test_csv_and_json_utf8_round_trip_with_chinese_ascii_and_surrogate(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        csv_path = Path(temp.name) / "rows.csv"
        _write_csv(csv_path, ("text",), [{"text": "中文 ascii bad\udcff"}])
        with csv_path.open(newline="", encoding="utf-8") as handle:
            self.assertEqual(list(csv.reader(handle)), [["text"], ["中文 ascii bad\\udcff"]])

        json_path = Path(temp.name) / "summary.json"
        summary = {"text": "中文 ascii bad\udcff"}
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(audit._sanitize_json_value(summary), handle, ensure_ascii=False)
        with json_path.open(encoding="utf-8") as handle:
            self.assertEqual(json.load(handle), {"text": "中文 ascii bad\\udcff"})

    def test_parse_target_spacing_requires_a_finite_positive_pair(self):
        self.assertEqual(parse_target_spacing(None, None), DEFAULT_TARGET_SPACING_XY_MM)
        self.assertEqual(parse_target_spacing(0.5, 0.75), (0.5, 0.75))
        for values in ((None, 0.5), (0.5, None), (0, 0.5), (-1, 0.5), (math.nan, 0.5), (0.5, math.inf)):
            with self.assertRaises((TypeError, ValueError)):
                parse_target_spacing(*values)


class CI1RawDataDiscoveryTest(unittest.TestCase):
    def setUp(self):
        self.dicom_dir = Path("synthetic_patient")
        self.files = {
            "SERIES-B": ("z.dcm",),
            "SERIES-A": ("b.dcm", "a.dcm"),
        }
        self.uids = {
            "a.dcm": "1.2.3.1",
            "b.dcm": "1.2.3.1",
            "z.dcm": "1.2.3.2",
        }

    def test_distinct_gdcm_ids_and_uids_remain_independent_and_sorted(self):
        records = discover_dicom_series(
            self.dicom_dir,
            series_ids=("SERIES-B", "SERIES-A"),
            file_names_for_series=self.files,
            read_uid=lambda path: self.uids[Path(path).name],
        )

        self.assertEqual([record.gdcm_series_id for record in records], ["SERIES-A", "SERIES-B"])
        self.assertEqual([record.series_instance_uid for record in records], ["1.2.3.1", "1.2.3.2"])
        self.assertEqual([record.file_count for record in records], [2, 1])
        self.assertEqual(records[0].relative_file_paths, ("a.dcm", "b.dcm"))
        self.assertEqual(records[1].relative_file_paths, ("z.dcm",))
        self.assertEqual([record.read_status for record in records], ["readable", "readable"])
        self.assertEqual([record.uid_status for record in records], ["consistent", "consistent"])

    def test_uid_mismatch_is_a_failed_unmerged_record(self):
        records = discover_dicom_series(
            self.dicom_dir,
            series_ids=("SERIES-A",),
            file_names_for_series={"SERIES-A": ("b.dcm", "a.dcm")},
            read_uid=lambda path: {"a.dcm": "1.2.3.1", "b.dcm": "1.2.3.9"}[Path(path).name],
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].series_instance_uid, "")
        self.assertEqual(records[0].read_status, "failed")
        self.assertEqual(records[0].uid_status, "inconsistent")
        self.assertEqual(records[0].error_code, "uid_inconsistent")
        self.assertIn("1.2.3.1", records[0].error_message)
        self.assertIn("1.2.3.9", records[0].error_message)

    def test_missing_uid_is_an_explicit_failure(self):
        records = discover_dicom_series(
            self.dicom_dir,
            series_ids=("SERIES-A",),
            file_names_for_series={"SERIES-A": ("a.dcm", "b.dcm")},
            read_uid=lambda path: {"a.dcm": "1.2.3.1", "b.dcm": ""}[Path(path).name],
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].series_instance_uid, "")
        self.assertEqual(records[0].read_status, "failed")
        self.assertEqual(records[0].uid_status, "missing")
        self.assertEqual(records[0].error_code, "uid_missing")
        self.assertIn("b.dcm", records[0].error_message)

    def test_uid_reader_error_is_an_explicit_failure(self):
        def read_uid(path):
            if Path(path).name == "b.dcm":
                raise OSError("unreadable synthetic member")
            return "1.2.3.1"

        records = discover_dicom_series(
            self.dicom_dir,
            series_ids=("SERIES-A",),
            file_names_for_series={"SERIES-A": ("a.dcm", "b.dcm")},
            read_uid=read_uid,
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].series_instance_uid, "")
        self.assertEqual(records[0].read_status, "failed")
        self.assertEqual(records[0].uid_status, "unreadable")
        self.assertEqual(records[0].error_code, "uid_unreadable")
        self.assertEqual(records[0].error_message, "b.dcm: unreadable synthetic member")


class CI1MetadataAndModalityTest(unittest.TestCase):
    def test_metadata_statuses_values_and_varying_ipp(self):
        values = {
            "a.dcm": {
                "SeriesInstanceUID": "1.2.3",
                "SeriesDescription": "DWI axial",
                "ProtocolName": "stroke",
                "ImagePositionPatient": [0, 0, 0],
                "Rows": 64,
            },
            "b.dcm": {
                "SeriesInstanceUID": "1.2.3",
                "SeriesDescription": "DWI axial",
                "ProtocolName": "stroke",
                "ImagePositionPatient": [0, 0, 5],
                "Rows": 64,
            },
        }
        result = extract_series_metadata(
            (Path("b.dcm"), Path("a.dcm")),
            lambda path: values[path.name],
        )

        self.assertEqual(result["values"]["SeriesDescription"], "DWI axial")
        self.assertEqual(result["values"]["Rows"], 64)
        self.assertEqual(result["field_status"]["SeriesDescription"], "present")
        self.assertEqual(result["field_status"]["Rows"], "present")
        self.assertEqual(result["field_status"]["Manufacturer"], "absent")
        self.assertEqual(result["field_status"]["ImagePositionPatient"], "present")
        self.assertEqual(result["per_file"]["a.dcm"]["ImagePositionPatient"], [0, 0, 0])
        self.assertEqual(result["per_file"]["b.dcm"]["ImagePositionPatient"], [0, 0, 5])

    def test_inconsistent_series_field_is_explicit(self):
        result = extract_series_metadata(
            (Path("a.dcm"), Path("b.dcm")),
            lambda path: {"SeriesDescription": "DWI" if path.name == "a.dcm" else "ADC"},
        )

        self.assertEqual(result["field_status"]["SeriesDescription"], "inconsistent")
        self.assertIsNone(result["values"]["SeriesDescription"])
        self.assertEqual(result["per_file"]["a.dcm"]["SeriesDescription"], "DWI")

    def test_metadata_reader_error_is_unreadable_without_crashing(self):
        def reader(path):
            if path.name == "b.dcm":
                raise OSError("synthetic metadata failure")
            return {"SeriesDescription": "DWI"}

        result = extract_series_metadata((Path("a.dcm"), Path("b.dcm")), reader)

        self.assertEqual(result["field_status"]["SeriesDescription"], "present")
        self.assertEqual(result["field_status"]["Manufacturer"], "absent")
        self.assertEqual(result["errors"]["b.dcm"], "synthetic metadata failure")

    def test_modality_classification_rules_and_evidence(self):
        dwi = classify_modality({"SeriesDescription": "DWI axial", "ProtocolName": "stroke"})
        adc = classify_modality({"SeriesDescription": "ADC map"})
        flair = classify_modality({"ProtocolName": "3D FLAIR brain"})
        conflict = classify_modality({"SeriesDescription": "DWI ADC"})
        unknown = classify_modality({"SeriesDescription": "T2 axial"})

        self.assertEqual(dwi["modality_class"], "dwi")
        self.assertEqual(dwi["classification_source"], ["dicom:SeriesDescription", "dicom:ProtocolName"])
        self.assertEqual(dwi["classification_confidence"], "high")
        self.assertEqual(dwi["classification_evidence"], ["SeriesDescription=DWI axial", "ProtocolName=stroke"])
        self.assertEqual(adc["modality_class"], "adc")
        self.assertEqual(adc["classification_confidence"], "medium")
        self.assertEqual(flair["modality_class"], "flair")
        self.assertEqual(conflict["modality_class"], "unknown")
        self.assertEqual(conflict["classification_confidence"], "none")
        self.assertEqual(unknown["modality_class"], "unknown")
        self.assertEqual(unknown["classification_source"], ["dicom:SeriesDescription"])

    def test_classifier_never_creates_formal_pairing(self):
        result = classify_modality({"SeriesDescription": "ADC map"})

        self.assertNotIn("pairing_status", result)
        self.assertNotIn("confirmed_dwi_case_id", result)


class CI1GeometryTest(unittest.TestCase):
    def test_geometry_reports_xyz_zyx_and_exact_fov(self):
        image = sitk.Image([4, 3, 2], sitk.sitkFloat32)
        image.SetSpacing((0.5, 2.0, 3.0))
        image.SetOrigin((1.0, 2.0, 3.0))
        image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

        result = summarize_image_geometry(image)

        self.assertEqual(result["size_xyz"], [4, 3, 2])
        self.assertEqual(result["shape_zyx"], [2, 3, 4])
        self.assertEqual(result["spacing_xyz_mm"], [0.5, 2.0, 3.0])
        self.assertEqual(result["origin_xyz_mm"], [1.0, 2.0, 3.0])
        self.assertEqual(result["slice_count"], 2)
        self.assertEqual(result["inplane_spacing_xy_mm"], [0.5, 2.0])
        self.assertEqual(result["z_spacing_sitk_mm"], 3.0)
        self.assertEqual(result["fov_xyz_mm"], [2.0, 6.0, 6.0])

    def test_ipp_spacing_axial_uniform(self):
        result = ipp_spacing(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 8.0], [0.0, 0.0, 16.0]],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        )
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["deltas_mm"], [8.0, 8.0])
        self.assertEqual(result["min_mm"], 8.0)
        self.assertEqual(result["median_mm"], 8.0)
        self.assertEqual(result["max_mm"], 8.0)
        self.assertTrue(result["uniform"])

    def test_ipp_spacing_uses_oblique_normal_projection(self):
        angle = math.radians(45.0)
        orientation = [1.0, 0.0, 0.0, 0.0, math.cos(angle), math.sin(angle)]
        normal = [0.0, -math.sin(angle), math.cos(angle)]
        positions = [[normal[i] * 8.0 * n for i in range(3)] for n in range(3)]

        result = ipp_spacing(positions, orientation)

        self.assertEqual(result["status"], "available")
        self.assertAlmostEqual(positions[1][2] - positions[0][2], 8.0 * math.cos(angle))
        for delta in result["deltas_mm"]:
            self.assertAlmostEqual(delta, 8.0)
        self.assertAlmostEqual(result["median_mm"], 8.0)

    def test_ipp_spacing_reports_nonuniform_projection(self):
        result = ipp_spacing(
            [[0.0, 0.0, z] for z in (0.0, 8.0, 16.0, 28.0)],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        )
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["deltas_mm"], [8.0, 8.0, 12.0])
        self.assertEqual(result["median_mm"], 8.0)
        self.assertFalse(result["uniform"])

    def test_ipp_spacing_rejects_invalid_orientation(self):
        for orientation in (
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 2.0, 0.0],
            [math.nan, 0.0, 0.0, 0.0, 1.0, 0.0],
        ):
            result = ipp_spacing([[0.0, 0.0, 0.0], [0.0, 0.0, 8.0]], orientation)
            self.assertEqual(result["status"], "failed")
            self.assertIn(result["error_code"], {"orientation_zero_normal", "orientation_invalid"})

    def test_ipp_spacing_rejects_missing_or_invalid_positions(self):
        missing = ipp_spacing([[0.0, 0.0, 0.0], None], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        invalid = ipp_spacing([[0.0, 0.0, 0.0], [0.0, math.inf, 8.0]], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        too_few = ipp_spacing([[0.0, 0.0, 0.0]], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        for result in (missing, invalid, too_few):
            self.assertEqual(result["status"], "unavailable")
            self.assertIsNone(result["median_mm"])
            self.assertTrue(result["error_code"])

    def test_metadata_spacing_preference_fallback_and_discrepancy(self):
        self.assertEqual(metadata_z_spacing({"SpacingBetweenSlices": "4.5", "SliceThickness": "7"}), {"status": "available", "value_mm": 4.5, "source": "SpacingBetweenSlices"})
        self.assertEqual(metadata_z_spacing({"SliceThickness": "7"}), {"status": "available", "value_mm": 7.0, "source": "SliceThickness"})
        self.assertEqual(metadata_z_spacing({}), {"status": "unavailable", "value_mm": None, "source": ""})
        self.assertEqual(z_spacing_discrepancy(4.5, 4.0), 0.5)
        self.assertIsNone(z_spacing_discrepancy(None, 4.0))
        self.assertIsNone(z_spacing_discrepancy(4.5, None))

    def test_index_xyz_to_physical_identity_and_nonidentity(self):
        self.assertEqual(
            index_xyz_to_physical([2.0, 3.0, 4.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            [2.0, 3.0, 4.0],
        )
        self.assertEqual(
            index_xyz_to_physical([1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [2.0, 3.0, 4.0], [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            [4.0, 22.0, 42.0],
        )

    def test_geometry_failure_blanks_dependent_fields(self):
        result = geometry_failure("image_read_failed", "synthetic read failure")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_code"], "image_read_failed")
        self.assertEqual(result["error_message"], "synthetic read failure")
        self.assertIsNone(result["size_xyz"])
        self.assertIsNone(result["fov_xyz_mm"])
        self.assertIsNone(result["slice_count"])


class CI1Task5CaseAuditTest(unittest.TestCase):
    def _write_image(
        self,
        path,
        *,
        size=(2, 2, 2),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        direction=None,
        pixel_type=sitk.sitkFloat32,
        value=1,
    ):
        image = sitk.Image(list(size), pixel_type)
        image.SetSpacing(spacing)
        image.SetOrigin(origin)
        if direction is not None:
            image.SetDirection(direction)
        image += value
        sitk.WriteImage(image, str(path))
        return image

    def _fixture(self, *, modality_class="dwi", mask_name="mask.nii.gz"):
        temp = TemporaryDirectory()
        root = Path(temp.name)
        dwi_dir = root / "patient" / "D1" / "source"
        dwi_dir.mkdir(parents=True)
        dwi_paths = (dwi_dir / "dwi0.dcm", dwi_dir / "dwi1.dcm")
        mask_path = root / mask_name
        for index, dwi_path in enumerate(dwi_paths):
            dwi_slice = sitk.Image([2, 2], sitk.sitkInt16)
            dwi_slice += 1
            dwi_slice.SetSpacing((1.0, 1.0))
            dwi_slice.SetOrigin((0.0, 0.0))
            dwi_slice.SetMetaData("0008|0060", "MR")
            dwi_slice.SetMetaData("0020|000e", "1.2.826.0.1")
            dwi_slice.SetMetaData("0020|000d", "1.2.826.0.2")
            dwi_slice.SetMetaData("0008|0018", f"1.2.826.0.3.{index + 1}")
            dwi_slice.SetMetaData("0020|0032", f"0\\0\\{index}")
            dwi_slice.SetMetaData("0020|0037", "1\\0\\0\\0\\1\\0")
            dwi_slice.SetMetaData("0020|0013", str(index + 1))
            sitk.WriteImage(dwi_slice, str(dwi_path))
        self._write_image(mask_path, pixel_type=sitk.sitkUInt8, value=1)
        record = SeriesRecord(
            series_directory=str(dwi_dir),
            gdcm_series_id="synthetic-series",
            series_instance_uid="1.2.826.0.1",
            relative_file_paths=tuple(path.name for path in dwi_paths),
            file_count=2,
            read_status="readable",
            uid_status="consistent",
            modality_class=modality_class,
        )
        row = {
            "patient": "patient",
            "timepoint": "D1",
            "modality": "DWI",
            "segmentation_path": str(mask_path),
            "dicom_dir": str(dwi_dir),
            "dicom_count": "1",
            "match_status": "matched",
        }
        return temp, root, row, record, dwi_paths[0], mask_path

    def test_unique_source_link_uses_resolved_source_directory(self):
        temp, _, row, record, _, _ = self._fixture()
        self.addCleanup(temp.cleanup)

        result = link_build_index_source(row["dicom_dir"], [record])

        self.assertEqual(result.link_status, "linked")
        self.assertEqual(result.series_instance_uid, "1.2.826.0.1")
        self.assertEqual(result.series_path, str(Path(row["dicom_dir"]).resolve()))

    def test_source_link_selects_dwi_from_records_in_same_directory(self):
        temp, _, row, record, _, _ = self._fixture()
        self.addCleanup(temp.cleanup)
        records = [
            replace(record, series_instance_uid="1.2.826.0.2", modality_class="adc"),
            replace(record, series_instance_uid="1.2.826.0.3", modality_class="flair"),
            replace(record, series_instance_uid="1.2.826.0.1", modality_class="DWI"),
        ]

        result = link_build_index_source(row["dicom_dir"], records)

        self.assertEqual(result.link_status, "linked")
        self.assertEqual(result.series_instance_uid, "1.2.826.0.1")

    def test_source_link_selects_only_dwi_when_multiple_series_share_directory(self):
        temp, _, row, record, _, _ = self._fixture()
        self.addCleanup(temp.cleanup)
        records = [
            replace(record, series_instance_uid="1.2.826.0.2", modality_class="adc"),
            replace(record, series_instance_uid="1.2.826.0.1", modality_class="dwi"),
        ]

        result = link_build_index_source(row["dicom_dir"], records)

        self.assertEqual(result.link_status, "linked")
        self.assertEqual(result.series_instance_uid, "1.2.826.0.1")

    def test_source_link_rejects_two_dwi_records_without_arbitrary_selection(self):
        temp, _, row, record, _, _ = self._fixture()
        self.addCleanup(temp.cleanup)
        first = replace(record, series_instance_uid="1.2.826.0.2", modality_class="dWi")
        second = replace(record, series_instance_uid="1.2.826.0.1", modality_class="DWI")

        result = link_build_index_source(row["dicom_dir"], [first, second])

        self.assertEqual(result.link_status, "multiple_matching_series")
        self.assertIsNone(result.record)

    def test_source_link_distinguishes_no_match_multiple_uid_and_unreadable_states(self):
        temp, _, row, record, _, _ = self._fixture()
        self.addCleanup(temp.cleanup)

        no_match = replace(record, series_directory=str(Path(row["dicom_dir"]).parent / "other"))
        self.assertEqual(link_build_index_source(row["dicom_dir"], [no_match]).link_status, "no_matching_series")

        second = replace(record, series_instance_uid="1.2.826.0.2")
        self.assertEqual(
            link_build_index_source(row["dicom_dir"], [record, second]).link_status,
            "multiple_matching_series",
        )

        invalid_uid = replace(record, series_instance_uid="not-a-dicom-uid")
        self.assertEqual(link_build_index_source(row["dicom_dir"], [invalid_uid]).link_status, "uid_invalid")

        unreadable = replace(record, read_status="failed")
        self.assertEqual(link_build_index_source(row["dicom_dir"], [unreadable]).link_status, "series_unreadable")

        self.assertEqual(link_build_index_source(str(Path(row["dicom_dir"]) / "missing"), [record]).link_status, "source_unreadable")

    def test_unknown_and_adc_metadata_do_not_qualify_as_dwi(self):
        temp, _, row, record, _, _ = self._fixture(modality_class="unknown")
        self.addCleanup(temp.cleanup)

        unknown = audit_case(row, [record])
        self.assertEqual(unknown.pairing_status, "confirmed")
        self.assertEqual(unknown.link_status, "no_matching_series")
        self.assertEqual(unknown.metadata_modality_consistency, "unknown")

        adc = audit_case(row, [replace(record, modality_class="adc")])
        self.assertEqual(adc.link_status, "no_matching_series")
        self.assertEqual(adc.metadata_modality_consistency, "unknown")
        self.assertEqual(adc.pairing_status, "confirmed")

    def test_compare_geometry_accepts_exact_native_geometry_and_boundaries(self):
        image = sitk.Image([2, 2, 2], sitk.sitkFloat32)
        mask = sitk.Image([2, 2, 2], sitk.sitkUInt8)

        self.assertEqual(compare_geometry(image, mask), ("match", []))

        spacing_boundary = sitk.Image([2, 2, 2], sitk.sitkUInt8)
        spacing_boundary.SetSpacing((1.0 + SPACING_TOLERANCE_MM, 1.0, 1.0))
        self.assertEqual(compare_geometry(image, spacing_boundary), ("match", []))

        spacing_over = sitk.Image([2, 2, 2], sitk.sitkUInt8)
        spacing_over.SetSpacing((1.0 + SPACING_TOLERANCE_MM * 1.01, 1.0, 1.0))
        self.assertEqual(compare_geometry(image, spacing_over), ("mismatch", ["spacing"]))

        origin_boundary = sitk.Image([2, 2, 2], sitk.sitkUInt8)
        origin_boundary.SetOrigin((ORIGIN_TOLERANCE_MM, 0.0, 0.0))
        self.assertEqual(compare_geometry(image, origin_boundary), ("match", []))

        direction_boundary = sitk.Image([2, 2, 2], sitk.sitkUInt8)
        direction = list(direction_boundary.GetDirection())
        direction[1] = DIRECTION_TOLERANCE
        direction_boundary.SetDirection(direction)
        self.assertEqual(compare_geometry(image, direction_boundary), ("match", []))

    def test_compare_geometry_reports_deterministic_multi_mismatch(self):
        image = sitk.Image([2, 2, 2], sitk.sitkFloat32)
        other = sitk.Image([3, 2, 2], sitk.sitkUInt8)
        other.SetSpacing((1.0 + SPACING_TOLERANCE_MM * 2, 1.0, 1.0))
        other.SetOrigin((ORIGIN_TOLERANCE_MM * 2, 0.0, 0.0))
        direction = list(other.GetDirection())
        direction[1] = DIRECTION_TOLERANCE * 2
        other.SetDirection(direction)

        self.assertEqual(compare_geometry(image, other), ("mismatch", ["size", "spacing", "origin", "direction"]))

    def test_raw_mask_is_native_unchanged_and_task6_metrics_are_created(self):
        temp, _, row, record, _, mask_path = self._fixture()
        self.addCleanup(temp.cleanup)
        before = sitk.ReadImage(str(mask_path))

        result = audit_case(row, [record])
        after = sitk.ReadImage(str(mask_path))

        self.assertIsInstance(result, CaseRecord)
        self.assertEqual(result.geometry_status, "match")
        self.assertEqual(result.dwi_mask_comparison_status, "match")
        self.assertEqual(result.derived_metrics["mask_voxel_count"], 8)
        self.assertEqual(result.derived_metrics["lesion_mean"], 1.0)
        self.assertEqual(before.GetSize(), after.GetSize())
        self.assertEqual(before.GetSpacing(), after.GetSpacing())
        self.assertEqual(before.GetOrigin(), after.GetOrigin())
        self.assertEqual(before.GetDirection(), after.GetDirection())
        self.assertEqual(sitk.GetArrayFromImage(before).tolist(), sitk.GetArrayFromImage(after).tolist())

    def test_geometry_mismatch_preserves_native_metadata_and_keeps_only_dwi_metrics(self):
        temp, _, row, record, dwi_path, _ = self._fixture()
        self.addCleanup(temp.cleanup)
        before = sitk.ReadImage(str(dwi_path))
        mismatched_mask = Path(temp.name) / "mismatch.nii.gz"
        self._write_image(mismatched_mask, size=(3, 2, 2), pixel_type=sitk.sitkUInt8, value=1)
        row["segmentation_path"] = str(mismatched_mask)

        result = audit_case(row, [record])

        self.assertEqual(result.audit_status, "failed")
        self.assertEqual(result.dwi_mask_comparison_status, "failed")
        self.assertEqual(result.geometry_status, "mismatch")
        self.assertEqual(result.geometry_mismatch_fields, ("size",))
        self.assertEqual(result.dwi_size_xyz, (2, 2, 2))
        self.assertEqual(result.mask_size_xyz, (3, 2, 2))
        self.assertEqual(result.derived_metrics["dwi_finite_voxel_count"], 8)
        self.assertIsNone(result.derived_metrics["mask_voxel_count"])
        self.assertEqual(result.derived_metrics["lesion_intensity_status"], "skipped_due_to_geometry_mismatch")
        after = sitk.ReadImage(str(dwi_path))
        self.assertEqual(before.GetSize(), after.GetSize())
        self.assertEqual(before.GetSpacing(), after.GetSpacing())
        self.assertEqual(before.GetOrigin(), after.GetOrigin())
        self.assertEqual(before.GetDirection(), after.GetDirection())
        self.assertEqual(sitk.GetArrayFromImage(before).tolist(), sitk.GetArrayFromImage(after).tolist())

    def test_mask_missing_unreadable_and_not_3d_are_explicit(self):
        temp, root, row, record, _, _ = self._fixture()
        self.addCleanup(temp.cleanup)

        missing_row = dict(row, segmentation_path=str(root / "missing-mask.nii.gz"))
        self.assertEqual(audit_case(missing_row, [record]).read_status, "mask_missing")

        unreadable_path = root / "unreadable-mask.nii.gz"
        unreadable_path.write_bytes(b"not an image")
        unreadable_row = dict(row, segmentation_path=str(unreadable_path))
        self.assertEqual(audit_case(unreadable_row, [record]).read_status, "mask_unreadable")

        mask_2d = root / "mask-2d.nii.gz"
        image_2d = sitk.Image([2, 2], sitk.sitkUInt8)
        image_2d += 1
        sitk.WriteImage(image_2d, str(mask_2d))
        mask_2d_row = dict(row, segmentation_path=str(mask_2d))
        self.assertEqual(audit_case(mask_2d_row, [record]).read_status, "mask_not_3d")

    def test_nonmatched_build_index_rows_are_excluded_before_linkage(self):
        temp, _, row, record, _, _ = self._fixture()
        self.addCleanup(temp.cleanup)
        missing = dict(row, match_status="missing_dicom", dicom_dir=str(Path(row["dicom_dir"]) / "missing"))

        cases = audit_confirmed_cases(
            [missing, row],
            {str(Path(row["dicom_dir"]).resolve()): [record]},
        )

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].build_index_match_status, "matched")
        self.assertEqual(cases[0].pairing_status, "confirmed")


class CI1RawMaskReaderTest(unittest.TestCase):
    def _write_synthetic_mask(self, path):
        array = np.array(
            [
                [[0, 1], [2, 0]],
                [[3, 0], [0, 4]],
            ],
            dtype=np.uint8,
        )
        image = sitk.GetImageFromArray(array)
        image.SetSpacing((0.7, 0.8, 4.5))
        image.SetOrigin((11.0, -2.0, 7.5))
        image.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
        suffix = ".nii.gz" if path.name.lower().endswith(".nii.gz") else ".nii"
        with tempfile.TemporaryDirectory() as ascii_temp:
            write_path = Path(ascii_temp) / f"synthetic{suffix}"
            sitk.WriteImage(image, str(write_path))
            path.write_bytes(write_path.read_bytes())
        return array, image

    def test_unicode_nifti_falls_back_by_copying_unchanged_bytes_and_cleans_temp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        source = Path(temp.name) / "中文目录" / "mask.NII.GZ"
        source.parent.mkdir()
        expected_array, expected_image = self._write_synthetic_mask(source)
        before_bytes = source.read_bytes()
        calls = []
        real_read_image = sitk.ReadImage

        def read_image(path):
            calls.append(Path(path))
            if Path(path) == source:
                raise OSError("synthetic Unicode-path reader failure")
            self.assertEqual(Path(path).read_bytes(), before_bytes)
            return real_read_image(path)

        with mock.patch.object(sitk, "ReadImage", side_effect=read_image):
            result = _read_raw_mask(source)

        self.assertEqual(calls[0], source)
        self.assertEqual(len(calls), 2)
        fallback_path = calls[1]
        self.assertTrue(str(fallback_path).isascii())
        self.assertTrue(str(fallback_path).lower().endswith(".nii.gz"))
        self.assertFalse(fallback_path.exists())
        self.assertFalse(fallback_path.parent.exists())
        self.assertEqual(source.read_bytes(), before_bytes)
        self.assertEqual(sitk.GetArrayFromImage(result).tolist(), expected_array.tolist())
        self.assertEqual(result.GetSize(), expected_image.GetSize())
        np.testing.assert_allclose(result.GetSpacing(), expected_image.GetSpacing(), rtol=0.0, atol=1e-6)
        self.assertEqual(result.GetOrigin(), expected_image.GetOrigin())
        self.assertEqual(result.GetDirection(), expected_image.GetDirection())

    def test_existing_nifti_directory_reraises_exact_direct_error_without_fallback(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        source = Path(temp.name) / "mask.nii"
        source.mkdir()
        calls = []
        direct_error = OSError("synthetic direct reader failure")

        def read_image(path):
            calls.append(Path(path))
            raise direct_error

        with mock.patch.object(sitk, "ReadImage", side_effect=read_image):
            with self.assertRaises(OSError) as raised:
                _read_raw_mask(source)

        self.assertIs(raised.exception, direct_error)
        self.assertEqual(calls, [source])

    def test_missing_and_unsupported_sources_reraise_direct_reader_without_fallback(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        missing = Path(temp.name) / "missing.nii.gz"
        unsupported = Path(temp.name) / "mask.txt"
        unsupported.write_bytes(b"synthetic")

        for source in (missing, unsupported):
            with self.subTest(source=source):
                calls = []
                direct_error = OSError("synthetic direct reader failure")

                def read_image(path):
                    calls.append(Path(path))
                    raise direct_error

                with mock.patch.object(sitk, "ReadImage", side_effect=read_image):
                    with self.assertRaises(Exception) as raised:
                        _read_raw_mask(source)

                self.assertIs(raised.exception, direct_error)
                self.assertEqual(calls, [source])

    def test_fallback_reader_failure_is_propagated_and_temp_is_cleaned(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        source = Path(temp.name) / "中文目录" / "mask.nii"
        source.parent.mkdir()
        self._write_synthetic_mask(source)
        calls = []
        direct_error = OSError("synthetic direct reader failure")
        fallback_error = RuntimeError("synthetic fallback reader failure")

        def read_image(path):
            calls.append(Path(path))
            raise direct_error if Path(path) == source else fallback_error

        with mock.patch.object(sitk, "ReadImage", side_effect=read_image):
            with self.assertRaisesRegex(RuntimeError, "synthetic fallback reader failure") as raised:
                _read_raw_mask(source)

        self.assertIs(raised.exception, fallback_error)
        self.assertEqual(calls[0], source)
        self.assertEqual(len(calls), 2)
        self.assertFalse(calls[1].exists())
        self.assertFalse(calls[1].parent.exists())

    def test_fallback_copy_error_is_propagated_and_temp_is_cleaned(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        source = Path(temp.name) / "中文目录" / "mask.nii"
        source.parent.mkdir()
        self._write_synthetic_mask(source)
        calls = []
        created_dirs = []
        direct_error = OSError("synthetic direct reader failure")
        copy_error = PermissionError("synthetic fallback copy failure")
        real_temporary_directory = tempfile.TemporaryDirectory

        def temporary_directory(*args, **kwargs):
            directory = real_temporary_directory(*args, **kwargs)
            created_dirs.append(Path(directory.name))
            return directory

        def read_image(path):
            calls.append(Path(path))
            raise direct_error

        with mock.patch.object(tempfile, "TemporaryDirectory", side_effect=temporary_directory), \
                mock.patch.object(Path, "write_bytes", side_effect=copy_error), \
                mock.patch.object(sitk, "ReadImage", side_effect=read_image):
            with self.assertRaises(PermissionError) as raised:
                _read_raw_mask(source)

        self.assertIs(raised.exception, copy_error)
        self.assertEqual(calls, [source])
        self.assertEqual(len(created_dirs), 1)
        self.assertFalse(created_dirs[0].exists())


class CI1Task6MetricsTest(unittest.TestCase):
    def _image(self, array, *, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), direction=None):
        image = sitk.GetImageFromArray(np.asarray(array))
        image.SetSpacing(spacing)
        image.SetOrigin(origin)
        if direction is not None:
            image.SetDirection(direction)
        return image

    def _case(self, dwi_array, mask_array, *, mask_spacing=None, dwi_spacing=None):
        helper = CI1Task5CaseAuditTest()
        temp, _, row, record, _, mask_path = helper._fixture()
        dwi = self._image(dwi_array, spacing=dwi_spacing or (1.0, 1.0, 1.0))
        mask = self._image(mask_array, spacing=mask_spacing or dwi.GetSpacing())
        sitk.WriteImage(mask, str(mask_path))
        return temp, row, record, dwi, mask

    def test_mask_metrics_exact_volume_and_foreground_slices(self):
        mask_array = np.zeros((5, 4, 3), dtype=np.uint8)
        mask_array[1, 0, :3] = 1
        mask_array[1, 1, :2] = 1
        mask_array[3, 0, :] = 1
        mask_array[3, 1, :2] = 1
        result = audit.mask_metrics(mask_array, self._image(mask_array, spacing=(0.5, 0.5, 8.0)))

        self.assertEqual(result["mask_voxel_count"], 10)
        self.assertEqual(result["mask_volume_mm3"], 20.0)
        self.assertEqual(result["mask_volume_ml"], 0.02)
        self.assertEqual(result["foreground_slice_count"], 2)
        self.assertEqual(result["foreground_slice_ratio"], 0.4)
        self.assertEqual(result["mask_unique_values"], [0, 1])

    def test_empty_mask_is_valid_with_zero_and_blank_lesion_fields(self):
        mask_array = np.zeros((3, 2, 2), dtype=np.uint8)
        result = audit.mask_metrics(mask_array, self._image(mask_array))

        self.assertTrue(result["empty_mask"])
        self.assertEqual(result["mask_voxel_count"], 0)
        self.assertEqual(result["component_count_26"], 0)
        for key in ("largest_component_voxels_26", "largest_component_volume_mm3_26",
                    "smallest_component_voxels_26", "smallest_component_volume_mm3_26"):
            self.assertIsNone(result[key], key)
        self.assertEqual(result["foreground_slice_count"], 0)
        self.assertEqual(result["foreground_slice_ratio"], 0.0)
        self.assertIsNone(result["lesion_finite_voxel_count"])
        for key in ("bbox_voxel_size_xyz", "bbox_physical_size_mm_xyz", "centroid_voxel_xyz",
                    "centroid_physical_xyz_mm", "lesion_index_min_xyz", "lesion_index_max_xyz",
                    "lesion_physical_min_xyz_mm", "lesion_physical_max_xyz_mm", "lesion_min",
                    "lesion_max", "lesion_mean", "lesion_std", "lesion_median", "lesion_p5",
                    "lesion_p25", "lesion_p75", "lesion_p95"):
            self.assertIsNone(result[key], key)
        self.assertEqual(result["lesion_intensity_status"], "empty_mask")

    def test_components_use_26_connectivity_and_report_largest_smallest(self):
        mask_array = np.zeros((4, 4, 4), dtype=np.uint8)
        mask_array[0, 0, 0] = 1
        mask_array[1, 1, 1] = 1
        mask_array[3, 0, 3] = 1
        result = audit.mask_metrics(mask_array, self._image(mask_array, spacing=(2.0, 3.0, 4.0)))

        self.assertEqual(result["component_connectivity"], 26)
        self.assertEqual(result["component_count_26"], 2)
        self.assertEqual(result["largest_component_voxels_26"], 2)
        self.assertEqual(result["largest_component_volume_mm3_26"], 48.0)
        self.assertEqual(result["smallest_component_voxels_26"], 1)
        self.assertEqual(result["smallest_component_volume_mm3_26"], 24.0)

    def test_bbox_is_inclusive_xyz_and_physical_size(self):
        mask_array = np.zeros((4, 4, 5), dtype=np.uint8)
        mask_array[1:4, 1:3, 1:4] = 1
        result = audit.mask_metrics(mask_array, self._image(mask_array, spacing=(2.0, 3.0, 4.0)))

        self.assertEqual(result["lesion_index_min_xyz"], [1, 1, 1])
        self.assertEqual(result["lesion_index_max_xyz"], [3, 2, 3])
        self.assertEqual(result["bbox_voxel_size_xyz"], [3, 2, 3])
        self.assertEqual(result["bbox_physical_size_mm_xyz"], [6.0, 6.0, 12.0])

    def test_fractional_centroid_uses_xyz_and_existing_physical_transform(self):
        mask_array = np.zeros((2, 2, 3), dtype=np.uint8)
        mask_array[0, 0, 0] = 1
        mask_array[0, 1, 2] = 1
        mask_array[1, 1, 0] = 1
        direction = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        image = self._image(mask_array, spacing=(2.0, 3.0, 4.0), origin=(10.0, 20.0, 30.0), direction=direction)
        result = audit.mask_metrics(mask_array, image)

        expected_centroid = [2.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0]
        self.assertEqual(result["centroid_voxel_xyz"], expected_centroid)
        self.assertEqual(
            result["centroid_physical_xyz_mm"],
            audit.index_xyz_to_physical(expected_centroid, image.GetOrigin(), image.GetSpacing(), image.GetDirection()),
        )

    def test_rotated_physical_bounds_transform_all_bbox_corners(self):
        mask_array = np.zeros((2, 3, 3), dtype=np.uint8)
        mask_array[:, 1:3, 1:3] = 1
        direction = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        image = self._image(mask_array, spacing=(2.0, 3.0, 4.0), origin=(10.0, 20.0, 30.0), direction=direction)
        result = audit.mask_metrics(mask_array, image)

        self.assertEqual(result["lesion_physical_min_xyz_mm"], [4.0, 22.0, 30.0])
        self.assertEqual(result["lesion_physical_max_xyz_mm"], [7.0, 24.0, 34.0])

    def test_intensity_summary_reports_exact_all_and_nonzero_statistics(self):
        values = np.arange(10, dtype=np.float64)
        result = audit.summarize_intensity(values)

        self.assertEqual(result["finite_voxel_count"], 10)
        self.assertEqual(result["nonfinite_voxel_count"], 0)
        for prefix, expected_values in (("all", values), ("nonzero", values[1:])):
            self.assertEqual(result[f"{prefix}_min"], float(expected_values.min()))
            self.assertEqual(result[f"{prefix}_max"], float(expected_values.max()))
            self.assertEqual(result[f"{prefix}_mean"], float(expected_values.mean()))
            self.assertEqual(result[f"{prefix}_std"], float(expected_values.std()))
            for name, percentile in (("p1", 1), ("p5", 5), ("p25", 25), ("median", 50),
                                     ("p75", 75), ("p95", 95), ("p99", 99)):
                self.assertEqual(result[f"{prefix}_{name}"], float(np.percentile(expected_values, percentile)))

    def test_nonfinite_intensity_fails_and_blanks_all_nonzero_and_lesion_stats(self):
        summary = audit.summarize_intensity(np.array([1.0, np.nan, np.inf, 0.0]))
        self.assertEqual(summary["finite_voxel_count"], 2)
        self.assertEqual(summary["nonfinite_voxel_count"], 2)
        self.assertIsNone(summary["all_min"])
        self.assertIsNone(summary["nonzero_p95"])

        temp, row, record, dwi, mask = self._case(
            np.array([[[1.0, np.nan], [3.0, 4.0]]]),
            np.array([[[1, 0], [0, 0]]], dtype=np.uint8),
        )
        self.addCleanup(temp.cleanup)
        result = audit.audit_case(row, [record], read_dwi=lambda _: dwi, read_mask=lambda _: mask)
        self.assertEqual(result.audit_status, "failed")
        self.assertEqual(result.derived_metrics["dwi_nonfinite_voxel_count"], 1)
        self.assertIsNone(result.derived_metrics["dwi_all_mean"])
        self.assertIsNone(result.derived_metrics["lesion_mean"])

    def test_lesion_intensity_percentiles_use_finite_masked_dwi_values(self):
        dwi_array = np.array([[[10.0, 20.0], [30.0, 40.0]]])
        mask_array = np.array([[[1, 0], [1, 1]]], dtype=np.uint8)
        result = audit.mask_metrics(mask_array, self._image(dwi_array))

        lesion = dwi_array[mask_array > 0]
        self.assertEqual(result["lesion_finite_voxel_count"], 3)
        for name, percentile in (("p5", 5), ("p25", 25), ("p95", 95)):
            self.assertEqual(result[f"lesion_{name}"], float(np.percentile(lesion, percentile)))
        self.assertEqual(result["lesion_median"], 30.0)
        self.assertEqual(result["lesion_mean"], float(lesion.mean()))

    def test_audit_case_keeps_reader_output_rescale_semantics(self):
        temp, row, record, dwi, mask = self._case(
            np.array([[[1.0, 2.0], [3.0, 4.0]]]),
            np.array([[[1, 0], [0, 0]]], dtype=np.uint8),
        )
        self.addCleanup(temp.cleanup)
        result = audit.audit_case(row, [record], read_dwi=lambda _: dwi, read_mask=lambda _: mask)

        self.assertEqual(result.derived_metrics["intensity_value_semantics"], "reader_output_unverified_rescale")
        self.assertIsNone(result.derived_metrics["rescale_applied"])
        self.assertEqual(result.derived_metrics["intensity_metadata_source"], "public DICOM rescale tags if available")

    def test_geometry_mismatch_reports_dwi_intensity_but_skips_mask_and_lesion_metrics(self):
        temp, row, record, dwi, mask = self._case(
            np.array([[[1.0, 2.0], [3.0, 4.0]]]),
            np.ones((1, 3, 2), dtype=np.uint8),
        )
        self.addCleanup(temp.cleanup)
        result = audit.audit_case(row, [record], read_dwi=lambda _: dwi, read_mask=lambda _: mask)

        self.assertEqual(result.geometry_status, "mismatch")
        self.assertEqual(result.derived_metrics["dwi_finite_voxel_count"], 4)
        self.assertIsNone(result.derived_metrics["mask_voxel_count"])
        self.assertIsNone(result.derived_metrics["lesion_mean"])
        self.assertEqual(result.derived_metrics["lesion_intensity_status"], "skipped_due_to_geometry_mismatch")

    def test_stats_do_not_mutate_mask_array_or_native_image_metadata(self):
        mask_array = np.array([[[0, 4], [1, 0]]], dtype=np.uint8)
        direction = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        image = self._image(mask_array, spacing=(2.0, 3.0, 4.0), origin=(5.0, 6.0, 7.0), direction=direction)
        before_array = mask_array.copy()
        before_metadata = (image.GetSize(), image.GetSpacing(), image.GetOrigin(), image.GetDirection())

        audit.mask_metrics(mask_array, image)

        self.assertTrue(np.array_equal(mask_array, before_array))
        self.assertEqual((image.GetSize(), image.GetSpacing(), image.GetOrigin(), image.GetDirection()), before_metadata)


class CI1Task7DatasetAuditTest(unittest.TestCase):
    def _case(self, *, case_id="patient_D1", source="", mask=""):
        return CaseRecord(
            case_id=case_id,
            patient_id="patient",
            timepoint="D1",
            pairing_status="confirmed",
            pairing_source="prepare_ci1_dwi_dataset.build_index",
            build_index_match_status="matched",
            dwi_source_path=source,
            mask_path=mask,
            link_status="linked",
            metadata_modality_consistency="match",
            dwi_series_uid="1.2.3.4",
            dwi_series_path=source,
            audit_status="passed",
            read_status="readable",
            error_code="",
            error_message="",
            geometry_status="match",
            geometry_mismatch_fields=(),
            dwi_mask_comparison_status="match",
            derived_metrics={"mask_voxel_count": 4, "xy_z_preserved": True},
        )

    def _pipeline_patches(self, root, dicom_dir, record, rows, cases):
        metadata = {
            "values": {"SeriesDescription": "DWI", "Modality": "MR"},
            "field_status": {"SeriesDescription": "present", "Modality": "present"},
            "per_file": {},
            "errors": {},
        }
        classification = {
            "modality_class": "dwi",
            "classification_source": ["dicom:SeriesDescription"],
            "classification_confidence": "medium",
            "classification_evidence": ["SeriesDescription=DWI"],
        }
        return (
            mock.patch.object(audit.prepare_ci1_dwi_dataset, "build_index", return_value=rows),
            mock.patch.object(audit, "discover_dicom_series", return_value=[record]),
            mock.patch.object(audit, "extract_series_metadata", return_value=metadata),
            mock.patch.object(audit, "classify_modality", return_value=classification),
            mock.patch.object(audit, "audit_confirmed_cases", return_value=cases),
        )

    def test_audit_dataset_runs_pipeline_once_and_writes_exact_deterministic_reports(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name) / "ci1"
        dicom_dir = root / "patient" / "D1-DWI"
        dicom_dir.mkdir(parents=True)
        member = dicom_dir / "member.dcm"
        member.write_text("synthetic", encoding="utf-8")
        output_dir = Path(temp.name) / "reports"
        rows = [
            {
                "patient": "patient",
                "timepoint": "D1",
                "segmentation_path": str(root / "mask.nii.gz"),
                "dicom_dir": str(dicom_dir),
                "match_status": "matched",
            },
            {
                "patient": "unmatched",
                "timepoint": "D2",
                "segmentation_path": str(root / "unmatched.nii.gz"),
                "dicom_dir": "",
                "match_status": "missing_dicom",
            },
        ]
        record = SeriesRecord(
            series_directory=str(dicom_dir), gdcm_series_id="series-b",
            series_instance_uid="1.2.3.4", relative_file_paths=("member.dcm",),
            file_count=1, read_status="readable", uid_status="consistent",
        )
        case = self._case(source=str(dicom_dir), mask=str(root / "mask.nii.gz"))
        patches = self._pipeline_patches(root, dicom_dir, record, rows, [case])

        with patches[0] as build_index, patches[1] as discover, patches[2] as extract, patches[3] as classify, patches[4] as confirmed:
            def verify_confirmed_input(actual_rows, records_by_dir):
                self.assertEqual(actual_rows, rows)
                self.assertIn(str(dicom_dir.resolve()), records_by_dir)
                return [case]

            confirmed.side_effect = verify_confirmed_input
            summary = audit.audit_dataset(root, output_dir)

        self.assertEqual(build_index.call_count, 1)
        build_index.assert_called_once_with(root.resolve())
        discover.assert_called_once_with(dicom_dir.resolve())
        self.assertEqual(extract.call_count, 1)
        self.assertEqual(tuple(extract.call_args.args[0]), (member.resolve(),))
        self.assertTrue(callable(extract.call_args.args[1]))
        classify.assert_called_once_with({"SeriesDescription": "DWI", "Modality": "MR"})
        confirmed.assert_called_once()
        self.assertEqual(set(output_dir.iterdir()), {output_dir / name for name in RESULT_FILES})
        with (output_dir / RESULT_FILES[0]).open(newline="", encoding="utf-8") as handle:
            series_rows = list(csv.reader(handle))
        with (output_dir / RESULT_FILES[1]).open(newline="", encoding="utf-8") as handle:
            case_rows = list(csv.reader(handle))
        self.assertEqual(series_rows[0], list(SERIES_COLUMNS))
        self.assertEqual(case_rows[0], list(CASE_COLUMNS))
        self.assertEqual(len(series_rows), 2)
        self.assertEqual(len(case_rows), 2)
        self.assertEqual(case_rows[1][0], "patient_D1")
        self.assertEqual(summary["result_files"], list(RESULT_FILES))
        self.assertEqual(summary["counts"], {
            "build_index_total": 2,
            "build_index_matched": 1,
            "case_rows": 1,
            "series_rows": 1,
        })
        self.assertEqual(summary["inputs"]["target_spacing_xy_mm"], list(DEFAULT_TARGET_SPACING_XY_MM))
        self.assertTrue(summary["protocol"]["read_only"])
        with (output_dir / RESULT_FILES[2]).open(encoding="utf-8") as handle:
            self.assertEqual(json.load(handle), summary)

    def test_audit_dataset_allows_empty_or_absent_output_but_rejects_input_containment(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name) / "ci1"
        root.mkdir()
        for unsafe in (root, root / "reports"):
            with self.assertRaises(ValueError):
                audit.audit_dataset(root, unsafe)
        for output_dir in (Path(temp.name) / "absent", Path(temp.name) / "empty"):
            if output_dir.name == "empty":
                output_dir.mkdir()
            with mock.patch.object(audit.prepare_ci1_dwi_dataset, "build_index", return_value=[]):
                audit.audit_dataset(root, output_dir)
            self.assertEqual({path.name for path in output_dir.iterdir()}, set(RESULT_FILES))

    def test_audit_dataset_sanitizes_surrogates_in_nested_failures_and_rows(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name) / "ci1"
        dicom_dir = root / "patient" / "D1-DWI"
        dicom_dir.mkdir(parents=True)
        member = dicom_dir / "member.dcm"
        member.write_text("synthetic", encoding="utf-8")
        output_dir = Path(temp.name) / "reports"
        rows = [{
            "patient": "patient",
            "timepoint": "D1",
            "segmentation_path": str(root / "mask.nii.gz"),
            "dicom_dir": str(dicom_dir),
            "match_status": "matched",
        }]
        record = SeriesRecord(
            series_directory=str(dicom_dir), gdcm_series_id="series-b",
            series_instance_uid="1.2.3.4", relative_file_paths=("member.dcm",),
            file_count=1, read_status="failed", uid_status="missing",
            error_code="uid_missing", error_message="bad\udcff",
        )
        case = replace(
            self._case(source=str(dicom_dir), mask=str(root / "mask.nii.gz"), case_id="case\udcff"),
            audit_status="failed", error_code="audit_failed", error_message="bad\udcff",
        )
        patches = self._pipeline_patches(root, dicom_dir, record, rows, [case])
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            summary = audit.audit_dataset(root, output_dir)

        self.assertEqual(summary["failures"][0]["message"], "bad\\udcff")
        with (output_dir / RESULT_FILES[0]).open(newline="", encoding="utf-8") as handle:
            series_rows = list(csv.reader(handle))
        self.assertIn("bad\\udcff", series_rows[1])
        with (output_dir / RESULT_FILES[1]).open(newline="", encoding="utf-8") as handle:
            case_rows = list(csv.reader(handle))
        self.assertEqual(case_rows[1][0], "case\\udcff")
        with (output_dir / RESULT_FILES[2]).open(encoding="utf-8") as handle:
            self.assertEqual(json.load(handle), summary)

    def test_audit_dataset_is_atomic_when_staged_writer_fails(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name) / "ci1"
        root.mkdir()
        output_dir = Path(temp.name) / "reports"
        with mock.patch.object(audit.prepare_ci1_dwi_dataset, "build_index", return_value=[]), \
                mock.patch.object(audit, "_write_csv", side_effect=OSError("synthetic write failure")):
            with self.assertRaises(OSError):
                audit.audit_dataset(root, output_dir)
        self.assertFalse(any(output_dir.glob("*")) if output_dir.exists() else False)
        self.assertFalse(any(path.name.startswith("ci1_audit_") for path in output_dir.parent.iterdir()))

    def test_cli_help_success_and_invalid_unpaired_target(self):
        with self.assertRaises(SystemExit) as help_exit:
            audit.main(["--help"])
        self.assertEqual(help_exit.exception.code, 0)
        self.assertEqual(
            audit.main([
                "--ci1-root", "synthetic-root", "--output-dir", "synthetic-output",
                "--target-spacing-x-mm", "0.5",
            ]),
            2,
        )

    def test_main_returns_zero_and_finalizes_exact_reports_for_empty_synthetic_root(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name) / "ci1"
        root.mkdir()
        output_dir = Path(temp.name) / "reports"
        with mock.patch.object(audit.prepare_ci1_dwi_dataset, "build_index", return_value=[]):
            status = audit.main(["--ci1-root", str(root), "--output-dir", str(output_dir)])
        self.assertEqual(status, 0)
        self.assertEqual({path.name for path in output_dir.iterdir()}, set(RESULT_FILES))


if __name__ == "__main__":
    unittest.main()
