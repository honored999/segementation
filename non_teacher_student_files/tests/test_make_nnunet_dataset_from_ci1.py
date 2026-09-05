import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from make_nnunet_dataset_from_CI1 import (
    build_dataset_json,
    CaseCandidate,
    convert_cases,
    find_case_candidates,
    is_forbidden_label_path,
    is_nifti_path,
    pick_best_dwi_image,
    pick_best_label,
    save_binary_label,
)


class MakeNnunetDatasetFromCI1Test(unittest.TestCase):
    def test_rejects_forbidden_label_paths(self):
        self.assertTrue(is_forbidden_label_path(Path("case_deepisles_pred.nii.gz")))
        self.assertTrue(is_forbidden_label_path(Path("xy_fixed/output_label.nii.gz")))
        self.assertFalse(is_forbidden_label_path(Path("patient-D1-DWI.nii.gz")))

    def test_identifies_nii_and_niigz_files(self):
        self.assertTrue(is_nifti_path(Path("a.nii")))
        self.assertTrue(is_nifti_path(Path("a.nii.gz")))
        self.assertFalse(is_nifti_path(Path("a.png")))

    def test_picks_dwi_image_before_other_modalities(self):
        candidates = [
            Path("patient-D1-FLAIR.nii.gz"),
            Path("patient-D1-b800.nii.gz"),
            Path("patient-D1-ADC.nii.gz"),
        ]

        self.assertEqual(pick_best_dwi_image(candidates), Path("patient-D1-b800.nii.gz"))

    def test_picks_readable_dwi_image_over_unreadable_candidate(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            bad_path = root / "patient-D1-DWI..nii.gz"
            good_path = root / "patient-D1-b800.nii.gz"
            bad_path.write_text("not a nifti", encoding="utf-8")
            sitk.WriteImage(
                sitk.GetImageFromArray(np.zeros((1, 2, 3), dtype=np.float32)),
                str(good_path),
            )

            self.assertEqual(pick_best_dwi_image([bad_path, good_path]), good_path)

    def test_picks_label_keyword_before_dwi_fallback(self):
        candidates = [
            Path("patient-D1-DWI.nii.gz"),
            Path("patient-D1-lesion_mask.nii.gz"),
        ]

        self.assertEqual(pick_best_label(candidates), Path("patient-D1-lesion_mask.nii.gz"))

    def test_save_binary_label_converts_all_nonzero_values_to_one(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "label.nii.gz"
            output_path = Path(tmp_dir) / "case001.nii.gz"
            image = sitk.GetImageFromArray(np.asarray([[[0, 2], [5, 0]]], dtype=np.int16))
            sitk.WriteImage(image, str(input_path))

            save_binary_label(input_path, output_path)

            saved = sitk.GetArrayFromImage(sitk.ReadImage(str(output_path)))
            self.assertEqual(saved.dtype, np.uint8)
            np.testing.assert_array_equal(saved, np.asarray([[[0, 1], [1, 0]]], dtype=np.uint8))

    def test_find_case_candidates_uses_dicom_dir_and_dwi_label(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            patient_dir = root / "CI-1" / "patientA"
            dicom_dir = patient_dir / "patientA_D1-123"
            dicom_dir.mkdir(parents=True)
            (dicom_dir / "slice001.dcm").write_bytes(b"fake dicom for indexing")
            label_path = patient_dir / "patientA-D1-DWI.nii.gz"
            sitk.WriteImage(
                sitk.GetImageFromArray(np.ones((1, 2, 3), dtype=np.uint8)),
                str(label_path),
            )

            candidates = find_case_candidates(root / "CI-1")

            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0].dicom_dir, dicom_dir)
            self.assertEqual(candidates[0].label_path, label_path)

    def test_convert_cases_skips_unreadable_dicom_series_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            bad_dicom_dir = root / "bad_dicom"
            bad_dicom_dir.mkdir()
            label_path = root / "label.nii.gz"
            output_root = root / "Dataset501_StrokeLesion"
            sitk.WriteImage(
                sitk.GetImageFromArray(np.ones((1, 2, 3), dtype=np.uint8)),
                str(label_path),
            )

            written = convert_cases(
                [
                    CaseCandidate(
                        patient="patient",
                        timepoint="D1",
                        dicom_dir=bad_dicom_dir,
                        label_path=label_path,
                    )
                ],
                output_root=output_root,
            )

            self.assertEqual(written, [])

    def test_build_dataset_json_uses_nnunet_v2_keys(self):
        data = build_dataset_json(num_training=3)

        self.assertEqual(data["channel_names"], {"0": "DWI"})
        self.assertEqual(data["labels"], {"background": 0, "lesion": 1})
        self.assertEqual(data["numTraining"], 3)
        self.assertEqual(data["file_ending"], ".nii.gz")
        self.assertEqual(data["overwrite_image_reader_writer"], "SimpleITKIO")
        json.dumps(data)


if __name__ == "__main__":
    unittest.main()
