from __future__ import annotations

import csv

import pytest
import SimpleITK as sitk

from standalone_nnunet2d.tools.build_multimodal_dataset import (
    AuditRow,
    build_dataset,
    build_dataset_json,
    normalize_modalities,
    resample_to_reference,
)


def test_dwi_adc_dataset_json_declares_ordered_two_channels() -> None:
    assert build_dataset_json(("DWI", "ADC"), 95) == {
        "channel_names": {"0": "DWI", "1": "ADC"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": 95,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }


def test_dwi_adc_flair_dataset_json_declares_ordered_three_channels() -> None:
    dataset = build_dataset_json(("DWI", "ADC", "FLAIR"), 95)
    assert dataset["channel_names"] == {"0": "DWI", "1": "ADC", "2": "FLAIR"}


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (("dwi", "adc"), ("DWI", "ADC")),
        (("DWI", "ADC", "FLAIR"), ("DWI", "ADC", "FLAIR")),
    ],
)
def test_normalize_modalities_accepts_only_supported_ordered_input(
    requested: tuple[str, ...], expected: tuple[str, ...]
) -> None:
    assert normalize_modalities(requested) == expected


@pytest.mark.parametrize(
    "requested",
    [("ADC", "DWI"), ("DWI",), ("DWI", "FLAIR"), ("DWI", "ADC", "T1")],
)
def test_normalize_modalities_rejects_ambiguous_or_unsupported_inputs(
    requested: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError):
        normalize_modalities(requested)


def test_build_rejects_failed_audit_before_creating_output(tmp_path) -> None:
    output_root = tmp_path / "Dataset502"
    with pytest.raises(RuntimeError, match="audit"):
        build_dataset(
            tmp_path / "manifest.tsv",
            output_root,
            ("DWI", "ADC"),
            [AuditRow("case001", "", "", "", "ADC", "", "failed", "missing")],
        )
    assert not output_root.exists()


def test_resample_to_reference_preserves_linear_interpolation_as_float32() -> None:
    source = sitk.GetImageFromArray([[0, 10], [20, 30]])
    source = sitk.Cast(source, sitk.sitkInt16)
    reference = sitk.Image([4, 4], sitk.sitkInt16)
    reference.SetSpacing((0.5, 0.5))

    result = resample_to_reference(source, reference)

    assert result.GetPixelID() == sitk.sitkFloat32


def test_build_failure_does_not_create_final_output_root(tmp_path) -> None:
    manifest_path = tmp_path / "conversion_manifest.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "patient",
                "timepoint",
                "source_dicom_dir",
                "source_label",
                "output_image",
                "output_label",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for index in range(95):
            writer.writerow(
                {
                    "case_id": f"case{index:03d}",
                    "patient": "p",
                    "timepoint": "D1",
                    "source_dicom_dir": str(tmp_path),
                    "source_label": str(tmp_path / "missing_label.nii.gz"),
                    "output_image": str(tmp_path / "missing_dwi.nii.gz"),
                    "output_label": str(tmp_path / "missing_label.nii.gz"),
                }
            )

    output_root = tmp_path / "Dataset502"
    rows = [
        AuditRow(f"case{index:03d}", "", "", str(tmp_path), "ADC", "ADC", "passed", "")
        for index in range(95)
    ]
    with pytest.raises(RuntimeError):
        build_dataset(manifest_path, output_root, ("DWI", "ADC"), rows)
    assert not output_root.exists()
