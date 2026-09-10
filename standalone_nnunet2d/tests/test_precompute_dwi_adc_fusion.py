from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import standalone_nnunet2d.data.dataset as dataset_module
import standalone_nnunet2d.tools.precompute_dwi_adc_fusion as precompute_module
from standalone_nnunet2d.data.fusion import build_dwi_adc_fusion_channel
from standalone_nnunet2d.data.nifti_io import NiftiVolume, read_nifti, write_nifti
from standalone_nnunet2d.tools.precompute_dwi_adc_fusion import build_dataset


IDENTITY_DIRECTION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def _volume(
    array: np.ndarray,
    *,
    spacing: tuple[float, float, float] = (1.0, 1.0, 4.0),
) -> NiftiVolume:
    return NiftiVolume(
        np.asarray(array), spacing, (2.0, 3.0, 4.0), IDENTITY_DIRECTION
    )


def _write_source(root: Path, *, adc_spacing: tuple[float, float, float] = (1.0, 1.0, 4.0)) -> None:
    (root / "imagesTr").mkdir(parents=True)
    (root / "labelsTr").mkdir()
    (root / "dataset.json").write_text(
        json.dumps(
            {
                "channel_names": {"0": "DWI", "1": "ADC"},
                "labels": {"background": 0, "lesion": 1},
                "numTraining": 1,
                "file_ending": ".nii.gz",
            }
        ),
        encoding="utf-8",
    )
    dwi = np.array([[[0.0, 1.0, 3.0], [0.0, 2.0, 3.0]]], dtype=np.float32)
    adc = np.array([[[0.0, 4.0, 2.0], [0.0, 3.0, 2.0]]], dtype=np.float32)
    label = np.array([[[0, 1, 0], [0, 0, 1]]], dtype=np.uint8)
    write_nifti(root / "imagesTr" / "case001_0000.nii.gz", _volume(dwi))
    write_nifti(
        root / "imagesTr" / "case001_0001.nii.gz",
        _volume(adc, spacing=adc_spacing),
    )
    write_nifti(root / "labelsTr" / "case001.nii.gz", _volume(label))


def test_build_dwi_adc_fusion_channel_matches_masked_formula() -> None:
    dwi = np.array([[[0.0, 1.0, 3.0]]], dtype=np.float32)
    adc = np.array([[[0.0, 4.0, 2.0]]], dtype=np.float32)

    fusion = build_dwi_adc_fusion_channel(dwi, adc)

    np.testing.assert_array_equal(fusion, np.array([[[0.0, 0.0, 2.0]]], dtype=np.float32))
    assert fusion.dtype == np.float32


def test_online_fusion_preparation_uses_shared_pure_builder(monkeypatch) -> None:
    calls = 0
    real_builder = build_dwi_adc_fusion_channel

    def tracked_builder(dwi: np.ndarray, adc: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return real_builder(dwi, adc)

    monkeypatch.setattr(dataset_module, "build_dwi_adc_fusion_channel", tracked_builder)
    dwi = _volume(np.array([[[0.0, 1.0, 3.0]]], dtype=np.float32))
    adc = _volume(np.array([[[0.0, 4.0, 2.0]]], dtype=np.float32))

    prepared = dataset_module.prepare_dwi_adc_fusion_images(
        dwi, adc, target_spacing_xy=(1.0, 1.0)
    )

    assert calls == 1
    np.testing.assert_array_equal(prepared.fusion, real_builder(dwi.array, adc.array))


def test_build_dataset_copies_sources_and_writes_float32_fusion_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "Dataset502_StrokeLesion_DWI_ADC"
    output = tmp_path / "Dataset505_StrokeLesion_DWI_ADC_Fusion"
    _write_source(source)
    monkeypatch.setattr(precompute_module, "EXPECTED_NUM_TRAINING", 1)

    build_dataset(source, output)

    for relative in (
        "imagesTr/case001_0000.nii.gz",
        "imagesTr/case001_0001.nii.gz",
        "labelsTr/case001.nii.gz",
    ):
        assert (output / relative).read_bytes() == (source / relative).read_bytes()
    dwi = read_nifti(output / "imagesTr" / "case001_0000.nii.gz")
    fusion = read_nifti(output / "imagesTr" / "case001_0002.nii.gz")
    assert fusion.array.dtype == np.float32
    assert fusion.array[0, 0, 0] == 0.0
    np.testing.assert_array_equal(
        fusion.array, build_dwi_adc_fusion_channel(dwi.array, read_nifti(output / "imagesTr" / "case001_0001.nii.gz").array)
    )
    assert fusion.spacing_xyz == dwi.spacing_xyz
    assert fusion.origin_xyz == dwi.origin_xyz
    assert fusion.direction == dwi.direction

    dataset_json = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
    assert dataset_json == {
        "channel_names": {"0": "DWI", "1": "ADC", "2": "noNorm"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": 1,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["source_dataset"] == "Dataset502_StrokeLesion_DWI_ADC"
    assert provenance["channels"]["2"] == {
        "semantic_name": "DWI_ADC_FUSION",
        "nnunet_channel_name": "noNorm",
        "recipe": "DWI_01 + (1 - ADC_01)",
        "valid_region": "finite and nonzero intersection",
        "output_range": [0, 2],
        "normalization_during_nnunet_preprocessing": "none",
    }


def test_build_dataset_rejects_geometry_mismatch_without_partial_output(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "Dataset502_StrokeLesion_DWI_ADC"
    output = tmp_path / "Dataset505_StrokeLesion_DWI_ADC_Fusion"
    _write_source(source, adc_spacing=(1.1, 1.0, 4.0))
    monkeypatch.setattr(precompute_module, "EXPECTED_NUM_TRAINING", 1)

    with pytest.raises(ValueError, match="geometry mismatch"):
        build_dataset(source, output)

    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}_building_*"))


def test_build_dataset_rejects_existing_output_without_modifying_it(tmp_path: Path) -> None:
    source = tmp_path / "Dataset502_StrokeLesion_DWI_ADC"
    output = tmp_path / "Dataset505_StrokeLesion_DWI_ADC_Fusion"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("unchanged", encoding="utf-8")

    with pytest.raises(FileExistsError, match="must not already exist"):
        build_dataset(source, output)

    assert marker.read_text(encoding="utf-8") == "unchanged"


def test_build_dataset_requires_dataset505_training_count_of_95(tmp_path: Path) -> None:
    source = tmp_path / "Dataset502_StrokeLesion_DWI_ADC"
    output = tmp_path / "Dataset505_StrokeLesion_DWI_ADC_Fusion"
    _write_source(source)

    with pytest.raises(ValueError, match="numTraining must be 95"):
        build_dataset(source, output)

    assert not output.exists()


def test_build_dataset_rejects_orphan_adc_and_label_cases(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "Dataset502_StrokeLesion_DWI_ADC"
    output = tmp_path / "Dataset505_StrokeLesion_DWI_ADC_Fusion"
    _write_source(source)
    monkeypatch.setattr(precompute_module, "EXPECTED_NUM_TRAINING", 1)
    orphan = np.ones((1, 2, 2), dtype=np.float32)
    write_nifti(source / "imagesTr" / "case999_0001.nii.gz", _volume(orphan))
    write_nifti(source / "labelsTr" / "case999.nii.gz", _volume(orphan))

    with pytest.raises(ValueError, match="case ID sets"):
        build_dataset(source, output)

    assert not output.exists()
