from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from standalone_nnunet2d.data import dataset as dataset_module
from standalone_nnunet2d.data.dataset import StrokeSliceDataset, resolve_input_channels
from standalone_nnunet2d.data.input_mode import InputMode
from standalone_nnunet2d.data.nifti_io import NiftiVolume, write_nifti
from standalone_nnunet2d.data.preprocessing import z_score_normalize
from standalone_nnunet2d.data.symmetry_alignment import bilateral_difference
from standalone_nnunet2d.training.formal_dataset import FormalPatchDataset
from standalone_nnunet2d.training.batch_sampler import PatchRequest


IDENTITY_DIRECTION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def _symmetric_dwi(shape: tuple[int, int, int] = (3, 33, 33)) -> np.ndarray:
    z, y, x = np.indices(shape)
    del z
    return (((x - 16) / 10.0) ** 2 + ((y - 16) / 5.0) ** 2 <= 1.0).astype(np.float32)


def _volume(array: np.ndarray) -> NiftiVolume:
    return NiftiVolume(
        array=array.astype(np.float32, copy=False),
        spacing_xyz=(1.0, 1.0, 4.0),
        origin_xyz=(0.0, 0.0, 0.0),
        direction=IDENTITY_DIRECTION,
    )


def _write_single_dwi_case(root: Path, case_id: str, image: np.ndarray, label: np.ndarray) -> Path:
    (root / "imagesTr").mkdir(parents=True)
    (root / "labelsTr").mkdir()
    (root / "dataset.json").write_text('{"channel_names": {"0": "DWI"}}', encoding="utf-8")
    image_path = root / "imagesTr" / f"{case_id}_0000.nii.gz"
    write_nifti(image_path, _volume(image))
    write_nifti(root / "labelsTr" / f"{case_id}.nii.gz", NiftiVolume(label.astype(np.int16), (1.0, 1.0, 4.0), (0.0, 0.0, 0.0), IDENTITY_DIRECTION))
    return image_path


def _case_id(monkeypatch) -> str:
    case_id = "case_synthetic"
    monkeypatch.setattr(dataset_module, "load_fold_cases", lambda fold, split: (case_id,))
    return case_id


def test_dwi_adc_bilateral_dataset_builds_four_ordered_channels(monkeypatch, tmp_path: Path) -> None:
    case_id = _case_id(monkeypatch)
    dwi = _symmetric_dwi()
    adc = 2.0 * dwi + np.indices(dwi.shape)[2].astype(np.float32) / 20.0
    label = np.zeros_like(dwi, dtype=np.int16)
    (tmp_path / "imagesTr").mkdir(parents=True)
    (tmp_path / "labelsTr").mkdir()
    (tmp_path / "dataset.json").write_text(
        json.dumps({"channel_names": {"0": "DWI", "1": "ADC"}}), encoding="utf-8"
    )
    write_nifti(tmp_path / "imagesTr" / f"{case_id}_0000.nii.gz", _volume(dwi))
    write_nifti(tmp_path / "imagesTr" / f"{case_id}_0001.nii.gz", _volume(adc))
    write_nifti(tmp_path / "labelsTr" / f"{case_id}.nii.gz", NiftiVolume(label, (1.0, 1.0, 4.0), (0.0, 0.0, 0.0), IDENTITY_DIRECTION))

    dataset = StrokeSliceDataset(
        tmp_path,
        fold=0,
        split="val",
        case_ids=(case_id,),
        target_spacing_xy=(1.0, 1.0),
        input_mode=InputMode.DWI_ADC_BILATERAL,
    )
    channels, returned_label = dataset.load_case(case_id)

    assert dataset.input_channels == 4
    assert channels.shape == (4, 3, 33, 33)
    assert returned_label.shape == (3, 33, 33)


@pytest.mark.parametrize(
    "declared_channels",
    [
        {"0": "ADC", "1": "DWI"},
        {"0": "DWI", "1": "FLAIR"},
        {"0": "DWI"},
        {"0": "DWI", "1": "ADC", "2": "FLAIR"},
    ],
)
def test_dwi_adc_bilateral_requires_exact_dwi_adc_declaration(
    monkeypatch, tmp_path: Path, declared_channels: dict[str, str]
) -> None:
    case_id = _case_id(monkeypatch)
    (tmp_path / "imagesTr").mkdir(parents=True)
    (tmp_path / "labelsTr").mkdir()
    (tmp_path / "dataset.json").write_text(
        json.dumps({"channel_names": declared_channels}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="exact channel declaration"):
        StrokeSliceDataset(
            tmp_path,
            fold=0,
            split="val",
            case_ids=(case_id,),
            input_mode=InputMode.DWI_ADC_BILATERAL,
        )


def test_dwi_adc_bilateral_stack_has_separate_normalization_and_signed_lr_diffs(
    monkeypatch, tmp_path: Path
) -> None:
    case_id = _case_id(monkeypatch)
    dwi = _symmetric_dwi()
    dwi[:, 16, 22] = 5.0
    adc = 3.0 * _symmetric_dwi()
    adc[:, 16, 22] = 11.0
    label = np.zeros_like(dwi, dtype=np.int16)
    (tmp_path / "imagesTr").mkdir(parents=True)
    (tmp_path / "labelsTr").mkdir()
    (tmp_path / "dataset.json").write_text(
        json.dumps({"channel_names": {"0": "DWI", "1": "ADC"}}), encoding="utf-8"
    )
    dwi_volume = _volume(dwi)
    adc_volume = _volume(adc)
    write_nifti(tmp_path / "imagesTr" / f"{case_id}_0000.nii.gz", dwi_volume)
    write_nifti(tmp_path / "imagesTr" / f"{case_id}_0001.nii.gz", adc_volume)
    write_nifti(tmp_path / "labelsTr" / f"{case_id}.nii.gz", NiftiVolume(label, (1.0, 1.0, 4.0), (0.0, 0.0, 0.0), IDENTITY_DIRECTION))

    monkeypatch.setattr(dataset_module, "estimate_quasi_symmetric_alignment", lambda volume: None)
    monkeypatch.setattr(
        dataset_module,
        "apply_quasi_symmetric_alignment",
        lambda volume, estimate, *, is_segmentation: volume,
    )
    dataset = StrokeSliceDataset(
        tmp_path,
        fold=0,
        split="val",
        case_ids=(case_id,),
        target_spacing_xy=(1.0, 1.0),
        input_mode=InputMode.DWI_ADC_BILATERAL,
    )

    channels, _ = dataset.load_case(case_id)
    normalized_dwi = z_score_normalize(dwi)
    normalized_adc = z_score_normalize(adc)

    np.testing.assert_allclose(channels[0], normalized_dwi, atol=1e-6)
    np.testing.assert_allclose(channels[1], normalized_adc, atol=1e-6)
    np.testing.assert_allclose(
        channels[2], bilateral_difference(NiftiVolume(normalized_dwi, (1.0, 1.0, 4.0), (0.0, 0.0, 0.0), IDENTITY_DIRECTION), mode="signed"),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        channels[3], bilateral_difference(NiftiVolume(normalized_adc, (1.0, 1.0, 4.0), (0.0, 0.0, 0.0), IDENTITY_DIRECTION), mode="signed"),
        atol=1e-6,
    )
    assert channels[2, 1, 16, 22] == pytest.approx(-channels[2, 1, 16, 10])
    assert channels[3, 1, 16, 22] == pytest.approx(-channels[3, 1, 16, 10])
    assert channels[2, 1, 16, 22] != 0.0
    assert channels[3, 1, 16, 22] != 0.0


def test_dwi_adc_bilateral_symmetric_modalities_have_zero_signed_diffs(
    monkeypatch, tmp_path: Path
) -> None:
    case_id = _case_id(monkeypatch)
    dwi = _symmetric_dwi()
    adc = 7.0 * _symmetric_dwi() + 3.0
    label = np.zeros_like(dwi, dtype=np.int16)
    (tmp_path / "imagesTr").mkdir(parents=True)
    (tmp_path / "labelsTr").mkdir()
    (tmp_path / "dataset.json").write_text(
        json.dumps({"channel_names": {"0": "DWI", "1": "ADC"}}), encoding="utf-8"
    )
    write_nifti(tmp_path / "imagesTr" / f"{case_id}_0000.nii.gz", _volume(dwi))
    write_nifti(tmp_path / "imagesTr" / f"{case_id}_0001.nii.gz", _volume(adc))
    write_nifti(
        tmp_path / "labelsTr" / f"{case_id}.nii.gz",
        NiftiVolume(label, (1.0, 1.0, 4.0), (0.0, 0.0, 0.0), IDENTITY_DIRECTION),
    )
    monkeypatch.setattr(dataset_module, "estimate_quasi_symmetric_alignment", lambda volume: None)
    monkeypatch.setattr(
        dataset_module,
        "apply_quasi_symmetric_alignment",
        lambda volume, estimate, *, is_segmentation: volume,
    )

    dataset = StrokeSliceDataset(
        tmp_path,
        fold=0,
        split="val",
        case_ids=(case_id,),
        target_spacing_xy=(1.0, 1.0),
        input_mode=InputMode.DWI_ADC_BILATERAL,
    )
    channels, _ = dataset.load_case(case_id)

    assert channels.shape == (4, 3, 33, 33)
    np.testing.assert_allclose(channels[2], 0.0, atol=1e-7)
    np.testing.assert_allclose(channels[3], 0.0, atol=1e-7)


def test_dwi_adc_bilateral_rejects_adc_geometry_mismatch_against_label(
    monkeypatch, tmp_path: Path
) -> None:
    case_id = _case_id(monkeypatch)
    dwi = _volume(_symmetric_dwi())
    adc = NiftiVolume(
        _symmetric_dwi(),
        (2.0, 1.0, 4.0),
        (0.0, 0.0, 0.0),
        IDENTITY_DIRECTION,
    )
    label = NiftiVolume(
        np.zeros_like(dwi.array, dtype=np.int16),
        (1.0, 1.0, 4.0),
        (0.0, 0.0, 0.0),
        IDENTITY_DIRECTION,
    )
    (tmp_path / "imagesTr").mkdir(parents=True)
    (tmp_path / "labelsTr").mkdir()
    (tmp_path / "dataset.json").write_text(
        json.dumps({"channel_names": {"0": "DWI", "1": "ADC"}}), encoding="utf-8"
    )
    write_nifti(tmp_path / "imagesTr" / f"{case_id}_0000.nii.gz", dwi)
    write_nifti(tmp_path / "imagesTr" / f"{case_id}_0001.nii.gz", adc)
    write_nifti(tmp_path / "labelsTr" / f"{case_id}.nii.gz", label)

    dataset = StrokeSliceDataset(
        tmp_path,
        fold=0,
        split="val",
        case_ids=(case_id,),
        input_mode=InputMode.DWI_ADC_BILATERAL,
    )

    with pytest.raises(ValueError, match=f"case {case_id}.*channel 1.*spacing"):
        dataset.load_case(case_id)


def test_dwi_adc_bilateral_estimates_from_dwi_once_and_shares_estimate(
    monkeypatch, tmp_path: Path
) -> None:
    case_id = _case_id(monkeypatch)
    dwi = _symmetric_dwi()
    adc = 2.0 * _symmetric_dwi() + 1.0
    label = np.zeros_like(dwi, dtype=np.int16)
    (tmp_path / "imagesTr").mkdir(parents=True)
    (tmp_path / "labelsTr").mkdir()
    (tmp_path / "dataset.json").write_text(
        json.dumps({"channel_names": {"0": "DWI", "1": "ADC"}}), encoding="utf-8"
    )
    write_nifti(tmp_path / "imagesTr" / f"{case_id}_0000.nii.gz", _volume(dwi))
    write_nifti(tmp_path / "imagesTr" / f"{case_id}_0001.nii.gz", _volume(adc))
    write_nifti(
        tmp_path / "labelsTr" / f"{case_id}.nii.gz",
        NiftiVolume(label, (1.0, 1.0, 4.0), (0.0, 0.0, 0.0), IDENTITY_DIRECTION),
    )

    estimate = object()
    estimated_inputs: list[NiftiVolume] = []
    applications: list[tuple[NiftiVolume, object, bool]] = []

    def estimate_from(volume: NiftiVolume) -> object:
        estimated_inputs.append(volume)
        return estimate

    def apply_shared(volume: NiftiVolume, applied_estimate: object, *, is_segmentation: bool) -> NiftiVolume:
        applications.append((volume, applied_estimate, is_segmentation))
        return volume

    monkeypatch.setattr(dataset_module, "estimate_quasi_symmetric_alignment", estimate_from)
    monkeypatch.setattr(dataset_module, "apply_quasi_symmetric_alignment", apply_shared)
    dataset = StrokeSliceDataset(
        tmp_path,
        fold=0,
        split="val",
        case_ids=(case_id,),
        target_spacing_xy=(1.0, 1.0),
        input_mode=InputMode.DWI_ADC_BILATERAL,
    )

    dataset.load_case(case_id)

    assert len(estimated_inputs) == 1
    np.testing.assert_array_equal(estimated_inputs[0].array, dwi)
    assert len(applications) == 3
    np.testing.assert_array_equal(applications[0][0].array, dwi)
    np.testing.assert_array_equal(applications[1][0].array, adc)
    np.testing.assert_array_equal(applications[2][0].array, label)
    assert [item[1] for item in applications] == [estimate, estimate, estimate]
    assert [item[2] for item in applications] == [False, False, True]


def test_dwi_adc_bilateral_input_and_estimate_are_invariant_to_label_content(
    monkeypatch, tmp_path: Path
) -> None:
    case_id = _case_id(monkeypatch)
    dwi = _symmetric_dwi()
    dwi[:, 16, 22] = 4.0
    adc = 2.0 * _symmetric_dwi()
    adc[:, 16, 22] = 8.0
    first_label = np.zeros_like(dwi, dtype=np.int16)
    second_label = first_label.copy()
    second_label[:, 10, 10] = 1
    (tmp_path / "imagesTr").mkdir(parents=True)
    (tmp_path / "labelsTr").mkdir()
    (tmp_path / "dataset.json").write_text(
        json.dumps({"channel_names": {"0": "DWI", "1": "ADC"}}), encoding="utf-8"
    )
    write_nifti(tmp_path / "imagesTr" / f"{case_id}_0000.nii.gz", _volume(dwi))
    write_nifti(tmp_path / "imagesTr" / f"{case_id}_0001.nii.gz", _volume(adc))
    label_path = tmp_path / "labelsTr" / f"{case_id}.nii.gz"
    write_nifti(label_path, NiftiVolume(first_label, (1.0, 1.0, 4.0), (0.0, 0.0, 0.0), IDENTITY_DIRECTION))

    real_read = dataset_module.read_nifti
    active_label = [first_label]
    estimates: list[object] = []
    real_estimate = dataset_module.estimate_quasi_symmetric_alignment

    def read_with_active_label(path: Path) -> NiftiVolume:
        if "labelsTr" in str(path):
            return NiftiVolume(active_label[0], (1.0, 1.0, 4.0), (0.0, 0.0, 0.0), IDENTITY_DIRECTION)
        return real_read(path)

    def estimate_and_record(volume: NiftiVolume) -> object:
        estimate = real_estimate(volume)
        estimates.append(estimate)
        return estimate

    monkeypatch.setattr(dataset_module, "read_nifti", read_with_active_label)
    monkeypatch.setattr(dataset_module, "estimate_quasi_symmetric_alignment", estimate_and_record)
    dataset = StrokeSliceDataset(
        tmp_path,
        fold=0,
        split="val",
        case_ids=(case_id,),
        target_spacing_xy=(1.0, 1.0),
        input_mode=InputMode.DWI_ADC_BILATERAL,
    )

    first_channels, _ = dataset.load_case(case_id)
    active_label[0] = second_label
    second_channels, second_aligned_label = dataset.load_case(case_id)

    np.testing.assert_allclose(first_channels, second_channels, atol=1e-6)
    assert estimates[0] == estimates[1]
    assert not np.array_equal(first_label, second_aligned_label)
    assert label_path.is_file()


def test_legacy_dwi_bilateral_dataset_remains_c2_absolute_and_nonnegative(
    monkeypatch, tmp_path: Path
) -> None:
    case_id = _case_id(monkeypatch)
    image = _symmetric_dwi()
    image[:, 16, 22] = 5.0
    label = np.zeros_like(image, dtype=np.int16)
    _write_single_dwi_case(tmp_path, case_id, image, label)

    dataset = StrokeSliceDataset(
        tmp_path,
        fold=0,
        split="val",
        case_ids=(case_id,),
        target_spacing_xy=(1.0, 1.0),
        bilateral_asymmetry_channel=True,
    )
    channels, _ = dataset.load_case(case_id)

    assert dataset.input_channels == 2
    assert channels.shape == (2, 3, 33, 33)
    assert np.all(channels[1] >= 0.0)
    assert channels[1, :, 16, 22].min() > 0.0
    np.testing.assert_allclose(channels[1, :, 16, 22], channels[1, :, 16, 10])


def test_explicit_dwi_bilateral_mode_uses_legacy_c2_math_without_legacy_flag(
    monkeypatch, tmp_path: Path
) -> None:
    case_id = _case_id(monkeypatch)
    image = _symmetric_dwi()
    image[:, 16, 22] = 5.0
    label = np.zeros_like(image, dtype=np.int16)
    _write_single_dwi_case(tmp_path, case_id, image, label)

    explicit = StrokeSliceDataset(
        tmp_path,
        fold=0,
        split="val",
        case_ids=(case_id,),
        target_spacing_xy=(1.0, 1.0),
        input_mode=InputMode.DWI_BILATERAL,
        bilateral_asymmetry_channel=False,
    )
    assert explicit.input_channels == 2
    assert explicit.derived_input_channels == 1
    explicit_channels, _ = explicit.load_case(case_id)

    assert explicit_channels.shape == (2, 3, 33, 33)
    expected_difference = np.abs(explicit_channels[0] - explicit_channels[0][:, :, ::-1])
    np.testing.assert_allclose(explicit_channels[1], expected_difference, atol=1e-6)
    assert np.all(explicit_channels[1] >= 0.0)
    assert not (tmp_path / "imagesTr" / f"{case_id}_0001.nii.gz").exists()

    legacy = StrokeSliceDataset(
        tmp_path,
        fold=0,
        split="val",
        case_ids=(case_id,),
        target_spacing_xy=(1.0, 1.0),
        bilateral_asymmetry_channel=True,
    )
    legacy_channels, _ = legacy.load_case(case_id)
    np.testing.assert_allclose(explicit_channels, legacy_channels, atol=1e-6)


def test_default_opt_in_off_preserves_single_channel_case_loading(monkeypatch, tmp_path: Path) -> None:
    case_id = _case_id(monkeypatch)
    image = _symmetric_dwi()
    label = np.zeros_like(image, dtype=np.int16)
    _write_single_dwi_case(tmp_path, case_id, image, label)

    default = StrokeSliceDataset(tmp_path, fold=0, split="val", case_ids=(case_id,), target_spacing_xy=(1.0, 1.0))
    explicit_false = StrokeSliceDataset(
        tmp_path,
        fold=0,
        split="val",
        case_ids=(case_id,),
        target_spacing_xy=(1.0, 1.0),
        bilateral_asymmetry_channel=False,
    )

    default_image, default_label = default.load_case(case_id)
    explicit_image, explicit_label = explicit_false.load_case(case_id)

    assert default.input_channels == 1
    assert default_image.shape == (1, 3, 33, 33)
    np.testing.assert_array_equal(default_image, explicit_image)
    np.testing.assert_array_equal(default_label, explicit_label)


def test_default_three_physical_channels_keep_reported_and_returned_c3(
    monkeypatch, tmp_path: Path
) -> None:
    case_id = _case_id(monkeypatch)
    (tmp_path / "imagesTr").mkdir(parents=True)
    (tmp_path / "labelsTr").mkdir()
    (tmp_path / "dataset.json").write_text(
        json.dumps({"channel_names": {"0": "DWI", "1": "ADC", "2": "FLAIR"}}),
        encoding="utf-8",
    )
    shape = (3, 33, 33)
    for channel_index, offset in enumerate((0.0, 10.0, 20.0)):
        write_nifti(
            tmp_path / "imagesTr" / f"{case_id}_{channel_index:04d}.nii.gz",
            _volume(np.full(shape, offset, dtype=np.float32) + np.indices(shape)[2]),
        )
    write_nifti(
        tmp_path / "labelsTr" / f"{case_id}.nii.gz",
        NiftiVolume(
            np.zeros(shape, dtype=np.int16),
            (1.0, 1.0, 4.0),
            (0.0, 0.0, 0.0),
            IDENTITY_DIRECTION,
        ),
    )

    dataset = StrokeSliceDataset(
        tmp_path,
        fold=0,
        split="val",
        case_ids=(case_id,),
        target_spacing_xy=(1.0, 1.0),
        bilateral_asymmetry_channel=False,
    )
    channels, returned_label = dataset.load_case(case_id)
    image_tensor, label_tensor = dataset[0]

    assert dataset.physical_input_channels == 3
    assert dataset.derived_input_channels == 0
    assert dataset.input_channels == 3
    assert channels.shape == (3, 3, 33, 33)
    assert returned_label.shape == (3, 33, 33)
    assert tuple(image_tensor.shape) == (3, 33, 33)
    assert tuple(label_tensor.shape) == (33, 33)


def test_derived_channel_is_built_from_normalized_aligned_dwi_in_required_order(monkeypatch, tmp_path: Path) -> None:
    case_id = _case_id(monkeypatch)
    image = _symmetric_dwi()
    image[:, 16, 22] = 4.0
    label = np.zeros_like(image, dtype=np.int16)
    _write_single_dwi_case(tmp_path, case_id, image, label)

    from standalone_nnunet2d.data import symmetry_alignment

    events: list[str] = []
    real_resample = dataset_module.resample_inplane
    real_align = symmetry_alignment.align_case
    real_normalize = dataset_module.z_score_normalize
    real_derive = symmetry_alignment.build_bilateral_asymmetry_channels

    def tracked_resample(volume, *args, **kwargs):
        if not kwargs["is_segmentation"]:
            events.append("resample")
        return real_resample(volume, *args, **kwargs)

    def tracked_align(image_volume, label_volume):
        events.append("alignment")
        return real_align(image_volume, label_volume)

    def tracked_normalize(array):
        events.append("normalize")
        return real_normalize(array)

    def tracked_derive(aligned_normalized_image):
        events.append("difference")
        return real_derive(aligned_normalized_image)

    monkeypatch.setattr(dataset_module, "resample_inplane", tracked_resample)
    monkeypatch.setattr(dataset_module, "align_case", tracked_align)
    monkeypatch.setattr(dataset_module, "z_score_normalize", tracked_normalize)
    monkeypatch.setattr(dataset_module, "build_bilateral_asymmetry_channels", tracked_derive)

    dataset = StrokeSliceDataset(
        tmp_path,
        fold=0,
        split="val",
        case_ids=(case_id,),
        target_spacing_xy=(1.0, 1.0),
        bilateral_asymmetry_channel=True,
    )
    derived, _ = dataset.load_case(case_id)

    assert derived.shape == (2, 3, 33, 33)
    assert events == ["resample", "alignment", "normalize", "difference"]


def test_symmetric_normalized_dwi_has_zero_asymmetry_channel() -> None:
    from standalone_nnunet2d.data.symmetry_alignment import build_bilateral_asymmetry_channels

    channels = build_bilateral_asymmetry_channels(_volume(_symmetric_dwi()))

    assert channels.shape == (2, 3, 33, 33)
    np.testing.assert_allclose(channels[1], 0.0, atol=1e-7)


def test_unilateral_abnormality_and_its_mirror_have_expected_absolute_difference() -> None:
    from standalone_nnunet2d.data.symmetry_alignment import build_bilateral_asymmetry_channels

    image = _symmetric_dwi()
    image[:, 16, 22] = 5.0
    channels = build_bilateral_asymmetry_channels(_volume(image))
    expected = np.abs(image - image[:, :, ::-1])

    np.testing.assert_array_equal(channels[0], image)
    np.testing.assert_array_equal(channels[1], expected)
    assert np.all(channels[1, :, 16, 22] > 0.0)
    assert np.all(channels[1, :, 16, 10] > 0.0)


def test_derived_builder_preserves_alignment_direction_rejection_semantics() -> None:
    from standalone_nnunet2d.data.symmetry_alignment import build_bilateral_asymmetry_channels

    oblique = NiftiVolume(
        _symmetric_dwi(),
        (1.0, 1.0, 4.0),
        (0.0, 0.0, 0.0),
        (1.0, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )

    with pytest.raises(ValueError, match="unsupported direction"):
        build_bilateral_asymmetry_channels(oblique)


def test_derived_channel_uses_only_dwi_and_not_gt_contents(monkeypatch, tmp_path: Path) -> None:
    case_id = _case_id(monkeypatch)
    image = _symmetric_dwi()
    image[:, 16, 22] = 4.0
    first_label = np.zeros_like(image, dtype=np.int16)
    image_path = _write_single_dwi_case(tmp_path, case_id, image, first_label)
    original_bytes = image_path.read_bytes()
    second_label = first_label.copy()
    second_label[:, 10, 10] = 1

    original_read = dataset_module.read_nifti
    active_label = [NiftiVolume(first_label, (1.0, 1.0, 4.0), (0.0, 0.0, 0.0), IDENTITY_DIRECTION)]
    from standalone_nnunet2d.data import symmetry_alignment

    estimates = []
    real_align = symmetry_alignment.align_case

    def read_with_changed_gt(path: Path):
        return active_label[0] if "labelsTr" in str(path) else original_read(path)

    def align_and_record(image_volume, label_volume):
        aligned_image, aligned_label, estimate = real_align(image_volume, label_volume)
        estimates.append(estimate)
        return aligned_image, aligned_label, estimate

    monkeypatch.setattr(dataset_module, "read_nifti", read_with_changed_gt)
    monkeypatch.setattr(dataset_module, "align_case", align_and_record)
    dataset = StrokeSliceDataset(
        tmp_path,
        fold=0,
        split="val",
        case_ids=(case_id,),
        target_spacing_xy=(1.0, 1.0),
        bilateral_asymmetry_channel=True,
    )

    first_channels, _ = dataset.load_case(case_id)
    active_label[0] = NiftiVolume(second_label, (1.0, 1.0, 4.0), (0.0, 0.0, 0.0), IDENTITY_DIRECTION)
    second_channels, _ = dataset.load_case(case_id)

    np.testing.assert_allclose(first_channels, second_channels, atol=1e-6)
    assert estimates[0] == estimates[1]
    assert image_path.read_bytes() == original_bytes


def test_derived_channel_needs_only_physical_dwi_and_reports_physical_and_derived_counts(monkeypatch, tmp_path: Path) -> None:
    case_id = _case_id(monkeypatch)
    image = _symmetric_dwi()
    label = np.zeros_like(image, dtype=np.int16)
    _write_single_dwi_case(tmp_path, case_id, image, label)

    dataset = StrokeSliceDataset(
        tmp_path,
        fold=0,
        split="val",
        case_ids=(case_id,),
        target_spacing_xy=(1.0, 1.0),
        bilateral_asymmetry_channel=True,
    )

    channels, _ = dataset.load_case(case_id)
    assert not (tmp_path / "imagesTr" / f"{case_id}_0001.nii.gz").exists()
    assert dataset.physical_input_channels == 1
    assert dataset.derived_input_channels == 1
    assert dataset.input_channels == 2
    assert resolve_input_channels(tmp_path, bilateral_asymmetry_channel=True) == 2
    assert channels.shape == (2, 3, 33, 33)


def test_formal_train_opt_in_resolves_two_input_channels_without_starting_training(monkeypatch, tmp_path: Path, capsys) -> None:
    case_id = _case_id(monkeypatch)
    image = _symmetric_dwi()
    label = np.zeros_like(image, dtype=np.int16)
    _write_single_dwi_case(tmp_path, case_id, image, label)
    plans = tmp_path / "plans.json"
    plans.write_text(
        json.dumps({"configurations": {"2d": {"patch_size": [13, 15], "use_mask_for_norm": [False]}}}),
        encoding="utf-8",
    )

    from standalone_nnunet2d import formal_train

    assert formal_train.main(
        [
            "--raw-root", str(tmp_path),
            "--output-root", str(tmp_path / "unstarted_output"),
            "--plans", str(plans),
            "--device", "cpu",
            "--bilateral-asymmetry-channel",
        ]
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["execution"] == "not-confirmed"
    assert payload["config"]["input_channels"] == 2
    assert payload["config"]["experimental_extension"] == "multichannel"
    assert not (tmp_path / "unstarted_output").exists()


def test_existing_physical_multichannel_loading_is_unchanged_when_opt_in_is_off(monkeypatch, tmp_path: Path) -> None:
    case_id = _case_id(monkeypatch)
    image = _symmetric_dwi()
    label = np.zeros_like(image, dtype=np.int16)
    _write_single_dwi_case(tmp_path, case_id, image, label)
    (tmp_path / "dataset.json").write_text('{"channel_names": {"0": "DWI", "1": "ADC"}}', encoding="utf-8")
    write_nifti(tmp_path / "imagesTr" / f"{case_id}_0001.nii.gz", _volume(image + 10.0))

    dataset = StrokeSliceDataset(tmp_path, fold=0, split="val", case_ids=(case_id,), target_spacing_xy=(1.0, 1.0))
    channels, _ = dataset.load_case(case_id)

    assert dataset.physical_input_channels == 2
    assert dataset.derived_input_channels == 0
    assert channels.shape == (2, 3, 33, 33)
    with pytest.raises(ValueError, match="exactly one physical DWI"):
        StrokeSliceDataset(
            tmp_path,
            fold=0,
            split="val",
            case_ids=(case_id,),
            target_spacing_xy=(1.0, 1.0),
            bilateral_asymmetry_channel=True,
        )


def test_formal_patch_crop_keeps_two_channels_and_validation_is_deterministic(monkeypatch, tmp_path: Path) -> None:
    case_id = _case_id(monkeypatch)
    image = _symmetric_dwi()
    image[:, 16, 22] = 4.0
    label = np.zeros_like(image, dtype=np.int16)
    _write_single_dwi_case(tmp_path, case_id, image, label)
    request = PatchRequest(case_id=case_id, z_index=1, center_yx=(16, 16), force_foreground=False)

    dataset = FormalPatchDataset(
        tmp_path,
        fold=0,
        split="val",
        case_ids=(case_id,),
        patch_size=(13, 15),
        augment=False,
        patch_request=request,
        bilateral_asymmetry_channel=True,
    )

    first_image, first_label = dataset[0]
    second_image, second_label = dataset[0]
    assert tuple(first_image.shape) == (2, 13, 15)
    assert tuple(first_label.shape) == (13, 15)
    assert dataset.use_mask_for_norm == (False, False)
    assert torch.equal(first_image, second_image)
    assert torch.equal(first_label, second_label)


def test_spatial_augmentation_preserves_dwi_and_asymmetry_correspondence(monkeypatch, tmp_path: Path) -> None:
    case_id = _case_id(monkeypatch)
    image = _symmetric_dwi()
    image[:, 12, 22] = 4.0
    label = np.zeros_like(image, dtype=np.int16)
    _write_single_dwi_case(tmp_path, case_id, image, label)

    dataset = StrokeSliceDataset(
        tmp_path,
        fold=0,
        split="val",
        case_ids=(case_id,),
        target_spacing_xy=(1.0, 1.0),
        bilateral_asymmetry_channel=True,
        augmentation_config=dataset_module.AugmentationConfig(
            horizontal_flip_probability=1.0,
            vertical_flip_probability=0.0,
            intensity_scale_range=(1.0, 1.0),
        ),
    )
    unaugmented, _ = dataset.load_case(case_id)
    augmented, _ = dataset[0]

    np.testing.assert_allclose(augmented.numpy(), unaugmented[:, 1, :, ::-1], atol=1e-6)
