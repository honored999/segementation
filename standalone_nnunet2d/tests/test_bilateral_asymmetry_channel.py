from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from standalone_nnunet2d.data import dataset as dataset_module
from standalone_nnunet2d.data.dataset import StrokeSliceDataset, resolve_input_channels
from standalone_nnunet2d.data.nifti_io import NiftiVolume, write_nifti
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
