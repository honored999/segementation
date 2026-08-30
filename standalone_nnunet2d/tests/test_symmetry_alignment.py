from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from standalone_nnunet2d.data.dataset import StrokeSliceDataset
from standalone_nnunet2d.data.nifti_io import NiftiVolume
from standalone_nnunet2d.data.preprocessing import resample_inplane, z_score_normalize
from standalone_nnunet2d.training.formal_dataset import FormalPatchDataset


IDENTITY_DIRECTION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def _alignment_api():
    from standalone_nnunet2d.data.symmetry_alignment import (
        AlignmentResult,
        build_alignment_qc,
        estimate_quasi_symmetric_alignment,
        align_case,
    )

    return AlignmentResult, build_alignment_qc, estimate_quasi_symmetric_alignment, align_case


def _physical_grid(
    shape_zyx: tuple[int, int, int],
    *,
    spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 4.0),
    origin_xyz: tuple[float, float, float] = (10.0, 20.0, 30.0),
    direction: tuple[float, ...] = IDENTITY_DIRECTION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z, y, x = np.indices(shape_zyx, dtype=np.float64)
    index_xyz = np.stack((x, y, z), axis=-1)
    direction_matrix = np.asarray(direction, dtype=np.float64).reshape(3, 3)
    physical_xyz = np.asarray(origin_xyz) + index_xyz @ direction_matrix.T * np.asarray(spacing_xyz)
    return physical_xyz[..., 0], physical_xyz[..., 1], physical_xyz[..., 2]


def _ellipse_volume(
    *,
    centre_xy: tuple[float, float],
    angle_degrees: float = 0.0,
    marker_xy: tuple[float, float] | None = None,
    direction: tuple[float, ...] = IDENTITY_DIRECTION,
) -> tuple[NiftiVolume, NiftiVolume]:
    shape = (3, 65, 65)
    px, py, _ = _physical_grid(shape, direction=direction)
    centre = np.asarray(centre_xy, dtype=np.float64)
    angle = np.deg2rad(angle_degrees)
    relative = np.stack((px - centre[0], py - centre[1]), axis=-1)
    rotation = np.array(((np.cos(angle), np.sin(angle)), (-np.sin(angle), np.cos(angle))))
    principal_xy = relative @ rotation.T
    image_array = np.where((principal_xy[..., 0] / 16.0) ** 2 + (principal_xy[..., 1] / 7.0) ** 2 <= 1.0, 10.0, 0.0)
    label_array = np.zeros(shape, dtype=np.int16)
    if marker_xy is not None:
        distances = (px - marker_xy[0]) ** 2 + (py - marker_xy[1]) ** 2
        marker = np.unravel_index(np.argmin(distances[1]), distances[1].shape)
        label_array[1, marker[0], marker[1]] = 2
    image = NiftiVolume(image_array.astype(np.float32), (1.0, 1.0, 4.0), (10.0, 20.0, 30.0), direction)
    label = NiftiVolume(label_array, image.spacing_xyz, image.origin_xyz, direction)
    return image, label


def _reference_center(volume: NiftiVolume) -> np.ndarray:
    size_xyz = np.asarray(volume.array.shape[::-1], dtype=np.float64)
    index_xyz = (size_xyz - 1.0) / 2.0
    direction = np.asarray(volume.direction, dtype=np.float64).reshape(3, 3)
    return np.asarray(volume.origin_xyz) + direction @ (index_xyz * np.asarray(volume.spacing_xyz))


def _foreground_centroid(volume: NiftiVolume, value: int | float = 0) -> np.ndarray:
    positions = np.argwhere(volume.array != value)
    xyz = positions[:, ::-1].astype(np.float64)
    direction = np.asarray(volume.direction, dtype=np.float64).reshape(3, 3)
    physical = np.asarray(volume.origin_xyz) + (xyz * np.asarray(volume.spacing_xyz)) @ direction.T
    return physical.mean(axis=0)


def test_identity_alignment_preserves_a_centered_physical_volume() -> None:
    _, _, estimate_alignment, align_case = _alignment_api()
    image, label = _ellipse_volume(centre_xy=(42.0, 52.0))

    estimate = estimate_alignment(image)
    aligned_image, aligned_label, result_estimate = align_case(image, label)

    np.testing.assert_allclose(estimate.center_xyz, (42.0, 52.0, 34.0), atol=0.2)
    assert estimate.rotation_angle_radians == pytest.approx(0.0, abs=1e-3)
    np.testing.assert_allclose(aligned_image.array, image.array, atol=1e-5)
    np.testing.assert_array_equal(aligned_label.array, label.array)
    assert result_estimate == estimate


def test_alignment_moves_known_physical_translation_to_grid_center() -> None:
    _, _, estimate_alignment, align_case = _alignment_api()
    image, label = _ellipse_volume(centre_xy=(47.0, 49.0), marker_xy=(47.0, 49.0))

    estimate = estimate_alignment(image)
    _, aligned_label, _ = align_case(image, label)

    reference = _reference_center(label)
    assert estimate.center_xyz[0] > reference[0]
    assert estimate.center_xyz[1] < reference[1]
    aligned_centroid = _foreground_centroid(aligned_label, value=0)
    np.testing.assert_allclose(aligned_centroid[:2], reference[:2], atol=0.75)


def test_alignment_rotates_known_physical_marker_in_the_correction_direction() -> None:
    _, _, estimate_alignment, align_case = _alignment_api()
    angle_degrees = 27.0
    image, label = _ellipse_volume(
        centre_xy=(42.0, 52.0),
        angle_degrees=angle_degrees,
        marker_xy=(52.0, 57.0),
    )

    estimate = estimate_alignment(image)
    _, aligned_label, _ = align_case(image, label)

    assert estimate.rotation_angle_radians == pytest.approx(np.deg2rad(angle_degrees), abs=0.04)
    reference = _reference_center(label)
    marker_input = np.asarray((52.0, 57.0))
    correction = np.array(
        ((np.cos(np.deg2rad(-angle_degrees)), -np.sin(np.deg2rad(-angle_degrees))),
         (np.sin(np.deg2rad(-angle_degrees)), np.cos(np.deg2rad(-angle_degrees))))
    )
    expected_marker = reference[:2] + correction @ (marker_input - np.asarray((42.0, 52.0)))
    actual_marker = _foreground_centroid(aligned_label, value=0)
    np.testing.assert_allclose(actual_marker[:2], expected_marker, atol=1.25)


def test_image_and_label_share_geometry_transform_and_labels_stay_discrete() -> None:
    _, _, _, align_case = _alignment_api()
    image, label = _ellipse_volume(centre_xy=(47.0, 49.0), marker_xy=(52.0, 55.0))

    aligned_image, aligned_label, _ = align_case(image, label)

    assert aligned_image.array.shape == image.array.shape
    assert aligned_label.array.shape == label.array.shape
    assert aligned_image.spacing_xyz == image.spacing_xyz
    assert aligned_label.spacing_xyz == label.spacing_xyz
    assert aligned_image.origin_xyz == image.origin_xyz
    assert aligned_label.origin_xyz == label.origin_xyz
    assert aligned_image.direction == image.direction
    assert aligned_label.direction == label.direction
    assert aligned_image.array.shape[0] == image.array.shape[0]
    assert aligned_image.spacing_xyz[2] == image.spacing_xyz[2]
    assert aligned_label.array.dtype == np.int16
    assert set(np.unique(aligned_label.array)).issubset({0, 1, 2})


def test_alignment_uses_physical_coordinates_for_axis_swaps_and_flips() -> None:
    _, _, estimate_alignment, _ = _alignment_api()
    direction = (0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0)
    reference_volume = NiftiVolume(np.zeros((3, 65, 65), dtype=np.float32), (1.0, 1.0, 4.0), (10.0, 20.0, 30.0), direction)
    reference_center = _reference_center(reference_volume)
    image, _ = _ellipse_volume(centre_xy=tuple(reference_center[:2]), direction=direction)

    estimate = estimate_alignment(image)

    np.testing.assert_allclose(estimate.center_xyz[:2], reference_center[:2], atol=0.5)
    assert np.isfinite(estimate.rotation_angle_radians)


@pytest.mark.parametrize(
    ("image", "match"),
    [
        (NiftiVolume(np.full((3, 9, 9), np.nan, dtype=np.float32), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)), "non-finite"),
        (NiftiVolume(np.zeros((3, 9, 9), dtype=np.float32), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)), "foreground"),
        (NiftiVolume(np.ones((3, 9, 9), dtype=np.float32), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)), "foreground"),
        (NiftiVolume(np.ones((3, 9, 9), dtype=np.float32), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), (1.0, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)), "direction"),
    ],
)
def test_alignment_rejects_invalid_image_inputs(image: NiftiVolume, match: str) -> None:
    _, _, estimate_alignment, _ = _alignment_api()

    with pytest.raises(ValueError, match=match):
        estimate_alignment(image)


def test_alignment_rejects_a_degenerate_principal_axis() -> None:
    _, _, estimate_alignment, _ = _alignment_api()
    image = NiftiVolume(np.zeros((3, 33, 33), dtype=np.float32), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    image_array = image.array.copy()
    yy, xx = np.indices(image_array.shape[1:])
    image_array[:, (xx - 16) ** 2 + (yy - 16) ** 2 <= 8**2] = 1.0
    image = NiftiVolume(image_array, image.spacing_xyz, image.origin_xyz, image.direction)

    with pytest.raises(ValueError, match="principal axis"):
        estimate_alignment(image)


def test_alignment_qc_contains_requested_slice_data_and_estimate() -> None:
    AlignmentResult, build_qc, _, align_case = _alignment_api()
    image, label = _ellipse_volume(centre_xy=(47.0, 49.0))
    aligned_image, _, estimate = align_case(image, label)
    result = AlignmentResult(image=aligned_image, label=None, estimate=estimate)

    qc = build_qc(image, result=result, slice_index=1)

    assert qc.original.shape == image.array.shape[1:]
    assert qc.aligned.shape == image.array.shape[1:]
    assert qc.mirrored.shape == image.array.shape[1:]
    assert qc.absolute_difference.shape == image.array.shape[1:]
    assert qc.center_xyz == estimate.center_xyz
    assert qc.rotation_angle_radians == estimate.rotation_angle_radians
    np.testing.assert_allclose(qc.absolute_difference, np.abs(qc.aligned - qc.mirrored))

    with pytest.raises(IndexError, match="slice index"):
        build_qc(image, result=result, slice_index=3)


def test_stroke_slice_dataset_default_off_matches_existing_preprocessing(monkeypatch, tmp_path: Path) -> None:
    case_id = "case_synthetic"
    image = NiftiVolume(np.arange(75, dtype=np.float32).reshape(3, 5, 5), (1.0, 1.0, 4.0), (1.0, 2.0, 3.0))
    label = NiftiVolume((image.array > 30).astype(np.int16), image.spacing_xyz, image.origin_xyz)
    (tmp_path / "imagesTr").mkdir()
    (tmp_path / "labelsTr").mkdir()

    import standalone_nnunet2d.data.dataset as dataset_module

    monkeypatch.setattr(dataset_module, "load_fold_cases", lambda fold, split: (case_id,))
    monkeypatch.setattr(dataset_module, "read_nifti", lambda path: image if "imagesTr" in str(path) else label)
    expected_image = z_score_normalize(resample_inplane(image, (1.0, 1.0), is_segmentation=False).array)
    expected_label = resample_inplane(label, (1.0, 1.0), is_segmentation=True).array

    default_dataset = StrokeSliceDataset(tmp_path, fold=0, split="val", case_ids=(case_id,), target_spacing_xy=(1.0, 1.0))
    explicit_dataset = StrokeSliceDataset(
        tmp_path,
        fold=0,
        split="val",
        case_ids=(case_id,),
        target_spacing_xy=(1.0, 1.0),
        symmetry_alignment=False,
    )

    default_result = default_dataset.load_case(case_id)
    explicit_result = explicit_dataset.load_case(case_id)
    np.testing.assert_array_equal(default_result[0], expected_image)
    np.testing.assert_array_equal(default_result[1], expected_label)
    np.testing.assert_array_equal(default_result[0], explicit_result[0])
    np.testing.assert_array_equal(default_result[1], explicit_result[1])


def test_formal_patch_dataset_default_off_matches_explicit_false(monkeypatch, tmp_path: Path) -> None:
    case_id = "case_synthetic"
    image = NiftiVolume(np.arange(75, dtype=np.float32).reshape(3, 5, 5), (1.0, 1.0, 4.0), (1.0, 2.0, 3.0))
    label = NiftiVolume((image.array > 30).astype(np.int16), image.spacing_xyz, image.origin_xyz)
    (tmp_path / "imagesTr").mkdir()
    (tmp_path / "labelsTr").mkdir()

    import standalone_nnunet2d.data.dataset as dataset_module

    monkeypatch.setattr(dataset_module, "load_fold_cases", lambda fold, split: (case_id,))
    monkeypatch.setattr(dataset_module, "read_nifti", lambda path: image if "imagesTr" in str(path) else label)

    default_dataset = FormalPatchDataset(tmp_path, fold=0, split="train", case_ids=(case_id,), patch_size=(5, 5), augment=False)
    explicit_dataset = FormalPatchDataset(
        tmp_path,
        fold=0,
        split="train",
        case_ids=(case_id,),
        patch_size=(5, 5),
        augment=False,
        symmetry_alignment=False,
    )

    np.testing.assert_array_equal(default_dataset.load_case(case_id)[0], explicit_dataset.load_case(case_id)[0])
    np.testing.assert_array_equal(default_dataset.load_case(case_id)[1], explicit_dataset.load_case(case_id)[1])
