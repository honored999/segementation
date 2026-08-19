from dataclasses import replace

import numpy as np
import pytest

from coarse_to_fine_dwi.nifti import (
    NiftiVolume,
    assert_compatible,
    crop_xy,
    restore_xy,
)
from coarse_to_fine_dwi.roi import compute_prediction_roi, validate_binary_prediction


def make_volume(array: np.ndarray) -> NiftiVolume:
    return NiftiVolume(
        array=array,
        spacing_xyz=(2.0, 3.0, 4.0),
        origin_xyz=(10.0, 20.0, 30.0),
        direction=(
            0.0,
            -1.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
    )


def test_prediction_roi_unions_disconnected_foreground_across_all_z_slices():
    prediction = np.zeros((3, 8, 10), dtype=np.uint8)
    prediction[0, 1, 1] = 1
    prediction[2, 6, 8] = 1

    roi = compute_prediction_roi(prediction, margin=1)

    assert roi.bbox == (0, 0, 10, 8)
    assert roi.fallback is False


def test_prediction_roi_empty_prediction_falls_back_to_full_xy():
    roi = compute_prediction_roi(np.zeros((2, 5, 7), dtype=np.uint8))

    assert roi.bbox == (0, 0, 7, 5)
    assert roi.fallback is True


def test_validate_binary_prediction_rejects_nonbinary_nonfinite_and_non3d_inputs():
    with pytest.raises(ValueError, match="0 and 1"):
        validate_binary_prediction(np.array([[[0, 2]]], dtype=np.int8))
    with pytest.raises(ValueError, match="finite"):
        validate_binary_prediction(np.array([[[0.0, np.nan]]]))
    with pytest.raises(ValueError, match="3-dimensional"):
        validate_binary_prediction(np.zeros((4, 5), dtype=np.uint8))


def test_prediction_roi_respects_edges_and_minimum_size():
    prediction = np.zeros((1, 6, 9), dtype=np.uint8)
    prediction[0, 0, 0] = 1

    roi = compute_prediction_roi(prediction, margin=1, min_width=4, min_height=3)

    assert roi.bbox == (0, 0, 4, 3)
    assert 0 <= roi.x0 < roi.x1 <= 9
    assert 0 <= roi.y0 < roi.y1 <= 6


def test_crop_and_restore_preserve_array_and_reference_space_with_direction():
    bbox = (2, 1, 7, 5)
    array = np.zeros((2, 6, 8), dtype=np.int16)
    array[:, 1:5, 2:7] = np.arange(2 * 4 * 5, dtype=np.int16).reshape(2, 4, 5)
    source = make_volume(array)

    cropped = crop_xy(source, bbox)
    restored = restore_xy(cropped, source, bbox)

    assert cropped.array.shape == (2, 4, 5)
    assert cropped.origin_xyz == pytest.approx((7.0, 24.0, 30.0))
    assert np.array_equal(restored.array, source.array)
    assert_compatible(restored, source)


def test_crop_and_restore_reject_invalid_bbox_and_mismatched_crop_metadata():
    source = make_volume(np.zeros((2, 6, 8), dtype=np.uint8))

    with pytest.raises(ValueError, match="bbox"):
        crop_xy(source, (2, 1, 2, 5))

    cropped = crop_xy(source, (1, 1, 5, 5))
    bad_origin = replace(cropped, origin_xyz=(999.0, 999.0, 999.0))
    with pytest.raises(ValueError, match="metadata"):
        restore_xy(bad_origin, source, (1, 1, 5, 5))

    bad_shape = replace(cropped, array=np.zeros((2, 3, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="shape"):
        restore_xy(bad_shape, source, (1, 1, 5, 5))


def test_nifti_round_trip_uses_simpleitk_metadata(tmp_path):
    sitk = pytest.importorskip("SimpleITK")
    source = make_volume(np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4))
    path = tmp_path / "volume.nii.gz"

    source.write(path)
    loaded = NiftiVolume.read(path)

    assert np.array_equal(loaded.array, source.array)
    assert_compatible(loaded, source)


def test_restore_xy_accepts_serialized_oblique_crop_origin_roundoff(tmp_path):
    pytest.importorskip("SimpleITK")
    source = NiftiVolume(
        array=np.zeros((16, 512, 512), dtype=np.uint8),
        spacing_xyz=(0.4892368018627167, 0.4892368018627167, 7.999998569488525),
        origin_xyz=(-114.60359191894531, -105.38716888427734, 51.46718215942383),
        direction=(
            0.9470567631478918,
            -0.038241729030622694,
            0.3187805757673911,
            -0.1062297453790248,
            0.8996375798568397,
            0.42351796805047803,
            -0.3029830499756473,
            -0.4349595326810035,
            0.8479454435586082,
        ),
    )
    source_path = tmp_path / "source.nii.gz"
    crop_path = tmp_path / "crop.nii.gz"
    source.write(source_path)
    serialized_source = NiftiVolume.read(source_path)
    bbox = (51, 323, 242, 450)
    crop_xy(serialized_source, bbox).write(crop_path)
    serialized_crop = NiftiVolume.read(crop_path)

    expected_origin = (
        serialized_source.origin_xyz
        + serialized_source.direction_matrix
        @ np.array(
            [
                bbox[0] * serialized_source.spacing_xyz[0],
                bbox[1] * serialized_source.spacing_xyz[1],
                0.0,
            ]
        )
    )
    assert max(abs(actual - expected) for actual, expected in zip(serialized_crop.origin_xyz, expected_origin)) > 1e-6
    restore_xy(serialized_crop, serialized_source, bbox)


def test_restore_xy_rejects_crop_origin_beyond_serialization_roundoff_tolerance():
    source = make_volume(np.zeros((2, 6, 8), dtype=np.uint8))
    cropped = crop_xy(source, (1, 1, 5, 5))
    shifted = replace(cropped, origin_xyz=(cropped.origin_xyz[0] + 1.1e-5, *cropped.origin_xyz[1:]))

    with pytest.raises(ValueError, match="origin"):
        restore_xy(shifted, source, (1, 1, 5, 5))


def test_direction_metadata_allows_round_trip_noise_but_rejects_material_difference():
    source = make_volume(np.zeros((2, 6, 8), dtype=np.uint8))
    nearly_identical = replace(
        source,
        direction=(1e-8, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )
    materially_different = replace(
        source,
        direction=(2e-6, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )

    assert_compatible(source, nearly_identical)
    with pytest.raises(ValueError, match="metadata mismatch"):
        assert_compatible(source, materially_different)

    cropped = crop_xy(source, (1, 1, 5, 5))
    assert_compatible(restore_xy(replace(cropped, direction=nearly_identical.direction), source, (1, 1, 5, 5)), source)
    with pytest.raises(ValueError, match="direction"):
        restore_xy(replace(cropped, direction=materially_different.direction), source, (1, 1, 5, 5))
