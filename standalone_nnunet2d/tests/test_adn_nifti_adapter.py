"""Synthetic tests for acquisition-preserving ADN canonicalization."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from standalone_nnunet2d.brain_alignment.nifti_adapter import OrientationError, canonicalize_nifti, compute_orientation_mapping
from standalone_nnunet2d.data.nifti_io import NiftiVolume


CASE005_DIRECTION = (
    0.999622, -0.004503, 0.027114,
    0.000044, 0.986747, 0.162266,
    -0.027485, -0.162204, 0.986374,
)


def _volume(direction: tuple[float, ...], shape: tuple[int, int, int] = (3, 4, 5)) -> NiftiVolume:
    return NiftiVolume(
        np.arange(np.prod(shape), dtype=np.float32).reshape(shape),
        spacing_xyz=(0.7, 0.8, 4.5),
        origin_xyz=(11.0, -2.0, 3.5),
        direction=direction,
    )


def _x_rotation(degrees: float) -> tuple[float, ...]:
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    return (1.0, 0.0, 0.0, 0.0, cosine, -sine, 0.0, sine, cosine)


def _z_rotation(degrees: float) -> tuple[float, ...]:
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    return (cosine, -sine, 0.0, sine, cosine, 0.0, 0.0, 0.0, 1.0)


def test_identity_preserves_array_and_establishes_acquisition_lr_contract() -> None:
    source = _volume((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    result = canonicalize_nifti(source)
    np.testing.assert_array_equal(result.array, source.array)
    assert result.permutation == (0, 1, 2)
    assert result.flips == (False, False, False)
    assert result.provenance["canonical_contract"] == "acquisition_preserving_lr"
    assert result.provenance["source_voxel_axis_for_dhw"] == ["z", "y", "x"]
    assert result.provenance["lr_source_voxel_axis"] == "x"
    assert result.provenance["lr_signed_component"] == 1.0
    assert result.provenance["lr_absolute_component"] == 1.0
    assert result.provenance["lr_second_best_component"] == 0.0
    assert result.provenance["lr_margin"] == 1.0
    np.testing.assert_array_equal(result.restore_array(), source.array)


def test_lr_in_voxel_y_permutates_only_in_plane_axes_and_keeps_depth_on_z() -> None:
    source = _volume((0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0), shape=(2, 3, 4))
    result = canonicalize_nifti(source)
    assert result.permutation == (0, 2, 1)
    assert result.array.shape == (2, 4, 3)
    assert result.flips == (True, False, False)
    assert result.provenance["source_voxel_axis_for_dhw"] == ["z", "x", "y"]
    assert result.provenance["lr_source_voxel_axis"] == "y"
    np.testing.assert_array_equal(result.restore_array(), source.array)


def test_negative_lr_flips_w_and_right_handed_rule_determines_h_flip() -> None:
    source = _volume((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
    result = canonicalize_nifti(source)
    assert result.permutation == (0, 1, 2)
    assert result.flips == (False, True, True)
    assert result.provenance["handedness_after_flips"] > 0.0
    assert result.provenance["flip_rules"] == {
        "D": "source_voxel_z_to_lps_positive_z",
        "H": "right_handed_w_h_d",
        "W": "lr_to_lps_positive_x",
    }
    np.testing.assert_array_equal(result.restore_array(), source.array)


@pytest.mark.parametrize("degrees", [35.0, 44.5, 45.0, 50.0])
def test_strong_ap_si_obliquity_is_accepted_when_lr_is_clear(degrees: float) -> None:
    source = _volume(_x_rotation(degrees), shape=(2, 3, 4))
    result = canonicalize_nifti(source)
    assert result.permutation == (0, 1, 2)
    assert result.provenance["lr_margin"] == pytest.approx(1.0)
    assert set(result.provenance["d_lps_components"]) == {"X", "Y", "Z"}
    assert set(result.provenance["h_lps_components"]) == {"X", "Y", "Z"}
    np.testing.assert_array_equal(result.restore_array(), source.array)


def test_case005_style_direction_is_accepted_and_reversible() -> None:
    source = _volume(CASE005_DIRECTION, shape=(2, 3, 4))
    result = canonicalize_nifti(source)
    assert result.permutation == (0, 1, 2)
    assert result.provenance["lr_source_voxel_axis"] == "x"
    np.testing.assert_array_equal(result.restore_array(), source.array)


@pytest.mark.parametrize(
    ("direction", "reason"),
    [
        ((math.nan, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), "nonfinite_direction"),
        ((1.001, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), "nonorthogonal_direction"),
        ((0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0), "lr_axis_is_voxel_z"),
        ((1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0), "d_lps_z_zero"),
        (_z_rotation(40.0), "lr_ambiguous"),
        (_z_rotation(45.0), "lr_ambiguous"),
    ],
)
def test_orientation_rejection_has_a_recordable_reason(direction: tuple[float, ...], reason: str) -> None:
    with pytest.raises(OrientationError) as error:
        compute_orientation_mapping(direction)
    assert error.value.reason == reason
    assert reason in str(error.value)
    assert isinstance(error.value.details, dict)


def test_provenance_is_json_serializable_complete_and_geometry_preserving() -> None:
    source = _volume(_x_rotation(45.0), shape=(2, 3, 4))
    result = canonicalize_nifti(source)
    assert json.dumps(result.provenance)
    assert result.provenance["original_array_order"] == "zyx"
    assert result.provenance["source_geometry"] == {
        "spacing_xyz": [0.7, 0.8, 4.5],
        "origin_xyz": [11.0, -2.0, 3.5],
        "direction": list(source.direction),
    }
    assert len(result.provenance["source_direction_vectors"]) == 3
    assert result.provenance["applied_permutation"] == [0, 1, 2]
    assert result.provenance["applied_flips"] == [False, False, False]
    assert result.provenance["interpolation_performed"] is False
    assert result.provenance["geometry_modified"] is False
    assert result.provenance["inverse_operations"] == {
        "flip_canonical_axes": [], "transpose_axes": [0, 1, 2]
    }


def test_restore_accepts_external_canonical_array_and_is_exact() -> None:
    source = _volume((0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0), shape=(2, 3, 4))
    result = canonicalize_nifti(source)
    np.testing.assert_array_equal(result.restore_array(result.array.copy()), source.array)
    with pytest.raises(ValueError, match="3D"):
        result.restore_array(result.array[0])
