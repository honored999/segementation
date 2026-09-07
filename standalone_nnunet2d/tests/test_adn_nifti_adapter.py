"""Synthetic tests for strict NIfTI orientation canonicalization."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from standalone_nnunet2d.brain_alignment.nifti_adapter import (
    OrientationError,
    canonicalize_nifti,
    compute_orientation_mapping,
)
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


def _z_rotation(degrees: float) -> tuple[float, ...]:
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    return (cosine, -sine, 0.0, sine, cosine, 0.0, 0.0, 0.0, 1.0)


def test_identity_preserves_canonical_array_and_raw_geometry() -> None:
    source = _volume((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    result = canonicalize_nifti(source)

    np.testing.assert_array_equal(result.array, source.array)
    assert result.permutation == (0, 1, 2)
    assert result.flips == (False, False, False)
    assert result.original_array_order == "zyx"
    assert result.spacing_xyz == source.spacing_xyz
    assert result.origin_xyz == source.origin_xyz
    assert result.direction == source.direction
    assert result.direction_labels == ("SI+", "AP+", "LR+")
    np.testing.assert_array_equal(result.restore_array(), source.array)


def test_permutation_and_flip_are_exactly_reversible_without_interpolation() -> None:
    # Raw x -> +SI, raw y -> -LR, raw z -> +AP.  The array is still z,y,x.
    direction = (0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
    source = _volume(direction, shape=(2, 3, 4))

    result = canonicalize_nifti(source)

    assert result.permutation == (2, 0, 1)
    assert result.flips == (False, False, True)
    assert result.provenance["canonical_axes"] == ["SI", "AP", "LR"]
    np.testing.assert_array_equal(result.restore_array(), source.array)
    np.testing.assert_array_equal(result.restore_array(result.array), source.array)
    assert result.provenance["original_shape"] == [2, 3, 4]


def test_case005_style_direction_is_accepted_and_reversible() -> None:
    source = _volume(CASE005_DIRECTION, shape=(2, 3, 4))

    result = canonicalize_nifti(source)

    assert result.permutation == (0, 1, 2)
    assert result.flips == (False, False, False)
    assert result.direction_labels == ("SI+", "AP+", "LR+")
    np.testing.assert_array_equal(result.restore_array(), source.array)


def test_oblique_direction_within_axis_gate_is_accepted_without_resampling() -> None:
    source = _volume(_z_rotation(5.0), shape=(2, 3, 4))
    before = source.array.copy()

    result = canonicalize_nifti(source)

    assert result.permutation == (0, 1, 2)
    assert result.flips == (False, False, False)
    np.testing.assert_array_equal(result.array, before)
    np.testing.assert_array_equal(result.restore_array(), before)
    assert result.provenance["geometry_modified"] is False


@pytest.mark.parametrize(
    ("direction", "reason"),
    [
        ((math.nan, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), "nonfinite_direction"),
        ((1.001, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), "nonorthogonal_direction"),
        (_z_rotation(25.0), "dominance_threshold"),
        (_z_rotation(46.0), "dominance_margin"),
        (_z_rotation(45.0), "ambiguous_assignment"),
    ],
)
def test_orientation_rejection_has_a_recordable_reason(
    direction: tuple[float, ...], reason: str
) -> None:
    with pytest.raises(OrientationError) as error:
        compute_orientation_mapping(direction)

    assert error.value.reason == reason
    assert reason in str(error.value)
    assert isinstance(error.value.details, dict)


def test_mapping_uses_one_global_assignment_not_independent_axis_argmax() -> None:
    mapping = compute_orientation_mapping(CASE005_DIRECTION)

    assert sorted(mapping.permutation) == [0, 1, 2]
    assert mapping.assignment_count == 1
    assert mapping.assignment_score > 2.9
