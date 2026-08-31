"""Synthetic checks for the isolated ADN transformation network."""

from __future__ import annotations

import math

import pytest
import torch

from standalone_nnunet2d.brain_alignment.adn_transform import (
    ADNTransformAligner,
    TransformRanges,
    alignment_losses,
    build_transform_matrices,
    left_right_flip,
    warp_volume,
)


def _cube(*, width_index: int = 10, batch: int = 1) -> torch.Tensor:
    volume = torch.zeros(batch, 1, 16, 16, 16)
    volume[:, :, 7:9, 7:9, width_index : width_index + 2] = 1.0
    return volume


def _center_of_mass(volume: torch.Tensor) -> torch.Tensor:
    indices = torch.arange(volume.shape[-1], dtype=volume.dtype).view(1, 1, 1, 1, -1)
    return (volume * indices).sum() / volume.sum()


def _label_center(volume: torch.Tensor, label: int) -> torch.Tensor:
    return torch.nonzero(volume[0, 0] == label, as_tuple=False).float().mean(dim=0)


def test_zero_parameters_produce_identity_and_exact_inverse() -> None:
    params = torch.zeros(2, 6)
    matrix, inverse = build_transform_matrices(params, spatial_shape=(16, 16, 16))
    identity = torch.eye(4).expand(2, -1, -1)

    assert torch.allclose(matrix, identity)
    assert torch.allclose(inverse, identity)
    assert torch.allclose(matrix @ inverse, identity, atol=1e-6)
    assert torch.allclose(inverse @ matrix, identity, atol=1e-6)


@pytest.mark.parametrize(("translation", "moves_left"), [(0.25, True), (-0.25, False)])
def test_sampling_x_translation_has_explicit_content_direction(
    translation: float, moves_left: bool
) -> None:
    volume = _cube()
    params = torch.tensor([[0.0, 0.0, 0.0, translation, 0.0, 0.0]])
    matrix, _ = build_transform_matrices(params, spatial_shape=(16, 16, 16))
    warped = warp_volume(volume, matrix)

    if moves_left:
        assert _center_of_mass(warped) < _center_of_mass(volume)
    else:
        assert _center_of_mass(warped) > _center_of_mass(volume)


def test_positive_z_rotation_moves_right_of_center_cube_toward_lower_height() -> None:
    volume = _cube(width_index=11)
    params = torch.tensor([[0.0, 0.0, math.pi / 2, 0.0, 0.0, 0.0]])
    matrix, _ = build_transform_matrices(params, spatial_shape=(16, 16, 16))
    warped = warp_volume(volume, matrix, mode="nearest")
    input_height = torch.nonzero(volume[0, 0], as_tuple=False)[:, 1].float().mean()
    output_height = torch.nonzero(warped[0, 0], as_tuple=False)[:, 1].float().mean()

    assert output_height < input_height


def test_non_square_voxel_rotation_preserves_center_and_known_landmark_geometry() -> None:
    """A 90-degree z rotation must be rigid in D/H/W voxel coordinates."""
    volume = torch.zeros(1, 1, 20, 48, 32)
    volume[:, :, 9:11, 23:25, 15:17] = 1
    volume[:, :, 9:11, 27:29, 23:25] = 2
    params = torch.tensor([[0.0, 0.0, math.pi / 2, 0.0, 0.0, 0.0]])
    matrix, _ = build_transform_matrices(params, spatial_shape=(20, 48, 32))
    rotated = warp_volume(volume, matrix, mode="nearest")

    assert torch.allclose(_label_center(rotated, 1), torch.tensor([9.5, 23.5, 15.5]))
    assert torch.allclose(_label_center(rotated, 2), torch.tensor([9.5, 15.5, 19.5]))


def test_square_axial_shape_matches_adn_normalized_z_rotation_matrix() -> None:
    params = torch.tensor([[0.0, 0.0, math.pi / 2, 0.25, 0.0, 0.0]])
    matrix, _ = build_transform_matrices(params, spatial_shape=(20, 32, 32))
    expected = torch.tensor(
        [[[0.0, -1.0, 0.0, 0.25], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]]
    )

    assert torch.allclose(matrix, expected, atol=1e-6)


def test_non_square_nearest_landmarks_survive_forward_inverse_round_trip() -> None:
    volume = torch.zeros(1, 1, 20, 48, 32)
    volume[:, :, 9:11, 23:25, 15:17] = 1
    volume[:, :, 9:11, 27:29, 23:25] = 2
    params = torch.tensor([[0.0, 0.0, math.pi / 2, 0.0, 0.0, 0.0]])
    matrix, inverse = build_transform_matrices(params, spatial_shape=(20, 48, 32))

    restored = warp_volume(warp_volume(volume, matrix, mode="nearest"), inverse, mode="nearest")
    assert torch.equal(restored, volume)


def test_identity_warp_preserves_volume_and_shape() -> None:
    torch.manual_seed(0)
    volume = torch.rand(2, 1, 16, 20, 24)
    matrix, _ = build_transform_matrices(torch.zeros(2, 6), spatial_shape=tuple(volume.shape[2:]))
    warped = warp_volume(volume, matrix)

    assert warped.shape == volume.shape
    assert torch.allclose(warped, volume, atol=2e-6, rtol=0)


def test_nearest_warp_does_not_create_new_label_values() -> None:
    labels = torch.zeros(1, 1, 16, 16, 16)
    labels[:, :, 6:10, 6:10, 9:13] = 2
    labels[:, :, 7:9, 7:9, 10:12] = 5
    matrix, _ = build_transform_matrices(
        torch.tensor([[0.0, 0.0, 0.0, 0.25, 0.0, 0.0]]), spatial_shape=(16, 16, 16)
    )
    warped = warp_volume(labels, matrix, mode="nearest")

    assert set(warped.unique().tolist()).issubset(set(labels.unique().tolist()))


def test_left_right_flip_uses_width_axis() -> None:
    volume = torch.arange(16).view(1, 1, 1, 1, 16)

    assert torch.equal(left_right_flip(volume), volume.flip(-1))


def test_network_returns_parameters_and_handles_variable_eligible_shapes_on_cpu() -> None:
    model = ADNTransformAligner().cpu().eval()
    with torch.no_grad():
        first = model(torch.rand(2, 1, 16, 16, 16))
        second = model(torch.rand(1, 1, 20, 24, 32))

    assert first.raw_params.shape == (2, 6)
    assert first.aligned.shape == (2, 1, 16, 16, 16)
    assert second.aligned.shape == (1, 1, 20, 24, 32)
    assert torch.all(first.raw_params.abs() <= 1.0)


def test_scaled_parameters_respect_configured_limits() -> None:
    ranges = TransformRanges(
        x_rotation_degrees=10,
        y_rotation_degrees=20,
        z_rotation_degrees=30,
        x_translation=0.4,
        y_translation=0.3,
        z_translation=0.2,
    )
    model = ADNTransformAligner(ranges=ranges).eval()
    with torch.no_grad():
        result = model(torch.rand(1, 1, 16, 16, 16))
    limits = torch.tensor([math.radians(10), math.radians(20), math.radians(30), 0.4, 0.3, 0.2])

    assert torch.all(result.scaled_params.abs() <= limits + 1e-6)


def test_network_rejects_spatial_sizes_too_small_for_four_downsamplings() -> None:
    with pytest.raises(ValueError, match="at least 16"):
        ADNTransformAligner()(torch.rand(1, 1, 15, 16, 16))


def test_losses_distinguish_symmetric_and_asymmetric_volumes() -> None:
    coordinate = torch.linspace(-1, 1, 16)
    symmetric = torch.exp(-8 * coordinate.square()).view(1, 1, 1, 1, 16).expand(1, 1, 16, 16, 16)
    asymmetric = symmetric.clone()
    asymmetric[..., 11:13] += 1
    identity, _ = build_transform_matrices(torch.zeros(1, 6), spatial_shape=(16, 16, 16))

    symmetric_loss = alignment_losses(symmetric, symmetric, identity)
    asymmetric_loss = alignment_losses(asymmetric, asymmetric, identity)
    assert symmetric_loss.flip_loss < 1e-6
    assert asymmetric_loss.flip_loss > symmetric_loss.flip_loss
    assert torch.isfinite(asymmetric_loss.total_loss)


def test_alignment_loss_backpropagates_finite_nonzero_network_gradients() -> None:
    torch.manual_seed(7)
    model = ADNTransformAligner()
    volume = torch.rand(1, 1, 16, 16, 16)
    result = model(volume)
    losses = alignment_losses(volume, result.aligned, result.inverse_sampling_matrix)
    losses.total_loss.backward()

    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.abs().sum() > 0 for gradient in gradients)
