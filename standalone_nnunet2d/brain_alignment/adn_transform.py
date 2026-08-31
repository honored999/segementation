"""Minimal, standalone ADN transformation network for canonical model-space volumes.

All functions here use PyTorch tensors shaped ``[B, C, D, H, W]``.  ``W`` /
normalized ``x`` is the left-right axis *only in this module's canonical model
space*.  It must not be assumed to be the anatomical left-right axis of a raw
DICOM or NIfTI array.  A future external adapter must use image
geometry/orientation to canonicalize inputs before this module is used.  That
adapter, image I/O, resampling, and real CI-1 validation are deliberately out
of scope here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# The only model-space anatomical convention in this module: [B, C, D, H, W].
LEFT_RIGHT_DIM = -1
_MIN_SPATIAL_SIZE = 16


@dataclass(frozen=True)
class TransformRanges:
    """Bounds used to scale the encoder's six ``tanh`` outputs.

    Defaults reproduce the nonzero ranges in the ADN official implementation:
    axial ``z`` rotation up to 40 degrees and normalized ``x`` translation up
    to 0.5.  The other parameters remain implemented and configurable.
    """

    x_rotation_degrees: float = 0.0
    y_rotation_degrees: float = 0.0
    z_rotation_degrees: float = 40.0
    x_translation: float = 0.5
    y_translation: float = 0.0
    z_translation: float = 0.0

    def scale(self, raw_params: Tensor) -> Tensor:
        """Scale raw values to voxel-space radians and normalized translations."""
        _validate_params(raw_params)
        ranges = raw_params.new_tensor(
            [
                math.radians(self.x_rotation_degrees),
                math.radians(self.y_rotation_degrees),
                math.radians(self.z_rotation_degrees),
                self.x_translation,
                self.y_translation,
                self.z_translation,
            ]
        )
        return raw_params * ranges


@dataclass(frozen=True)
class AlignmentResult:
    """Inference outputs for a predicted sampling transform.

    ``sampling_matrix`` (also exposed as ``forward_matrix``) is passed to
    :func:`torch.nn.functional.affine_grid`.  Therefore it maps normalized
    **output** ``(x, y, z)`` coordinates to normalized **input sampling**
    coordinates; it is not an input-to-output physical-space transform.
    """

    aligned: Tensor
    raw_params: Tensor
    scaled_params: Tensor
    sampling_matrix: Tensor
    inverse_sampling_matrix: Tensor

    @property
    def forward_matrix(self) -> Tensor:
        """Alias for the output-to-input normalized sampling matrix."""
        return self.sampling_matrix

    @property
    def inverse_matrix(self) -> Tensor:
        """Alias for the inverse of the output-to-input sampling matrix."""
        return self.inverse_sampling_matrix


@dataclass(frozen=True)
class AlignmentLosses:
    """The separate ADN-style symmetry and reconstruction objectives."""

    flip_loss: Tensor
    reconstruction_loss: Tensor
    total_loss: Tensor


def _validate_volume(volume: Tensor) -> None:
    if volume.ndim != 5:
        raise ValueError(f"volume must have shape [B, C, D, H, W], got {tuple(volume.shape)}")


def _validate_params(params: Tensor) -> None:
    if params.ndim != 2 or params.shape[1] != 6:
        raise ValueError(f"transform parameters must have shape [B, 6], got {tuple(params.shape)}")


def left_right_flip(volume: Tensor) -> Tensor:
    """Reflect a canonical model-space volume across the W/x left-right axis."""
    _validate_volume(volume)
    return torch.flip(volume, dims=(LEFT_RIGHT_DIM,))


def build_transform_matrices(
    scaled_params: Tensor, *, spatial_shape: tuple[int, int, int]
) -> tuple[Tensor, Tensor]:
    """Build voxel-rigid, output-to-input normalized sampling matrices.

    ``spatial_shape`` is ``(D, H, W)``.  Rotations in ``scaled_params`` are
    rigid rotations in voxel coordinates ``(x=W, y=H, z=D)`` about the image
    center, not rotations in normalized coordinates and not physical-space
    rotations.  With ``align_corners=False``, let ``A`` map normalized to voxel
    coordinates.  The rotational part supplied to ``affine_grid`` is
    ``A^-1 @ T(center) @ Rz @ Ry @ Rx @ T(-center) @ A``.

    ``tx, ty, tz`` deliberately retain ADN's normalized-coordinate convention:
    they are left-multiplied after that conversion.  Consequently the combined
    six-parameter matrix is not claimed to be one voxel-space or physical-space
    SE(3) transform.  For a square axial grid, z rotation reduces exactly to
    ADN's original normalized-coordinate matrix.
    """
    _validate_params(scaled_params)
    _validate_spatial_shape(spatial_shape)
    rx, ry, rz, tx, ty, tz = scaled_params.unbind(dim=1)
    zeros = torch.zeros_like(rx)
    ones = torch.ones_like(rx)

    rotation_x = torch.stack(
        (
            ones, zeros, zeros,
            zeros, rx.cos(), -rx.sin(),
            zeros, rx.sin(), rx.cos(),
        ),
        dim=1,
    ).reshape(-1, 3, 3)
    rotation_y = torch.stack(
        (
            ry.cos(), zeros, ry.sin(),
            zeros, ones, zeros,
            -ry.sin(), zeros, ry.cos(),
        ),
        dim=1,
    ).reshape(-1, 3, 3)
    rotation_z = torch.stack(
        (
            rz.cos(), -rz.sin(), zeros,
            rz.sin(), rz.cos(), zeros,
            zeros, zeros, ones,
        ),
        dim=1,
    ).reshape(-1, 3, 3)
    rotation = rotation_z @ rotation_y @ rotation_x

    batch_size = scaled_params.shape[0]
    normalized_to_voxel, voxel_to_normalized, center = _normalized_voxel_matrices(
        spatial_shape, batch_size, scaled_params
    )
    voxel_rigid_rotation = torch.eye(4, dtype=scaled_params.dtype, device=scaled_params.device).expand(
        batch_size, -1, -1
    ).clone()
    voxel_rigid_rotation[:, :3, :3] = rotation
    voxel_rigid_rotation[:, :3, 3] = center - (rotation @ center.unsqueeze(-1)).squeeze(-1)
    normalized_rotation = voxel_to_normalized @ voxel_rigid_rotation @ normalized_to_voxel

    normalized_translation = torch.eye(4, dtype=scaled_params.dtype, device=scaled_params.device).expand(
        batch_size, -1, -1
    ).clone()
    normalized_translation[:, :3, 3] = torch.stack((tx, ty, tz), dim=1)
    matrix = normalized_translation @ normalized_rotation
    inverse = torch.linalg.inv(matrix)
    return matrix, inverse


def _validate_spatial_shape(spatial_shape: tuple[int, int, int]) -> None:
    if len(spatial_shape) != 3 or any(not isinstance(size, int) or size < 2 for size in spatial_shape):
        raise ValueError("spatial_shape must be a (D, H, W) tuple with each size at least 2")


def _normalized_voxel_matrices(
    spatial_shape: tuple[int, int, int], batch_size: int, params: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Return the align_corners=False normalized↔voxel maps and voxel center."""
    depth, height, width = spatial_shape
    scales = params.new_tensor((width / 2, height / 2, depth / 2))
    center = params.new_tensor(((width - 1) / 2, (height - 1) / 2, (depth - 1) / 2))
    normalized_to_voxel = torch.eye(4, dtype=params.dtype, device=params.device).expand(batch_size, -1, -1).clone()
    normalized_to_voxel[:, :3, :3] = torch.diag(scales)
    normalized_to_voxel[:, :3, 3] = center
    voxel_to_normalized = torch.linalg.inv(normalized_to_voxel)
    return normalized_to_voxel, voxel_to_normalized, center.expand(batch_size, -1)


def warp_volume(volume: Tensor, sampling_matrix: Tensor, *, mode: str = "bilinear") -> Tensor:
    """Warp with an output-to-input normalized sampling matrix.

    ``mode='bilinear'`` is PyTorch's trilinear interpolation behavior for a
    5-D input.  ``mode='nearest'`` is available for future discrete masks.
    """
    _validate_volume(volume)
    if mode not in {"bilinear", "nearest"}:
        raise ValueError("mode must be 'bilinear' or 'nearest'")
    if sampling_matrix.shape != (volume.shape[0], 4, 4):
        raise ValueError(
            "sampling_matrix must have shape [B, 4, 4] matching volume batch size, "
            f"got {tuple(sampling_matrix.shape)}"
        )
    if sampling_matrix.device != volume.device or sampling_matrix.dtype != volume.dtype:
        raise ValueError("sampling_matrix must have the same device and dtype as volume")
    grid = F.affine_grid(sampling_matrix[:, :3, :], volume.shape, align_corners=False)
    return F.grid_sample(volume, grid, mode=mode, padding_mode="zeros", align_corners=False)


def alignment_losses(original: Tensor, aligned: Tensor, inverse_sampling_matrix: Tensor) -> AlignmentLosses:
    """Compute official ADN-style flip and inverse-warp reconstruction losses."""
    _validate_volume(original)
    _validate_volume(aligned)
    if original.shape != aligned.shape:
        raise ValueError("original and aligned volumes must have the same shape")
    aligned_flip = left_right_flip(aligned)
    flip_loss = F.l1_loss(aligned, aligned_flip)
    reconstructed = warp_volume(aligned_flip, inverse_sampling_matrix)
    reconstruction_loss = F.l1_loss(original, reconstructed)
    return AlignmentLosses(
        flip_loss=flip_loss,
        reconstruction_loss=reconstruction_loss,
        total_loss=flip_loss + reconstruction_loss,
    )


class BasicBlock(nn.Module):
    """The same pre-activation 3-D residual block used by ADN's encoder."""

    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, value: Tensor) -> Tensor:
        residual = value
        value = self.conv1(self.relu1(self.norm1(value)))
        value = self.conv2(self.relu2(self.norm2(value)))
        return value + residual


class _TransformationEncoder(nn.Module):
    """ADN's four-stage transformation encoder with variable-shape pooling."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        channels = (32, 64, 128, 256)
        groups = (8, 16, 32, 64)
        stages: list[nn.Module] = []
        current_channels = in_channels
        for output_channels, group_count in zip(channels, groups, strict=True):
            stages.append(
                nn.Sequential(
                    nn.Conv3d(current_channels, output_channels, kernel_size=4, stride=2, padding=1, bias=False),
                    BasicBlock(output_channels, group_count),
                )
            )
            current_channels = output_channels
        self.stages = nn.ModuleList(stages)
        # ADN's fixed AvgPool3d((2, 16, 16)) targets 40x256x256 CT.  This
        # AdaptiveAvgPool3d(1) is an MRI variable-shape engineering adaptation.
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(256, 6)
        # ADN does not state identity initialization.  Tiny weights and a zero
        # bias are a standard engineering choice to begin near identity while
        # retaining gradients into the preceding encoder on the first update.
        nn.init.normal_(self.fc.weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.fc.bias)

    def forward(self, volume: Tensor) -> Tensor:
        for stage in self.stages:
            volume = stage(volume)
        return torch.tanh(self.fc(self.pool(volume).flatten(1)))


class ADNTransformAligner(nn.Module):
    """Predict and apply an ADN-compatible canonical model-space alignment.

    The returned sampling matrix maps normalized output coordinates to input
    sampling coordinates.  This wording is intentional: the matrix is passed
    directly to ``affine_grid`` and should not be interpreted as an ambiguous
    physical-space forward transform.
    """

    def __init__(self, *, in_channels: int = 1, ranges: TransformRanges | None = None) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be positive")
        self.in_channels = in_channels
        self.ranges = TransformRanges() if ranges is None else ranges
        self.encoder = _TransformationEncoder(in_channels)

    def forward(self, volume: Tensor) -> AlignmentResult:
        _validate_volume(volume)
        if volume.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {volume.shape[1]}")
        if any(size < _MIN_SPATIAL_SIZE for size in volume.shape[2:]):
            raise ValueError("each spatial dimension must be at least 16 for four downsampling stages")
        raw_params = self.encoder(volume)
        scaled_params = self.ranges.scale(raw_params)
        sampling_matrix, inverse_sampling_matrix = build_transform_matrices(
            scaled_params, spatial_shape=tuple(volume.shape[2:])
        )
        aligned = warp_volume(volume, sampling_matrix)
        return AlignmentResult(
            aligned=aligned,
            raw_params=raw_params,
            scaled_params=scaled_params,
            sampling_matrix=sampling_matrix,
            inverse_sampling_matrix=inverse_sampling_matrix,
        )
