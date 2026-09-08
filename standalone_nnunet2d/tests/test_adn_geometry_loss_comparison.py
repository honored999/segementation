"""Synthetic tests for the read-only geometry-vs-loss diagnostic."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from standalone_nnunet2d.brain_alignment.adn_transform import (
    AlignmentResult,
    build_transform_matrices,
    warp_volume,
)
from standalone_nnunet2d.brain_alignment.nifti_adapter import canonicalize_nifti
from standalone_nnunet2d.data.nifti_io import NiftiVolume
from standalone_nnunet2d.tools import adn_geometry_loss_comparison as comparison


def _ellipse_slice(
    *,
    height: int = 80,
    width: int = 96,
    center_x: float | None = None,
    center_y: float | None = None,
    angle_degree: float = 0.0,
    radius_x: float = 20.0,
    radius_y: float = 8.0,
) -> np.ndarray:
    center_x = (width - 1) / 2 if center_x is None else center_x
    center_y = (height - 1) / 2 if center_y is None else center_y
    yy, xx = np.mgrid[:height, :width]
    angle = np.deg2rad(angle_degree)
    cosine, sine = np.cos(angle), np.sin(angle)
    x = xx - center_x
    y = yy - center_y
    major = cosine * x + sine * y
    minor = -sine * x + cosine * y
    return (((major / radius_x) ** 2 + (minor / radius_y) ** 2) <= 1.0).astype(np.float32)


def _ellipse_volume(
    *,
    depth: int = 9,
    center_x: float | None = None,
    angle_degree: float = 0.0,
    offsets: list[float] | None = None,
    angles: list[float] | None = None,
) -> np.ndarray:
    offsets = [0.0] * depth if offsets is None else offsets
    angles = [angle_degree] * depth if angles is None else angles
    base_x = (96 - 1) / 2 if center_x is None else center_x
    return np.stack(
        [
            _ellipse_slice(center_x=base_x + offsets[index], angle_degree=angles[index])
            for index in range(depth)
        ],
        axis=0,
    )


def _centroid_x(array: np.ndarray) -> float:
    coordinates = np.argwhere(array > 0.5)
    return float(coordinates[:, 1].mean())


def test_known_translated_ellipse_recovers_centroid_and_lr_offset() -> None:
    expected_center_x = (96 - 1) / 2 + 7.5
    estimate = comparison.estimate_slice_geometry(
        _ellipse_slice(center_x=expected_center_x), slice_index=4, percent=50
    )

    assert estimate.centroid_x == pytest.approx(expected_center_x, abs=0.35)
    assert estimate.lr_centroid_offset_pixels == pytest.approx(7.5, abs=0.35)


def test_known_rotated_ellipse_recovers_principal_axis_angle() -> None:
    estimate = comparison.estimate_slice_geometry(
        _ellipse_slice(angle_degree=27.0), slice_index=4, percent=50
    )

    assert estimate.principal_axis_angle_degree == pytest.approx(27.0, abs=1.5)


def test_case_geometry_uses_the_three_slice_median() -> None:
    volume = _ellipse_volume(
        offsets=[100.0] * 9,
        angles=[-60.0] * 9,
    )
    selected = comparison.slice_indices(volume.shape[0])
    for item, offset, angle in zip(
        selected,
        [4.0, 6.0, 8.0],
        [10.0, 14.0, 18.0],
        strict=True,
    ):
        volume[item.index] = _ellipse_slice(
            center_x=(96 - 1) / 2 + offset,
            angle_degree=angle,
        )

    estimate = comparison.estimate_case_geometry(volume)

    assert [item.percent for item in estimate.slices] == [25, 50, 75]
    assert [item.index for item in estimate.slices] == [2, 4, 6]
    assert estimate.estimated_content_lr_translation_pixels == pytest.approx(6.0, abs=0.4)
    assert estimate.estimated_content_rotation_degree == pytest.approx(14.0, abs=1.5)


def test_case_geometry_angle_median_is_axial_aware_across_vertical_boundary() -> None:
    volume = _ellipse_volume(depth=9)
    selected = comparison.slice_indices(volume.shape[0])
    for item, angle in zip(selected, [89.0, 0.0, -89.0], strict=True):
        volume[item.index] = _ellipse_slice(angle_degree=angle)

    estimate = comparison.estimate_case_geometry(volume)

    assert -90.0 <= estimate.estimated_content_rotation_degree < 90.0
    assert abs(abs(estimate.estimated_content_rotation_degree) - 90.0) < 3.0


def test_forward_content_transform_is_converted_to_inverse_sampling_and_roundtrips() -> None:
    volume = torch.zeros(1, 1, 16, 32, 40)
    volume[:, :, 7:9, 15:17, 10:12] = 1.0
    forward = comparison.build_forward_content_transform(
        rotation_degree=0.0,
        lr_translation_pixels=3.0,
        spatial_shape=(16, 32, 40),
    )
    sampling, inverse_sampling = comparison.content_transform_to_sampling_matrices(
        forward, spatial_shape=(16, 32, 40)
    )

    moved = warp_volume(volume, sampling, mode="nearest")
    restored = warp_volume(moved, inverse_sampling, mode="nearest")

    assert _centroid_x(moved[0, 0, 8].numpy()) > _centroid_x(volume[0, 0, 8].numpy())
    assert torch.equal(restored, volume)
    assert torch.allclose(sampling @ inverse_sampling, torch.eye(4).unsqueeze(0), atol=1e-6)


def test_identity_geometry_transform_does_not_change_volume() -> None:
    torch.manual_seed(11)
    volume = torch.rand(1, 1, 16, 32, 40)
    forward = comparison.build_forward_content_transform(
        rotation_degree=0.0,
        lr_translation_pixels=0.0,
        spatial_shape=tuple(volume.shape[-3:]),
    )
    sampling, _ = comparison.content_transform_to_sampling_matrices(
        forward, spatial_shape=tuple(volume.shape[-3:])
    )

    assert torch.equal(sampling, torch.eye(4).unsqueeze(0))
    torch.testing.assert_close(warp_volume(volume, sampling), volume, atol=3e-6, rtol=0)


def test_geometry_correction_reduces_synthetic_ellipse_centroid_and_tilt() -> None:
    volume = _ellipse_volume(depth=16, offsets=[7.0] * 16, angles=[25.0] * 16)
    before = comparison.estimate_case_geometry(volume)
    correction = comparison.build_geometry_correction_transform(
        before, spatial_shape=(16, 80, 96)
    )
    sampling, _ = comparison.content_transform_to_sampling_matrices(
        correction, spatial_shape=(16, 80, 96)
    )
    aligned = warp_volume(torch.from_numpy(volume)[None, None], sampling, mode="bilinear")
    after = comparison.estimate_case_geometry(aligned[0, 0].numpy())

    assert abs(after.estimated_content_lr_translation_pixels) < abs(
        before.estimated_content_lr_translation_pixels
    )
    assert abs(after.estimated_content_rotation_degree) < abs(
        before.estimated_content_rotation_degree
    )


class _FixedIdentityAligner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_channels = 1

    def forward(self, volume: torch.Tensor) -> AlignmentResult:
        parameters = torch.zeros(volume.shape[0], 6, dtype=volume.dtype, device=volume.device)
        matrix, inverse = build_transform_matrices(parameters, spatial_shape=tuple(volume.shape[-3:]))
        return AlignmentResult(
            aligned=warp_volume(volume, matrix),
            raw_params=parameters,
            scaled_params=parameters,
            sampling_matrix=matrix,
            inverse_sampling_matrix=inverse,
        )


def test_cli_writes_required_outputs_without_labels_or_training_when_checkpoint_is_omitted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = tmp_path / "dataset"
    image_path = dataset / "imagesTr" / "case001_0000.nii.gz"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"synthetic image placeholder")
    source = NiftiVolume(
        _ellipse_volume(depth=9),
        spacing_xyz=(0.7, 0.8, 4.5),
        origin_xyz=(11.0, -2.0, 3.5),
    )
    calls: list[str] = []
    original_normalize = comparison.normalize_volume
    original_pad = comparison.pad_model_input_depth

    def read(path: Path) -> NiftiVolume:
        calls.append("read_nifti")
        assert "labels" not in str(path).lower()
        return source

    def canonicalize(volume: NiftiVolume):
        calls.append("canonicalize_nifti")
        return canonicalize_nifti(volume)

    def normalize(array: np.ndarray) -> np.ndarray:
        calls.append("normalize_volume")
        return original_normalize(array)

    def pad(tensor: torch.Tensor):
        calls.append("pad_model_input_depth")
        return original_pad(tensor)

    monkeypatch.setattr(comparison, "read_nifti", read)
    monkeypatch.setattr(comparison, "canonicalize_nifti", canonicalize)
    monkeypatch.setattr(comparison, "normalize_volume", normalize)
    monkeypatch.setattr(comparison, "pad_model_input_depth", pad)
    monkeypatch.setattr(
        comparison,
        "load_diagnostic_checkpoint",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("checkpoint should not load")),
    )
    output = tmp_path / "comparison-output"

    assert comparison.main(
        [
            "--dataset-dir", str(dataset),
            "--cases", "case001",
            "--device", "cpu",
            "--output-dir", str(output),
        ]
    ) == 0

    assert calls == ["read_nifti", "canonicalize_nifti", "normalize_volume", "pad_model_input_depth"]
    case_output = output / "case001"
    assert (case_output / "summary.json").is_file()
    assert (case_output / "comparison.csv").is_file()
    assert (case_output / "geometry_qc.png").stat().st_size > 0
    summary = json.loads((case_output / "summary.json").read_text(encoding="utf-8"))
    assert summary["identity_total_loss"] is not None
    assert summary["adn_total_loss"] is None
    assert summary["geometry_aligned_total_loss"] is not None
    assert summary["adn_available"] is False
    assert summary["labels_accessed"] is False
    assert summary["training_called"] is False
    with (case_output / "comparison.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["state"] for row in rows] == ["identity", "adn_prediction", "geometry_estimate"]
    assert rows[1]["status"] == "not_requested"


def test_cli_uses_optional_checkpoint_prediction_and_does_not_call_training(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = tmp_path / "dataset"
    image_path = dataset / "imagesTr" / "case002_0000.nii.gz"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"synthetic image placeholder")
    source = NiftiVolume(_ellipse_volume(depth=9), (0.7, 0.8, 4.5), (0.0, 0.0, 0.0))
    calls: list[str] = []

    monkeypatch.setattr(comparison, "read_nifti", lambda _: source)
    monkeypatch.setattr(comparison, "load_diagnostic_checkpoint", lambda *args, **kwargs: SimpleNamespace(
        model=_FixedIdentityAligner(), metadata={}
    ))
    training_calls: list[str] = []
    monkeypatch.setattr(
        comparison,
        "train_fold0",
        lambda *args, **kwargs: training_calls.append("training"),
        raising=False,
    )
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"synthetic checkpoint placeholder")

    original_model = comparison.load_diagnostic_checkpoint

    def load(*args, **kwargs):
        calls.append("checkpoint")
        return original_model(*args, **kwargs)

    monkeypatch.setattr(comparison, "load_diagnostic_checkpoint", load)
    output = tmp_path / "comparison-output"

    assert comparison.main(
        [
            "--dataset-dir", str(dataset),
            "--cases", "case002",
            "--checkpoint", str(checkpoint),
            "--device", "cpu",
            "--output-dir", str(output),
        ]
    ) == 0

    assert calls == ["checkpoint"]
    summary = json.loads((output / "case002" / "summary.json").read_text(encoding="utf-8"))
    assert summary["adn_available"] is True
    assert summary["adn_total_loss"] is not None
    assert summary["checkpoint"] == str(checkpoint.resolve())
    assert training_calls == []
