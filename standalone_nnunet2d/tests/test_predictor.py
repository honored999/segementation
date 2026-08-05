from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from standalone_nnunet2d.data.nifti_io import NiftiVolume
from standalone_nnunet2d.engine.predictor import predict_logits_2d, predict_volume


class _BinaryModel(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.stack((-image[:, 0], image[:, 0]), dim=1)


class _OrientationModel(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        signal = image[:, :1]
        return torch.cat((torch.zeros_like(signal), signal), dim=1)


class _CountingOrientationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_batches: list[int] = []

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.forward_batches.append(int(image.shape[0]))
        signal = image[:, :1]
        return torch.cat((torch.zeros_like(signal), signal), dim=1)


class _FixedSpatialLogitModel(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        difference = torch.tensor(
            [[1.0, 1.0], [1.0, -5.0]], dtype=image.dtype, device=image.device
        ).view(1, 1, 2, 2)
        difference = difference.expand(image.shape[0], -1, -1, -1)
        return torch.cat((torch.zeros_like(difference), difference), dim=1)


def test_predict_logits_2d_unflips_each_mirror_before_averaging() -> None:
    image = torch.tensor([[[[2.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]]])

    logits = predict_logits_2d(
        _OrientationModel(), image, torch.device("cpu"), mirror_axes=(0, 1)
    )

    expected = torch.cat((torch.zeros_like(image), image), dim=1)
    torch.testing.assert_close(logits, expected)


def test_predict_volume_preserves_zyx_shape_and_binary_labels() -> None:
    image = NiftiVolume(np.array([[[-1.0, 1.0]], [[2.0, -2.0]]], dtype=np.float32), (1, 1, 1), (0, 0, 0))
    prediction = predict_volume(_BinaryModel(), image, torch.device("cpu"))
    assert prediction.shape == image.array.shape
    assert prediction.dtype == np.uint8
    assert set(np.unique(prediction)) <= {0, 1}


def test_predict_volume_slice_batch_matches_serial_orientation_sensitive_masks_and_reduces_forwards() -> None:
    image = NiftiVolume(
        np.arange(30, dtype=np.float32).reshape(5, 2, 3),
        (1, 1, 1),
        (0, 0, 0),
    )
    serial_model = _CountingOrientationModel()
    batched_model = _CountingOrientationModel()

    serial = predict_volume(
        serial_model,
        image,
        torch.device("cpu"),
        mirror_axes=(0, 1),
        patch_size=(2, 3),
    )
    batched = predict_volume(
        batched_model,
        image,
        torch.device("cpu"),
        mirror_axes=(0, 1),
        patch_size=(2, 3),
        slice_batch_size=2,
    )

    np.testing.assert_array_equal(batched, serial)
    assert batched.dtype == np.uint8
    assert len(batched_model.forward_batches) < len(serial_model.forward_batches)
    assert batched_model.forward_batches == [2, 2, 1] * 4


def test_predict_volume_slice_batch_argmaxes_after_all_tta_logits_are_averaged() -> None:
    image = NiftiVolume(np.zeros((3, 2, 2), dtype=np.float32), (1, 1, 1), (0, 0, 0))

    serial = predict_volume(
        _FixedSpatialLogitModel(),
        image,
        torch.device("cpu"),
        mirror_axes=(0, 1),
        patch_size=(2, 2),
    )
    batched = predict_volume(
        _FixedSpatialLogitModel(),
        image,
        torch.device("cpu"),
        mirror_axes=(0, 1),
        patch_size=(2, 2),
        slice_batch_size=2,
    )

    np.testing.assert_array_equal(batched, serial)
    np.testing.assert_array_equal(batched, np.zeros((3, 2, 2), dtype=np.uint8))


@pytest.mark.parametrize("slice_batch_size", [0, -1])
def test_predict_volume_rejects_nonpositive_slice_batch_size(slice_batch_size: int) -> None:
    image = NiftiVolume(np.zeros((1, 2, 2), dtype=np.float32), (1, 1, 1), (0, 0, 0))

    with pytest.raises(ValueError, match="slice_batch_size"):
        predict_volume(_BinaryModel(), image, torch.device("cpu"), slice_batch_size=slice_batch_size)
