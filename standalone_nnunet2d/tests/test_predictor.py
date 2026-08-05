from __future__ import annotations

import numpy as np
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
