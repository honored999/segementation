from __future__ import annotations

import pytest
import torch
from torch import nn

from standalone_nnunet2d.models.lite_upernet import LiteUPerDecoder


def _batch_norm(channels: int) -> nn.Module:
    return nn.BatchNorm2d(channels)


def test_lite_uper_decoder_returns_exact_output_size_and_finite_logits() -> None:
    decoder = LiteUPerDecoder(
        in_channels=(8, 16, 32, 64),
        num_classes=2,
        fpn_channels=8,
        pool_scales=(1, 2, 4),
        norm_factory=_batch_norm,
    ).eval()
    features = (
        torch.randn(2, 8, 64, 64),
        torch.randn(2, 16, 32, 32),
        torch.randn(2, 32, 16, 16),
        torch.randn(2, 64, 8, 8),
    )
    with torch.inference_mode():
        logits = decoder(features, output_size=(128, 128))
    assert logits.shape == (2, 2, 128, 128)
    assert torch.isfinite(logits).all()
    assert not any(
        isinstance(module, (nn.Softmax, nn.LogSoftmax, nn.Sigmoid))
        for module in decoder.modules()
    )


def test_ppm_branches_are_normalization_free_and_support_one_by_one_pooling() -> None:
    decoder = LiteUPerDecoder(
        in_channels=(8, 16, 32, 64),
        num_classes=2,
        fpn_channels=8,
        pool_scales=(1, 2, 4),
        norm_factory=lambda channels: nn.InstanceNorm2d(channels, affine=True),
    ).train()
    features = (
        torch.randn(2, 8, 32, 32),
        torch.randn(2, 16, 16, 16),
        torch.randn(2, 32, 8, 8),
        torch.randn(2, 64, 4, 4),
    )
    logits = decoder(features, output_size=(64, 64))
    logits.square().mean().backward()
    assert logits.shape == (2, 2, 64, 64)
    assert all(
        not any(isinstance(module, nn.InstanceNorm2d) for module in branch.modules())
        for branch in decoder.ppm_branches
    )


@pytest.mark.parametrize(
    ("features", "output_size", "message"),
    [
        ((torch.zeros(1, 8, 16, 16),), (32, 32), "four"),
        ((torch.zeros(1, 8, 16),) * 4, (32, 32), "BCHW"),
        (
            (
                torch.zeros(1, 7, 16, 16),
                torch.zeros(1, 16, 8, 8),
                torch.zeros(1, 32, 4, 4),
                torch.zeros(1, 64, 2, 2),
            ),
            (32, 32),
            "channels",
        ),
        (
            (
                torch.zeros(1, 8, 8, 8),
                torch.zeros(1, 16, 16, 16),
                torch.zeros(1, 32, 4, 4),
                torch.zeros(1, 64, 2, 2),
            ),
            (32, 32),
            "highest to lowest",
        ),
        (
            (
                torch.zeros(1, 8, 16, 16),
                torch.zeros(2, 16, 8, 8),
                torch.zeros(1, 32, 4, 4),
                torch.zeros(1, 64, 2, 2),
            ),
            (32, 32),
            "batch",
        ),
        (
            (
                torch.zeros(1, 8, 16, 16),
                torch.zeros(1, 16, 8, 8),
                torch.zeros(1, 32, 4, 4),
                torch.zeros(1, 64, 2, 2),
            ),
            (0, 32),
            "output_size",
        ),
    ],
)
def test_lite_uper_decoder_rejects_invalid_contracts(features, output_size, message: str) -> None:
    decoder = LiteUPerDecoder((8, 16, 32, 64), 2, 8, (1, 2, 4), _batch_norm)
    with pytest.raises((TypeError, ValueError), match=message):
        decoder(features, output_size=output_size)
