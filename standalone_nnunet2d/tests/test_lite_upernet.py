from __future__ import annotations

import pytest
import torch
from torch import nn

from standalone_nnunet2d.config import load_model_config
from standalone_nnunet2d.losses.compound import DiceCrossEntropyLoss
from standalone_nnunet2d.models.h2former import H2Former
from standalone_nnunet2d.models.lite_upernet import LiteUPerDecoder
from standalone_nnunet2d.models.h2former_lite_upernet import H2FormerLiteUPerNet
from standalone_nnunet2d.models.plain_conv_unet import PlainConvUNet2D
from standalone_nnunet2d.models.plain_conv_unet_lite_upernet import PlainConvUNetLiteUPerNet


def _batch_norm(channels: int) -> nn.Module:
    return nn.BatchNorm2d(channels)


def _parameters(modules) -> int:
    return sum(parameter.numel() for module in modules for parameter in module.parameters())


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


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_h2former_lite_upernet_returns_selected_features_and_full_resolution_logits() -> None:
    model = H2FormerLiteUPerNet(in_channels=1, num_classes=2, image_size=512).to(_device()).eval()
    image = torch.zeros((1, 1, 512, 512), device=_device())
    with torch.inference_mode():
        logits = model(image)

    assert logits.shape == (1, 2, 512, 512)
    assert torch.isfinite(logits).all()
    assert model.last_feature_shapes == (
        (1, 64, 256, 256),
        (1, 128, 128, 128),
        (1, 256, 64, 64),
        (1, 512, 32, 32),
    )


def test_h2former_lite_upernet_backward_reaches_encoder_and_decoder() -> None:
    model = H2FormerLiteUPerNet(in_channels=1, num_classes=2, image_size=512).to(_device()).eval()
    image = torch.zeros((1, 1, 512, 512), device=_device())
    image[:, :, 256, 256] = 1.0
    logits = model(image)
    loss = logits.square().mean() + logits.mean()
    loss.backward()

    named_parameters = dict(model.named_parameters())
    for name in (
        "conv1.weight",
        "swin_layers.0.blocks.0.attn.qkv.weight",
        "swin_layers.3.blocks.0.attn.qkv.weight",
        "decoder.classifier.weight",
    ):
        gradient = named_parameters[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert torch.count_nonzero(gradient) > 0, name


def test_plain_conv_unet_lite_upernet_returns_selected_features_and_logits() -> None:
    model = PlainConvUNetLiteUPerNet(load_model_config()).eval()
    image = torch.zeros((1, 1, 512, 512))
    with torch.inference_mode():
        logits = model(image)

    assert logits.shape == (1, 2, 512, 512)
    assert torch.isfinite(logits).all()
    assert model.last_encoder_shapes == (
        (1, 32, 512, 512),
        (1, 64, 256, 256),
        (1, 128, 128, 128),
        (1, 256, 64, 64),
        (1, 512, 32, 32),
        (1, 512, 16, 16),
        (1, 512, 8, 8),
        (1, 512, 4, 4),
    )
    assert model.last_selected_feature_shapes == (
        (1, 64, 256, 256),
        (1, 256, 64, 64),
        (1, 512, 16, 16),
        (1, 512, 4, 4),
    )


def test_plain_conv_unet_lite_upernet_backward_reaches_all_encoder_stages() -> None:
    model = PlainConvUNetLiteUPerNet(load_model_config()).train()
    image = torch.zeros((1, 1, 512, 512))
    image[:, :, 256, 256] = 1.0
    target = torch.zeros((1, 512, 512), dtype=torch.long)
    logits = model(image)
    loss = DiceCrossEntropyLoss()(logits, target)
    loss.backward()

    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    named_parameters = dict(model.named_parameters())
    for prefix in (
        "encoder_stages.0",
        "encoder_stages.2",
        "encoder_stages.4",
        "encoder_stages.6",
        "encoder_stages.7",
        "decoder.classifier",
    ):
        gradients = [
            parameter.grad
            for name, parameter in named_parameters.items()
            if name.startswith(prefix)
        ]
        assert gradients, prefix
        assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients), prefix
        assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients if gradient is not None), prefix


def test_lite_upernet_decoders_have_fewer_parameters_than_baseline_decoders() -> None:
    config = load_model_config()
    h2_baseline = H2Former(in_channels=1, num_classes=2, image_size=512)
    h2_variant = H2FormerLiteUPerNet(in_channels=1, num_classes=2, image_size=512)
    plain_baseline = PlainConvUNet2D(config)
    plain_variant = PlainConvUNetLiteUPerNet(config)

    h2_baseline_decoder = _parameters(
        (h2_baseline.decode4, h2_baseline.decode3, h2_baseline.decode2, h2_baseline.decode0)
    )
    plain_baseline_decoder = _parameters(
        (
            plain_baseline.transposed_convolutions,
            plain_baseline.decoder_stages,
            plain_baseline.segmentation_heads,
        )
    )
    h2_lite_decoder = _parameters((h2_variant.decoder,))
    plain_lite_decoder = _parameters((plain_variant.decoder,))

    print(f"h2_baseline_decoder_parameters={h2_baseline_decoder}")
    print(f"h2_lite_decoder_parameters={h2_lite_decoder}")
    print(f"plain_baseline_decoder_parameters={plain_baseline_decoder}")
    print(f"plain_lite_decoder_parameters={plain_lite_decoder}")

    assert h2_lite_decoder < h2_baseline_decoder
    assert plain_lite_decoder < plain_baseline_decoder
