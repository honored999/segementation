import pytest
import torch
from torch import nn

from optical_deeplab2d.models.backbone import build_deepseg_modules
from optical_deeplab2d.models.electronic_deepseg_decoder import (
    ElectronicDeepSegDecoder,
    ModifiedUNetDecoderStage,
)


def test_build_deepseg_modules_exposes_mobilenet_encoder_and_aspp():
    encoder, aspp, resolved = build_deepseg_modules("mobilenet_v2", None)

    assert resolved == "mobilenet_v2"
    assert encoder.out_channels == [3, 16, 24, 32, 96, 1280]
    assert aspp is not None


def test_build_deepseg_modules_falls_back_to_resnet18_for_unavailable_encoder():
    with pytest.warns(RuntimeWarning):
        encoder, aspp, resolved = build_deepseg_modules("unavailable_encoder", None)

    assert resolved == "resnet18"
    assert encoder is not None
    assert aspp is not None


def test_decoder_stage_upsamples_concatenates_skip_and_refines_twice():
    stage = ModifiedUNetDecoderStage(8, 5, 4)

    output = stage(torch.randn(2, 8, 4, 5), torch.randn(2, 5, 9, 11))

    assert output.shape == (2, 4, 9, 11)
    assert sum(isinstance(module, nn.Conv2d) for module in stage.refine) == 2
    assert sum(isinstance(module, nn.BatchNorm2d) for module in stage.refine) == 2
    assert sum(isinstance(module, nn.ReLU) for module in stage.refine) == 2


def test_electronic_deepseg_decoder_restores_odd_grayscale_spatial_shape():
    model = ElectronicDeepSegDecoder(encoder_weights=None)

    logits = model(torch.randn(2, 1, 65, 81))

    assert logits.shape == (2, 1, 65, 81)
    assert len(model.decoder_stages) == 4
    assert not any("optical" in name for name, _ in model.named_parameters())


def test_electronic_deepseg_decoder_backpropagates_through_encoder_aspp_and_decoder():
    model = ElectronicDeepSegDecoder(encoder_weights=None)

    model(torch.randn(2, 1, 64, 80)).square().mean().backward()

    for module in (model.encoder, model.aspp, model.decoder_stages):
        assert any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in module.parameters()
        )


def test_electronic_deepseg_decoder_trains_with_a_single_odd_sized_image():
    model = ElectronicDeepSegDecoder(encoder_weights=None)
    model.train()

    logits = model(torch.randn(1, 1, 65, 81))

    assert logits.shape == (1, 1, 65, 81)


def test_electronic_deepseg_decoder_supports_resnet18_gradients():
    model = ElectronicDeepSegDecoder(encoder_name="resnet18", encoder_weights=None)

    logits = model(torch.randn(2, 1, 65, 81))
    logits.square().mean().backward()

    assert logits.shape == (2, 1, 65, 81)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for module in (model.decoder_stages, model.aspp)
        for parameter in module.parameters()
    )


def test_electronic_deepseg_decoder_falls_back_to_resnet18_with_gradients():
    with pytest.warns(RuntimeWarning):
        model = ElectronicDeepSegDecoder(
            encoder_name="unavailable_encoder", encoder_weights=None
        )

    logits = model(torch.randn(2, 1, 65, 81))
    logits.square().mean().backward()

    assert model.resolved_encoder == "resnet18"
    assert logits.shape == (2, 1, 65, 81)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.aspp.parameters()
    )
