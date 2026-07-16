import pytest
import torch
from torch import nn

from optical_deeplab2d.models.backbone import build_deepseg_modules
from optical_deeplab2d.models.electronic_deepseg_decoder import ModifiedUNetDecoderStage


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
