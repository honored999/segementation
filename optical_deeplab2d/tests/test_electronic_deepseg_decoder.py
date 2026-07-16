import pytest

from optical_deeplab2d.models.backbone import build_deepseg_modules


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
