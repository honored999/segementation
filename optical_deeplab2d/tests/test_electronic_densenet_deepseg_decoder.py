import pytest

from optical_deeplab2d.models.backbone import build_densenet121_deepseg_encoder


def test_densenet121_deepseg_factory_returns_four_downsampling_encoder() -> None:
    encoder, resolved_encoder = build_densenet121_deepseg_encoder(None)

    assert resolved_encoder == "densenet121"
    assert len(encoder.out_channels) == 5
    assert encoder.out_channels[0] == 3


def test_densenet121_deepseg_factory_does_not_fallback_when_encoder_build_fails(
    monkeypatch,
) -> None:
    import segmentation_models_pytorch as smp

    def fail_get_encoder(*args, **kwargs):
        raise ValueError("DenseNet unavailable")

    monkeypatch.setattr(smp.encoders, "get_encoder", fail_get_encoder)

    with pytest.raises(RuntimeError, match="DenseNet121 encoder could not be created"):
        build_densenet121_deepseg_encoder(None)
