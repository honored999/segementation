import pytest
import torch

from optical_deeplab2d.models.backbone import build_densenet121_deepseg_encoder
from optical_deeplab2d.models.electronic_densenet_deepseg_decoder import (
    ElectronicDenseNetDeepSegDecoder,
)


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


def test_densenet_deepseg_returns_input_sized_logits_without_aspp() -> None:
    model = ElectronicDenseNetDeepSegDecoder(encoder_weights=None)

    logits = model(torch.randn(2, 1, 65, 81))

    assert logits.shape == (2, 1, 65, 81)
    assert model.resolved_encoder == "densenet121"
    assert model.context_module == "none"
    assert len(model.decoder_stages) == 4
    assert not any("aspp" in name.lower() for name, _ in model.named_parameters())
    assert not any("aspp" in name.lower() for name, _ in model.named_modules())


def test_densenet_deepseg_backpropagates_through_encoder_and_decoder() -> None:
    model = ElectronicDenseNetDeepSegDecoder(encoder_weights=None)

    model(torch.randn(2, 1, 64, 80)).square().mean().backward()

    for module in (model.encoder, model.decoder_stages):
        assert any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in module.parameters()
        )
