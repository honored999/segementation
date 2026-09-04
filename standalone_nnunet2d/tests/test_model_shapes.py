from __future__ import annotations

import torch

from standalone_nnunet2d.config import load_model_config
from standalone_nnunet2d.models.plain_conv_unet import PlainConvUNet2D


def _model(deep_supervision: bool) -> PlainConvUNet2D:
    return PlainConvUNet2D(load_model_config(), deep_supervision=deep_supervision).eval()


def test_plain_conv_unet_returns_full_resolution_logits_and_records_shapes() -> None:
    model = _model(deep_supervision=False)
    image = torch.randn(2, 1, 512, 512)

    with torch.inference_mode():
        logits = model(image)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (2, 2, 512, 512)
    assert model.last_encoder_shapes == (
        (2, 32, 512, 512), (2, 64, 256, 256), (2, 128, 128, 128),
        (2, 256, 64, 64), (2, 512, 32, 32), (2, 512, 16, 16),
        (2, 512, 8, 8), (2, 512, 4, 4),
    )
    assert model.last_decoder_shapes == (
        (2, 512, 8, 8), (2, 512, 16, 16), (2, 512, 32, 32),
        (2, 256, 64, 64), (2, 128, 128, 128), (2, 64, 256, 256),
        (2, 32, 512, 512),
    )
    assert not any(isinstance(module, (torch.nn.Softmax, torch.nn.LogSoftmax)) for module in model.modules())
    print(f"parameter_count={sum(parameter.numel() for parameter in model.parameters())}")


def test_deep_supervision_returns_main_logits_then_descending_auxiliary_scales() -> None:
    model = _model(deep_supervision=True)
    image = torch.randn(1, 1, 512, 512)

    with torch.inference_mode():
        outputs = model(image)

    assert isinstance(outputs, tuple)
    assert [tuple(output.shape) for output in outputs] == [
        (1, 2, 512, 512), (1, 2, 256, 256), (1, 2, 128, 128),
        (1, 2, 64, 64), (1, 2, 32, 32), (1, 2, 16, 16), (1, 2, 8, 8),
    ]
