from __future__ import annotations

import gc
import inspect
import re

import pytest
import torch
from torch import nn

from standalone_nnunet2d.models import h2former as h2former_module
from standalone_nnunet2d.models.h2former import H2Former


IMAGE_SIZE = 512


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_model() -> H2Former:
    return H2Former(in_channels=1, num_classes=2, image_size=IMAGE_SIZE).to(_device())


def test_h2former_import_is_self_contained_and_has_expected_parameter_scale() -> None:
    blocks_module = __import__(
        "standalone_nnunet2d.models.h2former_blocks", fromlist=["h2former_blocks"]
    )
    module_source = inspect.getsource(h2former_module)
    blocks_source = inspect.getsource(blocks_module)

    assert h2former_module.__file__ is not None
    assert "standalone_nnunet2d" in h2former_module.__file__
    for source in (module_source, blocks_source):
        assert not re.search(r"^\s*(?:from|import)\s+third_party", source, re.MULTILINE)
        assert "sys.path" not in source
        assert not re.search(r"^\s*(?:from|import)\s+timm", source, re.MULTILINE)

    model = _make_model()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"h2former_parameter_count={parameter_count}")
    assert 30_000_000 <= parameter_count <= 40_000_000
    del model
    gc.collect()


def test_h2former_zero_forward_returns_finite_full_resolution_logits() -> None:
    model = _make_model().eval()
    image = torch.zeros((1, 1, IMAGE_SIZE, IMAGE_SIZE), device=_device())

    with torch.inference_mode():
        logits = model(image)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (1, 2, IMAGE_SIZE, IMAGE_SIZE)
    assert torch.isfinite(logits).all()
    assert isinstance(model.decode0[-1], nn.Conv2d)
    assert not isinstance(model.decode0[-1], (nn.Softmax, nn.LogSoftmax, nn.Sigmoid))
    del logits, image, model
    gc.collect()


def test_h2former_backward_reaches_stem_transformer_and_decoder() -> None:
    model = _make_model().eval()
    image = torch.zeros((1, 1, IMAGE_SIZE, IMAGE_SIZE), device=_device())
    image[:, :, IMAGE_SIZE // 2, IMAGE_SIZE // 2] = 1.0

    logits = model(image)
    loss = logits.square().mean() + logits.mean()
    loss.backward()

    representative_names = (
        "conv1.weight",
        "swin_layers.0.blocks.0.attn.qkv.weight",
        "decode0.1.weight",
    )
    named_parameters = dict(model.named_parameters())
    for name in representative_names:
        gradient = named_parameters[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert torch.count_nonzero(gradient) > 0, name

    del loss, logits, image, model
    gc.collect()


def test_h2former_state_dict_round_trip_restores_forward(tmp_path: pytest.TempPathFactory) -> None:
    model = _make_model().eval()
    image = torch.zeros((1, 1, IMAGE_SIZE, IMAGE_SIZE), device=_device())
    with torch.inference_mode():
        reference = model(image).cpu()

    checkpoint_path = tmp_path / "h2former_state.pt"
    torch.save(model.state_dict(), checkpoint_path)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    restored = _make_model().eval()
    state_dict = torch.load(checkpoint_path, map_location=_device(), weights_only=True)
    restored.load_state_dict(state_dict)
    del state_dict
    with torch.inference_mode():
        actual = restored(image).cpu()

    assert actual.shape == reference.shape
    assert torch.isfinite(actual).all()
    assert torch.allclose(actual, reference, rtol=1e-5, atol=1e-6)
    del actual, reference, image, restored
    gc.collect()


def test_h2former_rejects_non_bchw_wrong_channels_and_wrong_spatial_size() -> None:
    model = _make_model().eval()
    invalid_inputs = (
        (torch.zeros((1, 1, IMAGE_SIZE)), "BCHW"),
        (torch.zeros((1, 2, IMAGE_SIZE, IMAGE_SIZE)), "in_channels"),
        (torch.zeros((1, 1, 256, IMAGE_SIZE)), "image_size"),
    )

    for image, message in invalid_inputs:
        with pytest.raises(ValueError, match=message):
            model(image.to(_device()))

    del model
    gc.collect()
