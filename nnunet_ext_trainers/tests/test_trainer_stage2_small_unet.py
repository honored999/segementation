from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
if str(EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTENSION_ROOT))

from small_unet_2d import ConvBlock2D, SmallUNet2D


def _load_trainer_types():
    pytest.importorskip("nnunetv2")

    from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import (
        nnUNetTrainerNoDeepSupervision,
    )

    from nnunet_ext_trainers.nnUNetTrainerStage2SmallUNet import (
        nnUNetTrainerStage2SmallUNet,
    )

    return nnUNetTrainerNoDeepSupervision, nnUNetTrainerStage2SmallUNet


@pytest.mark.parametrize("height,width", [(17, 23), (31, 18), (9, 13)])
def test_small_unet_preserves_arbitrary_valid_roi_shape(height: int, width: int) -> None:
    model = SmallUNet2D(in_channels=1, num_classes=2)

    logits = model(torch.randn(2, 1, height, width))

    assert logits.shape == (2, 2, height, width)


def test_small_unet_skip_concats_have_expected_channels_and_sizes() -> None:
    model = SmallUNet2D(in_channels=1, num_classes=2)
    captured_shapes: dict[str, tuple[int, ...]] = {}

    def capture_shape(name: str):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            captured_shapes[name] = tuple(inputs[0].shape)

        return hook

    hooks = [
        model.decoder32.register_forward_pre_hook(capture_shape("decoder32")),
        model.decoder16.register_forward_pre_hook(capture_shape("decoder16")),
    ]
    try:
        model(torch.randn(1, 1, 17, 23))
    finally:
        for hook in hooks:
            hook.remove()

    assert captured_shapes == {
        "decoder32": (1, 96, 8, 11),
        "decoder16": (1, 48, 17, 23),
    }


def test_small_unet_returns_raw_logits() -> None:
    model = SmallUNet2D(in_channels=1, num_classes=2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.final_conv.bias.copy_(torch.tensor([2.0, -1.0]))

    logits = model(torch.zeros(1, 1, 9, 13))
    expected = torch.tensor([2.0, -1.0]).view(1, 2, 1, 1).expand_as(logits)

    assert torch.equal(logits, expected)
    assert logits.min().item() < 0.0 < 1.0 < logits.max().item()


def test_small_unet_supports_classification_loss_and_finite_backward() -> None:
    model = SmallUNet2D(in_channels=1, num_classes=2)
    image = torch.randn(2, 1, 15, 21)
    target = torch.randint(0, 2, (2, 15, 21), dtype=torch.long)

    loss = nn.CrossEntropyLoss()(model(image), target)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_small_unet_is_compatible_with_synthetic_nnunet_2d_batch() -> None:
    """Synthetic B,C,H,W/B,1,H,W smoke only; not Dataset504 evidence."""
    model = SmallUNet2D(in_channels=1, num_classes=2)
    data = torch.randn(2, 1, 15, 21)
    target = torch.randint(0, 2, (2, 1, 15, 21), dtype=torch.long)

    logits = model(data)
    loss = nn.CrossEntropyLoss()(logits, target[:, 0])

    assert logits.shape[:2] == (2, 2)
    assert logits.shape[2:] == target.shape[2:]
    assert torch.isfinite(loss)


def test_small_unet_has_exact_small_unet_structure() -> None:
    model = SmallUNet2D(in_channels=1, num_classes=2)

    blocks = [module for module in model.modules() if isinstance(module, ConvBlock2D)]
    assert len(blocks) == 5
    for block in blocks:
        assert [type(module) for module in block.children()] == [
            nn.Conv2d,
            nn.ReLU,
            nn.Conv2d,
            nn.ReLU,
        ]
        convolutions = [module for module in block.children() if isinstance(module, nn.Conv2d)]
        assert all(
            convolution.kernel_size == (3, 3)
            and convolution.stride == (1, 1)
            and convolution.dilation == (1, 1)
            and convolution.padding == (1, 1)
            for convolution in convolutions
        )

    assert len([module for module in model.modules() if isinstance(module, nn.MaxPool2d)]) == 2
    assert not any(
        isinstance(
            module,
            (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm, nn.LayerNorm),
        )
        for module in model.modules()
    )
    forbidden_names = ("attention", "residual", "transformer", "aspp")
    assert not any(
        any(name in type(module).__name__.lower() for name in forbidden_names)
        for module in model.modules()
    )

    convolutions = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    assert [
        (convolution.in_channels, convolution.out_channels)
        for convolution in convolutions
    ] == [
        (1, 16),
        (16, 16),
        (16, 32),
        (32, 32),
        (32, 64),
        (64, 64),
        (96, 32),
        (32, 32),
        (48, 16),
        (16, 16),
        (16, 2),
    ]


def test_small_unet_has_reproducible_small_parameter_count() -> None:
    model = SmallUNet2D(in_channels=1, num_classes=2)

    assert sum(parameter.numel() for parameter in model.parameters()) == 118_002


def test_trainer_inherits_no_deep_supervision() -> None:
    no_deep_supervision, trainer = _load_trainer_types()

    assert issubclass(trainer, no_deep_supervision)


def test_trainer_build_hook_uses_channels_for_2d_configuration() -> None:
    _, trainer = _load_trainer_types()

    model = trainer.build_network_architecture(
        plans_manager=object(),
        configuration_manager=SimpleNamespace(patch_size=(32, 24)),
        num_input_channels=3,
        num_output_channels=5,
        enable_deep_supervision=False,
    )

    assert isinstance(model, SmallUNet2D)
    convolutions = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    assert convolutions[0].in_channels == 3
    assert convolutions[-1].out_channels == 5


def test_trainer_rejects_non_2d_patch_size() -> None:
    _, trainer = _load_trainer_types()

    with pytest.raises(ValueError, match="2D"):
        trainer.build_network_architecture(
            plans_manager=object(),
            configuration_manager=SimpleNamespace(patch_size=(16, 16, 16)),
            num_input_channels=1,
            num_output_channels=2,
            enable_deep_supervision=False,
        )
