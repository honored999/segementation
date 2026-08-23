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

from shallow_res_fcn_2d import ResidualBlock, ShallowResFCN2D


def _load_trainer_types():
    pytest.importorskip("nnunetv2")

    from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import (
        nnUNetTrainerNoDeepSupervision,
    )

    from nnunet_ext_trainers.nnUNetTrainerStage2Base import (
        nnUNetTrainerStage2Base,
    )
    from nnunet_ext_trainers.nnUNetTrainerStage2LiteFCN import (
        nnUNetTrainerStage2LiteFCN,
    )
    from nnunet_ext_trainers.nnUNetTrainerStage2ShallowResFCN import (
        nnUNetTrainerStage2ShallowResFCN,
    )
    from nnunet_ext_trainers.nnUNetTrainerStage2SmallUNet import (
        nnUNetTrainerStage2SmallUNet,
    )

    return (
        nnUNetTrainerNoDeepSupervision,
        nnUNetTrainerStage2Base,
        nnUNetTrainerStage2LiteFCN,
        nnUNetTrainerStage2ShallowResFCN,
        nnUNetTrainerStage2SmallUNet,
    )


def test_residual_block_uses_real_identity_addition() -> None:
    block = ResidualBlock(channels=32, dilation=2)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.zero_()

    image = torch.linspace(-1.0, 1.0, steps=32 * 5 * 7).reshape(1, 32, 5, 7)
    output = block(image)

    assert output.shape == image.shape
    assert torch.equal(output, torch.relu(image))


@pytest.mark.parametrize("dilation", [1, 2, 4])
def test_residual_block_dilation_and_padding_preserve_shape(dilation: int) -> None:
    block = ResidualBlock(channels=32, dilation=dilation)

    assert block.conv1.dilation == (dilation, dilation)
    assert block.conv1.padding == (dilation, dilation)
    assert block.conv2.dilation == (dilation, dilation)
    assert block.conv2.padding == (dilation, dilation)
    assert block(torch.randn(2, 32, 9, 13)).shape == (2, 32, 9, 13)


@pytest.mark.parametrize("height,width", [(1, 1), (17, 23)])
def test_shallow_res_fcn_forward_preserves_shape_and_returns_raw_logits(
    height: int, width: int
) -> None:
    model = ShallowResFCN2D(in_channels=3, num_classes=5)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.classifier.bias.copy_(torch.tensor([2.0, -1.0, 0.5, 0.0, -0.25]))

    logits = model(torch.zeros(2, 3, height, width))
    expected = torch.tensor([2.0, -1.0, 0.5, 0.0, -0.25]).view(1, 5, 1, 1)
    expected = expected.expand_as(logits)

    assert logits.shape == (2, 5, height, width)
    assert torch.equal(logits, expected)
    assert logits.min().item() < 0.0 < logits.max().item()


def test_shallow_res_fcn_has_only_32_channel_hidden_residual_path() -> None:
    model = ShallowResFCN2D(in_channels=3, num_classes=5)
    convolutions = [module for module in model.modules() if isinstance(module, nn.Conv2d)]

    assert len(model.residual_blocks) == 3
    assert [block.dilation for block in model.residual_blocks] == [1, 2, 4]
    assert [
        (convolution.in_channels, convolution.out_channels)
        for convolution in convolutions
    ] == [
        (3, 32),
        (32, 32),
        (32, 32),
        (32, 32),
        (32, 32),
        (32, 32),
        (32, 32),
        (32, 5),
    ]
    assert all(convolution.bias is not None for convolution in convolutions)
    assert not any(
        isinstance(
            module,
            (
                nn.MaxPool2d,
                nn.AvgPool2d,
                nn.BatchNorm2d,
                nn.InstanceNorm2d,
                nn.GroupNorm,
                nn.LayerNorm,
                nn.MultiheadAttention,
            ),
        )
        for module in model.modules()
    )
    assert not any(
        name in type(module).__name__.lower()
        for module in model.modules()
        for name in ("attention", "aspp", "decoder", "transformer", "unet")
    )


def test_shallow_res_fcn_supports_loss_and_finite_backward() -> None:
    model = ShallowResFCN2D(in_channels=1, num_classes=2)
    image = torch.randn(2, 1, 9, 13)
    target = torch.randint(0, 2, (2, 9, 13), dtype=torch.long)

    loss = nn.CrossEntropyLoss()(model(image), target)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_shallow_res_fcn_is_compatible_with_synthetic_nnunet_2d_batch() -> None:
    """Synthetic B,C,H,W/B,1,H,W smoke only; not evidence from real Dataset504."""
    model = ShallowResFCN2D(in_channels=1, num_classes=2)
    data = torch.randn(2, 1, 15, 21)
    target = torch.randint(0, 2, (2, 1, 15, 21), dtype=torch.long)

    logits = model(data)
    loss = nn.CrossEntropyLoss()(logits, target[:, 0])

    assert logits.shape[:2] == (2, 2)
    assert logits.shape[2:] == target.shape[2:]
    assert torch.isfinite(loss)


def test_all_stage2_trainers_share_the_lightweight_base_and_no_deep_supervision():
    (
        no_deep_supervision,
        shared_base,
        lite_trainer,
        shallow_trainer,
        small_trainer,
    ) = _load_trainer_types()

    assert issubclass(shared_base, no_deep_supervision)
    assert all(
        issubclass(trainer, shared_base)
        for trainer in (lite_trainer, shallow_trainer, small_trainer)
    )
    assert all(
        "build_network_architecture" not in trainer.__dict__
        for trainer in (lite_trainer, shallow_trainer, small_trainer)
    )


def test_shallow_trainer_builds_2d_network_from_plan_channels():
    _, _, _, trainer, _ = _load_trainer_types()

    model = trainer.build_network_architecture(
        plans_manager=object(),
        configuration_manager=SimpleNamespace(patch_size=(32, 24)),
        num_input_channels=3,
        num_output_channels=5,
        enable_deep_supervision=False,
    )

    assert isinstance(model, ShallowResFCN2D)
    convolutions = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    assert convolutions[0].in_channels == 3
    assert convolutions[-1].out_channels == 5


def test_shared_stage2_trainer_rejects_non_2d_patch_size():
    _, _, _, trainer, _ = _load_trainer_types()

    with pytest.raises(ValueError, match="2D"):
        trainer.build_network_architecture(
            plans_manager=object(),
            configuration_manager=SimpleNamespace(patch_size=(16, 16, 16)),
            num_input_channels=1,
            num_output_channels=2,
            enable_deep_supervision=False,
        )
