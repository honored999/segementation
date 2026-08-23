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


def _load_model_type():
    try:
        from lightweight_aspp_2d import LightweightASPP2D
    except ModuleNotFoundError as error:
        pytest.fail(
            "LightweightASPP2D is not implemented yet; expected RED from the "
            f"missing module: {error}"
        )
    return LightweightASPP2D


def _load_trainer_types():
    pytest.importorskip("nnunetv2")

    from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import (
        nnUNetTrainerNoDeepSupervision,
    )

    from nnunet_ext_trainers.nnUNetTrainerStage2LightweightASPP import (
        nnUNetTrainerStage2LightweightASPP,
    )

    return nnUNetTrainerNoDeepSupervision, nnUNetTrainerStage2LightweightASPP


def _conv_spec(module: nn.Conv2d) -> tuple[object, ...]:
    return (
        module.in_channels,
        module.out_channels,
        module.kernel_size,
        module.dilation,
        module.padding,
        module.stride,
        module.groups,
        module.bias is not None,
    )


def test_lightweight_aspp_has_exact_multiscale_module_structure() -> None:
    model_type = _load_model_type()
    model = model_type(in_channels=1, num_classes=2)

    assert [type(module) for module in model.children()] == [
        nn.Sequential,
        nn.ModuleList,
        nn.Sequential,
        nn.Conv2d,
    ]
    assert [type(module) for module in model.stem.children()] == [nn.Conv2d, nn.ReLU]
    assert _conv_spec(model.stem[0]) == (1, 32, (3, 3), (1, 1), (1, 1), (1, 1), 1, True)

    assert len(model.branches) == 4
    assert all(isinstance(branch, nn.Conv2d) for branch in model.branches)
    assert [_conv_spec(branch) for branch in model.branches] == [
        (32, 16, (1, 1), (1, 1), (0, 0), (1, 1), 1, True),
        (32, 16, (3, 3), (1, 1), (1, 1), (1, 1), 1, True),
        (32, 16, (3, 3), (2, 2), (2, 2), (1, 1), 1, True),
        (32, 16, (3, 3), (4, 4), (4, 4), (1, 1), 1, True),
    ]
    assert all(len(list(branch.children())) == 0 for branch in model.branches)

    assert [type(module) for module in model.fusion.children()] == [nn.Conv2d, nn.ReLU]
    assert _conv_spec(model.fusion[0]) == (
        64,
        32,
        (3, 3),
        (1, 1),
        (1, 1),
        (1, 1),
        1,
        True,
    )
    assert _conv_spec(model.classifier) == (
        32,
        2,
        (1, 1),
        (1, 1),
        (0, 0),
        (1, 1),
        1,
        True,
    )
    forbidden_names = {"MaxPool2d", "AvgPool2d", "AdaptiveAvgPool2d", "MultiheadAttention"}
    assert all(type(module).__name__ not in forbidden_names for module in model.modules())


def test_lightweight_aspp_keeps_all_branch_spatial_shapes_identical() -> None:
    model_type = _load_model_type()
    model = model_type(in_channels=1, num_classes=2)
    branch_shapes: list[tuple[int, ...]] = []
    handles = [
        branch.register_forward_hook(
            lambda _module, _inputs, output: branch_shapes.append(tuple(output.shape))
        )
        for branch in model.branches
    ]

    try:
        logits = model(torch.randn(2, 1, 17, 23))
    finally:
        for handle in handles:
            handle.remove()

    assert branch_shapes == [(2, 16, 17, 23)] * 4
    assert logits.shape == (2, 2, 17, 23)


def test_lightweight_aspp_concatenates_64_channels_without_resizing() -> None:
    model_type = _load_model_type()
    model = model_type(in_channels=1, num_classes=2)
    fusion_input_shapes: list[tuple[int, ...]] = []
    handle = model.fusion[0].register_forward_pre_hook(
        lambda _module, inputs: fusion_input_shapes.append(tuple(inputs[0].shape))
    )

    try:
        model(torch.randn(2, 1, 11, 19))
    finally:
        handle.remove()

    assert fusion_input_shapes == [(2, 64, 11, 19)]


def test_lightweight_aspp_returns_raw_logits_with_dataset504_dwi_batch() -> None:
    model_type = _load_model_type()
    model = model_type(in_channels=1, num_classes=2)
    data = torch.randn(2, 1, 9, 13)
    target = torch.randint(0, 2, (2, 1, 9, 13), dtype=torch.long)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.classifier.bias.copy_(torch.tensor([2.0, -1.0]))

    logits = model(data)
    expected = torch.tensor([2.0, -1.0]).view(1, 2, 1, 1).expand_as(logits)

    assert data.shape == (2, 1, 9, 13)
    assert logits.shape == (2, 2, 9, 13)
    assert logits.shape[2:] == target.shape[2:]
    assert torch.equal(logits, expected)
    assert logits.min().item() < 0.0 < 1.0 < logits.max().item()
    assert target.shape == (2, 1, 9, 13)
    assert target.dtype == torch.long


def test_lightweight_aspp_cross_entropy_backward_is_finite_for_dwi_batch() -> None:
    model_type = _load_model_type()
    model = model_type(in_channels=1, num_classes=2)
    data = torch.randn(2, 1, 9, 13)
    target = torch.randint(0, 2, (2, 1, 9, 13), dtype=torch.long)

    logits = model(data)
    assert logits.shape[2:] == target.shape[2:]

    loss = nn.CrossEntropyLoss()(logits, target[:, 0])
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_lightweight_aspp_parameter_count_is_below_50000() -> None:
    model_type = _load_model_type()
    model = model_type(in_channels=1, num_classes=2)

    assert sum(parameter.numel() for parameter in model.parameters()) < 50_000


def test_trainer_inherits_no_deep_supervision_and_builds_lightweight_aspp() -> None:
    no_deep_supervision, trainer = _load_trainer_types()
    model_type = _load_model_type()

    assert issubclass(trainer, no_deep_supervision)
    model = trainer.build_network_architecture(
        plans_manager=object(),
        configuration_manager=SimpleNamespace(patch_size=(32, 24)),
        num_input_channels=1,
        num_output_channels=2,
        enable_deep_supervision=False,
    )

    assert isinstance(model, model_type)
    assert model.stem[0].in_channels == 1
    assert model.classifier.out_channels == 2


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
