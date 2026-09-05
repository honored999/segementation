from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
if str(EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTENSION_ROOT))

from lite_fcn_2d import LiteFCN2D


def _load_trainer_types():
    try:
        from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import (
            nnUNetTrainerNoDeepSupervision,
        )
    except ModuleNotFoundError as error:
        if error.name != "nnunetv2":
            raise

        package_names = (
            "nnunetv2",
            "nnunetv2.training",
            "nnunetv2.training.nnUNetTrainer",
            "nnunetv2.training.nnUNetTrainer.variants",
            "nnunetv2.training.nnUNetTrainer.variants.network_architecture",
        )
        for package_name in package_names:
            package = ModuleType(package_name)
            package.__path__ = []
            sys.modules[package_name] = package

        class nnUNetTrainerNoDeepSupervision:
            def set_deep_supervision_enabled(self, enabled: bool) -> None:
                self.network.decoder.deep_supervision = enabled

        base_module_name = (
            "nnunetv2.training.nnUNetTrainer.variants.network_architecture."
            "nnUNetTrainerNoDeepSupervision"
        )
        base_module = ModuleType(base_module_name)
        base_module.nnUNetTrainerNoDeepSupervision = nnUNetTrainerNoDeepSupervision
        sys.modules[base_module_name] = base_module

    from nnunet_ext_trainers.nnUNetTrainerStage2LiteFCN import (
        nnUNetTrainerStage2LiteFCN,
    )

    return nnUNetTrainerNoDeepSupervision, nnUNetTrainerStage2LiteFCN


def test_lite_fcn_has_exact_conv_relu_sequence() -> None:
    model = LiteFCN2D(in_channels=3, num_classes=5)

    operators = list(model.layers.children())

    assert [type(module) for module in operators] == [
        nn.Conv2d,
        nn.ReLU,
        nn.Conv2d,
        nn.ReLU,
        nn.Conv2d,
        nn.ReLU,
        nn.Conv2d,
        nn.ReLU,
        nn.Conv2d,
    ]

    convolutions = [module for module in operators if isinstance(module, nn.Conv2d)]
    assert [
        (
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.dilation,
            module.padding,
            module.stride,
            module.groups,
            module.bias is not None,
        )
        for module in convolutions
    ] == [
        (3, 32, (3, 3), (1, 1), (1, 1), (1, 1), 1, True),
        (32, 32, (3, 3), (2, 2), (2, 2), (1, 1), 1, True),
        (32, 32, (3, 3), (4, 4), (4, 4), (1, 1), 1, True),
        (32, 32, (3, 3), (1, 1), (1, 1), (1, 1), 1, True),
        (32, 5, (1, 1), (1, 1), (0, 0), (1, 1), 1, True),
    ]


@pytest.mark.parametrize("height,width", [(1, 1), (17, 23)])
def test_lite_fcn_preserves_spatial_shape(height: int, width: int) -> None:
    model = LiteFCN2D(in_channels=1, num_classes=2)
    logits = model(torch.randn(2, 1, height, width))

    assert logits.shape == (2, 2, height, width)


def test_lite_fcn_supports_categorical_cross_entropy_and_finite_backward() -> None:
    model = LiteFCN2D(in_channels=1, num_classes=2)
    image = torch.randn(2, 1, 9, 13)
    target = torch.randint(0, 2, (2, 9, 13), dtype=torch.long)

    loss = nn.CrossEntropyLoss()(model(image), target)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_lite_fcn_returns_unnormalized_raw_logits() -> None:
    model = LiteFCN2D(in_channels=1, num_classes=2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.layers[-1].bias.copy_(torch.tensor([2.0, -1.0]))

    logits = model(torch.zeros(1, 1, 2, 3))
    expected = torch.tensor([2.0, -1.0]).view(1, 2, 1, 1).expand_as(logits)

    assert torch.equal(logits, expected)
    assert logits.min().item() < 0.0 < 1.0 < logits.max().item()


def test_trainer_inherits_no_deep_supervision() -> None:
    no_deep_supervision, trainer = _load_trainer_types()

    assert issubclass(trainer, no_deep_supervision)


@pytest.mark.parametrize("enabled", [True, False])
def test_trainer_deep_supervision_toggle_is_noop_for_lite_fcn(enabled: bool) -> None:
    _, trainer_type = _load_trainer_types()

    trainer = object.__new__(trainer_type)
    trainer.network = LiteFCN2D(in_channels=1, num_classes=2)

    assert not hasattr(trainer.network, "decoder")
    trainer.set_deep_supervision_enabled(enabled)


def test_trainer_build_hook_uses_channels_for_2d_configuration() -> None:
    _, trainer = _load_trainer_types()

    model = trainer.build_network_architecture(
        plans_manager=object(),
        configuration_manager=SimpleNamespace(patch_size=(32, 24)),
        num_input_channels=3,
        num_output_channels=5,
        enable_deep_supervision=False,
    )

    assert isinstance(model, LiteFCN2D)
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
