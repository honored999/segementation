from __future__ import annotations

from copy import deepcopy
import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
if str(EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTENSION_ROOT))

from dynamic_network_architectures.architectures.unet import PlainConvUNet
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import (
    nnUNetTrainerNoDeepSupervision,
)
from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name

from nnUNetTrainerUPerNetTopK10EarlyStopping import nnUNetTrainerUPerNetTopK10EarlyStopping


TRAINER_MODULE = "nnUNetTrainerUPerNetNoStage7TopK10EarlyStopping"
TRAINER_NAME = "nnUNetTrainerUPerNetNoStage7TopK10EarlyStopping"


def _trainer_class() -> type | None:
    if importlib.util.find_spec(TRAINER_MODULE) is None:
        return None
    return getattr(importlib.import_module(TRAINER_MODULE), TRAINER_NAME)


def _configuration(
    *, patch_size: tuple[int, int] = (512, 512), n_stages: int = 8
) -> SimpleNamespace:
    features = [4, 8, 8, 16, 16, 32, 32, 32][:n_stages]
    strides = [[1, 1]] + [[2, 2] for _ in range(n_stages - 1)]
    return SimpleNamespace(
        patch_size=patch_size,
        batch_dice=True,
        network_arch_class_name="dynamic_network_architectures.architectures.unet.PlainConvUNet",
        network_arch_init_kwargs={
            "n_stages": n_stages,
            "features_per_stage": features,
            "conv_op": "torch.nn.modules.conv.Conv2d",
            "kernel_sizes": [[3, 3] for _ in range(n_stages)],
            "strides": strides,
            "n_conv_per_stage": [1 for _ in range(n_stages)],
            "n_conv_per_stage_decoder": [1 for _ in range(n_stages - 1)],
            "conv_bias": False,
            "norm_op": "torch.nn.modules.instancenorm.InstanceNorm2d",
            "norm_op_kwargs": {"eps": 1e-5, "affine": True},
            "dropout_op": None,
            "dropout_op_kwargs": None,
            "nonlin": "torch.nn.ReLU",
            "nonlin_kwargs": {"inplace": True},
            "nonlin_first": False,
        },
        network_arch_init_kwargs_req_import=["conv_op", "norm_op", "dropout_op", "nonlin"],
    )


def _build_network(trainer_class: type, configuration: SimpleNamespace) -> torch.nn.Module:
    return trainer_class.build_network_architecture(
        plans_manager=object(),
        configuration_manager=configuration,
        num_input_channels=1,
        num_output_channels=3,
        enable_deep_supervision=False,
    )


def test_variant_is_discoverable_and_inherits_the_baseline_training_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer_class = _trainer_class()
    assert trainer_class is not None, "the stage7-removal external Trainer module is missing"

    assert trainer_class.__base__ is nnUNetTrainerUPerNetTopK10EarlyStopping
    assert issubclass(trainer_class, nnUNetTrainer)
    assert nnUNetTrainerNoDeepSupervision in trainer_class.__mro__
    assert trainer_class.__mro__.count(nnUNetTrainer) == 1
    assert trainer_class._build_loss is nnUNetTrainerUPerNetTopK10EarlyStopping._build_loss
    assert trainer_class.run_training is nnUNetTrainerUPerNetTopK10EarlyStopping.run_training
    assert trainer_class.on_epoch_end is nnUNetTrainerUPerNetTopK10EarlyStopping.on_epoch_end
    assert trainer_class.MAX_EPOCHS == nnUNetTrainerUPerNetTopK10EarlyStopping.MAX_EPOCHS
    assert trainer_class.MIN_TRAINING_EPOCHS == nnUNetTrainerUPerNetTopK10EarlyStopping.MIN_TRAINING_EPOCHS
    assert trainer_class.PATIENCE == nnUNetTrainerUPerNetTopK10EarlyStopping.PATIENCE
    assert trainer_class.MIN_DELTA == nnUNetTrainerUPerNetTopK10EarlyStopping.MIN_DELTA

    monkeypatch.setenv("nnUNet_extTrainer", str(EXTENSION_ROOT))
    resolved = recursive_find_trainer_class_by_name(TRAINER_NAME)
    assert resolved is trainer_class


@pytest.mark.parametrize(
    ("incompatible_plan", "message"),
    [
        ("architecture", "PlainConvUNet"),
        ("stage_count", "eight stages"),
        ("patch_size", "512 x 512"),
        ("stride_geometry", "stride geometry"),
        ("stage_metadata", "features_per_stage"),
    ],
)
def test_variant_rejects_incompatible_plans(incompatible_plan: str, message: str) -> None:
    trainer_class = _trainer_class()
    assert trainer_class is not None, "the stage7-removal external Trainer module is missing"
    configuration = _configuration()

    if incompatible_plan == "architecture":
        configuration.network_arch_class_name = "other.architecture"
    elif incompatible_plan == "stage_count":
        configuration.network_arch_init_kwargs["n_stages"] = 7
    elif incompatible_plan == "patch_size":
        configuration.patch_size = (256, 256)
    elif incompatible_plan == "stride_geometry":
        configuration.network_arch_init_kwargs["strides"][6] = [1, 1]
    elif incompatible_plan == "stage_metadata":
        configuration.network_arch_init_kwargs["features_per_stage"].pop()

    with pytest.raises(ValueError, match=message):
        _build_network(trainer_class, configuration)


def test_variant_rejects_deep_supervision() -> None:
    trainer_class = _trainer_class()
    assert trainer_class is not None, "the stage7-removal external Trainer module is missing"

    with pytest.raises(ValueError, match="deep supervision"):
        trainer_class.build_network_architecture(
            plans_manager=object(),
            configuration_manager=_configuration(),
            num_input_channels=1,
            num_output_channels=3,
            enable_deep_supervision=True,
        )


def test_builder_removes_stage7_and_forward_uses_only_selected_native_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer_class = _trainer_class()
    assert trainer_class is not None, "the stage7-removal external Trainer module is missing"
    trainer_module = importlib.import_module(TRAINER_MODULE)
    original_builder = trainer_module.get_network_from_plans
    captured: dict[str, object] = {}

    def capture_official_network(*args, **kwargs):
        official = original_builder(*args, **kwargs)
        captured["official_type"] = type(official)
        captured["retained_state"] = {
            name: value.clone() for name, value in official.encoder.state_dict().items()
            if not name.startswith("stages.7.")
        }
        stage7 = official.encoder.stages[7]
        captured["stage7"] = stage7
        captured["stage7_parameter_ids"] = {id(parameter) for parameter in stage7.parameters()}
        captured["stage7_calls"] = []
        stage7.register_forward_hook(lambda *_: captured["stage7_calls"].append(True))
        return official

    monkeypatch.setattr(trainer_module, "get_network_from_plans", capture_official_network)
    configuration = _configuration()
    original_kwargs = deepcopy(configuration.network_arch_init_kwargs)
    original_imports = deepcopy(configuration.network_arch_init_kwargs_req_import)
    model = _build_network(trainer_class, configuration)

    assert isinstance(model.encoder, torch.nn.Module)
    assert captured["official_type"] is PlainConvUNet
    assert model.selected_feature_indices == (1, 3, 5, 6)
    assert len(model.encoder.stages) == 7
    assert len(model.encoder.output_channels) == 7
    assert len(model.encoder.strides) == 7
    assert len(model.encoder.kernel_sizes) == 7
    assert model.encoder.output_channels == original_kwargs["features_per_stage"][:7]
    assert model.encoder.strides == original_kwargs["strides"][:7]
    assert model.encoder.kernel_sizes == original_kwargs["kernel_sizes"][:7]
    assert model.decoder.fpn_channels == 128
    assert model.decoder.pool_scales == (1, 2, 4)
    assert len(model.decoder.encoder_spatial_geometry) == 7
    assert model.encoder.state_dict().keys() == captured["retained_state"].keys()
    assert all(torch.equal(value, captured["retained_state"][name])
               for name, value in model.encoder.state_dict().items())
    assert not any(name.startswith("encoder.stages.7.") for name, _ in model.named_parameters())
    assert captured["stage7"] not in tuple(model.modules())
    assert captured["stage7_parameter_ids"].isdisjoint({id(parameter) for parameter in model.parameters()})

    assert configuration.network_arch_init_kwargs == original_kwargs
    assert configuration.network_arch_init_kwargs_req_import == original_imports
    assert configuration.patch_size == (512, 512)

    observed_shapes: dict[int, tuple[int, int]] = {}
    for index in model.selected_feature_indices:
        model.encoder.stages[index].register_forward_hook(
            lambda _module, _inputs, output, index=index: observed_shapes.__setitem__(
                index, tuple(output.shape[-2:])
            )
        )

    conv_outputs: list[int] = []
    handles = [module.register_forward_hook(lambda _module, _inputs, output: conv_outputs.append(output.numel()))
               for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(1, 1, 128, 128))
    for handle in handles:
        handle.remove()

    assert captured["stage7_calls"] == []
    assert observed_shapes == {1: (64, 64), 3: (16, 16), 5: (4, 4), 6: (2, 2)}
    assert logits.shape == (1, 3, 128, 128)
    assert torch.isfinite(logits).all()
    assert int(model.compute_conv_feature_map_size((128, 128))) == sum(conv_outputs)

    native_shapes: dict[int, tuple[int, int]] = {}
    size = [512, 512]
    for index, stride in enumerate(model.encoder.strides):
        size = [dimension // step for dimension, step in zip(size, stride)]
        native_shapes[index] = tuple(size)
    assert tuple(native_shapes[index] for index in model.selected_feature_indices) == (
        (256, 256),
        (64, 64),
        (16, 16),
        (8, 8),
    )

    restored = _build_network(trainer_class, _configuration())
    restored.load_state_dict(model.state_dict(), strict=True)


def test_fresh_official_external_resolver_from_unrelated_cwd(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["nnUNet_extTrainer"] = str(EXTENSION_ROOT)
    environment["PYTHONIOENCODING"] = "utf-8"
    code = (
        "from importlib.metadata import version; "
        "from pathlib import Path; import inspect; "
        "from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name; "
        "assert version('nnunetv2') == '2.8.1'; "
        f"cls = recursive_find_trainer_class_by_name('{TRAINER_NAME}'); "
        f"assert Path(inspect.getfile(cls.build_network_architecture)).resolve() == Path({str(EXTENSION_ROOT / (TRAINER_MODULE + '.py'))!r}).resolve(); "
        "assert tuple(inspect.signature(cls.build_network_architecture).parameters) == "
        "('plans_manager', 'configuration_manager', 'num_input_channels', 'num_output_channels', 'enable_deep_supervision'); "
        "print('NO_STAGE7_OFFICIAL_RESOLVER_OK nnunetv2=2.8.1')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=environment,
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "NO_STAGE7_OFFICIAL_RESOLVER_OK nnunetv2=2.8.1" in completed.stdout


def test_actual_plans_and_configuration_managers_remain_immutable() -> None:
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    trainer_class = _trainer_class()
    assert trainer_class is not None
    fixture = _configuration()
    plans = {
        "dataset_name": "Dataset501_StrokeLesion", "plans_name": "nnUNetPlans",
        "configurations": {"2d": {
            "patch_size": list(fixture.patch_size),
            "architecture": {
                "network_class_name": fixture.network_arch_class_name,
                "arch_kwargs": fixture.network_arch_init_kwargs,
                "_kw_requires_import": fixture.network_arch_init_kwargs_req_import,
            },
        }},
    }
    before = deepcopy(plans)
    manager = PlansManager(plans)
    configuration = manager.get_configuration("2d")
    configuration_before = deepcopy(configuration.configuration)
    model = trainer_class.build_network_architecture(manager, configuration, 1, 3, False)
    assert len(model.encoder.stages) == 7
    assert plans == before
    assert manager.plans == before
    assert configuration.configuration == configuration_before
