from __future__ import annotations

from copy import deepcopy
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
if str(EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTENSION_ROOT))

from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss, DC_and_topk_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import (
    nnUNetTrainerNoDeepSupervision,
)
from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name

from nnUNetTrainerNoDeepSupervisionEarlyStopping import (
    nnUNetTrainerNoDeepSupervisionEarlyStopping,
)
from nnUNetTrainerNoDeepSupervisionTopK10EarlyStopping import (
    nnUNetTrainerNoDeepSupervisionTopK10EarlyStopping,
)
from nnUNetTrainerUPerNetEarlyStopping import nnUNetTrainerUPerNetEarlyStopping
from nnUNetTrainerUPerNetTopK10EarlyStopping import nnUNetTrainerUPerNetTopK10EarlyStopping
from nnUNetTrainerMixins import select_upernet_feature_indices
import nnUNetTrainerMixins as mixins


TRAINERS = (
    nnUNetTrainerNoDeepSupervisionEarlyStopping,
    nnUNetTrainerUPerNetEarlyStopping,
    nnUNetTrainerNoDeepSupervisionTopK10EarlyStopping,
    nnUNetTrainerUPerNetTopK10EarlyStopping,
)
UPERNET_TRAINERS = (nnUNetTrainerUPerNetEarlyStopping, nnUNetTrainerUPerNetTopK10EarlyStopping)
TOPK_TRAINERS = (
    nnUNetTrainerNoDeepSupervisionTopK10EarlyStopping,
    nnUNetTrainerUPerNetTopK10EarlyStopping,
)
DEFAULT_DECODER_TRAINERS = (
    nnUNetTrainerNoDeepSupervisionEarlyStopping,
    nnUNetTrainerNoDeepSupervisionTopK10EarlyStopping,
)


def _configuration(
    *, patch_size: tuple[int, ...] = (256, 256), n_stages: int = 8
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


def _build_network(
    trainer_class: type, *, output_channels: int = 3, configuration: SimpleNamespace | None = None,
    enable_deep_supervision: bool = False,
) -> nn.Module:
    configuration = configuration or _configuration()
    return trainer_class.build_network_architecture(
        plans_manager=object(),
        configuration_manager=configuration,
        num_input_channels=1,
        num_output_channels=output_channels,
        enable_deep_supervision=enable_deep_supervision,
    )


def test_four_trainers_are_unique_external_nnunet_identities(monkeypatch: pytest.MonkeyPatch) -> None:
    assert len({trainer.__name__ for trainer in TRAINERS}) == 4
    for trainer in TRAINERS:
        assert issubclass(trainer, nnUNetTrainer)
        assert trainer.__mro__.count(nnUNetTrainer) == 1
        assert trainer.__mro__.count(nnUNetTrainerNoDeepSupervision) == 1

    monkeypatch.setenv("nnUNet_extTrainer", str(EXTENSION_ROOT))
    for trainer in TRAINERS:
        resolved = recursive_find_trainer_class_by_name(trainer.__name__)
        assert resolved.__name__ == trainer.__name__
        assert issubclass(resolved, nnUNetTrainer)


def test_upernet_selection_is_plan_derived_and_matches_current_eight_stage_reference() -> None:
    assert select_upernet_feature_indices(
        [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
        8,
    ) == (1, 3, 5, 7)
    selected = select_upernet_feature_indices([[1, 1], [2, 2], [2, 2], [2, 2], [2, 2]], 5)
    assert len(selected) == 4
    assert len(set(selected)) == 4
    assert selected == tuple(sorted(selected))

    with pytest.raises(ValueError, match="four"):
        select_upernet_feature_indices([[1, 1], [2, 2], [2, 2]], 3)


def test_upernet_reuses_the_official_plans_encoder_without_default_decoder() -> None:
    configuration = _configuration()
    default_network = _build_network(nnUNetTrainerNoDeepSupervision, configuration=configuration)
    upernet = _build_network(nnUNetTrainerUPerNetEarlyStopping, configuration=configuration)

    assert isinstance(default_network, PlainConvUNet)
    assert isinstance(upernet.encoder, PlainConvEncoder)
    assert tuple(default_network.encoder.output_channels) == tuple(upernet.encoder.output_channels)
    assert default_network.encoder.strides == upernet.encoder.strides
    assert default_network.encoder.kernel_sizes == upernet.encoder.kernel_sizes
    assert default_network.encoder.norm_op is upernet.encoder.norm_op
    assert default_network.encoder.norm_op_kwargs == upernet.encoder.norm_op_kwargs
    assert upernet.selected_feature_indices == (1, 3, 5, 7)
    assert not hasattr(upernet, "transposed_convolutions")
    assert not any("transposed_convolutions" in name for name in upernet.state_dict())

    default_encoder_shapes = {
        name: tuple(parameter.shape) for name, parameter in default_network.encoder.state_dict().items()
    }
    upernet_encoder_shapes = {
        name: tuple(parameter.shape) for name, parameter in upernet.encoder.state_dict().items()
    }
    assert default_encoder_shapes == upernet_encoder_shapes


@pytest.mark.parametrize("trainer_class", DEFAULT_DECODER_TRAINERS)
def test_default_decoder_trainers_return_single_logits_and_support_cpu_backward(trainer_class: type) -> None:
    model = _build_network(trainer_class, output_channels=3).train()
    image = torch.randn(1, 1, 256, 256)
    target = torch.randint(0, 3, (1, 256, 256), dtype=torch.long)

    logits = model(image)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (1, 3, 256, 256)
    loss = nn.CrossEntropyLoss()(logits, target)
    loss.backward()
    assert torch.isfinite(loss)


@pytest.mark.parametrize("trainer_class", UPERNET_TRAINERS)
def test_upernet_is_single_output_raw_logits_and_supports_cpu_backward(trainer_class: type) -> None:
    model = _build_network(trainer_class, output_channels=3).train()
    image = torch.randn(1, 1, 256, 256)
    target = torch.randint(0, 3, (1, 256, 256), dtype=torch.long)

    logits = model(image)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (1, 3, 256, 256)
    assert torch.isfinite(logits).all()
    assert not any(isinstance(module, (nn.Softmax, nn.Sigmoid, nn.LogSoftmax)) for module in model.modules())
    assert model.decoder.fpn_channels == 128
    assert model.decoder.pool_scales == (1, 2, 4)
    assert all(not any(isinstance(module, nn.InstanceNorm2d) for module in branch.modules())
               for branch in model.decoder.ppm_branches)

    loss = nn.CrossEntropyLoss()(logits, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
               for parameter in model.parameters())


def test_upernet_lifecycle_setter_never_switches_to_deep_supervision() -> None:
    model = _build_network(nnUNetTrainerUPerNetEarlyStopping)
    trainer = object.__new__(nnUNetTrainerUPerNetEarlyStopping)
    trainer.network = model
    trainer.is_ddp = False

    trainer.set_deep_supervision_enabled(True)
    trainer.set_deep_supervision_enabled(False)
    with torch.inference_mode():
        output = model(torch.zeros(1, 1, 256, 256))
    assert isinstance(output, torch.Tensor)


def test_upernet_compute_feature_map_size_and_state_dict_identity_are_stable() -> None:
    train_model = _build_network(nnUNetTrainerUPerNetEarlyStopping)
    inference_model = _build_network(nnUNetTrainerUPerNetEarlyStopping)

    feature_map_size = train_model.compute_conv_feature_map_size((256, 256))
    assert int(feature_map_size) > 0
    assert {
        key: tuple(value.shape) for key, value in train_model.state_dict().items()
    } == {
        key: tuple(value.shape) for key, value in inference_model.state_dict().items()
    }


@pytest.mark.parametrize("input_size", [(128, 192), (129, 193)], ids=["divisible", "odd"])
def test_upernet_decoder_feature_map_count_matches_conv_hooks(input_size: tuple[int, int]) -> None:
    configuration = _configuration(patch_size=input_size, n_stages=6)
    configuration.network_arch_init_kwargs["strides"] = [
        1, [2, 2], [2, 1], [2, 2], [2, 2], [2, 2],
    ]
    model = _build_network(nnUNetTrainerUPerNetEarlyStopping, configuration=configuration).eval()
    observed = 0

    def count_conv(_module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        nonlocal observed
        observed += output.numel()

    handles = [
        module.register_forward_hook(count_conv)
        for module in model.decoder.modules()
        if isinstance(module, nn.Conv2d)
    ]
    try:
        with torch.inference_mode():
            model(torch.zeros(1, 1, *input_size))
    finally:
        for handle in handles:
            handle.remove()

    estimated = model.decoder.compute_conv_feature_map_size(input_size)
    assert observed > 0
    assert int(estimated) > 0
    assert int(estimated) == observed


def test_upernet_preserves_official_initialized_encoder_and_initializes_decoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_get = mixins.get_network_from_plans
    original_initialize = PlainConvUNet.initialize
    initialized_encoder_state: dict[str, torch.Tensor] = {}
    official_encoder: PlainConvEncoder | None = None
    decoder_conv_calls: list[nn.Module] = []
    decoder_modules: set[nn.Module] = set()

    def record_official(*args, **kwargs):
        nonlocal official_encoder
        assert kwargs["allow_init"] is True
        network = original_get(*args, **kwargs)
        official_encoder = network.encoder
        initialized_encoder_state.update(
            (name, value.detach().clone()) for name, value in network.encoder.state_dict().items()
        )
        return network

    def record_initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d) and module in decoder_modules:
            decoder_conv_calls.append(module)
        original_initialize(module)

    monkeypatch.setattr(mixins, "get_network_from_plans", record_official)
    monkeypatch.setattr(PlainConvUNet, "initialize", staticmethod(record_initialize))
    original_decoder_init = mixins.UPerNetDecoder.__init__

    def record_decoder_init(self, *args, **kwargs):
        original_decoder_init(self, *args, **kwargs)
        decoder_modules.update(self.modules())

    monkeypatch.setattr(mixins.UPerNetDecoder, "__init__", record_decoder_init)
    model = _build_network(nnUNetTrainerUPerNetEarlyStopping)
    assert model.encoder is official_encoder
    assert initialized_encoder_state
    assert all(
        torch.equal(value, initialized_encoder_state[name])
        for name, value in model.encoder.state_dict().items()
    )
    assert set(decoder_conv_calls) == {
        module for module in model.decoder.modules() if isinstance(module, nn.Conv2d)
    }


def test_upernet_rejects_deep_supervision_and_non_2d_plans() -> None:
    with pytest.raises(ValueError, match="deep supervision"):
        _build_network(nnUNetTrainerUPerNetEarlyStopping, enable_deep_supervision=True)

    with pytest.raises(ValueError, match="2D"):
        _build_network(
            nnUNetTrainerUPerNetEarlyStopping,
            configuration=_configuration(patch_size=(64, 64, 64)),
        )


@pytest.mark.parametrize("trainer_class", TRAINERS)
def test_new_trainer_losses_are_single_output_and_preserve_official_weights(trainer_class: type) -> None:
    trainer = object.__new__(trainer_class)
    trainer.label_manager = SimpleNamespace(has_regions=False, ignore_label=None)
    trainer.configuration_manager = SimpleNamespace(batch_dice=True)
    trainer.is_ddp = False
    trainer.enable_deep_supervision = False
    trainer._do_i_compile = lambda: False

    loss = trainer._build_loss()
    assert not isinstance(loss, DeepSupervisionWrapper)
    assert loss.weight_ce == 1
    assert loss.weight_dice == 1

    if trainer_class in TOPK_TRAINERS:
        assert isinstance(loss, DC_and_topk_loss)
        assert loss.ce.k == 10
    else:
        assert isinstance(loss, DC_and_CE_loss)


@pytest.mark.parametrize("trainer_class", TOPK_TRAINERS)
def test_new_topk_trainers_fall_back_to_official_region_loss(trainer_class: type) -> None:
    trainer = object.__new__(trainer_class)
    trainer.label_manager = SimpleNamespace(has_regions=True, ignore_label=None)
    trainer.configuration_manager = SimpleNamespace(batch_dice=True)
    trainer.is_ddp = False
    trainer.enable_deep_supervision = False
    trainer._do_i_compile = lambda: False

    assert isinstance(trainer._build_loss(), DC_and_BCE_loss)


def test_new_early_stopping_trainers_keep_the_same_contract_constants() -> None:
    for trainer in TRAINERS:
        assert trainer.MAX_EPOCHS == 1000
        assert trainer.MIN_TRAINING_EPOCHS == 300
        assert trainer.PATIENCE == 100
        assert trainer.MIN_DELTA == pytest.approx(1e-4)
        assert tuple(inspect.signature(trainer.__init__).parameters) == (
            "self",
            "plans",
            "configuration",
            "fold",
            "dataset_json",
            "device",
        )


class _Stateful:
    def __init__(self, state: dict) -> None:
        self.state = state

    def state_dict(self) -> dict:
        return self.state

    def load_state_dict(self, state: dict) -> None:
        self.state = state


class _Logger:
    def __init__(self) -> None:
        self.loaded: dict | None = None

    def load_checkpoint(self, state: dict) -> None:
        self.loaded = state


@pytest.mark.parametrize("trainer_class", TRAINERS)
def test_each_new_trainer_round_trips_early_stopping_state(trainer_class: type) -> None:
    trainer = object.__new__(trainer_class)
    trainer.was_initialized = True
    trainer.network = _Stateful({})
    trainer.optimizer = _Stateful({})
    trainer.grad_scaler = None
    trainer.logger = _Logger()
    trainer.is_ddp = False
    trainer.device = torch.device("cpu")
    trainer.inference_allowed_mirroring_axes = None

    checkpoint = {
        "network_weights": {},
        "optimizer_state": {},
        "grad_scaler_state": None,
        "logging": {"ema_fg_dice": [0.71]},
        "_best_ema": 0.72,
        "current_epoch": 407,
        "init_args": {"configuration": "2d"},
        "trainer_name": trainer_class.__name__,
        "inference_allowed_mirroring_axes": None,
        "early_stopping_state": {
            "best_monitored_ema_dice": 0.70,
            "epochs_without_improvement": 19,
            "triggered": False,
        },
    }

    trainer.load_checkpoint(deepcopy(checkpoint))

    assert trainer.current_epoch == 407
    assert trainer._best_ema == pytest.approx(0.72)
    assert trainer._early_stopping_best_ema == pytest.approx(0.70)
    assert trainer._epochs_without_improvement == 19
    assert trainer._early_stopping_triggered is False
    assert trainer.logger.loaded == checkpoint["logging"]


def test_terminal_early_stop_resume_on_upernet_identity_skips_epoch_loop() -> None:
    trainer = object.__new__(nnUNetTrainerUPerNetTopK10EarlyStopping)
    trainer.current_epoch = 407
    trainer.num_epochs = 1000
    trainer.num_iterations_per_epoch = 1
    trainer.num_val_iterations_per_epoch = 1
    trainer.dataloader_train = iter([None])
    trainer.dataloader_val = iter([None])
    trainer._early_stopping_triggered = True
    calls: list[str] = []
    trainer.on_train_start = lambda: calls.append("train_start")
    trainer.on_epoch_start = lambda: calls.append("epoch_start")
    trainer.on_train_epoch_start = lambda: calls.append("train_epoch_start")
    trainer.train_step = lambda batch: calls.append("train_step")
    trainer.on_train_epoch_end = lambda outputs: calls.append("train_epoch_end")
    trainer.on_validation_epoch_start = lambda: calls.append("val_epoch_start")
    trainer.validation_step = lambda batch: calls.append("val_step")
    trainer.on_validation_epoch_end = lambda outputs: calls.append("val_epoch_end")
    trainer.on_epoch_end = lambda: calls.append("epoch_end")
    trainer.on_train_end = lambda: calls.append("train_end")

    trainer.run_training()

    assert calls == ["train_start", "train_end"]
