from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from standalone_nnunet2d import formal_train
from standalone_nnunet2d.engine import trainer
from standalone_nnunet2d.engine.validator import ValidationEpochResult
from standalone_nnunet2d.engine.trainer import TrainEpochResult, TrainStepResult
from standalone_nnunet2d.training.formal_trainer import run_formal_epoch
from standalone_nnunet2d.training.official_config import OfficialTrainerSchedule, PolyLRScheduler


class _TinyDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        del index
        return torch.zeros(1, 4, 4), torch.zeros(4, 4, dtype=torch.long)


def _required(module: object, name: str):
    value = getattr(module, name, None)
    assert value is not None, f"expected {module!r} to expose {name}"
    return value


def test_alignment_profile_preserves_conservative_loader_and_runtime_defaults() -> None:
    resolve = _required(formal_train, "resolve_performance_config")

    config = resolve("alignment", device="cuda:0")

    assert config.profile == "alignment"
    assert config.num_workers == 0
    assert config.pin_memory is False
    assert config.persistent_workers is False
    assert config.prefetch_factor is None
    assert config.non_blocking is False
    assert config.amp is False
    assert config.tf32 is False
    assert config.compile is False


def test_throughput_profile_resolves_safe_cuda_loader_defaults() -> None:
    resolve = _required(formal_train, "resolve_performance_config")

    config = resolve("throughput", device="cuda:0")

    assert (config.num_workers, config.pin_memory) == (2, True)
    assert config.persistent_workers is True
    assert config.prefetch_factor == 2
    assert config.non_blocking is True
    assert config.amp is False
    assert config.tf32 is False
    assert config.compile is False


def test_explicit_cpu_pin_memory_does_not_enable_non_blocking_transfer() -> None:
    resolve = _required(formal_train, "resolve_performance_config")

    config = resolve("alignment", device="cpu", pin_memory="on")

    assert config.pin_memory is True
    assert config.non_blocking is False


def test_loader_factory_applies_resolved_settings_to_train_and_validation() -> None:
    resolve = _required(formal_train, "resolve_performance_config")
    build_loaders = _required(formal_train, "build_formal_loaders")
    performance = resolve("throughput", device="cuda:0")

    train_loader, val_loader = build_loaders(
        _TinyDataset(),
        _TinyDataset(),
        performance=performance,
        batch_size=2,
    )

    for loader in (train_loader, val_loader):
        assert loader.num_workers == 2
        assert loader.pin_memory is True
        assert loader.persistent_workers is True
        assert loader.prefetch_factor == 2


def test_formal_epoch_helper_reuses_the_same_loaders_for_every_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    run_epochs = _required(formal_train, "run_formal_epochs")
    train_loader = object()
    val_loader = object()
    seen_train_loaders: list[object] = []
    seen_val_loaders: list[object] = []

    def fake_train_epoch(*args, **kwargs):
        seen_train_loaders.append(args[1])
        return TrainEpochResult(1, 0.0, ()) , 0.01

    def fake_validation(*args, **kwargs):
        seen_val_loaders.append(args[1])
        return ValidationEpochResult(1, 0.0, 0.0, 0.0)

    monkeypatch.setattr(formal_train, "run_formal_epoch", fake_train_epoch)
    monkeypatch.setattr(formal_train, "run_formal_validation", fake_validation)

    rows = list(
        run_epochs(
            model=object(),
            train_loader=train_loader,
            val_loader=val_loader,
            loss=object(),
            validation_loss=object(),
            optimizer=object(),
            scheduler=object(),
            device=torch.device("cpu"),
            start_epoch=0,
            end_epoch=3,
            schedule=object(),
            non_blocking=False,
        )
    )

    assert len(rows) == 3
    assert seen_train_loaders == [train_loader] * 3
    assert seen_val_loaders == [val_loader] * 3


def test_run_train_epoch_forwards_non_blocking_to_each_train_step(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[bool] = []

    def fake_train_step(*args, non_blocking: bool = False, **kwargs):
        seen.append(non_blocking)
        return TrainStepResult(0.0, ())

    monkeypatch.setattr(trainer, "train_step", fake_train_step)

    result = trainer.run_train_epoch(
        model=nn.Identity(),
        batches=[(torch.zeros(1), torch.zeros(1))] * 2,
        loss_fn=nn.Identity(),
        optimizer=object(),
        device=torch.device("cpu"),
        non_blocking=True,
    )

    assert result.batch_count == 2
    assert seen == [True, True]


def test_formal_epoch_forwards_non_blocking_to_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[bool] = []

    def fake_run_train_epoch(*args, non_blocking: bool = False, **kwargs):
        seen.append(non_blocking)
        return TrainEpochResult(1, 0.0, ())

    monkeypatch.setattr("standalone_nnunet2d.training.formal_trainer.run_train_epoch", fake_run_train_epoch)

    model = nn.Conv2d(1, 2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = PolyLRScheduler(optimizer, 0.01, 1000)
    schedule = OfficialTrainerSchedule(num_iterations_per_epoch=1)

    run_formal_epoch(
        model,
        [(torch.zeros(1), torch.zeros(1))],
        nn.Identity(),
        optimizer,
        scheduler,
        torch.device("cuda:0"),
        0,
        schedule,
        non_blocking=True,
    )

    assert seen == [True]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"persistent_workers": True}, "persistent_workers.*num_workers"),
        ({"prefetch_factor": 2}, "prefetch_factor.*num_workers"),
        ({"num_workers": -1}, "num_workers"),
        ({"prefetch_factor": 0, "num_workers": 1}, "prefetch_factor"),
        ({"pin_memory": "invalid"}, "pin_memory"),
    ],
)
def test_performance_options_reject_invalid_combinations(kwargs: dict[str, object], message: str) -> None:
    resolve = _required(formal_train, "resolve_performance_config")

    with pytest.raises(ValueError, match=message):
        resolve("alignment", device="cpu", **kwargs)


def test_parser_exposes_explicit_performance_profile_and_loader_options() -> None:
    build_parser = _required(formal_train, "build_parser")
    parser = build_parser()

    args = parser.parse_args(
        [
            "--raw-root",
            "raw",
            "--output-root",
            "out",
            "--performance-profile",
            "throughput",
            "--num-workers",
            "3",
            "--pin-memory",
            "on",
            "--persistent-workers",
            "--prefetch-factor",
            "4",
        ]
    )

    assert args.performance_profile == "throughput"
    assert args.num_workers == 3
    assert args.pin_memory == "on"
    assert args.persistent_workers is True
    assert args.prefetch_factor == 4


def test_resolved_config_records_profile_loader_settings_and_disabled_optimizations() -> None:
    resolve = _required(formal_train, "resolve_performance_config")
    performance = resolve("alignment", device="cuda:0")

    config = formal_train.build_formal_config(
        fold=0,
        epochs=2,
        schedule=OfficialTrainerSchedule(),
        performance=performance,
    )

    assert config["performance_profile"] == "alignment"
    assert config["performance"]["profile"] == "alignment"
    assert config["performance"]["loader"] == {
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": None,
        "non_blocking": False,
    }
    assert config["performance"]["optimizations"] == {
        "amp": False,
        "tf32": False,
        "compile": False,
    }


def test_public_transfer_signatures_expose_non_blocking_keyword() -> None:
    assert "non_blocking" in inspect.signature(trainer.train_step).parameters
    assert "non_blocking" in inspect.signature(trainer.run_train_epoch).parameters
