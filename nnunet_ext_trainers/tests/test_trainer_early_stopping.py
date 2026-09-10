"""Synthetic contract tests for the nnU-Net 2.8.1 early-stopping Trainer."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
if str(EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTENSION_ROOT))

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnUNetTrainerEarlyStopping import nnUNetTrainerEarlyStopping


class _Logger:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def get_value(self, key: str, step: int) -> float:
        assert key == "ema_fg_dice"
        assert step == -1
        return self.values[-1]

    def get_checkpoint(self) -> dict:
        return {"ema_fg_dice": self.values}

    def load_checkpoint(self, checkpoint: dict) -> None:
        self.values = checkpoint.get("ema_fg_dice", [])


class _Stateful:
    def __init__(self, state: dict) -> None:
        self.state = state

    def state_dict(self) -> dict:
        return self.state

    def load_state_dict(self, state: dict) -> None:
        self.state = state


def _state(epoch: int, values: list[float]) -> nnUNetTrainerEarlyStopping:
    trainer = object.__new__(nnUNetTrainerEarlyStopping)
    trainer.current_epoch = epoch
    trainer.logger = _Logger(values)
    trainer._early_stopping_best_ema = None
    trainer._epochs_without_improvement = 0
    trainer._early_stopping_triggered = False
    trainer.local_rank = 0
    trainer.print_to_log_file = lambda *args, **kwargs: None
    return trainer


def test_defaults_and_official_trainer_are_unchanged() -> None:
    assert issubclass(nnUNetTrainerEarlyStopping, nnUNetTrainer)
    assert nnUNetTrainerEarlyStopping.MAX_EPOCHS == 1000
    assert nnUNetTrainerEarlyStopping.MIN_TRAINING_EPOCHS == 300
    assert nnUNetTrainerEarlyStopping.PATIENCE == 100
    assert nnUNetTrainerEarlyStopping.MIN_DELTA == pytest.approx(1e-4)
    assert nnUNetTrainer.num_epochs if hasattr(nnUNetTrainer, "num_epochs") else True
    assert not hasattr(nnUNetTrainer, "_early_stopping_triggered")


def test_minimum_epochs_never_triggers_stopping() -> None:
    trainer = _state(299, [0.5])
    trainer._early_stopping_best_ema = 0.5
    trainer._epochs_without_improvement = trainer.PATIENCE - 1

    trainer._update_early_stopping()

    assert trainer._epochs_without_improvement == trainer.PATIENCE - 1
    assert trainer._early_stopping_triggered is False


def test_improvement_resets_patience() -> None:
    trainer = _state(300, [0.5002])
    trainer._early_stopping_best_ema = 0.5
    trainer._epochs_without_improvement = 17

    trainer._update_early_stopping()

    assert trainer._early_stopping_best_ema == pytest.approx(0.5002)
    assert trainer._epochs_without_improvement == 0


def test_patience_stops_after_complete_epoch() -> None:
    trainer = _state(399, [0.5])
    trainer._early_stopping_best_ema = 0.5
    trainer._epochs_without_improvement = trainer.PATIENCE - 1

    trainer._update_early_stopping()

    assert trainer._epochs_without_improvement == trainer.PATIENCE
    assert trainer._early_stopping_triggered is True


def test_min_delta_must_be_met() -> None:
    trainer = _state(300, [0.500099])
    trainer._early_stopping_best_ema = 0.5

    trainer._update_early_stopping()

    assert trainer._early_stopping_best_ema == pytest.approx(0.5)
    assert trainer._epochs_without_improvement == 1


def test_run_training_preserves_hook_order_and_finishes_after_stop() -> None:
    trainer = object.__new__(nnUNetTrainerEarlyStopping)
    trainer.current_epoch = 0
    trainer.num_epochs = 3
    trainer.num_iterations_per_epoch = 1
    trainer.num_val_iterations_per_epoch = 1
    trainer.dataloader_train = iter(["train"] * 3)
    trainer.dataloader_val = iter(["val"] * 3)
    trainer._early_stopping_triggered = False
    calls: list[str] = []
    trainer.on_train_start = lambda: calls.append("train_start")
    trainer.on_epoch_start = lambda: calls.append("epoch_start")
    trainer.on_train_epoch_start = lambda: calls.append("train_epoch_start")
    trainer.train_step = lambda batch: calls.append("train_step") or batch
    trainer.on_train_epoch_end = lambda outputs: calls.append("train_epoch_end")
    trainer.on_validation_epoch_start = lambda: calls.append("val_epoch_start")
    trainer.validation_step = lambda batch: calls.append("val_step") or batch
    trainer.on_validation_epoch_end = lambda outputs: calls.append("val_epoch_end")

    def on_epoch_end() -> None:
        calls.append("epoch_end")
        trainer.current_epoch += 1
        trainer._early_stopping_triggered = trainer.current_epoch == 1

    trainer.on_epoch_end = on_epoch_end
    trainer.on_train_end = lambda: calls.append("train_end")

    trainer.run_training()

    assert calls == [
        "train_start",
        "epoch_start",
        "train_epoch_start",
        "train_step",
        "train_epoch_end",
        "val_epoch_start",
        "val_step",
        "val_epoch_end",
        "epoch_end",
        "train_end",
    ]


def test_run_training_without_stop_matches_official_hook_order() -> None:
    def run(trainer_class):
        trainer = object.__new__(trainer_class)
        trainer.current_epoch = 0
        trainer.num_epochs = 2
        trainer.num_iterations_per_epoch = 1
        trainer.num_val_iterations_per_epoch = 1
        trainer.dataloader_train = iter([None] * 2)
        trainer.dataloader_val = iter([None] * 2)
        if trainer_class is nnUNetTrainerEarlyStopping:
            trainer._early_stopping_triggered = False
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
        return calls

    assert run(nnUNetTrainerEarlyStopping) == run(nnUNetTrainer)


def test_resumed_triggered_checkpoint_runs_no_additional_epoch_and_still_finishes() -> None:
    trainer = object.__new__(nnUNetTrainerEarlyStopping)
    trainer.current_epoch = 400
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


def _loadable_trainer() -> nnUNetTrainerEarlyStopping:
    trainer = object.__new__(nnUNetTrainerEarlyStopping)
    trainer.was_initialized = True
    trainer.network = _Stateful({})
    trainer.optimizer = _Stateful({})
    trainer.grad_scaler = None
    trainer.logger = _Logger([])
    trainer.is_ddp = False
    trainer.inference_allowed_mirroring_axes = None
    return trainer


def test_checkpoint_round_trip_preserves_early_stopping_state() -> None:
    checkpoint = {
        "network_weights": {},
        "optimizer_state": {},
        "grad_scaler_state": None,
        "logging": {},
        "_best_ema": 0.71,
        "current_epoch": 351,
        "init_args": {},
        "trainer_name": "nnUNetTrainerEarlyStopping",
        "inference_allowed_mirroring_axes": None,
    }
    source = _state(350, [0.7])
    source._early_stopping_best_ema = 0.69
    source._epochs_without_improvement = 42
    source._early_stopping_triggered = False
    saved = source._early_stopping_checkpoint_state()
    checkpoint["early_stopping_state"] = deepcopy(saved)
    restored = _loadable_trainer()

    restored.load_checkpoint(checkpoint)

    assert restored._early_stopping_best_ema == pytest.approx(0.69)
    assert restored._epochs_without_improvement == 42
    assert restored._early_stopping_triggered is False


def test_saved_checkpoint_keeps_official_fields_and_adds_early_stopping_state(monkeypatch) -> None:
    trainer = _state(350, [0.7])
    trainer._early_stopping_best_ema = 0.69
    trainer._epochs_without_improvement = 42
    trainer.disable_checkpointing = False
    trainer.is_ddp = False
    trainer.network = _Stateful({"weight": torch.tensor([1.0])})
    trainer.optimizer = _Stateful({"state": "optimizer"})
    trainer.grad_scaler = None
    trainer._best_ema = 0.7
    trainer.my_init_kwargs = {"configuration": "2d"}
    trainer.inference_allowed_mirroring_axes = (0, 1)
    captured: dict = {}
    monkeypatch.setattr(torch, "save", lambda payload, filename: captured.update(payload))

    trainer.save_checkpoint("ignored.pth")

    assert set(captured) == {
        "network_weights",
        "optimizer_state",
        "grad_scaler_state",
        "logging",
        "_best_ema",
        "current_epoch",
        "init_args",
        "trainer_name",
        "inference_allowed_mirroring_axes",
        "early_stopping_state",
    }
    assert captured["current_epoch"] == 351
    assert captured["_best_ema"] == pytest.approx(0.7)
    assert captured["early_stopping_state"] == {
        "best_monitored_ema_dice": pytest.approx(0.69),
        "epochs_without_improvement": 42,
        "triggered": False,
    }


def test_loading_official_checkpoint_uses_reasonable_initial_state() -> None:
    checkpoint = {
        "network_weights": {},
        "optimizer_state": {},
        "grad_scaler_state": None,
        "logging": {},
        "_best_ema": 0.73,
        "current_epoch": 351,
        "init_args": {},
        "trainer_name": "nnUNetTrainer",
        "inference_allowed_mirroring_axes": None,
    }
    trainer = _loadable_trainer()

    trainer.load_checkpoint(checkpoint)

    assert trainer._early_stopping_best_ema == pytest.approx(0.73)
    assert trainer._epochs_without_improvement == 0
    assert trainer._early_stopping_triggered is False
