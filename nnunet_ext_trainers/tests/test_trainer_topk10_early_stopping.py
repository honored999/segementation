"""Focused composition contracts; runtime execution is deferred to the server."""

from pathlib import Path
import sys


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
if str(EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTENSION_ROOT))

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnUNetTrainerEarlyStopping import nnUNetTrainerEarlyStopping
from nnUNetTrainerTopK10 import nnUNetTrainerTopK10
from nnUNetTrainerTopK10EarlyStopping import nnUNetTrainerTopK10EarlyStopping


Trainer = nnUNetTrainerTopK10EarlyStopping


def test_mro_preserves_exact_parent_method_bindings() -> None:
    assert Trainer.__mro__ == (
        Trainer, nnUNetTrainerEarlyStopping, nnUNetTrainerTopK10,
        nnUNetTrainer, object,
    )
    assert Trainer._build_loss is nnUNetTrainerTopK10._build_loss
    for name in (
        "__init__", "run_training", "on_epoch_end", "save_checkpoint",
        "load_checkpoint", "_update_early_stopping",
        "_early_stopping_checkpoint_state",
    ):
        assert getattr(Trainer, name) is getattr(nnUNetTrainerEarlyStopping, name)
    for name in (
        "build_network_architecture", "configure_optimizers", "train_step",
        "validation_step", "on_train_end",
    ):
        assert getattr(Trainer, name) is getattr(nnUNetTrainer, name)
    assert Trainer.TOPK_PERCENT == 10
    assert (Trainer.MAX_EPOCHS, Trainer.MIN_TRAINING_EPOCHS,
            Trainer.PATIENCE, Trainer.MIN_DELTA) == (1000, 300, 100, 1e-4)


def test_cooperative_init_calls_official_base_once_and_keeps_deep_supervision(
    monkeypatch,
) -> None:
    calls = []

    def base_init(self, plans, configuration, fold, dataset_json, device):
        calls.append((plans, configuration, fold, dataset_json, device))
        self.enable_deep_supervision = True

    monkeypatch.setattr(nnUNetTrainer, "__init__", base_init)
    plans, dataset_json, device = {}, {}, object()
    trainer = Trainer(plans, "2d", 0, dataset_json, device)
    assert calls == [(plans, "2d", 0, dataset_json, device)]
    assert trainer.enable_deep_supervision is True
    assert trainer.num_epochs == 1000
    assert trainer._early_stopping_checkpoint_state() == {
        "best_monitored_ema_dice": None,
        "epochs_without_improvement": 0,
        "triggered": False,
    }


def test_terminal_resume_runs_cleanup_without_another_epoch() -> None:
    trainer = object.__new__(Trainer)
    trainer._early_stopping_triggered = True
    events = []
    trainer.on_train_start = lambda: events.append("start")
    trainer.on_train_end = lambda: events.append("end")
    trainer.on_epoch_start = lambda: events.append("unexpected epoch")
    trainer.run_training()
    assert events == ["start", "end"]


def test_inherited_patience_boundary_and_checkpoint_state() -> None:
    trainer = object.__new__(Trainer)
    trainer.current_epoch = 299
    trainer._early_stopping_best_ema = 0.5
    trainer._epochs_without_improvement = Trainer.PATIENCE - 1
    trainer._early_stopping_triggered = False
    trainer.print_to_log_file = lambda *args, **kwargs: None

    class Logger:
        def get_value(self, key, step):
            assert (key, step) == ("ema_fg_dice", -1)
            return 0.5

    trainer.logger = Logger()
    trainer._update_early_stopping()
    assert trainer._epochs_without_improvement == 99
    assert trainer._early_stopping_triggered is False
    trainer.current_epoch = 300
    trainer._update_early_stopping()
    assert trainer._early_stopping_checkpoint_state() == {
        "best_monitored_ema_dice": 0.5,
        "epochs_without_improvement": 100,
        "triggered": True,
    }
