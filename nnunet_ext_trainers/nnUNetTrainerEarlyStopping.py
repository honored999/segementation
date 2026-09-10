"""Early stopping for the official nnU-Net 2.8.1 Trainer."""

from __future__ import annotations

from typing import Union

import torch
from torch._dynamo import OptimizedModule

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerEarlyStopping(nnUNetTrainer):
    """Official nnU-Net 2.8.1 training with EMA pseudo-Dice early stopping."""

    MAX_EPOCHS = 1000
    MIN_TRAINING_EPOCHS = 300
    PATIENCE = 100
    MIN_DELTA = 1e-4

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = self.MAX_EPOCHS
        self._early_stopping_best_ema: float | None = None
        self._epochs_without_improvement = 0
        self._early_stopping_triggered = False

    def _update_early_stopping(self) -> None:
        monitored_ema = float(self.logger.get_value("ema_fg_dice", step=-1))
        completed_epochs = self.current_epoch + 1

        if self._early_stopping_best_ema is None:
            self._early_stopping_best_ema = monitored_ema
            return

        if monitored_ema >= self._early_stopping_best_ema + self.MIN_DELTA:
            self._early_stopping_best_ema = monitored_ema
            self._epochs_without_improvement = 0
            return

        if completed_epochs <= self.MIN_TRAINING_EPOCHS:
            return

        self._epochs_without_improvement += 1
        if self._epochs_without_improvement >= self.PATIENCE:
            self._early_stopping_triggered = True
            self.print_to_log_file(
                "Early stopping at epoch "
                f"{completed_epochs}: best monitored EMA Dice="
                f"{self._early_stopping_best_ema:.6f}, patience={self.PATIENCE}, "
                f"min_delta={self.MIN_DELTA}"
            )

    def on_epoch_end(self) -> None:
        self._update_early_stopping()
        super().on_epoch_end()

    def _early_stopping_checkpoint_state(self) -> dict:
        return {
            "best_monitored_ema_dice": self._early_stopping_best_ema,
            "epochs_without_improvement": self._epochs_without_improvement,
            "triggered": self._early_stopping_triggered,
        }

    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                if self.is_ddp:
                    mod = self.network.module
                else:
                    mod = self.network
                if isinstance(mod, OptimizedModule):
                    mod = mod._orig_mod

                checkpoint = {
                    "network_weights": mod.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "grad_scaler_state": self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    "logging": self.logger.get_checkpoint(),
                    "_best_ema": self._best_ema,
                    "current_epoch": self.current_epoch + 1,
                    "init_args": self.my_init_kwargs,
                    "trainer_name": self.__class__.__name__,
                    "inference_allowed_mirroring_axes": self.inference_allowed_mirroring_axes,
                    "early_stopping_state": self._early_stopping_checkpoint_state(),
                }
                torch.save(checkpoint, filename)
            else:
                self.print_to_log_file("No checkpoint written, checkpointing is disabled")

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)
        else:
            checkpoint = filename_or_checkpoint
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = {}
        for k, value in checkpoint["network_weights"].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith("module."):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint["init_args"]
        self.current_epoch = checkpoint["current_epoch"]
        self.logger.load_checkpoint(checkpoint["logging"])
        self._best_ema = checkpoint["_best_ema"]
        self.inference_allowed_mirroring_axes = (
            checkpoint["inference_allowed_mirroring_axes"]
            if "inference_allowed_mirroring_axes" in checkpoint.keys()
            else self.inference_allowed_mirroring_axes
        )

        # messing with state dict naming schemes. Facepalm.
        if self.is_ddp:
            if isinstance(self.network.module, OptimizedModule):
                self.network.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.module.load_state_dict(new_state_dict)
        else:
            if isinstance(self.network, OptimizedModule):
                self.network._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.load_state_dict(new_state_dict)
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.grad_scaler is not None:
            if checkpoint["grad_scaler_state"] is not None:
                self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state"])

        state = checkpoint.get("early_stopping_state")
        if state is None:
            self._early_stopping_best_ema = self._best_ema
            self._epochs_without_improvement = 0
            self._early_stopping_triggered = False
        else:
            self._early_stopping_best_ema = state["best_monitored_ema_dice"]
            self._epochs_without_improvement = state["epochs_without_improvement"]
            self._early_stopping_triggered = state.get("triggered", False)

    def run_training(self):
        self.on_train_start()
        if self._early_stopping_triggered:
            self.on_train_end()
            return

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()
            if self._early_stopping_triggered:
                break

        self.on_train_end()
