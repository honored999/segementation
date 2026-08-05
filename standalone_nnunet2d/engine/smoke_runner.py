"""One explicit synthetic-or-server smoke epoch; no scheduling policy."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from standalone_nnunet2d.engine.checkpoint import save_checkpoint
from standalone_nnunet2d.engine.trainer import run_train_epoch
from standalone_nnunet2d.engine.validator import run_validation_epoch


def smoke_case_ids(fold: int) -> tuple[str, str]:
    """Return the first supplied train and validation case for one smoke run."""
    from standalone_nnunet2d.data.dataset import load_fold_cases

    return load_fold_cases(fold, "train")[0], load_fold_cases(fold, "val")[0]


def run_smoke_epoch(
    model: nn.Module,
    train_batches: Iterable[tuple[Tensor, Tensor]],
    validation_batches: Iterable[tuple[Tensor, Tensor]],
    loss_fn: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Run exactly one supplied train/validation sequence and save local state."""
    train = run_train_epoch(model, train_batches, loss_fn, optimizer, device)
    validation = run_validation_epoch(model, validation_batches, loss_fn, device)
    save_checkpoint(model, optimizer, checkpoint_path, {"smoke_run_only": True})
    return {
        "train": {"batch_count": train.batch_count, "mean_loss": train.mean_loss},
        "validation": {"batch_count": validation.batch_count, "mean_loss": validation.mean_loss, "dice": validation.dice, "iou": validation.iou},
        "checkpoint": str(checkpoint_path),
        "smoke_run_only": True,
    }
