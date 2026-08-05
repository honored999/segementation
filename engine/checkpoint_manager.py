"""Latest/best checkpoint policy for bounded smoke-only workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import nn
from torch.optim import Optimizer

from standalone_nnunet2d.engine.checkpoint import save_checkpoint


def save_smoke_checkpoints(model: nn.Module, optimizer: Optimizer, root: Path, *, epoch: int, global_step: int, fold: int, validation_dice: float, best_dice: float, config: dict[str, Any]) -> float:
    """Always save latest; replace best only on a strict Dice improvement."""
    metadata = {"run_type": "smoke_run_only", "smoke_run_only": True, "epoch": epoch, "global_step": global_step, "fold": fold, "best_validation_dice": max(best_dice, validation_dice), "config": config}
    save_checkpoint(model, optimizer, root / "checkpoints" / "checkpoint_latest.pth", metadata)
    if validation_dice > best_dice:
        save_checkpoint(model, optimizer, root / "checkpoints" / "checkpoint_best.pth", metadata)
        return validation_dice
    return best_dice
