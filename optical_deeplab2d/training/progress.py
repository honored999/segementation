"""Formatting helpers for terminal training progress."""

from __future__ import annotations

import math


def format_duration(seconds: float | None) -> str:
    """Format a non-negative duration as ``MM:SS`` or ``HH:MM:SS``."""
    if seconds is None:
        return "N/A"

    try:
        duration = float(seconds)
    except (TypeError, ValueError):
        return "N/A"

    if not math.isfinite(duration) or duration < 0:
        return "N/A"

    total_seconds = math.ceil(duration)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def build_batch_postfix(
    loss: float,
    average_loss: float,
    gpu_mib: float,
    epoch_eta_seconds: float | None,
) -> dict[str, str]:
    """Build the fixed progress-bar postfix fields for one training batch."""
    return {
        "loss": f"{loss:.4f}",
        "avg_loss": f"{average_loss:.4f}",
        "gpu_mib": f"{gpu_mib:.0f}",
        "epoch_eta": format_duration(epoch_eta_seconds),
    }


def format_epoch_summary(
    epoch: int,
    total_epochs: int,
    epoch_seconds: float | None,
    global_dice: float,
    patient_dice: float,
    total_eta_seconds: float | None,
) -> str:
    """Format a concise validation summary for a completed epoch."""
    return (
        f"Epoch {epoch}/{total_epochs} | "
        f"epoch_time={format_duration(epoch_seconds)} | "
        f"val_dice={global_dice:.4f} | "
        f"patient_dice={patient_dice:.4f} | "
        f"total_eta={format_duration(total_eta_seconds)}"
    )
