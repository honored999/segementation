"""One explicit, caller-invoked optimization step; no epoch training loop."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


@dataclass(frozen=True)
class TrainStepResult:
    loss: float
    output_shapes: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class TrainEpochResult:
    batch_count: int
    mean_loss: float
    output_shapes: tuple[tuple[int, ...], ...]


def train_step(
    model: nn.Module,
    batch: tuple[Tensor, Tensor],
    loss_fn: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    *,
    non_blocking: bool = False,
) -> TrainStepResult:
    """Execute exactly one supplied optimization step; callers control all policy."""
    model.train()
    image, target = (value.to(device, non_blocking=non_blocking) for value in batch)
    optimizer.zero_grad(set_to_none=True)
    outputs = model(image)
    loss = loss_fn(outputs, target)
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite training loss")
    loss.backward()
    optimizer.step()
    levels = (outputs,) if isinstance(outputs, Tensor) else tuple(outputs)
    return TrainStepResult(float(loss.detach().cpu()), tuple(tuple(level.shape) for level in levels))


def run_train_epoch(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
    loss_fn: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    *,
    non_blocking: bool = False,
) -> TrainEpochResult:
    """Run one caller-supplied sequence of optimization batches."""
    batch_count = 0
    total_loss = 0.0
    output_shapes: tuple[tuple[int, ...], ...] = ()

    for batch in batches:
        result = train_step(model, batch, loss_fn, optimizer, device, non_blocking=non_blocking)
        batch_count += 1
        total_loss += result.loss
        output_shapes = result.output_shapes

    if batch_count == 0:
        raise ValueError("empty training batches")

    return TrainEpochResult(
        batch_count=batch_count,
        mean_loss=total_loss / batch_count,
        output_shapes=output_shapes,
    )
