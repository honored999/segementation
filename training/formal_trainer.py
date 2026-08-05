"""Pure-PyTorch formal Trainer epoch mechanics, separate from smoke workflows."""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Iterable
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from standalone_nnunet2d.engine.trainer import run_train_epoch, TrainEpochResult
from standalone_nnunet2d.training.official_config import OfficialTrainerSchedule, PolyLRScheduler

@dataclass(frozen=True)
class FormalEpochResult:
 epoch: int
 train: TrainEpochResult
 learning_rate: float

def run_formal_epoch(model: nn.Module,batches: Iterable[tuple[Tensor,Tensor]],loss: nn.Module,optimizer: Optimizer,scheduler: PolyLRScheduler,device: torch.device,epoch: int,schedule: OfficialTrainerSchedule) -> FormalEpochResult:
 scheduler.step(epoch)
 result=run_train_epoch(model,batches,loss,optimizer,device)
 return FormalEpochResult(epoch,result,scheduler.get_last_lr()[0])
