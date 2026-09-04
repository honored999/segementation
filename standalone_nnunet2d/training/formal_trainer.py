"""Formal Trainer epoch/validation mechanics, separate from smoke workflows."""
from __future__ import annotations
from collections.abc import Iterable, Sequence
from itertools import islice
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from standalone_nnunet2d.engine.trainer import TrainEpochResult, run_train_epoch
from standalone_nnunet2d.engine.validator import ValidationEpochResult, run_validation_epoch
from standalone_nnunet2d.training.official_config import OfficialTrainerSchedule, PolyLRScheduler

def limit_iterations(batches: Iterable[tuple[Tensor,Tensor]],limit:int) -> Iterable[tuple[Tensor,Tensor]]:
 if limit<=0: raise ValueError('iteration limit must be positive')
 if isinstance(batches,Sequence):
  if not batches: raise ValueError('batches must not be empty')
  return islice(iter(batches),min(limit,len(batches)))
 def repeated():
  while True:
   count=0
   for batch in batches: count+=1; yield batch
   if count==0: raise ValueError('batches must not be empty')
 return islice(repeated(),limit)
def run_formal_epoch(model: nn.Module,batches: Iterable[tuple[Tensor,Tensor]],loss: nn.Module,optimizer: Optimizer,scheduler: PolyLRScheduler,device: torch.device,epoch:int,schedule: OfficialTrainerSchedule,*,non_blocking: bool = False) -> tuple[TrainEpochResult,float]:
 scheduler.step(epoch); scheduler._formal_last_step=epoch; scheduler.ctr=epoch+1
 result=run_train_epoch(model,limit_iterations(batches,schedule.num_iterations_per_epoch),loss,optimizer,device,non_blocking=non_blocking)
 return result,scheduler.get_last_lr()[0]
def run_formal_validation(model: nn.Module,batches: Iterable[tuple[Tensor,Tensor]],loss: nn.Module,device: torch.device,schedule: OfficialTrainerSchedule)->ValidationEpochResult:
 return run_validation_epoch(model,limit_iterations(batches,schedule.num_val_iterations_per_epoch),loss,device)
