"""Confirmed constants from inspected nnUNetTrainer source."""
from __future__ import annotations
from torch import nn
import torch
from dataclasses import dataclass

DEFAULT_RUN_STATE = "official_alignment_pending"


@dataclass(frozen=True)
class OfficialInferencePolicy:
    postprocessing: str = "argmax"
    mirror_axes: tuple[int, int] = (0, 1)
    tile_step_size: float = 0.5
    aggregation: str = "case_macro_mean"


@dataclass(frozen=True)
class OfficialTrainerSchedule:
 num_epochs: int=1000
 num_iterations_per_epoch: int=250
 num_val_iterations_per_epoch: int=50
 oversample_foreground_percent: float=.33
 patch_size: tuple[int,int]=(512,512)
 rotation_radians: tuple[float,float]=(-torch.pi,torch.pi)
 mirror_axes: tuple[int,int]=(0,1)

class PolyLRScheduler:
 def __init__(self, optimizer: torch.optim.Optimizer, initial_lr: float, max_steps: int, exponent: float=.9) -> None:
  self.optimizer,self.initial_lr,self.max_steps,self.exponent,self.ctr=optimizer,initial_lr,max_steps,exponent,0
 def step(self,current_step: int|None=None) -> None:
  step=self.ctr if current_step is None else current_step
  if current_step is None: self.ctr+=1
  lr=self.initial_lr*(1-step/self.max_steps)**self.exponent
  for group in self.optimizer.param_groups: group['lr']=lr
 def get_last_lr(self) -> list[float]: return [float(group['lr']) for group in self.optimizer.param_groups]

def make_official_optimizer(model: nn.Module) -> torch.optim.SGD:
 return torch.optim.SGD(model.parameters(),lr=.01,momentum=.99,nesterov=True,weight_decay=3e-5)

def deep_supervision_weights(outputs: int) -> tuple[float,...]:
 if outputs < 2: raise ValueError('official deep supervision requires at least two outputs')
 raw=[1/(2**i) for i in range(outputs)]; raw[-1]=0.; total=sum(raw)
 return tuple(weight/total for weight in raw)
