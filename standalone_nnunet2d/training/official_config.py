"""Confirmed constants from inspected nnUNetTrainer source."""
from __future__ import annotations
import math
from torch import nn
import torch
from dataclasses import dataclass

DEFAULT_RUN_STATE = "official_alignment_pending"
DEFAULT_OPTIMIZER = "sgd"
ADAMW_OPTIMIZER = "adamw"
H2FORMER_MODEL = "h2former"
H2FORMER_LITE_UPERNET_MODEL = "h2former_lite_upernet"
H2FORMER_LITE_UPERNET_W128_MODEL = "h2former_lite_upernet_w128"
H2FORMER_MODELS = (H2FORMER_MODEL, H2FORMER_LITE_UPERNET_MODEL, H2FORMER_LITE_UPERNET_W128_MODEL)


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
 def __init__(self, optimizer: torch.optim.Optimizer, initial_lr: float|int|None = None, max_steps: int|float|None = None, exponent: float=.9) -> None:
  legacy_initial_lr = None if max_steps is None else initial_lr
  if max_steps is None:
   if initial_lr is None:
    raise ValueError('max_steps must be provided')
   max_steps = initial_lr
  try:
   normalized_max_steps = int(max_steps)
  except (TypeError, ValueError, OverflowError):
   raise ValueError(f'max_steps must be a positive integer, got {max_steps}') from None
  if isinstance(max_steps,bool) or normalized_max_steps!=max_steps or normalized_max_steps<=0:
   raise ValueError(f'max_steps must be a positive integer, got {max_steps}')
  self.optimizer=optimizer
  self.initial_lrs=tuple(float(group['lr']) for group in optimizer.param_groups)
  if not self.initial_lrs:
   raise ValueError('optimizer must contain at least one parameter group')
  if legacy_initial_lr is not None:
   try:
    legacy_initial_lr=float(legacy_initial_lr)
   except (TypeError, ValueError, OverflowError):
    raise ValueError(f'legacy initial_lr must be numeric, got {legacy_initial_lr!r}') from None
   if any(not math.isclose(legacy_initial_lr, group_lr, rel_tol=1e-9, abs_tol=1e-12) for group_lr in self.initial_lrs):
    raise ValueError(
     'legacy initial_lr is inconsistent with optimizer param-group LR: '
     f'initial_lr={legacy_initial_lr!r}, optimizer_lrs={self.initial_lrs!r}'
    )
  self.initial_lr=self.initial_lrs[0]
  self.max_steps=normalized_max_steps
  self.exponent=exponent
  self.ctr=0
 def step(self,current_step: int|None=None) -> None:
  step=self.ctr if current_step is None else current_step
  if current_step is None: self.ctr+=1
  factor=max(0.0,1-step/self.max_steps)**self.exponent
  for group,initial_lr in zip(self.optimizer.param_groups,self.initial_lrs): group['lr']=initial_lr*factor
 def get_last_lr(self) -> list[float]: return [float(group['lr']) for group in self.optimizer.param_groups]

def resolve_optimizer_config(*, model_name: str='plain_conv_unet', optimizer_name: str=DEFAULT_OPTIMIZER) -> dict[str, object]:
 name=str(optimizer_name).lower()
 if name==DEFAULT_OPTIMIZER:
  return {'name':'SGD','lr':.01,'momentum':.99,'nesterov':True,'weight_decay':3e-5}
 if name==ADAMW_OPTIMIZER:
  if model_name not in H2FORMER_MODELS: raise ValueError('AdamW is only supported for H2Former')
  return {'name':'AdamW','lr':1e-4,'betas':(.9,.999),'eps':1e-8,'weight_decay':3e-5}
 raise ValueError(f'unsupported optimizer_name {optimizer_name!r}; choices are {DEFAULT_OPTIMIZER!r} and {ADAMW_OPTIMIZER!r}')

def make_official_optimizer(model: nn.Module, *, model_name: str='plain_conv_unet', optimizer_name: str=DEFAULT_OPTIMIZER) -> torch.optim.Optimizer:
 config=resolve_optimizer_config(model_name=model_name,optimizer_name=optimizer_name)
 if config['name']=='SGD':
  return torch.optim.SGD(model.parameters(),lr=config['lr'],momentum=config['momentum'],nesterov=config['nesterov'],weight_decay=config['weight_decay'])
 return torch.optim.AdamW(model.parameters(),lr=config['lr'],betas=config['betas'],eps=config['eps'],weight_decay=config['weight_decay'])

def deep_supervision_weights(outputs: int) -> tuple[float,...]:
 if outputs < 2: raise ValueError('official deep supervision requires at least two outputs')
 raw=[1/(2**i) for i in range(outputs)]; raw[-1]=0.; total=sum(raw)
 return tuple(weight/total for weight in raw)
