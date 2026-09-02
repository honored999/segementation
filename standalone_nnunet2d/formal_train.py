"""Explicit formal-alignment training entry point; never starts by default."""
from __future__ import annotations
import argparse
import csv
from copy import deepcopy
from dataclasses import asdict
import json
import random
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from standalone_nnunet2d.performance import PerformanceConfig, build_formal_loaders, resolve_performance_config
from standalone_nnunet2d.config import load_model_config
from standalone_nnunet2d.data.dataset import resolve_channel_specs, resolve_input_channels
from standalone_nnunet2d.data.input_mode import InputMode, input_spec
from standalone_nnunet2d.training.formal_dataset import FormalPatchDataset, resolve_input_mode
from standalone_nnunet2d.training.formal_trainer import run_formal_epoch, run_formal_validation
from standalone_nnunet2d.training.official_config import DEFAULT_RUN_STATE, OfficialTrainerSchedule, PolyLRScheduler, make_official_optimizer
from standalone_nnunet2d.losses.compound import DiceCrossEntropyLoss
from standalone_nnunet2d.losses.deep_supervision import DeepSupervisionLoss
from standalone_nnunet2d.models import PlainConvUNet2D
from standalone_nnunet2d.training.official_config import deep_supervision_weights
from standalone_nnunet2d.predict import _read_checkpoint
from standalone_nnunet2d.training.formal_checkpoint import checkpoint_input_channels, checkpoint_input_mode, FormalTrainerState, compute_plan_hash, load_formal_checkpoint, save_formal_checkpoint
from standalone_nnunet2d.alignment_evidence import OFFICIAL_ALIGNED, resolve_alignment_state, validate_alignment_evidence_record


def resolve_cli_input_mode(
 input_mode: InputMode | str | None,
 bilateral_asymmetry_channel: bool,
) -> InputMode | None:
 return resolve_input_mode(input_mode,bilateral_asymmetry_channel=bilateral_asymmetry_channel)


def build_formal_config(*, fold: int, epochs: int, schedule: OfficialTrainerSchedule, performance: PerformanceConfig | None = None, alignment_evidence: dict[str, object] | None = None, input_channels: int | None = None, input_mode: InputMode | str | None = None, bilateral_asymmetry_channel: bool = False, physical_input_channels: int | None = None) -> dict[str, object]:
 resolved_input_mode=resolve_input_mode(input_mode,bilateral_asymmetry_channel=bilateral_asymmetry_channel)
 mode_spec=input_spec(resolved_input_mode) if resolved_input_mode is not None else None
 if input_channels is None: input_channels=mode_spec.effective_input_channels if mode_spec is not None else 1
 if isinstance(input_channels, bool) or input_channels < 1:
  raise ValueError(f'input_channels must be a positive integer, got {input_channels}')
 if mode_spec is not None:
  if input_channels != mode_spec.effective_input_channels:
   raise ValueError(
    f'input_mode={resolved_input_mode.value} requires input_channels='
    f'{mode_spec.effective_input_channels}, got {input_channels}'
   )
  expected_physical_input_channels=mode_spec.physical_input_channels
  if physical_input_channels is None: physical_input_channels=expected_physical_input_channels
  if physical_input_channels != expected_physical_input_channels:
   raise ValueError(
    f'input_mode={resolved_input_mode.value} requires physical_input_channels='
    f'{expected_physical_input_channels}, got {physical_input_channels}'
   )
 if performance is None: performance=resolve_performance_config('alignment',device='cpu')
 if input_channels > 1:
  if alignment_evidence is not None:
   raise ValueError('multichannel runs are experimental and cannot carry official alignment evidence')
  run_state=DEFAULT_RUN_STATE
  validated_evidence=None
 elif alignment_evidence is None:
  run_state=DEFAULT_RUN_STATE
  validated_evidence=None
 else:
  validated_evidence=validate_alignment_evidence_record(alignment_evidence)
  run_state=OFFICIAL_ALIGNED
 schedule_config=asdict(schedule)
 optimizer_config={'name':'SGD','lr':.01,'momentum':.99,'nesterov':True,'weight_decay':3e-5}
 policies={'scheduler':{'name':'poly','exponent':.9,'initial_lr':.01,'max_steps':schedule.num_epochs},'training':{'iterations_per_epoch':schedule.num_iterations_per_epoch,'oversample_foreground_percent':schedule.oversample_foreground_percent},'validation':{'iterations_per_epoch':schedule.num_val_iterations_per_epoch}}
 performance_config={'profile':performance.profile,'loader':performance.as_dict(),'optimizations':{'amp':performance.amp,'tf32':performance.tf32,'compile':performance.compile}}
 plan={'run_type':run_state,'run_state':run_state,'alignment_evidence':validated_evidence,'input_channels':input_channels,'schedule':schedule_config,'optimizer':optimizer_config,'policies':policies,'performance':performance_config}
 config={'run_type':run_state,'run_state':run_state,'fold':fold,'epochs':epochs,'input_channels':input_channels,'schedule':schedule_config,'optimizer':optimizer_config,'policies':policies,'performance_profile':performance.profile,'performance':performance_config}
 if mode_spec is not None:
  provenance={'input_mode':resolved_input_mode.value,'physical_input_channels':physical_input_channels,'effective_model_input_channels':input_channels}
  plan.update(provenance); config.update(provenance)
  if resolved_input_mode is InputMode.DWI_BILATERAL:
   plan['bilateral_asymmetry_channel']=True; config['bilateral_asymmetry_channel']=True
 config['plan_hash']=compute_plan_hash(plan)
 if input_channels > 1: config['experimental_extension']='multichannel'
 if validated_evidence is not None:
  config['alignment_evidence']=deepcopy(validated_evidence)
 return config


def write_resolved_config(path: Path, config: dict[str, object]) -> None:
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(config,indent=2,sort_keys=True,default=str),encoding='utf-8')

def load_2d_plan_config(path: Path) -> tuple[tuple[int, int], tuple[bool, ...]]:
 with path.open(encoding='utf-8') as handle:
  configuration=json.load(handle)['configurations']['2d']
 return tuple(int(value) for value in configuration['patch_size']), tuple(bool(value) for value in configuration['use_mask_for_norm'])

def resolve_use_mask_for_channels(values: tuple[bool, ...] | list[bool], input_channels: int) -> tuple[bool, ...]:
 if input_channels < 1:
  raise ValueError(f'input_channels must be positive, got {input_channels}')
 resolved=tuple(bool(value) for value in values)
 if len(resolved)==1:
  return resolved * input_channels
 if len(resolved)!=input_channels:
  raise ValueError(f'use_mask_for_norm must have length 1 or {input_channels}, got {len(resolved)}')
 return resolved


def _validate_resume_input_contract(path: Path, config: Mapping[str, object]) -> None:
 _, metadata=_read_checkpoint(path)
 checkpoint_mode=checkpoint_input_mode(metadata)
 checkpoint_channels=checkpoint_input_channels(metadata)
 runtime_channels=config['input_channels']
 runtime_mode_value=config.get('input_mode')
 runtime_mode=(
  None
  if runtime_mode_value is None
  else runtime_mode_value if isinstance(runtime_mode_value,InputMode) else InputMode(runtime_mode_value)
 )
 if runtime_mode is not None and runtime_mode is not checkpoint_mode:
  raise ValueError(
   f'runtime input_mode={runtime_mode.value} conflicts with '
   f'checkpoint input_mode={checkpoint_mode.value}'
  )
 if checkpoint_channels != runtime_channels:
  raise ValueError(
   f'runtime input_channels={runtime_channels} conflicts with '
   f'checkpoint input_channels={checkpoint_channels}'
  )

def build_formal_datasets(raw_root: Path, *, fold: int, patch_size: tuple[int, int], use_mask_for_norm: tuple[bool, ...], input_mode: InputMode | str | None = None, bilateral_asymmetry_channel: bool = False) -> tuple[FormalPatchDataset, FormalPatchDataset]:
 resolved_input_mode=resolve_input_mode(input_mode,bilateral_asymmetry_channel=bilateral_asymmetry_channel)
 derived_channel_kwargs={'input_mode':resolved_input_mode} if resolved_input_mode is not None else {}
 train=FormalPatchDataset(raw_root,fold=fold,split='train',patch_size=patch_size,use_mask_for_norm=use_mask_for_norm,augment=True,**derived_channel_kwargs)
 validation=FormalPatchDataset(raw_root,fold=fold,split='val',patch_size=patch_size,use_mask_for_norm=use_mask_for_norm,augment=False,oversample_foreground_percent=0.0,**derived_channel_kwargs)
 return train,validation

def build_parser() -> argparse.ArgumentParser:
 p=argparse.ArgumentParser(description='Explicit formal-alignment training entry point')
 p.add_argument('--raw-root',required=True,type=Path); p.add_argument('--output-root',required=True,type=Path); p.add_argument('--plans',required=True,type=Path); p.add_argument('--fold',type=int,default=0); p.add_argument('--device',default='cuda:0'); p.add_argument('--epochs',type=int,default=1000); p.add_argument('--resume',type=Path); p.add_argument('--confirm-run',action='store_true')
 p.add_argument('--performance-profile',choices=('alignment','throughput'),default='alignment')
 p.add_argument('--num-workers',type=int)
 p.add_argument('--pin-memory',choices=('auto','on','off'),default='auto')
 p.add_argument('--persistent-workers',dest='persistent_workers',action='store_true')
 p.add_argument('--no-persistent-workers',dest='persistent_workers',action='store_false')
 p.set_defaults(persistent_workers=None)
 p.add_argument('--prefetch-factor',type=int)
 p.add_argument('--transform-parity-report',type=Path)
 p.add_argument('--inference-parity-report',type=Path)
 p.add_argument('--input-mode',type=InputMode,choices=tuple(InputMode))
 p.add_argument('--bilateral-asymmetry-channel',action='store_true')
 return p


def run_formal_epochs(*, model: nn.Module, train_loader: Iterable[tuple[torch.Tensor,torch.Tensor]], val_loader: Iterable[tuple[torch.Tensor,torch.Tensor]], loss: nn.Module, validation_loss: nn.Module, optimizer: Optimizer, scheduler: PolyLRScheduler, device: torch.device, start_epoch: int, end_epoch: int, schedule: OfficialTrainerSchedule, non_blocking: bool = False) -> Iterator[tuple[int,object,object,float]]:
 for epoch in range(start_epoch,end_epoch):
  train_result,lr=run_formal_epoch(model,train_loader,loss,optimizer,scheduler,device,epoch,schedule,non_blocking=non_blocking); validation=run_formal_validation(model,val_loader,validation_loss,device,schedule)
  yield epoch,train_result,validation,lr


def main(arguments: Sequence[str] | None = None) -> int:
 p=build_parser(); a=p.parse_args(arguments)
 try:
  input_mode=resolve_cli_input_mode(a.input_mode,a.bilateral_asymmetry_channel)
 except ValueError as exc:
  p.error(str(exc))
 try:
  performance=resolve_performance_config(a.performance_profile,device=a.device,num_workers=a.num_workers,pin_memory=a.pin_memory,persistent_workers=a.persistent_workers,prefetch_factor=a.prefetch_factor)
 except ValueError as exc:
  p.error(str(exc))
 try:
  _, alignment_evidence=resolve_alignment_state(a.transform_parity_report,a.inference_parity_report)
 except ValueError as exc:
  p.error(str(exc))
 patch_size,plan_use_mask_for_norm=load_2d_plan_config(a.plans)
 schedule=OfficialTrainerSchedule()
 try:
  if input_mode is None:
   input_channels=resolve_input_channels(a.raw_root)
   physical_input_channels=None
  else:
   mode_spec=input_spec(input_mode)
   dataset_json=a.raw_root.resolve()/'dataset.json'
   if dataset_json.is_file():
    declared=resolve_channel_specs(a.raw_root)
    expected=tuple(enumerate(mode_spec.physical_modalities))
    if declared != expected:
     raise ValueError(
      f'input mode {input_mode.value} requires exact channel declaration {expected}; '
      f'found {declared}'
     )
   elif mode_spec.physical_input_channels != 1:
    raise ValueError(
     f'input mode {input_mode.value} requires dataset.json with '
     f'{mode_spec.physical_input_channels} physical channels'
    )
   input_channels=mode_spec.effective_input_channels
   physical_input_channels=mode_spec.physical_input_channels
 except ValueError as exc:
  p.error(str(exc))
 config=build_formal_config(fold=a.fold,epochs=a.epochs,schedule=schedule,performance=performance,alignment_evidence=alignment_evidence,input_channels=input_channels,input_mode=input_mode,physical_input_channels=physical_input_channels)
 if not a.confirm_run: print(json.dumps({'execution':'not-confirmed','config':config},indent=2,default=str)); return 0
 if not 1<=a.epochs<=schedule.num_epochs: p.error('epochs must be in [1,1000]')
 use_mask_for_norm=resolve_use_mask_for_channels(plan_use_mask_for_norm,input_channels)
 config=build_formal_config(fold=a.fold,epochs=a.epochs,schedule=schedule,performance=performance,alignment_evidence=alignment_evidence,input_channels=input_channels,input_mode=input_mode,physical_input_channels=physical_input_channels)
 if a.resume is not None: _validate_resume_input_contract(a.resume,config)
 random.seed(0); np.random.seed(0); torch.manual_seed(0)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(0)
 device=torch.device(a.device); a.output_root.mkdir(parents=True,exist_ok=True); write_resolved_config(a.output_root/'resolved_config.json',config)
 train,val=build_formal_datasets(a.raw_root,fold=a.fold,patch_size=patch_size,use_mask_for_norm=use_mask_for_norm,input_mode=input_mode)
 train_loader,val_loader=build_formal_loaders(train,val,performance=performance,batch_size=12)
 model=PlainConvUNet2D(load_model_config(input_channels=input_channels),deep_supervision=True).to(device); optimizer=make_official_optimizer(model); scheduler=PolyLRScheduler(optimizer,.01,schedule.num_epochs); validation_loss=DiceCrossEntropyLoss(); loss=DeepSupervisionLoss(validation_loss,weights=deep_supervision_weights(7))
 state=FormalTrainerState(0,0,-1.,a.fold)
 if a.resume is not None: state=load_formal_checkpoint(model,optimizer,scheduler,a.resume,fold=a.fold,plan_hash=str(config['plan_hash']),policies=config['policies'],run_state=str(config['run_state']),alignment_evidence=config.get('alignment_evidence')).state
 log=(a.output_root/'training_log.csv').open('a',newline='',encoding='utf-8'); writer=csv.DictWriter(log,fieldnames=('epoch','global_step','train_loss','validation_dice','best_dice','lr')); 
 if log.tell()==0: writer.writeheader()
 for epoch,train_result,validation,lr in run_formal_epochs(model=model,train_loader=train_loader,val_loader=val_loader,loss=loss,validation_loss=validation_loss,optimizer=optimizer,scheduler=scheduler,device=device,start_epoch=state.epoch,end_epoch=a.epochs,schedule=schedule,non_blocking=performance.non_blocking):
  print({'run_type':config['run_type'],'run_state':config['run_state'],'epoch':epoch,'train_loss':train_result.mean_loss,'validation_dice':validation.dice,'lr':lr})
  improved=validation.dice>state.best_validation_dice; state=FormalTrainerState(epoch+1,state.global_step+schedule.num_iterations_per_epoch,max(state.best_validation_dice,validation.dice),a.fold); save_formal_checkpoint(model,optimizer,scheduler,a.output_root/'checkpoint_latest.pth',state,config,plan_hash=str(config['plan_hash']),policies=config['policies'],run_state=str(config['run_state']),alignment_evidence=config.get('alignment_evidence'))
  if improved: save_formal_checkpoint(model,optimizer,scheduler,a.output_root/'checkpoint_best.pth',state,config,plan_hash=str(config['plan_hash']),policies=config['policies'],run_state=str(config['run_state']),alignment_evidence=config.get('alignment_evidence'))
  writer.writerow({'epoch':epoch,'global_step':state.global_step,'train_loss':train_result.mean_loss,'validation_dice':validation.dice,'best_dice':state.best_validation_dice,'lr':lr}); log.flush()
 log.close()
 return 0
if __name__=='__main__': raise SystemExit(main())
