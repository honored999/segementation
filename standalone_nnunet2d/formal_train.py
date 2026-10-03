"""Explicit formal-alignment training entry point; never starts by default."""
from __future__ import annotations
import argparse
import csv
from copy import deepcopy
from dataclasses import asdict
import json
import random
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from standalone_nnunet2d.data.data_source import (
    DataSourceName,
    NNUNET_PREPROCESSED_B2ND,
    RAW_NIFTI_ONLINE,
    resolve_data_source_root,
    validate_data_source,
)
from standalone_nnunet2d.data.dataset import read_splits, load_fold_cases
from standalone_nnunet2d.performance import PerformanceConfig, build_formal_loaders, resolve_performance_config
from standalone_nnunet2d.engine.formal_validation import select_fold
from standalone_nnunet2d.training.formal_dataset import FormalPatchDataset
from standalone_nnunet2d.training.formal_trainer import run_formal_epoch, run_formal_validation
from standalone_nnunet2d.training.official_config import DEFAULT_RUN_STATE, OfficialTrainerSchedule, PolyLRScheduler, make_official_optimizer, resolve_optimizer_config
from standalone_nnunet2d.losses.compound import DiceCrossEntropyLoss
from standalone_nnunet2d.losses.deep_supervision import DeepSupervisionLoss
from standalone_nnunet2d.models.factory import (
 DEEP_SUPERVISION,
 MODEL_NAMES,
 PLAIN_CONV_UNET,
 SINGLE_OUTPUT,
 build_model,
 get_model_contract,
)
from standalone_nnunet2d.training.official_config import deep_supervision_weights
from standalone_nnunet2d.training.formal_checkpoint import FormalTrainerState, compute_plan_hash, load_formal_checkpoint, save_formal_checkpoint, read_formal_payload, validate_source_identity, initialize_pretrained_model, validate_pretrained_payload, validate_initialization_provenance
from standalone_nnunet2d.alignment_evidence import OFFICIAL_ALIGNED, resolve_alignment_state, validate_alignment_evidence_record


def build_formal_config(*, fold: int, epochs: int, schedule: OfficialTrainerSchedule, performance: PerformanceConfig | None = None, alignment_evidence: dict[str, object] | None = None, model_name: str = PLAIN_CONV_UNET, supervision_mode: str | None = None, batch_size: int = 12, data_source: DataSourceName = RAW_NIFTI_ONLINE, data_root: Path | None = None, optimizer_name: str = 'sgd') -> dict[str, object]:
 if batch_size<=0: raise ValueError(f'batch_size must be positive, got {batch_size}')
 if performance is None: performance=resolve_performance_config('alignment',device='cpu')
 source=validate_data_source(data_source)
 resolved_data_root=None if data_root is None else str(Path(data_root).expanduser().resolve())
 model_contract=get_model_contract(model_name, supervision_mode=supervision_mode)
 model_config=model_contract.as_dict()
 if alignment_evidence is None:
  run_state=DEFAULT_RUN_STATE
  validated_evidence=None
 else:
  validated_evidence=validate_alignment_evidence_record(alignment_evidence)
  run_state=OFFICIAL_ALIGNED
 schedule_config=asdict(schedule)
 optimizer_config=resolve_optimizer_config(model_name=model_contract.name,optimizer_name=optimizer_name)
 policies={'scheduler':{'name':'poly','exponent':.9,'initial_lr':optimizer_config['lr'],'max_steps':schedule.num_epochs},'training':{'iterations_per_epoch':schedule.num_iterations_per_epoch,'oversample_foreground_percent':schedule.oversample_foreground_percent},'validation':{'iterations_per_epoch':schedule.num_val_iterations_per_epoch}}
 selection_config={'interval_epochs':10,'mirror_axes':[],'aggregation':'case_macro_mean'}
 early_stopping_config={'max_epochs':schedule.num_epochs,'start_epoch':100,'patience':10,'min_delta':.001}
 performance_config={'profile':performance.profile,'loader':performance.as_dict(),'optimizations':{'amp':performance.amp,'tf32':performance.tf32,'compile':performance.compile}}
 data_source_config={'type':source,'root':resolved_data_root}
 plan={'run_type':run_state,'run_state':run_state,'alignment_evidence':validated_evidence,'schedule':schedule_config,'optimizer':optimizer_config,'policies':policies,'selection':selection_config,'early_stopping':early_stopping_config,'performance':performance_config,'model':model_config,'batch_size':batch_size,'data_source':{'type':source}}
 config={'run_type':run_state,'run_state':run_state,'fold':fold,'epochs':epochs,'batch_size':batch_size,'schedule':schedule_config,'optimizer':optimizer_config,'policies':policies,'selection':selection_config,'early_stopping':early_stopping_config,'performance_profile':performance.profile,'performance':performance_config,'model':deepcopy(model_config),'data_source':data_source_config,'plan_hash':compute_plan_hash(plan)}
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

def build_formal_datasets(data_root: Path, *, fold: int, patch_size: tuple[int, int], use_mask_for_norm: tuple[bool, ...], data_source: DataSourceName = RAW_NIFTI_ONLINE, splits_file: Path | None = None) -> tuple[FormalPatchDataset, FormalPatchDataset]:
 source_kwargs = {} if data_source == RAW_NIFTI_ONLINE else {'data_source': data_source}
 if splits_file is not None: source_kwargs['splits_file']=splits_file
 train=FormalPatchDataset(data_root,fold=fold,split='train',patch_size=patch_size,use_mask_for_norm=use_mask_for_norm,augment=True,**source_kwargs)
 validation=FormalPatchDataset(data_root,fold=fold,split='val',patch_size=patch_size,use_mask_for_norm=use_mask_for_norm,augment=False,oversample_foreground_percent=0.0,**source_kwargs)
 return train,validation

def build_parser() -> argparse.ArgumentParser:
 p=argparse.ArgumentParser(description='Explicit formal-alignment training entry point')
 source_group=p.add_mutually_exclusive_group(required=True); source_group.add_argument('--raw-root',type=Path); source_group.add_argument('--preprocessed-root',type=Path); p.add_argument('--output-root',required=True,type=Path); p.add_argument('--plans',required=True,type=Path); p.add_argument('--fold',type=int,default=0); p.add_argument('--device',default='cuda:0'); p.add_argument('--epochs',type=int,default=1000); p.add_argument('--batch-size',type=int,default=12); resume_group=p.add_mutually_exclusive_group(); resume_group.add_argument('--resume',type=Path); resume_group.add_argument('--pretrained-checkpoint',type=Path); p.add_argument('--confirm-run',action='store_true')
 p.add_argument('--splits-file',type=Path); p.add_argument('--dataset-id'); p.add_argument('--stage',choices=('pretrain','finetune')); p.add_argument('--source-version'); p.add_argument('--patient-map',type=Path)
 p.add_argument('--performance-profile',choices=('alignment','throughput'),default='alignment')
 p.add_argument('--model',choices=MODEL_NAMES,default=PLAIN_CONV_UNET); p.add_argument('--optimizer',choices=('sgd','adamw'),default='sgd')
 p.add_argument('--supervision-mode',choices=(DEEP_SUPERVISION,SINGLE_OUTPUT),default=None)
 p.add_argument('--num-workers',type=int)
 p.add_argument('--pin-memory',choices=('auto','on','off'),default='auto')
 p.add_argument('--persistent-workers',dest='persistent_workers',action='store_true')
 p.add_argument('--no-persistent-workers',dest='persistent_workers',action='store_false')
 p.set_defaults(persistent_workers=None)
 p.add_argument('--prefetch-factor',type=int)
 p.add_argument('--transform-parity-report',type=Path)
 p.add_argument('--inference-parity-report',type=Path)
 return p


def build_training_losses(model_name: str = PLAIN_CONV_UNET, *, supervision_mode: str | None = None) -> tuple[nn.Module, nn.Module]:
 model_contract=get_model_contract(model_name, supervision_mode=supervision_mode)
 validation_loss=DiceCrossEntropyLoss()
 if model_contract.supervision_mode==DEEP_SUPERVISION:
  return DeepSupervisionLoss(validation_loss,weights=deep_supervision_weights(7)), validation_loss
 return validation_loss, validation_loss


def run_formal_epochs(*, model: nn.Module, train_loader: Iterable[tuple[torch.Tensor,torch.Tensor]], val_loader: Iterable[tuple[torch.Tensor,torch.Tensor]], loss: nn.Module, validation_loss: nn.Module, optimizer: Optimizer, scheduler: PolyLRScheduler, device: torch.device, start_epoch: int, end_epoch: int, schedule: OfficialTrainerSchedule, non_blocking: bool = False) -> Iterator[tuple[int,object,object,float]]:
 for epoch in range(start_epoch,end_epoch):
  train_result,lr=run_formal_epoch(model,train_loader,loss,optimizer,scheduler,device,epoch,schedule,non_blocking=non_blocking); validation=run_formal_validation(model,val_loader,validation_loss,device,schedule)
  yield epoch,train_result,validation,lr


def update_selection_state(
    state: FormalTrainerState,
    selection_dice: float,
    *,
    completed_epoch: int,
    start_epoch: int = 100,
    patience: int = 10,
    min_delta: float = .001,
) -> tuple[FormalTrainerState, bool, bool]:
 strict_improvement=selection_dice>state.best_selection_dice
 best_dice=max(state.best_selection_dice,selection_dice)
 best_epoch=completed_epoch if strict_improvement else state.best_selection_epoch
 reference=state.early_stop_reference_dice
 checks=state.checks_without_improvement
 stop=False
 if completed_epoch < start_epoch:
  checks=0
 elif selection_dice>=reference+min_delta:
  reference=selection_dice
  checks=0
 else:
  checks+=1
  stop=checks>=patience
 return FormalTrainerState(state.epoch,state.global_step,state.best_validation_dice,state.fold,best_dice,best_epoch,reference,checks),strict_improvement,stop


def build_source_identity(*, stage: str | None, dataset_id: str | None, splits_file: Path | None,
                          plans: Path, fold: int, data_source: str, data_root: Path,
                          patch_size: tuple[int, int], model_name: str,
                          source_version: str | None = None, patient_map: Path | None = None) -> dict[str, object] | None:
 if stage is None:
  if any(v is not None for v in (dataset_id, splits_file, source_version, patient_map)):
   raise ValueError('source options require explicit stage')
  return None
 if stage not in ('pretrain','finetune'): raise ValueError('unsupported stage')
 if model_name != 'h2former' or patch_size != (512,512):
  raise ValueError('explicit stages require base h2former and plans patch_size 512x512')
 if stage == 'pretrain' and (not dataset_id or splits_file is None or dataset_id.lower().startswith('dataset501') or dataset_id == '501'):
  raise ValueError('pretrain requires external dataset-id and splits-file; Dataset501 forbidden')
 if stage == 'finetune':
  if dataset_id not in (None,'Dataset501','Dataset501_DWI','501'):
   raise ValueError('finetune requires Dataset501')
  dataset_id='Dataset501'
 mapping=None if patient_map is None else json.loads(patient_map.read_text(encoding='utf-8'))
 if mapping is not None and not isinstance(mapping,dict): raise ValueError('patient map must be case ID to patient ID object')
 folds=read_splits(splits_file,patient_map=mapping)
 canonical=lambda fs: [{k:sorted(f[k]) for k in ('train','val')} for f in fs]
 if stage == 'finetune' and canonical(folds) != canonical(read_splits()):
  raise ValueError('finetune cannot replace Dataset501 fixed splits')
 train=load_fold_cases(fold,'train',splits_file=splits_file)
 val=load_fold_cases(fold,'val',splits_file=splits_file)
 # Locate only source metadata, never load medical images during config checks.
 candidates=[data_root/'dataset.json',data_root.parent/'dataset.json']
 manifest=next((p for p in candidates if p.is_file()),None)
 if manifest is None: raise ValueError('source dataset.json is required for explicit stages')
 dataset=json.loads(manifest.read_text(encoding='utf-8'))
 if len(dataset.get('channel_names',{})) != 1 or dataset.get('labels') != {'background':0,'lesion':1}:
  raise ValueError('source must be single-channel background/lesion 0/1')
 manifest_mapping=dataset.get('patient_map')
 if manifest_mapping is not None:
  if mapping is not None and mapping != manifest_mapping: raise ValueError('patient maps conflict')
  mapping=manifest_mapping
  if not isinstance(mapping,dict): raise ValueError('manifest patient map must be an object')
  read_splits(splits_file,patient_map=mapping)
 plan=json.loads(plans.read_text(encoding='utf-8'))
 if data_source == RAW_NIFTI_ONLINE:
  spacing=plan['configurations']['2d'].get('spacing')
  if spacing is None or len(spacing)!=2 or not np.allclose(spacing,(.4892368018627167,.4892368018627167),rtol=0.,atol=1e-8):
   raise ValueError('raw online path uses fixed in-plane spacing; use approved source b2nd for different plans')
 if plan['configurations']['2d'].get('patch_size') != [512,512]:
  raise ValueError('source plans patch_size must be exactly 512x512')
 if len(plan['configurations']['2d']['use_mask_for_norm']) != 1:
  raise ValueError('source plans must be single channel')
 return {'stage':stage,'dataset_id':dataset_id,'source_version':source_version or dataset.get('source_version') or 'UNVERIFIED',
         'split_sha256':compute_plan_hash(canonical(folds)),
         'cases_sha256':compute_plan_hash({'train':sorted(train),'val':sorted(val)}),
         'plans_sha256':compute_plan_hash(plan),'dataset_metadata_sha256':compute_plan_hash(dataset),
         'patient_map_sha256':None if mapping is None else compute_plan_hash(mapping),
         'type':data_source,'root':str(data_root.expanduser().resolve()),
         'channels':1,'labels':{'background':0,'lesion':1},'patch_size':[512,512]}


def validate_stage_output(output_root: Path, *, data_root: Path, pretrained: Path | None = None,
                          resume: Path | None = None, protected_roots: Sequence[Path] = ()) -> Path:
 root=output_root.expanduser().resolve()
 protected=[data_root.resolve(),Path(__file__).resolve().parents[1]]
 protected.extend(Path(p).resolve() for p in protected_roots)
 protected.extend(p for p in Path(__file__).resolve().parents if (p/'.git').is_dir())
 # Parent-level nnU-Net raw/preprocessed roots protect sibling datasets too.
 for parent in data_root.resolve().parents:
  if parent.name.lower() in {'nnunet_raw','nnunet_preprocessed'}: protected.append(parent)
 if pretrained is not None: protected.append(pretrained.resolve().parent)
 for other in protected:
  if root == other or root in other.parents or other in root.parents:
   raise ValueError(f'output overlaps protected path: {other}')
 if resume is None:
  if root.exists(): raise ValueError('new stage output must not exist')
  # Existing experiment directories may not contain a new run.
  for parent in root.parents:
   if (parent/'resolved_config.json').exists() or any(parent.glob('checkpoint_*.pth')):
    raise ValueError('output is nested inside existing experiment')
 else:
  if resume.resolve().parent != root or not root.is_dir():
   raise ValueError('resume requires its verified stage output directory')
  if not (root/'resolved_config.json').is_file():
   raise ValueError('resume output lacks resolved configuration')
 return root


def main(arguments: Sequence[str] | None = None) -> int:
 p=build_parser(); a=p.parse_args(arguments)
 if a.batch_size<=0: p.error('batch_size must be positive')
 try:
  model_contract=get_model_contract(a.model, supervision_mode=a.supervision_mode)
 except ValueError as exc:
  p.error(str(exc))
 try:
  resolve_optimizer_config(model_name=model_contract.name,optimizer_name=a.optimizer)
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
 patch_size,use_mask_for_norm=load_2d_plan_config(a.plans)
 try:
  data_source, data_root = resolve_data_source_root(a.raw_root, a.preprocessed_root, data_source=RAW_NIFTI_ONLINE if a.raw_root is not None else NNUNET_PREPROCESSED_B2ND)
 except ValueError as exc:
  p.error(str(exc))
 schedule=OfficialTrainerSchedule()
 try:
  config=build_formal_config(fold=a.fold,epochs=a.epochs,schedule=schedule,performance=performance,alignment_evidence=alignment_evidence,model_name=model_contract.name,supervision_mode=model_contract.supervision_mode,batch_size=a.batch_size,data_source=data_source,data_root=data_root,optimizer_name=a.optimizer)
 except ValueError as exc:
  p.error(str(exc))
 try:
  identity=build_source_identity(stage=a.stage,dataset_id=a.dataset_id,splits_file=a.splits_file,plans=a.plans,fold=a.fold,data_source=data_source,data_root=data_root,patch_size=patch_size,model_name=a.model,source_version=a.source_version,patient_map=a.patient_map)
  if a.pretrained_checkpoint is not None and a.stage != 'finetune': raise ValueError('pretrained-checkpoint requires finetune')
  if identity is not None:
   config['source_identity']=identity
   config['initialization_provenance']=None
   validate_stage_output(a.output_root,data_root=data_root,pretrained=a.pretrained_checkpoint,resume=a.resume)
   if a.pretrained_checkpoint is not None:
    source_payload,_=read_formal_payload(a.pretrained_checkpoint)
    source_config=validate_pretrained_payload(source_payload)
    source_root=source_config.get('source_identity',{}).get('root')
    if source_root: validate_stage_output(a.output_root,data_root=data_root,pretrained=a.pretrained_checkpoint,protected_roots=(Path(source_root),))
    del source_payload
   if a.resume is not None:
    # This metadata check never restores model/optimizer/RNG or requires source weights.
    payload,_=read_formal_payload(a.resume)
    saved=payload['metadata'].get('resolved_config',{})
    validate_source_identity(saved.get('source_identity'),identity)
    disk_config=json.loads((a.output_root/'resolved_config.json').read_text(encoding='utf-8'))
    validate_source_identity(disk_config.get('source_identity'),identity)
    if disk_config.get('initialization_provenance') != saved.get('initialization_provenance'):
     raise ValueError('resume output initialization provenance mismatch')
    config['initialization_provenance']=deepcopy(saved.get('initialization_provenance'))
    provenance=config['initialization_provenance']
    validate_initialization_provenance(provenance)
    if provenance is not None:
     validate_stage_output(a.output_root,data_root=data_root,resume=a.resume,pretrained=Path(provenance['path']),protected_roots=(Path(provenance['source_identity']['root']),))
    del payload
 except (ValueError,FileNotFoundError) as exc:
  p.error(str(exc))
 if not a.confirm_run: print(json.dumps({'execution':'not-confirmed','config':config},indent=2,default=str)); return 0
 if not 1<=a.epochs<=schedule.num_epochs: p.error('epochs must be in [1,1000]')
 random.seed(0); np.random.seed(0); torch.manual_seed(0)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(0)
 device=torch.device(a.device)
 train,val=build_formal_datasets(data_root,fold=a.fold,patch_size=patch_size,use_mask_for_norm=use_mask_for_norm,data_source=data_source,**({'splits_file':a.splits_file} if a.splits_file is not None else {}))
 train_loader,val_loader=build_formal_loaders(train,val,performance=performance,batch_size=a.batch_size)
 model=build_model(a.model, supervision_mode=model_contract.supervision_mode).to(device); optimizer=make_official_optimizer(model,model_name=model_contract.name,optimizer_name=a.optimizer); scheduler=PolyLRScheduler(optimizer,schedule.num_epochs); loss,validation_loss=build_training_losses(a.model, supervision_mode=model_contract.supervision_mode)
 if a.pretrained_checkpoint is not None: config['initialization_provenance']=initialize_pretrained_model(model,a.pretrained_checkpoint)
 state=FormalTrainerState(0,0,-1.,a.fold)
 if a.resume is not None: state=load_formal_checkpoint(model,optimizer,scheduler,a.resume,fold=a.fold,plan_hash=str(config['plan_hash']),policies=config['policies'],run_state=str(config['run_state']),alignment_evidence=config.get('alignment_evidence'),model_name=model_contract.name,supervision_mode=model_contract.supervision_mode,checkpoint_root=a.output_root,**({'source_identity':identity,'initialization_provenance':config['initialization_provenance']} if identity is not None else {})).state
 a.output_root.mkdir(parents=True,exist_ok=identity is None or a.resume is not None)
 write_resolved_config(a.output_root/'resolved_config.json',config)
 log=(a.output_root/'training_log.csv').open('a',newline='',encoding='utf-8'); writer=csv.DictWriter(log,fieldnames=('epoch','global_step','train_loss','validation_dice','best_dice','selection_dice','lr'));
 if log.tell()==0: writer.writeheader()
 early_stopping=config['early_stopping']
 if (state.epoch>=int(early_stopping['start_epoch']) and state.checks_without_improvement>=int(early_stopping['patience'])):
  log.close()
  return 0
 for epoch,train_result,validation,lr in run_formal_epochs(model=model,train_loader=train_loader,val_loader=val_loader,loss=loss,validation_loss=validation_loss,optimizer=optimizer,scheduler=scheduler,device=device,start_epoch=state.epoch,end_epoch=a.epochs,schedule=schedule,non_blocking=performance.non_blocking):
  print({'run_type':config['run_type'],'run_state':config['run_state'],'epoch':epoch,'train_loss':train_result.mean_loss,'validation_dice':validation.dice,'lr':lr})
  state=FormalTrainerState(epoch+1,state.global_step+schedule.num_iterations_per_epoch,max(state.best_validation_dice,validation.dice),a.fold,state.best_selection_dice,state.best_selection_epoch,state.early_stop_reference_dice,state.checks_without_improvement)
  selection_dice=None; selection_saved=False; should_stop=False
  if state.epoch % int(config['selection']['interval_epochs'])==0:
   selection=select_fold(model,data_root,data_source=data_source,fold=a.fold,device=device,case_source=getattr(val,'_case_source',None),**({'splits_file':a.splits_file} if a.splits_file is not None else {}))
   state,selection_saved,should_stop=update_selection_state(state,selection['selection_dice'],completed_epoch=state.epoch,start_epoch=int(config['early_stopping']['start_epoch']),patience=int(config['early_stopping']['patience']),min_delta=float(config['early_stopping']['min_delta']))
   selection_dice=selection['selection_dice']
  save_formal_checkpoint(model,optimizer,scheduler,a.output_root/'checkpoint_latest.pth',state,config,plan_hash=str(config['plan_hash']),policies=config['policies'],run_state=str(config['run_state']),alignment_evidence=config.get('alignment_evidence'),model_name=model_contract.name,supervision_mode=model_contract.supervision_mode,checkpoint_root=a.output_root)
  if selection_saved: save_formal_checkpoint(model,optimizer,scheduler,a.output_root/'checkpoint_best.pth',state,config,plan_hash=str(config['plan_hash']),policies=config['policies'],run_state=str(config['run_state']),alignment_evidence=config.get('alignment_evidence'),model_name=model_contract.name,supervision_mode=model_contract.supervision_mode,checkpoint_root=a.output_root)
  writer.writerow({'epoch':epoch,'global_step':state.global_step,'train_loss':train_result.mean_loss,'validation_dice':validation.dice,'best_dice':state.best_selection_dice,'selection_dice':selection_dice,'lr':lr}); log.flush()
  if should_stop: break
 log.close()
 return 0
if __name__=='__main__': raise SystemExit(main())
