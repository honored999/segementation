"""Explicit formal-alignment training entry point; never starts by default."""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from standalone_nnunet2d.config import load_model_config
from standalone_nnunet2d.training.formal_dataset import FormalPatchDataset
from standalone_nnunet2d.training.formal_trainer import run_formal_epoch, run_formal_validation
from standalone_nnunet2d.training.official_config import DEFAULT_RUN_STATE, OfficialTrainerSchedule, PolyLRScheduler, make_official_optimizer
from standalone_nnunet2d.losses.compound import DiceCrossEntropyLoss
from standalone_nnunet2d.losses.deep_supervision import DeepSupervisionLoss
from standalone_nnunet2d.models import PlainConvUNet2D
from standalone_nnunet2d.training.official_config import deep_supervision_weights
from standalone_nnunet2d.training.formal_checkpoint import FormalTrainerState, compute_plan_hash, load_formal_checkpoint, save_formal_checkpoint


def build_formal_config(*, fold: int, epochs: int, schedule: OfficialTrainerSchedule) -> dict[str, object]:
 schedule_config=asdict(schedule)
 optimizer_config={'name':'SGD','lr':.01,'momentum':.99,'nesterov':True,'weight_decay':3e-5}
 policies={'scheduler':{'name':'poly','exponent':.9,'initial_lr':.01,'max_steps':schedule.num_epochs},'training':{'iterations_per_epoch':schedule.num_iterations_per_epoch,'oversample_foreground_percent':schedule.oversample_foreground_percent},'validation':{'iterations_per_epoch':schedule.num_val_iterations_per_epoch}}
 plan={'schedule':schedule_config,'optimizer':optimizer_config,'policies':policies}
 return {'run_type':DEFAULT_RUN_STATE,'run_state':DEFAULT_RUN_STATE,'fold':fold,'epochs':epochs,'schedule':schedule_config,'optimizer':optimizer_config,'policies':policies,'plan_hash':compute_plan_hash(plan)}


def write_resolved_config(path: Path, config: dict[str, object]) -> None:
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(config,indent=2,sort_keys=True,default=str),encoding='utf-8')

def main() -> int:
 p=argparse.ArgumentParser(); p.add_argument('--raw-root',required=True,type=Path); p.add_argument('--output-root',required=True,type=Path); p.add_argument('--fold',type=int,default=0); p.add_argument('--device',default='cuda:0'); p.add_argument('--epochs',type=int,default=1000); p.add_argument('--resume',type=Path); p.add_argument('--confirm-run',action='store_true'); a=p.parse_args()
 schedule=OfficialTrainerSchedule(); config=build_formal_config(fold=a.fold,epochs=a.epochs,schedule=schedule)
 if not a.confirm_run: print(json.dumps({'execution':'not-confirmed','config':config},indent=2,default=str)); return 0
 if not 1<=a.epochs<=schedule.num_epochs: p.error('epochs must be in [1,1000]')
 random.seed(0); np.random.seed(0); torch.manual_seed(0)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(0)
 device=torch.device(a.device); a.output_root.mkdir(parents=True,exist_ok=True); write_resolved_config(a.output_root/'resolved_config.json',config)
 train=FormalPatchDataset(a.raw_root,fold=a.fold,split='train',augment=True); val=FormalPatchDataset(a.raw_root,fold=a.fold,split='val',augment=False,oversample_foreground_percent=0.)
 model=PlainConvUNet2D(load_model_config(),deep_supervision=True).to(device); optimizer=make_official_optimizer(model); scheduler=PolyLRScheduler(optimizer,.01,schedule.num_epochs); validation_loss=DiceCrossEntropyLoss(); loss=DeepSupervisionLoss(validation_loss,weights=deep_supervision_weights(7))
 state=FormalTrainerState(0,0,-1.,a.fold)
 if a.resume is not None: state=load_formal_checkpoint(model,optimizer,scheduler,a.resume,fold=a.fold,plan_hash=str(config['plan_hash']),policies=config['policies']).state
 log=(a.output_root/'training_log.csv').open('a',newline='',encoding='utf-8'); writer=csv.DictWriter(log,fieldnames=('epoch','global_step','train_loss','validation_dice','best_dice','lr')); 
 if log.tell()==0: writer.writeheader()
 for epoch in range(state.epoch,a.epochs):
  train_result,lr=run_formal_epoch(model,DataLoader(train,batch_size=12,shuffle=True),loss,optimizer,scheduler,device,epoch,schedule); validation=run_formal_validation(model,DataLoader(val,batch_size=12,shuffle=False),validation_loss,device,schedule); print({'run_type':'official_alignment_pending','epoch':epoch,'train_loss':train_result.mean_loss,'validation_dice':validation.dice,'lr':lr})
  improved=validation.dice>state.best_validation_dice; state=FormalTrainerState(epoch+1,state.global_step+schedule.num_iterations_per_epoch,max(state.best_validation_dice,validation.dice),a.fold); save_formal_checkpoint(model,optimizer,scheduler,a.output_root/'checkpoint_latest.pth',state,config,plan_hash=str(config['plan_hash']),policies=config['policies'],run_state=DEFAULT_RUN_STATE)
  if improved: save_formal_checkpoint(model,optimizer,scheduler,a.output_root/'checkpoint_best.pth',state,config,plan_hash=str(config['plan_hash']),policies=config['policies'],run_state=DEFAULT_RUN_STATE)
  writer.writerow({'epoch':epoch,'global_step':state.global_step,'train_loss':train_result.mean_loss,'validation_dice':validation.dice,'best_dice':state.best_validation_dice,'lr':lr}); log.flush()
 log.close()
 return 0
if __name__=='__main__': raise SystemExit(main())
