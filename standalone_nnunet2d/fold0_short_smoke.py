"""Bounded fold-0 smoke-only workflow; never a formal benchmark."""
from __future__ import annotations
import argparse
import csv
import json
from itertools import islice
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from standalone_nnunet2d.config import load_model_config
from standalone_nnunet2d.data.dataset import StrokeSliceDataset, load_fold_cases
from standalone_nnunet2d.data.nifti_io import read_nifti
from standalone_nnunet2d.engine.case_validation import validate_cases
from standalone_nnunet2d.engine.checkpoint import load_checkpoint
from standalone_nnunet2d.engine.checkpoint_manager import save_smoke_checkpoints
from standalone_nnunet2d.engine.trainer import run_train_epoch
from standalone_nnunet2d.losses.compound import DiceCrossEntropyLoss
from standalone_nnunet2d.models import PlainConvUNet2D

def main() -> int:
 p=argparse.ArgumentParser(); p.add_argument('--raw-root',required=True,type=Path); p.add_argument('--output-root',required=True,type=Path); p.add_argument('--device',default='cuda:0'); p.add_argument('--confirm-run',action='store_true'); a=p.parse_args()
 if not a.confirm_run: p.error('--confirm-run is required for smoke-only execution')
 device=torch.device(a.device); fold=0; train_ids=load_fold_cases(fold,'train'); val_ids=load_fold_cases(fold,'val')[:3]
 train=StrokeSliceDataset(a.raw_root,fold=fold,split='train',case_ids=train_ids); loader=DataLoader(train,batch_size=1,shuffle=False)
 model=PlainConvUNet2D(load_model_config()).to(device); optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9); loss=DiceCrossEntropyLoss(); best=-1.; step=0
 a.output_root.mkdir(parents=True,exist_ok=True); config={'run_type':'smoke_run_only','smoke_run_only':True,'fold':0,'epochs':2,'max_train_batches':8,'validation_cases':3,'optimizer':{'name':'SGD','lr':.01,'momentum':.9,'weight_decay':0.0},'seed':None}; (a.output_root/'resolved_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
 log=(a.output_root/'training_log.csv').open('w',newline='',encoding='utf-8'); writer=csv.DictWriter(log,fieldnames=('epoch','global_step','train_loss','validation_dice','best_dice','run_type')); writer.writeheader()
 for epoch in range(1,3):
  result=run_train_epoch(model,islice(loader,8),loss,optimizer,device); step+=result.batch_count
  records=validate_cases(model,a.raw_root,val_ids,a.output_root,device,fold=fold,checkpoint_path=a.output_root/'checkpoints'/'checkpoint_latest.pth')
  dice=sum(float(r['dice']) for r in records)/len(records); best=save_smoke_checkpoints(model,optimizer,a.output_root,epoch=epoch,global_step=step,fold=fold,validation_dice=dice,best_dice=best,config={'run_type':'smoke_run_only','epochs':2,'max_train_batches':8,'validation_cases':3,'optimizer':'SGD','lr':.01,'momentum':.9})
  print({'smoke_run_only':True,'epoch':epoch,'train_loss':result.mean_loss,'validation_dice':dice,'best_dice':best})
  writer.writerow({'epoch':epoch,'global_step':step,'train_loss':result.mean_loss,'validation_dice':dice,'best_dice':best,'run_type':'smoke_run_only'}); log.flush()
 restored=PlainConvUNet2D(load_model_config()).to(device); restored_optimizer=torch.optim.SGD(restored.parameters(),lr=.01,momentum=.9); load_checkpoint(restored,restored_optimizer,a.output_root/'checkpoints'/'checkpoint_best.pth',{'smoke_run_only':True})
 replica=PlainConvUNet2D(load_model_config()).to(device); replica_optimizer=torch.optim.SGD(replica.parameters(),lr=.01,momentum=.9); load_checkpoint(replica,replica_optimizer,a.output_root/'checkpoints'/'checkpoint_best.pth',{'smoke_run_only':True})
 sample=torch.from_numpy(read_nifti(a.raw_root/'imagesTr'/f'{val_ids[0]}_0000.nii.gz').array[0:1]).unsqueeze(1).float().to(device)
 with torch.no_grad(): first=restored(sample); second=replica(sample)
 if not torch.equal(first.argmax(1),second.argmax(1)) or float((first-second).abs().max()) != 0.0: raise RuntimeError('checkpoint restore consistency check failed')
 validate_cases(restored,a.raw_root,val_ids,a.output_root,device,fold=fold,checkpoint_path=a.output_root/'checkpoints'/'checkpoint_best.pth')
 log.close()
 summary_path=a.output_root/'validation'/'summary.json'; summary=json.loads(summary_path.read_text(encoding='utf-8')); summary['optimizer']=config['optimizer']; summary['network']={'class_name':'PlainConvUNet2D','input_channels':1,'output_channels':2}; summary['data_preprocessing']='full-image Z-score; XY resampling bypassed when spacing matches'; summary['inference']='2D per-z argmax, reassembled zyx; saved masks validated against label space'; summary['resolved_config_path']=str(a.output_root/'resolved_config.json'); summary_path.write_text(json.dumps(summary,indent=2),encoding='utf-8')
 return 0
if __name__=='__main__': raise SystemExit(main())
