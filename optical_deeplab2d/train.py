"""Training entry point with reproducible validation and checkpoint selection."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch, yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
if __package__ in {None,""}: sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from optical_deeplab2d.datasets.dataset_2d import DwiSliceDataset,collate_samples,read_manifest
from optical_deeplab2d.datasets.split import build_patient_folds,save_folds
from optical_deeplab2d.evaluation.io import write_evaluation
from optical_deeplab2d.models.hybrid_deeplabv3plus import HybridOpticalDeepLabV3Plus
from optical_deeplab2d.models.electronic_deeplabv3plus import ElectronicDeepLabV3Plus
from optical_deeplab2d.models.electronic_deepseg_decoder import ElectronicDeepSegDecoder
from optical_deeplab2d.training.checkpoint import load_checkpoint,save_checkpoint,validate_resume
from optical_deeplab2d.training.logging import append_log
from optical_deeplab2d.training.losses import CombinedBCEDiceLoss
from optical_deeplab2d.training.seed import seed_everything

MODEL_TYPES = {
 'hybrid_ideal': HybridOpticalDeepLabV3Plus,
 'electronic_baseline': ElectronicDeepLabV3Plus,
 'electronic_deepseg_decoder': ElectronicDeepSegDecoder,
}

def args():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--fold',type=int,choices=range(5),required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--resume',type=Path);p.add_argument('--overfit-small-batch',action='store_true');return p.parse_args()
@torch.no_grad()
def validate(model,loader,criterion,device,threshold):
 model.eval();rows=[];total=0.
 for image,mask,records in loader:
  logits=model(image.to(device));total+=criterion(logits,mask.to(device)).item();prediction=(logits.sigmoid().cpu().numpy()>=threshold)
  for i,r in enumerate(records):rows.append({'patient':r.patient,'target':mask[i].numpy(),'prediction':prediction[i]})
 return total/max(len(loader),1),rows
def main():
 a=args();cfg=yaml.safe_load(a.config.read_text());seed_everything(cfg['seed']);a.output_dir.mkdir(parents=True,exist_ok=True);records=read_manifest(a.data_root);fold=build_patient_folds(records,cfg['seed'])[a.fold];save_folds(build_patient_folds(records,cfg['seed']),a.output_dir/'splits_final.json');train=[r for r in records if r.patient in fold.train_patients];val=[r for r in records if r.patient in fold.val_patients]
 if a.overfit_small_batch:train,val=train[:4],train[:4]
 pos=sum(r.has_mask for r in train);neg=len(train)-pos;weights=[1/max(pos,1) if r.has_mask else 1/max(neg,1) for r in train];sampler=WeightedRandomSampler(weights,len(train),replacement=True) if pos and neg else None
 tl=DataLoader(DwiSliceDataset(train),batch_size=cfg['training']['batch_size'],sampler=sampler,shuffle=sampler is None,num_workers=cfg['training']['num_workers'],collate_fn=collate_samples);vl=DataLoader(DwiSliceDataset(val),batch_size=cfg['training']['batch_size'],shuffle=False,num_workers=cfg['training']['num_workers'],collate_fn=collate_samples);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 try: cls=MODEL_TYPES[cfg['model']['type']]
 except KeyError as error: raise ValueError(f"Unknown model type: {cfg['model']['type']}") from error
 model=cls(cfg['model']['encoder_name'],cfg['model']['encoder_weights']).to(device);criterion=CombinedBCEDiceLoss(min(20.,max(1.,neg/max(pos,1)))).to(device);optim=torch.optim.AdamW(model.parameters(),lr=cfg['training']['new_layers_lr'],weight_decay=cfg['training']['weight_decay']);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optim,mode='max',patience=6,factor=.5);best=-1.;start=0
 if a.resume:
  ck=load_checkpoint(a.resume,device);validate_resume(ck,cfg);model.load_state_dict(ck['model_state_dict']);optim.load_state_dict(ck['optimizer_state_dict']);scheduler.load_state_dict(ck['scheduler_state_dict']);start=ck['epoch']+1;best=ck['best_metric']
 (a.output_dir/'config_resolved.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False));stale=0
 for epoch in range(start,cfg['training']['epochs']):
  began=time.time();model.train();losses=[]
  for image,mask,_ in tl:optim.zero_grad(set_to_none=True);value=criterion(model(image.to(device)),mask.to(device));value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['training']['max_grad_norm']);optim.step();losses.append(value.item())
  _,rows=validate(model,vl,criterion,device,cfg['training']['threshold']);summary=write_evaluation(rows,a.output_dir);metric=summary['mean_patient_dice'];scheduler.step(metric);meta={'model_type':cfg['model']['type'],'encoder_name':model.resolved_encoder,'threshold':cfg['training']['threshold'],'fold':a.fold,'seed':cfg['seed'],'normalization':cfg['data']['normalization'],'pos_weight':float(criterion.pos_weight.item()),'pairing_rule':'manifest image_path to mask_path','patient_id_rule':'manifest patient column','config':cfg};save_checkpoint(a.output_dir/'last.pt',model=model,optimizer=optim,scheduler=scheduler,epoch=epoch,best_metric=best,metadata=meta)
  if metric>best:best=metric;stale=0;save_checkpoint(a.output_dir/'best.pt',model=model,optimizer=optim,scheduler=scheduler,epoch=epoch,best_metric=best,metadata=meta)
  else:stale+=1
  append_log(a.output_dir/'train_log.csv',{'epoch':epoch+1,'train_total_loss':np.mean(losses),'val_global_dice':summary['global']['dice'],'val_mean_image_dice':summary['mean_image_dice'],'val_mean_patient_dice':metric,'val_precision':summary['global']['precision'],'val_recall':summary['global']['recall'],'encoder_lr':optim.param_groups[0]['lr'],'new_layers_lr':optim.param_groups[0]['lr'],'epoch_time':time.time()-began,'gpu_memory_mb':torch.cuda.max_memory_allocated()/1048576 if torch.cuda.is_available() else 0})
  if stale>=cfg['training']['early_stopping_patience']:break
if __name__=='__main__':main()
