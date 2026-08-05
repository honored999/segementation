"""Server-only smoke diagnostic for deliberately overfitting one case."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
import numpy as np
from standalone_nnunet2d.config import load_model_config
from standalone_nnunet2d.data.dataset import StrokeSliceDataset
from standalone_nnunet2d.engine.trainer import train_step
from standalone_nnunet2d.engine.predictor import predict_volume
from standalone_nnunet2d.metrics.case_metrics import volume_metrics
from standalone_nnunet2d.data.nifti_io import NiftiVolume
from standalone_nnunet2d.data.nifti_io import read_nifti
from standalone_nnunet2d.engine.predictor import save_and_validate_prediction
from standalone_nnunet2d.metrics.overlays import write_overlay
from standalone_nnunet2d.losses.compound import DiceCrossEntropyLoss
from standalone_nnunet2d.models import PlainConvUNet2D

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--raw-root",required=True,type=Path); p.add_argument("--output-root",required=True,type=Path); p.add_argument("--case-id",required=True); p.add_argument("--fold",type=int,default=0); p.add_argument("--device",default="cuda"); p.add_argument("--iterations",type=int,default=200); p.add_argument("--seed",type=int,default=0); a=p.parse_args()
    if a.iterations <= 0: p.error("iterations must be positive")
    torch.manual_seed(a.seed); device=torch.device(a.device)
    dataset=StrokeSliceDataset(a.raw_root,fold=a.fold,split="train",case_ids=(a.case_id,),foreground_probability=0.0)
    image, label = dataset.load_case(a.case_id)
    nonempty_z=np.where(label.sum(axis=(1,2))>0)[0]
    if len(nonempty_z)==0: p.error("overfit case must contain at least one lesion slice")
    batches=[(torch.from_numpy(image[z:z+1]).unsqueeze(1).float(), torch.from_numpy(label[z:z+1]).long()) for z in nonempty_z]
    model=PlainConvUNet2D(load_model_config()).to(device); optimizer=torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=0.0); loss=DiceCrossEntropyLoss()
    for step in range(1,a.iterations+1):
        result=train_step(model,batches[(step-1) % len(batches)],loss,optimizer,device)
        if step % 20 == 0 or step == 1:
            prediction=predict_volume(model,NiftiVolume(image,(0.4892368018627167,0.4892368018627167,1.0),(0,0,0)),device)
            print({"smoke_run_only":True,"case_id":a.case_id,"training_z_indices":nonempty_z.astype(int).tolist(),"step":step,"loss":result.loss,**volume_metrics(prediction,label.astype(np.uint8))})
    reference=read_nifti(a.raw_root / "labelsTr" / f"{a.case_id}.nii.gz")
    image_reference=read_nifti(a.raw_root / "imagesTr" / f"{a.case_id}_0000.nii.gz")
    prediction=predict_volume(model,image_reference,device)
    metrics=volume_metrics(prediction,reference.array.astype(np.uint8))
    save_and_validate_prediction(a.output_root / "validation" / "predictions" / f"{a.case_id}.nii.gz",prediction,reference)
    write_overlay(a.output_root / "validation" / "overlays" / f"{a.case_id}.png",image_reference.array,reference.array,prediction,case_id=a.case_id,dice=float(metrics["dice"]))
    return 0
if __name__=="__main__": raise SystemExit(main())
