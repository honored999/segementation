from __future__ import annotations
import argparse
from pathlib import Path
import sys, torch, numpy as np
if __package__ in {None, ""}: sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from optical_deeplab2d.datasets.dataset_2d import read_manifest,load_sample
from optical_deeplab2d.datasets.split import build_patient_folds
from optical_deeplab2d.evaluation.io import write_evaluation
from optical_deeplab2d.evaluation.visualization import select_representative_rows, save_validation_grid
from optical_deeplab2d.models.hybrid_deeplabv3plus import HybridOpticalDeepLabV3Plus
from optical_deeplab2d.models.electronic_deeplabv3plus import ElectronicDeepLabV3Plus
def main() -> None:
 p=argparse.ArgumentParser(description='Evaluate a checkpoint on one patient fold.');p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--fold',type=int,choices=range(5),required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();c=torch.load(a.checkpoint,map_location='cpu',weights_only=False);cls=HybridOpticalDeepLabV3Plus if c['model_type']=='hybrid_ideal' else ElectronicDeepLabV3Plus;m=cls(c['encoder_name'],None).eval();m.load_state_dict(c['model_state_dict']);fold=build_patient_folds(read_manifest(a.data_root),c['seed'])[a.fold]; rows=[]
 for r in read_manifest(a.data_root):
  if r.patient in fold.val_patients:
   x,y=load_sample(r); prediction=(m(x).sigmoid().detach().numpy()>=c['threshold']); rows.append({'sample_id':f'{r.patient}_{r.timepoint}_{r.image_path.stem}','patient':r.patient,'timepoint':r.timepoint,'slice_index':r.image_path.stem,'image':x.numpy(),'target':y.numpy(),'prediction':prediction})
 print(write_evaluation(rows,a.output_dir)); save_validation_grid(select_representative_rows(rows),a.output_dir/'validation_predictions_best.png')
if __name__=='__main__':main()
