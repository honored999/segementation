from __future__ import annotations
import argparse
from pathlib import Path
import sys, torch, numpy as np, json, random
if __package__ in {None, ""}: sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from optical_deeplab2d.datasets.dataset_2d import read_manifest,load_sample
from optical_deeplab2d.datasets.split import build_patient_folds
from optical_deeplab2d.evaluation.io import write_evaluation
from optical_deeplab2d.evaluation.visualization import select_representative_rows, save_validation_grid
from optical_deeplab2d.models.hybrid_deeplabv3plus import HybridOpticalDeepLabV3Plus
from optical_deeplab2d.models.electronic_deeplabv3plus import ElectronicDeepLabV3Plus
from optical_deeplab2d.models.electronic_deepseg_decoder import ElectronicDeepSegDecoder
MODEL_TYPES = {'hybrid_ideal': HybridOpticalDeepLabV3Plus, 'electronic_baseline': ElectronicDeepLabV3Plus, 'electronic_deepseg_decoder': ElectronicDeepSegDecoder}
def main() -> None:
 p=argparse.ArgumentParser(description='Evaluate a checkpoint on one patient fold.');p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--fold',type=int,choices=range(5),required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--visualize-random',type=int);a=p.parse_args();c=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
 try: cls=MODEL_TYPES[c['model_type']]
 except KeyError as error: raise ValueError(f"Unknown model type: {c['model_type']}") from error
 m=cls(c['encoder_name'],None).eval();m.load_state_dict(c['model_state_dict']);records=read_manifest(a.data_root);fold=build_patient_folds(records,c['seed'])[a.fold]; selected=[r for r in records if r.patient in fold.val_patients]; selected=random.Random(c['seed']).sample(selected,k=min(a.visualize_random,len(selected))) if a.visualize_random else selected; rows=[]
 for r in selected:
  if r.patient in fold.val_patients:
   x,y=load_sample(r); prediction=(m(x.unsqueeze(0)).sigmoid().detach().numpy()>=c['threshold']); rows.append({'sample_id':f'{r.patient}_{r.timepoint}_{r.image_path.stem}','patient':r.patient,'timepoint':r.timepoint,'slice_index':r.image_path.stem,'image':x.numpy(),'target':y.numpy(),'prediction':prediction})
 a.output_dir.mkdir(parents=True,exist_ok=True)
 if a.visualize_random: save_validation_grid(rows,a.output_dir/f'validation_predictions_random{a.visualize_random}.png'); (a.output_dir/'visualization_samples.json').write_text(json.dumps([row['sample_id'] for row in rows],ensure_ascii=False,indent=2)); return
 print(write_evaluation(rows,a.output_dir)); save_validation_grid(select_representative_rows(rows),a.output_dir/'validation_predictions_best.png')
if __name__=='__main__':main()
