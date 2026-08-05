"""Read-only validation of saved smoke-only prediction NIfTI files."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from standalone_nnunet2d.data.nifti_io import read_nifti

def main() -> int:
 p=argparse.ArgumentParser(); p.add_argument('--raw-root',required=True,type=Path); p.add_argument('--predictions-root',required=True,type=Path); p.add_argument('--case-id',action='append',required=True); a=p.parse_args()
 for case_id in a.case_id:
  gt=read_nifti(a.raw_root/'labelsTr'/f'{case_id}.nii.gz'); pred=read_nifti(a.predictions_root/f'{case_id}.nii.gz')
  if pred.array.shape!=gt.array.shape or pred.array.dtype!=np.uint8 or not np.isin(pred.array,(0,1)).all() or not np.allclose(pred.spacing_xyz,gt.spacing_xyz) or not np.allclose(pred.origin_xyz,gt.origin_xyz) or not np.allclose(pred.direction,gt.direction): raise ValueError(f'prediction validation failed for {case_id}')
  print({'case_id':case_id,'shape':pred.array.shape,'dtype':str(pred.array.dtype),'unique':np.unique(pred.array).tolist(),'spatial_match':True})
 return 0
if __name__=='__main__': raise SystemExit(main())
