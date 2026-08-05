from __future__ import annotations
import numpy as np
from standalone_nnunet2d.data.dataset import load_fold_cases
from standalone_nnunet2d.data.nifti_io import NiftiVolume, write_nifti
from standalone_nnunet2d.training.formal_dataset import FormalPatchDataset

def test_formal_dataset_oversamples_foreground_slice_and_returns_patch(tmp_path) -> None:
 case_id=load_fold_cases(0,'train')[0]; image=NiftiVolume(np.ones((3,4,4),dtype=np.float32),(1,1,1),(0,0,0)); label_array=np.zeros((3,4,4),dtype=np.int16); label_array[2,1,2]=1; label=NiftiVolume(label_array,(1,1,1),(0,0,0))
 write_nifti(tmp_path/'imagesTr'/f'{case_id}_0000.nii.gz',image); write_nifti(tmp_path/'labelsTr'/f'{case_id}.nii.gz',label)
 ds=FormalPatchDataset(tmp_path,fold=0,split='train',case_ids=(case_id,),patch_size=(4,4),oversample_foreground_percent=1.,rng=np.random.default_rng(0),augment=False)
 _,target=ds[0]
 assert tuple(target.shape)==(4,4) and target.sum().item()>0 and set(target.unique().tolist())<={0,1}
