from __future__ import annotations
import numpy as np
import pytest
from standalone_nnunet2d.data.dataset import load_fold_cases
from standalone_nnunet2d.data.nifti_io import NiftiVolume, write_nifti
from standalone_nnunet2d.training import formal_dataset
from standalone_nnunet2d.training.formal_dataset import FormalPatchDataset

def test_formal_dataset_oversamples_foreground_slice_and_returns_patch(tmp_path) -> None:
 case_id=load_fold_cases(0,'train')[0]; image=NiftiVolume(np.ones((3,4,4),dtype=np.float32),(1,1,1),(0,0,0)); label_array=np.zeros((3,4,4),dtype=np.int16); label_array[2,1,2]=1; label=NiftiVolume(label_array,(1,1,1),(0,0,0))
 write_nifti(tmp_path/'imagesTr'/f'{case_id}_0000.nii.gz',image); write_nifti(tmp_path/'labelsTr'/f'{case_id}.nii.gz',label)
 ds=FormalPatchDataset(tmp_path,fold=0,split='train',case_ids=(case_id,),patch_size=(4,4),oversample_foreground_percent=1.,rng=np.random.default_rng(0),augment=False)
 _,target=ds[0]
 assert tuple(target.shape)==(4,4) and target.sum().item()>0 and set(target.unique().tolist())<={0,1}


def test_formal_dataset_uses_plan_augmentation_config_and_deterministic_sample_seed(tmp_path, monkeypatch) -> None:
 case_id=load_fold_cases(0,'train')[0]
 image=NiftiVolume(np.ones((3,4,4),dtype=np.float32),(1,1,1),(0,0,0))
 label_array=np.zeros((3,4,4),dtype=np.int16); label_array[2,1,2]=1
 label=NiftiVolume(label_array,(1,1,1),(0,0,0))
 write_nifti(tmp_path/'imagesTr'/f'{case_id}_0000.nii.gz',image); write_nifti(tmp_path/'labelsTr'/f'{case_id}.nii.gz',label)
 calls=[]
 def fake_adapter(image, label, *, patch_size, use_mask_for_norm, seed):
  calls.append((patch_size, use_mask_for_norm, seed))
  return image, label
 monkeypatch.setattr(formal_dataset, 'apply_official_2d_batchgeneratorsv2', fake_adapter, raising=False)
 first=FormalPatchDataset(tmp_path,fold=0,split='train',case_ids=(case_id,),patch_size=(4,4),use_mask_for_norm=(True,),oversample_foreground_percent=1.,rng=np.random.default_rng(0),augment=True)
 second=FormalPatchDataset(tmp_path,fold=0,split='train',case_ids=(case_id,),patch_size=(4,4),use_mask_for_norm=(True,),oversample_foreground_percent=1.,rng=np.random.default_rng(0),augment=True)
 first[0]; second[0]
 assert calls[0][0:2]==((4,4),(True,))
 assert isinstance(calls[0][2],int) and calls[0][2]==calls[1][2]


def test_formal_dataset_returns_all_declared_channels_in_one_patch(tmp_path) -> None:
 (tmp_path/'dataset.json').write_text('{"channel_names": {"0": "DWI", "1": "ADC"}}',encoding='utf-8')
 case_id=load_fold_cases(0,'train')[0]
 image0=NiftiVolume(np.arange(48,dtype=np.float32).reshape(3,4,4),(1,1,1),(0,0,0))
 image1=NiftiVolume((100+np.arange(48,dtype=np.float32)).reshape(3,4,4),(1,1,1),(0,0,0))
 label_array=np.zeros((3,4,4),dtype=np.int16); label_array[2,1,2]=1
 label=NiftiVolume(label_array,(1,1,1),(0,0,0))
 write_nifti(tmp_path/'imagesTr'/f'{case_id}_0000.nii.gz',image0)
 write_nifti(tmp_path/'imagesTr'/f'{case_id}_0001.nii.gz',image1)
 write_nifti(tmp_path/'labelsTr'/f'{case_id}.nii.gz',label)
 ds=FormalPatchDataset(tmp_path,fold=0,split='train',case_ids=(case_id,),patch_size=(4,4),use_mask_for_norm=(False,False),oversample_foreground_percent=1.,rng=np.random.default_rng(0),augment=False)

 image_patch,target=ds[0]
 assert tuple(image_patch.shape)==(2,4,4)
 assert tuple(target.shape)==(4,4)


@pytest.mark.parametrize(
    ("use_mask_for_norm", "expected"),
    [((True,), (True, True)), ((False, True), (False, True))],
)
def test_formal_dataset_expands_single_mask_flag_for_multichannel_input(
    tmp_path, use_mask_for_norm, expected
) -> None:
 (tmp_path/'dataset.json').write_text('{"channel_names": {"0": "DWI", "1": "ADC"}}',encoding='utf-8')
 (tmp_path/'imagesTr').mkdir(); (tmp_path/'labelsTr').mkdir()
 case_id=load_fold_cases(0,'train')[0]

 ds=FormalPatchDataset(
     tmp_path, fold=0, split='train', case_ids=(case_id,), patch_size=(4,4),
     use_mask_for_norm=use_mask_for_norm, augment=False,
 )

 assert ds.use_mask_for_norm == expected


@pytest.mark.parametrize("use_mask_for_norm", [(), (True, False, True)])
def test_formal_dataset_rejects_wrong_mask_flag_count_for_multichannel_input(
    tmp_path, use_mask_for_norm
) -> None:
 (tmp_path/'dataset.json').write_text('{"channel_names": {"0": "DWI", "1": "ADC"}}',encoding='utf-8')
 (tmp_path/'imagesTr').mkdir(); (tmp_path/'labelsTr').mkdir()
 case_id=load_fold_cases(0,'train')[0]

 with pytest.raises(ValueError, match="expected 2, got"):
  FormalPatchDataset(
      tmp_path, fold=0, split='train', case_ids=(case_id,), patch_size=(4,4),
      use_mask_for_norm=use_mask_for_norm, augment=False,
  )
