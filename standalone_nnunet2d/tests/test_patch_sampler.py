from __future__ import annotations
import numpy as np
from standalone_nnunet2d.training.patch_sampler import crop_or_pad, sample_patch_center

def test_foreground_sampling_uses_lesion_coordinate_when_probability_one() -> None:
 label=np.zeros((4,4),dtype=np.uint8); label[2,3]=1
 assert sample_patch_center(label,np.random.default_rng(0),oversample_foreground_percent=1.)==(2,3)

def test_crop_or_pad_preserves_edge_label() -> None:
 image=np.ones((2,2),dtype=np.float32); label=np.array([[1,0],[0,0]],dtype=np.uint8)
 _,patch=crop_or_pad(image,label,(0,0),(4,4))
 assert patch.shape==(4,4) and patch.sum()==1 and set(np.unique(patch))<= {0,1}
