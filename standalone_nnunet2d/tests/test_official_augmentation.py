from __future__ import annotations
import numpy as np
from standalone_nnunet2d.training.official_augmentation import Official2DAugmentationConfig, apply_official_2d_augmentation, mirror_pair, rotate_pair

def test_official_2d_augmentation_constants() -> None:
 c=Official2DAugmentationConfig(); assert c.rotation_radians==(-np.pi,np.pi) and c.scaling_range==(.7,1.4) and c.mirror_axes==(0,1)
def test_mirror_pair_keeps_image_label_geometry_synced() -> None:
 image=np.array([[1,2],[3,4]]); label=np.array([[0,1],[0,0]])
 _,mirrored=mirror_pair(image,label,(1,)); np.testing.assert_array_equal(mirrored,np.array([[1,0],[0,0]]))
def test_rotate_pair_keeps_label_discrete() -> None:
 image=np.ones((5,5),dtype=np.float32); label=np.zeros((5,5),dtype=np.int16); label[2,2]=1
 _,rotated=rotate_pair(image,label,30); assert rotated.shape==label.shape and set(np.unique(rotated))<={-1,0,1}
def test_composed_augmentation_returns_fixed_patch_and_discrete_label() -> None:
 image=np.ones((8,8),dtype=np.float32); label=np.zeros((8,8),dtype=np.int16); label[3,4]=1
 augmented,seg=apply_official_2d_augmentation(image,label,np.random.default_rng(1),(8,8))
 assert augmented.shape==(8,8) and seg.shape==(8,8) and set(np.unique(seg))<={-1,0,1}
