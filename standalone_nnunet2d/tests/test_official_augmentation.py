from __future__ import annotations
import numpy as np
from standalone_nnunet2d.training.official_augmentation import (
    Official2DAugmentationConfig,
    apply_official_2d_augmentation,
    apply_official_2d_batchgeneratorsv2,
    mirror_pair,
    rotate_pair,
    rotation_for_patch_size,
)

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


def test_batchgeneratorsv2_adapter_is_deterministic_and_uses_plan_rotation() -> None:
    image = np.arange(32, dtype=np.float32).reshape(8, 4)
    label = np.zeros((8, 4), dtype=np.int16)
    label[2:6, 1:3] = 1

    anisotropic_rotation = rotation_for_patch_size((8, 4))
    isotropic_rotation = rotation_for_patch_size((8, 8))
    np.testing.assert_allclose(
        anisotropic_rotation,
        (-15.0 / 360.0 * 2.0 * np.pi, 15.0 / 360.0 * 2.0 * np.pi),
    )
    np.testing.assert_allclose(
        isotropic_rotation,
        (-180.0 / 360.0 * 2.0 * np.pi, 180.0 / 360.0 * 2.0 * np.pi),
    )
    assert anisotropic_rotation != isotropic_rotation

    first_image, first_label = apply_official_2d_batchgeneratorsv2(
        image,
        label,
        patch_size=(8, 4),
        use_mask_for_norm=(True,),
        seed=123,
    )
    second_image, second_label = apply_official_2d_batchgeneratorsv2(
        image,
        label,
        patch_size=(8, 4),
        use_mask_for_norm=(True,),
        seed=123,
    )

    assert first_image.shape == (8, 4)
    assert first_label.shape == (8, 4)
    assert isinstance(first_image, np.ndarray)
    assert isinstance(first_label, np.ndarray)
    assert not np.any(first_label == -1)
    np.testing.assert_array_equal(first_image, second_image)
    np.testing.assert_array_equal(first_label, second_label)
