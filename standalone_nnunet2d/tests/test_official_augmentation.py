from __future__ import annotations
import numpy as np
import pytest
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


def test_multichannel_spatial_mirror_keeps_channels_and_label_synchronized() -> None:
    image = np.stack(
        [np.array([[1, 2], [3, 4]]), np.array([[10, 20], [30, 40]])], axis=0
    )
    label = np.array([[0, 1], [0, 0]])

    mirrored, mirrored_label = mirror_pair(image, label, (1,))

    np.testing.assert_array_equal(mirrored[0], np.array([[2, 1], [4, 3]]))
    np.testing.assert_array_equal(mirrored[1], np.array([[20, 10], [40, 30]]))
    np.testing.assert_array_equal(mirrored_label, np.array([[1, 0], [0, 0]]))


def test_batchgeneratorsv2_adapter_preserves_multichannel_shape_and_syncs_spatial_transform() -> None:
    base = np.arange(32, dtype=np.float32).reshape(8, 4)
    image = np.stack([base, 100 + base], axis=0)
    label = np.zeros((8, 4), dtype=np.int16)
    label[2:6, 1:3] = 1

    first_image, first_label = apply_official_2d_batchgeneratorsv2(
        image, label, patch_size=(8, 4), use_mask_for_norm=(False, False), seed=123
    )
    second_image, second_label = apply_official_2d_batchgeneratorsv2(
        image, label, patch_size=(8, 4), use_mask_for_norm=(False, False), seed=123
    )

    assert first_image.shape == (2, 8, 4)
    assert first_label.shape == (8, 4)
    np.testing.assert_array_equal(first_image, second_image)
    np.testing.assert_array_equal(first_label, second_label)


def test_composed_augmentation_crops_multichannel_image_with_label_alignment() -> None:
    base = np.arange(20, dtype=np.float32).reshape(4, 5)
    image = np.stack([base, 100 + base], axis=0)
    label = np.zeros((4, 5), dtype=np.int16)
    label[1, 2] = 1

    augmented, augmented_label = apply_official_2d_augmentation(
        image, label, np.random.default_rng(14), patch_size=(2, 3)
    )

    np.testing.assert_array_equal(
        augmented,
        np.stack(
            [
                np.array([[6, 7, 8], [11, 12, 13]], dtype=np.float32),
                np.array([[106, 107, 108], [111, 112, 113]], dtype=np.float32),
            ],
            axis=0,
        ),
    )
    np.testing.assert_array_equal(
        augmented_label,
        np.array([[0, 1, 0], [0, 0, 0]], dtype=np.int16),
    )


def test_batchgeneratorsv2_adapter_expands_single_mask_flag_for_multichannel_input() -> None:
    base = np.arange(16, dtype=np.float32).reshape(4, 4)
    image = np.stack([base, 100 + base], axis=0)
    label = np.zeros((4, 4), dtype=np.int16)

    augmented, augmented_label = apply_official_2d_batchgeneratorsv2(
        image, label, patch_size=(4, 4), use_mask_for_norm=(True,), seed=123
    )

    assert augmented.shape == (2, 4, 4)
    assert augmented_label.shape == (4, 4)


@pytest.mark.parametrize("use_mask_for_norm", [(), (True, False, True)])
def test_batchgeneratorsv2_adapter_rejects_wrong_mask_flag_count_for_multichannel_input(
    use_mask_for_norm,
) -> None:
    image = np.zeros((2, 4, 4), dtype=np.float32)
    label = np.zeros((4, 4), dtype=np.int16)

    with pytest.raises(ValueError, match="expected 2, got"):
        apply_official_2d_batchgeneratorsv2(
            image, label, patch_size=(4, 4), use_mask_for_norm=use_mask_for_norm, seed=123
        )
