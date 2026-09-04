from __future__ import annotations

import numpy as np

from standalone_nnunet2d.training.formal_transforms import (
    FormalTransformResult,
    SpatialResult,
    apply_blur,
    apply_brightness,
    apply_contrast,
    apply_formal_spatial_transform,
    apply_gamma,
    apply_gamma_inversion,
    apply_low_resolution,
    apply_mirroring,
    apply_noise,
    make_deep_supervision_targets,
    remove_label_values,
)


def test_spatial_transform_uses_initial_patch_then_crops_final_patch() -> None:
    image = np.arange(64, dtype=np.float32).reshape(8, 8)
    label = np.zeros((8, 8), dtype=np.int16)
    result = apply_formal_spatial_transform(
        image,
        label,
        rng=np.random.default_rng(4),
        initial_patch_size=(12, 12),
        patch_size=(8, 8),
        rotation_probability=0.0,
        scale_probability=0.0,
    )

    assert isinstance(result, SpatialResult)
    assert result.image.shape == (8, 8)
    assert result.label.shape == (8, 8)
    assert set(np.unique(result.label)) <= {-1, 0, 1}


def test_spatial_transform_is_seed_reproducible_with_exact_labels_and_close_images() -> None:
    image = np.linspace(-1.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    label = np.zeros((8, 8), dtype=np.int16)
    label[2:5, 3:6] = 1
    kwargs = dict(
        initial_patch_size=(10, 10),
        patch_size=(8, 8),
        rotation_probability=1.0,
        rotation_radians=(0.2, 0.2),
        scale_probability=1.0,
        scale_range=(1.1, 1.1),
    )

    first = apply_formal_spatial_transform(image, label, rng=np.random.default_rng(9), **kwargs)
    second = apply_formal_spatial_transform(image, label, rng=np.random.default_rng(9), **kwargs)

    np.testing.assert_allclose(first.image, second.image)
    np.testing.assert_array_equal(first.label, second.label)
    assert set(np.unique(first.label)) <= {-1, 0, 1}


def test_each_intensity_operator_has_an_explicit_disabled_path() -> None:
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    label = np.arange(16, dtype=np.int16).reshape(4, 4)
    rng = np.random.default_rng(3)

    np.testing.assert_array_equal(apply_noise(image, rng, probability=0.0), image)
    np.testing.assert_array_equal(apply_blur(image, rng, probability=0.0), image)
    np.testing.assert_array_equal(apply_brightness(image, rng, probability=0.0), image)
    np.testing.assert_array_equal(apply_contrast(image, rng, probability=0.0), image)
    np.testing.assert_array_equal(apply_low_resolution(image, rng, probability=0.0), image)
    np.testing.assert_array_equal(apply_gamma(image, rng, probability=0.0), image)
    np.testing.assert_array_equal(apply_gamma_inversion(image, rng, probability=0.0), image)
    mirrored_image, mirrored_label = apply_mirroring(image, label, rng, probability=0.0)
    np.testing.assert_array_equal(mirrored_image, image)
    np.testing.assert_array_equal(mirrored_label, label)


def test_mirroring_is_paired_and_gamma_stages_are_reproducible() -> None:
    image = np.arange(9, dtype=np.float32).reshape(3, 3)
    label = np.arange(9, dtype=np.int16).reshape(3, 3)
    mirrored_image, mirrored_label = apply_mirroring(
        image, label, np.random.default_rng(1), probability=1.0, axes=(1,)
    )
    np.testing.assert_array_equal(mirrored_image, image[:, ::-1])
    np.testing.assert_array_equal(mirrored_label, label[:, ::-1])

    first = apply_gamma_inversion(
        image, np.random.default_rng(5), probability=1.0, gamma_range=(0.8, 0.8)
    )
    second = apply_gamma_inversion(
        image, np.random.default_rng(5), probability=1.0, gamma_range=(0.8, 0.8)
    )
    np.testing.assert_allclose(first, second)


def test_remove_label_values_is_explicit_and_deep_supervision_uses_nearest_values() -> None:
    label = np.array([[-1, 0, 1, 1], [-1, 1, 0, 0]], dtype=np.int16)
    removed = remove_label_values(label, values=(-1,), replacement=0)
    np.testing.assert_array_equal(removed, np.array([[0, 0, 1, 1], [0, 1, 0, 0]], dtype=np.int16))

    targets = make_deep_supervision_targets(removed, scales=((1.0, 1.0), (0.5, 0.5)))
    assert [target.shape for target in targets] == [(2, 4), (1, 2)]
    assert all(set(np.unique(target)) <= {0, 1} for target in targets)
    np.testing.assert_array_equal(targets[0], removed)


def test_formal_transform_result_keeps_image_and_label_contract() -> None:
    result = FormalTransformResult(
        image=np.zeros((2, 2), dtype=np.float32),
        label=np.zeros((2, 2), dtype=np.int16),
    )
    assert result.image.shape == result.label.shape
