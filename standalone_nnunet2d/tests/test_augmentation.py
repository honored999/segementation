from __future__ import annotations

import numpy as np

from standalone_nnunet2d.data.augmentation import AugmentationConfig, augment_slice


def test_default_augmentation_is_identity() -> None:
    image = np.arange(4, dtype=np.float32).reshape(2, 2)
    label = np.array([[0, 1], [1, 0]], dtype=np.int16)

    augmented_image, augmented_label = augment_slice(image, label, np.random.default_rng(1), AugmentationConfig())

    np.testing.assert_array_equal(augmented_image, image)
    np.testing.assert_array_equal(augmented_label, label)


def test_horizontal_flip_is_synchronized_for_image_and_label() -> None:
    image = np.array([[1, 2], [3, 4]], dtype=np.float32)
    label = np.array([[0, 1], [1, 0]], dtype=np.int16)
    config = AugmentationConfig(horizontal_flip_probability=1.0)

    result_image, result_label = augment_slice(image, label, np.random.default_rng(1), config)

    np.testing.assert_array_equal(result_image, image[:, ::-1])
    np.testing.assert_array_equal(result_label, label[:, ::-1])
