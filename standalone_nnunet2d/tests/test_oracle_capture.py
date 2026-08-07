from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from standalone_nnunet2d import oracle_capture


def _write_legacy_npz(root: Path, case_id: str) -> tuple[np.ndarray, np.ndarray]:
    image = np.arange(24, dtype=np.float64).reshape(1, 2, 3, 4)
    label = np.arange(24, dtype=np.int32).reshape(1, 2, 3, 4)
    np.savez_compressed(root / f"{case_id}.npz", data=image, seg=label)
    return image, label


class _LazyArray:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        return np.asarray(self.value, dtype=dtype)


def test_read_preprocessed_case_prefers_the_single_legacy_npz(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case_id = "case_legacy"
    image, label = _write_legacy_npz(tmp_path, case_id)
    (tmp_path / f"{case_id}.b2nd").touch()
    (tmp_path / f"{case_id}_seg.b2nd").touch()

    real_import = oracle_capture.importlib.import_module

    def reject_eager_blosc2(name: str, *args: object, **kwargs: object) -> object:
        if name == "blosc2":
            raise AssertionError("blosc2 must not be imported for an .npz case")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(oracle_capture.importlib, "import_module", reject_eager_blosc2)

    result_image, result_label = oracle_capture._read_preprocessed_case(tmp_path, case_id)

    np.testing.assert_array_equal(result_image, image[0].astype(np.float32))
    np.testing.assert_array_equal(result_label, label[0].astype(np.int16))
    assert result_image.dtype == np.float32
    assert result_label.dtype == np.int16


def test_read_preprocessed_case_reads_b2nd_data_and_seg_through_lazy_blosc2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_id = "case_b2nd"
    image = np.arange(24, dtype=np.float64).reshape(1, 2, 3, 4)
    label = np.arange(24, dtype=np.int32).reshape(1, 2, 3, 4)
    data_path = tmp_path / f"{case_id}.b2nd"
    seg_path = tmp_path / f"{case_id}_seg.b2nd"
    data_path.touch()
    seg_path.touch()
    opened: list[tuple[str, str]] = []

    def fake_open(*, urlpath: str, mode: str, **kwargs: object) -> _LazyArray:
        del kwargs
        name = Path(urlpath).name
        opened.append((name, mode))
        return _LazyArray({data_path.name: image, seg_path.name: label}[name])

    monkeypatch.setitem(sys.modules, "blosc2", SimpleNamespace(open=fake_open))

    result_image, result_label = oracle_capture._read_preprocessed_case(tmp_path, case_id)

    np.testing.assert_array_equal(result_image, image[0].astype(np.float32))
    np.testing.assert_array_equal(result_label, label[0].astype(np.int16))
    assert opened == [(data_path.name, "r"), (seg_path.name, "r")]


def test_read_preprocessed_case_reports_missing_b2nd_segmentation(tmp_path: Path) -> None:
    case_id = "case_missing_seg"
    (tmp_path / f"{case_id}.b2nd").touch()

    with pytest.raises(FileNotFoundError, match=rf"{case_id}_seg\.b2nd"):
        oracle_capture._read_preprocessed_case(tmp_path, case_id)


def test_official_transform_uses_current_nnunet_trainer_module_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.arange(6, dtype=np.float32).reshape(2, 3)
    label = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int16)
    plans_path = tmp_path / "nnUNetPlans.json"
    plans_path.write_text(
        json.dumps(
            {
                "configurations": {
                    "2d": {"patch_size": [512, 512], "use_mask_for_norm": [True]}
                }
            }
        ),
        encoding="utf-8",
    )
    imported: list[str] = []

    def factory(
        *,
        patch_size: tuple[int, ...],
        rotation_for_DA: tuple[float, float],
        deep_supervision_scales: None,
        mirror_axes: tuple[int, int],
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm: list[bool],
    ) -> object:
        assert patch_size == (512, 512)
        np.testing.assert_allclose(rotation_for_DA, (-np.pi, np.pi))
        assert deep_supervision_scales is None
        assert mirror_axes == (0, 1)
        assert do_dummy_2d_data_aug is False
        assert use_mask_for_norm == [True]

        def transform(*, image: np.ndarray, segmentation: np.ndarray) -> dict[str, np.ndarray]:
            return {"image": image + 1, "segmentation": segmentation}

        return transform

    class Trainer:
        @classmethod
        def get_training_transforms(cls, **kwargs: object) -> object:
            return factory(**kwargs)

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        del args, kwargs
        imported.append(name)
        if name != "nnunetv2.training.nnUNetTrainer.nnUNetTrainer":
            raise ModuleNotFoundError(name)
        return SimpleNamespace(get_training_transforms=factory, nnUNetTrainer=Trainer)

    monkeypatch.setattr(oracle_capture.importlib, "import_module", fake_import)

    result_image, result_label = oracle_capture._official_transform(image, label, seed=7, plans_path=plans_path)

    np.testing.assert_array_equal(result_image, image + 1)
    np.testing.assert_array_equal(result_label, label)
    assert imported == ["nnunetv2.training.nnUNetTrainer.nnUNetTrainer"]


def test_official_transform_looks_up_trainer_class_method_and_passes_rotation_for_da(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.arange(6, dtype=np.float32).reshape(2, 3)
    label = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int16)
    plans_path = tmp_path / "nnUNetPlans.json"
    plans_path.write_text(
        json.dumps(
            {
                "configurations": {
                    "2d": {"patch_size": [512, 512], "use_mask_for_norm": [True]}
                }
            }
        ),
        encoding="utf-8",
    )
    received: dict[str, object] = {}

    class Trainer:
        @classmethod
        def get_training_transforms(
            cls,
            *,
            patch_size: tuple[int, ...],
            rotation_for_DA: tuple[float, float],
            deep_supervision_scales: None,
            mirror_axes: tuple[int, int],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: list[bool],
        ) -> object:
            del cls
            received.update(
                {
                    "patch_size": patch_size,
                    "rotation_for_DA": rotation_for_DA,
                    "deep_supervision_scales": deep_supervision_scales,
                        "mirror_axes": mirror_axes,
                        "do_dummy_2d_data_aug": do_dummy_2d_data_aug,
                        "use_mask_for_norm": use_mask_for_norm,
                    }
                )

            def transform(*, image: np.ndarray, segmentation: np.ndarray) -> dict[str, np.ndarray]:
                return {"image": image, "segmentation": segmentation}

            return transform

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        del args, kwargs
        if name != "nnunetv2.training.nnUNetTrainer.nnUNetTrainer":
            raise ModuleNotFoundError(name)
        return SimpleNamespace(nnUNetTrainer=Trainer)

    monkeypatch.setattr(oracle_capture.importlib, "import_module", fake_import)

    result_image, result_label = oracle_capture._official_transform(image, label, seed=7, plans_path=plans_path)

    np.testing.assert_array_equal(result_image, image)
    np.testing.assert_array_equal(result_label, label)
    assert received == {
        "patch_size": (512, 512),
        "rotation_for_DA": (-np.pi, np.pi),
        "deep_supervision_scales": None,
        "mirror_axes": (0, 1),
        "do_dummy_2d_data_aug": False,
        "use_mask_for_norm": [True],
    }


def test_official_transform_passes_sample_fields_as_keyword_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.arange(6, dtype=np.float32).reshape(2, 3)
    label = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int16)
    plans_path = tmp_path / "nnUNetPlans.json"
    plans_path.write_text(
        json.dumps(
            {
                "configurations": {
                    "2d": {"patch_size": [512, 512], "use_mask_for_norm": [True]}
                }
            }
        ),
        encoding="utf-8",
    )

    class Trainer:
        @classmethod
        def get_training_transforms(cls, **kwargs: object) -> object:
            del cls, kwargs

            def transform(*, image: np.ndarray, segmentation: np.ndarray) -> dict[str, np.ndarray]:
                return {"image": image, "segmentation": segmentation}

            return transform

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        del args, kwargs
        if name != "nnunetv2.training.nnUNetTrainer.nnUNetTrainer":
            raise ModuleNotFoundError(name)
        return SimpleNamespace(nnUNetTrainer=Trainer)

    monkeypatch.setattr(oracle_capture.importlib, "import_module", fake_import)

    result_image, result_label = oracle_capture._official_transform(
        image, label, seed=7, plans_path=plans_path
    )

    np.testing.assert_array_equal(result_image, image)
    np.testing.assert_array_equal(result_label, label)


def test_official_transform_uses_trainer_class_and_plan_patch_size_for_rotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = np.arange(6, dtype=np.float32).reshape(2, 3)
    label = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int16)
    plans_path = tmp_path / "nnUNetPlans.json"
    plans_path.write_text(
        json.dumps(
            {
                "configurations": {
                    "2d": {"patch_size": [512, 256], "use_mask_for_norm": [True]}
                }
            }
        ),
        encoding="utf-8",
    )
    received: dict[str, object] = {}

    class Trainer:
        @classmethod
        def get_training_transforms(
            cls,
            *,
            patch_size: tuple[int, ...],
            rotation_for_DA: tuple[float, float],
            deep_supervision_scales: None,
            mirror_axes: tuple[int, int],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: list[bool],
        ) -> object:
            del cls
            received.update(
                {
                    "patch_size": patch_size,
                    "rotation_for_DA": rotation_for_DA,
                    "deep_supervision_scales": deep_supervision_scales,
                        "mirror_axes": mirror_axes,
                        "do_dummy_2d_data_aug": do_dummy_2d_data_aug,
                        "use_mask_for_norm": use_mask_for_norm,
                    }
                )

            def transform(*, image: np.ndarray, segmentation: np.ndarray) -> dict[str, np.ndarray]:
                return {"image": image, "segmentation": segmentation}

            return transform

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        del args, kwargs
        if name != "nnunetv2.training.nnUNetTrainer.nnUNetTrainer":
            raise ModuleNotFoundError(name)
        return SimpleNamespace(nnUNetTrainer=Trainer)

    monkeypatch.setattr(oracle_capture.importlib, "import_module", fake_import)

    result_image, result_label = oracle_capture._official_transform(
        image, label, seed=7, plans_path=plans_path
    )

    np.testing.assert_array_equal(result_image, image)
    np.testing.assert_array_equal(result_label, label)
    rotation = 15.0 / 360.0 * 2.0 * math.pi
    assert received == {
        "patch_size": (512, 256),
        "rotation_for_DA": (-rotation, rotation),
        "deep_supervision_scales": None,
        "mirror_axes": (0, 1),
        "do_dummy_2d_data_aug": False,
        "use_mask_for_norm": [True],
    }


def test_official_transform_uses_bgv2_fields_and_plan_mask_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.arange(6, dtype=np.float32).reshape(2, 3)
    label = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int16)
    plans_path = tmp_path / "nnUNetPlans.json"
    plans_path.write_text(
        json.dumps(
            {
                "configurations": {
                    "2d": {"patch_size": [512, 512], "use_mask_for_norm": [True]}
                }
            }
        ),
        encoding="utf-8",
    )
    received: dict[str, object] = {}

    class Trainer:
        @classmethod
        def get_training_transforms(
            cls,
            *,
            patch_size: tuple[int, ...],
            rotation_for_DA: tuple[float, float],
            deep_supervision_scales: None,
            mirror_axes: tuple[int, int],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: list[bool],
        ) -> object:
            del cls, patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes
            del do_dummy_2d_data_aug
            received["use_mask_for_norm"] = use_mask_for_norm

            def transform(*, image: np.ndarray, segmentation: np.ndarray) -> dict[str, np.ndarray]:
                received["input_fields"] = ("image", "segmentation")
                return {"image": image + 1, "segmentation": segmentation}

            return transform

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        del args, kwargs
        if name != "nnunetv2.training.nnUNetTrainer.nnUNetTrainer":
            raise ModuleNotFoundError(name)
        return SimpleNamespace(nnUNetTrainer=Trainer)

    monkeypatch.setattr(oracle_capture.importlib, "import_module", fake_import)

    result_image, result_label = oracle_capture._official_transform(
        image, label, seed=7, plans_path=plans_path
    )

    np.testing.assert_array_equal(result_image, image + 1)
    np.testing.assert_array_equal(result_label, label)
    assert received == {
        "use_mask_for_norm": [True],
        "input_fields": ("image", "segmentation"),
    }


def test_official_transform_passes_channel_and_spatial_axes_without_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.arange(6, dtype=np.float32).reshape(2, 3)
    label = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int16)
    plans_path = tmp_path / "nnUNetPlans.json"
    plans_path.write_text(
        json.dumps(
            {
                "configurations": {
                    "2d": {"patch_size": [512, 512], "use_mask_for_norm": [True]}
                }
            }
        ),
        encoding="utf-8",
    )

    class Trainer:
        @classmethod
        def get_training_transforms(cls, **kwargs: object) -> object:
            del cls, kwargs

            def transform(*, image: np.ndarray, segmentation: np.ndarray) -> dict[str, np.ndarray]:
                assert image.shape == (1, 2, 3)
                assert segmentation.shape == (1, 2, 3)
                return {"image": image, "segmentation": segmentation}

            return transform

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        del args, kwargs
        if name != "nnunetv2.training.nnUNetTrainer.nnUNetTrainer":
            raise ModuleNotFoundError(name)
        return SimpleNamespace(nnUNetTrainer=Trainer)

    monkeypatch.setattr(oracle_capture.importlib, "import_module", fake_import)

    result_image, result_label = oracle_capture._official_transform(
        image, label, seed=7, plans_path=plans_path
    )

    np.testing.assert_array_equal(result_image, image)
    np.testing.assert_array_equal(result_label, label)


def test_official_transform_passes_torch_tensors_to_bgv2_transform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    image = np.arange(6, dtype=np.float32).reshape(2, 3)
    label = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int16)
    plans_path = tmp_path / "nnUNetPlans.json"
    plans_path.write_text(
        json.dumps(
            {
                "configurations": {
                    "2d": {"patch_size": [512, 512], "use_mask_for_norm": [True]}
                }
            }
        ),
        encoding="utf-8",
    )

    class Trainer:
        @classmethod
        def get_training_transforms(cls, **kwargs: object) -> object:
            del cls, kwargs

            def transform(*, image: torch.Tensor, segmentation: torch.Tensor) -> dict[str, torch.Tensor]:
                return {
                    "image": image.contiguous(),
                    "segmentation": segmentation.contiguous(),
                }

            return transform

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        del args, kwargs
        if name != "nnunetv2.training.nnUNetTrainer.nnUNetTrainer":
            raise ModuleNotFoundError(name)
        return SimpleNamespace(nnUNetTrainer=Trainer)

    monkeypatch.setattr(oracle_capture.importlib, "import_module", fake_import)

    result_image, result_label = oracle_capture._official_transform(
        image, label, seed=7, plans_path=plans_path
    )

    np.testing.assert_array_equal(result_image, image)
    np.testing.assert_array_equal(result_label, label)
