from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from standalone_nnunet2d import predict as predict_module
from standalone_nnunet2d import standalone_capture
from standalone_nnunet2d import validate_cv
from standalone_nnunet2d.data import dataset as dataset_module
from standalone_nnunet2d.data.dataset import StrokeSliceDataset
from standalone_nnunet2d.data.nifti_io import NiftiVolume, write_nifti
from standalone_nnunet2d.engine import checkpoint as checkpoint_module
from standalone_nnunet2d.engine import formal_validation
from standalone_nnunet2d.formal_train import build_formal_config
from standalone_nnunet2d.training.formal_checkpoint import (
    FormalTrainerState,
    checkpoint_bilateral_asymmetry_channel,
    save_formal_checkpoint,
)
from standalone_nnunet2d.training.official_config import OfficialTrainerSchedule


IDENTITY_DIRECTION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def _dwi() -> np.ndarray:
    z, y, x = np.indices((3, 33, 33))
    del z
    image = (((x - 16) / 10.0) ** 2 + ((y - 16) / 5.0) ** 2 <= 1.0).astype(np.float32)
    image[:, 16, 22] = 4.0
    return image


def _volume(array: np.ndarray) -> NiftiVolume:
    return NiftiVolume(
        array.astype(np.float32, copy=False),
        spacing_xyz=(1.0, 1.0, 4.0),
        origin_xyz=(0.0, 0.0, 0.0),
        direction=IDENTITY_DIRECTION,
    )


def _write_case(root: Path, case_id: str) -> None:
    (root / "imagesTr").mkdir(parents=True)
    (root / "labelsTr").mkdir()
    (root / "dataset.json").write_text('{"channel_names": {"0": "DWI"}}', encoding="utf-8")
    image = _dwi()
    write_nifti(root / "imagesTr" / f"{case_id}_0000.nii.gz", _volume(image))
    write_nifti(
        root / "labelsTr" / f"{case_id}.nii.gz",
        NiftiVolume(np.zeros_like(image, dtype=np.uint8), (1.0, 1.0, 4.0), (0.0, 0.0, 0.0), IDENTITY_DIRECTION),
    )


def _bilateral_config() -> dict[str, object]:
    return build_formal_config(
        fold=0,
        epochs=1000,
        schedule=OfficialTrainerSchedule(),
        input_channels=2,
        bilateral_asymmetry_channel=True,
        physical_input_channels=1,
    )


def test_training_and_inference_preprocessing_build_elementwise_identical_bilateral_channels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from standalone_nnunet2d.data.inference_preprocessing import prepare_bilateral_asymmetry_case

    case_id = "case_synthetic"
    _write_case(tmp_path, case_id)
    monkeypatch.setattr(dataset_module, "load_fold_cases", lambda *_: (case_id,))
    training_dataset = StrokeSliceDataset(
        tmp_path,
        fold=0,
        split="val",
        case_ids=(case_id,),
        target_spacing_xy=(1.0, 1.0),
        bilateral_asymmetry_channel=True,
    )

    training_channels, _ = training_dataset.load_case(case_id)
    prepared = prepare_bilateral_asymmetry_case(
        tmp_path, case_id, target_spacing_xy=(1.0, 1.0)
    )

    assert not (tmp_path / "imagesTr" / f"{case_id}_0001.nii.gz").exists()
    assert prepared.physical_input_channels == 1
    assert prepared.effective_input_channels == 2
    np.testing.assert_allclose(prepared.model_input, training_channels, rtol=0.0, atol=1e-6)


def test_resolved_checkpoint_config_records_unambiguous_bilateral_channel_provenance() -> None:
    config = _bilateral_config()
    metadata = {"input_channels": 2, "resolved_config": config}

    assert config["bilateral_asymmetry_channel"] is True
    assert config["physical_input_channels"] == 1
    assert config["effective_model_input_channels"] == 2
    assert checkpoint_bilateral_asymmetry_channel(metadata) is True


def test_bilateral_checkpoint_rejects_ambiguous_channel_provenance() -> None:
    with pytest.raises(ValueError, match="physical_input_channels=1"):
        checkpoint_bilateral_asymmetry_channel(
            {
                "input_channels": 2,
                "resolved_config": {"bilateral_asymmetry_channel": True},
            }
        )


def test_persisted_checkpoint_restores_bilateral_channel_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(checkpoint_module, "PROJECT_OUTPUTS_DIRECTORY", tmp_path)
    config = _bilateral_config()
    model = nn.Conv2d(2, 2, 1)
    optimizer = torch.optim.SGD(model.parameters(), 0.01)
    checkpoint = save_formal_checkpoint(
        model,
        optimizer,
        tmp_path / "bilateral.pth",
        FormalTrainerState(epoch=1, global_step=1, best_validation_dice=0.0, fold=0),
        config,
    )

    metadata = torch.load(checkpoint, map_location="cpu", weights_only=False)["metadata"]

    assert metadata["resolved_config"]["bilateral_asymmetry_channel"] is True
    assert metadata["resolved_config"]["physical_input_channels"] == 1
    assert metadata["resolved_config"]["effective_model_input_channels"] == 2
    assert checkpoint_bilateral_asymmetry_channel(metadata) is True


def test_prediction_command_uses_derived_channels_without_physical_channel_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    case_id = "case_synthetic"
    raw_root = tmp_path / "raw"
    _write_case(raw_root, case_id)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    metadata = {
        "run_type": "official_alignment_pending",
        "run_state": "official_alignment_pending",
        "input_channels": 2,
        "resolved_config": _bilateral_config(),
    }
    calls: dict[str, object] = {}

    monkeypatch.setattr(predict_module, "_read_checkpoint", lambda *_: ({}, metadata))

    def fake_load_model(*_args: object, **kwargs: object) -> tuple[object, dict[str, object]]:
        calls["input_channels"] = kwargs["input_channels"]
        return object(), metadata

    def fake_predict_volume(_model: object, inputs: object, _device: torch.device, **kwargs: object) -> np.ndarray:
        channels = tuple(inputs)  # type: ignore[arg-type]
        calls["channel_count"] = len(channels)
        calls["normalise_inputs"] = kwargs.get("normalise_inputs")
        return np.zeros(channels[0].array.shape, dtype=np.uint8)

    monkeypatch.setattr(predict_module, "_load_model", fake_load_model)
    monkeypatch.setattr(predict_module, "predict_volume", fake_predict_volume)

    assert predict_module.main(
        [
            "--checkpoint", str(checkpoint), "--raw-root", str(raw_root),
            "--case-id", case_id, "--output-root", str(tmp_path / "output"),
            "--allow-pending", "--device", "cpu",
        ]
    ) == 0

    assert calls == {"input_channels": 2, "channel_count": 2, "normalise_inputs": False}
    assert not (raw_root / "imagesTr" / f"{case_id}_0001.nii.gz").exists()


def test_formal_fold_validation_uses_the_same_derived_bilateral_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    case_id = "case_synthetic"
    raw_root = tmp_path / "raw"
    _write_case(raw_root, case_id)
    calls: dict[str, object] = {}
    monkeypatch.setattr(formal_validation, "load_fold_cases", lambda *_: (case_id,))

    def fake_predict_volume(_model: object, inputs: object, _device: torch.device, **kwargs: object) -> np.ndarray:
        channels = tuple(inputs)  # type: ignore[arg-type]
        calls["channel_count"] = len(channels)
        calls["normalise_inputs"] = kwargs.get("normalise_inputs")
        return np.zeros(channels[0].array.shape, dtype=np.uint8)

    monkeypatch.setattr(formal_validation, "predict_volume", fake_predict_volume)

    report = formal_validation.validate_fold(
        object(), raw_root, fold=0, output_root=tmp_path / "output", device=torch.device("cpu"),
        bilateral_asymmetry_channel=True,
    )

    assert report["case_count"] == 1
    assert calls == {"channel_count": 2, "normalise_inputs": False}
    assert not (raw_root / "imagesTr" / f"{case_id}_0001.nii.gz").exists()


def test_standalone_inference_capture_uses_the_same_derived_bilateral_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    case_id = "case_synthetic"
    raw_root = tmp_path / "raw"
    _write_case(raw_root, case_id)
    oracle_root = tmp_path / "oracle"
    oracle_root.mkdir()
    (oracle_root / "manifest.json").write_text(
        json.dumps(
            {
                "case_id": case_id,
                "seed": 7,
                "transform_policy": {"mode": "inference"},
                "inference_context": {"fold": 0, "source_checkpoint_sha256": "a" * 64, "device": "cpu"},
            }
        ),
        encoding="utf-8",
    )
    plans = tmp_path / "plans.json"
    plans.write_text("{}", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    metadata = {
        "run_state": "official_alignment_pending", "fold": 0, "source_sha256": "a" * 64,
        "input_channels": 2, "resolved_config": _bilateral_config(),
    }
    calls: dict[str, object] = {}

    monkeypatch.setattr(standalone_capture, "_load_model", lambda *_: (object(), metadata))

    def fake_predict_volume(_model: object, inputs: object, _device: torch.device, **kwargs: object) -> np.ndarray:
        channels = tuple(inputs)  # type: ignore[arg-type]
        calls["channel_count"] = len(channels)
        calls["normalise_inputs"] = kwargs.get("normalise_inputs")
        return np.zeros(channels[0].array.shape, dtype=np.uint8)

    monkeypatch.setattr(standalone_capture, "predict_volume", fake_predict_volume)

    destination = standalone_capture.capture_standalone_inference(
        oracle_root=oracle_root, raw_root=raw_root, checkpoint=checkpoint,
        output_root=tmp_path / "output", plans_path=plans, device="cpu", case_id=case_id,
    )

    assert calls == {"channel_count": 2, "normalise_inputs": False}
    assert np.load(destination / "mask.npy").shape == _dwi().shape
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["transform_policy"]["bilateral_asymmetry_channel"] is True


def test_fold_validation_cli_restores_bilateral_mode_from_checkpoint_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    metadata = {
        "run_type": "official_alignment_pending", "run_state": "official_alignment_pending",
        "input_channels": 2, "resolved_config": _bilateral_config(),
    }
    calls: dict[str, object] = {}
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()

    monkeypatch.setattr(validate_cv, "_read_checkpoint", lambda *_: ({}, metadata))
    monkeypatch.setattr(validate_cv, "_load_model", lambda *_: (object(), metadata))

    def fake_validate_fold(*args: object, **kwargs: object) -> dict[str, object]:
        calls["model"] = args[0]
        calls["bilateral_asymmetry_channel"] = kwargs["bilateral_asymmetry_channel"]
        return {}

    monkeypatch.setattr(validate_cv, "validate_fold", fake_validate_fold)

    assert validate_cv.main(
        [
            "fold", "--checkpoint", str(checkpoint), "--raw-root", str(tmp_path / "raw"),
            "--fold", "0", "--output-root", str(tmp_path / "output"), "--allow-pending",
        ]
    ) == 0

    assert calls["model"] is not None
    assert calls["bilateral_asymmetry_channel"] is True
