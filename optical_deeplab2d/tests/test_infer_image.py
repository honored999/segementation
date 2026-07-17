from __future__ import annotations

import sys

import numpy as np
import pytest
import torch
from PIL import Image

from optical_deeplab2d import infer_image
from optical_deeplab2d.models.electronic_deepseg_decoder import ElectronicDeepSegDecoder


def _write_checkpoint(path, model_type: str, state_dict: dict) -> None:
    torch.save(
        {
            "model_type": model_type,
            "encoder_name": "mobilenet_v2",
            "model_state_dict": state_dict,
            "threshold": 0.5,
            "normalization": "minmax_uint8",
            "fold": 0,
            "seed": 7,
        },
        path,
    )


def test_infer_image_loads_electronic_deepseg_checkpoint(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "deepseg.pt"
    model = ElectronicDeepSegDecoder(encoder_weights=None)
    _write_checkpoint(checkpoint, "electronic_deepseg_decoder", model.state_dict())
    image_path = tmp_path / "image.png"
    Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(image_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "infer_image.py",
            "--image",
            str(image_path),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(tmp_path / "outputs"),
        ],
    )

    infer_image.main()

    assert (tmp_path / "outputs" / "prediction.npy").is_file()


def test_infer_image_rejects_unknown_model_type(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "unknown.pt"
    _write_checkpoint(checkpoint, "unsupported_model", {})
    image_path = tmp_path / "image.png"
    Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(image_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "infer_image.py",
            "--image",
            str(image_path),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(tmp_path / "outputs"),
        ],
    )

    with pytest.raises(ValueError, match="Unknown model type: unsupported_model"):
        infer_image.main()


def test_infer_image_rejects_checkpoint_without_model_type(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "missing-model-type.pt"
    _write_checkpoint(checkpoint, "electronic_baseline", {})
    checkpoint_data = torch.load(checkpoint, weights_only=False)
    checkpoint_data.pop("model_type")
    torch.save(checkpoint_data, checkpoint)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "infer_image.py",
            "--image",
            str(tmp_path / "not-needed.png"),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(tmp_path / "outputs"),
        ],
    )

    with pytest.raises(ValueError, match="model_type"):
        infer_image.main()
