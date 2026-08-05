from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from standalone_nnunet2d.config import load_model_config
from standalone_nnunet2d.data.nifti_io import NiftiVolume, read_nifti, write_nifti
from standalone_nnunet2d.engine import checkpoint as checkpoint_module
from standalone_nnunet2d.engine.checkpoint import save_checkpoint
from standalone_nnunet2d.models.plain_conv_unet import PlainConvUNet2D
from standalone_nnunet2d.predict import main


def _pending_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(checkpoint_module, "PROJECT_OUTPUTS_DIRECTORY", tmp_path.resolve())
    model = PlainConvUNet2D(load_model_config())
    return save_checkpoint(
        model,
        None,
        tmp_path / "pending.pt",
        {
            "run_type": "official_alignment_pending",
            "run_state": "official_alignment_pending",
            "fold": 0,
        },
    )


def test_prediction_command_requires_allow_pending_and_preserves_source_space_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw_root = tmp_path / "raw"
    (raw_root / "imagesTr").mkdir(parents=True)
    (raw_root / "labelsTr").mkdir()
    source = NiftiVolume(
        np.zeros((1, 512, 512), dtype=np.float32),
        spacing_xyz=(0.7, 0.8, 4.5),
        origin_xyz=(11.0, -2.0, 3.5),
        direction=(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )
    source_path = raw_root / "imagesTr" / "case001_0000.nii.gz"
    write_nifti(source_path, source)
    checkpoint = _pending_checkpoint(monkeypatch, tmp_path)
    output_root = tmp_path / "outputs"
    arguments = [
        "--checkpoint",
        str(checkpoint),
        "--raw-root",
        str(raw_root),
        "--case-id",
        "case001",
        "--output-root",
        str(output_root),
        "--device",
        "cpu",
        "--slice-batch-size",
        "2",
    ]

    with pytest.raises(ValueError, match="--allow-pending"):
        main(arguments)

    assert main([*arguments, "--allow-pending"]) == 0
    prediction_path = output_root / "predictions" / "case001.nii.gz"
    restored = read_nifti(prediction_path)
    assert restored.array.shape == source.array.shape
    assert restored.array.dtype == np.uint8
    np.testing.assert_allclose(restored.spacing_xyz, source.spacing_xyz)
    np.testing.assert_allclose(restored.origin_xyz, source.origin_xyz)
    np.testing.assert_allclose(restored.direction, source.direction)

    manifest = json.loads((output_root / "prediction_manifest.json").read_text(encoding="utf-8"))
    assert manifest["checkpoint"]["run_state"] == "official_alignment_pending"
    assert manifest["policy"]["slice_batch_size"] == 2
    assert manifest["cases"][0]["source_path"] == str(source_path)
    assert manifest["cases"][0]["nifti_validation"]["passed"] is True
