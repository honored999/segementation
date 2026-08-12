from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from standalone_nnunet2d.config import load_model_config
from standalone_nnunet2d.data.nifti_io import NiftiVolume, read_nifti, write_nifti
from standalone_nnunet2d.alignment_evidence import build_alignment_evidence
from standalone_nnunet2d import predict as predict_module
from standalone_nnunet2d.engine import checkpoint as checkpoint_module
from standalone_nnunet2d.engine.checkpoint import save_checkpoint
from standalone_nnunet2d.models.plain_conv_unet import PlainConvUNet2D
from standalone_nnunet2d.predict import main


def _aligned_evidence(tmp_path: Path) -> dict[str, object]:
    components = {
        name: {"status": "passed", "diagnostics": []}
        for name in ("image", "label", "manifest", "mask")
    }
    transform_path = tmp_path / "transform.json"
    inference_path = tmp_path / "inference.json"
    transform_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "run_state": "official_alignment_pending",
                "oracle_root": "/oracle/transform",
                "standalone_root": "/standalone/transform",
                "image_atol": 0.0,
                "components": components,
                "diagnostics": [],
            }
        ),
        encoding="utf-8",
    )
    inference_path.write_text(
        json.dumps(
            {
                "parity_policy": "repeat_oracle_stability_v1",
                "oracle_roots": ["/oracle/0", "/oracle/1", "/oracle/2"],
                "oracle_repeat_count": 3,
                "stable_mask_mismatch_count": 0,
                "stable_mask_mismatch_coordinates": [],
                "unobserved_standalone_label_count": 0,
                "unobserved_standalone_label_coordinates": [],
                "status": "passed",
                "run_state": "official_alignment_pending",
                "standalone_root": "/standalone/inference",
                "image_atol": 0.0,
                "components": components,
                "diagnostics": [],
            }
        ),
        encoding="utf-8",
    )
    return build_alignment_evidence(transform_path, inference_path)


def _checkpoint_with_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    metadata: dict[str, object],
    *,
    name: str,
) -> Path:
    monkeypatch.setattr(checkpoint_module, "PROJECT_OUTPUTS_DIRECTORY", tmp_path.resolve())
    model = PlainConvUNet2D(load_model_config())
    return save_checkpoint(model, None, tmp_path / f"{name}.pt", metadata)


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


def test_prediction_command_accepts_aligned_checkpoint_and_copies_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw_root = tmp_path / "raw"
    (raw_root / "imagesTr").mkdir(parents=True)
    (raw_root / "labelsTr").mkdir()
    source = NiftiVolume(
        np.zeros((1, 512, 512), dtype=np.float32),
        spacing_xyz=(0.7, 0.8, 4.5),
        origin_xyz=(0.0, 0.0, 0.0),
    )
    write_nifti(raw_root / "imagesTr" / "case001_0000.nii.gz", source)
    evidence = _aligned_evidence(tmp_path)
    checkpoint = _checkpoint_with_metadata(
        monkeypatch,
        tmp_path,
        {
            "run_type": "official_aligned",
            "run_state": "official_aligned",
            "alignment_evidence": evidence,
        },
        name="aligned",
    )
    output_root = tmp_path / "aligned-output"

    assert main(
        [
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
        ]
    ) == 0

    manifest = json.loads((output_root / "prediction_manifest.json").read_text(encoding="utf-8"))
    assert manifest["policy"]["run_state"] == "official_aligned"
    assert manifest["policy"]["alignment_status"] == "official_aligned"
    assert manifest["policy"]["alignment_evidence"] == evidence
    assert manifest["policy"]["alignment_evidence"] is not evidence


@pytest.mark.parametrize("case", ["pending_with_evidence", "aligned_without_evidence", "tampered", "unknown"])
def test_prediction_command_rejects_invalid_checkpoint_alignment_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str
) -> None:
    evidence = _aligned_evidence(tmp_path)
    if case == "pending_with_evidence":
        metadata = {
            "run_type": "official_alignment_pending",
            "run_state": "official_alignment_pending",
            "alignment_evidence": evidence,
        }
    elif case == "aligned_without_evidence":
        metadata = {"run_type": "official_aligned", "run_state": "official_aligned"}
    elif case == "tampered":
        tampered = deepcopy(evidence)
        tampered["sources"]["transform"]["snapshot"]["status"] = "failed"
        metadata = {
            "run_type": "official_aligned",
            "run_state": "official_aligned",
            "alignment_evidence": tampered,
        }
    else:
        metadata = {"run_type": "experimental", "run_state": "experimental"}
    checkpoint = _checkpoint_with_metadata(monkeypatch, tmp_path, metadata, name=case)
    raw_root = tmp_path / "raw"
    (raw_root / "imagesTr").mkdir(parents=True)
    write_nifti(
        raw_root / "imagesTr" / "case001_0000.nii.gz",
        NiftiVolume(
            np.zeros((1, 2, 2), dtype=np.float32),
            spacing_xyz=(0.7, 0.8, 4.5),
            origin_xyz=(0.0, 0.0, 0.0),
        ),
    )
    monkeypatch.setattr(
        predict_module,
        "predict_volume",
        lambda *_args, **_kwargs: np.zeros((1, 2, 2), dtype=np.uint8),
    )

    with pytest.raises(ValueError):
        main(
            [
                "--checkpoint",
                str(checkpoint),
                "--raw-root",
                str(raw_root),
                "--case-id",
                "case001",
                "--output-root",
                str(tmp_path / f"{case}-output"),
                "--allow-pending",
            ]
        )
