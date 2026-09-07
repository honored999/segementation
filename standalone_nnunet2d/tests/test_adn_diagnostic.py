"""Synthetic tests for the isolated ADN diagnostic workflow."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from standalone_nnunet2d.brain_alignment import diagnostic
from standalone_nnunet2d.brain_alignment.adn_transform import AlignmentResult
from standalone_nnunet2d.data.nifti_io import NiftiVolume
from standalone_nnunet2d.tools import adn_alignment_qc as qc_module
from standalone_nnunet2d.tools import train_adn_alignment as train_module


class _TinyAligner(nn.Module):
    """Small differentiable stand-in used only to keep CLI fixtures tiny."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.9))
        self.in_channels = 1

    def forward(self, volume: torch.Tensor) -> AlignmentResult:
        batch = volume.shape[0]
        identity = torch.eye(4, dtype=volume.dtype, device=volume.device).expand(batch, -1, -1).clone()
        params = torch.zeros(batch, 6, dtype=volume.dtype, device=volume.device)
        aligned = volume * self.scale
        return AlignmentResult(aligned, params, params, identity, identity)


def _split_file(tmp_path: Path, case_id: str = "case001") -> Path:
    path = tmp_path / "splits.json"
    folds = [{"train": [case_id], "val": []}] + [{"train": [], "val": []} for _ in range(4)]
    path.write_text(json.dumps(folds), encoding="utf-8")
    return path


def _synthetic_volume() -> NiftiVolume:
    array = np.linspace(-2.0, 2.0, 16 * 16 * 16, dtype=np.float32).reshape(16, 16, 16)
    return NiftiVolume(array, (0.7, 0.8, 4.5), (11.0, -2.0, 3.5))


def _install_tiny_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagnostic, "build_alignment_model", _TinyAligner)


def test_train_cli_requires_explicit_core_arguments() -> None:
    with pytest.raises(SystemExit) as error:
        train_module.main([])

    assert error.value.code == 2


def test_training_uses_only_fold0_train_images_and_writes_latest_best_history(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_tiny_model(monkeypatch)
    dataset_dir = tmp_path / "dataset"
    image_path = dataset_dir / "imagesTr" / "case001_0000.nii.gz"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"synthetic image placeholder")
    splits_file = _split_file(tmp_path)
    reads: list[Path] = []

    def fake_read(path: Path) -> NiftiVolume:
        reads.append(path)
        return _synthetic_volume()

    monkeypatch.setattr(diagnostic, "read_nifti", fake_read)
    output_dir = tmp_path / "diagnostic-output"

    assert train_module.main(
        [
            "--dataset-dir", str(dataset_dir),
            "--splits-file", str(splits_file),
            "--epochs", "2",
            "--lr", "0.01",
            "--device", "cpu",
            "--output-dir", str(output_dir),
        ]
    ) == 0

    assert reads == [image_path.resolve(), image_path.resolve()]
    assert all("labelsTr" not in str(path) for path in reads)
    assert (output_dir / "checkpoint_latest.pth").is_file()
    assert (output_dir / "checkpoint_best.pth").is_file()
    history = json.loads((output_dir / "history.json").read_text(encoding="utf-8"))
    assert len(history) == 2
    assert set(history[0]) >= {"epoch", "flip_loss", "reconstruction_loss", "total_loss"}
    assert len((output_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()) == 2

    latest = diagnostic.validate_checkpoint_payload(
        torch.load(output_dir / "checkpoint_latest.pth", map_location="cpu", weights_only=False)
    )
    best = diagnostic.validate_checkpoint_payload(
        torch.load(output_dir / "checkpoint_best.pth", map_location="cpu", weights_only=False)
    )
    assert latest["metadata"]["case_ids"] == ["case001"]
    assert latest["metadata"]["batch_size"] == 1
    assert latest["metadata"]["labels_accessed"] is False
    assert latest["history"]
    assert best["metadata"]["checkpoint_role"] == "best"


def test_checkpoint_loader_rejects_bad_contract_before_model_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "bad.pth"
    torch.save({"format_version": 1, "kind": "wrong"}, path)
    monkeypatch.setattr(
        diagnostic, "build_alignment_model", lambda: (_ for _ in ()).throw(AssertionError("model loaded"))
    )

    with pytest.raises(ValueError, match="checkpoint"):
        diagnostic.load_diagnostic_checkpoint(path, device=torch.device("cpu"))


def test_checkpoint_validator_rejects_missing_optimizer_state_dict(tmp_path: Path) -> None:
    model = diagnostic.build_alignment_model()
    path = tmp_path / "missing-optimizer-state.pth"
    diagnostic.save_diagnostic_checkpoint(
        path,
        model,
        optimizer=None,
        epoch=0,
        history=[],
        best_loss=0.0,
        metadata={"case_ids": [], "checkpoint_role": "best"},
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload.pop("optimizer_state_dict")

    with pytest.raises(ValueError, match="optimizer_state_dict"):
        diagnostic.validate_checkpoint_payload(payload)


def test_real_adn_state_dict_roundtrip_uses_one_synthetic_checkpoint(
    tmp_path: Path,
) -> None:
    model = diagnostic.build_alignment_model()
    path = tmp_path / "real-adn.pth"
    diagnostic.save_diagnostic_checkpoint(
        path,
        model,
        optimizer=None,
        epoch=0,
        history=[],
        best_loss=0.0,
        metadata={"case_ids": [], "checkpoint_role": "best"},
    )

    restored = diagnostic.load_diagnostic_checkpoint(path, device=torch.device("cpu"))

    assert restored.epoch == 0
    assert restored.model.__class__.__name__ == "ADNTransformAligner"
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.model.state_dict()[name])


def test_training_rejects_nonempty_or_overlapping_output(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "imagesTr").mkdir(parents=True)
    output_dir = dataset_dir / "outputs"

    with pytest.raises(ValueError, match="dataset"):
        diagnostic.ensure_diagnostic_output_dir(output_dir, dataset_dir=dataset_dir)

    separate = tmp_path / "separate"
    separate.mkdir()
    (separate / "sentinel").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        diagnostic.ensure_diagnostic_output_dir(separate, dataset_dir=dataset_dir)


def test_qc_cli_writes_three_four_panel_pngs_and_json_without_mutating_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_tiny_model(monkeypatch)
    image_path = tmp_path / "case005_0000.nii.gz"
    image_path.write_bytes(b"immutable synthetic image")
    image_bytes_before = image_path.read_bytes()
    volume = _synthetic_volume()
    checkpoint = tmp_path / "checkpoint.pth"
    diagnostic.save_diagnostic_checkpoint(
        checkpoint,
        _TinyAligner(),
        optimizer=None,
        epoch=2,
        history=[{"epoch": 1, "total_loss": 1.0}],
        best_loss=1.0,
        metadata={"case_ids": ["case005"], "checkpoint_role": "best"},
    )
    monkeypatch.setattr(qc_module, "read_nifti", lambda _: volume)
    output_dir = tmp_path / "qc-output"

    assert qc_module.main(
        [
            "--image", str(image_path),
            "--checkpoint", str(checkpoint),
            "--output-dir", str(output_dir),
            "--device", "cpu",
        ]
    ) == 0

    assert image_path.read_bytes() == image_bytes_before
    pngs = sorted(output_dir.glob("*.png"))
    assert [path.name for path in pngs] == ["slice_25.png", "slice_50.png", "slice_75.png"]
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["case_id"] == "case005"
    assert len(summary["raw_params"]) == 6
    assert len(summary["scaled_params"]) == 6
    assert "rz_degree" in summary and "tx" in summary
    assert "flip_loss_before" in summary and "flip_loss_after" in summary
    assert summary["raw_geometry"]["spacing_xyz"] == [0.7, 0.8, 4.5]
    assert summary["orientation_canonicalization"]["permutation"] == [0, 1, 2]
    assert [item["percent"] for item in summary["slice_indices"]] == [25, 50, 75]
    assert summary["checkpoint"] == str(checkpoint.resolve())


def test_qc_cli_requires_explicit_core_arguments_and_rejects_bad_checkpoint() -> None:
    with pytest.raises(SystemExit) as error:
        qc_module.main([])
    assert error.value.code == 2
