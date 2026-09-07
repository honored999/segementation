from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from standalone_nnunet2d.brain_alignment.adn_transform import AlignmentLosses
from standalone_nnunet2d.data.nifti_io import NiftiVolume
from standalone_nnunet2d.tools import adn_loss_landscape as landscape


def test_grid_contains_35_points_identity_and_correct_pixel_conversion() -> None:
    points = landscape.grid_points(width=80)

    assert len(points) == 35
    assert any(point.rz_degree == 0 and point.tx_pixels == 0 for point in points)
    assert next(point.tx_normalized for point in points if point.rz_degree == 0 and point.tx_pixels == 16) == pytest.approx(0.4)


def test_summary_selects_minimum_and_reports_boundary() -> None:
    rows = [
        {"case_id": "case001", "rz_degree": 0, "tx_pixels": 0, "tx_normalized": 0.0,
         "flip_loss": 2.0, "reconstruction_loss": 3.0, "total_loss": 5.0},
        {"case_id": "case001", "rz_degree": -15, "tx_pixels": 16, "tx_normalized": 0.5,
         "flip_loss": 1.0, "reconstruction_loss": 1.0, "total_loss": 2.0},
    ]

    summary = landscape.summarize_landscape("case001", rows)

    assert summary["identity"] == {"flip_loss": 2.0, "reconstruction_loss": 3.0, "total_loss": 5.0}
    assert summary["global_minimum"]["rz_degree"] == -15
    assert summary["global_minimum"]["tx_pixels"] == 16
    assert summary["global_minimum"]["total_loss"] == 2.0
    assert summary["best_to_identity_ratio"] == pytest.approx(0.4)
    assert summary["improvement_percent_relative_to_identity"] == pytest.approx(60.0)
    assert summary["minimum_on_grid_boundary"] is True


def test_cli_reuses_preprocessing_padding_and_never_loads_model_or_labels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = tmp_path / "dataset"
    image = dataset / "imagesTr" / "case001_0000.nii.gz"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"synthetic placeholder")
    calls: list[str] = []
    source = NiftiVolume(np.arange(13 * 4 * 80, dtype=np.float32).reshape(13, 4, 80), (1, 1, 1), (0, 0, 0))

    def read(path: Path) -> NiftiVolume:
        calls.append(f"read:{path.name}")
        assert "labels" not in str(path).lower()
        return source

    def canonicalize(volume: NiftiVolume) -> SimpleNamespace:
        calls.append("canonicalize")
        assert volume is source
        return SimpleNamespace(array=volume.array)

    def normalize(array: np.ndarray) -> np.ndarray:
        calls.append("normalize")
        return np.asarray(array, dtype=np.float32)

    def pad(tensor: torch.Tensor):
        calls.append("pad")
        assert tensor.shape == (1, 1, 13, 4, 80)
        return torch.nn.functional.pad(tensor, (0, 0, 0, 0, 1, 2)), {"padded": True}

    monkeypatch.setattr(landscape, "read_nifti", read)
    monkeypatch.setattr(landscape, "canonicalize_nifti", canonicalize)
    monkeypatch.setattr(landscape, "normalize_volume", normalize)
    monkeypatch.setattr(landscape, "pad_model_input_depth", pad)
    monkeypatch.setattr(landscape, "warp_volume", lambda volume, matrix: volume)

    def losses(_original: torch.Tensor, _aligned: torch.Tensor, inverse: torch.Tensor) -> AlignmentLosses:
        value = inverse[:, 0, 3].abs().mean()
        return AlignmentLosses(value, value + 1, value * 2 + 1)

    monkeypatch.setattr(landscape, "alignment_losses", losses)
    output = tmp_path / "output"

    assert landscape.main(["--dataset-dir", str(dataset), "--cases", "case001", "--device", "cpu", "--output-dir", str(output)]) == 0
    assert calls == ["read:case001_0000.nii.gz", "canonicalize", "normalize", "pad"]
    case_output = output / "case001"
    with (case_output / "landscape.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 35
    assert float(next(row["tx_normalized"] for row in rows if row["rz_degree"] == "0" and row["tx_pixels"] == "16")) == pytest.approx(0.4)
    assert (case_output / "total_loss_heatmap.png").is_file()
    summary = json.loads((case_output / "summary.json").read_text(encoding="utf-8"))
    assert summary["case_id"] == "case001"
    assert summary["model_input_depth_padding"] == {"padded": True}
    assert not hasattr(landscape, "load_diagnostic_checkpoint")
    assert not hasattr(landscape, "build_alignment_model")
