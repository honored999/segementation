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
from standalone_nnunet2d.brain_alignment.nifti_adapter import canonicalize_nifti
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


def _synthetic_volume_with_depth(depth: int) -> NiftiVolume:
    array = np.arange(depth * 3 * 5, dtype=np.float32).reshape(depth, 3, 5) + 1.0
    return NiftiVolume(array, (0.7, 0.8, 4.5), (11.0, -2.0, 3.5))


def _split_file_for_cases(tmp_path: Path, case_ids: list[str]) -> Path:
    path = tmp_path / "splits.json"
    folds = [{"train": case_ids, "val": []}] + [{"train": [], "val": []} for _ in range(4)]
    path.write_text(json.dumps(folds), encoding="utf-8")
    return path


def _install_tiny_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagnostic, "build_alignment_model", _TinyAligner)


def _valid_model_input_depth_contract() -> dict[str, object]:
    return {
        "axis": "D",
        "minimum_model_depth": 16,
        "padding_policy": "symmetric_constant_zero_D_only",
        "pad_value": 0.0,
    }


@pytest.mark.parametrize(
    ("depth", "expected_depth", "pad_before", "pad_after"),
    [(13, 16, 1, 2), (15, 16, 0, 1), (16, 16, 0, 0), (18, 18, 0, 0)],
)
def test_model_input_depth_padding_is_symmetric_deterministic_and_d_only(
    depth: int, expected_depth: int, pad_before: int, pad_after: int
) -> None:
    tensor = torch.arange(2 * 1 * depth * 3 * 5, dtype=torch.float32).reshape(2, 1, depth, 3, 5) + 1.0

    padded, metadata = diagnostic.pad_model_input_depth(tensor)

    assert padded.shape == (2, 1, expected_depth, 3, 5)
    assert metadata == {
        "axis": "D",
        "minimum_model_depth": 16,
        "padding_policy": "symmetric_constant_zero_D_only",
        "pad_value": 0.0,
        "input_depth_before": depth,
        "input_depth_after": expected_depth,
        "pad_before": pad_before,
        "pad_after": pad_after,
        "padded": depth < 16,
    }
    if depth < 16:
        assert torch.count_nonzero(padded[..., :pad_before, :, :]) == 0
        assert torch.count_nonzero(padded[..., expected_depth - pad_after :, :, :]) == 0
        torch.testing.assert_close(
            padded[..., pad_before : pad_before + depth, :, :], tensor
        )
    else:
        assert padded is tensor


def test_model_input_depth_unpadding_restores_canonical_array_before_inverse_mapping() -> None:
    source = NiftiVolume(
        np.arange(13 * 3 * 5, dtype=np.float32).reshape(13, 3, 5),
        spacing_xyz=(0.7, 0.8, 4.5),
        origin_xyz=(11.0, -2.0, 3.5),
        direction=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0),
    )
    canonical = canonicalize_nifti(source)
    model_tensor, metadata = diagnostic.pad_model_input_depth(torch.from_numpy(canonical.array))

    unpadded = diagnostic.unpad_model_input_depth(model_tensor, metadata)

    assert tuple(unpadded.shape) == canonical.array.shape
    np.testing.assert_array_equal(unpadded.numpy(), canonical.array)
    np.testing.assert_array_equal(canonical.restore_array(unpadded.numpy()), source.array)


@pytest.mark.parametrize(
    ("input_depth_after", "pad_before", "pad_after"),
    [(17, 2, 2), (16, 0, 3)],
)
def test_model_input_depth_unpadding_rejects_malformed_deterministic_metadata(
    input_depth_after: int, pad_before: int, pad_after: int
) -> None:
    metadata = {
        **_valid_model_input_depth_contract(),
        "input_depth_before": 13,
        "input_depth_after": input_depth_after,
        "pad_before": pad_before,
        "pad_after": pad_after,
        "padded": True,
    }

    with pytest.raises(ValueError, match="inconsistent"):
        diagnostic.unpad_model_input_depth(
            torch.zeros(1, 1, input_depth_after, 3, 5), metadata
        )


def test_training_metadata_records_padded_case_ids_and_depth_details(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_tiny_model(monkeypatch)
    case_ids = ["case018", "case037"]
    dataset_dir = tmp_path / "dataset"
    for case_id in case_ids:
        image_path = dataset_dir / "imagesTr" / f"{case_id}_0000.nii.gz"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"synthetic image placeholder")
    splits_file = _split_file_for_cases(tmp_path, case_ids)
    volumes = {
        "case018": _synthetic_volume_with_depth(13),
        "case037": _synthetic_volume_with_depth(15),
    }
    monkeypatch.setattr(diagnostic, "read_nifti", lambda path: volumes[path.name.split("_")[0]])
    output_dir = tmp_path / "diagnostic-output"

    assert train_module.main(
        [
            "--dataset-dir", str(dataset_dir),
            "--splits-file", str(splits_file),
            "--epochs", "1",
            "--lr", "0.01",
            "--device", "cpu",
            "--output-dir", str(output_dir),
        ]
    ) == 0

    payload = diagnostic.validate_checkpoint_payload(
        torch.load(output_dir / "checkpoint_latest.pth", map_location="cpu", weights_only=False)
    )
    assert payload["metadata"]["model_input_depth_padding"] == {
        "axis": "D",
        "minimum_model_depth": 16,
        "padding_policy": "symmetric_constant_zero_D_only",
        "pad_value": 0.0,
        "padded_case_ids": case_ids,
        "padded_cases": [
            {
                "case_id": "case018",
                "input_depth_before": 13,
                "input_depth_after": 16,
                "pad_before": 1,
                "pad_after": 2,
            },
            {
                "case_id": "case037",
                "input_depth_before": 15,
                "input_depth_after": 16,
                "pad_before": 0,
                "pad_after": 1,
            },
        ],
    }


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
    assert latest["model_contract"]["canonical_contract"] == "acquisition_preserving_lr"
    assert latest["model_contract"]["model_axis_semantics"] == {
        "D": "acquisition_through_plane",
        "H": "acquisition_in_plane_non_lr",
        "W": "anatomical_lr",
    }
    assert latest["model_contract"]["transform_semantics"]["tx"] == "model_space_lr_translation"
    assert latest["model_contract"]["transform_semantics"]["rz"] == "acquisition_model_in_plane_rotation"
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


@pytest.mark.parametrize("invalid_field", [None, "axis", "minimum_model_depth", "padding_policy", "pad_value"])
def test_checkpoint_validator_rejects_missing_or_invalid_model_input_depth_contract_before_model_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, invalid_field: str | None
) -> None:
    path = tmp_path / "bad-padding-contract.pth"
    diagnostic.save_diagnostic_checkpoint(
        path,
        _TinyAligner(),
        optimizer=None,
        epoch=0,
        history=[],
        best_loss=0.0,
        metadata={"case_ids": [], "checkpoint_role": "best"},
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    contract = payload["model_contract"]
    contract["model_input_depth_padding"] = _valid_model_input_depth_contract()
    if invalid_field is None:
        contract["model_input_depth_padding"].pop("axis")
    elif invalid_field == "axis":
        contract["model_input_depth_padding"][invalid_field] = "H"
    elif invalid_field == "minimum_model_depth":
        contract["model_input_depth_padding"][invalid_field] = 15
    elif invalid_field == "padding_policy":
        contract["model_input_depth_padding"][invalid_field] = "zero_pad"
    else:
        contract["model_input_depth_padding"][invalid_field] = 1.0
    torch.save(payload, path)
    monkeypatch.setattr(
        diagnostic, "build_alignment_model", lambda: (_ for _ in ()).throw(AssertionError("model loaded"))
    )

    with pytest.raises(ValueError, match="model_input_depth"):
        diagnostic.load_diagnostic_checkpoint(path, device=torch.device("cpu"))


def test_checkpoint_validator_rejects_inconsistent_padded_case_records_before_model_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "bad-padding-record.pth"
    diagnostic.save_diagnostic_checkpoint(
        path,
        _TinyAligner(),
        optimizer=None,
        epoch=0,
        history=[],
        best_loss=0.0,
        metadata={"case_ids": ["case018"], "checkpoint_role": "best"},
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["model_contract"]["model_input_depth_padding"] = _valid_model_input_depth_contract()
    payload["metadata"]["model_input_depth_padding"] = {
        **_valid_model_input_depth_contract(),
        "padded_case_ids": ["case018"],
        "padded_cases": [
            {
                "case_id": "case018",
                "input_depth_before": 13,
                "input_depth_after": 17,
                "pad_before": 2,
                "pad_after": 2,
            }
        ],
    }
    torch.save(payload, path)
    monkeypatch.setattr(
        diagnostic, "build_alignment_model", lambda: (_ for _ in ()).throw(AssertionError("model loaded"))
    )

    with pytest.raises(ValueError, match="model_input_depth"):
        diagnostic.load_diagnostic_checkpoint(path, device=torch.device("cpu"))


def test_synthetic_checkpoint_gets_stable_model_input_depth_contract(tmp_path: Path) -> None:
    path = tmp_path / "minimal-padding-contract.pth"
    diagnostic.save_diagnostic_checkpoint(
        path,
        _TinyAligner(),
        optimizer=None,
        epoch=0,
        history=[],
        best_loss=0.0,
        metadata={"case_ids": [], "checkpoint_role": "best"},
    )

    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert payload["model_contract"]["model_input_depth_padding"] == _valid_model_input_depth_contract()
    assert payload["metadata"]["model_input_depth_padding"] == {
        **_valid_model_input_depth_contract(),
        "padded_case_ids": [],
        "padded_cases": [],
    }


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
    assert summary["orientation_canonicalization"]["applied_permutation"] == [0, 1, 2]
    assert summary["orientation_canonicalization"]["canonical_contract"] == "acquisition_preserving_lr"
    assert summary["transform_semantics"] == {
        "tx": "model_space_lr_translation",
        "rz": "acquisition_model_in_plane_rotation",
        "physical_3d_rigid_registration": False,
    }
    assert [item["percent"] for item in summary["slice_indices"]] == [25, 50, 75]
    assert summary["checkpoint"] == str(checkpoint.resolve())


def test_qc_cli_uses_model_padding_and_unpads_outputs_before_display(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_tiny_model(monkeypatch)
    image_path = tmp_path / "case018_0000.nii.gz"
    image_path.write_bytes(b"immutable synthetic image")
    volume = _synthetic_volume_with_depth(13)
    checkpoint = tmp_path / "checkpoint.pth"
    diagnostic.save_diagnostic_checkpoint(
        checkpoint,
        _TinyAligner(),
        optimizer=None,
        epoch=2,
        history=[{"epoch": 1, "total_loss": 1.0}],
        best_loss=1.0,
        metadata={"case_ids": ["case018"], "checkpoint_role": "best"},
    )
    monkeypatch.setattr(qc_module, "read_nifti", lambda _: volume)
    panels: list[list[np.ndarray]] = []
    monkeypatch.setattr(qc_module, "_save_panel", lambda _path, arrays, _titles: panels.append(arrays))
    output_dir = tmp_path / "qc-output"

    assert qc_module.main(
        [
            "--image", str(image_path),
            "--checkpoint", str(checkpoint),
            "--output-dir", str(output_dir),
            "--device", "cpu",
        ]
    ) == 0

    assert len(panels) == 3
    assert all([array.shape for array in panel] == [(3, 5)] * 4 for panel in panels)
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["model_input_depth_padding"] == {
        "axis": "D",
        "minimum_model_depth": 16,
        "padding_policy": "symmetric_constant_zero_D_only",
        "pad_value": 0.0,
        "input_depth_before": 13,
        "input_depth_after": 16,
        "pad_before": 1,
        "pad_after": 2,
        "padded": True,
    }
    assert [item["index"] for item in summary["slice_indices"]] == [3, 6, 9]


def test_qc_rejects_checkpoint_padding_contract_mismatch_before_model_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_tiny_model(monkeypatch)
    image_path = tmp_path / "case018_0000.nii.gz"
    image_path.write_bytes(b"immutable synthetic image")
    checkpoint = tmp_path / "bad-padding-contract.pth"
    diagnostic.save_diagnostic_checkpoint(
        checkpoint,
        _TinyAligner(),
        optimizer=None,
        epoch=0,
        history=[],
        best_loss=0.0,
        metadata={"case_ids": ["case018"], "checkpoint_role": "best"},
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["model_contract"]["model_input_depth_padding"]["minimum_model_depth"] = 15
    torch.save(payload, checkpoint)
    monkeypatch.setattr(qc_module, "read_nifti", lambda _: _synthetic_volume_with_depth(13))
    monkeypatch.setattr(
        diagnostic, "build_alignment_model", lambda: (_ for _ in ()).throw(AssertionError("model loaded"))
    )

    with pytest.raises(ValueError, match="model_input_depth"):
        qc_module.main(
            [
                "--image", str(image_path),
                "--checkpoint", str(checkpoint),
                "--output-dir", str(tmp_path / "qc-output"),
                "--device", "cpu",
            ]
        )


def test_qc_cli_requires_explicit_core_arguments_and_rejects_bad_checkpoint() -> None:
    with pytest.raises(SystemExit) as error:
        qc_module.main([])
    assert error.value.code == 2
