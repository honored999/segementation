from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch

from standalone_nnunet2d.data.nifti_io import NiftiVolume, write_nifti
from standalone_nnunet2d import standalone_capture
from standalone_nnunet2d.standalone_capture import capture_standalone_transform


def _write_fixture(tmp_path: Path, *, oracle_mode: str = "transform") -> dict[str, Path | str]:
    case_id = "case001"
    raw_root = tmp_path / "raw"
    preprocessed_root = tmp_path / "preprocessed"
    oracle_root = tmp_path / "oracle"
    output_root = tmp_path / "standalone"
    plans_path = tmp_path / "nnUNetPlans.json"
    plans_path.write_text(
        json.dumps(
            {"configurations": {"2d": {"patch_size": [2, 3], "use_mask_for_norm": [True]}}}
        ),
        encoding="utf-8",
    )

    image = np.arange(32, dtype=np.float32).reshape(2, 4, 4)
    label = np.zeros((2, 4, 4), dtype=np.int16)
    label[1, 1:3, 1:3] = 1
    preprocessed_root.mkdir(parents=True)
    np.savez(
        preprocessed_root / f"{case_id}.npz",
        data=image[None],
        seg=label[None],
    )

    raw_volume = NiftiVolume(
        array=image,
        spacing_xyz=(0.5, 0.75, 3.0),
        origin_xyz=(1.0, 2.0, 3.0),
    )
    write_nifti(raw_root / "imagesTr" / f"{case_id}_0000.nii.gz", raw_volume)
    write_nifti(
        raw_root / "labelsTr" / f"{case_id}.nii.gz",
        NiftiVolume(
            array=label,
            spacing_xyz=raw_volume.spacing_xyz,
            origin_xyz=raw_volume.origin_xyz,
        ),
    )

    oracle_root.mkdir(parents=True)
    # Deliberately unrelated values make copying oracle arrays observable.
    np.save(oracle_root / "image.npy", np.full((4, 4), -999.0, dtype=np.float32))
    np.save(oracle_root / "label.npy", np.full((4, 4), 9, dtype=np.int16))
    np.save(oracle_root / "mask.npy", np.full((4, 4), 7, dtype=np.uint8))
    manifest = {
        "artifact_version": 1,
        "nnunetv2_version": "2.5.1",
        "plans_hash": "oracle-must-not-be-copied",
        "seed": 7,
        "case_id": case_id,
        "transform_policy": {
            "mode": oracle_mode,
            "z_index": 1,
            "source": "nnunetv2_training_transform",
        },
        "sampling_policy": {"seed": 7, "fold": 0, "implementation": "oracle"},
        "arrays": {
            "image": {"file": "image.npy", "shape": [4, 4], "dtype": "float32"},
            "label": {"file": "label.npy", "shape": [4, 4], "dtype": "int16"},
            "mask": {"file": "mask.npy", "shape": [4, 4], "dtype": "uint8"},
        },
        "nifti_metadata": {"space": "raw"},
        "run_state": "official_alignment_pending",
    }
    if oracle_mode == "inference":
        manifest["inference_context"] = {
            "fold": 0,
            "source_checkpoint_sha256": "a" * 64,
            "device": "cpu",
        }
    (oracle_root / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return {
        "case_id": case_id,
        "raw_root": raw_root,
        "preprocessed_root": preprocessed_root,
        "oracle_root": oracle_root,
        "output_root": output_root,
        "plans_path": plans_path,
    }


def test_capture_standalone_transform_writes_source_derived_artifact(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)

    artifact_root = capture_standalone_transform(**fixture)

    assert artifact_root == fixture["output_root"] / "transform" / fixture["case_id"]
    image = np.load(artifact_root / "image.npy")
    label = np.load(artifact_root / "label.npy")
    mask = np.load(artifact_root / "mask.npy")
    np.testing.assert_array_equal(mask, (label > 0).astype(np.uint8))
    assert image.shape == label.shape == mask.shape == (2, 3)
    assert not np.array_equal(image, np.load(fixture["oracle_root"] / "image.npy"))
    assert not np.array_equal(label, np.load(fixture["oracle_root"] / "label.npy"))

    manifest = json.loads((artifact_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["implementation"] == "standalone"
    assert manifest["capture_mode"] == "transform"
    assert manifest["case_id"] == fixture["case_id"]
    assert manifest["seed"] == 7
    assert manifest["transform_policy"]["mode"] == "transform"
    assert manifest["transform_policy"]["implementation"] == "standalone"
    assert manifest["transform_policy"]["z_index"] == 1
    assert manifest["plans_hash"] == hashlib.sha256(
        fixture["plans_path"].read_bytes()
    ).hexdigest()
    assert manifest["nifti_metadata"]["space"] == "raw"
    assert manifest["nifti_metadata"]["spacing_xyz"] == [0.5, 0.75, 3.0]


def test_capture_standalone_transform_uses_plan_config_and_oracle_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _write_fixture(tmp_path)
    calls: list[tuple[tuple[int, int], tuple[bool, ...], int]] = []

    def fake_adapter(image, label, *, patch_size, use_mask_for_norm, seed):
        calls.append((patch_size, tuple(use_mask_for_norm), seed))
        return image[:2, :3], label[:2, :3]

    monkeypatch.setattr(standalone_capture, "apply_official_2d_batchgeneratorsv2", fake_adapter, raising=False)

    capture_standalone_transform(**fixture)

    assert calls == [((2, 3), (True,), 7)]


def test_capture_standalone_transform_rejects_non_transform_oracle(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path, oracle_mode="sample")

    with pytest.raises(ValueError, match="transform"):
        capture_standalone_transform(**fixture)


def test_read_preprocessed_case_supports_b2nd_pair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    case_id = "case001"
    data_path = tmp_path / f"{case_id}.b2nd"
    seg_path = tmp_path / f"{case_id}_seg.b2nd"
    data_path.write_bytes(b"data")
    seg_path.write_bytes(b"seg")
    data = np.arange(8, dtype=np.float64).reshape(1, 2, 2, 2)
    seg = np.zeros((1, 2, 2, 2), dtype=np.int32)
    seg[0, 1, 1, 1] = 1
    calls: list[tuple[str, str]] = []

    def fake_open(*, urlpath: str, mode: str) -> np.ndarray:
        calls.append((urlpath, mode))
        return data if urlpath == str(data_path) else seg

    monkeypatch.setitem(sys.modules, "blosc2", SimpleNamespace(open=fake_open))

    image, label = standalone_capture._read_preprocessed_case(tmp_path, case_id)

    np.testing.assert_array_equal(image, data[0].astype(np.float32))
    np.testing.assert_array_equal(label, seg[0].astype(np.int16))
    assert calls == [(str(data_path), "r"), (str(seg_path), "r")]


def test_capture_standalone_inference_writes_source_derived_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = _write_fixture(tmp_path, oracle_mode="inference")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint handled by mocked project loader")
    loaded_model = object()
    expected_mask = np.zeros((2, 4, 4), dtype=np.uint8)
    expected_mask[0, 0, 0] = 1
    loader_calls: list[tuple[Path, torch.device]] = []
    predictor_calls: list[tuple[object, NiftiVolume, torch.device, int]] = []

    def fake_load_model(path: Path, device: torch.device) -> tuple[object, dict[str, object]]:
        loader_calls.append((path, device))
        return loaded_model, {
            "run_state": "official_alignment_pending",
            "fold": 0,
            "source_sha256": "a" * 64,
        }

    def fake_predict_volume(
        model: object,
        image: NiftiVolume,
        device: torch.device,
        *,
        slice_batch_size: int,
    ) -> np.ndarray:
        predictor_calls.append((model, image, device, slice_batch_size))
        return expected_mask

    monkeypatch.setattr(standalone_capture, "_load_model", fake_load_model, raising=False)
    monkeypatch.setattr(standalone_capture, "predict_volume", fake_predict_volume, raising=False)

    artifact_root = standalone_capture.capture_standalone_inference(
        oracle_root=fixture["oracle_root"],
        raw_root=fixture["raw_root"],
        checkpoint=checkpoint,
        output_root=fixture["output_root"],
        plans_path=fixture["plans_path"],
        device="cpu",
        case_id=fixture["case_id"],
        slice_batch_size=2,
    )

    assert artifact_root == fixture["output_root"] / "inference" / fixture["case_id"]
    image = np.load(artifact_root / "image.npy")
    label = np.load(artifact_root / "label.npy")
    mask = np.load(artifact_root / "mask.npy")
    raw_image = np.load(fixture["preprocessed_root"] / f"{fixture['case_id']}.npz")["data"][0]
    raw_label = np.load(fixture["preprocessed_root"] / f"{fixture['case_id']}.npz")["seg"][0]
    np.testing.assert_array_equal(image, raw_image.astype(np.float32))
    np.testing.assert_array_equal(label, raw_label.astype(np.int16))
    np.testing.assert_array_equal(mask, expected_mask)
    assert image.dtype == np.float32
    assert label.dtype == np.int16
    assert mask.dtype == np.uint8
    assert not np.array_equal(mask, np.load(fixture["oracle_root"] / "mask.npy"))
    assert loader_calls == [(checkpoint, torch.device("cpu"))]
    assert predictor_calls[0][0] is loaded_model
    assert predictor_calls[0][2:] == (torch.device("cpu"), 2)

    manifest = json.loads((artifact_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["implementation"] == "standalone"
    assert manifest["capture_mode"] == "inference"
    assert manifest["transform_policy"]["mode"] == "inference"
    assert manifest["transform_policy"]["implementation"] == "standalone"
    assert manifest["transform_policy"]["slice_batch_size"] == 2
    assert manifest["case_id"] == fixture["case_id"]
    assert manifest["seed"] == 7
    assert manifest["plans_hash"] == hashlib.sha256(
        fixture["plans_path"].read_bytes()
    ).hexdigest()
    assert manifest["nifti_metadata"]["space"] == "raw"
    assert manifest["nifti_metadata"]["spacing_xyz"] == [0.5, 0.75, 3.0]
    assert manifest["inference_context"] == {
        "fold": 0,
        "source_checkpoint_sha256": "a" * 64,
        "device": "cpu",
    }
    assert manifest["run_state"] == "official_alignment_pending"


@pytest.mark.parametrize(
    "checkpoint_metadata",
    [
        {"run_state": "official_alignment_pending", "source_sha256": "a" * 64},
        {
            "run_state": "official_alignment_pending",
            "fold": -1,
            "source_sha256": "a" * 64,
        },
        {
            "run_state": "official_alignment_pending",
            "fold": 0,
            "source_sha256": "not-a-sha256",
        },
    ],
)
def test_capture_standalone_inference_rejects_missing_or_invalid_checkpoint_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    checkpoint_metadata: dict[str, object],
) -> None:
    fixture = _write_fixture(tmp_path, oracle_mode="inference")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint handled by mocked project loader")

    monkeypatch.setattr(
        standalone_capture,
        "_load_model",
        lambda path, device: (object(), checkpoint_metadata),
        raising=False,
    )
    monkeypatch.setattr(
        standalone_capture,
        "predict_volume",
        lambda model, image, device, *, slice_batch_size: np.zeros(
            image.array.shape, dtype=np.uint8
        ),
        raising=False,
    )

    with pytest.raises(ValueError, match="fold|source_sha256|provenance"):
        standalone_capture.capture_standalone_inference(
            oracle_root=fixture["oracle_root"],
            raw_root=fixture["raw_root"],
            checkpoint=checkpoint,
            output_root=fixture["output_root"],
            plans_path=fixture["plans_path"],
            device="cpu",
        )


def test_capture_standalone_inference_rejects_non_inference_oracle(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path, oracle_mode="transform")

    with pytest.raises(ValueError, match="inference"):
        standalone_capture.capture_standalone_inference(
            oracle_root=fixture["oracle_root"],
            raw_root=fixture["raw_root"],
            checkpoint=tmp_path / "checkpoint.pt",
            output_root=fixture["output_root"],
            plans_path=fixture["plans_path"],
            device="cpu",
        )


def test_capture_cli_defaults_to_transform_and_prints_pending_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []
    destination = tmp_path / "transform" / "case001"

    def fake_transform(**kwargs: object) -> Path:
        calls.append(kwargs)
        return destination

    monkeypatch.setattr(standalone_capture, "capture_standalone_transform", fake_transform)

    result = standalone_capture.main(
        [
            "--oracle-root",
            str(tmp_path / "oracle"),
            "--preprocessed-root",
            str(tmp_path / "preprocessed"),
            "--raw-root",
            str(tmp_path / "raw"),
            "--output-root",
            str(tmp_path / "output"),
            "--plans",
            str(tmp_path / "plans.json"),
        ]
    )

    assert result == 0
    assert calls == [
        {
            "oracle_root": tmp_path / "oracle",
            "preprocessed_root": tmp_path / "preprocessed",
            "raw_root": tmp_path / "raw",
            "output_root": tmp_path / "output",
            "plans_path": tmp_path / "plans.json",
        }
    ]
    assert json.loads(capsys.readouterr().out) == {
        "artifact_root": str(destination),
        "run_state": "official_alignment_pending",
    }


def test_capture_cli_dispatches_inference_arguments(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []
    destination = tmp_path / "inference" / "case001"

    def fake_inference(**kwargs: object) -> Path:
        calls.append(kwargs)
        return destination

    monkeypatch.setattr(standalone_capture, "capture_standalone_inference", fake_inference)

    result = standalone_capture.main(
        [
            "--mode",
            "inference",
            "--oracle-root",
            str(tmp_path / "oracle"),
            "--raw-root",
            str(tmp_path / "raw"),
            "--output-root",
            str(tmp_path / "output"),
            "--plans",
            str(tmp_path / "plans.json"),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--device",
            "cuda:0",
            "--slice-batch-size",
            "4",
        ]
    )

    assert result == 0
    assert calls == [
        {
            "oracle_root": tmp_path / "oracle",
            "raw_root": tmp_path / "raw",
            "checkpoint": tmp_path / "checkpoint.pt",
            "output_root": tmp_path / "output",
            "plans_path": tmp_path / "plans.json",
            "device": "cuda:0",
            "slice_batch_size": 4,
        }
    ]
    assert json.loads(capsys.readouterr().out) == {
        "artifact_root": str(destination),
        "run_state": "official_alignment_pending",
    }


def test_capture_cli_reports_missing_preprocessed_root_in_transform_mode(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    with pytest.raises(SystemExit) as error:
        standalone_capture.main(
            [
                "--oracle-root",
                str(tmp_path / "oracle"),
                "--raw-root",
                str(tmp_path / "raw"),
                "--output-root",
                str(tmp_path / "output"),
                "--plans",
                str(tmp_path / "plans.json"),
            ]
        )

    assert error.value.code == 2
    assert "transform mode requires --preprocessed-root" in capsys.readouterr().err


def test_capture_cli_reports_missing_checkpoint_in_inference_mode(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    with pytest.raises(SystemExit) as error:
        standalone_capture.main(
            [
                "--mode",
                "inference",
                "--oracle-root",
                str(tmp_path / "oracle"),
                "--raw-root",
                str(tmp_path / "raw"),
                "--output-root",
                str(tmp_path / "output"),
                "--plans",
                str(tmp_path / "plans.json"),
            ]
        )

    assert error.value.code == 2
    assert "inference mode requires --checkpoint" in capsys.readouterr().err
