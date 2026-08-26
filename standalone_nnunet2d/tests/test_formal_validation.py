from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from standalone_nnunet2d.data.nifti_io import NiftiVolume, read_nifti, write_nifti
from standalone_nnunet2d.alignment_evidence import build_alignment_evidence
from standalone_nnunet2d.engine import formal_validation


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


def _write_raw_case(raw_root: Path, case_id: str, label_array: np.ndarray) -> None:
    image = NiftiVolume(
        np.arange(label_array.size, dtype=np.float32).reshape(label_array.shape),
        spacing_xyz=(0.7, 0.8, 4.5),
        origin_xyz=(11.0, -2.0, 3.5),
    )
    label = NiftiVolume(label_array.astype(np.uint8), image.spacing_xyz, image.origin_xyz, image.direction)
    write_nifti(raw_root / "imagesTr" / f"{case_id}_0000.nii.gz", image)
    write_nifti(raw_root / "labelsTr" / f"{case_id}.nii.gz", label)


def test_validate_fold_uses_full_volume_for_every_validation_case_and_writes_reports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw_root = tmp_path / "raw"
    (raw_root / "imagesTr").mkdir(parents=True)
    (raw_root / "labelsTr").mkdir()
    validation_ids = ("case001", "case002")
    _write_raw_case(raw_root, validation_ids[0], np.array([[[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]))
    _write_raw_case(raw_root, validation_ids[1], np.zeros((2, 3, 4), dtype=np.uint8))

    split_calls: list[tuple[int, str]] = []

    def fake_load_fold_cases(fold: int, split: str) -> tuple[str, ...]:
        split_calls.append((fold, split))
        return validation_ids

    full_volume_calls: list[tuple[object, tuple[int, ...], torch.device]] = []

    def fake_predict_volume(model: object, image: NiftiVolume, device: torch.device, **_: object) -> np.ndarray:
        full_volume_calls.append((model, image.array.shape, device))
        return np.zeros_like(image.array, dtype=np.uint8)

    def fail_if_online_validation_is_used(*_: object, **__: object) -> None:
        raise AssertionError("formal fold validation must not call run_validation_epoch")

    monkeypatch.setattr(formal_validation, "load_fold_cases", fake_load_fold_cases, raising=False)
    monkeypatch.setattr(formal_validation, "predict_volume", fake_predict_volume, raising=False)
    monkeypatch.setattr(formal_validation, "run_validation_epoch", fail_if_online_validation_is_used, raising=False)

    model = object()
    report = formal_validation.validate_fold(
        model,
        raw_root,
        fold=2,
        output_root=tmp_path / "fold-output",
        device=torch.device("cpu"),
    )

    assert split_calls == [(2, "val")]
    assert [shape for _, shape, _ in full_volume_calls] == [(2, 3, 4), (2, 3, 4)]
    assert [device for _, _, device in full_volume_calls] == [torch.device("cpu")] * 2
    assert report["case_count"] == len(validation_ids)
    assert report["aggregation"] == "case_macro_mean"

    prediction_paths = sorted((tmp_path / "fold-output").rglob("*.nii.gz"))
    assert len(prediction_paths) == len(validation_ids)
    assert {path.stem.removesuffix(".nii") for path in prediction_paths} == set(validation_ids)
    assert all(np.all(read_nifti(path).array == 0) for path in prediction_paths)

    csv_paths = sorted((tmp_path / "fold-output").rglob("*.csv"))
    assert csv_paths, "validate_fold must persist one case-metric CSV"
    with csv_paths[0].open(newline="", encoding="utf-8") as handle:
        case_rows = list(csv.DictReader(handle))
    assert len(case_rows) == len(validation_ids)
    assert {row["case_id"] for row in case_rows} == set(validation_ids)

    json_paths = sorted((tmp_path / "fold-output").rglob("*.json"))
    assert json_paths, "validate_fold must persist a fold report JSON"
    json_reports = [json.loads(path.read_text(encoding="utf-8")) for path in json_paths]
    assert any(
        report_json.get("case_count") == len(validation_ids)
        and report_json.get("aggregation") == "case_macro_mean"
        for report_json in json_reports
    )


def test_validate_fold_propagates_aligned_state_and_deep_copies_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw_root = tmp_path / "raw"
    (raw_root / "imagesTr").mkdir(parents=True)
    (raw_root / "labelsTr").mkdir()
    validation_ids = ("case001",)
    _write_raw_case(raw_root, validation_ids[0], np.zeros((1, 2, 2), dtype=np.uint8))
    monkeypatch.setattr(formal_validation, "load_fold_cases", lambda *_: validation_ids)
    monkeypatch.setattr(
        formal_validation,
        "predict_volume",
        lambda *_args, **_kwargs: np.zeros((1, 2, 2), dtype=np.uint8),
    )
    evidence = _aligned_evidence(tmp_path)

    report = formal_validation.validate_fold(
        object(),
        raw_root,
        fold=0,
        output_root=tmp_path / "aligned-fold",
        device=torch.device("cpu"),
        run_state="official_aligned",
        alignment_evidence=evidence,
    )

    assert report["run_state"] == "official_aligned"
    assert report["alignment_evidence"] == evidence
    assert report["alignment_evidence"] is not evidence
    report["alignment_evidence"]["sources"]["transform"]["snapshot"]["status"] = "changed"
    assert evidence["sources"]["transform"]["snapshot"]["status"] == "passed"
    persisted = json.loads(
        (tmp_path / "aligned-fold" / "fold_0_report.json").read_text(encoding="utf-8")
    )
    assert persisted["run_state"] == "official_aligned"
    assert persisted["alignment_evidence"] == evidence


def test_validate_fold_rejects_aligned_state_without_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(formal_validation, "validate_raw_root", lambda root: root)
    with pytest.raises(ValueError, match="alignment evidence"):
        formal_validation.validate_fold(
            object(),
            tmp_path / "raw",
            fold=0,
            output_root=tmp_path / "output",
            device=torch.device("cpu"),
            run_state="official_aligned",
        )


def test_validate_fold_checks_label_geometry_against_every_channel_before_predict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw_root = tmp_path / "multichannel-raw"
    (raw_root / "imagesTr").mkdir(parents=True)
    (raw_root / "labelsTr").mkdir()
    (raw_root / "dataset.json").write_text(
        json.dumps({"channel_names": {"0": "DWI", "1": "ADC"}}),
        encoding="utf-8",
    )
    case_id = "case001"
    reference = NiftiVolume(
        np.zeros((2, 2, 2), dtype=np.float32),
        spacing_xyz=(0.7, 0.8, 4.5),
        origin_xyz=(11.0, -2.0, 3.5),
    )
    label = NiftiVolume(
        np.zeros((2, 2, 2), dtype=np.uint8),
        reference.spacing_xyz,
        reference.origin_xyz,
        reference.direction,
    )
    mismatched_channel = NiftiVolume(
        np.zeros((2, 2, 3), dtype=np.float32),
        reference.spacing_xyz,
        reference.origin_xyz,
        reference.direction,
    )
    write_nifti(raw_root / "imagesTr" / f"{case_id}_0000.nii.gz", reference)
    write_nifti(raw_root / "imagesTr" / f"{case_id}_0001.nii.gz", mismatched_channel)
    write_nifti(raw_root / "labelsTr" / f"{case_id}.nii.gz", label)
    monkeypatch.setattr(formal_validation, "load_fold_cases", lambda *_: (case_id,))

    def fail_if_prediction_is_attempted(*_: object, **__: object) -> None:
        raise AssertionError("predict_volume must not run before channel geometry validation")

    monkeypatch.setattr(formal_validation, "predict_volume", fail_if_prediction_is_attempted)

    report = formal_validation.validate_fold(
        object(),
        raw_root,
        fold=0,
        output_root=tmp_path / "fold-output",
        device=torch.device("cpu"),
    )

    assert report["case_count"] == 0
    assert report["failed_case_count"] == 1
    error = report["failed_cases"][0]["error"]
    assert "channel 1" in error
    assert "geometry mismatch against label" in error
    assert "shape expected (2, 2, 2), got (2, 2, 3)" in error
