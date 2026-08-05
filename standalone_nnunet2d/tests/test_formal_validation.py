from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from standalone_nnunet2d.data.nifti_io import NiftiVolume, read_nifti, write_nifti
from standalone_nnunet2d.engine import formal_validation


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
