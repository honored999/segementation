"""Whole-case smoke-only validation outputs."""
from __future__ import annotations
from pathlib import Path
import torch
from torch import nn
from standalone_nnunet2d.data.nifti_io import read_nifti
from standalone_nnunet2d.engine.predictor import predict_volume, save_and_validate_prediction
from standalone_nnunet2d.metrics.case_metrics import volume_metrics
from standalone_nnunet2d.metrics.overlays import write_overlay
from standalone_nnunet2d.metrics.reporting import write_case_reports

def validate_cases(model: nn.Module, raw_root: Path, case_ids: tuple[str, ...], output_root: Path, device: torch.device, *, fold: int, checkpoint_path: Path) -> list[dict[str, object]]:
    validation = output_root / "validation"; records: list[dict[str, object]] = []
    for case_id in case_ids:
        image = read_nifti(raw_root / "imagesTr" / f"{case_id}_0000.nii.gz"); label = read_nifti(raw_root / "labelsTr" / f"{case_id}.nii.gz")
        prediction = predict_volume(model, image, device); save_and_validate_prediction(validation / "predictions" / f"{case_id}.nii.gz", prediction, label)
        record = {"case_id": case_id, **volume_metrics(prediction, label.array.astype("uint8"))}; records.append(record)
        write_overlay(validation / "overlays" / f"{case_id}.png", image.array, label.array, prediction, case_id=case_id, dice=float(record["dice"]))
    write_case_reports(records, validation, fold=fold, checkpoint_path=checkpoint_path)
    return records
