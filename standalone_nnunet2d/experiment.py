"""Validated experiment contract that intentionally does not start training."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from standalone_nnunet2d.engine.checkpoint import PROJECT_OUTPUTS_DIRECTORY
from standalone_nnunet2d.engine.smoke_runner import run_smoke_epoch, smoke_case_ids
from standalone_nnunet2d.engine.case_validation import validate_cases
from standalone_nnunet2d.data.dataset import StrokeSliceDataset
from standalone_nnunet2d.config import load_model_config
from standalone_nnunet2d.losses.compound import DiceCrossEntropyLoss
from standalone_nnunet2d.models import PlainConvUNet2D
from standalone_nnunet2d.tools.server_preflight import inspect_server_readiness
import torch
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class ExperimentRequest:
    raw_root: Path
    preprocessed_root: Path
    results_root: Path
    output_root: Path
    fold: int
    epochs: int
    device: str | None


def _fold(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= 4:
        raise argparse.ArgumentTypeError("fold must be between 0 and 4")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("epochs must be positive")
    return parsed


def _output_path(value: str) -> Path:
    resolved = Path(value).expanduser().resolve()
    try:
        resolved.relative_to(PROJECT_OUTPUTS_DIRECTORY.resolve())
    except ValueError as error:
        raise argparse.ArgumentTypeError("output root must be under standalone_nnunet2d/outputs") from error
    return resolved


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validated standalone nnU-Net experiment request")
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--preprocessed-root", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=_output_path)
    parser.add_argument("--fold", required=True, type=_fold)
    parser.add_argument("--epochs", required=True, type=_positive_int)
    parser.add_argument("--device")
    parser.add_argument("--confirm-run", action="store_true")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    """Print a validated request; confirmed execution remains deliberately deferred."""
    parsed = _parser().parse_args(arguments)
    request = ExperimentRequest(
        raw_root=parsed.raw_root.expanduser().resolve(),
        preprocessed_root=parsed.preprocessed_root.expanduser().resolve(),
        results_root=parsed.results_root.expanduser().resolve(),
        output_root=parsed.output_root,
        fold=parsed.fold,
        epochs=parsed.epochs,
        device=parsed.device,
    )
    request_payload = {
        "raw_root": str(request.raw_root), "preprocessed_root": str(request.preprocessed_root),
        "results_root": str(request.results_root), "output_root": str(request.output_root),
        "fold": request.fold, "epochs": request.epochs, "device": request.device,
    }
    readiness = inspect_server_readiness(
        request.raw_root, request.preprocessed_root, request.results_root, device=request.device
    )
    if parsed.confirm_run and request.epochs != 1:
        _parser().error("smoke run requires --epochs 1")
    if parsed.confirm_run and readiness["ready"]:
        train_case, validation_case = smoke_case_ids(request.fold)
        required_files = (
            request.raw_root / "imagesTr" / f"{train_case}_0000.nii.gz",
            request.raw_root / "labelsTr" / f"{train_case}.nii.gz",
            request.raw_root / "imagesTr" / f"{validation_case}_0000.nii.gz",
            request.raw_root / "labelsTr" / f"{validation_case}.nii.gz",
        )
        missing = [str(path) for path in required_files if not path.is_file()]
        if missing:
            readiness["ready"] = False
            readiness["diagnostics"].append(f"smoke-run case files are missing: {missing}")
    if parsed.confirm_run and readiness["ready"]:
        train_loader = DataLoader(StrokeSliceDataset(request.raw_root, fold=request.fold, split="train", case_ids=(train_case,)), batch_size=1)
        validation_loader = DataLoader(StrokeSliceDataset(request.raw_root, fold=request.fold, split="val", case_ids=(validation_case,)), batch_size=1)
        device = torch.device(request.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = PlainConvUNet2D(load_model_config()).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0)
        smoke = run_smoke_epoch(model, train_loader, validation_loader, DiceCrossEntropyLoss(), optimizer, device, request.output_root / "smoke_checkpoint.pt")
        validation = validate_cases(model, request.raw_root, (validation_case,), request.output_root, device, fold=request.fold, checkpoint_path=Path(smoke["checkpoint"]))
        payload = {"request": request_payload, "readiness": readiness, "execution": "completed", "smoke_run_only": True, "smoke": smoke, "validation_cases": validation}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    payload = {
        "request": request_payload,
        "readiness": readiness,
        "execution": "deferred" if parsed.confirm_run else "not-confirmed",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 3 if parsed.confirm_run else (0 if readiness["ready"] else 2)


if __name__ == "__main__":
    raise SystemExit(main())
