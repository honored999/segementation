"""Command-line formal full-volume prediction for standalone nnUNet 2D."""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from standalone_nnunet2d.config import load_model_config
from standalone_nnunet2d.data.dataset import load_fold_cases, read_case_images, resolve_channel_specs
from standalone_nnunet2d.data.inference_preprocessing import (
    prepare_bilateral_asymmetry_case,
    prepare_dwi_adc_bilateral_case,
    restore_bilateral_asymmetry_prediction,
)
from standalone_nnunet2d.data.input_mode import InputMode, input_spec
from standalone_nnunet2d.alignment_evidence import (
    OFFICIAL_ALIGNED,
    validate_checkpoint_alignment_metadata,
)
from standalone_nnunet2d.engine.predictor import (
    DEFAULT_MIRROR_AXES,
    DEFAULT_PATCH_SIZE,
    DEFAULT_TILE_STEP_SIZE,
    predict_volume,
    save_and_validate_prediction,
)
from standalone_nnunet2d.models.plain_conv_unet import PlainConvUNet2D
from standalone_nnunet2d.training.formal_dataset import resolve_input_mode
from standalone_nnunet2d.training.official_config import DEFAULT_RUN_STATE
from standalone_nnunet2d.training.formal_checkpoint import (
    checkpoint_input_channels,
    checkpoint_input_mode,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "values": _json_safe(value.tolist())}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict source-space masks from a formal 2D checkpoint")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--case-id", nargs="+", default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--slice-batch-size", type=int, default=1)
    parser.add_argument("--allow-pending", action="store_true")
    parser.add_argument("--input-mode", type=InputMode, choices=tuple(InputMode))
    parser.add_argument("--bilateral-asymmetry-channel", action="store_true")
    return parser


def _read_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    if not isinstance(payload.get("model_state_dict"), Mapping):
        raise ValueError("checkpoint is missing model_state_dict")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata must be a dictionary")
    return payload["model_state_dict"], dict(metadata)


def _load_model(
    path: Path, device: torch.device, *, input_channels: int | None = None
) -> tuple[PlainConvUNet2D, dict[str, Any]]:
    state_dict, metadata = _read_checkpoint(path)
    resolved_input_channels = checkpoint_input_channels(metadata) if input_channels is None else input_channels
    model = PlainConvUNet2D(
        load_model_config(input_channels=resolved_input_channels), deep_supervision=False
    )
    model.load_state_dict(state_dict)
    return model.to(device), metadata


def _resolve_case_ids(raw_root: Path, case_ids: list[str] | None, fold: int | None) -> tuple[str, ...]:
    image_root = raw_root.resolve() / "imagesTr"
    if not image_root.is_dir():
        raise FileNotFoundError(f"raw root is missing imagesTr: {image_root}")
    if case_ids is not None and fold is not None:
        raise ValueError("use either --case-id or --fold, not both")
    if case_ids:
        resolved = tuple(case_ids)
    elif fold is not None:
        resolved = load_fold_cases(fold, "val")
    else:
        raise ValueError("one or more --case-id values or --fold is required")
    if not resolved:
        raise ValueError("at least one case must be selected")
    return resolved


def _source_path(raw_root: Path, case_id: str) -> Path:
    return raw_root.resolve() / "imagesTr" / f"{case_id}_0000.nii.gz"


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        runtime_input_mode = resolve_input_mode(
            arguments.input_mode,
            bilateral_asymmetry_channel=arguments.bilateral_asymmetry_channel,
        )
    except ValueError as exc:
        parser.error(str(exc))
    _, checkpoint_metadata = _read_checkpoint(arguments.checkpoint)
    run_state, alignment_evidence = validate_checkpoint_alignment_metadata(
        checkpoint_metadata
    )
    if run_state == DEFAULT_RUN_STATE and not arguments.allow_pending:
        raise ValueError("pending checkpoint requires explicit --allow-pending")

    resolved_input_mode = checkpoint_input_mode(checkpoint_metadata)
    if runtime_input_mode is not None and runtime_input_mode is not resolved_input_mode:
        raise ValueError(
            f"runtime input_mode={runtime_input_mode.value} conflicts with "
            f"checkpoint input_mode={resolved_input_mode.value}"
        )
    resolved_input_spec = input_spec(resolved_input_mode)
    channel_specs = resolve_channel_specs(arguments.raw_root)
    physical_channels = len(channel_specs)
    checkpoint_channels = resolved_input_spec.effective_input_channels
    if physical_channels != resolved_input_spec.physical_input_channels:
        raise ValueError(
            "checkpoint input_channels="
            f"{checkpoint_channels} does not match dataset channels={physical_channels}"
        )
    declared_modalities = tuple(modality for _, modality in channel_specs)
    legacy_single_channel = channel_specs == ((0, "legacy_single_channel"),)
    if (
        declared_modalities != resolved_input_spec.physical_modalities
        and not (legacy_single_channel and resolved_input_spec.physical_input_channels == 1)
    ):
        raise ValueError(
            f"checkpoint input_mode={resolved_input_mode.value} requires physical modalities="
            f"{resolved_input_spec.physical_modalities}, found {declared_modalities}"
        )

    device = torch.device(arguments.device)
    model, loaded_metadata = _load_model(
        arguments.checkpoint, device, input_channels=checkpoint_channels
    )
    case_ids = _resolve_case_ids(arguments.raw_root, arguments.case_id, arguments.fold)
    prediction_root = arguments.output_root.resolve() / "predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)
    case_records: list[dict[str, Any]] = []
    for case_id in case_ids:
        if resolved_input_mode is InputMode.DWI_ADC_BILATERAL:
            prepared = prepare_dwi_adc_bilateral_case(arguments.raw_root, case_id)
            source = prepared.source_image
            prediction = predict_volume(
                model, prepared.model_volumes, device, mirror_axes=DEFAULT_MIRROR_AXES,
                patch_size=DEFAULT_PATCH_SIZE, tile_step_size=DEFAULT_TILE_STEP_SIZE,
                slice_batch_size=arguments.slice_batch_size, normalise_inputs=False,
            )
            prediction = restore_bilateral_asymmetry_prediction(prepared, prediction)
            source_paths = [
                str(arguments.raw_root.resolve() / "imagesTr" / f"{case_id}_{index:04d}.nii.gz")
                for index, _ in channel_specs
            ]
        elif resolved_input_mode is InputMode.DWI_BILATERAL:
            prepared = prepare_bilateral_asymmetry_case(arguments.raw_root, case_id)
            source = prepared.source_image
            sources = prepared.model_volumes
            prediction = predict_volume(
                model, sources, device, mirror_axes=DEFAULT_MIRROR_AXES,
                patch_size=DEFAULT_PATCH_SIZE, tile_step_size=DEFAULT_TILE_STEP_SIZE,
                slice_batch_size=arguments.slice_batch_size, normalise_inputs=False,
            )
            prediction = restore_bilateral_asymmetry_prediction(prepared, prediction)
            source_paths = [str(_source_path(arguments.raw_root, case_id))]
        else:
            sources = read_case_images(arguments.raw_root, case_id)
            source = sources[0]
            source_paths = [
                str(arguments.raw_root.resolve() / "imagesTr" / f"{case_id}_{index:04d}.nii.gz")
                for index, _ in resolve_channel_specs(arguments.raw_root)
            ]
            prediction = predict_volume(
                model, sources if len(sources) > 1 else source, device,
                mirror_axes=DEFAULT_MIRROR_AXES, patch_size=DEFAULT_PATCH_SIZE,
                tile_step_size=DEFAULT_TILE_STEP_SIZE, slice_batch_size=arguments.slice_batch_size,
            )
        prediction_path = prediction_root / f"{case_id}.nii.gz"
        validation = save_and_validate_prediction(prediction_path, prediction, source)
        case_records.append(
            {
                "case_id": case_id,
                "source_path": source_paths[0],
                "source_paths": source_paths,
                "prediction_path": str(prediction_path),
                "nifti_validation": validation,
            }
        )

    manifest = {
        "schema_version": 1,
        "policy": {
            "run_state": run_state,
            "alignment_status": run_state,
            "full_resolution_logits": True,
            "mirror_axes": list(DEFAULT_MIRROR_AXES),
            "mirror_aggregation": "unflip_logits_then_mean",
            "tile_patch_size": list(DEFAULT_PATCH_SIZE),
            "tile_step_size": DEFAULT_TILE_STEP_SIZE,
            "tile_aggregation": "logit_mean_then_argmax",
            "postprocessing": "class_argmax_after_all_tta_and_tiles",
            "slice_batch_size": arguments.slice_batch_size,
            "output_space": "source",
            "output_dtype": "uint8",
            "input_mode": resolved_input_mode.value,
            "physical_input_channels": physical_channels,
            "effective_model_input_channels": checkpoint_channels,
            "bilateral_asymmetry_channel": resolved_input_mode is InputMode.DWI_BILATERAL,
        },
        "checkpoint": {
            "path": str(arguments.checkpoint.resolve()),
            **_json_safe(loaded_metadata),
        },
        "cases": case_records,
    }
    if alignment_evidence is not None:
        manifest["policy"]["alignment_evidence"] = copy.deepcopy(alignment_evidence)
    manifest_path = arguments.output_root.resolve() / "prediction_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
