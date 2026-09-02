"""Capture one standalone transform artifact from a fixed oracle request."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from standalone_nnunet2d.data.nifti_io import read_nifti
from standalone_nnunet2d.data.input_mode import InputMode
from standalone_nnunet2d.data.inference_preprocessing import (
    prepare_bilateral_asymmetry_volume,
    restore_bilateral_asymmetry_prediction,
)
from standalone_nnunet2d.engine.predictor import predict_volume
from standalone_nnunet2d.predict import _load_model
from standalone_nnunet2d.training.formal_checkpoint import checkpoint_input_mode
from standalone_nnunet2d.tools.parity_report import RUN_STATE
from standalone_nnunet2d.training.official_augmentation import (
    apply_official_2d_batchgeneratorsv2,
)


MANIFEST_NAME = "manifest.json"
TRANSFORM_MODE = "transform"
INFERENCE_MODE = "inference"


def _load_manifest(root: Path) -> dict[str, Any]:
    path = root / MANIFEST_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"oracle manifest is unreadable: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"oracle manifest must contain a JSON object: {path}")
    return payload


def _required_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"oracle manifest field '{name}' must be an integer")
    return value


def _oracle_fields(manifest: Mapping[str, Any]) -> tuple[str, int, int, int]:
    case_id = manifest.get("case_id")
    if not isinstance(case_id, str) or not case_id or Path(case_id).name != case_id:
        raise ValueError("oracle manifest field 'case_id' must be a safe non-empty string")

    seed = _required_int(manifest.get("seed"), name="seed")
    policy_value = manifest.get("transform_policy")
    if policy_value is None:
        policy: Mapping[str, Any] = {}
    elif isinstance(policy_value, Mapping):
        policy = policy_value
    else:
        raise ValueError("oracle manifest field 'transform_policy' must be an object")

    mode = policy.get("mode", manifest.get("capture_mode"))
    if mode != TRANSFORM_MODE:
        raise ValueError("oracle artifact capture mode must be 'transform'")

    z_value = policy["z_index"] if "z_index" in policy else manifest.get("z_index")
    if z_value is None:
        raise ValueError("oracle manifest must provide a sampled z_index")
    z_index = _required_int(z_value, name="transform_policy.z_index")
    return case_id, seed, z_index, 0


def _find_preprocessed_case(
    root: Path, case_id: str, suffix: str, *, allow_missing: bool = False
) -> Path | None:
    direct = root / f"{case_id}{suffix}"
    if direct.is_file():
        return direct
    candidates = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name == direct.name
    )
    if allow_missing and not candidates:
        return None
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected exactly one {suffix} artifact for {case_id!r} below {root}, found {len(candidates)}"
        )
    return candidates[0]


def _channel_free(array: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(array)
    if result.ndim == 4 and result.shape[0] == 1:
        result = result[0]
    if result.ndim not in (2, 3):
        raise ValueError(f"official preprocessed {name} must be 2D or 3D, got {result.shape}")
    return result


def _read_preprocessed_case(root: Path, case_id: str) -> tuple[np.ndarray, np.ndarray]:
    npz_path = _find_preprocessed_case(root, case_id, ".npz", allow_missing=True)
    if npz_path is not None:
        with np.load(npz_path, allow_pickle=False) as payload:
            if "data" not in payload or "seg" not in payload:
                raise ValueError(f"official preprocessed artifact is missing data/seg arrays: {npz_path}")
            image = _channel_free(payload["data"], name="data")
            label = _channel_free(payload["seg"], name="seg")
    else:
        data_path = _find_preprocessed_case(root, case_id, ".b2nd")
        seg_path = root / f"{case_id}_seg.b2nd"
        if not seg_path.is_file():
            candidates = sorted(
                path for path in root.rglob("*") if path.is_file() and path.name == seg_path.name
            )
            if len(candidates) == 1:
                seg_path = candidates[0]
            else:
                raise FileNotFoundError(
                    f"official preprocessed segmentation artifact is missing: {seg_path}"
                )
        try:
            import blosc2
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "b2nd preprocessed artifacts require blosc2 in the server environment"
            ) from error
        image = _channel_free(
            np.asarray(blosc2.open(urlpath=str(data_path), mode="r")), name="data"
        )
        label = _channel_free(
            np.asarray(blosc2.open(urlpath=str(seg_path), mode="r")), name="seg"
        )
    if image.shape != label.shape:
        raise ValueError(f"preprocessed data/seg shapes differ: {image.shape} != {label.shape}")
    return image.astype(np.float32, copy=False), label.astype(np.int16, copy=False)


def _raw_metadata(raw_root: Path, case_id: str) -> dict[str, object]:
    image_path = raw_root / "imagesTr" / f"{case_id}_0000.nii.gz"
    volume = read_nifti(image_path)
    return {
        "space": "raw",
        "spacing_xyz": [float(value) for value in volume.spacing_xyz],
        "origin_xyz": [float(value) for value in volume.origin_xyz],
        "direction": [float(value) for value in volume.direction],
    }


def _plans_hash(plans_path: Path) -> str:
    if not plans_path.is_file():
        raise FileNotFoundError(f"plans file does not exist: {plans_path}")
    return hashlib.sha256(plans_path.read_bytes()).hexdigest()


def _normalize_device(device: str | torch.device) -> str:
    try:
        resolved = torch.device(device)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"invalid inference device: {device!r}") from error
    if resolved.type == "cpu":
        return "cpu"
    if resolved.type == "cuda":
        index = resolved.index
        if index is None:
            try:
                index = int(torch.cuda.current_device())
            except (RuntimeError, AssertionError) as error:
                raise ValueError(
                    "cannot resolve the active CUDA index for inference device 'cuda'"
                ) from error
        return f"cuda:{index}"
    raise ValueError(f"inference device must be cpu or cuda, got {resolved}")


def _inference_context_from_metadata(
    metadata: Mapping[str, Any], device: torch.device
) -> dict[str, object]:
    fold = metadata.get("fold")
    if isinstance(fold, bool) or not isinstance(fold, int) or fold < 0:
        raise ValueError("checkpoint metadata provenance field 'fold' must be a non-negative integer")
    source_sha256 = metadata.get("source_sha256")
    if not isinstance(source_sha256, str) or len(source_sha256) != 64:
        raise ValueError(
            "checkpoint metadata provenance field 'source_sha256' must be a 64-character SHA256"
        )
    try:
        int(source_sha256, 16)
    except ValueError as error:
        raise ValueError(
            "checkpoint metadata provenance field 'source_sha256' must be a 64-character SHA256"
        ) from error
    return {
        "fold": fold,
        "source_checkpoint_sha256": source_sha256,
        "device": _normalize_device(device),
    }


def _array_description(array: np.ndarray, filename: str) -> dict[str, object]:
    return {
        "file": filename,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def capture_standalone_transform(
    *,
    oracle_root: Path,
    preprocessed_root: Path,
    raw_root: Path,
    output_root: Path,
    plans_path: Path,
    case_id: str | None = None,
) -> Path:
    """Capture a standalone transform artifact for the oracle's fixed slice."""
    oracle_root = Path(oracle_root).resolve()
    preprocessed_root = Path(preprocessed_root).resolve()
    raw_root = Path(raw_root).resolve()
    output_root = Path(output_root).resolve()
    plans_path = Path(plans_path).resolve()

    if not oracle_root.is_dir():
        raise FileNotFoundError(f"oracle artifact root does not exist: {oracle_root}")
    if not preprocessed_root.is_dir():
        raise FileNotFoundError(f"preprocessed root does not exist: {preprocessed_root}")
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw root does not exist: {raw_root}")
    if any(output_root == source or source in output_root.parents for source in (
        oracle_root,
        preprocessed_root,
        raw_root,
    )):
        raise ValueError("output_root must not be inside an input root")

    manifest = _load_manifest(oracle_root)
    oracle_case_id, seed, z_index, _ = _oracle_fields(manifest)
    if case_id is not None:
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("case_id must be a non-empty string when supplied")
        if case_id != oracle_case_id:
            raise ValueError("case_id must match the oracle manifest case_id")
    case_id = oracle_case_id
    image_volume, label_volume = _read_preprocessed_case(preprocessed_root, case_id)
    if image_volume.ndim != 3 or label_volume.ndim != 3:
        raise ValueError("transform capture requires 3D official preprocessed data")
    if not 0 <= z_index < image_volume.shape[0]:
        raise ValueError(
            f"oracle manifest z_index {z_index} is outside preprocessed depth {image_volume.shape[0]}"
        )

    source_image = np.array(image_volume[z_index], dtype=np.float32, copy=True)
    source_label = np.array(label_volume[z_index], dtype=np.int16, copy=True)
    with plans_path.open(encoding="utf-8") as handle:
        plans = json.load(handle)
    configuration = plans["configurations"]["2d"]
    patch_size = tuple(int(value) for value in configuration["patch_size"])
    use_mask_for_norm = configuration["use_mask_for_norm"]
    transformed_image, transformed_label = apply_official_2d_batchgeneratorsv2(
        source_image,
        source_label,
        patch_size=patch_size,
        use_mask_for_norm=use_mask_for_norm,
        seed=seed,
    )
    image = np.asarray(transformed_image, dtype=np.float32)
    label = np.asarray(transformed_label, dtype=np.int16)
    mask = (label > 0).astype(np.uint8, copy=False)
    arrays = {"image": image, "label": label, "mask": mask}

    destination = output_root / TRANSFORM_MODE / case_id
    destination.mkdir(parents=True, exist_ok=True)
    array_manifest: dict[str, dict[str, object]] = {}
    for name, array in arrays.items():
        filename = f"{name}.npy"
        np.save(destination / filename, array)
        array_manifest[name] = _array_description(array, filename)

    output_manifest: dict[str, object] = {
        "artifact_version": 1,
        "nnunetv2_version": "not_applicable",
        "plans_hash": _plans_hash(plans_path),
        "seed": seed,
        "case_id": case_id,
        "implementation": "standalone",
        "capture_mode": TRANSFORM_MODE,
        "transform_policy": {
            "mode": TRANSFORM_MODE,
            "implementation": "standalone",
            "source": "official_preprocessed_npz",
            "z_index": z_index,
        },
        "sampling_policy": {
            "seed": seed,
            "implementation": "standalone",
            "z_index": z_index,
        },
        "arrays": array_manifest,
        "nifti_metadata": _raw_metadata(raw_root, case_id),
        "run_state": RUN_STATE,
    }
    (destination / MANIFEST_NAME).write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return destination


def capture_standalone_inference(
    *,
    oracle_root: Path,
    raw_root: Path,
    checkpoint: Path,
    output_root: Path,
    plans_path: Path,
    device: str | torch.device,
    case_id: str | None = None,
    slice_batch_size: int = 1,
) -> Path:
    """Capture one standalone source-space inference artifact."""
    oracle_root = Path(oracle_root).resolve()
    raw_root = Path(raw_root).resolve()
    checkpoint = Path(checkpoint).resolve()
    output_root = Path(output_root).resolve()
    plans_path = Path(plans_path).resolve()

    if not oracle_root.is_dir():
        raise FileNotFoundError(f"oracle artifact root does not exist: {oracle_root}")
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw root does not exist: {raw_root}")
    if any(output_root == source or source in output_root.parents for source in (oracle_root, raw_root)):
        raise ValueError("output_root must not be inside an input root")
    if slice_batch_size <= 0:
        raise ValueError(f"slice_batch_size must be positive, got {slice_batch_size}")

    oracle_manifest = _load_manifest(oracle_root)
    oracle_case_id = oracle_manifest.get("case_id")
    if not isinstance(oracle_case_id, str) or not oracle_case_id or Path(oracle_case_id).name != oracle_case_id:
        raise ValueError("oracle manifest field 'case_id' must be a safe non-empty string")
    seed = _required_int(oracle_manifest.get("seed"), name="seed")
    policy_value = oracle_manifest.get("transform_policy")
    if policy_value is None:
        policy: Mapping[str, Any] = {}
    elif isinstance(policy_value, Mapping):
        policy = policy_value
    else:
        raise ValueError("oracle manifest field 'transform_policy' must be an object")
    mode = policy.get("mode", oracle_manifest.get("capture_mode"))
    if mode != INFERENCE_MODE:
        raise ValueError("oracle artifact capture mode must be 'inference'")
    if case_id is not None:
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("case_id must be a non-empty string when supplied")
        if case_id != oracle_case_id:
            raise ValueError("case_id must match the oracle manifest case_id")
    case_id = oracle_case_id

    raw_image = read_nifti(raw_root / "imagesTr" / f"{case_id}_0000.nii.gz")
    raw_label = read_nifti(raw_root / "labelsTr" / f"{case_id}.nii.gz")
    if raw_image.array.shape != raw_label.array.shape:
        raise ValueError(f"raw image/label shapes differ: {raw_image.array.shape} != {raw_label.array.shape}")
    if not np.allclose(raw_image.spacing_xyz, raw_label.spacing_xyz, rtol=0.0, atol=1e-6):
        raise ValueError("raw image/label spacing differs")

    source_image = np.asarray(raw_image.array, dtype=np.float32).copy()
    source_label = np.asarray(raw_label.array, dtype=np.int16).copy()
    torch_device = torch.device(device)
    model, checkpoint_metadata = _load_model(checkpoint, torch_device)
    inference_context = _inference_context_from_metadata(checkpoint_metadata, torch_device)
    input_mode = checkpoint_input_mode(checkpoint_metadata)
    bilateral_asymmetry_channel = input_mode is InputMode.DWI_BILATERAL
    if bilateral_asymmetry_channel:
        prepared = prepare_bilateral_asymmetry_volume(raw_image)
        prediction = restore_bilateral_asymmetry_prediction(
            prepared,
            np.asarray(
                predict_volume(
                    model,
                    prepared.model_volumes,
                    torch_device,
                    slice_batch_size=slice_batch_size,
                    normalise_inputs=False,
                )
            ),
        )
    else:
        prediction = np.asarray(
            predict_volume(model, raw_image, torch_device, slice_batch_size=slice_batch_size)
        )
    if prediction.shape != source_image.shape:
        raise ValueError(
            "prediction shape must match raw image shape: "
            f"{prediction.shape} != {source_image.shape}"
        )
    if not np.isin(prediction, (0, 1)).all():
        raise ValueError("prediction labels must contain only 0 and 1")
    mask = prediction.astype(np.uint8, copy=False)
    arrays = {"image": source_image, "label": source_label, "mask": mask}

    destination = output_root / INFERENCE_MODE / case_id
    destination.mkdir(parents=True, exist_ok=True)
    array_manifest: dict[str, dict[str, object]] = {}
    for name, array in arrays.items():
        filename = f"{name}.npy"
        np.save(destination / filename, array)
        array_manifest[name] = _array_description(array, filename)

    output_manifest: dict[str, object] = {
        "artifact_version": 1,
        "nnunetv2_version": "not_applicable",
        "plans_hash": _plans_hash(plans_path),
        "seed": seed,
        "case_id": case_id,
        "implementation": "standalone",
        "capture_mode": INFERENCE_MODE,
        "transform_policy": {
            "mode": INFERENCE_MODE,
            "implementation": "standalone",
            "source": "raw_nifti",
            "predictor": "standalone_nnunet2d.engine.predictor.predict_volume",
            "device": str(torch_device),
            "slice_batch_size": slice_batch_size,
            "bilateral_asymmetry_channel": bilateral_asymmetry_channel,
        },
        "sampling_policy": {
            "seed": seed,
            "fold": inference_context["fold"],
            "implementation": "standalone",
            "source": "oracle_manifest",
        },
        "inference_context": inference_context,
        "arrays": array_manifest,
        "nifti_metadata": {
            "space": "raw",
            "spacing_xyz": [float(value) for value in raw_image.spacing_xyz],
            "origin_xyz": [float(value) for value in raw_image.origin_xyz],
            "direction": [float(value) for value in raw_image.direction],
        },
        "run_state": RUN_STATE,
    }
    (destination / MANIFEST_NAME).write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture one standalone artifact")
    parser.add_argument("--mode", choices=(TRANSFORM_MODE, INFERENCE_MODE), default=TRANSFORM_MODE)
    parser.add_argument("--oracle-root", required=True, type=Path)
    parser.add_argument("--preprocessed-root", type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--plans", "--plans-path", dest="plans_path", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--slice-batch-size", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.mode == TRANSFORM_MODE:
        if arguments.preprocessed_root is None:
            parser.error("transform mode requires --preprocessed-root")
        destination = capture_standalone_transform(
            oracle_root=arguments.oracle_root,
            preprocessed_root=arguments.preprocessed_root,
            raw_root=arguments.raw_root,
            output_root=arguments.output_root,
            plans_path=arguments.plans_path,
        )
    else:
        if arguments.checkpoint is None:
            parser.error("inference mode requires --checkpoint")
        destination = capture_standalone_inference(
            oracle_root=arguments.oracle_root,
            raw_root=arguments.raw_root,
            checkpoint=arguments.checkpoint,
            output_root=arguments.output_root,
            plans_path=arguments.plans_path,
            device=arguments.device,
            slice_batch_size=arguments.slice_batch_size,
        )
    print(json.dumps({"artifact_root": str(destination), "run_state": RUN_STATE}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
