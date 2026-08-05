"""Server-only capture of fixed nnU-Net oracle artifacts.

This module is safe to import in the standalone environment. The installed
``nnunetv2`` package is imported only after the command has validated an
explicit capture mode and paths. No capture mode starts or resumes a run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from standalone_nnunet2d.data.nifti_io import NiftiVolume, read_nifti
from standalone_nnunet2d.tools.parity_report import RUN_STATE


CAPTURE_MODES = ("preprocess", "sample", "transform", "deep_supervision", "inference")
DEFAULT_DATASET_ID = "Dataset501_StrokeLesion"
DEFAULT_PLANS = Path(__file__).resolve().parent / "reference" / "nnUNetPlans.json"


class OracleCaptureError(RuntimeError):
    """Raised when a server-side oracle capture cannot be completed safely."""


@dataclass(frozen=True)
class CaptureContext:
    mode: str
    raw_root: Path
    preprocessed_root: Path
    results_root: Path
    output_root: Path
    case_id: str
    fold: int
    seed: int
    plans_path: Path
    model_folder: Path | None
    device: str
    checkpoint_name: str
    nnunetv2_version: str


def _load_nnunetv2() -> Any:
    """Load nnU-Net only from the server-side capture path."""
    try:
        import nnunetv2
    except ModuleNotFoundError as error:
        raise OracleCaptureError(
            "oracle capture requires the server environment with nnunetv2 installed; "
            "the standalone environment must not run this command"
        ) from error
    return nnunetv2


def _module_version(nnunetv2: Any) -> str:
    version = getattr(nnunetv2, "__version__", None)
    if version is not None:
        return str(version)
    try:
        version_module = importlib.import_module("nnunetv2._version")
    except ModuleNotFoundError:
        return "unknown"
    return str(getattr(version_module, "__version__", "unknown"))


def _find_case_file(root: Path, case_id: str, suffix: str) -> Path:
    direct = root / f"{case_id}{suffix}"
    if direct.is_file():
        return direct
    candidates = sorted(path for path in root.rglob("*") if path.is_file() and path.name == direct.name)
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected exactly one {suffix} artifact for {case_id!r} below {root}, found {len(candidates)}"
        )
    return candidates[0]


def _raw_case_paths(raw_root: Path, case_id: str) -> tuple[Path, Path]:
    image_path = raw_root / "imagesTr" / f"{case_id}_0000.nii.gz"
    label_path = raw_root / "labelsTr" / f"{case_id}.nii.gz"
    if not image_path.is_file():
        raise FileNotFoundError(f"raw image does not exist: {image_path}")
    if not label_path.is_file():
        raise FileNotFoundError(f"raw label does not exist: {label_path}")
    return image_path, label_path


def _as_channel_free(array: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(array)
    if result.ndim == 4 and result.shape[0] == 1:
        result = result[0]
    if result.ndim not in (2, 3):
        raise OracleCaptureError(f"{name} must be 2D or 3D after channel removal, got {result.shape}")
    return result


def _read_preprocessed_case(root: Path, case_id: str) -> tuple[np.ndarray, np.ndarray]:
    path = _find_case_file(root, case_id, ".npz")
    with np.load(path, allow_pickle=False) as payload:
        if "data" not in payload or "seg" not in payload:
            raise OracleCaptureError(f"official preprocessed artifact is missing data/seg arrays: {path}")
        image = _as_channel_free(payload["data"], name="preprocessed data")
        label = _as_channel_free(payload["seg"], name="preprocessed seg")
    if image.shape != label.shape:
        raise OracleCaptureError(f"preprocessed data/seg shapes differ: {image.shape} != {label.shape}")
    return image.astype(np.float32, copy=False), label.astype(np.int16, copy=False)


def _read_raw_case(ctx: CaptureContext) -> tuple[NiftiVolume, NiftiVolume, Path, Path]:
    image_path, label_path = _raw_case_paths(ctx.raw_root, ctx.case_id)
    image = read_nifti(image_path)
    label = read_nifti(label_path)
    if image.array.shape != label.array.shape:
        raise OracleCaptureError(f"raw image/label shapes differ: {image.array.shape} != {label.array.shape}")
    if not np.allclose(image.spacing_xyz, label.spacing_xyz, rtol=0.0, atol=1e-6):
        raise OracleCaptureError("raw image/label spacing differs")
    return image, label, image_path, label_path


def _nifti_metadata(volume: NiftiVolume | None) -> dict[str, object]:
    if volume is None:
        return {"space": "preprocessed_or_array", "spacing_xyz": None, "origin_xyz": None, "direction": None}
    return {
        "space": "raw",
        "spacing_xyz": list(volume.spacing_xyz),
        "origin_xyz": list(volume.origin_xyz),
        "direction": list(volume.direction),
    }


def _load_sample(image: np.ndarray, label: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    if image.ndim != 3 or label.ndim != 3:
        raise OracleCaptureError("sampling requires 3D preprocessed data ordered as (z, y, x)")
    rng = np.random.default_rng(seed)
    z_index = int(rng.integers(image.shape[0]))
    return image[z_index], label[z_index], z_index


def _to_numpy(value: Any, *, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if result.size == 0:
        raise OracleCaptureError(f"official transform returned an empty {name}")
    return result


def _official_transform(image: np.ndarray, label: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        module = importlib.import_module("nnunetv2.training.data_augmentation.default_data_augmentation")
        factory = getattr(module, "get_training_transforms")
    except (ImportError, AttributeError) as error:
        raise OracleCaptureError("server nnunetv2 does not expose get_training_transforms") from error

    np.random.seed(seed)
    try:
        transform = factory(
            patch_size=tuple(int(value) for value in image.shape),
            deep_supervision_scales=None,
            mirror_axes=(0, 1),
            do_dummy_2d_data_aug=False,
        )
        result = transform({"data": image[None, None], "seg": label[None, None]})
    except (TypeError, ValueError, RuntimeError) as error:
        raise OracleCaptureError(f"official transform capture failed: {error}") from error
    if not isinstance(result, Mapping) or "data" not in result or "seg" not in result:
        raise OracleCaptureError("official transform did not return data and seg mappings")
    transformed_image = _to_numpy(result["data"], name="transformed image")
    transformed_label = _to_numpy(result["seg"], name="transformed label")
    while transformed_image.ndim > 2 and transformed_image.shape[0] == 1:
        transformed_image = transformed_image[0]
    while transformed_label.ndim > 2 and transformed_label.shape[0] == 1:
        transformed_label = transformed_label[0]
    if transformed_image.shape != transformed_label.shape:
        raise OracleCaptureError("official transform returned mismatched data and seg shapes")
    return transformed_image.astype(np.float32, copy=False), transformed_label.astype(np.int16, copy=False)


def _official_deep_supervision(label: np.ndarray) -> tuple[np.ndarray, ...]:
    try:
        module = importlib.import_module("nnunetv2.training.data_augmentation.custom_transforms.deep_supervision")
        transform_class = getattr(module, "DownsampleSegForDSTransform2", None)
        if transform_class is None:
            transform_class = getattr(module, "DownsampleSegForDSTransform")
    except (ImportError, AttributeError) as error:
        raise OracleCaptureError("server nnunetv2 does not expose a deep-supervision transform") from error

    scales = ((1.0, 1.0), (0.5, 0.5))
    try:
        transform = transform_class(ds_scales=scales, input_key="seg", output_key="target")
        result = transform({"seg": label[None, None]})
    except (TypeError, ValueError, RuntimeError) as error:
        raise OracleCaptureError(f"official deep-supervision capture failed: {error}") from error
    if not isinstance(result, Mapping) or "target" not in result:
        raise OracleCaptureError("official deep-supervision transform did not return target")
    targets = result["target"]
    if not isinstance(targets, (tuple, list)):
        targets = (targets,)
    return tuple(_to_numpy(target, name="deep-supervision target").astype(np.int16, copy=False) for target in targets)


def _official_inference(ctx: CaptureContext) -> tuple[np.ndarray, np.ndarray, np.ndarray, NiftiVolume]:
    if ctx.model_folder is None:
        raise OracleCaptureError("inference capture requires --model-folder")
    image, label, image_path, _ = _read_raw_case(ctx)
    try:
        module = importlib.import_module("nnunetv2.inference.predict_from_raw_data")
        predictor_class = getattr(module, "nnUNetPredictor")
        import torch
    except (ImportError, AttributeError) as error:
        raise OracleCaptureError("server nnunetv2 does not expose nnUNetPredictor") from error

    prediction_path = ctx.output_root / "_server_prediction" / f"{ctx.case_id}.nii.gz"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    predictor_kwargs = {
        "tile_step_size": 0.5,
        "use_gaussian": True,
        "use_mirroring": True,
        "perform_everything_on_device": True,
        "device": torch.device(ctx.device),
        "verbose": False,
        "verbose_preprocessing": False,
        "allow_tqdm": False,
    }
    try:
        try:
            predictor = predictor_class(**predictor_kwargs)
        except TypeError:
            predictor = predictor_class(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                device=torch.device(ctx.device),
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=False,
            )
        predictor.initialize_from_trained_model_folder(
            str(ctx.model_folder),
            use_folds=(ctx.fold,),
            checkpoint_name=ctx.checkpoint_name,
        )
        predictor.predict_from_files(
            [[str(image_path)]],
            [str(prediction_path)],
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        raise OracleCaptureError(f"official inference capture failed: {error}") from error
    prediction = read_nifti(prediction_path)
    return image.array.astype(np.float32, copy=False), label.array.astype(np.int16, copy=False), prediction.array.astype(np.uint8, copy=False), image


def _plans_hash(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"plans file does not exist: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_artifact(
    ctx: CaptureContext,
    arrays: Mapping[str, np.ndarray],
    *,
    nifti_metadata: Mapping[str, object],
    extra_policy: Mapping[str, object] | None = None,
) -> Path:
    destination = ctx.output_root / ctx.mode / ctx.case_id
    destination.mkdir(parents=True, exist_ok=True)
    saved_arrays: dict[str, dict[str, object]] = {}
    for name, value in arrays.items():
        array = np.asarray(value)
        filename = f"{name}.npy"
        np.save(destination / filename, array)
        saved_arrays[name] = {"file": filename, "shape": list(array.shape), "dtype": str(array.dtype)}
    manifest: dict[str, object] = {
        "artifact_version": 1,
        "nnunetv2_version": ctx.nnunetv2_version,
        "plans_hash": _plans_hash(ctx.plans_path),
        "seed": ctx.seed,
        "case_id": ctx.case_id,
        "transform_policy": {"mode": ctx.mode, **dict(extra_policy or {})},
        "sampling_policy": {"seed": ctx.seed, "fold": ctx.fold},
        "arrays": saved_arrays,
        "nifti_metadata": dict(nifti_metadata),
        "run_state": RUN_STATE,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return destination


def capture_oracle(
    *,
    mode: str,
    raw_root: Path,
    preprocessed_root: Path,
    results_root: Path,
    output_root: Path,
    case_id: str,
    fold: int = 0,
    seed: int = 0,
    plans_path: Path = DEFAULT_PLANS,
    model_folder: Path | None = None,
    device: str = "cuda",
    checkpoint_name: str = "checkpoint_best.pth",
) -> Path:
    """Capture one fixed oracle artifact using only non-training server APIs."""
    if mode not in CAPTURE_MODES:
        raise ValueError(f"mode must be one of {CAPTURE_MODES}, got {mode!r}")
    if not case_id:
        raise ValueError("case_id must not be empty")
    if fold < 0:
        raise ValueError("fold must be non-negative")
    raw_root = Path(raw_root).resolve()
    preprocessed_root = Path(preprocessed_root).resolve()
    results_root = Path(results_root).resolve()
    output_root = Path(output_root).resolve()
    if any(output_root == source or source in output_root.parents for source in (raw_root, preprocessed_root, results_root)):
        raise ValueError("output_root must not be inside an input or results root")

    nnunetv2 = _load_nnunetv2()
    ctx = CaptureContext(
        mode=mode,
        raw_root=raw_root,
        preprocessed_root=preprocessed_root,
        results_root=results_root,
        output_root=output_root,
        case_id=case_id,
        fold=fold,
        seed=seed,
        plans_path=Path(plans_path).resolve(),
        model_folder=None if model_folder is None else Path(model_folder).resolve(),
        device=device,
        checkpoint_name=checkpoint_name,
        nnunetv2_version=_module_version(nnunetv2),
    )

    raw_volume: NiftiVolume | None = None
    extra_policy: dict[str, object] = {}
    if mode == "preprocess":
        image, label = _read_preprocessed_case(ctx.preprocessed_root, ctx.case_id)
        extra_policy["source"] = "official_preprocessed_npz"
    elif mode == "sample":
        image, label = _read_preprocessed_case(ctx.preprocessed_root, ctx.case_id)
        image, label, z_index = _load_sample(image, label, seed=ctx.seed)
        extra_policy.update({"source": "official_preprocessed_npz", "z_index": z_index})
    elif mode == "transform":
        image, label = _read_preprocessed_case(ctx.preprocessed_root, ctx.case_id)
        image, label, z_index = _load_sample(image, label, seed=ctx.seed)
        image, label = _official_transform(image, label, seed=ctx.seed)
        extra_policy.update({"source": "nnunetv2_training_transform", "z_index": z_index})
    elif mode == "deep_supervision":
        image, label = _read_preprocessed_case(ctx.preprocessed_root, ctx.case_id)
        image, label, z_index = _load_sample(image, label, seed=ctx.seed)
        targets = _official_deep_supervision(label)
        extra_policy.update({"source": "nnunetv2_deep_supervision_transform", "z_index": z_index})
    else:
        image, label, mask, raw_volume = _official_inference(ctx)
        extra_policy["source"] = "nnunetv2_predictor"

    if mode != "inference":
        mask = (label > 0).astype(np.uint8, copy=False)
    arrays: dict[str, np.ndarray] = {
        "image": np.asarray(image),
        "label": np.asarray(label),
        "mask": np.asarray(mask),
    }
    if mode == "deep_supervision":
        arrays.update({f"deep_supervision_{index}": target for index, target in enumerate(targets)})
    if raw_volume is None:
        try:
            raw_volume = read_nifti(_raw_case_paths(ctx.raw_root, ctx.case_id)[0])
        except (FileNotFoundError, OSError):
            raw_volume = None
    return _write_artifact(ctx, arrays, nifti_metadata=_nifti_metadata(raw_volume), extra_policy=extra_policy)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture one nnU-Net oracle artifact on the server")
    parser.add_argument("--mode", required=True, choices=CAPTURE_MODES)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--preprocessed-root", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plans", type=Path, default=DEFAULT_PLANS)
    parser.add_argument("--model-folder", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-name", default="checkpoint_best.pth")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    destination = capture_oracle(
        mode=arguments.mode,
        raw_root=arguments.raw_root,
        preprocessed_root=arguments.preprocessed_root,
        results_root=arguments.results_root,
        output_root=arguments.output_root,
        case_id=arguments.case_id,
        fold=arguments.fold,
        seed=arguments.seed,
        plans_path=arguments.plans,
        model_folder=arguments.model_folder,
        device=arguments.device,
        checkpoint_name=arguments.checkpoint_name,
    )
    print(json.dumps({"artifact_root": str(destination), "run_state": RUN_STATE}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
