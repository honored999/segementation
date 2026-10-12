"""Create a bounded, source-train-only CKPD artifact for the nnU-Net run.

This tool reads Dataset508 prepared inputs and its existing fold split. It does
not use labels for sampling, writes no files under raw/preprocessed roots, and
uses nnU-Net 2.8.1's official pretrained loader (including seg-head exclusion).
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

EXTENSION_ROOT = Path(__file__).resolve().parents[1] / "nnunet_ext_trainers"
if str(EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTENSION_ROOT))

from ckpd import (  # noqa: E402
    CKPD_SCHEMA_VERSION,
    atomic_torch_save,
    decompose_conv2d,
    encoder_conv_layers,
    encoder_semantic_signature,
    hash_file,
    hash_identifier,
    module_fingerprint,
    sample_conv2d_patches,
    select_smallest_asr,
)


def _read_fold(path: Path, fold: int) -> tuple[str, list[str], list[str]]:
    if not path.is_file():
        raise RuntimeError(f"required existing splits_final.json not found: {path}")
    digest = hash_file(path)
    splits = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(splits, list) or fold < 0 or fold >= len(splits):
        raise RuntimeError(f"fold {fold} is absent from {path}")
    split = splits[fold]
    if not isinstance(split, dict) or not isinstance(split.get("train"), list) or not isinstance(split.get("val"), list):
        raise RuntimeError(f"fold {fold} in {path} must contain train and val case lists")
    train = [str(identifier) for identifier in split["train"]]
    val = [str(identifier) for identifier in split["val"]]
    if not train or not val or len(set(train)) != len(train) or set(train).intersection(val):
        raise RuntimeError(f"fold {fold} has empty, duplicate, or overlapping case IDs")
    return digest, train, val


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _paths_overlap(first: Path, second: Path) -> bool:
    return _is_within(first, second) or _is_within(second, first)


def _resolve_configuration_data_dir(preprocessed_root: Path, data_identifier: str) -> Path:
    if not isinstance(data_identifier, str) or not data_identifier or Path(data_identifier).name != data_identifier:
        raise RuntimeError(f"invalid nnU-Net data_identifier: {data_identifier!r}")
    root = preprocessed_root.resolve()
    resolved = (root / data_identifier).resolve()
    if resolved == root or not _is_within(resolved, root) or not resolved.is_dir():
        raise RuntimeError(f"configuration prepared-data folder is absent or escapes Dataset508 root: {resolved}")
    return resolved


def _assert_output_is_safe(output: Path, protected_roots) -> Path:
    resolved = output.resolve()
    for root in protected_roots:
        protected = Path(root).resolve()
        if _paths_overlap(resolved, protected):
            raise RuntimeError(f"calibration output overlaps protected source/output root: {resolved} vs {protected}")
    if resolved.exists():
        raise RuntimeError(f"refusing to overwrite an existing calibration artifact: {resolved}")
    return resolved


def _configured_output_protected_roots(args) -> list[Path]:
    from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw

    roots = [args.preprocessed_dir, args.dataset_json.parent, args.plans_json.parent]
    for env_path in (nnUNet_raw, nnUNet_preprocessed):
        if env_path.is_set():
            roots.append(Path(os.fspath(env_path)))
    # Protect the whole source experiment hierarchy, including the Dataset508
    # result folder, rather than only the fold directory containing the file.
    checkpoint = args.source_checkpoint.resolve()
    roots.extend(checkpoint.parents[:4])
    return roots


def _patch_from_slice(data: np.ndarray, patch_size: tuple[int, int], rng: random.Random) -> np.ndarray:
    if data.ndim != 3:
        raise ValueError("prepared image slice must be [channels, height, width]")
    channels, height, width = data.shape
    target_h, target_w = patch_size
    crop_h, crop_w = min(height, target_h), min(width, target_w)
    y0 = rng.randrange(height - crop_h + 1) if height > crop_h else 0
    x0 = rng.randrange(width - crop_w + 1) if width > crop_w else 0
    cropped = data[:, y0:y0 + crop_h, x0:x0 + crop_w]
    pad_h, pad_w = target_h - crop_h, target_w - crop_w
    if pad_h or pad_w:
        cropped = np.pad(cropped, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
    return np.asarray(cropped, dtype=np.float32)


def _collect_source_slices(args, train_ids: list[str], rng: random.Random) -> tuple[list[str], list[torch.Tensor]]:
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

    data_folder = Path(args.configuration_data_dir)
    dataset_class = infer_dataset_class(str(data_folder))
    available = set(dataset_class.get_identifiers(str(data_folder)))
    chosen = sorted(rng.sample(sorted(train_ids), args.cases))
    missing = sorted(set(chosen) - available)
    if missing:
        raise RuntimeError(f"selected Dataset508 train cases are missing from {data_folder}: {missing[:5]}")
    dataset = dataset_class(str(data_folder), identifiers=chosen)
    patches = []
    for identifier in chosen:
        data, _seg, _previous_stage, _properties = dataset.load_case(identifier)
        if len(data.shape) != 4:
            raise RuntimeError(f"prepared Dataset508 case {identifier} is not a 2D nnU-Net array")
        depth = int(data.shape[1])
        z = rng.randrange(depth)
        image_slice = np.asarray(data[:, z, :, :], dtype=np.float32)
        patch = _patch_from_slice(image_slice, args.patch_size, rng)
        patches.append(torch.from_numpy(patch).unsqueeze(0))
        # nnU-Net's case API also materializes the paired segmentation; it is
        # discarded without inspection and never enters patch selection.
        del data, _seg, _previous_stage, _properties, image_slice, patch
    return chosen, patches


def _host_resource_snapshot() -> tuple[float, int, int]:
    try:
        import psutil
        cpu_pct = float(psutil.cpu_percent(interval=0.5))
        memory = psutil.virtual_memory()
        return cpu_pct, int(memory.total), int(memory.available)
    except ImportError:
        if os.name != "nt":
            raise RuntimeError("resource preflight requires psutil or Windows host resource APIs")

    import ctypes.wintypes as wintypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise RuntimeError("Windows GlobalMemoryStatusEx failed during CKPD preflight")
    kernel32 = ctypes.windll.kernel32

    def cpu_times():
        idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
        if not kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            raise RuntimeError("Windows GetSystemTimes failed during CKPD preflight")
        def value(filetime):
            return (int(filetime.dwHighDateTime) << 32) | int(filetime.dwLowDateTime)
        return value(idle), value(kernel), value(user)

    before = cpu_times()
    time.sleep(0.5)
    after = cpu_times()
    idle_delta = after[0] - before[0]
    total_delta = (after[1] - before[1]) + (after[2] - before[2])
    cpu_pct = 0.0 if total_delta <= 0 else 100.0 * (1.0 - idle_delta / total_delta)
    return cpu_pct, int(status.ullTotalPhys), int(status.ullAvailPhys)


def _physical_gpu_utilizations() -> list[dict[str, int]]:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=10,
    )
    rows = []
    for line in query.stdout.splitlines():
        if not line.strip():
            continue
        pieces = [piece.strip() for piece in line.split(",")]
        if len(pieces) != 2:
            raise RuntimeError(f"unexpected nvidia-smi utilization row: {line!r}")
        rows.append({"physical_index": int(pieces[0]), "utilization_pct": int(pieces[1])})
    return rows


def _require_gpu_utilization_headroom(rows: list[dict[str, int]]) -> None:
    busy = [row for row in rows if row["utilization_pct"] >= 80]
    if busy:
        raise RuntimeError(f"CKPD calibration rejected physical GPU utilization >=80%: {busy}")


def _early_resource_preflight(args) -> None:
    cpu_pct, ram_total, ram_available = _host_resource_snapshot()
    checkpoint_bytes = args.source_checkpoint.stat().st_size
    # A checkpoint-sized host reserve is a lower-bound check before plans or
    # model construction. Detailed model/covariance/activation estimates follow.
    host_reserve = checkpoint_bytes + 256 * 1024**2
    host_used = ram_total - ram_available
    report = {
        "stage": "before_checkpoint_or_network_load",
        "cpu_logical": os.cpu_count(),
        "cpu_load_pct": round(cpu_pct, 1),
        "ram_total_gib": round(ram_total / 1024**3, 2),
        "ram_available_gib": round(ram_available / 1024**3, 2),
        "source_checkpoint_gib": round(checkpoint_bytes / 1024**3, 2),
        "minimum_host_reserve_gib": round(host_reserve / 1024**3, 2),
        "host_used_after_minimum_reserve_pct": round(100 * (host_used + host_reserve) / ram_total, 1),
    }
    if args.device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA calibration requested but CUDA is unavailable")
        visible_index = args.device.index if args.device.index is not None else torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(visible_index)
        gpu_reserve = checkpoint_bytes + 256 * 1024**2
        rows = _physical_gpu_utilizations()
        _require_gpu_utilization_headroom(rows)
        report.update({
            "visible_gpu": torch.cuda.get_device_name(visible_index),
            "visible_gpu_index": visible_index,
            "gpu_free_gib": round(free / 1024**3, 2),
            "gpu_total_gib": round(total / 1024**3, 2),
            "minimum_gpu_reserve_gib": round(gpu_reserve / 1024**3, 2),
            "physical_gpu_utilization": rows,
        })
        if gpu_reserve > free or (total - free + gpu_reserve) / total >= 0.80:
            raise RuntimeError(f"early CKPD preflight lacks minimum visible-GPU headroom: {json.dumps(report)}")
    else:
        try:
            report["physical_gpu_utilization"] = _physical_gpu_utilizations()
        except (OSError, subprocess.SubprocessError, FileNotFoundError):
            report["physical_gpu_utilization"] = None
    print("CKPD_EARLY_RESOURCE_PREFLIGHT " + json.dumps(report, sort_keys=True), flush=True)
    if cpu_pct >= 80 or host_reserve > ram_available or (host_used + host_reserve) / ram_total >= 0.80:
        raise RuntimeError(f"early CKPD preflight lacks minimum host/CPU headroom: {json.dumps(report)}")


def _resource_preflight(args, network: nn.Module, max_dimension: int) -> None:
    cpu_pct, ram_total, ram_available = _host_resource_snapshot()
    model_bytes = sum(t.numel() * t.element_size() for t in list(network.parameters()) + list(network.buffers()))
    covariance_bytes = max_dimension * max_dimension * 8
    sampled_patches_bytes = args.max_positions * max_dimension * 4 * 2
    patch_h, patch_w = args.patch_size
    max_channels = max((module.out_channels for module in network.modules() if isinstance(module, nn.Conv2d)), default=1)
    forward_activation_bytes = patch_h * patch_w * max_channels * 4 * 3
    checkpoint_bytes = args.source_checkpoint.stat().st_size
    host_reserve = 2 * checkpoint_bytes + 2 * model_bytes + covariance_bytes + sampled_patches_bytes + forward_activation_bytes
    host_used = ram_total - ram_available
    report = {
        "stage": "after_network_before_official_weight_load",
        "cpu_logical": os.cpu_count(),
        "cpu_load_pct": round(cpu_pct, 1),
        "ram_total_gib": round(ram_total / 1024**3, 2),
        "ram_available_gib": round(ram_available / 1024**3, 2),
        "estimated_host_reserve_gib": round(host_reserve / 1024**3, 2),
        "host_used_after_estimated_reserve_pct": round(100 * (host_used + host_reserve) / ram_total, 1),
        "largest_covariance_mib": round(covariance_bytes / 1024**2, 1),
        "source_checkpoint_gib": round(checkpoint_bytes / 1024**3, 2),
        "network_gib": round(model_bytes / 1024**3, 2),
    }
    if args.device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA calibration requested but CUDA is unavailable")
        visible_index = args.device.index if args.device.index is not None else torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(visible_index)
        gpu_rows = _physical_gpu_utilizations()
        _require_gpu_utilization_headroom(gpu_rows)
        gpu_reserve = 2 * checkpoint_bytes + 2 * model_bytes + forward_activation_bytes + sampled_patches_bytes
        report.update({
            "visible_gpu": torch.cuda.get_device_name(visible_index),
            "visible_gpu_index": visible_index,
            "gpu_free_gib": round(free / 1024**3, 2),
            "gpu_total_gib": round(total / 1024**3, 2),
            "estimated_gpu_reserve_gib": round(gpu_reserve / 1024**3, 2),
            "physical_gpu_utilization": gpu_rows,
        })
        if gpu_reserve > free or (total - free + gpu_reserve) / total >= 0.80:
            raise RuntimeError(f"CKPD preflight rejected estimated GPU reserve: {json.dumps(report)}")
    else:
        report["physical_gpu_utilization"] = _physical_gpu_utilizations() if torch.cuda.is_available() else None
    print("CKPD_RESOURCE_PREFLIGHT " + json.dumps(report, sort_keys=True), flush=True)
    if cpu_pct >= 80 or host_reserve > ram_available or (host_used + host_reserve) / ram_total >= 0.80:
        raise RuntimeError(f"CKPD preflight rejected estimated host/CPU reserve: {json.dumps(report)}")


def _stable_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _validate_source_checkpoint_device(args, init_args: dict) -> None:
    if "device" not in init_args:
        raise RuntimeError("source checkpoint init_args.device is required to validate official loader placement")
    try:
        source_device = torch.device(init_args["device"])
    except (TypeError, RuntimeError) as exc:
        raise RuntimeError(f"source checkpoint init_args.device is malformed: {exc}") from exc
    if source_device.type not in {"cpu", "cuda"}:
        raise RuntimeError(f"unsupported source checkpoint device type: {source_device.type}")
    if source_device.type == "cuda":
        if args.device.type != "cuda":
            raise RuntimeError(
                "CPU calibration cannot load a CUDA-origin checkpoint: the pinned official loader calls "
                "torch.load without map_location; rerun with --device cuda"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA-origin checkpoint requires an available CUDA device")
        visible_count = torch.cuda.device_count()
        target_index = args.device.index if args.device.index is not None else torch.cuda.current_device()
        source_index = source_device.index
        if source_index is not None and source_index >= visible_count:
            raise RuntimeError(
                f"source checkpoint CUDA index {source_index} is hidden or unavailable under CUDA_VISIBLE_DEVICES "
                f"(visible device count={visible_count})"
            )
        if source_index is not None and source_index != target_index:
            raise RuntimeError(
                f"source checkpoint CUDA index {source_index} differs from requested visible device {target_index}"
            )


def _validate_source_checkpoint_metadata(args, plans: dict, dataset_json: dict, plans_manager) -> dict:
    checkpoint = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    try:
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("network_weights"), dict):
            raise RuntimeError("source checkpoint must contain a network_weights mapping")
        trainer_name = checkpoint.get("trainer_name")
        expected_trainer = "nnUNetTrainerTopK10EarlyStopping"
        init_args = checkpoint.get("init_args")
        if trainer_name != expected_trainer or not isinstance(init_args, dict):
            raise RuntimeError("source checkpoint must come from ordinary nnUNetTrainerTopK10EarlyStopping with init_args")
        _validate_source_checkpoint_device(args, init_args)
        if init_args.get("configuration") != args.configuration or init_args.get("fold") != args.fold:
            raise RuntimeError("source checkpoint configuration/fold metadata differs from requested Dataset508 fold 0 2d")
        if not isinstance(init_args.get("plans"), dict) or _stable_json(init_args["plans"]) != _stable_json(plans):
            raise RuntimeError("source checkpoint plans metadata differs from --plans-json")
        if not isinstance(init_args.get("dataset_json"), dict) or _stable_json(init_args["dataset_json"]) != _stable_json(dataset_json):
            raise RuntimeError("source checkpoint dataset_json metadata differs from --dataset-json")
        dataset_name = plans_manager.dataset_name
        declared_dataset_name = dataset_json.get("name")
        if (dataset_name != "Dataset508_ISLES2022DWI"
                or declared_dataset_name not in (None, "Dataset508_ISLES2022DWI")):
            raise RuntimeError(
                "source plans must identify Dataset508_ISLES2022DWI and any dataset.json name must agree"
            )
        plans_name = plans_manager.plans_name
        if plans_name != "nnUNetPlansFrom501":
            raise RuntimeError("source plan identifier must be nnUNetPlansFrom501 for the prepared Dataset508 input")
        run_folder = f"{expected_trainer}__{plans_name}__{args.configuration}"
        checkpoint_path = args.source_checkpoint.resolve()
        if (checkpoint_path.parent.name != f"fold_{args.fold}"
                or checkpoint_path.parent.parent.name != run_folder
                or checkpoint_path.parent.parent.parent.name != dataset_name):
            raise RuntimeError("source checkpoint path does not match its declared Dataset508 Trainer/plans/config/fold")
        return {
            "source_trainer_name": trainer_name,
            "source_plans_name": plans_name,
            "source_data_identifier": plans_manager.get_configuration(args.configuration).data_identifier,
            "source_dataset_name": dataset_name,
        }
    finally:
        del checkpoint
        gc.collect()


def _estimate_covariance(network: nn.Module, name: str, conv: nn.Conv2d,
                         samples: list[torch.Tensor], args, layer_index: int) -> torch.Tensor:
    dimension = conv.weight.numel() // conv.out_channels
    covariance = torch.zeros((dimension, dimension), dtype=torch.float64, device="cpu")
    for case_index, sample in enumerate(samples):
        captured = {"count": 0}

        def hook(module, inputs, _outputs):
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise RuntimeError(f"could not capture input to encoder convolution {name}")
            patches = sample_conv2d_patches(
                inputs[0], module, args.max_positions,
                seed=args.seed + layer_index * 100_003 + case_index * 997,
            )
            values = patches.detach().to(device="cpu", dtype=torch.float64)
            covariance.add_(values.T @ values / max(1, values.shape[0]))
            captured["count"] += 1

        handle = conv.register_forward_hook(hook)
        try:
            with torch.no_grad():
                network(sample.to(device=args.device, dtype=torch.float32))
        finally:
            handle.remove()
        if captured["count"] != 1:
            raise RuntimeError(f"encoder convolution {name} executed {captured['count']} times for one calibration case")
    return covariance / len(samples)


def calibrate(args) -> dict:
    from nnunetv2.run.load_pretrained_weights import load_pretrained_weights
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    _early_resource_preflight(args)
    split_path = args.preprocessed_dir / "splits_final.json"
    split_sha, train_ids, val_ids = _read_fold(split_path, args.fold)
    if args.cases > len(train_ids):
        raise ValueError(f"requested {args.cases} cases but Dataset508 fold train has {len(train_ids)}")
    plans = json.loads(args.plans_json.read_text(encoding="utf-8"))
    dataset_json = json.loads(args.dataset_json.read_text(encoding="utf-8"))
    if len(dataset_json.get("channel_names", {})) < 1:
        raise RuntimeError("Dataset508 dataset.json has no image channels")
    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(args.configuration)
    if not configuration_manager.data_identifier:
        raise RuntimeError("source nnU-Net configuration has no data_identifier")
    args.configuration_data_dir = _resolve_configuration_data_dir(
        args.preprocessed_dir, configuration_manager.data_identifier,
    )
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
    dataset_class = infer_dataset_class(str(args.configuration_data_dir))
    available_ids = set(dataset_class.get_identifiers(str(args.configuration_data_dir)))
    missing_train_ids = sorted(set(train_ids) - available_ids)
    if missing_train_ids:
        raise RuntimeError(f"Dataset508 fold-train cases are absent from the resolved prepared configuration folder: {missing_train_ids[:5]}")
    source_metadata = _validate_source_checkpoint_metadata(args, plans, dataset_json, plans_manager)
    if not configuration_manager.network_arch_class_name.endswith(".PlainConvUNet"):
        raise RuntimeError("CKPD calibration supports only the ordinary 2D PlainConvUNet architecture")
    patch_size = tuple(int(value) for value in configuration_manager.patch_size)
    if len(patch_size) != 2:
        raise RuntimeError("Dataset508 2d configuration must define a two-dimensional patch_size")
    args.patch_size = patch_size
    label_manager = plans_manager.get_label_manager(dataset_json)
    network = nnUNetTrainer.build_network_architecture(
        plans_manager,
        configuration_manager,
        len(dataset_json["channel_names"]),
        label_manager.num_segmentation_heads,
        enable_deep_supervision=True,
    )
    if type(network).__name__ != "PlainConvUNet" or not hasattr(network, "encoder") or not hasattr(network, "decoder"):
        raise RuntimeError("source architecture is not the ordinary 2D PlainConvUNet")
    convs = encoder_conv_layers(network)
    eligible = {
        name: module for name, module in convs.items()
        if min(module.out_channels, module.weight.numel() // module.out_channels) >= args.rank
    }
    if args.layers > len(eligible):
        raise RuntimeError(f"requested {args.layers} selected layers but only {len(eligible)} Conv2d layers support rank {args.rank}")
    max_dimension = max(module.weight.numel() // module.out_channels for module in eligible.values())
    maximum_covariance = max_dimension * max_dimension * 8
    if maximum_covariance > args.max_covariance_mib * 1024**2:
        raise RuntimeError(
            f"largest float64 covariance needs {maximum_covariance / 1024**2:.1f} MiB, "
            f"over --max-covariance-mib={args.max_covariance_mib}"
        )
    network_bytes = sum(t.numel() * t.element_size() for t in list(network.parameters()) + list(network.buffers()))
    checkpoint_bytes = args.source_checkpoint.stat().st_size
    if network_bytes > 2**63 - checkpoint_bytes:
        raise RuntimeError("network/checkpoint memory estimate overflow")
    _resource_preflight(args, network, max_dimension)
    network.to(args.device)
    network.eval()
    # Same pinned official loader used by nnUNetv2_train; seg_layers are excluded.
    load_pretrained_weights(network, str(args.source_checkpoint), verbose=False)
    encoder_fingerprint = module_fingerprint(network.encoder)

    rng = random.Random(args.seed)
    selected_case_ids, samples = _collect_source_slices(args, train_ids, rng)
    singular_values_by_name = {}
    decompositions = {}
    for layer_index, (name, conv) in enumerate(sorted(eligible.items())):
        covariance = _estimate_covariance(network, name, conv, samples, args, layer_index)
        decomposition = decompose_conv2d(
            conv.weight.detach().cpu(), covariance, args.rank, args.damping_relative,
        )
        singular_values_by_name[name] = decomposition.singular_values
        decompositions[name] = decomposition
        del covariance
    selected_layers = select_smallest_asr(singular_values_by_name, args.rank, args.layers)

    artifact = {
        "schema_version": CKPD_SCHEMA_VERSION,
        "paper": "CKPD arXiv:2501.05017v2 equations 1-5 and ASR equation 7",
        "paper_url": "https://arxiv.org/html/2501.05017v2",
        "source_dataset": "Dataset508_ISLES2022DWI",
        "source_fold": args.fold,
        **source_metadata,
        "source_preprocessed_dir": str(args.preprocessed_dir.resolve()),
        "source_split_path": str(split_path.resolve()),
        "source_split_sha256": split_sha,
        "source_train_case_hashes": sorted(hash_identifier(identifier) for identifier in train_ids),
        "source_val_case_hashes": sorted(hash_identifier(identifier) for identifier in val_ids),
        "selected_case_hashes": sorted(hash_identifier(identifier) for identifier in selected_case_ids),
        "source_plans_path": str(args.plans_json.resolve()),
        "source_plans_sha256": hash_file(args.plans_json),
        "source_dataset_json_path": str(args.dataset_json.resolve()),
        "source_dataset_json_sha256": hash_file(args.dataset_json),
        "source_checkpoint_path": str(args.source_checkpoint.resolve()),
        "source_checkpoint_sha256": hash_file(args.source_checkpoint),
        "source_encoder_fingerprint": encoder_fingerprint,
        "source_encoder_semantic_signature": encoder_semantic_signature(network),
        "configuration": args.configuration,
        "patch_size": list(args.patch_size),
        "sample_count": len(selected_case_ids),
        "seed": args.seed,
        "covariance": "per-case mean of sampled uncentered Conv2d input-patch second moments; case means averaged equally",
        "max_positions_per_case_per_layer": args.max_positions,
        "damping_relative": args.damping_relative,
        "max_covariance_mib": args.max_covariance_mib,
        "rank": args.rank,
        "layer_count": args.layers,
        "asr_definition": "descending SVD(W C_reg): sigma[-rank] / sigma[-1]; lowest K scores selected (paper v2)",
        "encoder_conv_metadata": nnUNetTrainerCKPDMetadata(convs),
        "layer_scores": {
            name: {
                "asr": float(decompositions[name].asr),
                "singular_values": decompositions[name].singular_values.cpu(),
                "damping": float(decompositions[name].damping),
            }
            for name in sorted(decompositions)
        },
        "selected_layers": selected_layers,
        "factors": {
            name: {
                "base_weight": decompositions[name].base_weight.detach().cpu(),
                "A": decompositions[name].A.detach().cpu(),
                "B": decompositions[name].B.detach().cpu(),
                "singular_values": decompositions[name].singular_values.cpu(),
                "asr": decompositions[name].asr,
                "damping": decompositions[name].damping,
            }
            for name in selected_layers
        },
    }
    return artifact


def nnUNetTrainerCKPDMetadata(convs: dict[str, nn.Conv2d]) -> dict:
    return {
        name: {
            "shape": list(module.weight.shape),
            "dtype": str(module.weight.dtype),
            "stride": list(module.stride),
            "padding": list(module.padding),
            "dilation": list(module.dilation),
        }
        for name, module in sorted(convs.items())
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessed-dir", type=Path, required=True)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--plans-json", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configuration", default="2d")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--cases", type=int, default=8)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--max-positions", type=int, default=256)
    parser.add_argument("--max-covariance-mib", type=int, default=512)
    parser.add_argument("--damping-relative", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.fold != 0 or args.configuration != "2d":
        raise SystemExit("this approved calibration is restricted to Dataset508 fold 0, configuration 2d")
    if min(args.cases, args.rank, args.layers, args.max_positions, args.max_covariance_mib, args.cpu_threads) < 1:
        raise SystemExit("cases/rank/layers/positions/covariance budget/thread count must be positive")
    if not math.isfinite(args.damping_relative) or args.damping_relative < 0:
        raise SystemExit("damping-relative must be finite and nonnegative")
    for path in (args.preprocessed_dir, args.dataset_json, args.plans_json, args.source_checkpoint):
        if not path.exists():
            raise SystemExit(f"required source path does not exist: {path}")
    args.source_checkpoint = args.source_checkpoint.resolve()
    args.preprocessed_dir = args.preprocessed_dir.resolve()
    args.dataset_json = args.dataset_json.resolve()
    args.plans_json = args.plans_json.resolve()
    output = _assert_output_is_safe(args.output, _configured_output_protected_roots(args))
    args.device = torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    artifact = calibrate(args)
    atomic_torch_save(artifact, output)
    print("CKPD_CALIBRATION_CREATED " + json.dumps({
        "path": str(output),
        "sha256": hash_file(output),
        "source_cases": artifact["sample_count"],
        "selected_layers": artifact["selected_layers"],
        "rank": artifact["rank"],
        "layer_count": artifact["layer_count"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
