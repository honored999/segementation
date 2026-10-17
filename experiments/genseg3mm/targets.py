"""Train-only 3 mm retention targets for the exploratory GenSeg preview."""
from __future__ import annotations

from datetime import datetime, timezone
import csv
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from scipy.ndimage import distance_transform_edt

from experiments.genseg30.contract import (
    assert_identities, capture_identities, check_output, checked_file,
    read_json, require, sha256, write_json,
)
from experiments.genseg30.pipeline import load_run

CLASSIFICATION = "exploratory_dataset501_fold0_genseg3mm_preview"
TARGET_MANIFEST = "targets.json"
RECIPE = {
    "hd_bet_version": "2.0.1", "hd_bet_tta": True, "hd_bet_device": "cuda",
    "mask_margin_mm": 3.0, "mask_margin_axes": "native_slice_plane_only",
    "empty_slice": "retain_entire_native_slice", "empty_volume": "fail",
    "mapping": "source_reader_transpose_original_crop_nnUNet_seg_resampling",
    "target_fill": "baseline_generator_training_global_minimum",
    "target_space": "existing_preprocessed_target",
}
EXPECTED_SETTINGS = {
    "epochs": 200, "max_steps": 5000, "width": 64, "downs": 8,
    "discriminator_width": 64, "seed": 20261014, "lr": 2e-5,
    "l1_weight": 100.0, "batch_size": 1, "betas": [0.5, 0.999],
}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _safe_relative_file(root, name):
    root = Path(root).resolve(strict=True)
    require(isinstance(name, str) and name and not Path(name).is_absolute(),
            "Only relative target artifact paths are allowed")
    path = (root / name).resolve(strict=True)
    require(root in path.parents and path.is_file(), "Target artifact escapes its receipt or is missing")
    return path


def baseline_training_kwargs(metadata, *, device):
    settings = metadata.get("settings", {})
    for key, value in EXPECTED_SETTINGS.items():
        require(settings.get(key) == value, "Baseline generator setting mismatch: " + key)
    require(metadata.get("actual_steps") == 5000,
            "Baseline must be the complete approved 5000-update run")
    require(settings.get("norm") == "batch" and settings.get("dropout") is False and
            settings.get("gan_loss") == "vanilla" and settings.get("scheduler") == "constant_lr",
            "Baseline generator training contract mismatch")
    require(device == "cuda:0", "Selected physical GPU must be bound as cuda:0")
    return {key: settings[key] for key in (
        "epochs", "max_steps", "width", "downs", "discriminator_width", "seed", "lr", "l1_weight"
    )} | {"device": device}


def expand_keep_mask_3mm(brain_mask_zyx, *, spacing_xyz, margin_mm=3.0):
    """Expand each native z plane by physical y/x distance; never grow along z."""
    brain = np.asarray(brain_mask_zyx, dtype=bool)
    spacing = np.asarray(spacing_xyz, dtype=np.float64)
    require(brain.ndim == 3 and all(n > 0 for n in brain.shape), "Expected nonempty [Z,Y,X] brain mask")
    require(spacing.shape == (3,) and np.isfinite(spacing).all() and np.all(spacing > 0),
            "Finite positive native x/y/z spacing required")
    require(type(margin_mm) in (int, float) and math.isfinite(margin_mm) and margin_mm == 3.0,
            "Approved physical in-plane margin is fixed at 3 mm")
    require(brain.any(), "HD-BET predicted an entirely empty brain volume")
    keep = np.empty(brain.shape, dtype=bool)
    for z, plane in enumerate(brain):
        keep[z] = True if not plane.any() else (
            distance_transform_edt(~plane, sampling=(float(spacing[1]), float(spacing[0]))) <= margin_mm
        )
    return keep


def _check_sitk_geometry(image, *, context):
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    require(np.isfinite(spacing).all() and np.all(spacing > 0), context + ": invalid spacing")
    require(np.isfinite(direction).all() and
            np.allclose(direction.T @ direction, np.eye(3), atol=1e-4, rtol=0),
            context + ": sheared/nonorthogonal geometry is unsupported")
    require(abs(abs(float(np.linalg.det(direction))) - 1.0) <= 1e-4,
            context + ": invalid direction matrix")
    return spacing, direction


def validate_nifti_mm_orthogonal(path):
    """Reject unknown units or shear while allowing oblique orthogonal millimeter grids."""
    import nibabel as nib

    image = nib.load(str(path))
    unit, _ = image.header.get_xyzt_units()
    require(unit == "mm", "Raw DWI NIfTI must declare millimeter spatial units")
    linear = np.asarray(image.affine[:3, :3], dtype=np.float64)
    spacing = np.linalg.norm(linear, axis=0)
    require(np.isfinite(linear).all() and np.all(spacing > 0), "Invalid raw DWI affine")
    direction = linear / spacing
    require(np.allclose(direction.T @ direction, np.eye(3), atol=1e-4, rtol=0) and
            abs(abs(float(np.linalg.det(direction))) - 1.0) <= 1e-4,
            "Raw DWI affine contains shear or invalid directions")
    return image


def mask_image_to_raw_grid(mask_image, raw_image):
    """Reorient the HD-BET result to original raw axes and verify the physical voxel grid."""
    import SimpleITK as sitk

    _check_sitk_geometry(mask_image, context="HD-BET mask")
    _check_sitk_geometry(raw_image, context="Original raw DWI")
    orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(raw_image.GetDirection())
    aligned = sitk.DICOMOrient(mask_image, orientation)
    require(aligned.GetSize() == raw_image.GetSize() and
            np.allclose(aligned.GetSpacing(), raw_image.GetSpacing(), atol=1e-5, rtol=0) and
            np.allclose(aligned.GetOrigin(), raw_image.GetOrigin(), atol=1e-4, rtol=0) and
            np.allclose(aligned.GetDirection(), raw_image.GetDirection(), atol=1e-5, rtol=0),
            "Reoriented mask does not occupy the exact original raw grid")
    array = sitk.GetArrayFromImage(aligned)
    require(array.shape == tuple(reversed(raw_image.GetSize())) and np.isfinite(array).all() and
            np.isin(array, (0, 1)).all(), "Mask array is nonbinary or differs from original [Z,Y,X] grid")
    return aligned, array > 0


def map_keep_mask_to_preprocessed(raw_keep_zyx, plans_manager, configuration_manager,
                                  original_properties, *, expected_shape):
    """Apply original transpose, stored crop, and nnU-Net's segmentation resampler."""
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import compute_new_shape

    raw_keep = np.asarray(raw_keep_zyx, dtype=bool)
    require(raw_keep.ndim == 3 and raw_keep.any(), "Expected a nonempty native retention mask")
    transpose = tuple(plans_manager.transpose_forward)
    require(sorted(transpose) == [0, 1, 2], "Invalid original plans transpose")
    transposed = raw_keep.transpose(transpose)
    before = tuple(int(x) for x in original_properties["shape_before_cropping"])
    require(transposed.shape == before, "Mask does not match the original transposed source grid")
    bbox = original_properties["bbox_used_for_cropping"]
    require(len(bbox) == 3, "Original crop bbox is missing")
    slices = []
    for axis, pair in enumerate(bbox):
        require(len(pair) == 2 and isinstance(pair[0], (int, np.integer)) and
                isinstance(pair[1], (int, np.integer)) and
                0 <= pair[0] < pair[1] <= before[axis], "Invalid recorded original crop bbox")
        slices.append(slice(pair[0], pair[1]))
    cropped = transposed[tuple(slices)]
    crop_shape = tuple(int(x) for x in original_properties["shape_after_cropping_and_before_resampling"])
    require(cropped.shape == crop_shape, "Mask crop differs from original saved crop")
    spacing = [float(original_properties["spacing"][i]) for i in transpose]
    target_spacing = list(configuration_manager.spacing)
    if len(target_spacing) < len(crop_shape):
        require(len(target_spacing) == 2 and len(crop_shape) == 3,
                "Only original 2d configuration spacing convention is supported")
        target_spacing = [spacing[0], *target_spacing]
    require(len(spacing) == len(target_spacing) == len(crop_shape), "Spacing dimensionality mismatch")
    new_shape = tuple(int(x) for x in compute_new_shape(crop_shape, spacing, target_spacing))
    saved_shape = tuple(int(x) for x in expected_shape[1:])
    require(tuple(expected_shape[:1]) == (1,) and new_shape == saved_shape,
            "Mapped mask grid differs from saved preprocessed image")
    mapped = configuration_manager.resampling_fn_seg(
        cropped.astype(np.uint8, copy=False)[None], new_shape, spacing, target_spacing
    )
    mapped = np.asarray(mapped)
    require(mapped.shape == (1, *saved_shape) and np.isfinite(mapped).all() and
            np.isin(mapped, (0, 1)).all(), "Original nnU-Net segmentation resampler returned invalid mask")
    return mapped[0] == 1


def _raw_folder(source):
    return Path(source.raw_root).resolve() / source.base.name / "imagesTr"


def _raw_case_path(source, identifier):
    from experiments.genseg30.contract import valid_id
    valid_id(identifier)
    return checked_file(_raw_folder(source), identifier + "_0000.nii.gz")


def _target_case_paths(root, record):
    fields = ("mask_file", "native_mask_file", "input_las_file", "log_file",
              "command_started_file", "command_result_file")
    return {field: _safe_relative_file(root, record[field]) for field in fields}


def _hd_bet_runtime():
    version = importlib.metadata.version("hd-bet")
    require(version == RECIPE["hd_bet_version"], "HD-BET 2.0.1 is required")
    executable = shutil.which("hd-bet")
    require(executable, "HD-BET CLI executable is unavailable")
    from HD_BET.paths import folder_with_parameter_files
    weights = Path(folder_with_parameter_files).resolve() / "fold_all" / "checkpoint_final.pth"
    require(weights.is_file(), "HD-BET weights missing; automatic download is refused")
    return version, executable, weights, sha256(weights)


def _resource_snapshot(gpu_index):
    try:
        import psutil
        cpu = float(psutil.cpu_percent(interval=0.5))
        memory = psutil.virtual_memory()
        ram_used, ram_available = float(memory.percent), int(memory.available)
    except ImportError:
        if os.name == "nt":
            script = (
                "$c=(Get-Counter '\\Processor(_Total)\\% Processor Time' -SampleInterval 1 -MaxSamples 2)"
                ".CounterSamples | Measure-Object -Property CookedValue -Average;"
                "$o=Get-CimInstance Win32_OperatingSystem;"
                "[pscustomobject]@{cpu=$c.Average;total=$o.TotalVisibleMemorySize;free=$o.FreePhysicalMemory}|ConvertTo-Json -Compress"
            )
            result = subprocess.run(["powershell", "-NoProfile", "-Command", script],
                                    check=True, capture_output=True, text=True)
            values = json.loads(result.stdout)
            cpu = float(values["cpu"])
            total, free = int(values["total"]) * 1024, int(values["free"]) * 1024
            ram_available, ram_used = free, 100 * (1 - free / total)
        else:
            cpu = min(100.0, 100 * os.getloadavg()[0] / max(1, os.cpu_count() or 1))
            values = {}
            with open("/proc/meminfo", encoding="ascii") as stream:
                for line in stream:
                    key, raw, *_ = line.split()
                    values[key.rstrip(":")] = int(raw) * 1024
            ram_available = values["MemAvailable"]
            ram_used = 100 * (1 - ram_available / values["MemTotal"])
    command = ["nvidia-smi", "-i", str(gpu_index),
               "--query-gpu=uuid,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    row = next(csv.reader([result.stdout.strip()]))
    require(len(row) == 4, "Could not read selected physical GPU resources")
    total_vram, used_vram = int(row[3]) * 1024**2, int(row[2]) * 1024**2
    return {"cpu_pct": cpu, "ram_used_pct": ram_used, "ram_available": ram_available,
            "gpu_uuid": row[0].strip(), "gpu_util_pct": float(row[1]),
            "vram_available": total_vram - used_vram}


def bind_gpu(gpu_index, *, min_vram_gib=6, min_ram_gib=8):
    require(type(gpu_index) is int and gpu_index >= 0, "Physical nvidia-smi GPU index is required")
    snapshot = _resource_snapshot(gpu_index)
    for key in ("cpu_pct", "ram_used_pct", "gpu_util_pct"):
        require(snapshot[key] < 80.0, "Resource guard refused: " + key + " >= 80 percent")
    require(snapshot["vram_available"] >= min_vram_gib * 1024**3,
            "Resource guard refused: insufficient free VRAM")
    require(snapshot["ram_available"] >= min_ram_gib * 1024**3,
            "Resource guard refused: insufficient free RAM")
    os.environ["CUDA_VISIBLE_DEVICES"] = snapshot["gpu_uuid"]
    return snapshot


def _build_hd_bet_command(executable, input_file, output_file):
    output_file = Path(output_file)
    require(output_file.name.lower().endswith(".nii.gz"),
            "HD-BET single-file output argument must end in .nii.gz")
    return [
        str(executable), "-i", str(input_file), "-o", str(output_file),
        "-device", "cuda", "--save_bet_mask", "--no_bet_image",
    ]


def _run_hd_bet(command, *, case_dir, env, log_path):
    require("-o" in command, "HD-BET command has no single-file output argument")
    output_file = Path(command[command.index("-o") + 1])
    require(output_file.name.lower().endswith(".nii.gz"),
            "HD-BET single-file output argument must end in .nii.gz")
    mask_path = output_file.with_name(output_file.name[:-7] + "_bet.nii.gz")
    with Path(log_path).open("x", encoding="utf-8", newline="\n") as log:
        result = subprocess.run(command, cwd=case_dir, env=env, stdout=log,
                                stderr=subprocess.STDOUT, text=True, check=False)
    return result, mask_path

def _load_case_raw(source, identifier, plans_manager):
    raw_path = _raw_case_path(source, identifier)
    validate_nifti_mm_orthogonal(raw_path)
    reader = plans_manager.image_reader_writer_class()
    data, properties = reader.read_images([str(raw_path)])
    require(data.ndim == 4 and data.shape[0] == 1 and np.isfinite(data).all(),
            "Original plans reader did not return finite single-channel [1,Z,Y,X]")
    import SimpleITK as sitk
    image = sitk.ReadImage(str(raw_path))
    _check_sitk_geometry(image, context="Original raw DWI")
    require(tuple(data.shape[1:]) == tuple(reversed(image.GetSize())),
            "Original plans reader and physical raw grid disagree")
    return raw_path, data, properties, image


def prepare_targets(source, baseline_run, output, *, gpu_index):
    """Create a fresh train-only receipt; the completion manifest is written last."""
    require(source.dataset_class.__name__ == "nnUNetDatasetBlosc2",
            "Read-only source contract requires Blosc2; refusing NPZ unpack side effects")
    source.require_no_unpack_writes()
    gpu = bind_gpu(gpu_index)
    baseline_record, baseline_checkpoint = load_run(source, baseline_run)
    metadata = baseline_record["metadata"]
    baseline_training_kwargs(metadata, device="cuda:0")
    source.verify_inventory(metadata["inventory"])
    output = check_output(output, source.protected + [Path(baseline_run)], fresh=True)
    hd_version, hd_executable, weights, weights_hash = _hd_bet_runtime()
    require(hd_version == RECIPE["hd_bet_version"], "HD-BET runtime changed during preparation")
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    plans_manager = PlansManager(source.plans)
    configuration_manager = plans_manager.get_configuration("2d")
    train_dataset = source.training_dataset()
    source_inventory = {patient: source.inventory(patient) for patient in source.train}
    source.verify_inventory(metadata["inventory"])
    raw_paths = {patient: _raw_case_path(source, patient) for patient in source.train}
    raw_hashes = {patient: sha256(path) for patient, path in raw_paths.items()}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "started.json", {
        "classification": CLASSIFICATION, "complete": False, "started_utc": _now(),
        "source": source.identity, "baseline_checkpoint_sha256": baseline_record["checkpoint_sha256"],
        "recipe": RECIPE, "hd_bet_weights_sha256": weights_hash,
        "gpu": {"physical_index": gpu_index, "uuid": gpu["gpu_uuid"]},
    })
    cases = {}
    for index, patient in enumerate(source.train):
        case_gpu = bind_gpu(gpu_index)
        require(case_gpu["gpu_uuid"] == gpu["gpu_uuid"],
                "Selected physical GPU UUID changed during target preparation")
        case_dir = output / "cases" / patient
        case_dir.mkdir(parents=True, exist_ok=False)
        raw_path, _, _, raw_image = _load_case_raw(source, patient, plans_manager)
        require(sha256(raw_path) == raw_hashes[patient], "Raw DWI changed before HD-BET")
        import SimpleITK as sitk
        las_image = sitk.DICOMOrient(raw_image, "LAS")
        las_path = case_dir / "input_las.nii.gz"
        sitk.WriteImage(las_image, str(las_path), True)
        output_file = case_dir / "hd_bet.nii.gz"
        command = _build_hd_bet_command(hd_executable, las_path, output_file)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = case_gpu["gpu_uuid"]
        log_path = case_dir / "hd-bet.log"
        started_path = case_dir / "command_started.json"
        write_json(started_path, {
            "patient": patient, "status": "running", "command": command,
            "raw_file": str(raw_path), "raw_sha256": raw_hashes[patient],
            "input_las_file": str(las_path), "output_file": str(output_file),
            "gpu_uuid": case_gpu["gpu_uuid"], "default_tta": True,
        })
        result, mask_path = _run_hd_bet(command, case_dir=case_dir, env=env, log_path=log_path)
        result_path = case_dir / "command_result.json"
        has_mask = mask_path.is_file()
        write_json(result_path, {
            "patient": patient,
            "status": "command_completed" if result.returncode == 0 and has_mask else "command_failed",
            "return_code": result.returncode, "mask_file": str(mask_path),
            "mask_sha256": sha256(mask_path) if has_mask else None,
            "log_file": str(log_path), "log_sha256": sha256(log_path),
        })
        require(result.returncode == 0 and has_mask,
                "HD-BET failed for " + patient + "; retain the partial directory and logs")
        mask_image = sitk.ReadImage(str(mask_path))
        require(mask_image.GetSize() == las_image.GetSize() and
                np.allclose(mask_image.GetSpacing(), las_image.GetSpacing(), atol=1e-5, rtol=0) and
                np.allclose(mask_image.GetOrigin(), las_image.GetOrigin(), atol=1e-4, rtol=0) and
                np.allclose(mask_image.GetDirection(), las_image.GetDirection(), atol=1e-5, rtol=0),
                "HD-BET result does not match the LAS input grid")
        _, brain_raw = mask_image_to_raw_grid(mask_image, raw_image)
        native_keep = expand_keep_mask_3mm(brain_raw, spacing_xyz=raw_image.GetSpacing())
        image, label, properties = source.load_train(train_dataset, patient)
        require(label.shape == image.shape and properties.get("bbox_used_for_cropping") and
                properties.get("shape_before_cropping"),
                "Original preprocessed train arrays/crop properties are incomplete")
        mapped_keep = map_keep_mask_to_preprocessed(
            native_keep, plans_manager, configuration_manager, properties, expected_shape=image.shape
        )
        require(sha256(raw_path) == raw_hashes[patient], "Raw DWI changed during HD-BET for " + patient)
        mask_dir = output / "masks"
        mask_dir.mkdir(exist_ok=True)
        mask_file = mask_dir / (patient + ".npz")
        with mask_file.open("xb") as stream:
            np.savez_compressed(stream, keep=mapped_keep)
        cases[patient] = {
            "source_inventory": source_inventory[patient], "source_shape": list(image.shape),
            "raw_file": raw_path.name, "raw_sha256": raw_hashes[patient],
            "input_las_file": str(las_path.relative_to(output)), "input_las_sha256": sha256(las_path),
            "native_mask_file": str(mask_path.relative_to(output)), "native_mask_sha256": sha256(mask_path),
            "log_file": str(log_path.relative_to(output)), "log_sha256": sha256(log_path),
            "command_started_file": str(started_path.relative_to(output)),
            "command_started_sha256": sha256(started_path),
            "command_result_file": str(result_path.relative_to(output)),
            "command_result_sha256": sha256(result_path),
            "command": command, "return_code": result.returncode, "status": "complete",
            "mask_file": str(mask_file.relative_to(output)), "mask_sha256": sha256(mask_file),
            "mask_shape": list(mapped_keep.shape), "mask_voxels_kept": int(mapped_keep.sum()),
            "native_empty_plane_count": int(np.count_nonzero(~brain_raw.any(axis=(1, 2)))),
        }
        print(f"PREPARE {index + 1}/{len(source.train)}: {patient}", flush=True)
    source.verify_inventory(source_inventory)
    source.verify_inventory(metadata["inventory"])
    require(sha256(baseline_checkpoint) == baseline_record["checkpoint_sha256"],
            "Baseline generator changed during preparation")
    for patient, path in raw_paths.items():
        require(sha256(path) == raw_hashes[patient], "Raw DWI changed during preparation: " + patient)
    manifest = {
        "format": 1, "classification": CLASSIFICATION, "complete": True, "created_utc": _now(),
        "source": source.identity, "source_inventory": source_inventory,
        "baseline": {
            "run": str(Path(baseline_run).resolve()),
            "checkpoint_sha256": baseline_record["checkpoint_sha256"], "metadata": metadata,
        },
        "background_fill": float(metadata["scaling"]["min"]), "recipe": RECIPE,
        "hd_bet": {
            "version": hd_version, "weights_file": str(weights), "weights_sha256": weights_hash,
            "cli": hd_executable, "default_tta": True,
        },
        "gpu": {"physical_index": gpu_index, "uuid": gpu["gpu_uuid"]}, "cases": cases,
    }
    write_json(output / TARGET_MANIFEST, manifest)
    digest = sha256(output / TARGET_MANIFEST)
    print("TARGETS COMPLETE: manifest_sha256=" + digest, flush=True)
    return manifest


class TargetDataset:
    """Generator-only view that masks copied images and leaves segmentations untouched."""

    def __init__(self, source, target_root, baseline_run, *, expected_manifest_sha256):
        self.original = source
        self.target_root = check_output(target_root, source.protected + [Path(baseline_run)], fresh=False)
        self.manifest_path = checked_file(self.target_root, TARGET_MANIFEST)
        require(_is_sha256(expected_manifest_sha256), "An independently recorded target manifest SHA256 is required")
        digest = sha256(self.manifest_path)
        require(digest == expected_manifest_sha256, "Pinned target manifest SHA256 mismatch")
        manifest = read_json(self.manifest_path)
        require(manifest.get("format") == 1 and manifest.get("complete") is True and
                manifest.get("classification") == CLASSIFICATION,
                "Incomplete or foreign GenSeg3mm target manifest")
        require(manifest.get("recipe") == RECIPE, "GenSeg3mm target recipe mismatch")
        source.verify_identity(manifest["source"])
        require(set(manifest.get("cases", {})) == set(source.train),
                "Target receipt must contain every and only fixed training patient")
        baseline_record, checkpoint = load_run(source, baseline_run)
        metadata = baseline_record["metadata"]
        baseline_training_kwargs(metadata, device="cuda:0")
        baseline = manifest["baseline"]
        require(baseline.get("checkpoint_sha256") == baseline_record["checkpoint_sha256"] and
                baseline.get("metadata") == metadata and
                Path(baseline.get("run", "")).resolve() == Path(checkpoint).parent.resolve(),
                "Target receipt baseline generator/source mismatch")
        source.verify_inventory(metadata["inventory"])
        require(manifest.get("source_inventory") == metadata["inventory"],
                "Target receipt source inventory differs from baseline generator")
        require(manifest.get("background_fill") == metadata["scaling"]["min"],
                "Target fill differs from original generator training minimum")
        hd_bet = manifest.get("hd_bet", {})
        require(hd_bet.get("version") == RECIPE["hd_bet_version"] and
                hd_bet.get("default_tta") is True and _is_sha256(hd_bet.get("weights_sha256")),
                "HD-BET runtime/weight provenance is incomplete")
        self.baseline_record = baseline_record
        self.baseline_checkpoint = Path(checkpoint).resolve()
        self.baseline_run = Path(baseline_run).resolve()
        self.manifest = manifest
        self.manifest_sha256 = digest
        self.fill = float(manifest["background_fill"])
        self.train = list(source.train)
        self.val = list(source.val)
        self._case_paths = {}
        for patient in self.train:
            record = manifest["cases"][patient]
            require(record.get("status") == "complete" and record.get("return_code") == 0,
                    "Target case is incomplete: " + patient)
            require(record.get("source_inventory") == source.inventory(patient),
                    "Target case original source inventory mismatch: " + patient)
            raw_path = _raw_case_path(source, patient)
            require(record.get("raw_file") == raw_path.name and record.get("raw_sha256") == sha256(raw_path),
                    "Original raw DWI changed since target preparation: " + patient)
            paths = _target_case_paths(self.target_root, record)
            for field, hash_field in (
                ("mask_file", "mask_sha256"), ("native_mask_file", "native_mask_sha256"),
                ("input_las_file", "input_las_sha256"), ("log_file", "log_sha256"),
                ("command_started_file", "command_started_sha256"),
                ("command_result_file", "command_result_sha256"),
            ):
                require(_is_sha256(record.get(hash_field)) and sha256(paths[field]) == record[hash_field],
                        "Target artifact hash mismatch: " + patient + "/" + field)
            require(isinstance(record.get("command"), list) and record["command"] and
                    record["command"][0] == hd_bet["cli"] and record["command"][-1] == "--no_bet_image",
                    "Target case command receipt is incomplete: " + patient)
            started = read_json(paths["command_started_file"])
            result = read_json(paths["command_result_file"])
            require(started.get("patient") == patient and started.get("status") == "running" and
                    started.get("command") == record["command"] and
                    started.get("raw_sha256") == record["raw_sha256"] and
                    result.get("patient") == patient and result.get("status") == "command_completed" and
                    result.get("return_code") == 0 and
                    result.get("mask_sha256") == record["native_mask_sha256"] and
                    result.get("log_sha256") == record["log_sha256"],
                    "Per-case HD-BET command/result receipt mismatch: " + patient)
            require(record.get("source_shape") == metadata["cases"][patient]["shape"] and
                    record.get("mask_shape") == metadata["cases"][patient]["shape"][1:],
                    "Target mask does not match baseline preprocessed grid: " + patient)
            with np.load(paths["mask_file"], allow_pickle=False) as arrays:
                require(set(arrays.files) == {"keep"}, "Unexpected target mask cache fields")
                keep = np.asarray(arrays["keep"])
            require(keep.dtype == np.bool_ and list(keep.shape) == record["mask_shape"] and keep.any(),
                    "Target retention mask has invalid values or shape: " + patient)
            self._case_paths[patient] = paths
        self.original.verify_identity(manifest["source"])
        self.original.verify_inventory(metadata["inventory"])
        self.identity = dict(source.identity)
        self.identity["genseg3mm"] = {
            "classification": CLASSIFICATION, "target_manifest_sha256": digest,
            "baseline_checkpoint_sha256": baseline_record["checkpoint_sha256"], "recipe": RECIPE,
        }
        self.protected = list(source.protected) + [
            self.target_root, self.baseline_run, self.baseline_checkpoint.parent,
        ]
        self._source_inventory = manifest["source_inventory"]
        self._identities = self._capture_paths()
        assert_identities(self._identities)

    @property
    def folder(self):
        return self.original.folder

    @property
    def base(self):
        return self.original.base

    @property
    def raw_root(self):
        return self.original.raw_root

    @property
    def dataset_json(self):
        return self.original.dataset_json

    @property
    def plans(self):
        return self.original.plans

    @property
    def configuration(self):
        return self.original.configuration

    @property
    def dataset_class(self):
        return self.original.dataset_class

    def _capture_paths(self):
        paths = [
            self.target_root, self.manifest_path, self.baseline_checkpoint,
            self.baseline_checkpoint.parent / "generator.json",
            self.original.folder, self.original.base, self.original.dataset_path,
            self.original.plans_path, self.original.splits,
        ]
        from experiments.genseg30 import contract
        paths.append(contract.REFERENCE)
        for patient in self.train:
            paths.extend(self.original.case_files(patient))
            paths.append(_raw_case_path(self.original, patient))
            paths.extend(self._case_paths[patient].values())
        return capture_identities([path for path in paths if Path(path).exists()])

    def _paths_for(self, patient):
        global_paths = {
            str(self.target_root), str(self.manifest_path), str(self.baseline_checkpoint),
            str(self.baseline_checkpoint.parent / "generator.json"),
            str(self.original.folder), str(self.original.base),
            str(self.original.dataset_path), str(self.original.plans_path), str(self.original.splits),
        }
        from experiments.genseg30 import contract
        global_paths.add(str(contract.REFERENCE))
        paths = [path for path in self._identities if path in global_paths]
        paths.extend(str(path) for path in self.original.case_files(patient))
        paths.append(str(_raw_case_path(self.original, patient)))
        paths.extend(str(path) for path in self._case_paths[patient].values())
        return [path for path in paths if path in self._identities]

    def inventory(self, identifier):
        require(identifier in self.train, "Only original training cases may enter the generator")
        record = self.manifest["cases"][identifier]
        return {
            "source": self.original.inventory(identifier),
            "raw_sha256": record["raw_sha256"],
            "target_manifest_sha256": self.manifest_sha256,
            "mask_file": record["mask_file"],
            "mask_sha256": record["mask_sha256"],
        }

    def verify_identity(self, identity):
        require(identity == self.identity, "Original source or GenSeg3mm target identity mismatch")
        self.original.verify_identity(self.manifest["source"])
        assert_identities(self._identities)

    def verify_inventory(self, inventory):
        require(set(inventory) == set(self.train), "Incomplete/foreign GenSeg3mm training inventory")
        for patient in self.train:
            require(inventory[patient] == self.inventory(patient),
                    "GenSeg3mm training inventory changed: " + patient)
        self.original.verify_inventory(self._source_inventory)
        assert_identities(self._identities)

    def training_dataset(self):
        return self.original.training_dataset()

    def load_train(self, dataset, identifier):
        require(identifier in self.train and set(dataset.identifiers) == set(self.train),
                "Only fixed training patients may be read through the target adapter")
        assert_identities(self._identities, self._paths_for(identifier))
        image, seg, properties = self.original.load_train(dataset, identifier)
        record = self.manifest["cases"][identifier]
        require(list(image.shape) == record["source_shape"] and list(seg.shape) == record["source_shape"],
                "Original train arrays changed shape after target preparation")
        with np.load(self._case_paths[identifier]["mask_file"], allow_pickle=False) as arrays:
            keep = np.asarray(arrays["keep"])
        require(keep.dtype == np.bool_ and tuple(keep.shape) == image.shape[1:],
                "Cached target mask no longer matches source arrays")
        image = np.array(image, copy=True)
        image[0, ~keep] = self.fill
        assert_identities(self._identities, self._paths_for(identifier))
        return image, seg, properties

    def require_no_unpack_writes(self):
        return self.original.require_no_unpack_writes()





