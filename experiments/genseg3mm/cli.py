"""Exploratory generator-only 3 mm comparison: prepare -> train -> preview."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess

import numpy as np

from experiments.genseg30.contract import Source, check_output, require, sha256, write_json
from experiments.genseg30.models import Generator, pad_plane
from experiments.genseg30.pipeline import load_run, train_generator
from .targets import (
    CLASSIFICATION, TARGET_MANIFEST, TargetDataset, baseline_training_kwargs,
    bind_gpu, prepare_targets,
)


def parser():
    root = argparse.ArgumentParser(
        description="Train-only GenSeg 3 mm target preview; not a segmentation pool"
    )
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("prepare", "train", "preview"):
        sub = commands.add_parser(name)
        sub.add_argument("--source", type=Path, required=True,
                         help="Original read-only Dataset501 nnUNetPlans_2d folder")
        sub.add_argument("--raw-root", type=Path, required=True, help="Configured read-only nnUNet_raw root")
        sub.add_argument("--baseline-run", type=Path, required=True, help="Completed original GenSeg30 run")
        sub.add_argument("--gpu", type=int, required=True, help="Physical nvidia-smi GPU index, bound by UUID")
        if name == "prepare":
            sub.add_argument("--output", type=Path, required=True, help="Fresh external target directory")
            sub.add_argument("--check-only", action="store_true",
                             help="Check paths, baseline, HD-BET weights and resources without processing data")
        else:
            sub.add_argument("--targets", type=Path, required=True, help="Completed prepared target receipt")
            sub.add_argument("--targets-sha256", required=True,
                             help="Manifest SHA256 printed by prepare; pins immutable targets")
            if name == "train":
                sub.add_argument("--output", type=Path, required=True, help="Fresh external generator run")
            else:
                sub.add_argument("--run", type=Path, required=True, help="Completed GenSeg3mm generator run")
                sub.add_argument("--output", type=Path, required=True, help="Fresh external preview directory")
    return root


def select_preview_slices(train_ids, labels_by_patient, *, patient_limit=4):
    """Choose up to two valid no-lesion planes and the first lesion plane per train case."""
    require(type(patient_limit) is int and patient_limit > 0, "Positive preview patient limit required")
    selected = []
    for patient in list(train_ids)[:patient_limit]:
        require(patient in labels_by_patient, "Missing training-only labels for preview selection")
        seg = np.asarray(labels_by_patient[patient])
        require(seg.ndim == 4 and seg.shape[0] == 1, "Preview labels must be [1,Z,Y,X]")
        positive_z, background_z = [], []
        for z in range(seg.shape[1]):
            plane = seg[0, z]
            if np.any(plane == 1):
                positive_z.append(z)
            elif np.any(plane == 0):
                background_z.append(z)
        selected.extend({"patient": patient, "z": int(z), "has_lesion": False}
                        for z in background_z[:2])
        if positive_z:
            selected.append({"patient": patient, "z": int(positive_z[0]), "has_lesion": True})
    require(bool(selected), "No valid training-only preview planes")
    return selected


def _render_panels(path, panels, label, scaling):
    from PIL import Image, ImageDraw

    require(len(panels) == 5, "Preview must show source, target, old, new and overlay panels")
    arrays = [(name, np.asarray(image, dtype=np.float32)) for name, image in panels]
    shapes = {image.shape for _, image in arrays}
    require(len(shapes) == 1 and len(next(iter(shapes))) == 2 and
            all(np.isfinite(image).all() for _, image in arrays),
            "Preview panels must share a finite 2d grid")
    label = np.asarray(label)
    require(label.shape == next(iter(shapes)), "Preview segmentation shape mismatch")
    lo, hi = float(scaling["min"]), float(scaling["max"])
    require(np.isfinite(lo) and np.isfinite(hi) and hi > lo, "Invalid display scaling")
    rendered = []
    for name, image in arrays:
        pixels = np.clip((image - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        rgb = np.repeat(pixels[:, :, None], 3, axis=2)
        if name == "new + lesion overlay":
            rgb[label == 1] = (0.4 * rgb[label == 1] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)
            rgb[label == -1] = [255, 255, 0]
        rendered.append((name, Image.fromarray(rgb)))
    width = sum(image.width for _, image in rendered)
    height = max(image.height for _, image in rendered) + 22
    canvas = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(canvas)
    x = 0
    for name, image in rendered:
        canvas.paste(image, (x, 22))
        draw.text((x + 2, 4), name, fill="white")
        x += image.width
    with Path(path).open("xb") as stream:
        canvas.save(stream, format="PNG")


def _restore_generator(record, checkpoint, device):
    generator, metadata = Generator.load(checkpoint, device)
    require(metadata == record["metadata"], "Generator checkpoint/sidecar metadata mismatch")
    generator.net.eval()
    return generator


def _generate(generator, label, scaling):
    import torch

    multiple = 2 ** generator.config["downs"]
    condition = pad_plane((label == 1).astype(np.float32), multiple, 0)
    with torch.no_grad():
        result = generator.net(torch.from_numpy(condition[None, None]).to(generator.device))
    fake = result.detach().cpu().numpy()[0, 0]
    require(fake.shape == condition.shape and np.isfinite(fake).all(), "Generator output is invalid")
    lo, hi = float(scaling["min"]), float(scaling["max"])
    return ((fake[:label.shape[-2], :label.shape[-1]] + 1) * (hi - lo) / 2 + lo).astype(np.float32)


def preview(source, baseline_run, target_root, expected_target_sha256, run, output, *, gpu_index):
    bind_gpu(gpu_index)
    baseline_record, baseline_checkpoint = load_run(source, baseline_run)
    targets = TargetDataset(source, target_root, baseline_run,
                            expected_manifest_sha256=expected_target_sha256)
    run_record, run_checkpoint = load_run(targets, run)
    require(run_record["metadata"]["actual_steps"] == 5000 and
            run_record["metadata"]["settings"] == baseline_record["metadata"]["settings"],
            "New generator differs from the complete original training budget/settings")
    targets.verify_inventory(run_record["metadata"]["inventory"])
    output = check_output(output, targets.protected + [Path(run)], fresh=True)
    gpu = bind_gpu(gpu_index)
    device = "cuda:0"
    old_generator = _restore_generator(baseline_record, baseline_checkpoint, device)
    new_generator = _restore_generator(run_record, run_checkpoint, device)
    output.mkdir(parents=True, exist_ok=False)
    original_dataset = source.training_dataset()
    cases = {}
    for patient in source.train[:4]:
        original_image, seg, _ = source.load_train(original_dataset, patient)
        cases[patient] = {"source": original_image, "seg": seg}
    selected = select_preview_slices(source.train, {p: case["seg"] for p, case in cases.items()})
    target_dataset = targets.training_dataset()
    for patient, case in cases.items():
        target_image, target_seg, _ = targets.load_train(target_dataset, patient)
        require(np.array_equal(case["seg"], target_seg), "Target adapter altered preview segmentation")
        case["target"] = target_image
    files = {}
    for selection in selected:
        patient, z = selection["patient"], selection["z"]
        original_image, seg = cases[patient]["source"], cases[patient]["seg"]
        target_image = cases[patient]["target"]
        label = seg[0, z].copy()
        source_plane, target_plane = original_image[0, z].copy(), target_image[0, z].copy()
        old_fake = _generate(old_generator, label, baseline_record["metadata"]["scaling"])
        new_fake = _generate(new_generator, label, run_record["metadata"]["scaling"])
        old_fake[label == -1] = source_plane[label == -1]
        new_fake[label == -1] = target_plane[label == -1]
        name = f"{patient}_z{z:04d}.png"
        _render_panels(output / name, [
            ("source original", source_plane), ("target 3 mm", target_plane),
            ("old generator", old_fake), ("new generator", new_fake),
            ("new + lesion overlay", new_fake),
        ], label, baseline_record["metadata"]["scaling"])
        files[name] = sha256(output / name)
    targets.verify_inventory(run_record["metadata"]["inventory"])
    manifest = {
        "format": 1, "classification": CLASSIFICATION, "complete": True,
        "created_utc": datetime.now(timezone.utc).isoformat(), "source": source.identity,
        "target_manifest_sha256": targets.manifest_sha256,
        "baseline_checkpoint_sha256": baseline_record["checkpoint_sha256"],
        "new_checkpoint_sha256": run_record["checkpoint_sha256"],
        "gpu": {"physical_index": gpu_index, "uuid": gpu["gpu_uuid"]},
        "selection_rule": "first4_train_patients_up_to2_distinct_background_z_plus_first_lesion_z",
        "selection": selected,
        "panels": ["source original", "target 3 mm", "old generator", "new generator", "new + lesion overlay"],
        "files": files,
        "interpretation": "Exploratory image preview only; no heldout data, QA approval, segmentation pool or benefit claim.",
    }
    write_json(output / "preview.json", manifest)
    print("PREVIEW COMPLETE: " + str(len(files)) + " training-only images", flush=True)
    return manifest


def main(argv=None):
    root = parser()
    args = root.parse_args(argv)
    try:
        source = Source(args.source, args.raw_root)
        if args.command == "prepare":
            if args.check_only:
                from .targets import _hd_bet_runtime, _raw_case_path
                source.require_no_unpack_writes()
                baseline_record, _ = load_run(source, args.baseline_run)
                baseline_training_kwargs(baseline_record["metadata"], device="cuda:0")
                bind_gpu(args.gpu)
                for patient in source.train:
                    _raw_case_path(source, patient)
                _hd_bet_runtime()
                check_output(args.output, source.protected + [args.baseline_run], fresh=True)
                print("CHECK PASS: source, baseline, training raw paths, HD-BET weights and resources; no data processed")
                return
            prepare_targets(source, args.baseline_run, args.output, gpu_index=args.gpu)
        elif args.command == "train":
            source.require_no_unpack_writes()
            baseline_record, _ = load_run(source, args.baseline_run)
            kwargs = baseline_training_kwargs(baseline_record["metadata"], device="cuda:0")
            bind_gpu(args.gpu)
            targets = TargetDataset(source, args.targets, args.baseline_run,
                                    expected_manifest_sha256=args.targets_sha256)
            output = check_output(args.output, targets.protected, fresh=True)
            bind_gpu(args.gpu)
            result = train_generator(targets, output, **kwargs)
            require(result["actual_steps"] == 5000, "GenSeg3mm run did not complete all 5000 updates")
        else:
            preview(source, args.baseline_run, args.targets, args.targets_sha256,
                    args.run, args.output, gpu_index=args.gpu)
    except (ValueError, OSError, KeyError, TypeError, ImportError, RuntimeError,
            subprocess.SubprocessError) as exc:
        root.exit(2, "GenSeg3mm refused: " + str(exc) + "\n")


if __name__ == "__main__":
    main()

