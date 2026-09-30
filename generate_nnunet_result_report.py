"""Single-fold Dataset501 nnU-Net validation report from existing masks/metrics.

Feature intensity is a diagnostic recomputation from raw images, never a replay
claim about the saved validation masks. No labels enter preprocessing or forward.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile

from report_metrics_table import (metric_value, count_coverage, binary_labels, build_table, export_table)
from pathlib import Path


TRAINER = "nnUNetTrainerUPerNetTopK10EarlyStopping"
ORIGINAL_TRAINER = "nnUNetTrainerTopK10"
SUPPORTED_TRAINERS = (TRAINER, ORIGINAL_TRAINER)
METRICS = ("dice", "iou", "f2", "avd_percent", "lcd", "recall", "hd95_mm")


def resolved(path):
    return Path(path).resolve(strict=False)


def protected_output(output, sources):
    out = resolved(output)
    if out.exists():
        raise ValueError(f"output directory already exists: {out}")
    for source in sources:
        src = resolved(source)
        if out == src or src in out.parents or out in src.parents:
            raise ValueError(f"output overlaps source: {out} / {src}")
    return out


def collect(directory, *, input_channel=False):
    paths = sorted([*directory.glob("*.nii"), *directory.glob("*.nii.gz")])
    result = {}
    for path in paths:
        stem = path.name[:-7] if path.name.endswith(".nii.gz") else path.stem
        if input_channel:
            if not stem.endswith("_0000"):
                raise ValueError(f"Dataset501 DWI input must end in _0000: {path}")
            stem = stem[:-5]
        if stem in result:
            raise ValueError(f"duplicate case ID {stem}: {result[stem]}, {path}")
        result[stem] = path
    if not result:
        raise ValueError(f"no NIfTI files in {directory}")
    return result


def read_metrics(metrics_dir, ids):
    case_file = metrics_dir / "case_metrics.csv"
    summary_file = metrics_dir / "summary_metrics.json"
    if not case_file.is_file() or not summary_file.is_file():
        raise ValueError("existing case_metrics.csv and summary_metrics.json are required")
    with case_file.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"case_id", *METRICS}.issubset(reader.fieldnames or []):
            raise ValueError("case metrics columns missing")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("duplicate case metrics columns")
        rows = {}
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("malformed case metrics CSV row")
            case_id = row["case_id"]
            if not case_id or case_id in rows:
                raise ValueError(f"missing or duplicate metric case ID: {case_id!r}")
            try:
                for key in METRICS:
                    metric_value(row[key])
                dice = metric_value(row["dice"])
            except (ValueError, TypeError) as error:
                raise ValueError(f"invalid numeric metric for {case_id}: {error}") from error
            if dice is not None and not 0 <= dice <= 1:
                raise ValueError(f"invalid Dice for {case_id}: {dice}")
            rows[case_id] = row
    if set(rows) != set(ids):
        raise ValueError(f"metric coverage mismatch: missing={sorted(set(ids)-set(rows))}, extra={sorted(set(rows)-set(ids))}")
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    if summary.get("n_cases") != len(rows) or not isinstance(summary.get("metrics"), dict):
        raise ValueError("summary n_cases/metrics inconsistent with case metrics")
    for key in METRICS:
        item = summary["metrics"].get(key)
        if not isinstance(item, dict) or not 0 <= item.get("valid_cases", -1) <= len(rows):
            raise ValueError(f"summary metric coverage invalid: {key}")
        observed = sum(1 for row in rows.values() if _finite_or_missing(row[key]) is not None)
        if item["valid_cases"] != observed:
            raise ValueError(f"summary valid_cases mismatch for {key}: {item['valid_cases']} != {observed}")
    count_coverage(rows)  # metadata-only strict count parsing; no voxel reads
    return rows, summary


def _finite_or_missing(value):
    return metric_value(value)


def select_cases(rows):
    finite = {cid: row for cid, row in rows.items() if metric_value(row["dice"]) is not None}
    if len(finite) < 6:
        raise ValueError("at least six distinct validation cases with finite Dice are required")
    high = sorted(finite, key=lambda cid: (-float(finite[cid]["dice"]), cid))[:3]
    remaining = set(finite) - set(high)
    low = sorted(remaining, key=lambda cid: (float(finite[cid]["dice"]), cid))[:3]
    return high, low


def select_slices(mask):
    import numpy as np
    if mask.ndim != 3:
        raise ValueError("expected a three-dimensional original-space mask")
    area = np.count_nonzero(mask, axis=(1, 2))
    return sorted((int(i) for i in np.flatnonzero(area)), key=lambda i: (-int(area[i]), i))[:3]


def check_geometry(image, other, case_id):
    import numpy as np
    for key in ("Size", "Spacing", "Origin", "Direction"):
        a = getattr(image, "Get" + key)()
        b = getattr(other, "Get" + key)()
        if key == "Size":
            same = a == b
        else:
            same = bool(np.allclose(a, b, rtol=1e-5, atol=1e-5))
        if not same:
            raise ValueError(f"{case_id}: {key} mismatch: {a} != {b}")
    if image.GetDimension() != 3:
        raise ValueError(f"{case_id}: only 3D original volumes supported")


def _check_identity(model_dir, checkpoint, fold):
    trainer = model_dir.name.split("__")[0]
    if fold != 0 or trainer not in SUPPORTED_TRAINERS or model_dir.name != f"{trainer}__nnUNetPlans__2d" or model_dir.parent.name != "Dataset501_StrokeLesion":
        raise ValueError("only exact Dataset501 official TopK10/UPerNet TopK10 fold 0 2D model directories are supported")
    dataset = json.loads((model_dir / "dataset.json").read_text(encoding="utf-8"))
    plans = json.loads((model_dir / "plans.json").read_text(encoding="utf-8"))
    channels = dataset.get("channel_names")
    if not isinstance(channels, dict) or len(channels) != 1 or not isinstance(next(iter(channels.values())), str) or next(iter(channels.values())) != "DWI":
        raise ValueError("only one-channel DWI Dataset501 is supported")
    if "2d" not in plans.get("configurations", {}):
        raise ValueError("2d configuration missing")
    require_simpleitk_reader(plans)
    cp = resolved(checkpoint)
    if cp.parent != resolved(model_dir / "fold_0") or cp.name not in ("checkpoint_best.pth", "checkpoint_final.pth") or not cp.is_file():
        raise ValueError("checkpoint must explicitly identify existing fold_0 best/final file")
    return dataset, plans, cp


def require_simpleitk_reader(plans):
    reader = plans.get("image_reader_writer")
    if reader != "SimpleITKIO":
        raise ValueError(f"unsupported image_reader_writer {reader!r}; only exact SimpleITKIO 3D array axes are verified")


def verify_full_geometry(info):
    """Read headers for every metrics-covered case before any report output."""
    import SimpleITK as sitk
    for cid in sorted(info["rows"]):
        try:
            headers = []
            for name in ("images", "labels", "predictions"):
                reader = sitk.ImageFileReader()
                reader.SetFileName(str(info[name][cid]))
                reader.ReadImageInformation()
                headers.append(reader)
            check_geometry(headers[0], headers[1], cid)
            check_geometry(headers[0], headers[2], cid)
        except Exception as error:
            raise ValueError(f"{cid}: full-set geometry verification failed: {error}") from error
    return len(info["rows"])


def _hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_sources(args):
    model = resolved(args.model_dir)
    images = resolved(args.images_dir)
    labels = resolved(args.labels_dir)
    preds = resolved(args.prediction_dir or model / "fold_0" / "validation")
    if args.metrics_dir is not None:
        metrics = resolved(args.metrics_dir)
    else:
        candidates = [model / "fold_0" / name for name in ("multi_metrics", "multi_metric_evaluation")
                      if (model / "fold_0" / name).is_dir()]
        if len(candidates) != 1:
            raise ValueError(f"expected one metrics directory; found {candidates}; pass --metrics-dir explicitly")
        metrics = resolved(candidates[0])
    checkpoint = resolved(args.checkpoint)
    output = protected_output(args.output_dir, (model, images, labels, preds, metrics, checkpoint))
    dataset, plans, checkpoint = _check_identity(model, checkpoint, args.fold)
    if not str(dataset.get("name", "")).startswith("Dataset501") and "Dataset501" not in str(model.parent):
        raise ValueError("Dataset501 identity not established by model directory or dataset.json")
    image_map = collect(images, input_channel=True)
    label_map = collect(labels)
    pred_map = collect(preds)
    if set(image_map) != set(label_map) or set(pred_map) != set(label_map):
        raise ValueError(f"case coverage mismatch: images={sorted(image_map)}, labels={sorted(label_map)}, predictions={sorted(pred_map)}")
    rows, summary = read_metrics(metrics, pred_map)
    return dict(model=model, images=image_map, labels=label_map, predictions=pred_map,
                metrics=metrics, checkpoint=checkpoint, output=output, dataset=dataset,
                plans=plans, rows=rows, summary=summary)


def _add_window(sum_map, weight_map, value, weight, sl):
    """Add one feature patch to its actual image-window coordinates."""
    import numpy as np
    y, x = sl
    if value.shape != (y.stop-y.start, x.stop-x.start):
        raise ValueError("feature patch/window mismatch")
    sum_map[y, x] += np.asarray(value, dtype=np.float32) * weight
    weight_map[y, x] += weight


def _inverse_feature(preprocessed_map, properties, plans_manager, config):
    """Mirror nnU-Net export order: resample, uncrop, transpose backward."""
    import numpy as np
    source_shape = tuple(int(x) for x in properties["shape_after_cropping_and_before_resampling"])
    source_spacing = [properties["spacing"][i] for i in plans_manager.transpose_forward]
    current_spacing = list(config.spacing) if len(config.spacing) == 3 else [source_spacing[0], *config.spacing]
    restored = config.resampling_fn_probabilities(preprocessed_map[None], source_shape,
                                                    current_spacing, source_spacing)
    if hasattr(restored, "detach"):
        restored = restored.detach().cpu().numpy()
    restored = np.asarray(restored)[0]
    if restored.shape != source_shape:
        raise ValueError("inverse resampling returned unexpected shape")
    full = np.zeros(tuple(properties["shape_before_cropping"]), dtype=np.float32)
    bbox = properties["bbox_used_for_cropping"]
    dest = tuple(slice(int(a), int(b)) for a, b in bbox)
    if full[dest].shape != restored.shape:
        raise ValueError("crop geometry mismatch")
    full[dest] = restored
    return full.transpose(plans_manager.transpose_backward)


def _native_slice_index(original_index, properties, plans_manager, preprocessed_shape):
    """Map SimpleITK array-axis-0 to a preprocessed 2D slice, or refuse ambiguity."""
    forward = tuple(int(v) for v in plans_manager.transpose_forward)
    backward = tuple(int(v) for v in plans_manager.transpose_backward)
    if forward != (0, 1, 2) or backward != (0, 1, 2):
        raise ValueError("native channel window location requires identity axis transpose")
    bbox = properties["bbox_used_for_cropping"]
    crop_z = int(bbox[0][1]) - int(bbox[0][0])
    source_z = int(properties["shape_after_cropping_and_before_resampling"][0])
    if crop_z != source_z or source_z != int(preprocessed_shape[0]):
        raise ValueError("native channel window location requires unchanged slice-axis size")
    index = int(original_index) - int(bbox[0][0])
    if not 0 <= index < int(preprocessed_shape[0]):
        raise ValueError("selected original slice is outside preprocessed crop")
    return index


def _channel_ids(count, limit=8):
    import numpy as np
    if count < 1:
        raise ValueError("encoder feature has no channels")
    return np.linspace(0, count-1, min(limit, count), dtype=int).tolist()


def _diagnostic_feature(raw_path, predictor, stage, selected_slice=None, intermediate_stage=3):
    """One GT-free diagnostic pass; native channels come from one local window."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from acvl_utils.cropping_and_padding.padding import pad_nd_image
    from nnunetv2.inference.sliding_window_prediction import compute_gaussian

    pre = predictor.configuration_manager.preprocessor_class(verbose=False)
    data, _, properties = pre.run_case([str(raw_path)], None, predictor.plans_manager,
                                       predictor.configuration_manager, predictor.dataset_json)
    if data.ndim != 4 or data.shape[0] != 1:
        raise ValueError("only one-channel 2D volume preprocessing supported")
    target_z = None
    if selected_slice is not None:
        target_z = _native_slice_index(selected_slice, properties, predictor.plans_manager,
                                       data.shape[1:])
    stages = predictor.network.encoder.stages
    if intermediate_stage == stage or intermediate_stage < 0 or intermediate_stage >= len(stages):
        raise ValueError(f"intermediate encoder stage {intermediate_stage} unavailable or equals final stage {stage}")
    captured = {}
    handles = []
    window_index = None
    channels = predictor.network.encoder.output_channels
    def shape_of(value):
        return tuple(value.shape) if isinstance(value, torch.Tensor) else f"non-Tensor {type(value).__name__}"

    def expected(index):
        size = ", H=W=64" if index == intermediate_stage else ", H>0, W>0"
        return f"Tensor BCHW with B=1, C={channels[index]}{size}"

    def check_feature(index, value):
        shape = shape_of(value)
        if (not isinstance(value, torch.Tensor) or value.ndim != 4
                or value.shape[0] != 1 or value.shape[1] != channels[index]
                or value.shape[2] < 1 or value.shape[3] < 1
                or (index == intermediate_stage and tuple(value.shape[-2:]) != (64, 64))):
            raise ValueError(f"encoder.stages.{index} window {window_index}: expected {expected(index)}, got {shape}")

    def make_hook(index):
        def hook(_module, _inputs, output):
            if index in captured:
                raise ValueError(f"encoder.stages.{index} window {window_index}: expected one output {expected(index)}, got duplicate {shape_of(output)}; first {shape_of(captured[index])}")
            captured[index] = output.detach() if isinstance(output, torch.Tensor) else output
        return hook
    try:
        for index in (intermediate_stage, stage):
            handles.append(stages[index].register_forward_hook(make_hook(index)))
        predictor.network.eval().to(predictor.device)
        padded, undo = pad_nd_image(torch.from_numpy(np.ascontiguousarray(data)),
                                    predictor.configuration_manager.patch_size,
                                    "constant", {"value": 0}, True, None)
        z, h, w = padded.shape[1:]
        total = np.zeros((z, h, w), dtype=np.float32)
        weights = np.zeros_like(total)
        gaussian = compute_gaussian(tuple(predictor.configuration_manager.patch_size),
                                    sigma_scale=1/8, value_scaling_factor=10,
                                    device=torch.device("cpu")).float().numpy()
        shapes = set()
        window_count = 0
        native = None
        intermediate_native = None
        intermediate_shapes = set()
        # undo indexes the padded tensor; its start places the original preprocessed data.
        offset_z = int(undo[1].start or 0)
        offset_y = int(undo[2].start or 0)
        offset_x = int(undo[3].start or 0)
        target_padded_z = None if target_z is None else target_z + offset_z
        center_y = offset_y + data.shape[2] // 2
        center_x = offset_x + data.shape[3] // 2
        with torch.inference_mode():
            for window_index, sl in enumerate(predictor._internal_get_sliding_window_slicers((z, h, w))):
                if len(sl) != 4 or not isinstance(sl[1], int):
                    raise ValueError("expected 2D slice window")
                window_count += 1
                patch = padded[sl][None].to(predictor.device)
                captured.clear()
                predictor.network(patch)
                for index in (intermediate_stage, stage):
                    if index not in captured:
                        raise ValueError(f"encoder.stages.{index} window {window_index}: expected one output {expected(index)}, got missing")
                    check_feature(index, captured[index])
                intermediate = captured.pop(intermediate_stage)
                feature = captured.pop(stage)
                intermediate_shapes.add(tuple(intermediate.shape))
                shapes.add(tuple(feature.shape))
                if (native is None and target_padded_z == sl[1]
                        and sl[2].start <= center_y < sl[2].stop
                        and sl[3].start <= center_x < sl[3].stop):
                    ids = _channel_ids(int(feature.shape[1]))
                    native = dict(channels=feature[0, ids].float().cpu().numpy().copy(),
                                  channel_ids=ids, feature_shape=list(feature.shape),
                                  original_slice=int(selected_slice),
                                  preprocessed_slice=target_z,
                                  window_index=window_index,
                                  window_selection="first enumerated window containing preprocessed slice center",
                                  window_yx_preprocessed=[
                                      [int(sl[2].start-offset_y), int(sl[2].stop-offset_y)],
                                      [int(sl[3].start-offset_x), int(sl[3].stop-offset_x)]],
                                  window_yx_padded=[
                                      [int(sl[2].start), int(sl[2].stop)],
                                      [int(sl[3].start), int(sl[3].stop)]])
                    intermediate_ids = _channel_ids(int(intermediate.shape[1]))
                    intermediate_native = {**{k: v for k, v in native.items() if k != "channels"},
                        "channels": intermediate[0, intermediate_ids].float().cpu().numpy().copy(),
                        "channel_ids": intermediate_ids,
                        "feature_shape": list(intermediate.shape),
                        "stage": intermediate_stage,
                        "module": f"encoder.stages.{intermediate_stage}"}
                    native["stage"] = stage
                    native["module"] = f"encoder.stages.{stage}"
                intensity = feature.abs().mean(dim=1, keepdim=True)
                intensity = F.interpolate(intensity, size=tuple(predictor.configuration_manager.patch_size),
                                          mode="bilinear", align_corners=False)[0, 0].float().cpu().numpy()
                _add_window(total[sl[1]], weights[sl[1]], intensity, gaussian, sl[2:])
                del intermediate, feature, intensity, patch
        if target_z is not None and (native is None or intermediate_native is None):
            raise ValueError("no diagnostic window contains selected preprocessed slice center")
        if np.any(weights <= 0):
            raise ValueError("uncovered preprocessed feature pixel")
        map_pre = (total / weights)[undo[1:]]
        if map_pre.shape != data.shape[1:]:
            raise ValueError("padding removal mismatch")
        original = _inverse_feature(map_pre, properties, predictor.plans_manager,
                                    predictor.configuration_manager)
        provenance = dict(stage=stage, module=f"encoder.stages.{stage}",
                          feature_shapes=sorted(shapes), windows=window_count,
                          covered_preprocessed_voxels=int(np.count_nonzero(weights)),
                          native_window={k: v for k, v in native.items() if k != "channels"} if native else None,
                          intermediate_stage=intermediate_stage,
                          intermediate_module=f"encoder.stages.{intermediate_stage}",
                          intermediate_feature_shapes=sorted(intermediate_shapes),
                          intermediate_native_window={k: v for k, v in intermediate_native.items() if k != "channels"} if intermediate_native else None,
                          transpose_forward=list(predictor.plans_manager.transpose_forward),
                          transpose_backward=list(predictor.plans_manager.transpose_backward),
                          preprocessing_properties={k: properties[k] for k in
                              ("shape_before_cropping", "shape_after_cropping_and_before_resampling", "bbox_used_for_cropping")})
        return original, provenance, native, intermediate_native
    finally:
        for handle in handles:
            handle.remove()
        captured.clear()


def resolve_external_trainer(finder, trainer_name=TRAINER):
    """Use nnU-Net's temporary external import path and reject namesakes."""
    if trainer_name not in SUPPORTED_TRAINERS:
        raise ValueError("unsupported Trainer")
    discovered = finder(trainer_name)
    extension = resolved(Path(__file__).parent / "nnunet_ext_trainers")
    paths = os.environ.get("nnUNet_extTrainer", "").split(os.pathsep)
    expected_mixin = extension / "nnUNetTrainerMixins.py"
    architecture = getattr(discovered, "build_network_architecture", None)
    if (not isinstance(discovered, type) or discovered.__name__ != trainer_name
            or discovered.__module__ != trainer_name
            or len(paths) != 1 or resolved(paths[0]) != extension
            or architecture is None
            or (trainer_name == TRAINER and (
                resolved(architecture.__code__.co_filename) != expected_mixin
                or [base.__name__ for base in discovered.__bases__[:3]] !=
                ["EarlyStoppingMixin", "TopK10LossMixin", "UPerNetArchitectureMixin"]))
            or (trainer_name == ORIGINAL_TRAINER and (
                [base.__name__ for base in discovered.__bases__] != ["TopK10LossMixin", "nnUNetTrainer"]
                or resolved(discovered.__bases__[0].__dict__["_build_loss"].__code__.co_filename) != expected_mixin
                or architecture.__qualname__ != "nnUNetTrainer.build_network_architecture"))):
        raise ValueError("external Trainer identity mismatch")
    return discovered


def _predictor(info, args):
    from importlib.metadata import version
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
    if version("nnunetv2") != "2.8.1":
        raise ValueError("this feature mapping was audited only against nnunetv2==2.8.1")
    trainer_name = info["model"].name.split("__")[0]
    discovered = resolve_external_trainer(recursive_find_trainer_class_by_name, trainer_name)
    metadata = torch.load(info["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    try:
        if metadata.get("trainer_name") != trainer_name or metadata.get("init_args", {}).get("configuration") != "2d":
            raise ValueError("checkpoint Trainer/configuration mismatch")
        fold_value = metadata.get("init_args", {}).get("fold")
        if fold_value is not None and str(fold_value) != "0":
            raise ValueError("checkpoint fold mismatch")
    finally:
        del metadata
    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                                perform_everything_on_device=False, device=torch.device(args.device),
                                verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(info["model"]), use_folds=(0,),
                                                    checkpoint_name=info["checkpoint"].name)
    if predictor.trainer_name != trainer_name or len(predictor.configuration_manager.patch_size) != 2:
        raise ValueError("loaded Trainer/configuration identity mismatch")
    _network_kind(predictor.network, trainer_name)
    if trainer_name == ORIGINAL_TRAINER:
        if predictor.network.decoder.deep_supervision is not False:
            raise ValueError("original U-Net inference must return the main output with deep supervision disabled")
    return predictor


def _network_kind(network, trainer_name):
    if trainer_name == TRAINER and network.__class__.__name__ == "_PlainConvUNetUPerNet":
        return "upernet"
    if trainer_name == ORIGINAL_TRAINER:
        from dynamic_network_architectures.architectures.unet import PlainConvUNet
        from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
        from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
        if (type(network) is PlainConvUNet and type(network.encoder) is PlainConvEncoder
                and type(network.decoder) is UNetDecoder
                and len(network.encoder.stages) == len(network.encoder.output_channels)
                and len(network.decoder.transpconvs) == len(network.encoder.stages) - 1
                and len(network.decoder.stages) == len(network.decoder.transpconvs)
                and len(network.decoder.seg_layers) == len(network.decoder.stages)):
            return "plain_unet"
    raise ValueError(f"unsupported loaded network architecture for {trainer_name}: {type(network).__name__}")


def _original_feature_stages(network, patch_size):
    if len(patch_size) != 2:
        raise ValueError("expected 2D patch size")
    if len(network.encoder.strides) != len(network.encoder.stages):
        raise ValueError("encoder strides/stages mismatch")
    size = [int(v) for v in patch_size]
    matches = []
    for index, stride in enumerate(network.encoder.strides):
        pair = (stride, stride) if isinstance(stride, int) else stride
        if len(pair) != 2 or any(int(v) < 1 for v in pair):
            raise ValueError("invalid encoder stride")
        size = [int((v + int(st) - 1) // int(st)) for v, st in zip(size, pair)]
        if size == [64, 64]:
            matches.append(index)
    if len(matches) != 1 or matches[0] == len(network.encoder.stages) - 1:
        raise ValueError(f"expected one native 64x64 intermediate encoder stage; found {matches}")
    return len(network.encoder.stages) - 1, matches[0]


def _normalize_display(array):
    import numpy as np
    a = np.asarray(array, dtype=np.float32)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros_like(a)
    lo, hi = np.percentile(finite, (1, 99))
    if hi <= lo:
        return np.zeros_like(a)
    return np.clip((a-lo)/(hi-lo), 0, 1)


def _normalize_native_channel(values):
    """Scale one native channel for display; constant channels display at zero."""
    import numpy as np
    values = np.asarray(values)
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi == lo:
        return np.zeros(values.shape, dtype=np.float32)
    return (values - lo) / (hi - lo)


def _architecture(ax, network, stage, trainer_name=TRAINER, intermediate_stage=3, patch_size=None):
    if trainer_name == ORIGINAL_TRAINER:
        if patch_size is None:
            raise ValueError("original architecture overview requires patch size")
        from nnunet_report_architecture import draw_overview
        return draw_overview(ax, network, patch_size, stage, intermediate_stage)
    from matplotlib.patches import FancyBboxPatch
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    encoder, decoder = network.encoder, network.decoder
    chosen = tuple(network.selected_feature_indices)
    channels = tuple(encoder.output_channels)
    strides = getattr(encoder, "strides", None)
    scales = tuple(decoder.pool_scales)
    width = int(decoder.fpn_channels)
    if len(chosen) != 4 or len(channels) != len(encoder.stages):
        raise ValueError("unexpected UPerNet stage configuration")

    def box(x, y, w, h, label, color="#e9f3fa", size=8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004",
                                    facecolor=color, edgecolor="#31546a", linewidth=.8))
        ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=size)
    def arrow(x1, y1, x2, y2, color="#456879"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1, color=color,
                                    shrinkA=1, shrinkB=1))
    def route(points, color="#456879"):
        for start, end in zip(points, points[1:]):
            ax.plot((start[0], end[0]), (start[1], end[1]), color=color, lw=1)
        arrow(*points[-2], *points[-1], color=color)
    ax.text(.10, .97, "ENCODER", ha="center", fontsize=11, weight="bold")
    ax.text(.51, .97, "DECODER: PPM + FPN", ha="center", fontsize=11, weight="bold")
    ax.text(.87, .97, "FUSION / OUTPUT", ha="center", fontsize=11, weight="bold")
    box(.015, .88, .18, .055, "DWI input", "#f5f5f5")
    cumulative = [1, 1]
    stage_y = {}
    for i, count in enumerate(channels):
        if strides is not None:
            pair = strides[i]
            cumulative = [cumulative[j] * int(pair[j]) for j in (0, 1)]
            resolution = f"1/{cumulative[0]} x 1/{cumulative[1]}"
        else:
            resolution = "relative size unknown"
        y = .78 - i*.085
        stage_y[i] = y+.03
        label = f"stage {i}  |  {count} ch  |  {resolution}"
        if i == stage:
            label += "\nfinal encoder / pre-decoder"
        box(.01, y, .20, .064, label, "#ffe7bd" if i == stage else "#e9f3fa", 7)
        arrow(.11, .88 if i == 0 else stage_y[i-1]-.03, .11, y+.065)
    # Four scale rows. Each Pj exposes separate top-down and fusion ports.
    levels = [.78, .61, .44, .27]
    for j, index in enumerate(chosen):
        y = levels[j]
        arrow(.21, stage_y[index], .27, y+.025)
        if j < 3:
            box(.27, y, .12, .05, f"lateral {j}: 1x1", "#dff4eb", 7)
            box(.47, y, .14, .05, f"upsample + add\n3x3 refine = P{j}", "#dff4eb", 7)
            arrow(.39, y+.025, .47, y+.025)
        else:
            box(.27, y, .12, .05, "deepest", "#fcebdc", 7)
            box(.47, y, .14, .05, f"concat + 3x3\nbottleneck = P3", "#fcebdc", 7)
        # Refined pyramid tensor travels to the next adjacent higher scale.
        if j > 0:
            route([(.54, y+.05), (.54, levels[j-1])])
    # PPM remains local to P3; three branches and the identity input meet at concat.
    for k, scale in enumerate(scales):
        x = .30+k*.055
        box(x, .135, .05, .058, f"pool {scale}²\n1x1", "#fcebdc", 6)
        route([(.33, .27), (.33, .215), (x+.025, .215), (x+.025, .193)])
        lane_y = .07 + k*.015
        port_x = .56 - k*.04
        route([(x+.025, .135), (x+.025, lane_y), (port_x, lane_y), (port_x, .27)])
    route([(.39, .295), (.47, .295)])
    ax.text(.40, .045, "PPM branches resize to deepest before concat", fontsize=7, ha="center")
    # Dedicated fusion ports keep each scale visible; no shared bus before concat.
    box(.69, .43, .13, .09, f"resize P0..P3 to P0\nconcat + 3x3 fusion\n{width} ch", "#dff4eb", 7)
    for j, y in enumerate(levels):
        route([(.61, y+.025), (.64+j*.012, y+.025), (.64+j*.012, .50-j*.018), (.69, .50-j*.018)])
    box(.84, .45, .065, .05, f"1x1 classifier\n{decoder.classifier.out_channels} logits", "#ece6f8", 7)
    arrow(.82, .475, .84, .475)
    box(.92, .44, .07, .07, "bilinear upsample\nto input HxW", "#ece6f8", 7)
    arrow(.905, .475, .92, .475)
    ax.text(.71, .035, f"PPM scales {scales}; selected stages {chosen}. TopK10 / early stopping: training only.",
            ha="center", fontsize=8)


def _plain_unet_overview_figure(network, patch_size, stage, intermediate_stage, output):
    import matplotlib.pyplot as plt
    count = len(network.encoder.stages)
    fig, ax = plt.subplots(figsize=(19, 3.8 + 1.16*count), facecolor="#f7f9fc")
    try:
        _architecture(ax, network, stage, ORIGINAL_TRAINER, intermediate_stage, patch_size)
        fig.subplots_adjust(left=.015, right=.995, top=.99, bottom=.015)
        fig.savefig(output, dpi=160, facecolor=fig.get_facecolor())
    finally:
        plt.close(fig)


def _plain_unet_detail_figure(network, patch_size, stage, intermediate_stage, output):
    from nnunet_report_architecture import draw_detail
    draw_detail(network, patch_size, stage, intermediate_stage, output)


def _confusion_masks(gt, pred):
    import numpy as np
    gt, pred = np.asarray(gt, dtype=bool), np.asarray(pred, dtype=bool)
    if gt.shape != pred.shape:
        raise ValueError("GT/prediction shape mismatch")
    return gt & pred, ~gt & pred, gt & ~pred


def _overlay_rgba(gt, pred):
    import numpy as np
    tp, fp, fn = _confusion_masks(gt, pred)
    rgba = np.zeros((*tp.shape, 4), dtype=np.float32)
    for mask, color in ((tp, (0.12, .82, .30, .62)),
                        (fp, (1., .18, .18, .72)),
                        (fn, (.16, .42, 1., .72))):
        rgba[mask] = color
    return rgba


def _confusion_legend_handles():
    from matplotlib.patches import Patch
    return [Patch(color=(.12, .82, .30, .62), label="TP"),
            Patch(color=(1., .18, .18, .72), label="FP"),
            Patch(color=(.16, .42, 1., .72), label="FN")]


def _draw_case_panels(fig, subgrid, item, rows):
    import matplotlib.pyplot as plt
    outer = subgrid.subgridspec(4, 6, height_ratios=[.18, 1, 1, 1],
                                hspace=.06, wspace=.025)
    header = fig.add_subplot(outer[0, :])
    header.axis("off")
    header.text(.02, .5, f"{item['cid']}   Dice {float(rows[item['cid']]['dice']):.3f}",
                fontsize=11, weight="bold", va="center")
    names = ("DWI", "GT", "Saved pred.", "TP / FP / FN", "Feature magnitude", "DWI + feature")
    for row, index in enumerate(item["slices"] or [None]):
        for col, name in enumerate(names):
            ax = fig.add_subplot(outer[row+1, col])
            ax.set_xticks([]); ax.set_yticks([])
            if index is None:
                ax.text(.5, .5, "No GT-positive slice", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7)
            else:
                raw, gt, pred, feature = item["panels"][index]
                dwi = _normalize_display(raw)
                magnitude = _normalize_display(feature)
                if col == 0:
                    ax.imshow(dwi, cmap="gray", vmin=0, vmax=1)
                elif col == 1:
                    ax.imshow(gt, cmap="gray", vmin=0, vmax=1)
                elif col == 2:
                    ax.imshow(pred, cmap="gray", vmin=0, vmax=1)
                elif col == 3:
                    ax.imshow(dwi, cmap="gray", vmin=0, vmax=1)
                    ax.imshow(_overlay_rgba(gt, pred))
                elif col == 4:
                    ax.imshow(magnitude, cmap="inferno", vmin=0, vmax=1)
                else:
                    ax.imshow(dwi, cmap="gray", vmin=0, vmax=1)
                    ax.imshow(magnitude, cmap="inferno", vmin=0, vmax=1, alpha=.48)
            if row == 0:
                ax.set_title(name, fontsize=8, pad=2)
            if col == 0:
                ax.set_ylabel(f"slice {index}" if index is not None else "no slice",
                              fontsize=8)


def _native_channel_figure(items, output, *, layer="native"):
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    fig = plt.figure(figsize=(18, 2.3*len(items)+1.6))
    grid = fig.add_gridspec(len(items)+2, 8, height_ratios=[.35]+[1]*len(items)+[.4],
                            left=.06, right=.98, top=.96, bottom=.13, hspace=.48, wspace=.16)
    title = fig.add_subplot(grid[0, :]); title.axis("off")
    heading = ("Intermediate encoder features" if layer == "intermediate_native"
               else "Final encoder / pre-decoder features")
    title.text(.5, .5, f"{heading} | native channels | shared local window | nearest",
               ha="center", va="center", fontsize=12)
    norm = Normalize(0, 1)
    cmap = "coolwarm"
    for row, item in enumerate(items):
        native = item[layer]
        for col in range(8):
            ax = fig.add_subplot(grid[row+1, col])
            ax.set_xticks([]); ax.set_yticks([])
            if native is None or col >= len(native["channel_ids"]):
                ax.axis("off")
                continue
            values = native["channels"][col]
            ax.imshow(_normalize_native_channel(values), cmap=cmap, norm=norm,
                      interpolation="nearest")
            ax.set_title(f"stage {native.get('stage', '?')} | ch {native['channel_ids'][col]}\n{native.get('module', 'encoder stage')} | {native['feature_shape'][1] if 'feature_shape' in native else len(native['channel_ids'])}x{values.shape[0]}x{values.shape[1]}",
                         fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{item['cid']}\noriginal slice {native['original_slice']}\npreprocessed slice {native.get('preprocessed_slice', '?')}\nwindow {native['window_index']}\npadded y/x {native.get('window_yx_padded', '?')}",
                              fontsize=8)
    cax = fig.add_subplot(grid[-1, 2:6])
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    bar.set_label("Relative activation within each channel [0,1]; not lesion probability or absolute cross-channel/case intensity", fontsize=9)
    try:
        fig.savefig(output, dpi=140)
    finally:
        plt.close(fig)


def _summary_figure(items, rows, network, stage, output, trainer_name=TRAINER, intermediate_stage=3, source_note=None):
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    if trainer_name == ORIGINAL_TRAINER:
        # The original network has two standalone architecture PNGs. Keep the
        # case comparison free of diagram panels while preserving its content.
        fig = plt.figure(figsize=(30, 29))
        grid = fig.add_gridspec(7, 2, height_ratios=[.5, .5, .48, .55, 7.2, 7.2, 7.2],
                               hspace=.16, wspace=.04, left=.025, right=.99, bottom=.02, top=.985)
        base = 0
    else:
        fig = plt.figure(figsize=(30, 37))
        grid = fig.add_gridspec(8, 2, height_ratios=[8.2, .5, .5, .48, .55, 7.2, 7.2, 7.2],
                               hspace=.16, wspace=.04, left=.025, right=.99, bottom=.02, top=.985)
        _architecture(fig.add_subplot(grid[0, :]), network, stage, trainer_name, intermediate_stage)
        base = 1
    legend_ax = fig.add_subplot(grid[base, :]); legend_ax.axis("off")
    legend_ax.legend(handles=_confusion_legend_handles(), loc="center", ncol=3, fontsize=10)
    cax = fig.add_subplot(grid[base+1, :])
    bar = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap="inferno"),
                       cax=cax, orientation="horizontal")
    bar.set_label("Feature magnitude display: per-slice 1st-99th percentile -> [0,1]; overlay alpha 0.48", fontsize=8)
    note = fig.add_subplot(grid[base+2, :]); note.axis("off")
    note.text(.5, .5, source_note or "Diagnostic feature magnitude; colors are display-normalized and do not encode lesion probability or absolute cross-case intensity.",
              ha="center", va="center", fontsize=9)
    for col, title in enumerate(("HIGH DICE | 3 cases", "LOW DICE | 3 cases")):
        head = fig.add_subplot(grid[base+3, col])
        head.axis("off")
        head.text(.5, .5, title, ha="center", va="center", fontsize=16, weight="bold")
    for i, item in enumerate(items):
        col = 0 if i < 3 else 1
        row = base+4 + i % 3
        _draw_case_panels(fig, grid[row, col], item, rows)
    try:
        fig.savefig(output, dpi=130)
    finally:
        plt.close(fig)


def _source_statement(trainer_name, checkpoint_name, declaration):
    if trainer_name == ORIGINAL_TRAINER:
        if declaration is None:
            return (f"Saved-prediction checkpoint provenance: UNKNOWN. Specified {checkpoint_name} "
                    "extracts diagnostic features; historical prediction weights are unconfirmed.")
        if not declaration.strip():
            raise ValueError("user confirmation statement must be nonempty")
        return ("Saved-prediction checkpoint provenance: USER CONFIRMED, not independently verified. "
                f"Original user statement: {declaration}")
    return f"Saved-prediction/checkpoint declaration (USER DECLARED, not independently verified): {declaration}"


def create_report(info, predictor, declaration):
    import numpy as np
    import SimpleITK as sitk
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    require_simpleitk_reader(info["plans"])
    high, low = select_cases(info["rows"])
    trainer_name = info["model"].name.split("__")[0]
    kind = _network_kind(predictor.network, trainer_name)
    if kind == "plain_unet":
        stage, intermediate_stage = _original_feature_stages(
            predictor.network, predictor.configuration_manager.patch_size)
    else:
        stage, intermediate_stage = predictor.network.selected_feature_indices[-1], 3
    items = []
    for group, ids in (("High Dice", high), ("Low Dice", low)):
        for cid in ids:
            image = sitk.ReadImage(str(info["images"][cid]))
            gt = sitk.ReadImage(str(info["labels"][cid]))
            pred = sitk.ReadImage(str(info["predictions"][cid]))
            check_geometry(image, gt, cid); check_geometry(image, pred, cid)
            image_array = sitk.GetArrayFromImage(image)
            gt_array = sitk.GetArrayFromImage(gt) > 0
            pred_array = sitk.GetArrayFromImage(pred) > 0
            slices = select_slices(gt_array)
            feature, provenance, native, intermediate_native = _diagnostic_feature(
                info["images"][cid], predictor, stage, slices[0] if slices else None, intermediate_stage)
            if feature.shape != image_array.shape:
                raise ValueError(f"{cid}: inverse feature shape mismatch")
            panels = {index: (image_array[index].copy(), gt_array[index].copy(),
                              pred_array[index].copy(), feature[index].copy()) for index in slices}
            items.append(dict(group=group, cid=cid, panels=panels, slices=slices,
                              native=native, intermediate_native=intermediate_native,
                              geometry={k: getattr(image, "Get"+k)() for k in
                                        ("Size", "Spacing", "Origin", "Direction")},
                              provenance=provenance))
    source_note = (f"Diagnostic: {info['checkpoint'].name} features; saved prediction weights unknown."
                   if kind == "plain_unet" and declaration is None else
                   "Diagnostic features: specified checkpoint; saved prediction source USER CONFIRMED (user statement only)."
                   if kind == "plain_unet" else None)
    if kind == "plain_unet":
        patch_size = predictor.configuration_manager.patch_size
        _plain_unet_overview_figure(predictor.network, patch_size, stage,
                                    intermediate_stage, info["output"] / "architecture_overview.png")
        _plain_unet_detail_figure(predictor.network, patch_size, stage,
                                  intermediate_stage, info["output"] / "architecture_detail.png")
    _summary_figure(items, info["rows"], predictor.network, stage,
                    info["output"] / "summary.png", trainer_name, intermediate_stage, source_note)
    _native_channel_figure(items, info["output"] / "feature_channels.png")
    _native_channel_figure(items, info["output"] / "feature_channels_64x64.png", layer="intermediate_native")

    title = ("Dataset501 official nnU-Net 2D TopK10 PlainConvUNet: fold 0 single-fold validation report"
             if kind == "plain_unet" else
             "Dataset501 official nnU-Net 2D UPerNet TopK10 EarlyStopping: fold 0 single-fold validation report")
    source_line = _source_statement(trainer_name, info["checkpoint"].name, declaration)
    txt = [title,
           "This is not five-fold OOF or a clinical conclusion.",
           f"Model directory: {info['model']}", f"Checkpoint: {info['checkpoint']}",
           f"Checkpoint SHA256: {_hash_file(info['checkpoint'])}",
           f"Full-set geometry verified: {info['geometry_count']} of {len(info['rows'])} metrics-covered cases (DWI, GT, saved prediction; size, spacing, origin, direction).",
           "Reader/axis contract: plans image_reader_writer=SimpleITKIO; 3D GetArrayFromImage z/y/x.",
           source_line,
           f"Historical saved-prediction TTA status (user supplied, not independently verified): {info.get('historical_tta', 'unknown')}; original statement above may contain details.",
           "Diagnostic features: freshly preprocessed raw DWI, no GT and no TTA; they are not saved prediction features or a replay.",
           f"Feature: mean(abs(channel)) at final encoder stage {stage}; native 64x64 stage {intermediate_stage}; bilinear upsample per patch; Gaussian overlap mean; inverse padding/resample/crop/transpose.",
           "Native channels: fixed uniformly spaced IDs, at most eight per actual layer; final and intermediate encoder features share one local window containing the first displayed slice center; nearest pixel display. They are not full-image maps or lesion probabilities. Same column across layers has no guaranteed semantic match.",
           "Summary display: DWI and feature magnitude each use per-slice 1st-99th percentile normalization to [0,1]; constant maps become zero; DWI+feature overlay alpha=0.48. Intensities not comparable across cases. TP green, FP red, FN blue.",
            *(["Original PlainConvUNet architecture: architecture_overview.png shows the module flow; architecture_detail.png separates encoder, decoder and output-head panels. Both use loaded network metadata; summary.png contains case comparisons only."] if kind == "plain_unet" else []),
           "Native-channel display: each channel at its actual native CxHxW independently uses (value-min)/(max-min) to [0,1]; constant channels display uniformly at zero. Shared coolwarm blue/red means relative low/high only, not negative/positive. Colors cannot compare absolute intensity across channels or cases and are not lesion probabilities.",
           "Original-space array axis 0, zero-based index; no anatomical plane claim.",
           "Selection: full-case Dice descending, case_id tie break, top 3; among remaining, Dice ascending, case_id tie break, bottom 3; GT area descending, index tie break, max 3 positive slices.",
           f"Metrics source: {info['metrics']}", "Full validation set summary (source JSON, including units/aggregation/valid_cases/F2):",
           f"F2 definition from source: {info['summary'].get('f2_mode', 'unknown (source omitted)')}; "
           "AVD unit=percent, HD95 unit=mm; missing values remain missing, never zero-filled.",
           json.dumps(info["summary"], indent=2, ensure_ascii=False, allow_nan=True),
           f"Runtime: nnunetv2={__import__('importlib.metadata', fromlist=['version']).version('nnunetv2')}; "
           f"torch={__import__('torch').__version__}; device={predictor.device}; "
           "diagnostic tile_step_size=0.5, Gaussian=True, mirroring=False",
           "Network repr:", repr(predictor.network), "Selected cases:"]
    for item in items:
        txt.extend([f"{item['group']}: {item['cid']} | slices={item['slices'] or 'none (empty GT)'} | fewer than 3 positive slices={len(item['slices']) < 3}",
                    "Metrics: " + json.dumps(info["rows"][item["cid"]], ensure_ascii=False),
                    "Geometry: " + json.dumps(item["geometry"]),
                    "Feature mapping: " + json.dumps(item["provenance"], default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))])
    (info["output"] / "report.txt").write_text("\n".join(txt)+"\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("official-nnunet", "standalone-h2former"), default="official-nnunet")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path)
    parser.add_argument("--metrics-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="metadata only; no model or real-image loading")
    parser.add_argument("--prediction-checkpoint-declaration", help="user statement about saved-mask checkpoint provenance (official or standalone); required for official UPerNet")
    parser.add_argument("--confirm-prediction-checkpoint", action="store_true", help="explicit user confirmation for original TopK10 or standalone saved-mask checkpoint provenance; requires declaration")
    parser.add_argument("--historical-tta", choices=("unknown", "enabled", "disabled"), default="unknown", help="user-supplied historical prediction TTA status")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--manifest", type=Path, help="standalone existing prediction_manifest.json")
    parser.add_argument("--config", type=Path, help="standalone resolved_config.json")
    parser.add_argument("--allow-pending", action="store_true", help="explicitly allow pending standalone checkpoint/manifest")
    args = parser.parse_args(argv)
    try:
        if args.source == "standalone-h2former":
            if args.confirm_prediction_checkpoint != bool(args.prediction_checkpoint_declaration):
                raise ValueError("user confirmation requires both --confirm-prediction-checkpoint and --prediction-checkpoint-declaration")
            from standalone_h2former_report import run
            return run(args)
        info = inspect_sources(args)
        if args.check:
            print(json.dumps({"status": "METADATA ONLY; geometry, checkpoint content and provenance not verified",
                              "cases": len(info["rows"]), "selected": select_cases(info["rows"]),
                              "count_coverage": count_coverage(info["rows"]),
                              "checkpoint": str(info["checkpoint"]), "output": str(info["output"])}, indent=2))
            return 0
        if args.confirm_prediction_checkpoint and not args.prediction_checkpoint_declaration:
            raise ValueError("checkpoint confirmation requires original user statement")
        if info["model"].name.startswith(ORIGINAL_TRAINER + "__") and bool(args.prediction_checkpoint_declaration) != args.confirm_prediction_checkpoint:
            raise ValueError("original TopK10 user confirmation requires both --confirm-prediction-checkpoint and --prediction-checkpoint-declaration")
        if args.historical_tta != "unknown" and not args.prediction_checkpoint_declaration:
            raise ValueError("known historical TTA requires original user statement")
        info["historical_tta"] = args.historical_tta
        if info["model"].name.startswith(TRAINER + "__") and not args.prediction_checkpoint_declaration:
            raise ValueError("UPerNet full report requires --prediction-checkpoint-declaration; metadata-only --check is available")
        require_simpleitk_reader(info["plans"])
        info["geometry_count"] = verify_full_geometry(info)
        predictor = _predictor(info, args)
        output = info["output"]
        if not output.parent.is_dir():
            raise ValueError(f"output parent directory must exist: {output.parent}")
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
        try:
            work = staging / "report"
            work.mkdir()
            staged_info = dict(info, output=work)
            create_report(staged_info, predictor, args.prediction_checkpoint_declaration)
            table = build_table(info["rows"], info["summary"], info["predictions"], info["labels"],
                                label_contract=lambda: binary_labels(info["dataset"]),
                                check_geometry=check_geometry, geometry_verified=True)
            table_text = export_table(table, work)
            with (work / "report.txt").open("a", encoding="utf-8") as handle:
                handle.write(table_text)
            required = ["metrics_table.csv", "metrics_table.png", "summary.png", "feature_channels.png", "feature_channels_64x64.png", "report.txt"]
            if info["model"].name.startswith(ORIGINAL_TRAINER + "__"):
                required.extend(("architecture_overview.png", "architecture_detail.png"))
            if not all((work / name).is_file() for name in required):
                raise ValueError("report output incomplete")
            protected_output(output, (info["model"], *info["images"].values(),
                             *info["labels"].values(), *info["predictions"].values(),
                             info["metrics"], info["checkpoint"]))
            os.rename(work, output)
        finally:
            shutil.rmtree(staging)
        print("Wrote " + ", ".join(str(output / name) for name in required))
        return 0
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
