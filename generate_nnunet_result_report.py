"""Single-fold Dataset501 UPerNet validation report from existing masks/metrics.

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
from pathlib import Path


TRAINER = "nnUNetTrainerUPerNetTopK10EarlyStopping"
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
        rows = {}
        for row in reader:
            case_id = row["case_id"]
            if not case_id or case_id in rows:
                raise ValueError(f"missing or duplicate metric case ID: {case_id!r}")
            try:
                dice = float(row["dice"])
            except (ValueError, TypeError) as error:
                raise ValueError(f"invalid Dice for {case_id}") from error
            if not math.isfinite(dice) or not 0 <= dice <= 1:
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
    return rows, summary


def _finite_or_missing(value):
    if value is None or str(value).strip().lower() in ("", "nan", "none", "null"):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def select_cases(rows):
    if len(rows) < 6:
        raise ValueError("at least six distinct validation cases are required")
    high = sorted(rows, key=lambda cid: (-float(rows[cid]["dice"]), cid))[:3]
    remaining = set(rows) - set(high)
    low = sorted(remaining, key=lambda cid: (float(rows[cid]["dice"]), cid))[:3]
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
    if fold != 0 or model_dir.name != f"{TRAINER}__nnUNetPlans__2d":
        raise ValueError("only Dataset501 official UPerNet TopK10 fold 0 is supported")
    dataset = json.loads((model_dir / "dataset.json").read_text(encoding="utf-8"))
    plans = json.loads((model_dir / "plans.json").read_text(encoding="utf-8"))
    if len(dataset.get("channel_names", {})) != 1:
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
    metrics = resolved(args.metrics_dir or model / "fold_0" / "multi_metrics")
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


def _diagnostic_feature(raw_path, predictor, stage):
    """Re-preprocess raw input without GT; no TTA, patch-weighted map."""
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
    module = predictor.network.encoder.stages[stage]
    captured = []
    def hook(_module, _inputs, output):
        captured.append(output.detach())
    handle = module.register_forward_hook(hook)
    try:
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
        with torch.inference_mode():
            for sl in predictor._internal_get_sliding_window_slicers((z, h, w)):
                if len(sl) != 4 or not isinstance(sl[1], int):
                    raise ValueError("expected 2D slice window")
                patch = padded[sl][None].to(predictor.device)
                captured.clear()
                predictor.network(patch)
                if len(captured) != 1:
                    raise ValueError("deepest encoder stage hook did not fire exactly once")
                feature = captured.pop()
                shapes.add(tuple(feature.shape))
                intensity = feature.abs().mean(dim=1, keepdim=True)
                intensity = F.interpolate(intensity, size=tuple(predictor.configuration_manager.patch_size),
                                          mode="bilinear", align_corners=False)[0, 0].float().cpu().numpy()
                _add_window(total[sl[1]], weights[sl[1]], intensity, gaussian, sl[2:])
                del feature, intensity, patch
        if np.any(weights <= 0):
            raise ValueError("uncovered preprocessed feature pixel")
        map_pre = (total / weights)[undo[1:]]
        if map_pre.shape != data.shape[1:]:
            raise ValueError("padding removal mismatch")
        original = _inverse_feature(map_pre, properties, predictor.plans_manager,
                                    predictor.configuration_manager)
        return original, dict(stage=stage, module=f"encoder.stages.{stage}",
                              feature_shapes=sorted(shapes), windows=int(np.count_nonzero(weights)),
                              transpose_forward=list(predictor.plans_manager.transpose_forward),
                              transpose_backward=list(predictor.plans_manager.transpose_backward),
                              preprocessing_properties={k: properties[k] for k in
                                  ("shape_before_cropping", "shape_after_cropping_and_before_resampling", "bbox_used_for_cropping")})
    finally:
        handle.remove()
        captured.clear()


def resolve_external_trainer(finder):
    """Use nnU-Net's temporary external import path and reject namesakes."""
    discovered = finder(TRAINER)
    extension = resolved(Path(__file__).parent / "nnunet_ext_trainers")
    paths = os.environ.get("nnUNet_extTrainer", "").split(os.pathsep)
    expected_mixin = extension / "nnUNetTrainerMixins.py"
    architecture = getattr(discovered, "build_network_architecture", None)
    if (not isinstance(discovered, type) or discovered.__name__ != TRAINER
            or discovered.__module__ != TRAINER
            or len(paths) != 1 or resolved(paths[0]) != extension
            or architecture is None
            or resolved(architecture.__code__.co_filename) != expected_mixin
            or [base.__name__ for base in discovered.__bases__[:3]] !=
            ["EarlyStoppingMixin", "TopK10LossMixin", "UPerNetArchitectureMixin"]):
        raise ValueError("external Trainer identity mismatch")
    return discovered


def _predictor(info, args):
    from importlib.metadata import version
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
    if version("nnunetv2") != "2.8.1":
        raise ValueError("this feature mapping was audited only against nnunetv2==2.8.1")
    discovered = resolve_external_trainer(recursive_find_trainer_class_by_name)
    metadata = torch.load(info["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    try:
        if metadata.get("trainer_name") != TRAINER or metadata.get("init_args", {}).get("configuration") != "2d":
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
    if predictor.trainer_name != TRAINER or len(predictor.configuration_manager.patch_size) != 2:
        raise ValueError("loaded Trainer/configuration identity mismatch")
    if predictor.network.__class__.__name__ != "_PlainConvUNetUPerNet":
        raise ValueError("loaded network class mismatch")
    return predictor


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


def _architecture(ax, network, stage):
    from matplotlib.patches import FancyBboxPatch
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    channels = network.encoder.output_channels
    chosen = network.selected_feature_indices
    decoder = network.decoder
    nodes = [(0.02, "DWI input"), (0.22, f"PlainConv encoder\n{len(network.encoder.stages)} stages"),
             (0.48, "Selected stages\n" + ", ".join(f"{i}:{channels[i]}ch" for i in chosen)),
             (0.74, f"{type(decoder).__name__}\nPPM {tuple(decoder.pool_scales)}; FPN {decoder.fpn_channels}ch"),
             (0.92, f"{decoder.classifier.out_channels} logits")]
    widths = [0.14, 0.22, 0.24, 0.16, 0.07]
    for (x, label), width in zip(nodes, widths):
        ax.add_patch(FancyBboxPatch((x, .42), width, .25, boxstyle="round,pad=.01",
                                    facecolor="#e5f0fa", edgecolor="#1c567f"))
        ax.text(x+width/2, .545, label, ha="center", va="center", fontsize=9)
    for left, right in zip(nodes, nodes[1:]):
        ax.annotate("", xy=(right[0], .545), xytext=(left[0]+widths[nodes.index(left)], .545),
                    arrowprops={"arrowstyle": "->", "lw": 1.4})
    ax.text(.5, .17, f"Diagnostic hook: encoder.stages.{stage} (before PPM); training TopK10/early stopping are not forward layers",
            ha="center", fontsize=9)


def create_report(info, predictor, declaration):
    import numpy as np
    import SimpleITK as sitk
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    require_simpleitk_reader(info["plans"])
    high, low = select_cases(info["rows"])
    stage = predictor.network.selected_feature_indices[-1]
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
            feature, provenance = _diagnostic_feature(info["images"][cid], predictor, stage)
            if feature.shape != image_array.shape:
                raise ValueError(f"{cid}: inverse feature shape mismatch")
            slices = select_slices(gt_array)
            panels = {index: (image_array[index].copy(), gt_array[index].copy(),
                              pred_array[index].copy(), feature[index].copy()) for index in slices}
            items.append(dict(group=group, cid=cid, panels=panels, slices=slices,
                              geometry={k: getattr(image, "Get"+k)() for k in
                                        ("Size", "Spacing", "Origin", "Direction")},
                              provenance=provenance))
    rows = sum(max(1, len(item["slices"])) for item in items)
    fig = plt.figure(figsize=(15, max(11, rows*2.15+3)), layout="constrained")
    grid = fig.add_gridspec(rows+1, 4, height_ratios=[1.5]+[1]*rows)
    _architecture(fig.add_subplot(grid[0, :]), predictor.network, stage)
    cursor = 1
    for item in items:
        indices = item["slices"] or [None]
        for index in indices:
            for col, title in enumerate(("Original DWI", "GT", "Saved prediction", "Encoder intensity")):
                ax = fig.add_subplot(grid[cursor, col])
                ax.set_xticks([]); ax.set_yticks([])
                if index is None:
                    ax.text(.5, .5, "No GT-positive slice", transform=ax.transAxes, ha="center")
                else:
                    plane = item["panels"][index][col]
                    if col == 0:
                        ax.imshow(_normalize_display(plane), cmap="gray", vmin=0, vmax=1)
                    elif col in (1, 2):
                        ax.imshow(plane, cmap="gray", vmin=0, vmax=1)
                    else:
                        ax.imshow(_normalize_display(plane), cmap="inferno", vmin=0, vmax=1)
                ax.set_title(f"{item['group']} | {item['cid']} | Dice {float(info['rows'][item['cid']]['dice']):.3f}\n"
                             f"array axis 0 index {index} | {title}", fontsize=8)
            cursor += 1
    try:
        fig.savefig(info["output"] / "summary.png", dpi=110)
    finally:
        plt.close(fig)

    txt = ["Dataset501 official nnU-Net 2D UPerNet TopK10 EarlyStopping: fold 0 single-fold validation report",
           "This is not five-fold OOF or a clinical conclusion.",
           f"Model directory: {info['model']}", f"Checkpoint: {info['checkpoint']}",
           f"Checkpoint SHA256: {_hash_file(info['checkpoint'])}",
           f"Full-set geometry verified: {info['geometry_count']} of {len(info['rows'])} metrics-covered cases (DWI, GT, saved prediction; size, spacing, origin, direction).",
           "Reader/axis contract: plans image_reader_writer=SimpleITKIO; 3D GetArrayFromImage z/y/x.",
           f"Saved-prediction/checkpoint declaration (USER DECLARED, not independently verified): {declaration}",
           "Diagnostic features: freshly preprocessed raw DWI, no GT and no TTA; they are not saved prediction features or a replay.",
           "Feature: mean(abs(channel)) at deepest selected encoder stage, bilinear upsample per patch; Gaussian overlap mean; inverse padding/resample/crop/transpose.",
           "Display: per-slice 1st-99th percentile normalization; constant maps become zero; intensities not comparable across cases.",
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
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path)
    parser.add_argument("--metrics-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="metadata only; no model or real-image loading")
    parser.add_argument("--prediction-checkpoint-declaration", help="explicit user provenance statement for saved masks")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    try:
        info = inspect_sources(args)
        if args.check:
            print(json.dumps({"status": "METADATA ONLY; geometry, checkpoint content and provenance not verified",
                              "cases": len(info["rows"]), "selected": select_cases(info["rows"]),
                              "checkpoint": str(info["checkpoint"]), "output": str(info["output"])}, indent=2))
            return 0
        if not args.prediction_checkpoint_declaration:
            raise ValueError("full report requires --prediction-checkpoint-declaration; metadata-only --check is available")
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
            if not (work / "summary.png").is_file() or not (work / "report.txt").is_file():
                raise ValueError("report output incomplete")
            protected_output(output, (info["model"], *info["images"].values(),
                             *info["labels"].values(), *info["predictions"].values(),
                             info["metrics"], info["checkpoint"]))
            os.rename(work, output)
        finally:
            shutil.rmtree(staging)
        print(f"Wrote {output / 'summary.png'} and {output / 'report.txt'}")
        return 0
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
