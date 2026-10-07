"""Sequential Dataset501 reports from saved masks/metrics; optional diagnostics."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback
from generate_nnunet_result_report import (
    SUPPORTED_TRAINERS, FEATURE_TRAINERS, model_identity, TRAINER, collect, read_metrics, check_geometry,
    protected_output, _normalize_display, _overlay_rgba,
)
from report_comparison_selection import (
    DATASET, read_json, validate, load_selection, verify_selection,
    save_selection, check_selection_output, geometry, _volume,
)
from report_metrics_table import (
    binary_labels, build_table, export_table, macro_summary, voxel_counts, source_counts,
)

CODE = Path(__file__).resolve().parent
H2 = {"h2former", "h2former_lite_upernet", "h2former_lite_upernet_w128_ppm1236"}
PRUNE = re.compile(r"(?:^|[_ .-])(reports?|aborted|preflight|smoke|audit)(?:$|[_ .-])", re.I)
PATH_COLUMNS = ("niftipath", "prediction_path", "pred_path", "pred_file")


def exclusion(path):
    if PRUNE.search(path.name):
        return "report/aborted/preflight/smoke/audit directory"
    if re.match(r"(?:\d+_)?Dataset\d+", path.name) and path.name != DATASET:
        return "outside Dataset501 DWI cohort (Dataset508/multimodal/other dataset)"
    return ""


def scan(roots, records):
    index = {name: set() for name in ("dataset.json", "resolved_config.json",
                                     "prediction_manifest.json", "case_metrics.csv")}
    for root in roots:
        def onerror(error):
            records.append(dict(status="error", model=str(error.filename), reason=str(error)))
        for folder, dirs, files in os.walk(root, followlinks=False, onerror=onerror):
            here = Path(folder).resolve()
            reason = exclusion(here)
            if any(n.lower() in ("aborted", "aborted.json", "aborted.txt") for n in files):
                reason = "explicit aborted-run marker"
            if reason:
                records.append(dict(status="excluded", model=str(here), reason=reason))
                dirs[:] = []
                continue
            kept = []
            for name in sorted(dirs):
                child = here / name
                reason = exclusion(child)
                if child.is_symlink() or not child.resolve().is_relative_to(root):
                    reason = "linked directory/outside scan root"
                if reason:
                    records.append(dict(status="excluded", model=str(child), reason=reason))
                else:
                    kept.append(name)
            dirs[:] = kept
            for name in index:
                if name in files:
                    index[name].add(here / name)
    return index


def metrics_map(path, roots):
    if path is None:
        return {}
    document = read_json(path)
    if not isinstance(document, dict):
        raise ValueError("--metrics-map requires JSON absolute prediction dir -> absolute metrics dir")
    result = {}
    for pred, metrics in document.items():
        if not isinstance(pred, str) or not isinstance(metrics, str):
            raise ValueError("metrics-map paths must be strings")
        p, m = Path(pred), Path(metrics)
        if not p.is_absolute() or not m.is_absolute() or not p.is_dir() or not m.is_dir():
            raise ValueError(f"mapped paths must be existing absolute directories: {pred} -> {metrics}")
        p, m = p.resolve(), m.resolve()
        if not any(p.is_relative_to(root) for root in roots):
            raise ValueError(f"mapped prediction is outside results roots: {p}")
        if any(exclusion(part) for base in (p, m) for part in (base, *base.parents)):
            raise ValueError(f"mapped directory has excluded report/preflight/dataset ancestry: {p} -> {m}")
        if p in result:
            raise ValueError(f"duplicate resolved metrics-map key: {p}")
        if not all((m / n).is_file() for n in ("case_metrics.csv", "summary_metrics.json")):
            raise ValueError(f"mapped metrics CSV/summary missing: {m}")
        result[p] = m
    return result


def csv_references(metrics):
    with (metrics / "case_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = [c for c in PATH_COLUMNS if c in (reader.fieldnames or [])]
        if not columns:
            return None
        refs = {}
        for row in reader:
            cid = row.get("case_id")
            if not cid or cid in refs or None in row or any(v is None for v in row.values()):
                raise ValueError("malformed/duplicate metric case")
            paths = [Path(row[c]) for c in columns]
            if any(not p.is_absolute() for p in paths):
                raise ValueError("CSV prediction references must be absolute")
            paths = {p.resolve() for p in paths}
            if len(paths) != 1:
                raise ValueError("conflicting CSV prediction references")
            refs[cid] = paths.pop()
        return refs


def choose_metrics(job, predictions, metric_dirs, mapping):
    pred = job["prediction"]
    if pred in mapping:
        candidates, association = {mapping[pred]}, "explicit --metrics-map; caller association, provenance unverified"
    else:
        if job["source"] == "official-nnunet":
            names = [pred.parent / n for n in ("multi_metrics", "multi_metric_evaluation")]
        else:
            container = job["manifest"].parent
            name = container.name
            if name.startswith("full_volume_predictions"):
                names = [container.with_name("full_volume_metrics" + name[len("full_volume_predictions"):])]
            else:
                match = re.fullmatch(r"(.+)_predictions?", name)
                names = [container.with_name(match[1] + "_metrics")] if match else []
        candidates = {p.resolve() for p in names if p.is_dir()}
        for metrics in metric_dirs:
            try:
                if csv_references(metrics) == predictions:
                    candidates.add(metrics)
            except (OSError, ValueError, TypeError):
                continue
        association = "specific name / exact CSV prediction paths; provenance unverified"
    if len(candidates) != 1:
        raise ValueError(f"missing/ambiguous metrics: {sorted(map(str, candidates))}; supply --metrics-map (matching IDs alone is insufficient)")
    metrics = candidates.pop()
    refs = csv_references(metrics)
    if refs is not None and refs != predictions:
        raise ValueError("metric CSV paths conflict with saved predictions")
    return metrics, association


def h2_identity(config):
    model = config.get("model")
    if not isinstance(model, dict) or model.get("name") not in H2:
        raise ValueError("resolved_config lacks one of the three exact supported H2 identities")
    if model.get("supervision_mode") != "single_output" or type(model.get("num_classes")) is not int or model["num_classes"] != 2:
        raise ValueError("H2 requires explicit single_output and num_classes=2")
    expected = dict(in_channels=1, image_size=512, deep_supervision=False)
    if model["name"] == "h2former_lite_upernet_w128_ppm1236":
        expected.update(fpn_channels=128, ppm_scales=[1, 2, 3, 6], ppm_out_channels=128)
        if not all(k in model for k in ("fpn_channels", "ppm_scales", "ppm_out_channels")):
            raise ValueError("W128_PPM1236 architecture declaration missing")
    for key, value in expected.items():
        if key in model and (type(model[key]) is not type(value) or model[key] != value):
            raise ValueError(f"H2 model contract conflict: {key}")
    return model["name"]


def discover(index, records):
    jobs = []
    blocked = [Path(r["model"]) for r in records if r["status"] == "excluded"]
    for path in sorted(index["dataset.json"]):
        model = path.parent
        if not (model / "fold_0").is_dir():
            continue
        try:
            pred = model / "fold_0" / "validation"
            if any(pred.is_relative_to(p) for p in blocked):
                raise ValueError("validation lies under a pruned/aborted/linked directory")
            parts = model.name.split("__")
            if model.parent.name != DATASET or len(parts) != 3 or not parts[2].startswith("2d"):
                raise ValueError("outside exact Dataset501 / 2D config scope")
            dataset, plans = read_json(path), read_json(model / "plans.json")
            if dataset.get("channel_names") != {"0": "DWI"}:
                raise ValueError("multimodal/non-DWI metadata; requires channel 0=DWI only")
            for field in ("name", "dataset_name", "dataset_id"):
                if field in dataset and dataset[field] not in (DATASET, "Dataset501", "StrokeLesion", 501, "501"):
                    raise ValueError(f"conflicting dataset metadata: {field}")
            if parts[2] not in plans.get("configurations", {}):
                raise ValueError("configuration missing from plans")
            if binary_labels(dataset, plans) != 1:
                raise ValueError("only binary class-index labels 0/1 supported")
            jobs.append(dict(source="official-nnunet", model=model, identity=model.name,
                             prediction=(model / "fold_0" / "validation").resolve()))
        except Exception as error:
            records.append(dict(status="excluded", model=str(model), reason=str(error)))
    used = set()
    for path in sorted(index["prediction_manifest.json"]):
        try:
            configs = [p / "resolved_config.json" for p in [path.parent, *path.parent.parents]
                       if p / "resolved_config.json" in index["resolved_config.json"]]
            if not configs:
                raise ValueError("associated resolved_config missing; no identity inferred from folder names")
            config_path = configs[0]
            config = read_json(config_path)
            identity = h2_identity(config)
            jobs.append(dict(source="standalone-h2former", model=config_path.parent,
                             identity=identity, config=config_path, config_data=config,
                             manifest=path, prediction=(path.parent / "predictions").resolve()))
            used.add(config_path)
        except Exception as error:
            records.append(dict(status="excluded", model=str(path.parent), reason=str(error)))
    for path in sorted(index["resolved_config.json"] - used):
        records.append(dict(status="excluded", model=str(path.parent), reason="no supported H2 saved manifest associated with config"))
    unique = {}
    for job in jobs:
        unique.setdefault(job["prediction"], job)
    return list(unique.values())


def check_h2(job, info, allow_pending):
    # This validator is pure JSON; official_config/factory imports would import torch.
    from standalone_nnunet2d.alignment_evidence import validate_checkpoint_alignment_metadata
    config, manifest = job["config_data"], read_json(job["manifest"])
    if type(config.get("fold")) is not int or config["fold"] != 0:
        raise ValueError("H2 config must declare fold 0")
    for field in ("dataset", "dataset_name", "dataset_id"):
        if field in config and config[field] not in (DATASET, "Dataset501", "StrokeLesion", 501, "501"):
            raise ValueError(f"H2 conflicting dataset declaration: {field}")
    source = config.get("data_source", {})
    if isinstance(source, dict):
        if any(token != "Dataset501" for token in re.findall(r"Dataset\d+", str(source.get("root", "")))):
            raise ValueError("H2 data_source declares a different dataset")
    policy = manifest.get("policy", {})
    if manifest.get("schema_version") != 1 or not isinstance(policy, dict) or policy.get("output_space") != "source":
        raise ValueError("H2 manifest requires schema 1 / source space")
    state, _ = validate_checkpoint_alignment_metadata(dict(
        run_type=policy.get("run_state"), run_state=policy.get("alignment_status"),
        alignment_evidence=policy.get("alignment_evidence")))
    if validate_checkpoint_alignment_metadata(config)[0] != state:
        raise ValueError("H2 config/manifest alignment state mismatch")
    if state == "official_alignment_pending" and not allow_pending:
        raise ValueError("pending H2 alignment requires explicit --allow-pending")
    metadata = manifest.get("checkpoint")
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError("H2 manifest checkpoint model identity missing")
    if validate_checkpoint_alignment_metadata(metadata)[0] != state:
        raise ValueError("H2 checkpoint/manifest alignment state mismatch")
    for key in ("config", "resolved_config"):
        if key in metadata:
            nested = metadata[key]
            if not isinstance(nested, dict):
                raise ValueError(f"H2 checkpoint {key} must be a config object")
            if validate_checkpoint_alignment_metadata(nested)[0] != state:
                raise ValueError(f"H2 checkpoint {key}/manifest alignment state mismatch")
    identities = [h2_identity(metadata[key]) for key in ("config", "resolved_config") if key in metadata]
    if "model_name" in metadata or "supervision_mode" in metadata:
        architecture = metadata.get("architecture", {})
        if not isinstance(architecture, dict):
            raise ValueError("invalid H2 manifest architecture")
        identities.append(h2_identity({"model": dict(architecture, name=metadata.get("model_name"),
                          supervision_mode=metadata.get("supervision_mode"), num_classes=2)}))
    if not identities or any(name != job["identity"] for name in identities):
        raise ValueError("H2 config/manifest model identity conflict")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != len(info["predictions"]):
        raise ValueError("H2 manifest population mismatch")
    seen = set()
    for item in cases:
        if not isinstance(item, dict):
            raise ValueError("invalid H2 manifest case")
        cid = item.get("case_id")
        if cid in seen or cid not in info["predictions"]:
            raise ValueError("H2 manifest duplicate/unknown case")
        seen.add(cid)
        for key, files in (("source_path", info["images"]), ("prediction_path", info["predictions"])):
            value = item.get(key)
            if not isinstance(value, str) or not Path(value).is_absolute() or Path(value).resolve() != files[cid]:
                raise ValueError(f"{cid}: H2 manifest {key} mismatch")
    job["manifest_data"], job["run_state"] = manifest, state


def prepare(job, args, images, labels, population, metric_dirs, mapping):
    predictions = {cid: p.resolve() for cid, p in collect(job["prediction"]).items()}
    if set(predictions) != population or not population <= set(images) or not population <= set(labels):
        raise ValueError("full frozen population / predictions / DWI / GT coverage mismatch; no intersection")
    metrics, association = choose_metrics(job, predictions, metric_dirs, mapping)
    job.update(metrics=metrics, association=association)
    rows, summary = read_metrics(metrics, population)
    macro_summary(rows, summary)
    info = dict(images={cid: images[cid] for cid in population}, labels={cid: labels[cid] for cid in population},
                predictions=predictions, rows=rows, summary=summary)
    load_selection(args.selection_json, info)
    if job["source"] == "standalone-h2former":
        check_h2(job, info, args.allow_pending)
    return info


def verify_volumes(info):
    import numpy as np
    selected = {e["case_id"]: e for e in info["selection"]["entries"]}
    counts, panels = {}, {}
    for cid in sorted(info["rows"]):
        packets = [_volume(info[key][cid]) for key in ("images", "labels", "predictions")]
        for image, values in packets:
            geometry(image)
            if image.GetNumberOfComponentsPerPixel() != 1 or values.ndim != 3 or not np.isfinite(values).all():
                raise ValueError(f"{cid}: requires finite scalar 3D volumes")
        check_geometry(packets[0][0], packets[1][0], cid)
        check_geometry(packets[0][0], packets[2][0], cid)
        raw, gt, pred = [p[1] for p in packets]
        counts[cid] = voxel_counts(pred, gt, foreground=1)
        if any(v is not None and v != counts[cid][k] for k, v in source_counts(info["rows"][cid], cid).items()):
            raise ValueError(f"{cid}: source TP/FP/FN disagree with saved masks")
        if cid in selected:
            panels[cid] = {z: (raw[z].copy(), gt[z].copy(), pred[z].copy())
                           for z in selected[cid]["display_slices"]}
    verify_selection(info)
    return counts, panels


def summary_figure(info, panels, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from report_visuals import BODY, TITLE, setup_font, wrap, dice_label, save_figure
    setup_font()
    fig, axes = plt.subplots(18, 4, figsize=(16, 44), squeeze=False)
    try:
        for case_index, entry in enumerate(info["selection"]["entries"]):
            cid, slices = entry["case_id"], entry["display_slices"]
            for slot in range(3):
                z = slices[slot] if slot < len(slices) else None
                for col, title in enumerate(("DWI", "GT", "Saved pred", "TP / FP / FN")):
                    ax = axes[case_index * 3 + slot, col]
                    ax.set_xticks([]); ax.set_yticks([])
                    if slot == 0:
                        ax.set_title(title, fontsize=BODY)
                    if z is None:
                        ax.text(.5, .5, "No additional frozen slice", ha="center", va="center", transform=ax.transAxes, fontsize=BODY)
                    else:
                        raw, gt, pred = panels[cid][z]
                        value = _normalize_display(raw) if col in (0, 3) else gt if col == 1 else pred
                        ax.imshow(value, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                        if col == 3:
                            ax.imshow(_overlay_rgba(gt, pred), interpolation="nearest")
                    if col == 0:
                        label = f"{cid} | slice {z}" if z is not None else cid
                        if slot == 0:
                            label += f"\n{entry['size_group']} / baseline-{entry['baseline_role']}\ncurrent Dice {dice_label(info['rows'][cid]['dice'])}"
                        ax.set_ylabel(wrap(label, 30), fontsize=BODY)
        fig.suptitle("Frozen comparison cohort | saved results", fontsize=TITLE)
        fig.legend(handles=[Patch(color=c, label=t) for t, c in (("TP", "#1fd14d"), ("FP", "#ff2e2e"), ("FN", "#296bff"))],
                   loc="lower center", ncol=3, fontsize=BODY)
        fig.subplots_adjust(left=.20, right=.98, bottom=.035, top=.97, hspace=.45, wspace=.12)
        save_figure(fig, output / "summary.png", family="summary")
    finally:
        plt.close(fig)


def saved_report(job, info, output):
    counts, panels = verify_volumes(info)
    table = build_table(info["rows"], info["summary"], info["predictions"], info["labels"],
                        label_contract=lambda: 1, check_geometry=check_geometry,
                        loader=lambda pred, gt, cid, foreground, checker: counts[cid], geometry_verified=True)
    output.mkdir()
    note = (f"Saved-results report | {job['identity']}\nModel/run: {job['model']}\n"
            f"Prediction directory (snapshot identity retained): {job['prediction']}\nMetrics directory: {job['metrics']}\n"
            f"Association: {job['association']}\nSaved prediction checkpoint provenance: UNKNOWN; no checkpoint loaded.\n"
            f"Alignment state: {job.get('run_state', 'official saved result; checkpoint provenance UNKNOWN')}\n"
            f"All {len(info['rows'])} scalar 3D DWI/GT/pred geometries and binary 0/1 masks checked.\n"
            "Clinical metrics retained from saved CSV/JSON. Only missing TP/FP/FN are filled from saved masks; source counts cross-checked.\n"
            "Single fold 0; not five-fold OOF or clinical improvement evidence.\n"
            "Six frozen cases, up to three fixed slices each; empty slots are never substituted.\n"
            "Saved-only summary: DWI / GT / saved prediction / TP(green), FP(red), FN(blue). No architecture.\n")
    (output / "report.txt").write_text(note + save_selection(info, output), encoding="utf-8")
    text = export_table(table, output)
    (output / "metrics_table.txt").write_text(text, encoding="utf-8")
    with (output / "report.txt").open("a", encoding="utf-8") as handle:
        handle.write(text)
    summary_figure(info, panels, output)
    from report_visuals import validate_visual_outputs
    validate_visual_outputs(output, ["summary", "metrics_table"])
    check_selection_output(info, output)


def feature_command(job, args, target):
    if job["source"] == "official-nnunet":
        trainer = job["model"].name.split("__")[0]
        try:
            model_identity(job["model"])
        except ValueError as error:
            return None, str(error)
        if trainer not in SUPPORTED_TRAINERS and not getattr(args, "features_only", False):
            return None, "additional Trainer supported only with --features-only"
        extension = getattr(args, "trainer_extensions", {}).get(trainer)
        if trainer in FEATURE_TRAINERS and extension is None:
            return None, "correct Trainer source missing; supply --trainer-extension-map or --trainer-extension-dir"
        if extension is not None and not (extension / (trainer + ".py")).is_file():
            return None, "exact external Trainer class file missing: " + str(extension / (trainer + ".py"))
        checkpoint = job["model"] / "fold_0" / "checkpoint_best.pth"
        if not checkpoint.is_file():
            return None, "diagnostic checkpoint_best.pth missing; no substitution"
        extra = []
        if extension is not None:
            extra += ["--trainer-extension-dir", str(extension)]
        if trainer == TRAINER:
            extra += ["--prediction-checkpoint-declaration", "UNKNOWN; saved prediction checkpoint is unverified; named checkpoint is diagnostic only"]
    else:
        from report_comparison_selection import file_hash
        use_current = getattr(args, "use_current_h2_checkpoints", False)
        if use_current:
            validate_backfill_flags(args)
            from standalone_h2former_report import current_h2_checkpoint
            try:
                checkpoint = current_h2_checkpoint(job["model"], job["prediction"], job["manifest_data"])
            except ValueError as error:
                return None, str(error)
        else:
            metadata = job["manifest_data"]["checkpoint"]
            value, digest = metadata.get("path"), metadata.get("sha256", metadata.get("checkpoint_sha256"))
            if not isinstance(value, str) or not Path(value).is_absolute():
                return None, "H2 exact manifest checkpoint path missing"
            checkpoint = Path(value).resolve()
            if not checkpoint.is_file() or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                return None, "H2 manifest lacks existing exact checkpoint + SHA256 proof; historic mutable best/latest names cannot prove snapshot weights"
            if file_hash(checkpoint) != digest:
                return None, "H2 manifest checkpoint hash mismatch; snapshot features skipped"
        extra = ["--manifest", str(job["manifest"]), "--config", str(job["config"])]
        if use_current:
            extra += ["--diagnostic-current-checkpoint", "--final-h2-only"]
            if job.get("existing_report"):
                extra += ["--existing-report", str(job["existing_report"])]
        if args.allow_pending:
            extra.append("--allow-pending")
    if getattr(args, "features_only", False):
        extra.append("--features-only")
    command = [sys.executable, str(CODE / "generate_nnunet_result_report.py"), "--source", job["source"],
               "--model-dir", str(job["model"]), "--fold", "0", "--images-dir", str(args.images_dir),
               "--labels-dir", str(args.labels_dir), "--prediction-dir", str(job["prediction"]),
               "--metrics-dir", str(job["metrics"]), "--selection-json", str(args.selection_json),
               "--checkpoint", str(checkpoint), "--output-dir", str(target / "feature_diagnostics"),
               "--device", args.device, *extra]
    return command, "diagnostic features; existing generator may add architecture in feature_diagnostics; source remains UNKNOWN"


def features(job, args, target, log):
    command, note = feature_command(job, args, target)
    if command is None:
        return "skipped: " + note, False
    env = dict(os.environ)
    env["nnUNet_extTrainer"] = str(CODE / "nnunet_ext_trainers")
    with log.open("a", encoding="utf-8") as handle:
        handle.write("\nFeature command: " + json.dumps(command) + "\n" + note + "\n")
        handle.flush()
        result = subprocess.run(command, cwd=CODE, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        return f"failed: generator exit {result.returncode}; saved-only report preserved", True
    return "generated: " + note, False


def write_summary(output, records, args):
    document = dict(mode="features-only backfill" if getattr(args, "features_only", False) else "saved-results + optional features" if args.features else "saved-results",
                    dry_run=args.dry_run, records=records)
    if args.dry_run:
        print(json.dumps(document, ensure_ascii=False, indent=2))
        return
    (output / "batch_summary.json").write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = ("status", "source", "model", "prediction", "metrics", "association", "report", "existing_report", "features", "reason", "log")
    with (output / "batch_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(records)
    (output / "batch.log").write_text("\n".join(
        f"{r['status']}: {r.get('prediction', r.get('model', ''))}: {r.get('reason', '')}"
        for r in records) + "\n", encoding="utf-8")


def extension_map(args):
    document = read_json(args.trainer_extension_map) if args.trainer_extension_map else {}
    if not isinstance(document, dict):
        raise ValueError("Trainer extension map requires a JSON object")
    result = {}
    for trainer, value in document.items():
        if trainer not in SUPPORTED_TRAINERS + FEATURE_TRAINERS or not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError("Trainer extension map requires supported Trainer -> absolute source directory")
        result[trainer] = Path(value).resolve()
    if args.trainer_extension_dir:
        if not args.trainer_extension_dir.is_absolute():
            raise ValueError("Trainer extension directory must be absolute")
        for trainer in FEATURE_TRAINERS:
            result.setdefault(trainer, args.trainer_extension_dir.resolve())
    return result


def validate_backfill_flags(args):
    if getattr(args, "final_h2_only", False) and not getattr(args, "features_only", False):
        raise ValueError("--final-h2-only requires --features-only")
    if getattr(args, "use_current_h2_checkpoints", False) and not (
            getattr(args, "features_only", False) and getattr(args, "final_h2_only", False)):
        raise ValueError("--use-current-h2-checkpoints requires --features-only --final-h2-only")


def select_final_records(previous):
    """Pure metadata policy for final presentation/backfill; no metric or epoch ranking."""
    from standalone_h2former_report import final_h2_slot
    kept, excluded, groups = [], [], {}
    for row in previous:
        if row.get("status") not in ("saved", "feature_failed"):
            continue
        if row.get("source") == "official-nnunet":
            kept.append(row)
        elif row.get("source") == "standalone-h2former":
            try:
                slot = final_h2_slot(Path(row["model"]), Path(row["prediction"]))
                groups.setdefault((str(Path(row["model"])).casefold(), slot), []).append(row)
            except ValueError as error:
                excluded.append(dict(row, exclusion_reason=str(error)))
    for candidates in groups.values():
        preferred = [r for r in candidates if Path(r["prediction"]).parent.name.startswith("full_volume_predictions_")]
        winners = preferred if preferred else candidates
        if len(winners) != 1:
            excluded.extend(dict(r, exclusion_reason="ambiguous final exports for the same run/slot; no metric-based selection") for r in candidates)
            continue
        winner = winners[0]
        kept.append(winner)
        excluded.extend(dict(r, exclusion_reason="distinct alternate export superseded by full_volume_predictions final export; not averaged")
                        for r in candidates if r is not winner)
    order = {id(r): i for i, r in enumerate(previous)}
    kept.sort(key=lambda r: order[id(r)])
    return kept, excluded


def write_final_allowlist(output, previous, summary):
    kept, excluded = select_final_records(previous)
    from report_comparison_selection import file_hash
    document = dict(policy="exact user final exports; no Dice/epoch ranking; alternate exports never averaged",
                    source_summary=str(summary), source_summary_sha256=file_hash(summary),
                    retained_count=len(kept), excluded_count=len(excluded),
                    official_count=sum(r["source"] == "official-nnunet" for r in kept),
                    h2_count=sum(r["source"] == "standalone-h2former" for r in kept),
                    retained=[dict(r, report_basename=Path(r["report"]).name) for r in kept], excluded=excluded,
                    unrun=[r for r in previous if r.get("source") in ("official-nnunet", "standalone-h2former")
                           and r.get("status") not in ("saved", "feature_failed")])
    (output / "final_report_allowlist.json").write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, rows in (("final_report_allowlist.csv", document["retained"]), ("final_report_excluded.csv", excluded)):
        fields = list(dict.fromkeys(key for row in rows for key in row)) or ["source", "report", "exclusion_reason"]
        with (output / name).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
    return document


def backfill(args):
    """Replay completed saved jobs only. Never discover/evaluate masks or rewrite reports."""
    validate_backfill_flags(args)
    summary = args.existing_summary
    if summary is None:
        summary = args.output_dir.with_name(args.output_dir.name.removesuffix("_features_final_v1").removesuffix("_features_v1")) / "batch_summary.csv"
    summary = summary.resolve()
    with summary.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"status", "source", "model", "prediction", "metrics", "report"} <= set(reader.fieldnames or []):
            raise ValueError("existing summary missing saved job sources")
        previous = list(reader)
    eligible, final_excluded = select_final_records(previous) if getattr(args, "final_h2_only", False) else (previous, [])
    selection = validate(read_json(args.selection_json))
    if len(selection["population"]) != 19:
        raise ValueError("backfill requires frozen 19-case cohort")
    sources = [summary.parent, CODE, args.images_dir, args.labels_dir, args.selection_json,
               *args.trainer_extensions.values()]
    jobs = []
    for row in eligible:
        if row["status"] not in ("saved", "feature_failed"):
            continue  # Three genuinely unrun reports are not new jobs.
        paths = {}
        for key in ("model", "prediction", "metrics", "report"):
            path = Path(row[key])
            if not path.is_absolute() or not path.is_dir():
                raise ValueError("completed job needs existing absolute " + key + ": " + row[key])
            paths[key] = path.resolve()
        if validate(read_json(paths["report"] / "comparison_selection.json")) != selection:
            raise ValueError("existing report frozen selection differs from supplied selection")
        sources.extend(paths.values())
        jobs.append((row, paths))
    output = protected_output(args.output_dir, sources)
    if output.parent != summary.parent.parent:
        raise ValueError("feature output must be a fresh sibling of existing report root")
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=False)
        (output / "logs").mkdir()
        if getattr(args, "final_h2_only", False):
            write_final_allowlist(output, previous, summary)
    records = [dict(row, features="not run: existing report was not completed; no backfill attempted")
               for row in previous if row["status"] not in ("saved", "feature_failed")]
    records.extend(dict(row, status="feature_excluded", features="not run: excluded by final H2 policy", reason=row["exclusion_reason"])
                   for row in final_excluded)
    for number, (row, paths) in enumerate(jobs, 1):
        target = output / (str(number).zfill(3) + "_" + paths["report"].name)
        log = output / "logs" / f"job_{number:03d}.log"
        record = dict(row, report=str(target), existing_report=str(paths["report"]), log=str(log))
        records.append(record)
        job = dict(source=row["source"], model=paths["model"], prediction=paths["prediction"], metrics=paths["metrics"], existing_report=paths["report"])
        try:
            existing_features = paths["report"] / "feature_diagnostics"
            if row.get("features", "").startswith("generated:") and all(
                    (existing_features / name).is_file() for name in
                    ("encoder_stages_heatmap.png", "summary.png", "feature_channels.png", "feature_channels_64x64.png", "report.txt")):
                record.update(status="feature_skipped", features="skipped: existing completed feature diagnostics", reason="original feature files retained")
                continue
            if job["source"] == "standalone-h2former":
                job.update(manifest=paths["prediction"].parent / "prediction_manifest.json",
                           config=paths["model"] / "resolved_config.json")
                job["manifest_data"] = read_json(job["manifest"])
            elif job["source"] != "official-nnunet":
                raise ValueError("unsupported saved source")
            if args.dry_run:
                record.update(status="planned", features="PENDING exact checkpoint/source validation; no checkpoint reads")
            else:
                target.mkdir()
                record["features"], failed = features(job, args, target, log)
                record.update(status="feature_failed" if failed else "feature_skipped" if record["features"].startswith("skipped:") else "feature_saved", reason="diagnostics only; original reports/metrics preserved")
        except Exception as error:
            record.update(status="feature_failed", features="failed: " + str(error))
            if not args.dry_run:
                log.write_text(traceback.format_exc(), encoding="utf-8")
        if not args.dry_run:
            write_summary(output, records, args)
    write_summary(output, records, args)
    return int(any(r["status"] == "feature_failed" for r in records))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-roots", type=Path, nargs="+")
    for name in ("images-dir", "labels-dir", "selection-json", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--metrics-map", type=Path, help="JSON exact absolute prediction dir -> metrics dir")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--features", action="store_true")
    parser.add_argument("--features-only", action="store_true")
    parser.add_argument("--final-h2-only", action="store_true", help="retain only exact user final H2 exports; official saved jobs retained")
    parser.add_argument("--use-current-h2-checkpoints", action="store_true", help="DIAGNOSTIC ONLY current named best/latest slots; requires final features-only mode")
    parser.add_argument("--existing-summary", type=Path, help="existing completed batch_summary.csv; no scan or saved-report regeneration")
    parser.add_argument("--trainer-extension-map", type=Path, help="JSON Trainer -> absolute real SERVER source directory")
    parser.add_argument("--trainer-extension-dir", type=Path, help="default real source directory for added Trainers; map entries take precedence")
    parser.add_argument("--allow-pending", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="metadata/coverage plan only; no output, voxels, checkpoints or models")
    args = parser.parse_args(argv)
    records = []
    try:
        validate_backfill_flags(args)
        args.trainer_extensions = extension_map(args)
        if args.features_only:
            args.features = True
            return backfill(args)
        if args.existing_summary or args.trainer_extensions:
            raise ValueError("existing summary/external source overrides require --features-only")
        if not args.results_roots:
            raise ValueError("--results-roots required for ordinary batch reporting")
        roots = sorted({p.resolve() for p in args.results_roots})
        args.images_dir, args.labels_dir = args.images_dir.resolve(), args.labels_dir.resolve()
        args.selection_json = args.selection_json.resolve()
        if not roots or any(not p.is_dir() for p in [*roots, args.images_dir, args.labels_dir]):
            raise ValueError("all result and raw roots must be existing directories")
        mapping = metrics_map(args.metrics_map, roots)
        selection = validate(read_json(args.selection_json))
        if len(selection["population"]) != 19:
            raise ValueError("Dataset501 fold-0 batch requires all 19 frozen cases")
        population = set(selection["population"])
        images = {cid: p.resolve() for cid, p in collect(args.images_dir, input_channel=True).items()}
        labels = {cid: p.resolve() for cid, p in collect(args.labels_dir).items()}
        sources = [*roots, args.images_dir, args.labels_dir, args.selection_json.parent, CODE,
                   *mapping.keys(), *mapping.values(), *images.values(), *labels.values()]
        if args.metrics_map is not None:
            sources.append(args.metrics_map.resolve())
        git_file = CODE / ".git"
        if git_file.is_file():
            git_dir = Path(git_file.read_text(encoding="utf-8").strip().removeprefix("gitdir: "))
            if not git_dir.is_absolute():
                git_dir = CODE / git_dir
            sources.extend(p.parent for p in [git_dir.resolve(), *git_dir.resolve().parents] if p.name == ".git")
        output = protected_output(args.output_dir, sources)
        index = scan(roots, records)
        jobs = discover(index, records)
        metric_dirs = {p.parent.resolve() for p in index["case_metrics.csv"]}
        protected_output(output, [*sources, *metric_dirs, *(j["model"] for j in jobs)])
        if not args.dry_run:
            output.mkdir(parents=True, exist_ok=False)
            (output / "logs").mkdir()
        if not jobs:
            records.append(dict(status="error", reason="no jobs in supported saved-result scope"))
        for number, job in enumerate(jobs, 1):
            snapshot = job["prediction"].parent.name if job["source"] == "standalone-h2former" else "validation"
            slug = re.sub(r"[^A-Za-z0-9_-]", "_", job["model"].name)[:65]
            digest = hashlib.sha256(str(job["prediction"]).encode()).hexdigest()[:10]
            target = output / f"{number:03d}_{slug}_{snapshot[:45]}_{digest}"
            log = output / "logs" / f"job_{number:03d}.log"
            record = dict(source=job["source"], model=str(job["model"]), prediction=str(job["prediction"]),
                          report=str(target), features="not requested", log=str(log))
            records.append(record)
            try:
                info = prepare(job, args, images, labels, population, metric_dirs, mapping)
                record.update(metrics=str(job["metrics"]), association=job["association"])
            except Exception as error:
                record.update(status="skipped", reason=str(error))
                if not args.dry_run:
                    log.write_text(traceback.format_exc(), encoding="utf-8")
                    write_summary(output, records, args)
                continue
            if args.dry_run:
                record.update(status="planned", reason="metadata coverage checked; full-volume geometry/binary masks/frozen hashes PENDING")
                if args.features:
                    record["features"] = "PENDING adapter/checkpoint proof; no model or checkpoint read"
                continue
            try:
                saved_report(job, info, target)
                record.update(status="saved", reason="saved-only report complete; clinical metrics reused")
                log.write_text(record["reason"] + "\n", encoding="utf-8")
            except Exception as error:
                record.update(status="failed", reason=str(error))
                log.write_text(traceback.format_exc(), encoding="utf-8")
                if target.is_dir():
                    (target / "FAILED.txt").write_text(str(error) + "\nPartial report; see batch summary/log.\n", encoding="utf-8")
                write_summary(output, records, args)
                continue
            if args.features:
                try:
                    record["features"], failed = features(job, args, target, log)
                    if failed:
                        record["status"] = "feature_failed"
                except Exception as error:
                    record.update(status="feature_failed", features="failed: " + str(error) + "; saved-only report preserved")
                    with log.open("a", encoding="utf-8") as handle:
                        handle.write(traceback.format_exc())
                with (target / "report.txt").open("a", encoding="utf-8") as handle:
                    handle.write("\nOptional feature diagnostics: " + record["features"] + "\n")
            write_summary(output, records, args)
        write_summary(output, records, args)
        failed = any(r["status"] in ("error", "skipped", "failed", "feature_failed") for r in records)
        if not args.dry_run:
            print(f"Batch summary: {output / 'batch_summary.csv'}; exit={int(failed)}")
        return int(failed)
    except Exception as error:
        parser.exit(2, f"batch report: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
