"""History-selected exploratory CV70 preparation and complete five-fold aggregation.

Pure JSON/CSV metadata utilities: never import a model or read checkpoint content.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
import re

ROOT = Path(__file__).resolve().parent
EXPERIMENT_CLASS = "history-selected exploratory OOF; validation used for early stopping"
PROVENANCE_CAVEAT = "Source paths declare historical five-fold OOF; per-fold predictions/checkpoints not independently audited."
PLANS_POLICY = "Original 95-case plans/preprocessing statistics intentionally retained; no preprocessing retraining."
MARKERS = ("Dataset501_StrokeLesion", "nnUNetTrainer__nnUNetPlans__2d", "crossval_results_folds_0_1_2_3_4")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def digest(protocol):
    return config_digest({k: v for k, v in protocol.items() if k != "protocol_sha256"})


def case_id(path):
    return PurePosixPath(str(path).replace("\\", "/")).name.removesuffix(".nii.gz")


def original_ids(folds):
    if not isinstance(folds, list) or len(folds) != 5:
        raise ValueError("Expected five original folds")
    cohort = None
    vals = []
    for f in folds:
        train, val = f["train"], f["val"]
        if (len(train), len(val)) != (76, 19) or len(set(train)) != 76 or len(set(val)) != 19 or set(train) & set(val):
            raise ValueError("Invalid original 76/19 membership")
        current = set(train) | set(val)
        if any(not isinstance(x, str) or not x for x in current):
            raise ValueError("Invalid case ID")
        if cohort is not None and current != cohort:
            raise ValueError("Original cohort differs between folds")
        cohort = current
        vals += val
    if len(cohort) != 95 or len(set(vals)) != 95 or set(vals) != cohort:
        raise ValueError("Original validation must cover 95 once")
    return cohort


def finite_unit(value):
    if isinstance(value, bool):
        raise ValueError("Boolean metric")
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError("Invalid numeric metric") from None
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Metric must be finite in [0,1]")
    return value


def frozen_folds(retained, seed):
    shuffled = sorted(retained)
    random.Random(seed).shuffle(shuffled)
    return [{"fold": i, "val": shuffled[i*14:(i+1)*14],
             "train": sorted(set(retained) - set(shuffled[i*14:(i+1)*14]))} for i in range(5)]


def validate(protocol, original):
    cohort = original_ids(original)
    if protocol.get("version") != 1 or protocol.get("experiment_class") != EXPERIMENT_CLASS:
        raise ValueError("Unexpected CV70 protocol version/class")
    if protocol.get("protocol_sha256") != digest(protocol):
        raise ValueError("CV70 protocol digest mismatch")
    if protocol.get("original_splits_membership_sha256") != config_digest(original):
        raise ValueError("Original split exact membership/order mismatch")
    seed = protocol["split_seed"]
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Invalid split seed")
    source = protocol["ranking_source"]
    for key in ("summary_sha256", "original_splits_sha256"):
        if not re.fullmatch("[0-9a-f]{64}", source.get(key, "")):
            raise ValueError("Missing source SHA256")
    if source.get("caveat") != PROVENANCE_CAVEAT or source.get("markers") != list(MARKERS):
        raise ValueError("Missing ranking provenance caveat/markers")
    declared_parts = PurePosixPath(str(source.get("prediction_directory", "")).replace("\\", "/")).parts
    if tuple(declared_parts[-3:]) != MARKERS:
        raise ValueError("Ranking directory declaration lacks original OOF markers")
    if protocol.get("plans_policy") != PLANS_POLICY:
        raise ValueError("Missing original plans policy")
    ranking = protocol["ranking"]
    if len(ranking) != 95 or {r["case_id"] for r in ranking} != cohort:
        raise ValueError("Ranking must contain unique original95")
    numeric = [{"case_id": r["case_id"], "dice": finite_unit(r["dice"])} for r in ranking]
    if ranking != sorted(numeric, key=lambda r: (r["dice"], r["case_id"])):
        raise ValueError("Ranking must be sorted Dice then ID")
    excluded = [r["case_id"] for r in numeric[:25]]
    retained = sorted(cohort - set(excluded))
    if protocol["excluded"] != excluded or protocol["retained"] != retained or len(retained) != 70:
        raise ValueError("Exclusion/retained cohort differs from ranking")
    if protocol["folds"] != frozen_folds(retained, seed):
        raise ValueError("Frozen fivefold56/14 partition mismatch")
    return protocol


def load_protocol(path, original):
    return validate(read_json(path), original)


def protect_output(output, sources, protected_roots=()):
    output = Path(output).resolve()
    for source in sources:
        source = Path(source).resolve()
        if output == source or output in source.parents:
            raise ValueError("Output overlaps source metadata")
    for root in protected_roots:
        root = Path(root).resolve()
        if output == root or root in output.parents or output in root.parents:
            raise ValueError("Output overlaps protected source root")
    return output


def prepare(summary, splits, output, split_seed=0):
    summary, splits = Path(summary).resolve(), Path(splits).resolve()
    output = protect_output(output, (summary, splits), (summary.parent, splits.parent))
    if output.exists():
        raise FileExistsError("Protocol output exists; refusing overwrite")
    if output.name.lower() in ("splits_final.json", "summary.json"):
        raise ValueError("Refusing reserved source metadata output name")
    original = read_json(splits)
    cohort = original_ids(original)
    ranking = []
    prediction_dirs = set()
    for record in read_json(summary)["metric_per_case"]:
        path = str(record["prediction_file"]).replace("\\", "/")
        parts = PurePosixPath(path).parts
        if tuple(parts[-4:-1]) != MARKERS or not path.endswith(".nii.gz"):
            raise ValueError("Ranking source must declare original Dataset501 Trainer five-fold OOF path")
        prediction_dirs.add(str(PurePosixPath(path).parent))
        ranking.append({"case_id": case_id(path), "dice": finite_unit(record["metrics"]["1"]["Dice"])})
    if len(prediction_dirs) != 1 or len(ranking) != 95 or len({r["case_id"] for r in ranking}) != 95 or {r["case_id"] for r in ranking} != cohort:
        raise ValueError("Ranking must contain unique95 matching original cohort from one OOF directory")
    ranking.sort(key=lambda r: (r["dice"], r["case_id"]))
    excluded = [r["case_id"] for r in ranking[:25]]
    retained = sorted(cohort - set(excluded))
    protocol = {"version": 1, "experiment_class": EXPERIMENT_CLASS, "plans_policy": PLANS_POLICY,
                "split_seed": split_seed, "original_splits_membership_sha256": config_digest(original),
                "ranking_source": {"summary_path": str(summary), "summary_sha256": hash_file(summary),
                                   "original_splits_path": str(splits), "original_splits_sha256": hash_file(splits),
                                   "prediction_directory": next(iter(prediction_dirs)),
                                   "markers": list(MARKERS), "caveat": PROVENANCE_CAVEAT},
                "ranking": ranking, "excluded": excluded, "retained": retained,
                "folds": frozen_folds(retained, split_seed)}
    protocol["protocol_sha256"] = digest(protocol)
    validate(protocol, original)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(protocol, stream, indent=2, allow_nan=False)
    return protocol


def metric_values(row):
    values = {}
    for key in ("TP", "FP", "FN"):
        value = float(row[key + "_voxels"])
        if not math.isfinite(value) or value < 0 or not value.is_integer():
            raise ValueError("Invalid voxel count")
        values[key] = int(value)
    tp, fp, fn = (values[k] for k in ("TP", "FP", "FN"))
    if 2*tp+fp+fn == 0:
        raise ValueError("Undefined empty-case Dice/IoU; cannot aggregate finite metrics")
    dice, iou = finite_unit(row["dice_0_to_1"]), finite_unit(row["IoU_0_to_1"])
    if not math.isclose(dice, 2*tp/(2*tp+fp+fn), rel_tol=1e-6, abs_tol=1e-8) or not math.isclose(iou, tp/(tp+fp+fn), rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError("Dice/IoU do not match TP/FP/FN")
    values.update(dice=dice, iou=iou)
    values["recall"] = tp/(tp+fn) if tp+fn else None
    values["precision"] = tp/(tp+fp) if tp+fp else None
    values["paper_f2"] = 5*tp/(5*tp+4*fp+fn)
    values["avd_percent"] = 100*abs(fp-fn)/(tp+fn) if tp+fn else None
    return values


def resolve_protocol_path(config, config_path=None):
    path = Path(config["cv70_protocol"])
    return (path if path.is_absolute() else (Path(config_path).resolve().parent if config_path else ROOT) / path).resolve()


def aggregate(config_path, seed=0, original_path=None):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Invalid run seed")
    config = read_json(config_path)
    protocol_path = resolve_protocol_path(config, config_path)
    original = read_json(original_path or ROOT / "original_splits.json")
    p = load_protocol(protocol_path, original)
    output = Path(config["results_parent"]).resolve() / f"seed_{seed}"
    protect_output(output, (config_path, protocol_path, original_path or ROOT / "original_splits.json"),
                   (config["raw_root"], config["preprocessed_root"], Path(config["pretrained_checkpoint"]).resolve().parent, ROOT))
    csv_path, json_path = output / "oof70_case_metrics.csv", output / "oof70_summary.json"
    if csv_path.exists() or json_path.exists():
        raise FileExistsError("Aggregation output exists; refusing overwrite")
    all_rows, metrics = [], []
    pretrain = protected = None
    for f in p["folds"]:
        directory = output / f"fold_{f['fold']}"
        m = read_json(directory / "experiment_manifest.json")
        if (m.get("status") != "COMPLETE" or m.get("canonical_metadata_unchanged") is not True
                or m.get("fold") != f["fold"] or m.get("seed") != seed
                or m.get("experiment_class") != EXPERIMENT_CLASS or m.get("plans_policy") != PLANS_POLICY
                or m.get("ranking_provenance") != p["ranking_source"]
                or m.get("nnunetv2_version") != "2.8.1"
                or m.get("cohort_source_protocol_sha256") != p["protocol_sha256"]
                or m.get("config") != config or m.get("config_sha256") != config_digest(config)
                or m.get("cv70_protocol_sha256") != p["protocol_sha256"]
                or m.get("cv70_protocol_file_sha256") != hash_file(protocol_path)
                or m.get("train") != f["train"] or m.get("selection_validation") != f["val"]
                or m.get("evaluation", {}).get("evaluation_cases") != f["val"]
                or m.get("evaluation", {}).get("selection_cases") != f["val"]):
            raise ValueError("Incomplete or mismatched fold manifest")
        current_pretrain = m.get("pretrained_sha256", "")
        current_protected = m.get("protected_hashes")
        if not re.fullmatch("[0-9a-f]{64}", current_pretrain) or not current_protected:
            raise ValueError("Missing pretrain/canonical source hashes")
        if pretrain is not None and (current_pretrain != pretrain or current_protected != protected):
            raise ValueError("Different pretrain/canonical source hashes across folds")
        pretrain, protected = current_pretrain, current_protected
        with (directory / "evaluation14_case_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 14 or len({r["case_id"] for r in rows}) != 14 or {r["case_id"] for r in rows} != set(f["val"]):
            raise ValueError("Fold metrics must contain exactly its validation14 once")
        for row in rows:
            if row.get("selection14_member") != "True":
                raise ValueError("Evaluation14 must equal selection14")
            metric = metric_values(row)
            metrics.append(metric)
            all_rows.append({"fold": f["fold"], **row})
    if len(all_rows) != 70 or {r["case_id"] for r in all_rows} != set(p["retained"]):
        raise ValueError("Expected exact unique retained70 OOF")
    report = {"status": "COMPLETE", "experiment_class": EXPERIMENT_CLASS, "plans_policy": PLANS_POLICY,
              "ranking_provenance": p["ranking_source"], "cases": 70, "seed": seed,
              "config_sha256": config_digest(config), "cv70_protocol_sha256": p["protocol_sha256"],
              "cv70_protocol_file_sha256": hash_file(protocol_path), "pretrained_sha256": pretrain,
              "protected_hashes": protected, "canonical_metadata_unchanged": True}
    for key, label in (("dice", "dice_0_to_1"), ("iou", "iou_0_to_1"), ("recall", "recall_0_to_1"),
                       ("precision", "precision_0_to_1"), ("paper_f2", "paper_f2_0_to_1"), ("avd_percent", "avd_percent")):
        defined = [m[key] for m in metrics if m[key] is not None]
        report["macro_" + label] = sum(defined)/len(defined) if defined else None
        report[label + "_defined_cases"] = len(defined)
    for key in ("TP", "FP", "FN"):
        report[key + "_voxels_total"] = sum(m[key] for m in metrics)
    # Validate everything before creating either output; summary is the completion marker.
    # Reserve both exclusively before writing. On failure remove only our own newly-created files.
    created = []
    try:
        with csv_path.open("x", encoding="utf-8-sig", newline="") as csv_stream:
            created.append(csv_path)
            with json_path.open("x", encoding="utf-8") as json_stream:
                created.append(json_path)
                writer = csv.DictWriter(csv_stream, fieldnames=list(all_rows[0]))
                writer.writeheader()
                writer.writerows(sorted(all_rows, key=lambda r: r["case_id"]))
                csv_stream.flush()
                json.dump(report, json_stream, indent=2, allow_nan=False)
    except BaseException:
        for path in reversed(created):
            path.unlink()
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--summary", required=True)
    prepare_parser.add_argument("--splits", required=True)
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.add_argument("--split-seed", type=int, default=0)
    check = sub.add_parser("check")
    check.add_argument("--protocol", required=True)
    check.add_argument("--splits", default=str(ROOT / "original_splits.json"))
    agg = sub.add_parser("aggregate")
    agg.add_argument("--config", default=str(ROOT / "experiment_config.cv70.json"))
    agg.add_argument("--seed", type=int, default=0)
    agg.add_argument("--splits", default=str(ROOT / "original_splits.json"))
    args = parser.parse_args()
    if args.command == "prepare":
        p = prepare(args.summary, args.splits, args.output, args.split_seed)
        print("PREPARED: original95 excluded25 retained70; fivefold56/14; protocol", p["protocol_sha256"])
    elif args.command == "check":
        p = load_protocol(args.protocol, read_json(args.splits))
        print("CHECK_OK: frozen fivefold56/14; protocol", p["protocol_sha256"])
    else:
        report = aggregate(args.config, args.seed, args.splits)
        print("AGGREGATED: complete exploratory OOF70; macro Dice", report["macro_dice_0_to_1"])


if __name__ == "__main__":
    main()
