"""Isolated exploratory A/B fine-tuning; no edits to source data or canonical splits."""
import argparse
import csv
import hashlib
import importlib.metadata
import importlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from types import MethodType

ROOT = Path(__file__).resolve().parent
DATASET = "Dataset501_StrokeLesion"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def split_members(folds):
    return [(sorted(f["train"]), sorted(f["val"])) for f in folds]


def memberships(folds, excluded, arm):
    if len(folds) != 5:
        raise ValueError("Expected five original folds")
    all_ids = set(folds[0]["train"]) | set(folds[0]["val"])
    if len(all_ids) != 95 or len(excluded) != 15 or len(set(excluded)) != 15 or not set(excluded) <= all_ids:
        raise ValueError("Expected 95 original cases and 15 distinct exclusion IDs")
    for fold in folds:
        train, val = fold["train"], fold["val"]
        if len(set(train)) != len(train) or len(set(val)) != len(val) or set(train) & set(val):
            raise ValueError("Duplicate or overlapping fold membership")
        if set(train) | set(val) != all_ids:
            raise ValueError("Fold cohort mismatch")
    train = list(folds[0]["train"])
    evaluation = list(folds[0]["val"])
    selection = [x for x in evaluation if x not in excluded]
    if arm == "B":
        train = [x for x in train if x not in excluded]
    if len(train) != (76 if arm == "A" else 63) or len(selection) != 17 or len(evaluation) != 19:
        raise ValueError("Unexpected A/B membership sizes")
    if set(train) & set(evaluation):
        raise ValueError("Train/evaluation overlap")
    return train, selection, evaluation


def nested(child, parent):
    return child == parent or parent in child.parents


def resource_check(gpu):
    if os.name != "nt":
        raise RuntimeError("This launcher expects the verified Windows server")
    command = ("$c=@(1..3|ForEach-Object{(Get-CimInstance Win32_Processor|Measure-Object LoadPercentage -Average).Average;"
               "Start-Sleep -Milliseconds 400});$m=Get-CimInstance Win32_OperatingSystem;"
               "[pscustomobject]@{cpu=($c|Measure-Object -Maximum).Maximum;"
               "ram=100*(1-$m.FreePhysicalMemory/$m.TotalVisibleMemorySize)}|ConvertTo-Json -Compress")
    proc = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True, check=True)
    memory_cpu = json.loads(proc.stdout.strip())
    gpu_proc = subprocess.run(["nvidia-smi", f"--id={gpu}",
                               "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, check=True)
    compute, used, total = map(float, gpu_proc.stdout.strip().split(","))
    stats = {**memory_cpu, "selected_gpu": gpu, "gpu_compute": compute, "gpu_memory_percent": 100*used/total}
    print("RESOURCE_PREFLIGHT", json.dumps(stats), flush=True)
    if any(stats[k] >= 80 for k in ("cpu", "ram", "gpu_compute", "gpu_memory_percent")):
        raise RuntimeError("RESOURCE_GUARD: busy server; retry after resources are available")
    return stats


def verify(config, arm, seed, cv70_fold=None, config_path=None):
    raw = Path(config["raw_root"]).resolve()
    prep = Path(config["preprocessed_root"]).resolve()
    result = Path(config["results_parent"]).resolve() / f"seed_{seed}" / (arm if cv70_fold is None else f"fold_{cv70_fold}")
    checkpoint = Path(config["pretrained_checkpoint"]).resolve()
    if result.exists():
        raise ValueError(f"Run directory exists, refusing overwrite/resume: {result}")
    for protected in (raw, prep, checkpoint.parent, ROOT):
        if nested(result, protected) or nested(protected, result):
            raise ValueError(f"Result path overlaps source: {protected}")
    if importlib.metadata.version("nnunetv2") != "2.8.1":
        raise ValueError("Expected server nnunetv2==2.8.1")
    original = read_json(ROOT / "original_splits.json")
    actual_split = prep / DATASET / "splits_final.json"
    if split_members(read_json(actual_split)) != split_members(original):
        raise ValueError("Server canonical split differs from exported original")
    paths = {"split": actual_split, "plans": prep / DATASET / "nnUNetPlans.json",
             "dataset": prep / DATASET / "dataset.json"}
    if read_json(paths["plans"]) != read_json(ROOT / "expected_plans.json"):
        raise ValueError("Server plans changed")
    if read_json(paths["dataset"]) != read_json(ROOT / "expected_dataset.json"):
        raise ValueError("Server dataset metadata changed")
    cfg = read_json(paths["plans"])["configurations"]["2d"]
    if cfg["batch_size"] != 12 or cfg["patch_size"] != [512, 512] or cfg["data_identifier"] != "nnUNetPlans_2d":
        raise ValueError("Unexpected target configuration")
    source_cfg = read_json(config["pretrained_plans"])["configurations"]["2d"]
    if source_cfg["architecture"] != cfg["architecture"]:
        raise ValueError("Pretraining/target network architecture differs")
    if not checkpoint.is_file() or checkpoint.stat().st_size != config["pretrained_expected_bytes"]:
        raise ValueError("Pretraining checkpoint is missing or has changed size")
    for filename, expected in config["source_hashes"].items():
        if hash_file(ROOT / "trainers" / filename) != expected:
            raise ValueError(f"Trainer snapshot changed: {filename}")
    if cv70_fold is None:
        train, selection, evaluation = memberships(original, config["excluded_cases"], arm)
    else:
        from cv70 import load_protocol, resolve_protocol_path, protect_output
        if type(cv70_fold) is not int or not 0 <= cv70_fold <= 4:
            raise ValueError("CV70 fold must be 0..4")
        if config["trainer"] != "nnUNetTrainerTopK10EarlyStopping":
            raise ValueError("CV70 requires existing TopK10EarlyStopping")
        expected_sources = read_json(ROOT / "experiment_config.example.json")["source_hashes"]
        if config["source_hashes"] != expected_sources:
            raise ValueError("CV70 Trainer snapshot hash contract differs")
        cv_path = resolve_protocol_path(config, config_path)
        protect_output(result, (cv_path, config_path or ROOT / "experiment_config.json"))
        protocol = load_protocol(cv_path, original)
        fold = protocol["folds"][cv70_fold]
        train, selection, evaluation = fold["train"], fold["val"], fold["val"]
    protected_hashes = {str(p): hash_file(p) for p in paths.values()}
    return result, train, selection, evaluation, protected_hashes


def verify_trainer_source(trainer, trainer_module, trainer_name):
    source = Path(trainer_module.__file__).resolve()
    if source != (ROOT / "trainers" / (trainer_name + ".py")).resolve():
        raise ValueError(f"Unexpected Trainer source: {source}")
    if type(trainer) is not getattr(trainer_module, trainer_name):
        raise ValueError(f"Unexpected Trainer class: {type(trainer)}")
    return source


def bind_split(trainer, train, val, phase):
    def fixed_split(self):
        self.print_to_log_file(f"EXPLORATORY {phase}: {len(train)} train; {len(val)} validation/evaluation; canonical split unchanged")
        return list(train), list(val)
    trainer.do_split = MethodType(fixed_split, trainer)


def export_metrics(summary_path, evaluation, selection, output, cohort_label="evaluation19"):
    summary = read_json(summary_path)
    records = summary["metric_per_case"]
    from cv70 import case_id
    by_id = {case_id(r["prediction_file"]): r["metrics"]["1"] for r in records}
    if len(by_id) != len(records) or set(by_id) != set(evaluation):
        raise ValueError("Evaluation summary does not contain exactly the expected held-out cohort")
    for case, metric in by_id.items():
        if not math.isfinite(float(metric["Dice"])):
            raise ValueError(f"Nonfinite Dice: {case}")
    if cohort_label == "evaluation14":
        from cv70 import metric_values
        for metric in by_id.values():
            metric_values({"dice_0_to_1": metric["Dice"], "IoU_0_to_1": metric["IoU"],
                           "TP_voxels": metric["TP"], "FP_voxels": metric["FP"], "FN_voxels": metric["FN"]})
    elif cohort_label != "evaluation19":
        raise ValueError("Unexpected evaluation cohort label")
    with (output / (cohort_label + "_case_metrics.csv")).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "dice_0_to_1", "IoU_0_to_1", "TP_voxels", "FP_voxels", "FN_voxels", "selection14_member" if cohort_label == "evaluation14" else "selection17_member"])
        for case in sorted(evaluation):
            metric = by_id[case]
            writer.writerow([case, metric["Dice"], metric["IoU"], metric["TP"], metric["FP"], metric["FN"], case in selection])
    means = {"primary_original19_dice": sum(by_id[c]["Dice"] for c in evaluation)/len(evaluation),
             "secondary_remaining17_dice": sum(by_id[c]["Dice"] for c in selection)/len(selection),
             "evaluation_cases": evaluation, "selection_cases": selection,
             "experiment_class": "exploratory; selected from earlier OOF results"}
    if cohort_label == "evaluation14":
        from cv70 import EXPERIMENT_CLASS
        means = {"primary_validation14_dice": sum(by_id[c]["Dice"] for c in evaluation)/14,
                 "evaluation_cases": evaluation, "selection_cases": selection, "experiment_class": EXPERIMENT_CLASS}
    (output / "evaluation_summary.json").write_text(json.dumps(means, indent=2), encoding="utf-8")
    return means


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--arm", choices=("A", "B"))
    mode.add_argument("--cv70-fold", type=int, choices=range(5))
    parser.add_argument("--config", type=Path, default=ROOT / "experiment_config.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.seed < 0 or args.seed >= 2**32:
        parser.error("Seed must be between 0 and 2**32-1")
    config = read_json(args.config)
    result, train, selection, evaluation, original_hashes = verify(config, args.arm, args.seed, args.cv70_fold, args.config)
    cv_protocol = None
    if args.cv70_fold is not None:
        from cv70 import load_protocol, resolve_protocol_path, config_digest, EXPERIMENT_CLASS, PLANS_POLICY
        cv_path = resolve_protocol_path(config, args.config)
        cv_protocol = load_protocol(cv_path, read_json(ROOT / "original_splits.json"))
    print(f"ARM={args.arm}; fold={args.cv70_fold}; seed={args.seed}; train={len(train)}; checkpoint selection={len(selection)}; final evaluation={len(evaluation)}")
    print("RESULT_ROOT", result)
    print("PRETRAIN", config["pretrained_checkpoint"])
    if args.check_only:
        print("CHECK_OK: metadata only; no image/checkpoint/model loading")
        return
    resources = resource_check(config["gpu"])
    os.environ["nnUNet_raw"] = config["raw_root"]
    os.environ["nnUNet_preprocessed"] = config["preprocessed_root"]
    os.environ["nnUNet_results"] = str(result)
    os.environ["nnUNet_extTrainer"] = str(ROOT / "trainers")
    os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu"]
    sys.path.insert(0, str(ROOT / "trainers"))
    import numpy as np
    import torch
    from torch.backends import cudnn
    from nnunetv2.run.run_training import get_trainer_from_args, maybe_load_checkpoint
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = False
    cudnn.benchmark = True
    result.mkdir(parents=True, exist_ok=False)
    protocol = {"arm": args.arm, "seed": args.seed, "config": config, "train": train,
                "selection_validation": selection, "evaluation": evaluation, "protected_hashes": original_hashes,
                "pretrained_sha256": hash_file(config["pretrained_checkpoint"]),
                "resources_at_start": resources, "status": "STARTED",
                "randomness_limit": "Main-process RNG seeded; official nondeterministic augmentation workers and CUDA remain nondeterministic. Repeat paired runs; not bitwise reproducibility."}
    if cv_protocol is not None:
        protocol.update({"fold": args.cv70_fold, "experiment_class": EXPERIMENT_CLASS,
                         "plans_policy": PLANS_POLICY, "ranking_provenance": cv_protocol["ranking_source"],
                         "cohort_source_protocol_sha256": cv_protocol["protocol_sha256"],
                         "cv70_protocol_sha256": cv_protocol["protocol_sha256"],
                         "cv70_protocol_file_sha256": hash_file(cv_path),
                         "config_sha256": config_digest(config), "nnunetv2_version": "2.8.1"})
    protocol_path = result / "experiment_manifest.json"
    def save():
        protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    save()
    try:
        # nnU-Net's external lookup cleans up newly imported modules; retain this one first.
        trainer_module = importlib.import_module(config["trainer"])
        trainer = get_trainer_from_args(DATASET, "2d", 0 if args.cv70_fold is None else args.cv70_fold, config["trainer"], "nnUNetPlans", False, torch.device("cuda"))
        source = verify_trainer_source(trainer, trainer_module, config["trainer"])
        protocol["trainer_source"] = str(source)
        if not nested(Path(trainer.output_folder).resolve(), result):
            raise ValueError("Trainer output escaped isolated result root")
        bind_split(trainer, train, selection, "TRAINING_SELECTION17" if cv_protocol is None else "TRAINING_SELECTION14")
        maybe_load_checkpoint(trainer, False, False, config["pretrained_checkpoint"])
        if trainer.current_epoch != 0:
            raise ValueError("Fine-tuning did not start at epoch 0")
        expected = {"initial_lr": .01, "weight_decay": 3e-5, "num_epochs": 1000,
                    "num_iterations_per_epoch": 250, "num_val_iterations_per_epoch": 50,
                    "oversample_foreground_percent": .33, "enable_deep_supervision": True}
        for key, value in expected.items():
            if getattr(trainer, key) != value:
                raise ValueError(f"Unexpected training parameter {key}")
        save()
        trainer.run_training()
        best = Path(trainer.output_folder) / "checkpoint_best.pth"
        if not best.is_file():
            raise ValueError("Training did not produce checkpoint_best.pth")
        protocol["status"] = "TRAINED_EVALUATION_PENDING"
        save()
        trainer.load_checkpoint(str(best))
        bind_split(trainer, train, evaluation, "EVALUATE_ORIGINAL19" if cv_protocol is None else "EVALUATE_VALIDATION14")
        trainer.output_folder = str(Path(trainer.output_folder) / ("evaluation_original19" if cv_protocol is None else "evaluation_validation14"))
        Path(trainer.output_folder).mkdir(exist_ok=False)
        trainer.perform_actual_validation(False)
        means = export_metrics(Path(trainer.output_folder) / "validation" / "summary.json", evaluation, selection, result, cohort_label="evaluation19" if cv_protocol is None else "evaluation14")
        protocol["evaluation"] = means
        protocol["status"] = "COMPLETE"
        print("FULL_VOLUME_RESULT", json.dumps(means), flush=True)
    except BaseException as exc:
        protocol["status"] = "FAILED_OR_ABORTED"
        protocol["error"] = str(exc)
        raise
    finally:
        changed = [p for p, h in original_hashes.items() if hash_file(p) != h]
        if cv_protocol is not None:
            if hash_file(cv_path) != protocol["cv70_protocol_file_sha256"] or config_digest(read_json(args.config)) != protocol["config_sha256"]:
                changed.append("CV70 protocol/config")
        protocol["canonical_metadata_unchanged"] = not changed
        save()
        if changed:
            raise RuntimeError(f"Source metadata changed during run: {changed}")


if __name__ == "__main__":
    main()