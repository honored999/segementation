"""Fixed fold-0 pipeline preparation; never imports torch or reads image contents."""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

PLAN = "nnUNetPlansDropLastDS"
CONFIG = "2d_drop_last_ds"
SOURCE = "Dataset508_ISLES2022DWI"
TARGET = "Dataset501_StrokeLesion"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_config(plans, name, seen=()):
    require(name not in seen, "Configuration inheritance cycle")
    config = deepcopy(plans["configurations"][name])
    parent = config.pop("inherits_from", None)
    base = resolve_config(plans, parent, (*seen, name)) if parent else {}
    base.update(config)
    return base


def split(path):
    splits = read(path)
    require(len(splits) == 5, f"Expected established five-fold splits: {path}")
    for fold in splits:
        train, val = fold["train"], fold["val"]
        require(len(set(train)) == len(train) and len(set(val)) == len(val), "Duplicate split IDs")
        require(not set(train) & set(val), "Train/validation overlap")
        require(all(Path(x).name == x and x not in ("", ".", "..") for x in train + val), "Unsafe case ID")
    return splits


def guard(gpu, cpu_only=False):
    import psutil
    cpu, ram = psutil.cpu_percent(interval=1), psutil.virtual_memory().percent
    require(0 <= cpu < 80 and 0 <= ram < 80, f"CPU/RAM gate failed (>=80%): CPU={cpu}, RAM={ram}")
    if cpu_only:
        return dict(cpu_percent=cpu, ram_percent=ram)
    command = ["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]
    fields = subprocess.check_output(command, text=True).strip().split(",")
    require(len(fields) == 4, "Expected exactly one physical GPU")
    uuid, util, used, total = [x.strip() for x in fields]
    require(uuid.startswith("GPU-"), "Missing GPU UUID")
    values = dict(cpu_percent=cpu, ram_percent=ram, gpu_percent=float(util), vram_percent=100 * float(used) / float(total))
    require(all(0 <= x < 80 for x in values.values()), f"Resource gate failed (>=80%): {values}")
    return dict(physical_gpu=gpu, gpu_uuid=uuid, **values)


def prepare(args):
    repo = Path(__file__).resolve().parent.parent
    root = Path(args.root).resolve()
    raw_s, pp_s, raw_t, pp_t, original = [Path(x).resolve() for x in (args.source_raw, args.source_pp, args.target_raw, args.target_pp, args.original_raw)]
    for protected in (repo, raw_s, pp_s, raw_t, pp_t, original):
        require(root != protected and protected not in root.parents and root not in protected.parents, "Output root overlaps protected input")
    require(not root.exists(), "Output root already exists; no resume/overwrite allowed")
    resources = guard(args.gpu)
    plan_paths = [pp_s / SOURCE / "nnUNetPlansFrom501.json", pp_t / TARGET / "nnUNetPlans.json"]
    plans = [read(x) for x in plan_paths]
    for plan, expected in zip(plans, (SOURCE, TARGET)):
        require(plan.get('dataset_name') == expected, f'plans.dataset_name must be {expected}')
    configs = [resolve_config(p, "2d") for p in plans]
    require({k: v for k, v in configs[0].items() if k != "data_identifier"} == {k: v for k, v in configs[1].items() if k != "data_identifier"}, "Resolved source/target 2d settings differ (architecture/patch/batch/spacing/normalization)")
    for key in ("transpose_forward", "transpose_backward", "image_reader_writer"):
        require(plans[0].get(key) == plans[1].get(key), f"Plan metadata mismatch: {key}")
    config = configs[0]
    architecture = config["architecture"]
    require(architecture["network_class_name"] == "dynamic_network_architectures.architectures.unet.PlainConvUNet", "Expected official PlainConvUNet")
    stages = architecture["arch_kwargs"]["n_stages"]
    require(type(stages) is int and stages >= 3 and len(config["patch_size"]) == 2, "Invalid 2d depth")
    original_pp = Path(args.original_pp).resolve()
    require(root != original_pp and original_pp not in root.parents and root not in original_pp.parents, "Output overlaps original preprocessed input")
    split_paths = [pp_s / SOURCE / "splits_final.json", pp_t / TARGET / "splits_final.json", original_pp / TARGET / "splits_final.json"]
    splits = [split(x) for x in split_paths]
    require(splits[1] == splits[2], "HD-BET split differs from original fixed split")
    require(len(splits[1][0]["train"]) == 76 and len(splits[1][0]["val"]) == 19, "Dataset501 fold0 must be 76/19")
    datasets = [read(raw_s / SOURCE / "dataset.json"), read(raw_t / TARGET / "dataset.json")]
    for key in ("channel_names", "labels", "file_ending"):
        require(datasets[0][key] == datasets[1][key], f"Dataset interface mismatch: {key}")
    require(datasets[0]["file_ending"] == ".nii.gz", "Expected NIfTI .nii.gz")
    for pp, name, folds, gt in ((pp_s, SOURCE, splits[0], raw_s / SOURCE / "labelsTr"), (pp_t, TARGET, splits[1], original / TARGET / "labelsTr")):
        own_config = configs[0] if name == SOURCE else configs[1]
        identifier = own_config["data_identifier"]
        require(Path(identifier).name == identifier and identifier not in ("", ".", ".."), "Unsafe data_identifier")
        folder = pp / name / identifier
        require(folder.is_dir(), f"Missing preprocessed directory: {folder}")
        for case in set(folds[0]["train"] + folds[0]["val"]):
            require((folder / (case + ".pkl")).is_file(), f"Missing properties for {case}")
            require((folder / (case + ".npz")).is_file() or ((folder / (case + ".b2nd")).is_file() and (folder / (case + "_seg.b2nd")).is_file()) or ((folder / (case + ".npy")).is_file() and (folder / (case + "_seg.npy")).is_file()), f"Missing prepared arrays for {case}")
        require(all((gt / (case + ".nii.gz")).is_file() for case in folds[0]["val"]), f"Missing original validation GT: {gt}")
    derived = []
    for path, plan, config in zip(plan_paths, plans, configs):
        plan = deepcopy(plan)
        plan["plans_name"] = PLAN
        config = deepcopy(config)
        config.update(encoder_last_stage=stages - 2, enable_deep_supervision=True)
        plan["configurations"][CONFIG] = config
        destination = path.with_name(PLAN + ".json")
        require(not destination.exists() or read(destination) == plan, f"Conflicting derived plans: {destination}")
        derived.append((destination, plan))
    root.mkdir(parents=True, exist_ok=False)
    for destination, plan in derived:
        if not destination.exists():
            with destination.open("x", encoding="utf-8") as stream:
                json.dump(plan, stream, indent=2)
    tracked = plan_paths + split_paths + [x[0] for x in derived] + list((repo / "nnunet_ext_trainers").glob("*.py")) + [repo / "evaluate_segmentation_metrics.py", Path(__file__).resolve(), repo / "run_plainconv_depth_pretrain_finetune.bat"]
    receipt = dict(created=datetime.now().isoformat(), root=str(root), resources=resources, environment=os.environ.get("CONDA_DEFAULT_ENV"), fold=0, configuration=CONFIG, plans=PLAN, encoder_last_stage=stages - 2, enable_deep_supervision=True, files={str(p): sha(p) for p in tracked})
    (root / "run_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    (root / "gpu_uuid.txt").write_text(resources["gpu_uuid"], encoding="ascii")


def predictions(args):
    expected = set(split(Path(args.split))[0]["val"])
    folder = Path(args.pred)
    require(folder.is_dir(), f"Missing validation predictions: {folder}")
    actual = {p.name[:-7] for p in folder.glob("*.nii.gz") if p.is_file()}
    require(actual == expected, f"Prediction cohort mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    for name in ("root", "source-raw", "source-pp", "target-raw", "target-pp", "original-raw", "original-pp"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p = sub.add_parser("guard")
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--cpu-only", action="store_true")
    p = sub.add_parser("predictions")
    p.add_argument("--split", required=True)
    p.add_argument("--pred", required=True)
    p = sub.add_parser("finish")
    p.add_argument("--root", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "guard":
        print(json.dumps(guard(args.gpu, cpu_only=args.cpu_only)))
    elif args.command == "predictions":
        predictions(args)
    else:
        paths = [Path(args.root).resolve() / name for name in ("evaluation_isles508_best_fold0", "evaluation_hdbet501_best_fold0")]
        for path in paths:
            require(all((path / name).is_file() and (path / name).stat().st_size > 0 for name in ("case_metrics.csv", "summary_metrics.csv", "summary_metrics.json")), f"Incomplete metrics: {path}")
        for path in paths:
            print(str(path))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}")
