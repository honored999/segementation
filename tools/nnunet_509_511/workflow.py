"""Sequential fold-0 workflow. Standard library only; default mode writes metadata."""
import argparse
import copy
import csv
import ctypes
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime

REPO = Path(__file__).resolve().parents[2]
RAW = Path(r"C:\lijialin\models3d\nnUNet\nnUNet_raw")
DEST = Path(r"C:\lijialin\HD-BET_outputs\results")
SOURCE = Path(r"C:\lijialin\models3d\nnUNet\nnUNet_results") / (
    "Dataset508_ISLES2022DWI/nnUNetTrainerTopK10EarlyStopping__nnUNetPlansFrom501__2d")
BEST = SOURCE / "fold_0/checkpoint_best.pth"
DATASETS = {509: "Dataset509_CI1DWIHigh", 511: "Dataset511_CI1DWIHighB0BET3mm"}
SPLIT_SHA = "cfcad46bf1f47d397cf7ea8b5c3a40c2a49db77767eb343c6e3a92e3e88831d4"
PLAN = "nnUNetPlansFrom508"
EARLY = "nnUNetTrainerEarlyStopping"
TOPK = "nnUNetTrainerTopK10EarlyStopping"
GROUPS = [("01_509_CEDice", 509, EARLY, False), ("02_509_TopK10", 509, TOPK, False),
          ("03_511_TopK10_scratch", 511, TOPK, False), ("04_511_TopK10_ISLES508", 511, TOPK, True)]
METRICS = ["Dice (%)", "IOU (%)", "F2 (%)", "AVD (%)", "LCD", "Rec (%)", "HD95 (mm)"]
ENTRY = "nnunetv2.experiment_planning.plan_and_preprocess_entrypoints"


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def inside(base, *parts):
    base = base.resolve()
    path = base.joinpath(*parts).resolve()
    need(path != base and path.is_relative_to(base), f"Path escapes boundary: {path}")
    return path


def save(root, name, value):
    path = inside(root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def safe_id(value):
    reserved = {"CON", "PRN", "AUX", "NUL"} | {
        f"{prefix}{n}" for prefix in ("COM", "LPT") for n in range(1, 10)}
    return (isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value)
            and value.upper() not in reserved)


def metadata():
    splits_path = inside(RAW, DATASETS[511], "splits_final.json")
    split_bytes = splits_path.read_bytes()
    need(hashlib.sha256(split_bytes).hexdigest() == SPLIT_SHA, "511 split SHA256 mismatch")
    splits = json.loads(split_bytes)
    need(isinstance(splits, list) and len(splits) == 5, "Expected exactly five folds")
    universe, validation = None, []
    for i, fold in enumerate(splits):
        need(isinstance(fold, dict) and set(fold) == {"train", "val"}, "Invalid fold structure")
        train, val = fold["train"], fold["val"]
        need(isinstance(train, list) and isinstance(val, list), "Split lists required")
        ids = train + val
        need(all(safe_id(x) for x in ids), "Unsafe case ID")
        need(len(ids) == 128 and len({x.casefold() for x in ids}) == 128, "Duplicate/missing IDs")
        current = set(ids)
        if universe is None:
            universe = current
        need(current == universe, "Fold universes differ")
        if i == 0:
            need(len(train) == 100 and len(val) == 28, "Fold0 must be 100/28")
        validation.extend(val)
    need(len(validation) == 128 and set(validation) == universe, "Each case must validate exactly once")
    target_json = {}
    for number, name in DATASETS.items():
        need(inside(RAW, name, "splits_final.json").read_bytes() == split_bytes, f"{number} split differs")
        data = load(inside(RAW, name, "dataset.json"))
        need(data.get("numTraining") == 128 and data.get("file_ending") == ".nii.gz", "Dataset size/format mismatch")
        need(set(data.get("channel_names", {})) == {"0"}, "Exactly one image channel required")
        need(data.get("labels", {}).get("background") == 0 and
             sorted(data.get("labels", {}).values()) == [0, 1], "Binary labels 0/1 required")
        need("dataset" not in data and "regions_class_order" not in data, "Custom dataset mappings/regions refused")
        need(data.get("overwrite_image_reader_writer") in (None, "SimpleITKIO"), "Reader mismatch")
        for folder, suffix in (("imagesTr", "_0000.nii.gz"), ("labelsTr", ".nii.gz")):
            directory = inside(RAW, name, folder)
            need({p.name for p in directory.iterdir()} == {x + suffix for x in universe},
                 f"{name}/{folder}: exact 128 filenames required")
            need(all(inside(RAW, name, folder, x + suffix).is_file() for x in universe), "Non-file input")
        target_json[number] = data
    source = load(SOURCE / "plans.json")
    source_json = load(SOURCE / "dataset.json")
    config = source["configurations"]["2d"]
    need(source.get("dataset_name") == "Dataset508_ISLES2022DWI" and
         source.get("plans_name") == "nnUNetPlansFrom501", "Unexpected source plans identity")
    need(source.get("image_reader_writer") == "SimpleITKIO", "Source reader must be SimpleITKIO")
    need(config.get("batch_size") == 12 and config.get("patch_size") == [512, 512] and
         config["architecture"]["arch_kwargs"]["n_stages"] == 8, "Expected batch12 / 512 / eight stages")
    need(len(config["spacing"]) == 2 and all(abs(x - 0.489) < 0.002 for x in config["spacing"]),
         "Unexpected source 2D spacing")
    need(config.get("normalization_schemes") == ["ZScoreNormalization"] and
         config.get("use_mask_for_norm") == [False], "Unexpected normalization")
    need(set(source_json.get("channel_names", {})) == {"0"} and
         source_json.get("labels", {}).get("background") == 0 and
         sorted(source_json.get("labels", {}).values()) == [0, 1] and
         "regions_class_order" not in source_json, "Source label/channel mismatch")
    need(BEST.is_file() and BEST.stat().st_size == 370907859, "Source best missing or unexpected size")
    trainer_dir = REPO / "nnunet_ext_trainers"
    for filename in (EARLY + ".py", TOPK + ".py", "nnUNetTrainerTopK10.py", "evaluate_segmentation_metrics.py"):
        need((REPO / filename if filename.startswith("evaluate") else trainer_dir / filename).is_file(),
             f"Required existing source missing: {filename}")
    return split_bytes, splits[0]["val"], source


def transferred(source, name):
    plans = copy.deepcopy(source)
    old = source["plans_name"]
    plans.update(dataset_name=name, plans_name=PLAN, image_reader_writer="SimpleITKIO")
    for config in plans["configurations"].values():
        if "data_identifier" in config:
            value = config["data_identifier"]
            config["data_identifier"] = PLAN + (value[len(old):] if value.startswith(old) else "_" + value)
            need(safe_id(config["data_identifier"]), "Unsafe data_identifier")
    return plans


def entry(module, function, *args):
    return [sys.executable, "-c", f"from {module} import {function}; {function}()", *map(str, args)]


def resources(gpu, manual_cpu):
    class Memory(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (n, ctypes.c_ulonglong) for n in ("total", "available", "page", "free_page", "virtual", "free_virtual", "extended")]
    memory = Memory()
    memory.length = ctypes.sizeof(memory)
    need(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)), "RAM query failed")
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], text=True, capture_output=True, check=True, timeout=20)
    rows = list(csv.reader(query.stdout.splitlines(), skipinitialspace=True))
    matches = [row for row in rows if gpu in (row[0].strip(), row[1].strip())]
    need(len(matches) == 1, "Specify an existing physical GPU index or full UUID")
    index, uuid, utilization, used, total = [x.strip() for x in matches[0]]
    utilization, used, total = float(utilization), float(used), float(total)
    need(all(math.isfinite(x) for x in (utilization, used, total)) and total > 0, "GPU counters unavailable")
    snapshot = {"physical_index": index, "uuid": uuid, "gpu_util_percent": utilization,
                "vram_used_MiB": used, "vram_total_MiB": total, "ram_percent": memory.load,
                "cpu_percent": manual_cpu, "cpu_source": "TaskManager_manual" if manual_cpu is not None else "unavailable",
                "time": datetime.now().isoformat()}
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=["plan", "execute"], default="plan")
    parser.add_argument("gpu", nargs="?", default=os.environ.get("NNUNET_PHYSICAL_GPU"))
    parser.add_argument("cpu", nargs="?", type=float, default=None, help="Optional current Task Manager CPU percent")
    args = parser.parse_args()
    need(os.name == "nt", "This workflow requires Windows")
    need(os.environ.get("CONDA_DEFAULT_ENV", "").lower() == "nnunet5090" and
         Path(sys.executable).parent.resolve() == Path(os.environ.get("CONDA_PREFIX", "")).resolve(),
         "Activate nnunet5090; refusing a different Python")
    need(args.cpu is None or math.isfinite(args.cpu) and 0 <= args.cpu <= 100, "CPU percent must be 0..100")
    root = inside(DEST, "nnunet509_511_fold0_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    for protected in (RAW, SOURCE, REPO, Path(r"C:\lijialin\models3d\nnUNet\nnUNet_preprocessed")):
        protected = protected.resolve()
        need(not root.is_relative_to(protected) and not protected.is_relative_to(root), "Output/input overlap")
    need(not os.path.lexists(root), f"Output root already exists: {root}")
    root.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "PREPARING", "mode": args.mode, "root": str(root), "python": sys.executable,
               "created": datetime.now().isoformat(), "manual_cpu_percent": args.cpu,
               "gpu_requested": args.gpu, "steps": [], "resources": []}
    print(f"Output / receipt: {root}", flush=True)
    try:
        split_bytes, val, source = metadata()
        preprocessed = inside(root, "nnUNet_preprocessed")
        env = os.environ.copy()
        env.update(nnUNet_raw=str(RAW.resolve()), nnUNet_preprocessed=str(preprocessed),
                   nnUNet_results=str(inside(root, "preprocessing_results")),
                   nnUNet_extTrainer=str((REPO / "nnunet_ext_trainers").resolve()),
                   nnUNet_n_proc_DA="2", nnUNet_def_n_proc="2", ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="2",
                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2",
                   PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", CUDA_DEVICE_ORDER="PCI_BUS_ID")
        # Existing external trainers use sibling imports; the discovery folder must also be on sys.path.
        env["PYTHONPATH"] = str(REPO / "nnunet_ext_trainers") + os.pathsep + str(REPO)
        for key in ("TMP", "TEMP"):
            env[key] = str(inside(root, "tmp"))
        temp = inside(root, "tmp")
        temp.mkdir()
        for key in ("TORCHINDUCTOR_CACHE_DIR", "TORCH_HOME", "MPLCONFIGDIR", "CUDA_CACHE_PATH",
                    "TRITON_CACHE_DIR", "NUMBA_CACHE_DIR", "XDG_CACHE_HOME"):
            directory = inside(root, "runtime_cache", key)
            directory.mkdir(parents=True)
            env[key] = str(directory)
        commands = []
        for number in DATASETS:
            commands.append({"name": f"fingerprint_{number}", "argv": entry(
                ENTRY, "extract_fingerprint_entry", "-d", number, "-np", 2, "--verify_dataset_integrity", "--no_pbar")})
            commands.append({"name": f"preprocess_{number}", "argv": entry(
                ENTRY, "preprocess_entry", "-d", number, "-plans_name", PLAN, "-c", "2d", "-np", 2, "--no_pbar")})
        for group, number, trainer, pretrained in GROUPS:
            results = inside(root, group, "nnUNet_results")
            fold = inside(results, DATASETS[number], f"{trainer}__{PLAN}__2d", "fold_0")
            train = entry("nnunetv2.run.run_training", "run_training_entry",
                          number, "2d", 0, "-tr", trainer, "-p", PLAN, "--val_best")
            if pretrained:
                train += ["-pretrained_weights", str(BEST)]
            commands.extend([
                {"name": group + "_train", "results": str(results), "argv": train},
                {"name": group + "_evaluate", "results": str(results), "argv": [
                    sys.executable, str(REPO / "evaluate_segmentation_metrics.py"),
                    "--pred-dir", str(inside(fold, "validation")), "--gt-dir", str(inside(RAW, DATASETS[509], "labelsTr")),
                    "--output-dir", str(inside(root, group, "metrics")), "--f2-mode", "paper"]}])
        tracked = [SOURCE / "plans.json", SOURCE / "dataset.json", Path(__file__), Path(__file__).with_name("run.bat"),
                   REPO / "evaluate_segmentation_metrics.py"] + [
                       inside(RAW, name, "dataset.json") for name in DATASETS.values()] + [
                       REPO / "nnunet_ext_trainers" / (name + ".py") for name in (EARLY, TOPK, "nnUNetTrainerTopK10")]
        receipt.update(source_best={"path": str(BEST), "size": BEST.stat().st_size, "mtime_ns": BEST.stat().st_mtime_ns},
                       split_sha256=SPLIT_SHA, validation_ids=val, file_sha256={str(p): sha(p) for p in tracked},
                       package_versions={p: importlib.metadata.version(p) for p in ("nnunetv2", "torch")})
        save(root, "plan.json", {"commands": commands, "environment": {
            k: env[k] for k in env if k.startswith("nnUNet") or k in (
                "PYTHONPATH", "OMP_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "TEMP", "TMP",
                "TORCHINDUCTOR_CACHE_DIR", "TORCH_HOME", "MPLCONFIGDIR", "CUDA_CACHE_PATH",
                "TRITON_CACHE_DIR", "NUMBA_CACHE_DIR", "XDG_CACHE_HOME")},
            "source_plans": source, "target_plans": {str(n): transferred(source, name) for n, name in DATASETS.items()},
            "api": "Parent verified installed entrypoints; no runtime import probe"})
        if args.mode == "plan":
            receipt["status"] = "PLAN_ONLY_NOT_RUN"
            return
        need(args.gpu, "execute requires a physical GPU argument or NNUNET_PHYSICAL_GPU")
        selected_uuid = None

        def guard():
            nonlocal selected_uuid
            sample = resources(selected_uuid or args.gpu, args.cpu)
            receipt["resources"].append(sample)
            save(root, "receipt.json", receipt)
            need(sample["gpu_util_percent"] < 80 and sample["vram_used_MiB"] / sample["vram_total_MiB"] < 0.8
                 and sample["ram_percent"] < 80 and (args.cpu is None or args.cpu < 80),
                 "Resource threshold >=80%; stopped without queueing or selecting another GPU")
            selected_uuid = sample["uuid"]
            env["CUDA_VISIBLE_DEVICES"] = selected_uuid

        guard()
        receipt["status"] = "RUNNING"
        receipt["source_best"]["sha256"] = sha(BEST)
        receipt["validation_label_sha256"] = {}
        for cid in val:
            left = sha(inside(RAW, DATASETS[509], "labelsTr", cid + ".nii.gz"))
            right = sha(inside(RAW, DATASETS[511], "labelsTr", cid + ".nii.gz"))
            need(left == right, f"509/511 validation label bytes differ: {cid}")
            receipt["validation_label_sha256"][cid] = left
        for number, name in DATASETS.items():
            cache = inside(preprocessed, name)
            cache.mkdir(parents=True)
            with inside(cache, "dataset.json").open("xb") as stream:
                stream.write(inside(RAW, name, "dataset.json").read_bytes())
            inside(cache, "splits_final.json").write_bytes(split_bytes)
            save(root, str(inside(cache, PLAN + ".json").relative_to(root)), transferred(source, name))
        rows = []
        for command in commands:
            guard()
            if "-pretrained_weights" in command["argv"]:
                need(sha(BEST) == receipt["source_best"]["sha256"], "Source best changed before fine-tuning")
            if command["name"].endswith("_evaluate"):
                prediction = Path(command["argv"][3])
                need({p.name for p in prediction.glob("*.nii*")} == {cid + ".nii.gz" for cid in val},
                     "Predictions must be exactly the 28 fold0 .nii.gz files")
            if command["name"].endswith("_train"):
                for name in DATASETS.values():
                    need(inside(preprocessed, name, "splits_final.json").read_bytes() == split_bytes, "Cached split changed")
            env["nnUNet_results"] = command.get("results", str(inside(root, "preprocessing_results")))
            step = {"name": command["name"], "argv": command["argv"], "status": "RUNNING",
                    "nnUNet_results": env["nnUNet_results"], "CUDA_VISIBLE_DEVICES": selected_uuid}
            receipt["steps"].append(step)
            save(root, "receipt.json", receipt)
            print(subprocess.list2cmdline(command["argv"]), flush=True)
            with inside(root, command["name"] + ".log").open("w", encoding="utf-8") as log:
                with subprocess.Popen(command["argv"], env=env, cwd=temp, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                      bufsize=1) as child:
                    for line in child.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                    code = child.wait()
            step.update(returncode=code, status="COMPLETE" if code == 0 else "FAILED")
            save(root, "receipt.json", receipt)
            if code:
                raise subprocess.CalledProcessError(code, command["argv"])
            if command["name"].endswith("_evaluate"):
                folder = Path(command["argv"][7])
                with (folder / "summary_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
                    summary = list(csv.DictReader(stream))
                need(len(summary) == 1 and all(k in summary[0] for k in METRICS), "Evaluator summary schema mismatch")
                cases = load(folder / "summary_metrics.json")
                need(cases["n_cases"] == 28 and cases["f2_mode"] == "paper", "Incomplete/mismatched evaluation")
                rows.append({"experiment": command["name"].removesuffix("_evaluate"),
                             **{k: summary[0][k] for k in METRICS}})
        need(len(rows) == 4, "All four experiments must complete before comparison")
        with inside(root, "comparison.csv").open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["experiment", *METRICS])
            writer.writeheader()
            writer.writerows(rows)
        receipt["status"] = "COMPLETE_FOLD0_DEVELOPMENT"
    except BaseException as error:
        receipt.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt["updated"] = datetime.now().isoformat()
        save(root, "receipt.json", receipt)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(f"Child failed with exit code {error.returncode}; output and receipt preserved.", file=sys.stderr)
        sys.exit(error.returncode)
