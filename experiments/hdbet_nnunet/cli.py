"""Bounded Dataset501 CLI. check/ready use only stdlib metadata/file hashes.

Incomplete outputs are retained; prepare never resumes or overwrites.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[2]
NAME = "Dataset501_StrokeLesion"
TARGET_SHA = "23d25e2b2a6d7c7ca2a270690a9d4d35116ff4f834458b48a9049ff889894a56"
SPLIT_SHA = "02a84d777e4eb9f481c3a9d90529c1673c828e14f2699fd0e898596ce21574ed"
SPLIT_LF_SHA = "eb0d4b15920581fdac2fa158423b30fefce8465e961e4016d1e2b48ac55bd455"
TRAINER = "nnUNetTrainerTopK10EarlyStopping"
PROTOCOL = "hdbet3mm_fixed95_fold0"
FINGERPRINT_STATUS = "inherited_fixed_baseline_compatibility_not_derived"
RECIPE = {
    "hd_bet_version": "2.0.1", "hd_bet_tta": True, "hd_bet_device": "cuda",
    "mask_margin_mm": 3.0, "mask_margin_axes": "native_slice_plane_only",
    "empty_slice": "retain_entire_native_slice", "empty_volume": "fail",
    "mapping": "source_reader_transpose_original_crop_nnUNet_seg_resampling",
    "target_fill": "baseline_generator_training_global_minimum", "target_space": "existing_preprocessed_target",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def confined(root, relative):
    root = Path(root).resolve(strict=True)
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(), "Relative artifact required")
    path = (root / relative).resolve(strict=True)
    require(root in path.parents and path.is_file(), "Missing or escaping artifact: " + relative)
    return path


def separate(output, protected):
    output = Path(output).resolve()
    require("HD-BET_outputs" in output.parts, "Output must be below HD-BET_outputs")
    for path in protected:
        path = Path(path).resolve()
        require(output != path and path not in output.parents and output not in path.parents,
                "Output overlaps protected input: " + str(path))
    return output


def splits(path):
    reference = REPO / "standalone_nnunet2d/reference/splits_final.json"
    require(hashlib.sha256(reference.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == SPLIT_LF_SHA,
            "Canonical reference bytes changed")
    actual, expected = read(path), read(reference)
    require(isinstance(actual, list) and len(actual) == len(expected) == 5, "Canonical five folds required")
    for fold, ref in zip(actual, expected):
        require(isinstance(fold, dict) and set(fold) == {"train", "val"}, "Unexpected split fields")
        for role, count in (("train", 76), ("val", 19)):
            ids = fold[role]
            require(isinstance(ids, list) and all(isinstance(p, str) and p and
                    all(c.isascii() and (c.isalnum() or c in "_-") for c in p) for p in ids), "Unsafe case IDs")
            require(len(ids) == len(set(ids)) == count and sorted(ids) == sorted(ref[role]), "Canonical membership mismatch")
        require(not set(fold["train"]) & set(fold["val"]), "Split overlap")
    return sorted(actual[0]["train"]), sorted(actual[0]["val"])


def plan_contract(plan, dataset):
    require(plan.get("dataset_name") == NAME and plan.get("plans_name") == "nnUNetPlans", "Unexpected plans identity")
    cfg = plan.get("configurations", {}).get("2d", {})
    require("inherits_from" not in cfg and cfg.get("previous_stage") is None and
            cfg.get("preprocessor_name") == "DefaultPreprocessor", "Only original noncascade DefaultPreprocessor supported")
    require(cfg.get("normalization_schemes") == ["ZScoreNormalization"] and
            cfg.get("use_mask_for_norm") in ([True], [False]), "SingleDWI per-case ZScore with explicit mask policy required")
    require(len(cfg.get("spacing", [])) == len(cfg.get("patch_size", [])) == 2 and
            type(cfg.get("batch_size")) is int and cfg["batch_size"] > 0, "Original 2d plans incomplete")
    identifier = cfg.get("data_identifier")
    require(isinstance(identifier, str) and Path(identifier).name == identifier and identifier not in ("", ".", ".."),
            "Unsafe data_identifier")
    require(dataset.get("channel_names") == {"0": "DWI"} and dataset.get("numTraining") == 95 and
            dataset.get("file_ending") == ".nii.gz" and dataset.get("labels") == {"background": 0, "lesion": 1},
            "Expected exact Dataset501 singleDWI binary lesion metadata")
    require(plan.get("image_reader_writer") == dataset.get("overwrite_image_reader_writer") == "SimpleITKIO",
            "Original SimpleITK native reader required")
    return identifier


def versions():
    for name, expected in (("nnunetv2", "2.8.1"), ("hd-bet", "2.0.1")):
        require(importlib.metadata.version(name) == expected, name + " version must be " + expected)


def check(args):
    raw = Path(args.raw_dataset).resolve(strict=True)
    pre = Path(args.preprocessed_dataset).resolve(strict=True)
    targets = Path(args.targets).resolve(strict=True)
    require(raw.name == pre.name == NAME, "Source dataset names must remain Dataset501")
    output = separate(args.output, [raw.parent, pre.parent, targets, REPO])
    require(not output.exists(), "Fresh output required; no overwrite/resume")
    versions()
    train, val = splits(pre / "splits_final.json")
    dataset, plan = read(raw / "dataset.json"), read(pre / "nnUNetPlans.json")
    identifier = plan_contract(plan, dataset)
    require(read(pre / "dataset.json") == dataset, "Raw/preprocessed dataset metadata disagree")
    require(args.targets_sha256 == TARGET_SHA and sha(targets / "targets.json") == TARGET_SHA, "Pinned target SHA mismatch")
    manifest = read(targets / "targets.json")
    require(manifest.get("format") == 1 and manifest.get("complete") is True and manifest.get("recipe") == RECIPE and
            manifest.get("classification") == "exploratory_dataset501_fold0_genseg3mm_preview", "Foreign/incomplete target recipe")
    require(set(manifest.get("cases", {})) == set(train), "Targets must cover exactly train76")
    identity = manifest["source"]
    for field, value in (("dataset_name", NAME), ("configuration", "2d"), ("fold", 0),
                         ("train_ids", train), ("val_ids", val), ("backend", "nnUNetDatasetBlosc2"),
                         ("reference_sha256", SPLIT_SHA)):
        require(identity.get(field) == value, "Target source identity mismatch: " + field)
    for field, name in (("plans_sha256", "nnUNetPlans.json"), ("dataset_sha256", "dataset.json"), ("splits_sha256", "splits_final.json")):
        require(sha(pre / name) == identity.get(field), "Original metadata differs from pinned targets: " + name)
    hd = manifest["hd_bet"]
    weights = Path(hd["weights_file"]).resolve(strict=True)
    require(weights.is_file() and sha(weights) == hd.get("weights_sha256") and hd.get("version") == "2.0.1" and
            hd.get("default_tta") is True, "Pinned weights/defaultTTA mismatch")
    official_weights = Path.home() / "hd-bet_params/release_2.0.0/fold_all/checkpoint_final.pth"
    require(official_weights.resolve(strict=True) == weights, "Pinned weights must occupy official2.0.1 runtime location")
    require(shutil.which("hd-bet"), "HD-BET executable missing (no install/download)")
    # Prevent outputs from overlapping ANY source/cached/weight tree.
    separate(output, [raw.parent, pre.parent, targets, weights.parent, REPO])
    files = {}
    def pin(path):
        path = Path(path).resolve(strict=True)
        files[str(path)] = sha(path)
        return path
    for name in ("nnUNetPlans.json", "splits_final.json", "dataset_fingerprint.json", "dataset.json"):
        pin(confined(pre, name))
    fingerprint = read(pre / "dataset_fingerprint.json")
    require(isinstance(fingerprint, dict) and isinstance(fingerprint.get("spacings"), list) and
            isinstance(fingerprint.get("shapes_after_crop"), list) and
            len(fingerprint["spacings"]) == len(fingerprint["shapes_after_crop"]) == 95,
            "Original Dataset501 fingerprint metadata incomplete")
    pin(confined(raw, "dataset.json"))
    pin(confined(targets, "targets.json"))
    pin(weights)
    # Read-only implementation provenance, frozen with all other inputs. Hashes
    # suffice without requiring Git installation or querying/mutating repository state.
    for relative in (
        "experiments/hdbet_nnunet/__init__.py", "experiments/hdbet_nnunet/cli.py",
        "experiments/hdbet_nnunet/native.py", "nnunet_ext_trainers/nnUNetTrainerTopK10EarlyStopping.py",
        "nnunet_ext_trainers/nnUNetTrainerEarlyStopping.py", "nnunet_ext_trainers/nnUNetTrainerTopK10.py",
        "experiments/genseg3mm/targets.py", "experiments/genseg_joint/runtime.py", "experiments/genseg30/contract.py",
    ):
        pin(confined(REPO, relative))
    ids = train + val
    require({p.name for p in (raw / "imagesTr").iterdir()} == {p + "_0000.nii.gz" for p in ids}, "Exact95 raw images required")
    require({p.name for p in (raw / "labelsTr").iterdir()} == {p + ".nii.gz" for p in ids}, "Exact95 raw labels required")
    cases = {}
    for patient in ids:
        image = pin(confined(raw, "imagesTr/" + patient + "_0000.nii.gz"))
        label = pin(confined(raw, "labelsTr/" + patient + ".nii.gz"))
        case = {"image": str(image), "label": str(label), "raw_sha256": files[str(image)], "label_sha256": files[str(label)]}
        if patient in train:
            rec = manifest["cases"][patient]
            require(rec.get("status") == "complete" and rec.get("return_code") == 0 and
                    rec.get("raw_file") == image.name and rec.get("raw_sha256") == case["raw_sha256"], "Cached raw identity mismatch")
            inventory = rec["source_inventory"]
            require(inventory == manifest["source_inventory"][patient] and
                    set(inventory) == {patient + ext for ext in (".b2nd", "_seg.b2nd", ".pkl")}, "Cached source inventory mismatch")
            for filename, digest in inventory.items():
                path = pin(confined(pre / identifier, filename))
                require(files[str(path)] == digest, "Original cache changed: " + filename)
            artifacts = {}
            for field in ("mask", "native_mask", "input_las", "log", "command_started", "command_result"):
                path = pin(confined(targets, rec[field + "_file"]))
                require(files[str(path)] == rec[field + "_sha256"], "Cache artifact hash mismatch: " + field)
                artifacts[field] = path
            start, result = read(artifacts["command_started"]), read(artifacts["command_result"])
            require(start.get("patient") == result.get("patient") == patient and start.get("command") == rec.get("command") and
                    start.get("default_tta") is True and start.get("raw_sha256") == case["raw_sha256"] and
                    result.get("status") == "command_completed" and result.get("return_code") == 0 and
                    result.get("mask_sha256") == rec["native_mask_sha256"] and result.get("log_sha256") == rec["log_sha256"],
                    "Invalid cached command receipt")
            command = rec.get("command", [])
            require(command == [hd["cli"], "-i", start["input_las_file"], "-o", start["output_file"],
                                "-device", "cuda", "--save_bet_mask", "--no_bet_image"], "Cached command/defaultTTA differs")
            case["cached_mask"] = str(artifacts["native_mask"])
        cases[patient] = case
    return {"raw": str(raw), "pre": str(pre), "targets": str(targets), "output": str(output), "train": train, "val": val,
            "identifier": identifier, "files": files, "cases": cases, "weights": str(weights),
            "weights_sha256": hd["weights_sha256"], "targets_sha256": TARGET_SHA,
            "normalization_mask": plan["configurations"]["2d"]["use_mask_for_norm"]}


def verify_files(files):
    for name, digest in files.items():
        require(sha(name) == digest, "Audited file changed: " + name)


def guard(gpu, expected_uuid=None):
    from experiments.genseg_joint.runtime import gpu_guard
    return gpu_guard(gpu, expected_uuid=expected_uuid, min_free_vram_gib=6, min_free_ram_gib=8)


def prepare(args):
    info = check(args)
    # Expensive read-only audit is finished before resource guard and arrays/model.
    snap = guard(args.gpu)
    from .native import build
    build(args, info, snap)


def ready(args, *, before_preprocess=False):
    output = Path(args.output).resolve(strict=True)
    receipt = read(confined(output, "complete.json"))
    require(receipt.get("complete") is True and receipt.get("protocol") == PROTOCOL, "Completed preparation required")
    info = receipt["inputs"]
    require(Path(info["output"]) == output and info["targets_sha256"] == TARGET_SHA, "Receipt identity mismatch")
    separate(output, [Path(info["raw"]).parent, Path(info["pre"]).parent, info["targets"], REPO])
    verify_files(info["files"])
    for relative, digest in receipt["artifacts"].items():
        require(sha(confined(output, relative)) == digest, "Prepared artifact changed: " + relative)
    for env in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        require(os.environ.get(env) and Path(os.environ[env]).resolve(strict=True) == output / env, "Set isolated environment " + env)
    raw, pre = output / "nnUNet_raw" / NAME, output / "nnUNet_preprocessed" / NAME
    train, val = splits(pre / "splits_final.json")
    require(train == info["train"] and val == info["val"] and set(receipt["cases"]) == set(train + val), "Receipt cohort mismatch")
    for name in ("nnUNetPlans.json", "splits_final.json", "dataset.json", "dataset_fingerprint.json"):
        require(sha(pre / name) == info["files"][str(Path(info["pre"]) / name)], "Inherited metadata changed")
    fingerprint_sha = info["files"][str(Path(info["pre"]) / "dataset_fingerprint.json")]
    require(receipt.get("fingerprint_status") == FINGERPRINT_STATUS, "Fingerprint must be explicitly inherited, not derived")
    for relative in (Path("nnUNet_preprocessed") / NAME / "dataset_fingerprint.json",
                     Path("provenance/inherited_fixed_baseline_dataset_fingerprint.json")):
        require(receipt["artifacts"].get(str(relative)) == fingerprint_sha and
                sha(confined(output, str(relative))) == fingerprint_sha, "Inherited fingerprint/provenance hash mismatch")
    require(sha(raw / "dataset.json") == info["files"][str(Path(info["raw"]) / "dataset.json")], "Raw dataset metadata changed")
    identifier = plan_contract(read(pre / "nnUNetPlans.json"), read(pre / "dataset.json"))
    require(identifier == info["identifier"], "Configuration identifier changed")
    folder = pre / identifier
    require(raw.resolve(strict=True) == raw and pre.resolve(strict=True) == pre, "Derived dataset path redirected")
    fold = output / "nnUNet_results" / NAME / (TRAINER + "__nnUNetPlans__2d") / "fold_0"
    require(fold.resolve(strict=False) == fold and output in fold.resolve(strict=False).parents,
            "Results path redirected outside intended isolated path")
    require(not fold.exists() and not fold.is_symlink(), "Existing fold_0 refused, even if empty; fresh training only")
    require(all(r.get("readback_verified") is True for r in receipt["cases"].values()) and
            sum(r.get("mask_source") == "cached76" for r in receipt["cases"].values()) == 76 and
            sum(r.get("mask_source") == "new19" for r in receipt["cases"].values()) == 19, "Incomplete per-case preparation receipt")
    for patient in train + val:
        case = receipt["cases"][patient]
        require(case.get("mask_source") == ("cached76" if patient in train else "new19") and
                case.get("raw_sha256") == info["cases"][patient]["raw_sha256"] and
                case.get("label_sha256") == info["cases"][patient]["label_sha256"], "Per-case source/role receipt mismatch")
        for relative, digest in (
            (f"nnUNet_raw/{NAME}/imagesTr/{patient}_0000.nii.gz", case.get("image_sha256")),
            (f"nnUNet_raw/{NAME}/labelsTr/{patient}.nii.gz", case.get("label_sha256")),
        ):
            # Artifact maps use the platform's Path spelling, including Windows separators.
            key = str(Path(relative))
            require(isinstance(digest, str) and receipt["artifacts"].get(key) == digest,
                    "Missing or inconsistent per-case output hash")
    for directory, ending in ((raw / "imagesTr", "_0000.nii.gz"), (raw / "labelsTr", ".nii.gz")):
        require({p.name for p in directory.iterdir()} == {p + ending for p in train + val}, "Derived raw exact95 required")
    if before_preprocess:
        require(not folder.exists() and not folder.is_symlink(), "Preprocess target already exists; official preprocessor deletes it; refusing rerun")
    else:
        require(folder.resolve(strict=True) == folder, "Preprocessed data_identifier redirected")
        check_triplets(folder, train + val)
    check_trainer_metadata()
    print("PREPROCESS METADATA PASS" if before_preprocess else "READY METADATA PASS")
    print("Actual Trainer import and manual QA approval remain separate; run guard adjacent to compute")

def check_triplets(folder, ids):
    expected = {p + suffix for p in ids for suffix in (".b2nd", "_seg.b2nd", ".pkl")}
    actual = {p.name for p in folder.iterdir() if p.suffix in (".b2nd", ".pkl", ".npz", ".npy")}
    require(actual == expected and all(confined(folder, n).stat().st_size > 0 for n in expected), "Exact95 preprocessed triplets required")


def check_trainer_metadata():
    require(importlib.metadata.version("nnunetv2") == "2.8.1", "nnunetv2 2.8.1 required")
    trainer_dir = REPO / "nnunet_ext_trainers"
    require(os.environ.get("nnUNet_extTrainer") and Path(os.environ["nnUNet_extTrainer"]).resolve(strict=True) == trainer_dir,
            "Set nnUNet_extTrainer to this checkout's nnunet_ext_trainers")
    hashes = {str(confined(trainer_dir, name + ".py")): sha(confined(trainer_dir, name + ".py"))
              for name in (TRAINER, "nnUNetTrainerEarlyStopping", "nnUNetTrainerTopK10")}
    print("Trainer source hashes=" + json.dumps(hashes, sort_keys=True))
    return hashes


def trainer_check(args):
    """SERVER ONLY: actual official discovery; no Trainer/model construction."""
    hashes = check_trainer_metadata()
    guard(args.gpu)  # Adjacent to imports; no local execution is authorized.
    import importlib
    import inspect
    import sys
    previous_path = list(sys.path)
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        sys.path.insert(0, str(REPO / "nnunet_ext_trainers"))
        module = importlib.import_module(TRAINER)
        expected = getattr(module, TRAINER)
        require(Path(inspect.getfile(expected)).resolve() == REPO / "nnunet_ext_trainers" / (TRAINER + ".py"), "Foreign combination Trainer imported")
        from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
        actual = recursive_find_trainer_class_by_name(TRAINER)
        require(actual is expected and actual.__name__ == TRAINER, "Official discovery resolved a different Trainer")
        for method, owner in (("_build_loss", "nnUNetTrainerTopK10"), ("__init__", "nnUNetTrainerEarlyStopping")):
            require(Path(inspect.getfile(getattr(actual, method))).resolve() == REPO / "nnunet_ext_trainers" / (owner + ".py"), "Trainer method binding mismatch: " + method)
        verify_files(hashes)
        print("TRAINER DISCOVERY PASS: " + actual.__module__ + "." + actual.__name__)
        print("No Trainer instance, model, or checkpoint was constructed")
    finally:
        sys.path[:] = previous_path
        sys.dont_write_bytecode = previous_bytecode


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("check", "prepare"):
        lane = sub.add_parser(command)
        for flag in ("raw-dataset", "preprocessed-dataset", "targets", "targets-sha256", "output"):
            lane.add_argument("--" + flag, required=True)
        lane.add_argument("--gpu", type=int, default=1)
        if command == "prepare":
            lane.add_argument("--audit-label-coverage", action="store_true", help="Optional train76 GT report; never changes masks")
    for command in ("ready", "preprocess-check"):
        lane = sub.add_parser(command)
        lane.add_argument("--output", required=True)
        lane.add_argument("--gpu", type=int, default=1, help="Selection for subsequent guard; no resource queries")
    for command in ("guard", "trainer-check"):
        lane = sub.add_parser(command)
        lane.add_argument("--gpu", type=int, default=1)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "check":
            check(args)
            print("CHECK METADATA PASS:95 cases, canonical76/19, pinned targets/weights/plans; no arrays/model/output")
        elif args.command == "prepare":
            prepare(args)
        elif args.command in ("ready", "preprocess-check"):
            ready(args, before_preprocess=args.command == "preprocess-check")
        elif args.command == "trainer-check":
            trainer_check(args)
        else:
            print(json.dumps(guard(args.gpu), sort_keys=True))
    except (ValueError, OSError, KeyError, importlib.metadata.PackageNotFoundError) as exc:
        raise SystemExit("REFUSED: " + str(exc)) from exc


if __name__ == "__main__":
    main()
