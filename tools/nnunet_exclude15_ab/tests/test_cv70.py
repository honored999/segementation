"""Tiny synthetic CV70 tests; no medical data or model imports."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv70
import run_experiment as run


def inputs(base):
    ids = [f"synthetic_subject_{i:03d}" for i in range(95)]
    folds = [{"train": [x for x in ids if x not in ids[i*19:(i+1)*19]],
              "val": ids[i*19:(i+1)*19]} for i in range(5)]
    source = base / "source"
    source.mkdir()
    splits = source / "original.json"
    splits.write_text(json.dumps(folds), encoding="utf-8")
    summary = source / "summary.json"
    records = [{"prediction_file": "D:\\nnUNet_results\\Dataset501_StrokeLesion\\nnUNetTrainer__nnUNetPlans__2d\\crossval_results_folds_0_1_2_3_4\\" + x + ".nii.gz",
                "metrics": {"1": {"Dice": i/100}}} for i, x in enumerate(ids)]
    summary.write_text(json.dumps({"metric_per_case": records}), encoding="utf-8")
    return ids, folds, summary, splits


class CV70Tests(unittest.TestCase):
    def test_prepare_rank_split_reproducibility_load_and_nooverwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ids, folds, summary, splits = inputs(base)
            a = cv70.prepare(summary, splits, base / "private" / "a.json")
            b = cv70.prepare(summary, splits, base / "private" / "b.json")
            self.assertEqual(a, b)
            self.assertEqual(a["excluded"], ids[:25])
            self.assertEqual(a["retained"], ids[25:])
            self.assertEqual(cv70.load_protocol(base / "private" / "a.json", folds), a)
            val = []
            for f in a["folds"]:
                self.assertEqual((len(f["train"]), len(f["val"])), (56, 14))
                self.assertFalse(set(f["train"]) & set(f["val"]))
                val += f["val"]
            self.assertEqual(sorted(val), ids[25:])
            with self.assertRaises(FileExistsError):
                cv70.prepare(summary, splits, base / "private" / "a.json")
            with self.assertRaises(ValueError):
                cv70.prepare(summary, splits, base / "source" / "new.json")

    def test_ranking_invalid_and_ties_and_source_guard(self):
        for defect in ("nan", "range", "duplicate", "missing", "fold0", "dataset", "trainer", "fold_coverage", "tie"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                ids, folds, summary, splits = inputs(base)
                s = json.loads(summary.read_text())
                if defect == "nan": s["metric_per_case"][0]["metrics"]["1"]["Dice"] = float("nan")
                if defect == "range": s["metric_per_case"][0]["metrics"]["1"]["Dice"] = 1.1
                if defect == "duplicate": s["metric_per_case"][-1] = s["metric_per_case"][0]
                if defect == "missing": s["metric_per_case"].pop()
                if defect in ("fold0", "dataset", "trainer"):
                    old, new = {"fold0": ("crossval_results_folds_0_1_2_3_4", "fold_0"),
                                "dataset": ("Dataset501_StrokeLesion", "Dataset508_ISLES2022DWI"),
                                "trainer": ("nnUNetTrainer__", "nnUNetTrainerOther__")}[defect]
                    s["metric_per_case"][0]["prediction_file"] = s["metric_per_case"][0]["prediction_file"].replace(old, new)
                if defect == "fold_coverage":
                    folds[1] = copy.deepcopy(folds[0])
                    splits.write_text(json.dumps(folds))
                if defect == "tie":
                    for r in s["metric_per_case"]: r["metrics"]["1"]["Dice"] = .5
                    s["metric_per_case"].reverse()
                summary.write_text(json.dumps(s))
                if defect == "tie":
                    self.assertEqual(cv70.prepare(summary, splits, base / "private.json")["excluded"], ids[:25])
                else:
                    with self.assertRaises(ValueError):
                        cv70.prepare(summary, splits, base / "private.json")
                    self.assertFalse((base / "private.json").exists())

    def test_load_rejects_tamper_even_rehashed_invalid_structure(self):
        for defect in ("digest", "duplicate", "missing", "overlap", "unknown", "rank"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                ids, folds, summary, splits = inputs(base)
                path = base / "private.json"
                p = cv70.prepare(summary, splits, path)
                if defect == "digest": p["split_seed"] = 9
                if defect == "duplicate": p["folds"][1]["fold"] = 0
                if defect == "missing": p["folds"].pop()
                if defect == "overlap": p["folds"][0]["train"][0] = p["folds"][0]["val"][0]
                if defect == "unknown": p["retained"][0] = "synthetic_unknown"
                if defect == "rank": p["excluded"], p["retained"] = p["retained"][:25], p["excluded"] + p["retained"][25:]
                if defect != "digest": p["protocol_sha256"] = cv70.digest(p)
                path.write_text(json.dumps(p))
                with self.assertRaises(ValueError):
                    cv70.load_protocol(path, folds)

    def test_aggregation_complete_and_rejects_bad_runs(self):
        for defect in ("good", "incomplete", "duplicate_fold", "cohort", "config", "pretrain", "seed", "metric", "unchanged", "existing", "writefailure"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                ids, folds, summary, splits = inputs(base)
                protocol_path = base / "protocol.json"
                p = cv70.prepare(summary, splits, protocol_path)
                cfg = {"cv70_protocol": str(protocol_path), "results_parent": str(base / "results"),
                       "raw_root": str(base / "raw"), "preprocessed_root": str(base / "prep"),
                       "pretrained_checkpoint": str(base / "weights" / "initial.pth")}
                cfg_path = base / "config.json"
                cfg_path.write_text(json.dumps(cfg))
                for f in p["folds"]:
                    out = base / "results" / "seed_0" / f"fold_{f['fold']}"
                    out.mkdir(parents=True)
                    manifest = {"fold": f["fold"], "seed": 0, "config": cfg,
                                "config_sha256": cv70.config_digest(cfg),
                                "cv70_protocol_sha256": p["protocol_sha256"],
                                "experiment_class": cv70.EXPERIMENT_CLASS, "plans_policy": cv70.PLANS_POLICY,
                                "ranking_provenance": p["ranking_source"], "nnunetv2_version": "2.8.1",
                                "cohort_source_protocol_sha256": p["protocol_sha256"],
                                "cv70_protocol_file_sha256": cv70.hash_file(protocol_path),
                                "pretrained_sha256": "a"*64, "status": "COMPLETE",
                                "canonical_metadata_unchanged": True, "protected_hashes": {"synthetic": "b"*64},
                                "train": f["train"], "selection_validation": f["val"],
                                "evaluation": {"evaluation_cases": f["val"], "selection_cases": f["val"]}}
                    if f["fold"] == 1:
                        if defect == "incomplete": manifest["status"] = "FAILED_OR_ABORTED"
                        if defect == "duplicate_fold": manifest["fold"] = 0
                        if defect == "config": manifest["config"]["gpu"] = "9"
                        if defect == "pretrain": manifest["pretrained_sha256"] = "c"*64
                        if defect == "seed": manifest["seed"] = 1
                        if defect == "unchanged": manifest["canonical_metadata_unchanged"] = False
                    (out / "experiment_manifest.json").write_text(json.dumps(manifest))
                    # Prevent fixture mutation of shared cfg above affecting subsequent folds.
                    cfg.pop("gpu", None)
                    with (out / "evaluation14_case_metrics.csv").open("w", newline="") as stream:
                        w = csv.writer(stream)
                        w.writerow(["case_id", "dice_0_to_1", "IoU_0_to_1", "TP_voxels", "FP_voxels", "FN_voxels", "selection14_member"])
                        cases = f["val"] if not (defect == "cohort" and f["fold"] == 1) else [f["val"][0]]*14
                        for case in cases:
                            w.writerow([case, .9 if defect == "metric" and f["fold"] == 1 else .5, 1/3, 2, 2, 2, True])
                if defect == "existing":
                    (base / "results" / "seed_0" / "oof70_summary.json").write_text("preserve")
                if defect == "good":
                    report = cv70.aggregate(cfg_path, original_path=splits)
                    self.assertEqual(report["cases"], 70)
                    self.assertAlmostEqual(report["macro_dice_0_to_1"], .5)
                    self.assertEqual(report["TP_voxels_total"], 140)
                elif defect == "writefailure":
                    with patch.object(cv70.json, "dump", side_effect=OSError("synthetic write failure")):
                        with self.assertRaises(OSError):
                            cv70.aggregate(cfg_path, original_path=splits)
                    self.assertFalse((base / "results" / "seed_0" / "oof70_summary.json").exists())
                    self.assertFalse((base / "results" / "seed_0" / "oof70_case_metrics.csv").exists())
                else:
                    with self.assertRaises((ValueError, FileExistsError)):
                        cv70.aggregate(cfg_path, original_path=splits)
                    self.assertFalse((base / "results" / "seed_0" / "oof70_case_metrics.csv").exists())

    def test_cv_metadata_only_check_shared_guards_relative_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ids, folds, summary, splits = inputs(base)
            tool = base / "tool"
            tool.mkdir()
            cv70.prepare(summary, splits, tool / "cv70_protocol.json")
            plans = {"configurations": {"2d": {"batch_size": 12, "patch_size": [512,512],
                     "data_identifier": "nnUNetPlans_2d", "architecture": {"synthetic": True}}}}
            prep = base / "prep" / run.DATASET
            prep.mkdir(parents=True)
            for path, value in ((tool / "original_splits.json", folds), (prep / "splits_final.json", folds),
                                (tool / "expected_plans.json", plans), (prep / "nnUNetPlans.json", plans),
                                (tool / "expected_dataset.json", {"synthetic": True}),
                                (prep / "dataset.json", {"synthetic": True})):
                path.write_text(json.dumps(value))
            source_hashes = {}
            (tool / "trainers").mkdir()
            for name in ("nnUNetTrainerEarlyStopping.py", "nnUNetTrainerTopK10.py", "nnUNetTrainerTopK10EarlyStopping.py"):
                path = tool / "trainers" / name
                path.write_text("# synthetic metadata snapshot\\n")
                source_hashes[name] = run.hash_file(path)
            (tool / "experiment_config.example.json").write_text(json.dumps({"source_hashes": source_hashes}))
            weights = base / "weights"
            weights.mkdir()
            checkpoint = weights / "synthetic.bin"
            checkpoint.write_bytes(b"x")
            pretrained_plans = weights / "plans.json"
            pretrained_plans.write_text(json.dumps(plans))
            cfg = {"raw_root": str(base / "raw"), "preprocessed_root": str(base / "prep"),
                   "results_parent": str(base / "results"), "pretrained_checkpoint": str(checkpoint),
                   "pretrained_expected_bytes": 1, "pretrained_plans": str(pretrained_plans),
                   "trainer": "nnUNetTrainerTopK10EarlyStopping", "source_hashes": source_hashes,
                   "cv70_protocol": "cv70_protocol.json", "gpu": "1"}
            cfg_path = tool / "config.json"
            cfg_path.write_text(json.dumps(cfg))
            with patch.object(run, "ROOT", tool), patch.object(run.importlib.metadata, "version", return_value="2.8.1"):
                result, train, selection, evaluation, hashes = run.verify(cfg, None, 0, 4, cfg_path)
                self.assertEqual(result, base / "results" / "seed_0" / "fold_4")
                self.assertEqual((len(train), len(selection), len(evaluation)), (56,14,14))
                self.assertEqual(selection, evaluation)
                self.assertFalse(result.exists())
                argv = ["run_experiment.py", "--config", str(cfg_path), "--cv70-fold", "4", "--check-only"]
                with patch.object(sys, "argv", argv), patch.object(run.importlib, "import_module", side_effect=AssertionError("Trainer import forbidden")), patch.object(run, "resource_check", side_effect=AssertionError("Training forbidden")):
                    run.main()
                cfg["source_hashes"] = {}
                with self.assertRaisesRegex(ValueError, "snapshot hash contract"):
                    run.verify(cfg, None, 0, 4, cfg_path)

    def test_paper_f2_unequal_errors_and_zero_gt_nulls(self):
        m = cv70.metric_values({"dice_0_to_1": 4/9, "IoU_0_to_1": 2/7,
                               "TP_voxels": 2, "FP_voxels": 4, "FN_voxels": 1})
        self.assertAlmostEqual(m["paper_f2"], 10/27)
        m = cv70.metric_values({"dice_0_to_1": 0, "IoU_0_to_1": 0,
                               "TP_voxels": 0, "FP_voxels": 1, "FN_voxels": 0})
        self.assertIsNone(m["recall"])
        self.assertIsNone(m["avd_percent"])
        self.assertEqual(m["paper_f2"], 0)

    def test_old_ab_and_cv_export_windows_names(self):
        ids = [f"synthetic_subject_{i:03d}" for i in range(95)]
        folds = [{"train": [x for x in ids if x not in ids[i*19:(i+1)*19]], "val": ids[i*19:(i+1)*19]} for i in range(5)]
        self.assertEqual(tuple(map(len, run.memberships(folds, ids[:2]+ids[19:32], "A"))), (76,17,19))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            cases = ids[:14]
            records = [{"prediction_file": "D:\\pred\\"+c+".nii.gz", "metrics": {"1": {"Dice": .5, "IoU": 1/3, "TP": 2, "FP": 2, "FN": 2}}} for c in cases]
            summary = out / "summary.json"
            summary.write_text(json.dumps({"metric_per_case": records}))
            report = run.export_metrics(summary, cases, cases, out, cohort_label="evaluation14")
            self.assertEqual(report["evaluation_cases"], cases)
            self.assertTrue((out / "evaluation14_case_metrics.csv").is_file())


if __name__ == "__main__":
    unittest.main()
