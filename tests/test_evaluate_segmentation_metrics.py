import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk

SCRIPT = Path(__file__).resolve().parents[1] / "evaluate_segmentation_metrics.py"
spec = importlib.util.spec_from_file_location("segmentation_metrics", SCRIPT)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)
METRICS = ("dice", "iou", "f2", "avd_percent", "lcd", "recall", "hd95_mm")


def make_cases(root, mode):
    pred_dir, gt_dir = root / "pred", root / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    rows = []
    for name, shape, pred_indices, gt_indices in (
        ("small", (2, 2, 2), [0, 1], [0, 2]),
        ("large", (3, 3, 3), list(range(9)), list(range(8)) + [10, 11]),
    ):
        for directory, indices in ((pred_dir, pred_indices), (gt_dir, gt_indices)):
            array = np.zeros(shape, dtype=np.uint8)
            array.flat[indices] = 1
            sitk.WriteImage(sitk.GetImageFromArray(array), str(directory / (name + ".nii.gz")))
        rows.append(metrics.evaluate_case(pred_dir / (name + ".nii.gz"), gt_dir / (name + ".nii.gz"), mode))
    return pred_dir, gt_dir, pd.DataFrame(rows)


@pytest.mark.parametrize("mode", ["paper", "standard"])
def test_counts_and_macro_average(tmp_path, mode):
    _, _, df = make_cases(tmp_path, mode)
    summary = metrics.build_summary(df, mode)
    assert summary["voxel_counts"] == {
        "aggregation": "sum over evaluated cases", "unit": "voxels", "tp": 9, "fp": 2, "fn": 3
    }
    for key in ("tp", "fp", "fn"):
        assert type(summary["voxel_counts"][key]) is int
        assert summary["voxel_counts"][key] == sum(int(value) for value in df[key])
    for key in METRICS:
        assert summary["metrics"][key]["mean"] == pytest.approx(df[key].mean())
    assert summary["metrics"]["dice"]["mean"] != pytest.approx(18 / 23)
    weights = (4, 1) if mode == "paper" else (1, 4)
    for row in df.to_dict("records"):
        assert row["f2"] == pytest.approx(5 * row["tp"] / (5 * row["tp"] + weights[0] * row["fp"] + weights[1] * row["fn"]))
    df.loc[0, "hd95_mm"] = np.nan
    nan_summary = metrics.build_summary(df, mode)
    assert nan_summary["metrics"]["hd95_mm"]["valid_cases"] == 1
    assert nan_summary["metrics"]["hd95_mm"]["mean"] == df.loc[1, "hd95_mm"]
    assert nan_summary["voxel_counts"] == summary["voxel_counts"]


@pytest.mark.parametrize("mode", ["paper", "standard"])
def test_cli_counts(tmp_path, mode):
    pred_dir, gt_dir, df = make_cases(tmp_path, mode)
    sitk.WriteImage(sitk.GetImageFromArray(np.ones((2, 2, 2), dtype=np.uint8)), str(gt_dir / "extra.nii.gz"))
    output_dir = tmp_path / "output"
    result = subprocess.run(
        [sys.executable, "-B", str(SCRIPT), "--pred-dir", str(pred_dir), "--gt-dir", str(gt_dir), "--output-dir", str(output_dir), "--f2-mode", mode],
        capture_output=True, text=True, check=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    summary = json.loads((output_dir / "summary_metrics.json").read_text(encoding="utf-8"))
    assert summary["n_cases"] == 2
    assert summary["aggregation"] == "macro average over cases"
    assert summary["f2_mode"] == mode
    assert "lcd_connectivity" in summary and "hd95" in summary
    with (output_dir / "summary_metrics.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        row = next(reader)
        assert reader.fieldnames == ["Dice (%)", "IOU (%)", "F2 (%)", "AVD (%)", "LCD", "Rec (%)", "HD95 (mm)", "TP (total)", "FP (total)", "FN (total)"]
    case_df = pd.read_csv(output_dir / "case_metrics.csv")
    for key, expected in (("tp", 9), ("fp", 2), ("fn", 3)):
        assert type(summary["voxel_counts"][key]) is int
        assert summary["voxel_counts"][key] == expected == int(case_df[key].sum())
        assert row[key.upper() + " (total)"] == str(expected)
    lines = result.stdout.splitlines()
    header = next(index for index, line in enumerate(lines) if "TP (total)" in line)
    assert lines[header + 1].split()[-3:] == ["9", "2", "3"]
    assert "they will be ignored" in result.stdout
    for key in METRICS:
        assert summary["metrics"][key]["mean"] == pytest.approx(df[key].mean())
    for label, key, factor in (("Dice (%)", "dice", 100), ("IOU (%)", "iou", 100), ("F2 (%)", "f2", 100), ("AVD (%)", "avd_percent", 1), ("LCD", "lcd", 1), ("Rec (%)", "recall", 100), ("HD95 (mm)", "hd95_mm", 1)):
        assert float(row[label]) == pytest.approx(factor * df[key].mean())
