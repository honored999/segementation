"""Smoke-only per-case CSV and JSON summary reporting."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any


FIELDS = ("case_id", "dice", "iou", "precision", "recall", "gt_voxels", "pred_voxels", "tp", "fp", "fn")


def write_case_reports(records: list[dict[str, Any]], validation_dir: Path, *, fold: int, checkpoint_path: Path) -> None:
    validation_dir.mkdir(parents=True, exist_ok=True)
    with (validation_dir / "per_case_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerows(records)
    dices = [float(record["dice"]) for record in records]
    def mean(key: str) -> float: return sum(float(record[key]) for record in records) / len(records) if records else 0.0
    ranked = sorted(records, key=lambda record: float(record["dice"]), reverse=True)
    summary = {"run_type": "smoke_run_only", "fold": fold, "checkpoint_path": str(checkpoint_path), "num_cases": len(records), "mean_dice": mean("dice"), "median_dice": statistics.median(dices) if dices else 0.0, "std_dice": statistics.pstdev(dices) if len(dices)>1 else 0.0, "min_dice": min(dices) if dices else 0.0, "max_dice": max(dices) if dices else 0.0, "mean_iou": mean("iou"), "mean_precision": mean("precision"), "mean_recall": mean("recall"), "best_case": ranked[0]["case_id"] if ranked else None, "worst_case": ranked[-1]["case_id"] if ranked else None, "empty_gt_case_count": sum(int(record["gt_voxels"])==0 for record in records), "empty_prediction_case_count": sum(int(record["pred_voxels"])==0 for record in records), "failed_case_count": 0, "inference": "2D per-z argmax, reassembled zyx", "preprocessing": "Z-score; matching spacing bypasses resampling"}
    (validation_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    def write_ranked(path: Path, selected: list[dict[str, Any]]) -> None:
        path.write_text("\n".join(f"case_id={r['case_id']} Dice={r['dice']} IoU={r['iou']} GT_voxels={r['gt_voxels']} Pred_voxels={r['pred_voxels']}" for r in selected), encoding="utf-8")
    write_ranked(validation_dir / "best_cases.txt", ranked[:5])
    write_ranked(validation_dir / "worst_cases.txt", list(reversed(ranked[-5:])))
