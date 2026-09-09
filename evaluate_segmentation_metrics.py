#!/usr/bin/env python
"""
Evaluate binary 3D medical-image segmentation masks with SrSNet-style metrics.

Metrics:
- Dice
- IoU
- F2 (paper-compatible or standard)
- AVD (%)
- LCD
- Recall
- HD95 (mm)

Input:
  --pred-dir: predicted NIfTI masks (*.nii / *.nii.gz)
  --gt-dir:   ground-truth NIfTI masks with matching case filenames

Outputs:
  case_metrics.csv
  summary_metrics.csv
  summary_metrics.json

Notes
-----
1) This script computes metrics from prediction masks + GT masks. It does not
   "convert Dice" into the other metrics.
2) SrSNet (Li et al., 2024) prints F2 as:
       5TP / (5TP + 4FP + FN)
   which differs from the conventional F2:
       5TP / (5TP + FP + 4FN)
   Use --f2-mode paper to reproduce the paper's printed equation, or
   --f2-mode standard for the conventional F2 definition.
3) LCD uses full-connectivity connected components:
   26-connectivity in 3D, 8-connectivity in 2D.
4) HD95 is computed from bidirectional surface distances in physical mm.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage


def case_id_from_path(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def collect_nifti(directory: Path) -> dict[str, Path]:
    paths = list(directory.glob("*.nii.gz")) + list(directory.glob("*.nii"))
    result: dict[str, Path] = {}
    for p in sorted(paths):
        cid = case_id_from_path(p)
        if cid in result:
            raise RuntimeError(f"Duplicate case id {cid!r}: {result[cid]} and {p}")
        result[cid] = p
    return result


def assert_same_geometry(pred_img: sitk.Image, gt_img: sitk.Image, case_id: str) -> None:
    if pred_img.GetSize() != gt_img.GetSize():
        raise ValueError(
            f"{case_id}: size mismatch: pred={pred_img.GetSize()}, gt={gt_img.GetSize()}"
        )

    checks = [
        ("spacing", pred_img.GetSpacing(), gt_img.GetSpacing()),
        ("origin", pred_img.GetOrigin(), gt_img.GetOrigin()),
        ("direction", pred_img.GetDirection(), gt_img.GetDirection()),
    ]
    for name, a, b in checks:
        if not np.allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-5):
            raise ValueError(f"{case_id}: {name} mismatch: pred={a}, gt={b}")


def safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den != 0 else float("nan")


def lesion_count(mask: np.ndarray) -> int:
    if mask.ndim not in (2, 3):
        raise ValueError(f"LCD supports 2D/3D masks, got ndim={mask.ndim}")
    structure = np.ones((3,) * mask.ndim, dtype=np.uint8)
    _, n = ndimage.label(mask, structure=structure)
    return int(n)


def surface(mask: np.ndarray) -> np.ndarray:
    if mask.ndim not in (2, 3):
        raise ValueError(f"HD95 supports 2D/3D masks, got ndim={mask.ndim}")
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    return np.logical_xor(mask, eroded)


def hd95_mm(pred: np.ndarray, gt: np.ndarray, spacing_array_order: tuple[float, ...]) -> float:
    pred_any = bool(pred.any())
    gt_any = bool(gt.any())

    if not pred_any and not gt_any:
        return 0.0
    if not pred_any or not gt_any:
        # No finite surface-to-surface distance exists if only one mask is empty.
        return float("nan")

    pred_s = surface(pred)
    gt_s = surface(gt)

    # distance_transform_edt(~surface) gives distance to the nearest surface voxel.
    dt_to_gt = ndimage.distance_transform_edt(~gt_s, sampling=spacing_array_order)
    dt_to_pred = ndimage.distance_transform_edt(~pred_s, sampling=spacing_array_order)

    d_pred_to_gt = dt_to_gt[pred_s]
    d_gt_to_pred = dt_to_pred[gt_s]

    distances = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(distances, 95))


def evaluate_case(
    pred_path: Path,
    gt_path: Path,
    f2_mode: str,
) -> dict[str, float | int | str]:
    case_id = case_id_from_path(pred_path)

    pred_img = sitk.ReadImage(str(pred_path))
    gt_img = sitk.ReadImage(str(gt_path))
    assert_same_geometry(pred_img, gt_img, case_id)

    pred = sitk.GetArrayFromImage(pred_img) > 0
    gt = sitk.GetArrayFromImage(gt_img) > 0

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())

    pred_voxels = int(pred.sum())
    gt_voxels = int(gt.sum())

    dice = safe_ratio(2 * tp, 2 * tp + fp + fn)
    iou = safe_ratio(tp, tp + fp + fn)
    recall = safe_ratio(tp, tp + fn)

    if f2_mode == "paper":
        # SrSNet printed equation (Eq. 13)
        f2 = safe_ratio(5 * tp, 5 * tp + 4 * fp + fn)
    elif f2_mode == "standard":
        # Conventional F_beta with beta=2
        f2 = safe_ratio(5 * tp, 5 * tp + fp + 4 * fn)
    else:
        raise ValueError(f"Unknown f2_mode: {f2_mode}")

    if gt_voxels == 0:
        avd_percent = 0.0 if pred_voxels == 0 else float("nan")
    else:
        avd_percent = abs(pred_voxels - gt_voxels) / gt_voxels * 100.0

    pred_lesions = lesion_count(pred)
    gt_lesions = lesion_count(gt)
    lcd = abs(pred_lesions - gt_lesions)

    # SimpleITK spacing is (x, y, z), while GetArrayFromImage is (z, y, x).
    spacing_xyz = tuple(float(x) for x in pred_img.GetSpacing())
    spacing_array_order = tuple(reversed(spacing_xyz))
    if pred.ndim == 2:
        spacing_array_order = spacing_array_order[-2:]

    hd95 = hd95_mm(pred, gt, spacing_array_order)

    voxel_volume_mm3 = float(np.prod(spacing_xyz))
    pred_volume_ml = pred_voxels * voxel_volume_mm3 / 1000.0
    gt_volume_ml = gt_voxels * voxel_volume_mm3 / 1000.0

    return {
        "case_id": case_id,
        "dice": dice,
        "iou": iou,
        "f2": f2,
        "avd_percent": avd_percent,
        "lcd": lcd,
        "recall": recall,
        "hd95_mm": hd95,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pred_voxels": pred_voxels,
        "gt_voxels": gt_voxels,
        "pred_lesions": pred_lesions,
        "gt_lesions": gt_lesions,
        "pred_volume_ml": pred_volume_ml,
        "gt_volume_ml": gt_volume_ml,
    }


def build_summary(df: pd.DataFrame, f2_mode: str) -> dict:
    metric_columns = [
        "dice",
        "iou",
        "f2",
        "avd_percent",
        "lcd",
        "recall",
        "hd95_mm",
    ]

    metrics = {}
    for col in metric_columns:
        values = pd.to_numeric(df[col], errors="coerce")
        metrics[col] = {
            "mean": float(values.mean(skipna=True)),
            "median": float(values.median(skipna=True)),
            "std": float(values.std(skipna=True, ddof=1)) if values.notna().sum() > 1 else 0.0,
            "valid_cases": int(values.notna().sum()),
        }

    return {
        "n_cases": int(len(df)),
        "aggregation": "macro average over cases",
        "f2_mode": f2_mode,
        "lcd_connectivity": "26-connectivity for 3D / 8-connectivity for 2D",
        "hd95": "95th percentile of bidirectional surface distances in physical mm",
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute SrSNet-style segmentation metrics from NIfTI predictions and GT."
    )
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--f2-mode",
        choices=("paper", "standard"),
        default="paper",
        help=(
            "'paper': 5TP/(5TP+4FP+FN), matching the equation printed in SrSNet; "
            "'standard': conventional F2 = 5TP/(5TP+FP+4FN)."
        ),
    )
    args = parser.parse_args()

    pred_files = collect_nifti(args.pred_dir)
    gt_files = collect_nifti(args.gt_dir)

    if not pred_files:
        raise RuntimeError(f"No NIfTI files found in prediction directory: {args.pred_dir}")
    if not gt_files:
        raise RuntimeError(f"No NIfTI files found in GT directory: {args.gt_dir}")

    
    pred_ids = set(pred_files)
    gt_ids = set(gt_files)

    # Every prediction must have a corresponding GT.
    # Extra GT cases are allowed and ignored. This supports fold-level evaluation
    # against a full labelsTr directory.
    missing_gt = sorted(pred_ids - gt_ids)

    if missing_gt:
        raise RuntimeError(
            "Some predictions do not have matching GT.\n"
            f"Missing GT for: {missing_gt[:20]}"
        )

    extra_gt = sorted(gt_ids - pred_ids)
    if extra_gt:
        print(
            f"GT directory contains {len(extra_gt)} additional cases; "
            "they will be ignored."
        )

    rows = [
        evaluate_case(pred_files[cid], gt_files[cid], args.f2_mode)
        for cid in sorted(pred_ids)
    ]
    df = pd.DataFrame(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    case_csv = args.output_dir / "case_metrics.csv"
    df.to_csv(case_csv, index=False)

    summary = build_summary(df, args.f2_mode)
    summary_json = args.output_dir / "summary_metrics.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Paper-like one-row table: overlap metrics shown as percentages.
    paper_row = {
        "Dice (%)": 100.0 * summary["metrics"]["dice"]["mean"],
        "IOU (%)": 100.0 * summary["metrics"]["iou"]["mean"],
        "F2 (%)": 100.0 * summary["metrics"]["f2"]["mean"],
        "AVD (%)": summary["metrics"]["avd_percent"]["mean"],
        "LCD": summary["metrics"]["lcd"]["mean"],
        "Rec (%)": 100.0 * summary["metrics"]["recall"]["mean"],
        "HD95 (mm)": summary["metrics"]["hd95_mm"]["mean"],
    }
    summary_csv = args.output_dir / "summary_metrics.csv"
    pd.DataFrame([paper_row]).to_csv(summary_csv, index=False)

    print(f"Evaluated {len(df)} cases")
    print(pd.DataFrame([paper_row]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved:\n  {case_csv}\n  {summary_csv}\n  {summary_json}")


if __name__ == "__main__":
    main()
