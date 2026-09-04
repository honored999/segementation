from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk

CASE_RE = re.compile(r"(case\d+)", re.IGNORECASE)


def case_id_from_name(name: str) -> str:
    match = CASE_RE.search(name.replace("\\", "/"))
    if not match:
        raise ValueError(f"无法识别病例编号：{name}")
    return match.group(1).lower()


def find_cases(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        raise NotADirectoryError(f"目录不存在：{folder}")
    files: dict[str, Path] = {}
    for path in sorted(folder.glob("case*.nii*")):
        cid = case_id_from_name(path.name)
        if cid in files:
            raise RuntimeError(f"发现重复病例文件：{cid}")
        files[cid] = path
    if not files:
        raise RuntimeError(f"未找到 case*.nii 或 case*.nii.gz：{folder}")
    return files


def read_mask(path: Path) -> tuple[np.ndarray, sitk.Image]:
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image) > 0  # [z, y, x]
    return array, image


def same_geometry(a: sitk.Image, b: sitk.Image, atol: float = 1e-5) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=atol, rtol=0)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=atol, rtol=0)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=atol, rtol=0)
    )


def counts(gt: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.logical_and(gt, pred).sum(dtype=np.int64))
    fp = int(np.logical_and(~gt, pred).sum(dtype=np.int64))
    fn = int(np.logical_and(gt, ~pred).sum(dtype=np.int64))
    tn = int(np.logical_and(~gt, ~pred).sum(dtype=np.int64))
    return tp, fp, fn, tn


def dice(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return math.nan if denominator == 0 else 2.0 * tp / denominator


def iou(tp: int, fp: int, fn: int) -> float:
    denominator = tp + fp + fn
    return math.nan if denominator == 0 else tp / denominator


def finite(values: list[float]) -> np.ndarray:
    return np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)


def mean(values: list[float]) -> float:
    arr = finite(values)
    return float(arr.mean()) if arr.size else math.nan


def median(values: list[float]) -> float:
    arr = finite(values)
    return float(np.median(arr)) if arr.size else math.nan


def std(values: list[float]) -> float:
    arr = finite(values)
    return float(arr.std(ddof=0)) if arr.size else math.nan


def json_clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_clean(v) for v in value]
    return value


def load_reference_summary(path: Path | None) -> tuple[dict[str, float], float | None]:
    if path is None:
        return {}, None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    per_case: dict[str, float] = {}
    for item in data.get("metric_per_case", []):
        cid = case_id_from_name(item["reference_file"])
        metrics = item["metrics"]
        if "1" in metrics:
            fg = metrics["1"]
        elif "(1,)" in metrics:
            fg = metrics["(1,)"]
        else:
            keys = [k for k in metrics if k not in {"0", "(0,)"}]
            if len(keys) != 1:
                raise RuntimeError(f"{cid} 无法识别前景类别：{list(metrics)}")
            fg = metrics[keys[0]]
        per_case[cid] = float(fg["Dice"])

    return per_case, float(data["foreground_mean"]["Dice"])


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="计算3D病例Dice、2D单张Dice及多种平均口径。"
    )
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, default=None)
    parser.add_argument(
        "--allow-geometry-mismatch",
        action="store_true",
        help="仅在shape一致时继续；默认空间信息不一致会停止。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gt_files = find_cases(args.labels_dir)
    pred_files = find_cases(args.pred_dir)

    if set(gt_files) != set(pred_files):
        raise RuntimeError(
            f"病例不匹配。缺少预测：{sorted(set(gt_files)-set(pred_files))}；"
            f"缺少GT：{sorted(set(pred_files)-set(gt_files))}"
        )

    reference_case_dice, reference_mean = load_reference_summary(args.reference_summary)

    slice_rows: list[dict] = []
    case_rows: list[dict] = []

    all_union_dice: list[float] = []
    all_gt_positive_dice: list[float] = []
    all_empty_as_one_dice: list[float] = []
    case_balanced_union: list[float] = []
    case_balanced_gt_positive: list[float] = []
    case_dices: list[float] = []

    total_tp = total_fp = total_fn = total_tn = 0
    max_reference_diff = 0.0
    geometry_warnings: list[str] = []

    case_ids = sorted(gt_files)
    for index, cid in enumerate(case_ids, 1):
        print(f"[{index:03d}/{len(case_ids):03d}] {cid}")
        gt, gt_img = read_mask(gt_files[cid])
        pred, pred_img = read_mask(pred_files[cid])

        if gt.shape != pred.shape:
            raise ValueError(f"{cid} shape不一致：GT={gt.shape}, Pred={pred.shape}")

        if not same_geometry(gt_img, pred_img):
            message = f"{cid} spacing/origin/direction不一致"
            if args.allow_geometry_mismatch:
                geometry_warnings.append(message)
            else:
                raise ValueError(
                    message + "。确认预测已恢复到原始标签空间；"
                    "确需仅按数组计算时加 --allow-geometry-mismatch。"
                )

        ctp, cfp, cfn, ctn = counts(gt, pred)
        case_dice = dice(ctp, cfp, cfn)
        case_iou = iou(ctp, cfp, cfn)
        case_dices.append(case_dice)
        total_tp += ctp
        total_fp += cfp
        total_fn += cfn
        total_tn += ctn

        union_values: list[float] = []
        gt_positive_values: list[float] = []
        all_values_empty_as_one: list[float] = []

        gt_positive_slices = pred_positive_slices = union_slices = both_empty_slices = 0
        gt_only_slices = pred_only_slices = 0

        for z in range(gt.shape[0]):
            gt2d = gt[z]
            pred2d = pred[z]
            tp, fp, fn, tn = counts(gt2d, pred2d)
            gt_voxels = int(gt2d.sum(dtype=np.int64))
            pred_voxels = int(pred2d.sum(dtype=np.int64))
            gt_nonempty = gt_voxels > 0
            pred_nonempty = pred_voxels > 0
            union_nonempty = gt_nonempty or pred_nonempty

            gt_positive_slices += int(gt_nonempty)
            pred_positive_slices += int(pred_nonempty)
            union_slices += int(union_nonempty)
            both_empty_slices += int(not union_nonempty)
            gt_only_slices += int(gt_nonempty and not pred_nonempty)
            pred_only_slices += int(pred_nonempty and not gt_nonempty)

            d = dice(tp, fp, fn)
            j = iou(tp, fp, fn)

            # 推荐：GT或Pred任一非空，才纳入切片平均。
            if union_nonempty:
                union_values.append(d)
                all_union_dice.append(d)

            # 另一口径：只统计GT非空切片。
            if gt_nonempty:
                gt_positive_values.append(d)
                all_gt_positive_dice.append(d)

            # 仅对照，不推荐：双空切片按1。
            d_empty_as_one = 1.0 if not union_nonempty else d
            all_values_empty_as_one.append(d_empty_as_one)
            all_empty_as_one_dice.append(d_empty_as_one)

            slice_rows.append({
                "case_id": cid,
                "z_index": z,
                "gt_voxels": gt_voxels,
                "pred_voxels": pred_voxels,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "gt_nonempty": int(gt_nonempty),
                "pred_nonempty": int(pred_nonempty),
                "union_nonempty": int(union_nonempty),
                "dice": "" if not math.isfinite(d) else f"{d:.10f}",
                "iou": "" if not math.isfinite(j) else f"{j:.10f}",
                "dice_empty_as_one": f"{d_empty_as_one:.10f}",
            })

        union_mean = mean(union_values)
        gt_positive_mean = mean(gt_positive_values)
        case_balanced_union.append(union_mean)
        case_balanced_gt_positive.append(gt_positive_mean)

        ref_dice = reference_case_dice.get(cid, math.nan)
        ref_diff = abs(case_dice - ref_dice) if math.isfinite(ref_dice) else math.nan
        if math.isfinite(ref_diff):
            max_reference_diff = max(max_reference_diff, ref_diff)

        case_rows.append({
            "case_id": cid,
            "num_slices": gt.shape[0],
            "gt_positive_slices": gt_positive_slices,
            "pred_positive_slices": pred_positive_slices,
            "union_positive_slices": union_slices,
            "both_empty_slices": both_empty_slices,
            "gt_positive_pred_empty_slices": gt_only_slices,
            "gt_empty_pred_positive_slices": pred_only_slices,
            "gt_voxels_3d": int(gt.sum(dtype=np.int64)),
            "pred_voxels_3d": int(pred.sum(dtype=np.int64)),
            "tp_3d": ctp,
            "fp_3d": cfp,
            "fn_3d": cfn,
            "tn_3d": ctn,
            "case_dice_3d": f"{case_dice:.10f}",
            "case_iou_3d": f"{case_iou:.10f}",
            "mean_slice_dice_union_nonempty": f"{union_mean:.10f}",
            "median_slice_dice_union_nonempty": f"{median(union_values):.10f}",
            "std_slice_dice_union_nonempty": f"{std(union_values):.10f}",
            "mean_slice_dice_gt_nonempty": f"{gt_positive_mean:.10f}",
            "median_slice_dice_gt_nonempty": f"{median(gt_positive_values):.10f}",
            "mean_slice_dice_all_empty_as_one_not_recommended": f"{mean(all_values_empty_as_one):.10f}",
            "reference_case_dice": "" if not math.isfinite(ref_dice) else f"{ref_dice:.10f}",
            "reference_abs_diff": "" if not math.isfinite(ref_diff) else f"{ref_diff:.12g}",
        })

    recomputed_case_mean = mean(case_dices)
    overall = {
        "definitions": {
            "mean_case_dice_3d": "每个病例完整3D体计算Dice，再对病例平均。",
            "mean_slice_dice_union_nonempty": "所有GT或Pred任一非空的切片直接平均；推荐作为主要切片级结果。",
            "case_balanced_mean_slice_dice_union_nonempty": "先在每个病例内部平均有效切片，再对病例平均；每位患者等权。",
            "mean_slice_dice_gt_nonempty": "只统计GT非空切片，会忽略纯假阳性切片。",
            "mean_slice_dice_all_empty_as_one_not_recommended": "双空切片按Dice=1，容易被背景抬高，不推荐。",
        },
        "counts": {
            "num_cases": len(case_ids),
            "num_all_slices": len(slice_rows),
            "num_union_nonempty_slices": len(all_union_dice),
            "num_gt_nonempty_slices": len(all_gt_positive_dice),
            "num_both_empty_slices": len(slice_rows) - len(all_union_dice),
        },
        "case_level_3d": {
            "mean_dice": recomputed_case_mean,
            "median_dice": median(case_dices),
            "std_dice": std(case_dices),
            "global_pooled_dice": dice(total_tp, total_fp, total_fn),
            "global_pooled_iou": iou(total_tp, total_fp, total_fn),
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "total_tn": total_tn,
        },
        "slice_level_2d": {
            "mean_dice_union_nonempty": mean(all_union_dice),
            "median_dice_union_nonempty": median(all_union_dice),
            "std_dice_union_nonempty": std(all_union_dice),
            "case_balanced_mean_dice_union_nonempty": mean(case_balanced_union),
            "mean_dice_gt_nonempty": mean(all_gt_positive_dice),
            "median_dice_gt_nonempty": median(all_gt_positive_dice),
            "case_balanced_mean_dice_gt_nonempty": mean(case_balanced_gt_positive),
            "mean_dice_all_empty_as_one_not_recommended": mean(all_empty_as_one_dice),
        },
        "reference_check": {
            "reference_summary": str(args.reference_summary) if args.reference_summary else None,
            "reference_foreground_mean_dice": reference_mean,
            "recomputed_mean_case_dice": recomputed_case_mean,
            "mean_case_dice_abs_diff": abs(recomputed_case_mean-reference_mean) if reference_mean is not None else None,
            "max_per_case_abs_diff": max_reference_diff if reference_case_dice else None,
        },
        "geometry_warnings": geometry_warnings,
    }

    slice_columns = list(slice_rows[0].keys())
    case_columns = list(case_rows[0].keys())
    write_csv(args.output_dir / "slice_metrics.csv", slice_rows, slice_columns)
    write_csv(args.output_dir / "case_metrics.csv", case_rows, case_columns)
    with (args.output_dir / "overall_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(json_clean(overall), f, ensure_ascii=False, indent=2)

    print("\n================ 统计结果 ================")
    print(f"病例数：{len(case_ids)}")
    print(f"全部切片数：{len(slice_rows)}")
    print(f"有效切片数（GT或Pred任一非空）：{len(all_union_dice)}")
    print(f"GT非空切片数：{len(all_gt_positive_dice)}")
    print(f"3D病例级平均Dice：{recomputed_case_mean:.6f}")
    print(f"3D病例级中位Dice：{median(case_dices):.6f}")
    print(f"3D全局体素Dice：{dice(total_tp, total_fp, total_fn):.6f}")
    print(f"2D有效切片直接平均Dice（推荐）：{mean(all_union_dice):.6f}")
    print(f"2D病例平衡平均Dice：{mean(case_balanced_union):.6f}")
    print(f"2D仅GT非空切片平均Dice：{mean(all_gt_positive_dice):.6f}")
    print(f"2D双空切片按1平均Dice（不推荐）：{mean(all_empty_as_one_dice):.6f}")

    if reference_mean is not None:
        print(f"与summary病例级均值绝对差：{abs(recomputed_case_mean-reference_mean):.12g}")
        print(f"单病例最大绝对差：{max_reference_diff:.12g}")

    print("\n输出文件：")
    print(args.output_dir / "slice_metrics.csv")
    print(args.output_dir / "case_metrics.csv")
    print(args.output_dir / "overall_metrics.json")


if __name__ == "__main__":
    main()
