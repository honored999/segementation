import json
import argparse
import os


def load_summary(summary_path):
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_case_id_from_path(path_str):
    # 兼容 Windows / Linux 路径
    path_str = path_str.replace("\\", "/")
    name = os.path.basename(path_str)
    # 例如 case018.nii.gz -> case018
    if name.endswith(".nii.gz"):
        return name[:-7]
    elif name.endswith(".nii"):
        return name[:-4]
    return os.path.splitext(name)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=str, required=True)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    data = load_summary(args.summary)
    metric_per_case = data["metric_per_case"]

    rows = []
    for item in metric_per_case:
        case_id = get_case_id_from_path(item["reference_file"])

        # 二分类时常见键可能是 "1" 或 "(1,)"
        metrics = item["metrics"]
        if "1" in metrics:
            m = metrics["1"]
        elif "(1,)" in metrics:
            m = metrics["(1,)"]
        else:
            # 找非背景类
            keys = [k for k in metrics.keys() if k not in ["0", "(0,)"]]
            if len(keys) != 1:
                raise RuntimeError(f"Cannot determine foreground key for {case_id}: {list(metrics.keys())}")
            m = metrics[keys[0]]

        rows.append({
            "case_id": case_id,
            "dice": float(m["Dice"]),
            "iou": float(m["IoU"]),
            "n_ref": int(m["n_ref"]),
            "n_pred": int(m["n_pred"]),
        })

    # 从低到高排序
    rows_sorted = sorted(rows, key=lambda x: x["dice"])

    worst = rows_sorted[:args.topk]
    best = rows_sorted[-args.topk:][::-1]

    # 找接近中位数的病例
    median_dice = sorted([r["dice"] for r in rows])[len(rows) // 2]
    typical = sorted(rows, key=lambda x: abs(x["dice"] - median_dice))[:args.topk]

    print("\n=== Worst cases ===")
    for r in worst:
        print(f'{r["case_id"]}: Dice={r["dice"]:.4f}, IoU={r["iou"]:.4f}, GT={r["n_ref"]}, Pred={r["n_pred"]}')

    print("\n=== Best cases ===")
    for r in best:
        print(f'{r["case_id"]}: Dice={r["dice"]:.4f}, IoU={r["iou"]:.4f}, GT={r["n_ref"]}, Pred={r["n_pred"]}')

    print("\n=== Typical cases (closest to median Dice) ===")
    print(f"Median Dice ≈ {median_dice:.4f}")
    for r in typical:
        print(f'{r["case_id"]}: Dice={r["dice"]:.4f}, IoU={r["iou"]:.4f}, GT={r["n_ref"]}, Pred={r["n_pred"]}')


if __name__ == "__main__":
    main()