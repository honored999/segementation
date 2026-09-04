from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib

# 服务器没有图形界面时也能正常保存图片
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


def read_nifti(path: Path) -> tuple[np.ndarray, sitk.Image]:
    """读取NIfTI，数组顺序为[z, y, x]。"""
    if not path.is_file():
        raise FileNotFoundError(f"NIfTI文件不存在：{path}")

    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    return array, image


def normalize_slice(image_2d: np.ndarray) -> np.ndarray:
    """使用1%和99%分位数对单张DWI切片进行显示归一化。"""
    image_2d = image_2d.astype(np.float32)

    finite_mask = np.isfinite(image_2d)
    if not finite_mask.any():
        return np.zeros_like(image_2d, dtype=np.float32)

    finite_values = image_2d[finite_mask]
    low = float(np.percentile(finite_values, 1))
    high = float(np.percentile(finite_values, 99))

    if high <= low:
        return np.zeros_like(image_2d, dtype=np.float32)

    normalized = np.clip(image_2d, low, high)
    normalized = (normalized - low) / (high - low)
    normalized[~finite_mask] = 0
    return normalized


def dice_score(
    gt: np.ndarray,
    pred: np.ndarray,
    empty_value: Optional[float],
) -> Optional[float]:
    """
    计算二值Dice。

    empty_value:
    - 1.0：GT和Pred都为空时返回1
    - None：GT和Pred都为空时返回None
    """
    gt_bool = gt.astype(bool)
    pred_bool = pred.astype(bool)

    intersection = np.logical_and(gt_bool, pred_bool).sum(dtype=np.int64)
    denominator = (
        gt_bool.sum(dtype=np.int64)
        + pred_bool.sum(dtype=np.int64)
    )

    if denominator == 0:
        return empty_value

    return float(2.0 * intersection / denominator)


def calculate_confusion(
    gt: np.ndarray,
    pred: np.ndarray,
) -> tuple[int, int, int, int]:
    gt_bool = gt.astype(bool)
    pred_bool = pred.astype(bool)

    tp = int(np.logical_and(gt_bool, pred_bool).sum())
    fn = int(np.logical_and(gt_bool, ~pred_bool).sum())
    fp = int(np.logical_and(~gt_bool, pred_bool).sum())
    tn = int(np.logical_and(~gt_bool, ~pred_bool).sum())

    return tp, fn, fp, tn


def choose_slices(
    gt: np.ndarray,
    pred: np.ndarray,
    max_slices: int,
) -> list[int]:
    """
    优先选择GT或Pred病灶最多的切片。

    如果有效病灶切片不足max_slices，
    则从病灶中心附近补充相邻切片，
    尽量保证每个病例显示固定数量的切片。
    """
    if max_slices <= 0:
        raise ValueError("max_slices必须大于0")

    number_of_slices = gt.shape[0]
    max_slices = min(max_slices, number_of_slices)

    scores: list[tuple[int, int]] = []

    for z_index in range(number_of_slices):
        gt_count = int(np.count_nonzero(gt[z_index]))
        pred_count = int(np.count_nonzero(pred[z_index]))
        score = gt_count + pred_count

        if score > 0:
            scores.append((z_index, score))

    # 整个病例GT和Pred都没有病灶时，均匀选择切片
    if not scores:
        return np.linspace(
            0,
            number_of_slices - 1,
            max_slices,
            dtype=int,
        ).tolist()

    # 先选择病灶体素最多的切片
    scores.sort(key=lambda item: item[1], reverse=True)
    selected = [z for z, _ in scores[:max_slices]]

    # 以最显著病灶切片为中心补充邻近切片
    center_z = scores[0][0]

    distance_order = sorted(
        range(number_of_slices),
        key=lambda z: (abs(z - center_z), z),
    )

    for z_index in distance_order:
        if len(selected) >= max_slices:
            break

        if z_index not in selected:
            selected.append(z_index)

    selected.sort()
    return selected

def create_overlay(
    mask: np.ndarray,
    rgb_color: tuple[float, float, float],
    alpha: float,
) -> np.ndarray:
    """创建RGBA覆盖层。"""
    mask_bool = mask.astype(bool)

    overlay = np.zeros(
        (mask.shape[0], mask.shape[1], 4),
        dtype=np.float32,
    )

    overlay[..., 0] = rgb_color[0]
    overlay[..., 1] = rgb_color[1]
    overlay[..., 2] = rgb_color[2]
    overlay[..., 3] = mask_bool.astype(np.float32) * alpha

    return overlay


def show_mask_overlay(
    axis: plt.Axes,
    image_2d: np.ndarray,
    mask_2d: np.ndarray,
    rgb_color: tuple[float, float, float],
    title: str,
    alpha: float = 0.42,
) -> None:
    axis.imshow(image_2d, cmap="gray", vmin=0, vmax=1)
    axis.imshow(create_overlay(mask_2d, rgb_color, alpha))
    axis.set_title(title, fontsize=10)
    axis.axis("off")


def show_error_overlay(
    axis: plt.Axes,
    image_2d: np.ndarray,
    gt_2d: np.ndarray,
    pred_2d: np.ndarray,
) -> None:
    """
    误差颜色：
    绿色：TP
    红色：FN
    黄色：FP
    """
    gt_bool = gt_2d.astype(bool)
    pred_bool = pred_2d.astype(bool)

    tp = np.logical_and(gt_bool, pred_bool)
    fn = np.logical_and(gt_bool, ~pred_bool)
    fp = np.logical_and(~gt_bool, pred_bool)

    axis.imshow(image_2d, cmap="gray", vmin=0, vmax=1)
    axis.imshow(create_overlay(tp, (0.0, 1.0, 0.0), 0.42))
    axis.imshow(create_overlay(fn, (1.0, 0.0, 0.0), 0.55))
    axis.imshow(create_overlay(fp, (1.0, 1.0, 0.0), 0.55))

    axis.set_title("TP / FN / FP", fontsize=10)
    axis.axis("off")


def validate_geometry(
    case_id: str,
    image_array: np.ndarray,
    gt_array: np.ndarray,
    pred_array: np.ndarray,
    image_itk: sitk.Image,
    gt_itk: sitk.Image,
    pred_itk: sitk.Image,
) -> None:
    """检查数组shape和基础空间信息。"""
    if image_array.shape != gt_array.shape:
        raise ValueError(
            f"{case_id}原图与GT的shape不一致："
            f"image={image_array.shape}, gt={gt_array.shape}"
        )

    if gt_array.shape != pred_array.shape:
        raise ValueError(
            f"{case_id} GT与Pred的shape不一致："
            f"gt={gt_array.shape}, pred={pred_array.shape}"
        )

    image_size = image_itk.GetSize()
    gt_size = gt_itk.GetSize()
    pred_size = pred_itk.GetSize()

    if image_size != gt_size or gt_size != pred_size:
        raise ValueError(
            f"{case_id} ITK size不一致："
            f"image={image_size}, gt={gt_size}, pred={pred_size}"
        )


def add_row_label(axis: plt.Axes, text: str) -> None:
    axis.text(
        -0.16,
        0.5,
        text,
        transform=axis.transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )


def make_case_figure(
    case_id: str,
    image_path: Path,
    gt_path: Path,
    pred_path: Path,
    output_path: Path,
    max_slices: int,
    dpi: int,
) -> None:
    image, image_itk = read_nifti(image_path)
    gt, gt_itk = read_nifti(gt_path)
    pred, pred_itk = read_nifti(pred_path)

    validate_geometry(
        case_id=case_id,
        image_array=image,
        gt_array=gt,
        pred_array=pred,
        image_itk=image_itk,
        gt_itk=gt_itk,
        pred_itk=pred_itk,
    )

    gt = gt > 0
    pred = pred > 0

    selected_slices = choose_slices(
        gt=gt,
        pred=pred,
        max_slices=max_slices,
    )

    case_dice = dice_score(
        gt=gt,
        pred=pred,
        empty_value=1.0,
    )

    assert case_dice is not None

    tp, fn, fp, _ = calculate_confusion(gt, pred)

    gt_voxels = int(gt.sum())
    pred_voxels = int(pred.sum())

    number_of_columns = len(selected_slices)

    figure, axes = plt.subplots(
        nrows=4,
        ncols=number_of_columns,
        figsize=(3.2 * number_of_columns, 10.5),
        squeeze=False,
    )

    figure.suptitle(
        (
            f"{case_id} | 3D Case Dice={case_dice:.4f} | "
            f"GT voxels={gt_voxels} | Pred voxels={pred_voxels}"
        ),
        fontsize=15,
        y=0.985,
    )

    slice_dice_results: list[tuple[int, Optional[float]]] = []

    for column_index, z_index in enumerate(selected_slices):
        image_2d = normalize_slice(image[z_index])
        gt_2d = gt[z_index]
        pred_2d = pred[z_index]

        slice_dice = dice_score(
            gt=gt_2d,
            pred=pred_2d,
            empty_value=None,
        )
        slice_dice_results.append((z_index, slice_dice))

        if slice_dice is None:
            slice_dice_text = "N/A"
        else:
            slice_dice_text = f"{slice_dice:.4f}"

        axes[0, column_index].imshow(
            image_2d,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[0, column_index].set_title(
            f"Slice z={z_index}\n2D Slice Dice={slice_dice_text}",
            fontsize=10,
        )
        axes[0, column_index].axis("off")

        show_mask_overlay(
            axis=axes[1, column_index],
            image_2d=image_2d,
            mask_2d=gt_2d,
            rgb_color=(0.0, 1.0, 1.0),
            title="GT",
        )

        show_mask_overlay(
            axis=axes[2, column_index],
            image_2d=image_2d,
            mask_2d=pred_2d,
            rgb_color=(1.0, 0.0, 1.0),
            title="Prediction",
        )

        show_error_overlay(
            axis=axes[3, column_index],
            image_2d=image_2d,
            gt_2d=gt_2d,
            pred_2d=pred_2d,
        )

    add_row_label(axes[0, 0], "DWI")
    add_row_label(axes[1, 0], "GT")
    add_row_label(axes[2, 0], "Pred")
    add_row_label(axes[3, 0], "Error")

    figure.text(
        0.5,
        0.025,
        (
            "Error map: green=TP, red=FN, yellow=FP | "
            f"3D TP={tp}, FN={fn}, FP={fp}"
        ),
        ha="center",
        fontsize=11,
    )

    figure.text(
        0.5,
        0.008,
        (
            "Top title reports full-volume 3D case Dice; "
            "each column reports the corresponding 2D slice Dice."
        ),
        ha="center",
        fontsize=9,
    )

    plt.tight_layout(rect=(0.025, 0.055, 1.0, 0.95))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"\n{case_id}")
    print(f"  3D Case Dice: {case_dice:.6f}")
    print(f"  GT voxels:     {gt_voxels}")
    print(f"  Pred voxels:   {pred_voxels}")
    print(f"  TP/FN/FP:      {tp}/{fn}/{fp}")
    print("  Selected slices:")

    for z_index, slice_dice in slice_dice_results:
        if slice_dice is None:
            value = "N/A"
        else:
            value = f"{slice_dice:.6f}"
        print(f"    z={z_index}: 2D Dice={value}")

    print(f"  Saved: {output_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate DWI/GT/prediction/error comparison figures "
            "with both 3D case Dice and 2D slice Dice."
        )
    )

    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="nnU-Net imagesTr目录",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        required=True,
        help="nnU-Net labelsTr目录",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        required=True,
        help="五折out-of-fold预测结果目录",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        required=True,
        help="病例ID，例如case018 case075 case082",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nnunet_case_comparisons"),
        help="输出目录",
    )
    parser.add_argument(
        "--max-slices",
        type=int,
        default=6,
        help="每个病例最多展示的切片数",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="输出图片DPI",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    failed_cases: list[str] = []

    for case_id in args.cases:
        image_path = args.images_dir / f"{case_id}_0000.nii.gz"
        gt_path = args.labels_dir / f"{case_id}.nii.gz"
        pred_path = args.pred_dir / f"{case_id}.nii.gz"
        output_path = args.output_dir / f"{case_id}_comparison.png"

        print(f"\nProcessing {case_id}...")

        try:
            make_case_figure(
                case_id=case_id,
                image_path=image_path,
                gt_path=gt_path,
                pred_path=pred_path,
                output_path=output_path,
                max_slices=args.max_slices,
                dpi=args.dpi,
            )
        except Exception as error:
            failed_cases.append(case_id)
            print(f"  Failed: {error}")

    print("\nFinished.")

    if failed_cases:
        print("Failed cases:")
        for case_id in failed_cases:
            print(f"  {case_id}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()