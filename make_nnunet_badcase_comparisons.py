import os
import argparse
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


def read_nifti(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # [z, y, x]
    return arr, img


def normalize_slice(img2d):
    img2d = img2d.astype(np.float32)
    p1 = np.percentile(img2d, 1)
    p99 = np.percentile(img2d, 99)
    img2d = np.clip(img2d, p1, p99)
    if p99 > p1:
        img2d = (img2d - p1) / (p99 - p1)
    else:
        img2d = np.zeros_like(img2d)
    return img2d


def choose_slices(gt, pred, max_slices=6):
    """
    选择最值得展示的切片：
    优先选择 GT 或 Pred 非零最多的切片
    """
    z_scores = []
    for z in range(gt.shape[0]):
        score = int(np.count_nonzero(gt[z])) + int(np.count_nonzero(pred[z]))
        z_scores.append((z, score))

    z_scores = [x for x in z_scores if x[1] > 0]

    if len(z_scores) == 0:
        # 没有病灶就均匀挑几张
        z_list = np.linspace(0, gt.shape[0] - 1, min(max_slices, gt.shape[0]), dtype=int).tolist()
        return z_list

    z_scores.sort(key=lambda x: x[1], reverse=True)
    top_z = [z for z, _ in z_scores[:max_slices]]
    top_z.sort()
    return top_z


def overlay_mask(ax, base_img, mask, color='lime', alpha=0.35, title=None):
    ax.imshow(base_img, cmap='gray')
    masked = np.ma.masked_where(mask == 0, mask)
    ax.imshow(masked, cmap=plt.cm.colors.ListedColormap([color]), alpha=alpha)
    if title is not None:
        ax.set_title(title, fontsize=10)
    ax.axis('off')


def error_map(gt, pred):
    """
    返回误差图：
    0: 背景
    1: TP
    2: FN
    3: FP
    """
    out = np.zeros_like(gt, dtype=np.uint8)
    out[(gt == 1) & (pred == 1)] = 1   # TP
    out[(gt == 1) & (pred == 0)] = 2   # FN
    out[(gt == 0) & (pred == 1)] = 3   # FP
    return out


def show_error(ax, base_img, gt, pred, title=None):
    ax.imshow(base_img, cmap='gray')

    emap = error_map(gt, pred)

    tp = np.ma.masked_where(emap != 1, emap)
    fn = np.ma.masked_where(emap != 2, emap)
    fp = np.ma.masked_where(emap != 3, emap)

    ax.imshow(tp, cmap=plt.cm.colors.ListedColormap(['lime']), alpha=0.35)
    ax.imshow(fn, cmap=plt.cm.colors.ListedColormap(['red']), alpha=0.45)
    ax.imshow(fp, cmap=plt.cm.colors.ListedColormap(['yellow']), alpha=0.45)

    if title is not None:
        ax.set_title(title, fontsize=10)
    ax.axis('off')


def dice_score(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    inter = np.logical_and(gt, pred).sum()
    denom = gt.sum() + pred.sum()
    if denom == 0:
        return 1.0
    return 2.0 * inter / denom


def make_case_figure(case_id, image_path, gt_path, pred_path, output_path, max_slices=6):
    image, _ = read_nifti(image_path)
    gt, _ = read_nifti(gt_path)
    pred, _ = read_nifti(pred_path)

    if image.shape != gt.shape or gt.shape != pred.shape:
        raise ValueError(
            f"{case_id} shape mismatch: image={image.shape}, gt={gt.shape}, pred={pred.shape}"
        )

    z_list = choose_slices(gt, pred, max_slices=max_slices)

    ncols = len(z_list)
    fig, axes = plt.subplots(4, ncols, figsize=(3.2 * ncols, 10))

    if ncols == 1:
        axes = np.expand_dims(axes, axis=1)

    case_dice = dice_score(gt > 0, pred > 0)
    gt_vox = int(np.count_nonzero(gt))
    pred_vox = int(np.count_nonzero(pred))

    fig.suptitle(
        f"{case_id} | Dice={case_dice:.4f} | GT voxels={gt_vox} | Pred voxels={pred_vox}",
        fontsize=14
    )

    for i, z in enumerate(z_list):
        img2d = normalize_slice(image[z])
        gt2d = (gt[z] > 0).astype(np.uint8)
        pred2d = (pred[z] > 0).astype(np.uint8)

        axes[0, i].imshow(img2d, cmap='gray')
        axes[0, i].set_title(f"Slice z={z}", fontsize=10)
        axes[0, i].axis('off')

        overlay_mask(axes[1, i], img2d, gt2d, color='cyan', alpha=0.40, title="GT")
        overlay_mask(axes[2, i], img2d, pred2d, color='magenta', alpha=0.40, title="Pred")
        show_error(axes[3, i], img2d, gt2d, pred2d, title="TP/FN/FP")

    axes[0, 0].set_ylabel("DWI", fontsize=12)
    axes[1, 0].set_ylabel("GT", fontsize=12)
    axes[2, 0].set_ylabel("Pred", fontsize=12)
    axes[3, 0].set_ylabel("Error", fontsize=12)

    legend_text = "Error map: green=TP, red=FN, yellow=FP"
    fig.text(0.5, 0.02, legend_text, ha='center', fontsize=11)

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Path to imagesTr"
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        required=True,
        help="Path to labelsTr"
    )
    parser.add_argument(
        "--pred-dir",
        type=str,
        required=True,
        help="Path to crossval_results_folds_0_1_2_3_4"
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        required=True,
        help="case ids, e.g. case018 case075"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="bad_case_comparisons"
    )
    parser.add_argument(
        "--max-slices",
        type=int,
        default=6
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for case_id in args.cases:
        image_path = os.path.join(args.images_dir, f"{case_id}_0000.nii.gz")
        gt_path = os.path.join(args.labels_dir, f"{case_id}.nii.gz")
        pred_path = os.path.join(args.pred_dir, f"{case_id}.nii.gz")
        output_path = os.path.join(args.output_dir, f"{case_id}_comparison.png")

        print(f"Processing {case_id}...")
        make_case_figure(
            case_id=case_id,
            image_path=image_path,
            gt_path=gt_path,
            pred_path=pred_path,
            output_path=output_path,
            max_slices=args.max_slices
        )
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()