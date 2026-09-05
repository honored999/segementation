"""Create nnU-Net validation prediction comparison grids.

中文说明：这个脚本把 nnU-Net 的 validation 预测结果画成类似 teacher
`predictions_best.png` 的对比图。单模型图包含 DWI 原图、GT mask、prediction；
2D vs 3D 图包含 DWI 原图、GT mask、2D prediction、3D prediction。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


DEFAULT_RESULTS_ROOT = (
    Path("E:/学习/研一/work14-图像分割/nnUNet_results")
    / "Dataset501_StrokeLesion"
)
DEFAULT_OUTPUT_DIR = Path("results") / "nnunet_prediction_comparisons"
CONFIG_FOLDERS = {
    "2d": "nnUNetTrainer__nnUNetPlans__2d",
    "3d": "nnUNetTrainer__nnUNetPlans__3d_fullres",
}


def read_image(path: Path) -> sitk.Image:
    """Read a NIfTI image."""
    return sitk.ReadImage(str(path))


def image_to_array(path: Path) -> np.ndarray:
    """Read a NIfTI file into a z/y/x numpy array."""
    return sitk.GetArrayFromImage(read_image(path))


def normalize_slice(image: np.ndarray) -> np.ndarray:
    """Normalize a 2D slice to 0..1 for display."""
    image = image.astype(np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    low, high = np.percentile(finite, [1.0, 99.0])
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def case_id_from_prediction(path: Path) -> str:
    """Extract case id from case001.nii.gz."""
    name = path.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return path.stem


def prediction_paths(results_root: Path, config_key: str) -> list[Path]:
    """List validation predictions for one nnU-Net configuration."""
    validation_dir = results_root / CONFIG_FOLDERS[config_key] / "fold_0" / "validation"
    return sorted(
        path
        for path in validation_dir.glob("case*.nii.gz")
        if path.name != "summary.json"
    )


def raw_image_path(raw_dataset_root: Path, case_id: str) -> Path:
    """Return imagesTr path for a case."""
    return raw_dataset_root / "imagesTr" / f"{case_id}_0000.nii.gz"


def raw_label_path(raw_dataset_root: Path, case_id: str) -> Path:
    """Return labelsTr path for a case."""
    return raw_dataset_root / "labelsTr" / f"{case_id}.nii.gz"


def find_raw_dataset_root(user_path: Path | None) -> Path:
    """Find Dataset501_StrokeLesion raw root from common locations."""
    if user_path is not None:
        return user_path
    candidates = [
        Path("E:/学习/研一/work14-图像分割/nnUNet_raw/Dataset501_StrokeLesion"),
        Path("C:/lijialin/models3d/nnUNet/nnUNet_raw/Dataset501_StrokeLesion"),
    ]
    for candidate in candidates:
        if (candidate / "imagesTr").is_dir() and (candidate / "labelsTr").is_dir():
            return candidate
    raise FileNotFoundError(
        "Cannot find nnU-Net raw Dataset501_StrokeLesion. "
        "Please pass --raw-dataset-root."
    )


def largest_positive_slice(mask: np.ndarray, prediction: np.ndarray | None = None) -> int:
    """Pick the z slice with the largest GT/predicted lesion area."""
    mask_bool = mask != 0
    if prediction is not None:
        mask_bool = mask_bool | (prediction != 0)
    areas = mask_bool.reshape(mask_bool.shape[0], -1).sum(axis=1)
    return int(np.argmax(areas))


def select_cases_for_grid(
    raw_dataset_root: Path,
    prediction_files: Sequence[Path],
    num_samples: int,
) -> list[tuple[str, int, int]]:
    """Select cases and slices with the largest GT lesion areas."""
    scored: list[tuple[int, str, int]] = []
    for prediction_file in prediction_files:
        case_id = case_id_from_prediction(prediction_file)
        label_path = raw_label_path(raw_dataset_root, case_id)
        if not label_path.exists():
            continue
        label = image_to_array(label_path)
        prediction = image_to_array(prediction_file)
        z_index = largest_positive_slice(label, prediction)
        score = int(((label != 0) | (prediction != 0))[z_index].sum())
        if score > 0:
            scored.append((score, case_id, z_index))
    scored.sort(reverse=True)
    return [(case_id, z_index, score) for score, case_id, z_index in scored[:num_samples]]


def draw_single_model_grid(
    raw_dataset_root: Path,
    prediction_files: Sequence[Path],
    output_path: Path,
    title: str,
    num_samples: int,
) -> None:
    """Draw DWI/GT/prediction rows for one nnU-Net model."""
    selected = select_cases_for_grid(raw_dataset_root, prediction_files, num_samples)
    if not selected:
        raise RuntimeError("No drawable validation cases found.")

    fig, axes = plt.subplots(len(selected), 3, figsize=(9, 3 * len(selected)))
    if len(selected) == 1:
        axes = axes.reshape(1, -1)

    prediction_by_case = {case_id_from_prediction(path): path for path in prediction_files}
    for row, (case_id, z_index, score) in enumerate(selected):
        image = image_to_array(raw_image_path(raw_dataset_root, case_id))
        label = image_to_array(raw_label_path(raw_dataset_root, case_id)) != 0
        prediction = image_to_array(prediction_by_case[case_id]) != 0
        z_index = min(z_index, image.shape[0] - 1, label.shape[0] - 1, prediction.shape[0] - 1)

        panels = [
            (normalize_slice(image[z_index]), "DWI"),
            (label[z_index], "GT Mask"),
            (prediction[z_index], "Prediction"),
        ]
        for col, (array, panel_title) in enumerate(panels):
            axes[row, col].imshow(array, cmap="gray")
            axes[row, col].set_title(f"{panel_title}\n{case_id} z={z_index} area={score}")
            axes[row, col].axis("off")

    fig.suptitle(title, fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_2d_3d_grid(
    raw_dataset_root: Path,
    predictions_2d: Sequence[Path],
    predictions_3d: Sequence[Path],
    output_path: Path,
    num_samples: int,
) -> None:
    """Draw DWI/GT/2D pred/3D pred rows for shared validation cases."""
    pred2 = {case_id_from_prediction(path): path for path in predictions_2d}
    pred3 = {case_id_from_prediction(path): path for path in predictions_3d}
    shared_files = [pred2[case_id] for case_id in sorted(set(pred2) & set(pred3))]
    selected = select_cases_for_grid(raw_dataset_root, shared_files, num_samples)
    if not selected:
        raise RuntimeError("No shared drawable validation cases found.")

    fig, axes = plt.subplots(len(selected), 4, figsize=(12, 3 * len(selected)))
    if len(selected) == 1:
        axes = axes.reshape(1, -1)

    for row, (case_id, z_index, score) in enumerate(selected):
        image = image_to_array(raw_image_path(raw_dataset_root, case_id))
        label = image_to_array(raw_label_path(raw_dataset_root, case_id)) != 0
        prediction_2d = image_to_array(pred2[case_id]) != 0
        prediction_3d = image_to_array(pred3[case_id]) != 0
        z_index = min(
            z_index,
            image.shape[0] - 1,
            label.shape[0] - 1,
            prediction_2d.shape[0] - 1,
            prediction_3d.shape[0] - 1,
        )
        panels = [
            (normalize_slice(image[z_index]), "DWI"),
            (label[z_index], "GT Mask"),
            (prediction_2d[z_index], "nnU-Net 2D"),
            (prediction_3d[z_index], "nnU-Net 3D"),
        ]
        for col, (array, panel_title) in enumerate(panels):
            axes[row, col].imshow(array, cmap="gray")
            axes[row, col].set_title(f"{panel_title}\n{case_id} z={z_index} area={score}")
            axes[row, col].axis("off")

    fig.suptitle("nnU-Net 2D vs 3D validation predictions", fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create comparison figures for nnU-Net validation predictions."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--raw-dataset-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-samples", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    raw_root = find_raw_dataset_root(args.raw_dataset_root)
    predictions_2d = prediction_paths(args.results_root, "2d")
    predictions_3d = prediction_paths(args.results_root, "3d")

    draw_single_model_grid(
        raw_dataset_root=raw_root,
        prediction_files=predictions_2d,
        output_path=args.output_dir / "nnunet_2d_predictions_best.png",
        title="nnU-Net 2D validation predictions",
        num_samples=args.num_samples,
    )
    draw_single_model_grid(
        raw_dataset_root=raw_root,
        prediction_files=predictions_3d,
        output_path=args.output_dir / "nnunet_3d_predictions_best.png",
        title="nnU-Net 3D fullres validation predictions",
        num_samples=args.num_samples,
    )
    draw_2d_3d_grid(
        raw_dataset_root=raw_root,
        predictions_2d=predictions_2d,
        predictions_3d=predictions_3d,
        output_path=args.output_dir / "nnunet_2d_vs_3d_predictions_best.png",
        num_samples=args.num_samples,
    )
    print(f"Wrote nnU-Net comparison figures to {args.output_dir}")


if __name__ == "__main__":
    main()
