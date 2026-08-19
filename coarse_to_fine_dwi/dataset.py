"""Build the prediction-guided Dataset504 coarse-to-fine training dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .nifti import NiftiVolume, assert_compatible, crop_xy
from .roi import compute_prediction_roi, validate_binary_prediction

EXPECTED_NUM_CASES = 95
EXPECTED_NUM_FOLDS = 5
DEFAULT_SPLITS_PATH = (
    Path(__file__).resolve().parents[1]
    / "standalone_nnunet2d"
    / "reference"
    / "splits_final.json"
)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"split file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"split file is not valid JSON: {path}") from error


def _fixed_splits() -> list[dict[str, list[str]]]:
    splits = _read_json(DEFAULT_SPLITS_PATH)
    if not isinstance(splits, list):
        raise ValueError("the bundled fixed splits must be a list")
    return splits


def _validate_fixed_splits(splits_path: Path) -> tuple[list[dict[str, list[str]]], tuple[str, ...]]:
    splits = _read_json(splits_path)
    expected = _fixed_splits()
    if splits != expected:
        raise ValueError("splits must exactly match the established fixed 5-fold Dataset501 split")
    if len(splits) != EXPECTED_NUM_FOLDS:
        raise ValueError(f"expected exactly {EXPECTED_NUM_FOLDS} folds")

    all_ids: set[str] = set()
    validation_counts: dict[str, int] = {}
    normalized: list[dict[str, list[str]]] = []
    for fold_index, fold in enumerate(splits):
        if not isinstance(fold, dict) or set(fold) != {"train", "val"}:
            raise ValueError(f"fold {fold_index} must contain only train and val lists")
        train = fold["train"]
        validation = fold["val"]
        if not isinstance(train, list) or not isinstance(validation, list):
            raise ValueError(f"fold {fold_index} train and val must be lists")
        if not all(isinstance(case_id, str) for case_id in [*train, *validation]):
            raise ValueError(f"fold {fold_index} contains a non-string case ID")
        if set(train) & set(validation):
            raise ValueError(f"fold {fold_index} train and val overlap")
        all_ids.update(train)
        all_ids.update(validation)
        for case_id in validation:
            validation_counts[case_id] = validation_counts.get(case_id, 0) + 1
        normalized.append({"train": list(train), "val": list(validation)})

    if len(all_ids) != EXPECTED_NUM_CASES or len(validation_counts) != EXPECTED_NUM_CASES:
        raise ValueError(f"fixed split must contain exactly {EXPECTED_NUM_CASES} unique case IDs")
    if any(count != 1 for count in validation_counts.values()):
        raise ValueError("each fixed case ID must occur in validation exactly once")
    if any(len(fold["val"]) != 19 or len(fold["train"]) != 76 for fold in normalized):
        raise ValueError("fixed folds must contain 76 train and 19 validation IDs")
    return normalized, tuple(sorted(all_ids))


def _nii_case_id(path: Path) -> str | None:
    if path.name.endswith(".nii.gz"):
        return path.name[:-7]
    if path.name.endswith(".nii"):
        return path.name[:-4]
    return None


def _validate_exact_files(directory: Path, expected_ids: tuple[str, ...], *, kind: str) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"required {kind} directory does not exist: {directory}")
    files = [path for path in directory.iterdir() if path.is_file() and _nii_case_id(path) is not None]
    found: dict[str, Path] = {}
    duplicates: set[str] = set()
    for path in files:
        case_id = _nii_case_id(path)
        assert case_id is not None
        if kind == "imagesTr":
            if not case_id.endswith("_0000"):
                duplicates.add(case_id)
                continue
            case_id = case_id[:-5]
        if case_id in found:
            duplicates.add(case_id)
        found[case_id] = path
    expected = set(expected_ids)
    if duplicates or set(found) != expected or len(found) != len(expected):
        missing = sorted(expected - set(found))
        extra = sorted(set(found) - expected | duplicates)
        raise ValueError(
            f"{kind} must contain exactly the fixed 95-case IDs; missing={missing}, extra={extra}"
        )
    return found


def _validate_oof_files(prediction_root: Path, expected_ids: tuple[str, ...]) -> dict[str, Path]:
    return _validate_exact_files(
        prediction_root,
        expected_ids,
        kind="exactly the fixed 95-case OOF IDs",
    )


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _ensure_output_isolated(raw_root: Path, prediction_root: Path, output_root: Path) -> None:
    if _paths_overlap(output_root, raw_root) or _paths_overlap(output_root, prediction_root):
        raise ValueError("Dataset504 output root must be isolated from raw and prediction roots")
    if output_root.exists():
        raise FileExistsError(f"Dataset504 output root already exists: {output_root}")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_dataset504(
    raw_dataset_root: str | Path,
    oof_prediction_root: str | Path,
    output_root: str | Path,
    *,
    splits_path: str | Path = DEFAULT_SPLITS_PATH,
    margin: int | tuple[int, int] = 0,
    min_width: int = 1,
    min_height: int = 1,
) -> Path:
    """Create Dataset504 from complete five-fold Stage-1 OOF predictions.

    The function requires the established 95-case split and exact OOF case-ID
    set. It reads each DWI and prediction before reading that case's GT; the
    GT is therefore never used to localize the ROI.
    """
    raw_root = Path(raw_dataset_root).resolve()
    prediction_root = Path(oof_prediction_root).resolve()
    destination = Path(output_root).resolve()
    _ensure_output_isolated(raw_root, prediction_root, destination)

    splits, case_ids = _validate_fixed_splits(Path(splits_path).resolve())
    image_paths = _validate_exact_files(raw_root / "imagesTr", case_ids, kind="imagesTr")
    label_paths = _validate_exact_files(raw_root / "labelsTr", case_ids, kind="labelsTr")
    prediction_paths = _validate_oof_files(prediction_root, case_ids)

    fold_by_case = {
        case_id: fold_index
        for fold_index, fold in enumerate(splits)
        for case_id in fold["val"]
    }
    destination_images = destination / "imagesTr"
    destination_labels = destination / "labelsTr"
    destination_images.mkdir(parents=True)
    destination_labels.mkdir()

    manifest_cases: dict[str, dict[str, Any]] = {}
    for case_id in case_ids:
        image = NiftiVolume.read(image_paths[case_id])
        prediction = NiftiVolume.read(prediction_paths[case_id])
        assert_compatible(image, prediction)
        roi = compute_prediction_roi(
            prediction,
            margin=margin,
            min_width=min_width,
            min_height=min_height,
        )

        # Keep GT access after prediction validation and ROI calculation.
        label = NiftiVolume.read(label_paths[case_id])
        assert_compatible(image, label)
        label = NiftiVolume(
            array=validate_binary_prediction(label.array),
            spacing_xyz=label.spacing_xyz,
            origin_xyz=label.origin_xyz,
            direction=label.direction,
        )

        cropped_image = crop_xy(image, roi.bbox)
        cropped_label = crop_xy(label, roi.bbox)
        cropped_label = NiftiVolume(
            array=cropped_label.array,
            spacing_xyz=cropped_image.spacing_xyz,
            origin_xyz=cropped_image.origin_xyz,
            direction=cropped_image.direction,
        )
        cropped_image.write(destination_images / f"{case_id}_0000.nii.gz")
        cropped_label.write(destination_labels / f"{case_id}.nii.gz")
        manifest_cases[case_id] = {
            "fold": fold_by_case[case_id],
            "split": "val",
            "roi": list(roi.bbox),
            "fallback": roi.fallback,
            "source_image": f"imagesTr/{case_id}_0000.nii.gz",
            "source_label": f"labelsTr/{case_id}.nii.gz",
            "stage1_prediction": prediction_paths[case_id].name,
        }

    _write_json(
        destination / "dataset.json",
        {
            "channel_names": {"0": "DWI"},
            "labels": {"background": 0, "lesion": 1},
            "numTraining": EXPECTED_NUM_CASES,
            "file_ending": ".nii.gz",
            "overwrite_image_reader_writer": "SimpleITKIO",
        },
    )
    _write_json(destination / "splits_final.json", splits)
    _write_json(
        destination / "manifest.json",
        {
            "dataset_name": "Dataset504_StrokeLesion_CoarseToFine",
            "source_dataset": "Dataset501_StrokeLesion",
            "stage1_prediction_source": "complete_5_fold_oof",
            "roi_source": "stage1_prediction_only",
            "num_cases": EXPECTED_NUM_CASES,
            "num_folds": EXPECTED_NUM_FOLDS,
            "case_ids": list(case_ids),
            "cases": manifest_cases,
        },
    )
    return destination
