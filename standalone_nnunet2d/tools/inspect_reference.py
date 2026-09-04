"""Validate the supplied nnU-Net reference JSON files without changing them."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ReferenceValidationError(ValueError):
    """Raised when a supplied reference artifact is absent or inconsistent."""


@dataclass(frozen=True)
class ReferenceInspection:
    dataset_name: str
    plans_name: str
    patch_size: tuple[int, int]
    batch_size: int
    spacing: tuple[float, float]
    normalization_schemes: tuple[str, ...]
    use_mask_for_norm: tuple[bool, ...]
    network_class_name: str
    n_stages: int
    features_per_stage: tuple[int, ...]
    kernel_sizes: tuple[tuple[int, int], ...]
    strides: tuple[tuple[int, int], ...]
    n_conv_per_stage: tuple[int, ...]
    n_conv_per_stage_decoder: tuple[int, ...]
    conv_bias: bool
    norm_op: str
    norm_op_kwargs: dict[str, Any]
    dropout_op: str | None
    nonlin: str
    nonlin_kwargs: dict[str, Any]
    batch_dice: bool
    fold_sizes: tuple[tuple[int, int], ...]
    validation_case_count: int
    validation_cases_appear_once: bool
    foreground_dice: float
    foreground_iou: float
    metric_per_case_count: int


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise ReferenceValidationError(f"required reference file is missing: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def inspect_reference(reference_dir: Path) -> ReferenceInspection:
    """Read plans, splits, summary, and optional dataset metadata; raise on mismatch."""
    plans = _load_json(reference_dir / "nnUNetPlans.json")
    splits = _load_json(reference_dir / "splits_final.json")
    summary = _load_json(reference_dir / "summary.json")
    dataset_path = reference_dir / "dataset.json"
    dataset = _load_json(dataset_path) if dataset_path.is_file() else None
    try:
        configuration = plans["configurations"]["2d"]
        architecture = configuration["architecture"]
        arch_kwargs = architecture["arch_kwargs"]
    except KeyError as error:
        raise ReferenceValidationError(f"nnUNetPlans.json is missing required 2D key: {error}") from error

    if architecture["network_class_name"] != "dynamic_network_architectures.architectures.unet.PlainConvUNet":
        raise ReferenceValidationError("2D plan is not a PlainConvUNet")
    if len(splits) != 5:
        raise ReferenceValidationError(f"expected 5 folds, found {len(splits)}")
    fold_sizes: list[tuple[int, int]] = []
    validation_cases: list[str] = []
    for fold_index, fold in enumerate(splits):
        train = fold.get("train")
        validation = fold.get("val")
        if not isinstance(train, list) or not isinstance(validation, list):
            raise ReferenceValidationError(f"fold {fold_index} must contain list-valued train and val keys")
        if len(train) != 76 or len(validation) != 19:
            raise ReferenceValidationError(f"fold {fold_index} expected 76 train / 19 val, found {len(train)} / {len(validation)}")
        if set(train) & set(validation):
            raise ReferenceValidationError(f"fold {fold_index} has overlapping train and val cases")
        fold_sizes.append((len(train), len(validation)))
        validation_cases.extend(validation)
    validation_counts = Counter(validation_cases)
    if len(validation_counts) != 95 or any(count != 1 for count in validation_counts.values()):
        raise ReferenceValidationError("five validation folds must cover each of 95 cases exactly once")

    metrics = summary.get("metric_per_case")
    foreground = summary.get("foreground_mean")
    if not isinstance(metrics, list) or len(metrics) != 95:
        raise ReferenceValidationError("summary metric_per_case must contain exactly 95 cases")
    if not isinstance(foreground, dict) or "Dice" not in foreground or "IoU" not in foreground:
        raise ReferenceValidationError("summary foreground_mean requires Dice and IoU")
    if dataset is not None and dataset.get("numTraining") != 95:
        raise ReferenceValidationError("dataset.json numTraining must equal the 95 summary cases")

    return ReferenceInspection(
        dataset_name=plans["dataset_name"], plans_name=plans["plans_name"],
        patch_size=tuple(configuration["patch_size"]), batch_size=configuration["batch_size"],
        spacing=tuple(configuration["spacing"]), normalization_schemes=tuple(configuration["normalization_schemes"]),
        use_mask_for_norm=tuple(configuration["use_mask_for_norm"]), network_class_name=architecture["network_class_name"],
        n_stages=arch_kwargs["n_stages"], features_per_stage=tuple(arch_kwargs["features_per_stage"]),
        kernel_sizes=tuple(tuple(value) for value in arch_kwargs["kernel_sizes"]), strides=tuple(tuple(value) for value in arch_kwargs["strides"]),
        n_conv_per_stage=tuple(arch_kwargs["n_conv_per_stage"]), n_conv_per_stage_decoder=tuple(arch_kwargs["n_conv_per_stage_decoder"]),
        conv_bias=arch_kwargs["conv_bias"], norm_op=arch_kwargs["norm_op"], norm_op_kwargs=arch_kwargs["norm_op_kwargs"],
        dropout_op=arch_kwargs["dropout_op"], nonlin=arch_kwargs["nonlin"], nonlin_kwargs=arch_kwargs["nonlin_kwargs"],
        batch_dice=configuration["batch_dice"], fold_sizes=tuple(fold_sizes), validation_case_count=len(validation_counts),
        validation_cases_appear_once=all(count == 1 for count in validation_counts.values()),
        foreground_dice=float(foreground["Dice"]), foreground_iou=float(foreground["IoU"]), metric_per_case_count=len(metrics),
    )


def main() -> None:
    report = inspect_reference(Path(__file__).resolve().parents[1] / "reference")
    for field, value in vars(report).items():
        print(f"{field}: {value}")


if __name__ == "__main__":
    main()
