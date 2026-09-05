"""Create a MONAI datalist JSON from converted CI-1 NVAUTO cases."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from ci1_nvauto_common import read_csv, safe_json_dump, write_csv


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MONAI datalist for CI-1 DWI+ADC dataset.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        row
        for row in read_csv(args.dataset_root / "conversion_report.csv")
        if row.get("status") == "ok"
    ]
    patients = sorted({row["patient"] for row in rows})
    rng = random.Random(args.seed)
    rng.shuffle(patients)
    val_count = 0
    if len(patients) >= 2 and args.val_ratio > 0:
        val_count = max(1, int(round(len(patients) * args.val_ratio)))
        val_count = min(val_count, len(patients) - 1)
    val_patients = set(patients[:val_count])

    training = []
    validation = []
    summary_rows = []
    for row in sorted(rows, key=lambda item: item["case_id"]):
        case_dir = args.dataset_root / "cases" / row["case_id"]
        item = {
            "image": [rel(case_dir / "dwi.nii.gz", args.dataset_root), rel(case_dir / "adc.nii.gz", args.dataset_root)],
            "label": rel(case_dir / "label.nii.gz", args.dataset_root),
        }
        split = "val" if row["patient"] in val_patients else "train"
        if split == "train":
            item["fold"] = 0
            training.append(item)
        else:
            validation.append(item)
        summary_rows.append(
            {
                "case_id": row["case_id"],
                "patient": row["patient"],
                "timepoint": row["timepoint"],
                "split": split,
            }
        )

    datalist = {
        "name": "CI1_DWI_ADC_Stroke",
        "description": "CI-1 DWI+ADC ischemic lesion segmentation",
        "modality": {"0": "DWI", "1": "ADC"},
        "labels": {"0": "background", "1": "lesion"},
        "training": training,
        "validation": validation,
    }
    safe_json_dump(datalist, args.dataset_root / "datalist_ci1_dwi_adc.json")
    write_csv(args.dataset_root / "split_summary.csv", summary_rows, ["case_id", "patient", "timepoint", "split"])
    print(f"Training cases: {len(training)}")
    print(f"Validation cases: {len(validation)}")


if __name__ == "__main__":
    main()
