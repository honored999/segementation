"""Compare CI-1 mask area distributions across axial/coronal/sagittal views.

中文说明：
统计并比较 CI-1 mask 在 axial、coronal、sagittal 三个视角下的面积分布。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


VIEW_AXES = {
    "axial": 0,
    "coronal": 1,
    "sagittal": 2,
}


@dataclass
class ViewStats:
    view: str
    total_slices: int
    positive_slices: int
    mean_positive_mask_area: float
    max_mask_area: int
    ge_1pct_slices: int
    ge_5pct_slices: int
    ge_10pct_slices: int


def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def group_rows(rows: Sequence[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row.get("patient", "unknown"), row.get("timepoint", ""))
        groups.setdefault(key, []).append(row)
    return groups


def load_mask_volume(rows: Sequence[dict[str, str]]) -> np.ndarray:
    sorted_rows = sorted(rows, key=lambda row: int(row["slice_index"]))
    masks = []
    for row in sorted_rows:
        mask = np.asarray(Image.open(row["mask_path"]).convert("L")) > 127
        masks.append(mask)
    if not masks:
        raise ValueError("Cannot build a volume from an empty row group.")
    return np.stack(masks, axis=0)


def view_slice_areas(mask_volume: np.ndarray, axis: int) -> tuple[np.ndarray, int]:
    axes_to_sum = tuple(index for index in range(mask_volume.ndim) if index != axis)
    areas = mask_volume.sum(axis=axes_to_sum).astype(np.int64)
    plane_area = int(np.prod([mask_volume.shape[index] for index in axes_to_sum]))
    return areas, plane_area


def summarize_areas(view: str, areas: np.ndarray, plane_area: int) -> ViewStats:
    positive_areas = areas[areas > 0]
    return ViewStats(
        view=view,
        total_slices=int(areas.size),
        positive_slices=int(positive_areas.size),
        mean_positive_mask_area=float(positive_areas.mean()) if positive_areas.size else 0.0,
        max_mask_area=int(positive_areas.max()) if positive_areas.size else 0,
        ge_1pct_slices=int(np.count_nonzero(areas >= 0.01 * plane_area)),
        ge_5pct_slices=int(np.count_nonzero(areas >= 0.05 * plane_area)),
        ge_10pct_slices=int(np.count_nonzero(areas >= 0.10 * plane_area)),
    )


def summarize_mask_volume(mask_volume: np.ndarray) -> dict[str, ViewStats]:
    stats = {}
    for view, axis in VIEW_AXES.items():
        areas, plane_area = view_slice_areas(mask_volume, axis)
        stats[view] = summarize_areas(view, areas, plane_area)
    return stats


def analyze_manifest(manifest_path: Path) -> list[dict[str, str]]:
    rows = read_manifest(manifest_path)
    report_rows: list[dict[str, str]] = []

    for (patient, timepoint), group in sorted(group_rows(rows).items()):
        mask_volume = load_mask_volume(group)
        stats_by_view = summarize_mask_volume(mask_volume)
        for view, stats in stats_by_view.items():
            report_rows.append(
                {
                    "patient": patient,
                    "timepoint": timepoint,
                    "view": view,
                    "volume_shape_zyx": "x".join(str(value) for value in mask_volume.shape),
                    "total_slices": str(stats.total_slices),
                    "positive_slices": str(stats.positive_slices),
                    "mean_positive_mask_area": f"{stats.mean_positive_mask_area:.6g}",
                    "max_mask_area": str(stats.max_mask_area),
                    "ge_1pct_slices": str(stats.ge_1pct_slices),
                    "ge_5pct_slices": str(stats.ge_5pct_slices),
                    "ge_10pct_slices": str(stats.ge_10pct_slices),
                }
            )

    return report_rows


def summarize_report(report_rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    summary_rows: list[dict[str, str]] = []
    for view in VIEW_AXES:
        rows = [row for row in report_rows if row["view"] == view]
        if not rows:
            continue
        summary_rows.append(
            {
                "view": view,
                "case_count": str(len(rows)),
                "total_slices": str(sum(int(row["total_slices"]) for row in rows)),
                "positive_slices": str(sum(int(row["positive_slices"]) for row in rows)),
                "ge_1pct_slices": str(sum(int(row["ge_1pct_slices"]) for row in rows)),
                "ge_5pct_slices": str(sum(int(row["ge_5pct_slices"]) for row in rows)),
                "ge_10pct_slices": str(sum(int(row["ge_10pct_slices"]) for row in rows)),
                "max_mask_area": str(max(int(row["max_mask_area"]) for row in rows)),
                "mean_case_positive_area": f"{np.mean([float(row['mean_positive_mask_area']) for row in rows]):.6g}",
            }
        )
    return summary_rows


def write_csv(rows: Sequence[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze CI-1 mask slice area statistics in axial/coronal/sagittal views."
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data") / "ci1_dwi_2d_dedup" / "manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "ci1_multiview_mask_stats",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_rows = analyze_manifest(args.manifest_path)
    summary_rows = summarize_report(report_rows)
    write_csv(report_rows, args.output_dir / "case_view_stats.csv")
    write_csv(summary_rows, args.output_dir / "summary.csv")

    print(f"Wrote case stats: {args.output_dir / 'case_view_stats.csv'}")
    print(f"Wrote summary:    {args.output_dir / 'summary.csv'}")
    for row in summary_rows:
        print(
            f"{row['view']}: positive={row['positive_slices']} "
            f"ge1={row['ge_1pct_slices']} ge5={row['ge_5pct_slices']} "
            f"ge10={row['ge_10pct_slices']} max={row['max_mask_area']}"
        )


if __name__ == "__main__":
    main()
