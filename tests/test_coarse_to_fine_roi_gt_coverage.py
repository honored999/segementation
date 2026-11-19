import csv
import json
from pathlib import Path

import numpy as np
import pytest

from coarse_to_fine_dwi.nifti import NiftiVolume


def _volume(array: np.ndarray) -> NiftiVolume:
    return NiftiVolume(
        array=array,
        spacing_xyz=(1.0, 1.0, 1.0),
        origin_xyz=(0.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )


def _inputs(
    tmp_path: Path,
    *,
    empty_gt: bool = False,
    invalid_bbox: bool = False,
    invalid_label: bool = False,
    include_bbox_xy: bool = True,
    mismatched_bbox_xy: bool = False,
):
    pytest.importorskip("SimpleITK")
    manifest_dir = tmp_path / "manifest"
    splits_dir = tmp_path / "splits"
    labels_dir = tmp_path / "labels"
    manifest_dir.mkdir()
    splits_dir.mkdir()
    labels_dir.mkdir()

    shape = (3, 10, 12)
    label = np.zeros(shape, dtype=np.uint8)
    if not empty_gt:
        # These two voxels are one lesion under explicit 26-connectivity.
        label[0, 3, 3] = 1
        label[1, 4, 4] = 1
        # This diagonal pair is a wholly missed lesion outside the final ROI.
        label[0, 8, 10] = 1
        label[1, 9, 11] = 1
    if invalid_label:
        label[0, 0, 0] = 2
    _volume(label).write(labels_dir / "caseA.nii.gz")

    final_bbox = [0, 0, 12, 10] if empty_gt else [2, 2, 6, 6]
    if invalid_bbox:
        final_bbox = [0, 0, 13, 5]
    manifest_case = {
        "case_id": "caseA",
        "fold": 0,
        "roi": final_bbox,
        "fallback": empty_gt,
        "raw_prediction_bbox": None if empty_gt else [3, 3, 4, 4],
        "roi_margin": 1,
        "min_roi_width": 4,
        "min_roi_height": 4,
        "roi_width": final_bbox[2] - final_bbox[0],
        "roi_height": final_bbox[3] - final_bbox[1],
    }
    if include_bbox_xy:
        manifest_case["bbox_xy"] = [0, 0, 12, 10] if mismatched_bbox_xy else final_bbox
    manifest = {"cases": {"caseA": manifest_case}}
    manifest_path = manifest_dir / "roi_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    splits_path = splits_dir / "splits_final.json"
    splits_path.write_text(json.dumps({"synthetic": True}), encoding="utf-8")
    return manifest_path, labels_dir, splits_path


def _patch_fixed_split(monkeypatch, audit_module):
    monkeypatch.setattr(
        audit_module,
        "_validate_fixed_splits",
        lambda path: ([{"train": [], "val": ["caseA"]}], ("caseA",)),
    )


def test_final_roi_coverage_and_wholly_missed_26_connected_lesion(tmp_path, monkeypatch):
    from coarse_to_fine_dwi.cli import audit_roi_gt_coverage as audit

    manifest, labels, splits = _inputs(tmp_path)
    _patch_fixed_split(monkeypatch, audit)
    output_dir = tmp_path / "output"

    csv_path, summary_path = audit.audit_roi_gt_coverage(
        manifest=manifest,
        labels_dir=labels,
        splits=splits,
        fold=0,
        output_dir=output_dir,
    )

    assert csv_path.name == "roi_gt_coverage_fold0.csv"
    assert summary_path.name == "roi_gt_coverage_fold0_summary.json"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert int(row["gt_total_foreground_voxels"]) == 4
    assert int(row["gt_roi_foreground_voxels"]) == 2
    assert int(row["gt_outside_roi_foreground_voxels"]) == 2
    assert float(row["coverage"]) == pytest.approx(0.5)
    assert int(row["gt_lesion_components"]) == 2
    assert row["has_complete_missed_lesion"] == "True"
    assert int(row["complete_missed_lesion_count"]) == 1
    assert int(row["complete_missed_lesion_voxels"]) == 2

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["case_count"] == 1
    assert summary["case_ids"] == ["caseA"]
    assert summary["mean_coverage"] == pytest.approx(0.5)
    assert summary["median_coverage"] == pytest.approx(0.5)
    assert summary["min_coverage"] == pytest.approx(0.5)
    assert summary["coverage_100_percent_case_count"] == 0
    assert summary["coverage_below_99_percent_case_count"] == 1
    assert summary["coverage_below_95_percent_case_count"] == 1
    assert summary["complete_miss_case_count"] == 1
    assert summary["total_missed_lesions"] == 1
    assert summary["total_missed_lesion_voxels"] == 2
    assert summary["worst_cases"] == [{"case_id": "caseA", "coverage": 0.5}]


def test_empty_gt_has_full_coverage_and_exact_outputs(tmp_path, monkeypatch):
    from coarse_to_fine_dwi.cli import audit_roi_gt_coverage as audit

    manifest, labels, splits = _inputs(tmp_path, empty_gt=True)
    _patch_fixed_split(monkeypatch, audit)

    csv_path, summary_path = audit.audit_roi_gt_coverage(
        manifest=manifest,
        labels_dir=labels,
        splits=splits,
        fold=0,
        output_dir=tmp_path / "output",
    )

    assert {csv_path.name, summary_path.name} == {
        "roi_gt_coverage_fold0.csv",
        "roi_gt_coverage_fold0_summary.json",
    }
    with csv_path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert float(row["coverage"]) == 1.0
    assert int(row["gt_lesion_components"]) == 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["coverage_100_percent_case_count"] == 1
    assert summary["complete_miss_case_count"] == 0
    assert summary["protocol"]["connectivity"] == 26
    assert summary["empty_gt_case_count"] == 1
    assert summary["protocol"]["empty_gt_coverage"] == 1.0


def test_non_binary_gt_is_rejected_before_output_writing(tmp_path, monkeypatch):
    from coarse_to_fine_dwi.cli import audit_roi_gt_coverage as audit

    manifest, labels, splits = _inputs(tmp_path, invalid_label=True)
    _patch_fixed_split(monkeypatch, audit)
    output_dir = tmp_path / "output"

    with pytest.raises(ValueError, match="0 and 1|finite"):
        audit.audit_roi_gt_coverage(
            manifest=manifest,
            labels_dir=labels,
            splits=splits,
            fold=0,
            output_dir=output_dir,
        )
    assert not output_dir.exists()


def test_roi_only_manifest_is_accepted(tmp_path, monkeypatch):
    from coarse_to_fine_dwi.cli import audit_roi_gt_coverage as audit

    manifest, labels, splits = _inputs(tmp_path, include_bbox_xy=False)
    _patch_fixed_split(monkeypatch, audit)

    audit.audit_roi_gt_coverage(
        manifest=manifest,
        labels_dir=labels,
        splits=splits,
        fold=0,
        output_dir=tmp_path / "output",
    )


def test_mismatched_roi_and_bbox_xy_are_rejected(tmp_path, monkeypatch):
    from coarse_to_fine_dwi.cli import audit_roi_gt_coverage as audit

    manifest, labels, splits = _inputs(tmp_path, mismatched_bbox_xy=True)
    _patch_fixed_split(monkeypatch, audit)

    with pytest.raises(ValueError, match="do not match"):
        audit.audit_roi_gt_coverage(
            manifest=manifest,
            labels_dir=labels,
            splits=splits,
            fold=0,
            output_dir=tmp_path / "output",
        )


def test_invalid_bbox_is_rejected_before_output_writing(tmp_path, monkeypatch):
    from coarse_to_fine_dwi.cli import audit_roi_gt_coverage as audit

    manifest, labels, splits = _inputs(tmp_path, invalid_bbox=True)
    _patch_fixed_split(monkeypatch, audit)
    output_dir = tmp_path / "output"

    with pytest.raises(ValueError, match="bbox"):
        audit.audit_roi_gt_coverage(
            manifest=manifest,
            labels_dir=labels,
            splits=splits,
            fold=0,
            output_dir=output_dir,
        )
    assert not output_dir.exists()


def test_output_isolation_rejects_labels_and_input_parent_overlap(tmp_path, monkeypatch):
    from coarse_to_fine_dwi.cli import audit_roi_gt_coverage as audit

    manifest, labels, splits = _inputs(tmp_path)
    _patch_fixed_split(monkeypatch, audit)

    with pytest.raises(ValueError, match="labels"):
        audit.audit_roi_gt_coverage(
            manifest=manifest,
            labels_dir=labels,
            splits=splits,
            fold=0,
            output_dir=labels / "output",
        )
    with pytest.raises(ValueError, match="manifest|splits"):
        audit.audit_roi_gt_coverage(
            manifest=manifest,
            labels_dir=labels,
            splits=splits,
            fold=0,
            output_dir=manifest.parent / "output",
        )


def test_parser_has_required_parameters_and_default_fold():
    from coarse_to_fine_dwi.cli.audit_roi_gt_coverage import build_parser

    args = build_parser().parse_args(
        ["--manifest", "m.json", "--labels-dir", "labels", "--splits", "s.json", "--output-dir", "out"]
    )
    assert args.fold == 0
