from __future__ import annotations

from pathlib import Path

from standalone_nnunet2d.tools.inspect_reference import inspect_reference


REFERENCE_DIR = Path(__file__).resolve().parents[1] / "reference"


def test_reference_files_match_the_expected_2d_plan_and_baseline() -> None:
    report = inspect_reference(REFERENCE_DIR)

    assert report.dataset_name == "Dataset501_StrokeLesion"
    assert report.plans_name == "nnUNetPlans"
    assert report.patch_size == (512, 512)
    assert report.batch_size == 12
    assert report.n_stages == 8
    assert report.features_per_stage == (32, 64, 128, 256, 512, 512, 512, 512)
    assert report.foreground_dice == 0.731103738314918
    assert report.foreground_iou == 0.5923877518050135
    assert report.metric_per_case_count == 95
