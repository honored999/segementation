from __future__ import annotations

from pathlib import Path

from standalone_nnunet2d.tools.inspect_reference import inspect_reference


REFERENCE_DIR = Path(__file__).resolve().parents[1] / "reference"


def test_five_fold_split_has_one_validation_occurrence_per_case() -> None:
    report = inspect_reference(REFERENCE_DIR)

    assert report.fold_sizes == ((76, 19),) * 5
    assert report.validation_case_count == 95
    assert report.validation_cases_appear_once is True
