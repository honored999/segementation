from __future__ import annotations

from pathlib import Path

from standalone_nnunet2d.tools.inspect_dataset import inspect_dataset


def test_inspect_dataset_reports_directory_status_without_loading_cases(tmp_path: Path) -> None:
    (tmp_path / "imagesTr").mkdir()
    (tmp_path / "labelsTr").mkdir()

    report = inspect_dataset(tmp_path)

    assert report["raw_root_exists"] is True
    assert report["imagesTr_exists"] is True
    assert report["labelsTr_exists"] is True
    assert "case_id" not in report
