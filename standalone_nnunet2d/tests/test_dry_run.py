from __future__ import annotations

import json
from pathlib import Path

import pytest

from standalone_nnunet2d.dry_run import main


def _roots(tmp_path: Path) -> tuple[Path, Path, Path]:
    roots = tuple(tmp_path / name for name in ("raw", "preprocessed", "results"))
    for root in roots:
        root.mkdir()
    (roots[0] / "imagesTr").mkdir()
    (roots[0] / "labelsTr").mkdir()
    return roots  # type: ignore[return-value]


def test_dry_run_prints_json_and_returns_zero_when_ready(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_root, preprocessed_root, results_root = _roots(tmp_path)

    exit_code = main(
        [
            "--raw-root", str(raw_root),
            "--preprocessed-root", str(preprocessed_root),
            "--results-root", str(results_root),
            "--device", "cpu",
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["ready"] is True


def test_dry_run_rejects_run_flag(tmp_path: Path) -> None:
    raw_root, preprocessed_root, results_root = _roots(tmp_path)

    with pytest.raises(SystemExit) as error:
        main(
            [
                "--raw-root", str(raw_root),
                "--preprocessed-root", str(preprocessed_root),
                "--results-root", str(results_root),
                "--run",
            ]
        )

    assert error.value.code == 2
