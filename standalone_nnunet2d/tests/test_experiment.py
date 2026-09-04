from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from standalone_nnunet2d.engine.checkpoint import PROJECT_OUTPUTS_DIRECTORY
from standalone_nnunet2d.experiment import main


def _roots(tmp_path: Path) -> tuple[Path, Path, Path]:
    roots = tuple(tmp_path / name for name in ("raw", "preprocessed", "results"))
    for root in roots:
        root.mkdir()
    (roots[0] / "imagesTr").mkdir()
    (roots[0] / "labelsTr").mkdir()
    return roots  # type: ignore[return-value]


def _output_path(label: str) -> Path:
    return PROJECT_OUTPUTS_DIRECTORY / f"pytest-{label}-{uuid4().hex}"


def _arguments(
    raw_root: Path,
    preprocessed_root: Path,
    results_root: Path,
    output_root: Path,
    *,
    fold: str = "0",
    confirm: bool = False,
) -> list[str]:
    arguments = [
        "--raw-root", str(raw_root),
        "--preprocessed-root", str(preprocessed_root),
        "--results-root", str(results_root),
        "--output-root", str(output_root),
        "--fold", fold,
        "--epochs", "1",
        "--device", "cpu",
    ]
    return [*arguments, "--confirm-run"] if confirm else arguments


def test_experiment_reports_valid_request_without_training(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_root, preprocessed_root, results_root = _roots(tmp_path)
    output_root = _output_path("default")

    assert main(_arguments(raw_root, preprocessed_root, results_root, output_root)) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["request"]["fold"] == 0
    assert payload["request"]["epochs"] == 1
    assert payload["execution"] == "not-confirmed"
    assert not output_root.exists()


def test_experiment_rejects_invalid_fold(tmp_path: Path) -> None:
    raw_root, preprocessed_root, results_root = _roots(tmp_path)

    with pytest.raises(SystemExit) as error:
        main(_arguments(raw_root, preprocessed_root, results_root, _output_path("bad-fold"), fold="5"))

    assert error.value.code == 2


def test_experiment_rejects_output_outside_project_outputs(tmp_path: Path) -> None:
    raw_root, preprocessed_root, results_root = _roots(tmp_path)

    with pytest.raises(SystemExit) as error:
        main(_arguments(raw_root, preprocessed_root, results_root, tmp_path / "outside"))

    assert error.value.code == 2


def test_confirmed_experiment_is_deferred_without_writing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_root, preprocessed_root, results_root = _roots(tmp_path)
    output_root = _output_path("confirmed")

    assert main(_arguments(raw_root, preprocessed_root, results_root, output_root, confirm=True)) == 3

    assert json.loads(capsys.readouterr().out)["execution"] == "deferred"
    assert not output_root.exists()
