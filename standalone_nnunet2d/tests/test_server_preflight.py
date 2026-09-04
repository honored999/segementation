from __future__ import annotations

from pathlib import Path

from standalone_nnunet2d.tools.server_preflight import inspect_server_readiness


def _roots(tmp_path: Path) -> tuple[Path, Path, Path]:
    roots = tuple(tmp_path / name for name in ("raw", "preprocessed", "results"))
    for root in roots:
        root.mkdir()
    (roots[0] / "imagesTr").mkdir()
    (roots[0] / "labelsTr").mkdir()
    return roots  # type: ignore[return-value]


def test_inspect_server_readiness_reports_cpu_and_reference_plan(tmp_path: Path) -> None:
    raw_root, preprocessed_root, results_root = _roots(tmp_path)

    report = inspect_server_readiness(raw_root, preprocessed_root, results_root, device="cpu")

    assert report["ready"] is True
    assert report["device"]["selected"] == "cpu"
    assert report["plan"]["patch_size"] == [512, 512]
    assert report["plan"]["batch_size"] == 12


def test_inspect_server_readiness_diagnoses_missing_directory(tmp_path: Path) -> None:
    raw_root, preprocessed_root, _ = _roots(tmp_path)
    missing_results_root = tmp_path / "missing"

    report = inspect_server_readiness(raw_root, preprocessed_root, missing_results_root, device="cpu")

    assert report["ready"] is False
    assert any("results" in message for message in report["diagnostics"])


def test_inspect_server_readiness_rejects_unavailable_cuda_device(tmp_path: Path) -> None:
    raw_root, preprocessed_root, results_root = _roots(tmp_path)

    report = inspect_server_readiness(raw_root, preprocessed_root, results_root, device="cuda:999")

    assert report["ready"] is False
    assert any("requested CUDA device is unavailable" in message for message in report["diagnostics"])
