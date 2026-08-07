from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from standalone_nnunet2d.tools.parity_report import compare_artifacts


def _write_artifact(
    root: Path,
    *,
    image: np.ndarray,
    label: np.ndarray,
    mask: np.ndarray,
    manifest_overrides: dict[str, object] | None = None,
) -> None:
    root.mkdir(parents=True)
    arrays = {
        "image": image,
        "label": label,
        "mask": mask,
    }
    array_manifest = {}
    for name, array in arrays.items():
        filename = f"{name}.npy"
        np.save(root / filename, array)
        array_manifest[name] = {
            "file": filename,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }

    manifest: dict[str, object] = {
        "artifact_version": "1",
        "nnunetv2_version": "2.5.1",
        "plans_hash": "plan-sha256",
        "seed": 17,
        "case_id": "case001",
        "transform_policy": {"mode": "transform", "name": "fixed"},
        "sampling_policy": {"name": "fixed"},
        "arrays": array_manifest,
        "nifti_metadata": {
            "spacing_xyz": [0.5, 0.5, 5.0],
            "origin_xyz": [0.0, 0.0, 0.0],
            "direction": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        },
    }
    if manifest_overrides:
        manifest.update(manifest_overrides)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_pair(
    tmp_path: Path,
    *,
    image_delta: float = 0.0,
    oracle_manifest_overrides: dict[str, object] | None = None,
    standalone_manifest_overrides: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    label = np.array([[0, 1], [1, 0]], dtype=np.int16)
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    image = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    _write_artifact(
        tmp_path / "oracle",
        image=image,
        label=label,
        mask=mask,
        manifest_overrides=oracle_manifest_overrides,
    )
    _write_artifact(
        tmp_path / "standalone",
        image=image + image_delta,
        label=label,
        mask=mask,
        manifest_overrides=standalone_manifest_overrides,
    )
    return tmp_path / "oracle", tmp_path / "standalone"


def test_compare_artifacts_allows_implementation_metadata_differences(tmp_path: Path) -> None:
    oracle, standalone = _write_pair(
        tmp_path,
        oracle_manifest_overrides={
            "nnunetv2_version": "2.5.1",
            "transform_policy": {
                "mode": "transform",
                "implementation": "oracle",
                "interpolation": "nearest",
            },
            "sampling_policy": {"seed": 17, "fold": 0, "implementation": "oracle"},
        },
        standalone_manifest_overrides={
            "nnunetv2_version": "2.6.0",
            "transform_policy": {
                "mode": "transform",
                "implementation": "standalone",
                "interpolation": "nearest",
            },
            "sampling_policy": {"seed": 17, "fold": 0, "implementation": "standalone"},
        },
    )

    report = compare_artifacts(oracle, standalone)

    assert report["status"] == "passed"
    assert report["components"]["manifest"]["status"] == "passed"
    assert report["run_state"] == "official_alignment_pending"


def test_compare_artifacts_rejects_seed_mismatch(tmp_path: Path) -> None:
    oracle, standalone = _write_pair(tmp_path, standalone_manifest_overrides={"seed": 18})

    report = compare_artifacts(oracle, standalone)

    assert report["status"] == "failed"
    assert "manifest field differs: seed" in report["diagnostics"]


def test_compare_artifacts_rejects_case_id_mismatch(tmp_path: Path) -> None:
    oracle, standalone = _write_pair(tmp_path, standalone_manifest_overrides={"case_id": "case002"})

    report = compare_artifacts(oracle, standalone)

    assert report["status"] == "failed"
    assert "manifest field differs: case_id" in report["diagnostics"]


def test_compare_artifacts_rejects_capture_mode_mismatch(tmp_path: Path) -> None:
    oracle, standalone = _write_pair(
        tmp_path,
        standalone_manifest_overrides={
            "transform_policy": {"mode": "inference", "implementation": "standalone"}
        },
    )

    report = compare_artifacts(oracle, standalone)

    assert report["status"] == "failed"
    assert "manifest capture mode differs" in report["diagnostics"]


def test_compare_artifacts_accepts_only_declared_float_tolerance(tmp_path: Path) -> None:
    oracle, standalone = _write_pair(tmp_path, image_delta=1e-7)

    report = compare_artifacts(oracle, standalone, image_atol=1e-6)

    assert report["status"] == "passed"
    assert report["run_state"] == "official_alignment_pending"

    oracle, standalone = _write_pair(tmp_path / "outside_tolerance", image_delta=2e-6)
    report = compare_artifacts(oracle, standalone, image_atol=1e-6)

    assert report["status"] == "failed"


def test_compare_artifacts_compares_integer_labels_exactly(tmp_path: Path) -> None:
    oracle, standalone = _write_pair(tmp_path)
    np.save(standalone / "label.npy", np.array([[0, 1], [1, 1]], dtype=np.int16))

    report = compare_artifacts(oracle, standalone)

    assert report["status"] == "failed"


def test_compare_artifacts_compares_integer_masks_exactly(tmp_path: Path) -> None:
    oracle, standalone = _write_pair(tmp_path)
    np.save(standalone / "mask.npy", np.array([[0, 1], [0, 0]], dtype=np.uint8))

    report = compare_artifacts(oracle, standalone)

    assert report["status"] == "failed"


def test_compare_artifacts_rejects_manifest_missing_mandatory_field(tmp_path: Path) -> None:
    oracle, standalone = _write_pair(tmp_path)
    manifest_path = standalone / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["plans_hash"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = compare_artifacts(oracle, standalone)

    assert report["status"] == "failed"
    assert report["run_state"] == "official_alignment_pending"
    assert "plans_hash" in json.dumps(report)
