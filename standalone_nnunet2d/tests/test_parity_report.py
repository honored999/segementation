from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from standalone_nnunet2d.tools.parity_report import (
    compare_artifacts,
    compare_repeated_oracle_inference,
)


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


def _write_repeated_inference_artifacts(
    tmp_path: Path,
    oracle_masks: tuple[np.ndarray, ...],
    standalone_mask: np.ndarray,
) -> tuple[tuple[Path, ...], Path]:
    image = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    label = np.array([[0, 0], [1, 0]], dtype=np.int16)
    oracle_roots: list[Path] = []
    for index, mask in enumerate(oracle_masks):
        root = tmp_path / f"oracle_{index}"
        _write_artifact(
            root,
            image=image,
            label=label,
            mask=mask,
            manifest_overrides={
                "transform_policy": {"mode": "inference", "implementation": "oracle"},
                "sampling_policy": {"seed": 17, "fold": 0, "implementation": "oracle"},
                "inference_context": {
                    "fold": 0,
                    "source_checkpoint_sha256": "a" * 64,
                    "device": "cpu",
                },
            },
        )
        oracle_roots.append(root)
    standalone_root = tmp_path / "standalone"
    _write_artifact(
        standalone_root,
        image=image,
        label=label,
        mask=standalone_mask,
            manifest_overrides={
                "transform_policy": {"mode": "inference", "implementation": "standalone"},
                "sampling_policy": {"seed": 17, "fold": 0, "implementation": "standalone"},
                "inference_context": {
                    "fold": 0,
                    "source_checkpoint_sha256": "a" * 64,
                    "device": "cpu",
                },
            },
    )
    return tuple(oracle_roots), standalone_root


def test_repeated_oracle_gate_accepts_only_observed_labels_on_unstable_voxels(
    tmp_path: Path,
) -> None:
    oracle_masks = (
        np.array([[0, 0], [1, 0]], dtype=np.uint8),
        np.array([[1, 0], [1, 1]], dtype=np.uint8),
        np.array([[1, 0], [1, 0]], dtype=np.uint8),
    )
    standalone_mask = np.array([[0, 0], [1, 1]], dtype=np.uint8)
    oracle_roots, standalone_root = _write_repeated_inference_artifacts(
        tmp_path, oracle_masks, standalone_mask
    )

    report = compare_repeated_oracle_inference(oracle_roots, standalone_root, 0.0)

    assert report["status"] == "passed"
    assert report["parity_policy"] == "repeat_oracle_stability_v1"
    assert report["oracle_repeat_count"] == 3
    assert report["oracle_unstable_voxel_count"] == 2
    assert report["oracle_unstable_voxel_coordinates"] == [[0, 0], [1, 1]]
    assert report["oracle_pairwise_mask_difference_counts"] == [
        {"left_index": 0, "right_index": 1, "difference_count": 2},
        {"left_index": 0, "right_index": 2, "difference_count": 1},
        {"left_index": 1, "right_index": 2, "difference_count": 1},
    ]
    assert report["stable_mask_mismatch_count"] == 0
    assert report["unobserved_standalone_label_count"] == 0
    assert report["run_state"] == "official_alignment_pending"


def test_repeated_oracle_gate_rejects_stable_voxel_mismatch(tmp_path: Path) -> None:
    oracle_masks = tuple(
        np.array([[0, 0], [1, 0]], dtype=np.uint8) for _ in range(3)
    )
    standalone_mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    roots, standalone = _write_repeated_inference_artifacts(
        tmp_path, oracle_masks, standalone_mask
    )

    report = compare_repeated_oracle_inference(roots, standalone)

    assert report["status"] == "failed"
    assert report["stable_mask_mismatch_count"] == 1
    assert report["stable_mask_mismatch_coordinates"] == [[0, 1]]
    assert report["run_state"] == "official_alignment_pending"


def test_repeated_oracle_gate_rejects_unobserved_unstable_label(tmp_path: Path) -> None:
    oracle_masks = (
        np.array([[0, 0], [1, 0]], dtype=np.uint8),
        np.array([[1, 0], [1, 0]], dtype=np.uint8),
        np.array([[0, 0], [1, 0]], dtype=np.uint8),
    )
    standalone_mask = np.array([[2, 0], [1, 0]], dtype=np.uint8)
    roots, standalone = _write_repeated_inference_artifacts(
        tmp_path, oracle_masks, standalone_mask
    )

    report = compare_repeated_oracle_inference(roots, standalone)

    assert report["status"] == "failed"
    assert report["unobserved_standalone_label_count"] == 1
    assert report["unobserved_standalone_label_coordinates"] == [[0, 0]]
    assert report["run_state"] == "official_alignment_pending"


@pytest.mark.parametrize("repeat_count", [0, 1, 2])
def test_repeated_oracle_gate_requires_three_distinct_roots(
    tmp_path: Path, repeat_count: int
) -> None:
    masks = tuple(np.zeros((2, 2), dtype=np.uint8) for _ in range(3))
    roots, standalone = _write_repeated_inference_artifacts(
        tmp_path, masks, np.zeros((2, 2), dtype=np.uint8)
    )

    with pytest.raises(ValueError, match="at least three"):
        compare_repeated_oracle_inference(roots[:repeat_count], standalone)


def test_repeated_oracle_gate_rejects_duplicate_roots(tmp_path: Path) -> None:
    masks = tuple(np.zeros((2, 2), dtype=np.uint8) for _ in range(3))
    roots, standalone = _write_repeated_inference_artifacts(
        tmp_path, masks, np.zeros((2, 2), dtype=np.uint8)
    )

    with pytest.raises(ValueError, match="distinct"):
        compare_repeated_oracle_inference([roots[0], roots[0], roots[0]], standalone)


@pytest.mark.parametrize("standalone_selector", ["same", "ancestor", "descendant"])
def test_repeated_oracle_gate_rejects_overlapping_standalone_root(
    tmp_path: Path, standalone_selector: str
) -> None:
    masks = tuple(np.zeros((2, 2), dtype=np.uint8) for _ in range(3))
    roots, standalone = _write_repeated_inference_artifacts(
        tmp_path / "artifacts", masks, np.zeros((2, 2), dtype=np.uint8)
    )
    if standalone_selector == "same":
        overlapping_root = roots[0]
    elif standalone_selector == "ancestor":
        overlapping_root = roots[0].parent
    else:
        overlapping_root = roots[0] / "nested"

    with pytest.raises(ValueError, match="overlap|independent"):
        compare_repeated_oracle_inference(roots, overlapping_root)


def test_repeated_oracle_gate_rejects_non_inference_mode(tmp_path: Path) -> None:
    masks = tuple(np.zeros((2, 2), dtype=np.uint8) for _ in range(3))
    roots, standalone = _write_repeated_inference_artifacts(
        tmp_path, masks, np.zeros((2, 2), dtype=np.uint8)
    )
    manifest_path = roots[1] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["transform_policy"] = {"mode": "transform", "implementation": "oracle"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="inference"):
        compare_repeated_oracle_inference(roots, standalone)


def test_repeated_oracle_gate_rejects_missing_inference_context(tmp_path: Path) -> None:
    masks = tuple(np.zeros((2, 2), dtype=np.uint8) for _ in range(3))
    roots, standalone = _write_repeated_inference_artifacts(
        tmp_path, masks, np.zeros((2, 2), dtype=np.uint8)
    )
    manifest_path = roots[1] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["inference_context"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = compare_repeated_oracle_inference(roots, standalone)

    assert report["status"] == "failed"
    assert report["components"]["manifest"]["status"] == "failed"
    assert any("inference_context" in diagnostic for diagnostic in report["diagnostics"])
    assert report["run_state"] == "official_alignment_pending"


def test_repeated_oracle_gate_rejects_mismatched_inference_context(tmp_path: Path) -> None:
    masks = tuple(np.zeros((2, 2), dtype=np.uint8) for _ in range(3))
    roots, standalone = _write_repeated_inference_artifacts(
        tmp_path, masks, np.zeros((2, 2), dtype=np.uint8)
    )
    manifest_path = standalone / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inference_context"]["source_checkpoint_sha256"] = "b" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = compare_repeated_oracle_inference(roots, standalone)

    assert report["status"] == "failed"
    assert any("inference_context" in diagnostic for diagnostic in report["diagnostics"])


def test_repeated_oracle_gate_rejects_oracle_sampling_policy_mismatch(
    tmp_path: Path,
) -> None:
    masks = tuple(np.zeros((2, 2), dtype=np.uint8) for _ in range(3))
    roots, standalone = _write_repeated_inference_artifacts(
        tmp_path, masks, np.zeros((2, 2), dtype=np.uint8)
    )
    manifest_path = roots[1] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sampling_policy"]["fold"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = compare_repeated_oracle_inference(roots, standalone)

    assert report["status"] == "failed"
    assert any("sampling_policy" in diagnostic for diagnostic in report["diagnostics"])


@pytest.mark.parametrize(
    ("array_name", "replacement"),
    [
        ("image", np.array([[1.0, 2.0], [3.0, 9.0]], dtype=np.float32)),
        ("label", np.array([[0, 0], [1, 1]], dtype=np.int16)),
    ],
)
def test_repeated_oracle_gate_rejects_image_or_label_difference(
    tmp_path: Path, array_name: str, replacement: np.ndarray
) -> None:
    masks = tuple(np.zeros((2, 2), dtype=np.uint8) for _ in range(3))
    roots, standalone = _write_repeated_inference_artifacts(
        tmp_path, masks, np.zeros((2, 2), dtype=np.uint8)
    )
    np.save(roots[1] / f"{array_name}.npy", replacement)

    report = compare_repeated_oracle_inference(roots, standalone)

    assert report["status"] == "failed"
    assert report["components"][array_name]["status"] == "failed"
    assert report["run_state"] == "official_alignment_pending"


@pytest.mark.parametrize("manifest_override", [{"plans_hash": "different"}, {"nifti_metadata": {
    "spacing_xyz": [0.6, 0.5, 5.0],
    "origin_xyz": [0.0, 0.0, 0.0],
    "direction": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
}}])
def test_repeated_oracle_gate_rejects_manifest_or_spatial_metadata_difference(
    tmp_path: Path, manifest_override: dict[str, object]
) -> None:
    masks = tuple(np.zeros((2, 2), dtype=np.uint8) for _ in range(3))
    roots, standalone = _write_repeated_inference_artifacts(
        tmp_path, masks, np.zeros((2, 2), dtype=np.uint8)
    )
    manifest_path = roots[1] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(manifest_override)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = compare_repeated_oracle_inference(roots, standalone)

    assert report["status"] == "failed"
    assert report["components"]["manifest"]["status"] == "failed"
    assert report["run_state"] == "official_alignment_pending"


@pytest.mark.parametrize(
    "standalone_mask",
    [
        np.zeros((1, 2), dtype=np.uint8),
        np.zeros((2, 2), dtype=np.int16),
        np.zeros((2, 2), dtype=np.float32),
    ],
)
def test_repeated_oracle_gate_rejects_mask_shape_or_dtype(
    tmp_path: Path, standalone_mask: np.ndarray
) -> None:
    masks = tuple(np.zeros((2, 2), dtype=np.uint8) for _ in range(3))
    roots, standalone = _write_repeated_inference_artifacts(
        tmp_path, masks, standalone_mask
    )

    report = compare_repeated_oracle_inference(roots, standalone)

    assert report["status"] == "failed"
    assert report["components"]["mask"]["status"] == "failed"
    assert report["run_state"] == "official_alignment_pending"


@pytest.mark.parametrize("image_atol", [1.0, -1.0, np.inf, np.nan])
def test_repeated_oracle_gate_requires_zero_finite_image_atol(
    tmp_path: Path, image_atol: float
) -> None:
    masks = tuple(np.zeros((2, 2), dtype=np.uint8) for _ in range(3))
    roots, standalone = _write_repeated_inference_artifacts(
        tmp_path, masks, np.zeros((2, 2), dtype=np.uint8)
    )

    with pytest.raises(ValueError, match="image_atol"):
        compare_repeated_oracle_inference(roots, standalone, image_atol=image_atol)


def test_cli_single_oracle_root_preserves_transform_contract(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    oracle, standalone = _write_pair(tmp_path)
    output = tmp_path / "single_report.json"

    from standalone_nnunet2d.tools.parity_report import main

    exit_code = main(
        [
            "--oracle-root",
            str(oracle),
            "--standalone-root",
            str(standalone),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["oracle_root"] == str(oracle.resolve())
    assert report["run_state"] == "official_alignment_pending"
    assert "parity_policy" not in report
    assert json.loads(capsys.readouterr().out)["status"] == "passed"


def test_cli_three_oracle_roots_dispatches_repeated_comparison(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    masks = tuple(np.zeros((2, 2), dtype=np.uint8) for _ in range(3))
    oracle_roots, standalone = _write_repeated_inference_artifacts(
        tmp_path, masks, np.zeros((2, 2), dtype=np.uint8)
    )
    output = tmp_path / "repeated_report.json"

    from standalone_nnunet2d.tools.parity_report import main

    arguments = [
        item
        for root in oracle_roots
        for item in ("--oracle-root", str(root))
    ]
    exit_code = main(
        [
            *arguments,
            "--standalone-root",
            str(standalone),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["parity_policy"] == "repeat_oracle_stability_v1"
    assert report["oracle_repeat_count"] == 3
    assert report["run_state"] == "official_alignment_pending"
    assert json.loads(capsys.readouterr().out)["status"] == "passed"


def test_cli_two_oracle_roots_is_a_parser_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    masks = tuple(np.zeros((2, 2), dtype=np.uint8) for _ in range(3))
    oracle_roots, standalone = _write_repeated_inference_artifacts(
        tmp_path, masks, np.zeros((2, 2), dtype=np.uint8)
    )

    from standalone_nnunet2d.tools.parity_report import main

    with pytest.raises(SystemExit) as error:
        main(
            [
                "--oracle-root",
                str(oracle_roots[0]),
                "--oracle-root",
                str(oracle_roots[1]),
                "--standalone-root",
                str(standalone),
            ]
        )

    assert error.value.code == 2
    assert "one oracle root or at least three distinct oracle roots" in capsys.readouterr().err


def test_cli_repeated_failure_returns_one_and_writes_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    masks = tuple(np.zeros((2, 2), dtype=np.uint8) for _ in range(3))
    oracle_roots, standalone = _write_repeated_inference_artifacts(
        tmp_path, masks, np.array([[0, 1], [0, 0]], dtype=np.uint8)
    )
    output = tmp_path / "failed_repeated_report.json"

    from standalone_nnunet2d.tools.parity_report import main

    arguments = [
        item
        for root in oracle_roots
        for item in ("--oracle-root", str(root))
    ]
    exit_code = main(
        [
            *arguments,
            "--standalone-root",
            str(standalone),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["stable_mask_mismatch_count"] == 1
    assert report["run_state"] == "official_alignment_pending"
    assert json.loads(capsys.readouterr().out)["status"] == "failed"


def test_cli_repeated_validation_errors_remain_explicit(tmp_path: Path) -> None:
    masks = tuple(np.zeros((2, 2), dtype=np.uint8) for _ in range(3))
    oracle_roots, standalone = _write_repeated_inference_artifacts(
        tmp_path, masks, np.zeros((2, 2), dtype=np.uint8)
    )

    from standalone_nnunet2d.tools.parity_report import main

    duplicate_arguments = [
        "--oracle-root",
        str(oracle_roots[0]),
        "--oracle-root",
        str(oracle_roots[0]),
        "--oracle-root",
        str(oracle_roots[0]),
        "--standalone-root",
        str(standalone),
    ]
    with pytest.raises(ValueError, match="distinct"):
        main(duplicate_arguments)

    with pytest.raises(ValueError, match="image_atol"):
        main(
            [
                "--oracle-root",
                str(oracle_roots[0]),
                "--oracle-root",
                str(oracle_roots[1]),
                "--oracle-root",
                str(oracle_roots[2]),
                "--standalone-root",
                str(standalone),
                "--image-atol",
                "1.0",
            ]
        )


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
