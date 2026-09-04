from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from standalone_nnunet2d.alignment_evidence import (
    ALIGNMENT_EVIDENCE_POLICY,
    OFFICIAL_ALIGNED,
    build_alignment_evidence,
    resolve_alignment_state,
    validate_alignment_evidence_record,
)


COMPONENT_NAMES = ("image", "label", "manifest", "mask")


def _components() -> dict[str, dict[str, object]]:
    return {
        name: {"status": "passed", "diagnostics": []}
        for name in COMPONENT_NAMES
    }


def _transform_report() -> dict[str, object]:
    return {
        "status": "passed",
        "run_state": "official_alignment_pending",
        "oracle_root": "/server/oracle/transform",
        "standalone_root": "/local/transform",
        "image_atol": 0.0,
        "components": _components(),
        "diagnostics": [],
    }


def _inference_report() -> dict[str, object]:
    return {
        "parity_policy": "repeat_oracle_stability_v1",
        "oracle_roots": [
            "/server/oracle/inference/0",
            "/server/oracle/inference/1",
            "/server/oracle/inference/2",
        ],
        "oracle_repeat_count": 3,
        "stable_mask_mismatch_count": 0,
        "stable_mask_mismatch_coordinates": [],
        "unobserved_standalone_label_count": 0,
        "unobserved_standalone_label_coordinates": [],
        "status": "passed",
        "run_state": "official_alignment_pending",
        "standalone_root": "/local/inference",
        "image_atol": 0.0,
        "components": _components(),
        "diagnostics": [],
    }


def _write_report(path: Path, report: object) -> Path:
    path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    return path


def _write_pair(
    tmp_path: Path,
    *,
    transform: dict[str, object] | None = None,
    inference: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    return (
        _write_report(
            tmp_path / "transform.json",
            _transform_report() if transform is None else transform,
        ),
        _write_report(
            tmp_path / "inference.json",
            _inference_report() if inference is None else inference,
        ),
    )


def test_valid_reports_produce_validated_official_evidence(tmp_path: Path) -> None:
    transform_path = _write_report(tmp_path / "transform.json", _transform_report())
    inference_path = _write_report(tmp_path / "inference.json", _inference_report())

    state, evidence = resolve_alignment_state(transform_path, inference_path)

    assert state == OFFICIAL_ALIGNED
    assert evidence is not None
    assert evidence["schema_version"] == 1
    assert evidence["policy"] == ALIGNMENT_EVIDENCE_POLICY
    assert evidence["run_state"] == OFFICIAL_ALIGNED
    assert evidence["sources"]["transform"]["path"] == str(transform_path.resolve())
    assert evidence["sources"]["inference"]["path"] == str(inference_path.resolve())
    assert evidence["sources"]["transform"]["sha256"] == hashlib.sha256(
        transform_path.read_bytes()
    ).hexdigest()
    assert evidence["sources"]["inference"]["sha256"] == hashlib.sha256(
        inference_path.read_bytes()
    ).hexdigest()
    assert evidence["inference"]["parity_policy"] == "repeat_oracle_stability_v1"
    assert evidence["inference"]["oracle_repeat_count"] == 3
    assert evidence["sources"]["transform"]["snapshot"]["status"] == "passed"
    assert evidence["sources"]["inference"]["snapshot"]["status"] == "passed"
    assert json.dumps(evidence, sort_keys=True)

    validated = validate_alignment_evidence_record(evidence)
    assert validated == evidence
    assert validated is not evidence
    validated["sources"]["transform"]["snapshot"]["components"]["image"][
        "status"
    ] = "tampered"
    assert evidence["sources"]["transform"]["snapshot"]["components"]["image"][
        "status"
    ] == "passed"


def test_resolve_without_reports_preserves_pending_state() -> None:
    assert resolve_alignment_state(None, None) == (
        "official_alignment_pending",
        None,
    )


def test_checkpoint_metadata_validator_accepts_pending_without_evidence() -> None:
    from standalone_nnunet2d import alignment_evidence

    state, evidence = alignment_evidence.validate_checkpoint_alignment_metadata(
        {
            "run_type": "official_alignment_pending",
            "run_state": "official_alignment_pending",
        }
    )

    assert state == "official_alignment_pending"
    assert evidence is None


def test_checkpoint_metadata_validator_rejects_pending_evidence_and_unknown_state(
    tmp_path: Path,
) -> None:
    from standalone_nnunet2d import alignment_evidence

    transform_path, inference_path = _write_pair(tmp_path)
    evidence = build_alignment_evidence(transform_path, inference_path)

    with pytest.raises(ValueError, match="pending.*evidence"):
        alignment_evidence.validate_checkpoint_alignment_metadata(
            {
                "run_type": "official_alignment_pending",
                "run_state": "official_alignment_pending",
                "alignment_evidence": evidence,
            }
        )

    with pytest.raises(ValueError, match="unsupported.*run_state|unknown"):
        alignment_evidence.validate_checkpoint_alignment_metadata(
            {"run_type": "experimental", "run_state": "experimental"}
        )


def test_checkpoint_metadata_validator_requires_and_validates_aligned_evidence(
    tmp_path: Path,
) -> None:
    from standalone_nnunet2d import alignment_evidence

    transform_path, inference_path = _write_pair(tmp_path)
    evidence = build_alignment_evidence(transform_path, inference_path)

    state, validated = alignment_evidence.validate_checkpoint_alignment_metadata(
        {
            "run_type": OFFICIAL_ALIGNED,
            "run_state": OFFICIAL_ALIGNED,
            "alignment_evidence": evidence,
        }
    )
    assert state == OFFICIAL_ALIGNED
    assert validated == evidence
    assert validated is not evidence

    with pytest.raises(ValueError, match="requires alignment evidence"):
        alignment_evidence.validate_checkpoint_alignment_metadata(
            {"run_type": OFFICIAL_ALIGNED, "run_state": OFFICIAL_ALIGNED}
        )

    tampered = deepcopy(evidence)
    tampered["sources"]["transform"]["snapshot"]["status"] = "failed"
    with pytest.raises(ValueError, match="alignment evidence"):
        alignment_evidence.validate_checkpoint_alignment_metadata(
            {
                "run_type": OFFICIAL_ALIGNED,
                "run_state": OFFICIAL_ALIGNED,
                "alignment_evidence": tampered,
            }
        )


@pytest.mark.parametrize(
    ("transform_path", "inference_path"),
    [
        (Path("transform.json"), None),
        (None, Path("inference.json")),
    ],
)
def test_resolve_requires_both_report_paths(
    transform_path: Path | None, inference_path: Path | None
) -> None:
    with pytest.raises(ValueError, match="both"):
        resolve_alignment_state(transform_path, inference_path)


@pytest.mark.parametrize("report_kind", ["transform", "inference"])
def test_invalid_json_is_rejected(tmp_path: Path, report_kind: str) -> None:
    transform_path, inference_path = _write_pair(tmp_path)
    report_path = transform_path if report_kind == "transform" else inference_path
    report_path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match=f"invalid JSON {report_kind} report"):
        build_alignment_evidence(transform_path, inference_path)


@pytest.mark.parametrize("report_kind", ["transform", "inference"])
@pytest.mark.parametrize("top_level", [[], "report", 7, None])
def test_reports_must_have_object_top_level(
    tmp_path: Path, report_kind: str, top_level: object
) -> None:
    transform_path, inference_path = _write_pair(tmp_path)
    report_path = _write_report(
        tmp_path / f"{report_kind}.json",
        top_level,
    )
    if report_kind == "transform":
        transform_path = report_path
    else:
        inference_path = report_path

    with pytest.raises(ValueError, match=f"{report_kind} report.*JSON object"):
        build_alignment_evidence(transform_path, inference_path)


@pytest.mark.parametrize("report_kind", ["transform", "inference"])
def test_report_status_must_be_passed(tmp_path: Path, report_kind: str) -> None:
    report = _transform_report() if report_kind == "transform" else _inference_report()
    report["status"] = "failed"
    transform_path, inference_path = _write_pair(
        tmp_path,
        transform=report if report_kind == "transform" else None,
        inference=report if report_kind == "inference" else None,
    )

    with pytest.raises(ValueError, match=f"{report_kind} report status must be passed"):
        build_alignment_evidence(transform_path, inference_path)


@pytest.mark.parametrize("report_kind", ["transform", "inference"])
def test_report_run_state_must_remain_pending(tmp_path: Path, report_kind: str) -> None:
    report = _transform_report() if report_kind == "transform" else _inference_report()
    report["run_state"] = "official_aligned"
    transform_path, inference_path = _write_pair(
        tmp_path,
        transform=report if report_kind == "transform" else None,
        inference=report if report_kind == "inference" else None,
    )

    with pytest.raises(
        ValueError,
        match=f"{report_kind} report run_state must be official_alignment_pending",
    ):
        build_alignment_evidence(transform_path, inference_path)


@pytest.mark.parametrize(
    "component_update",
    [
        {"status": "failed", "diagnostics": []},
        {"status": "passed", "diagnostics": ["unexpected"]},
    ],
)
def test_failed_or_diagnosed_component_is_rejected(
    tmp_path: Path, component_update: dict[str, object]
) -> None:
    transform = _transform_report()
    transform["components"] = deepcopy(transform["components"])
    transform["components"]["image"] = component_update
    transform_path, inference_path = _write_pair(tmp_path, transform=transform)

    with pytest.raises(ValueError, match="component"):
        build_alignment_evidence(transform_path, inference_path)


@pytest.mark.parametrize("report_kind", ["transform", "inference"])
@pytest.mark.parametrize("image_atol", [1.0, float("nan")])
def test_nonzero_or_nan_image_atol_is_rejected(
    tmp_path: Path, report_kind: str, image_atol: float
) -> None:
    report = _transform_report() if report_kind == "transform" else _inference_report()
    report["image_atol"] = image_atol
    transform_path, inference_path = _write_pair(
        tmp_path,
        transform=report if report_kind == "transform" else None,
        inference=report if report_kind == "inference" else None,
    )

    with pytest.raises(ValueError, match="image_atol"):
        build_alignment_evidence(transform_path, inference_path)


def test_transform_rejects_repeated_parity_schema(tmp_path: Path) -> None:
    transform = _transform_report()
    transform["parity_policy"] = "repeat_oracle_stability_v1"
    transform["oracle_roots"] = ["/server/oracle/0", "/server/oracle/1", "/server/oracle/2"]
    transform_path, inference_path = _write_pair(tmp_path, transform=transform)

    with pytest.raises(ValueError, match="single-root"):
        build_alignment_evidence(transform_path, inference_path)


@pytest.mark.parametrize(
    "repeated_only_field",
    [
        "oracle_repeat_count",
        "stable_mask_mismatch_count",
        "unobserved_standalone_label_count",
    ],
)
def test_transform_rejects_any_repeated_only_field(
    tmp_path: Path, repeated_only_field: str
) -> None:
    transform = _transform_report()
    transform[repeated_only_field] = 0
    transform_path, inference_path = _write_pair(tmp_path, transform=transform)

    with pytest.raises(ValueError, match=repeated_only_field):
        build_alignment_evidence(transform_path, inference_path)


def test_transform_allows_unrelated_telemetry_and_extra_component(
    tmp_path: Path,
) -> None:
    transform = _transform_report()
    transform["telemetry"] = {"elapsed_seconds": 1.25}
    transform["components"] = deepcopy(transform["components"])
    transform["components"]["probability"] = {
        "status": "passed",
        "diagnostics": ["component-specific telemetry"],
    }
    transform_path, inference_path = _write_pair(tmp_path, transform=transform)

    evidence = build_alignment_evidence(transform_path, inference_path)

    assert set(evidence["sources"]["transform"]["snapshot"]["components"]) == set(
        COMPONENT_NAMES
    )


def test_wrong_inference_policy_is_rejected(tmp_path: Path) -> None:
    inference = _inference_report()
    inference["parity_policy"] = "single_oracle_exact_v1"
    transform_path, inference_path = _write_pair(tmp_path, inference=inference)

    with pytest.raises(ValueError, match="parity_policy"):
        build_alignment_evidence(transform_path, inference_path)


def test_inference_repeat_count_must_be_at_least_three(tmp_path: Path) -> None:
    inference = _inference_report()
    inference["oracle_repeat_count"] = 2
    inference["oracle_roots"] = ["/server/oracle/0", "/server/oracle/1"]
    transform_path, inference_path = _write_pair(tmp_path, inference=inference)

    with pytest.raises(ValueError, match="three"):
        build_alignment_evidence(transform_path, inference_path)


def test_inference_oracle_roots_must_be_distinct(tmp_path: Path) -> None:
    inference = _inference_report()
    inference["oracle_roots"] = [
        "/server/oracle/0",
        "/server/oracle/0",
        "/server/oracle/2",
    ]
    transform_path, inference_path = _write_pair(tmp_path, inference=inference)

    with pytest.raises(ValueError, match="distinct"):
        build_alignment_evidence(transform_path, inference_path)


def test_stable_mask_mismatch_is_rejected(tmp_path: Path) -> None:
    inference = _inference_report()
    inference["stable_mask_mismatch_count"] = 1
    inference["stable_mask_mismatch_coordinates"] = [[0, 0]]
    transform_path, inference_path = _write_pair(tmp_path, inference=inference)

    with pytest.raises(ValueError, match="stable_mask_mismatch"):
        build_alignment_evidence(transform_path, inference_path)


def test_unobserved_standalone_label_is_rejected(tmp_path: Path) -> None:
    inference = _inference_report()
    inference["unobserved_standalone_label_count"] = 1
    inference["unobserved_standalone_label_coordinates"] = [[0, 0]]
    transform_path, inference_path = _write_pair(tmp_path, inference=inference)

    with pytest.raises(ValueError, match="unobserved_standalone_label"):
        build_alignment_evidence(transform_path, inference_path)


@pytest.mark.parametrize(
    ("count_field", "invalid_count"),
    [
        ("stable_mask_mismatch_count", False),
        ("stable_mask_mismatch_count", 0.0),
        ("unobserved_standalone_label_count", False),
        ("unobserved_standalone_label_count", 0.0),
    ],
)
def test_inference_counts_must_be_non_bool_int(
    tmp_path: Path, count_field: str, invalid_count: object
) -> None:
    inference = _inference_report()
    inference[count_field] = invalid_count
    transform_path, inference_path = _write_pair(tmp_path, inference=inference)

    with pytest.raises(ValueError, match=count_field):
        build_alignment_evidence(transform_path, inference_path)


def _valid_evidence(tmp_path: Path) -> dict[str, object]:
    transform_path, inference_path = _write_pair(tmp_path)
    return build_alignment_evidence(transform_path, inference_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "schema_version"),
        ("policy", "wrong-policy", "policy"),
        ("run_state", "official_alignment_pending", "run_state"),
    ],
)
def test_evidence_header_is_failure_closed(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    evidence = _valid_evidence(tmp_path)
    evidence[field] = value

    with pytest.raises(ValueError, match=message):
        validate_alignment_evidence_record(evidence)


def test_evidence_rejects_tampered_sha256(tmp_path: Path) -> None:
    evidence = _valid_evidence(tmp_path)
    evidence["sources"]["transform"]["sha256"] = "not-a-sha"

    with pytest.raises(ValueError, match="sha256"):
        validate_alignment_evidence_record(evidence)


def test_evidence_rejects_non_absolute_source_path(tmp_path: Path) -> None:
    evidence = _valid_evidence(tmp_path)
    evidence["sources"]["inference"]["path"] = "inference.json"

    with pytest.raises(ValueError, match="absolute"):
        validate_alignment_evidence_record(evidence)


def test_evidence_rejects_tampered_snapshot(tmp_path: Path) -> None:
    evidence = _valid_evidence(tmp_path)
    evidence["sources"]["transform"]["snapshot"]["components"]["mask"][
        "status"
    ] = "failed"

    with pytest.raises(ValueError, match="snapshot"):
        validate_alignment_evidence_record(evidence)


def test_evidence_rejects_inference_metadata_snapshot_mismatch(tmp_path: Path) -> None:
    evidence = _valid_evidence(tmp_path)
    evidence["inference"]["oracle_repeat_count"] = 4

    with pytest.raises(ValueError, match="metadata does not match snapshot"):
        validate_alignment_evidence_record(evidence)


def test_evidence_rejects_extra_schema_fields(tmp_path: Path) -> None:
    evidence = _valid_evidence(tmp_path)
    evidence["unexpected"] = True

    with pytest.raises(ValueError, match="schema"):
        validate_alignment_evidence_record(evidence)


def test_evidence_record_must_be_a_json_object() -> None:
    with pytest.raises(ValueError, match="object"):
        validate_alignment_evidence_record([])
