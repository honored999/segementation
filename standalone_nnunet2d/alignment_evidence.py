from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


OFFICIAL_ALIGNED = "official_aligned"
ALIGNMENT_EVIDENCE_POLICY = "transform_exact_plus_repeat_oracle_stability_v1"
_PENDING_STATE = "official_alignment_pending"
_INFERENCE_POLICY = "repeat_oracle_stability_v1"
_COMPONENT_NAMES = ("image", "label", "manifest", "mask")
_TRANSFORM_SNAPSHOT_KEYS = {
    "status",
    "run_state",
    "image_atol",
    "oracle_root",
    "standalone_root",
    "components",
}
_INFERENCE_SNAPSHOT_KEYS = {
    "status",
    "run_state",
    "image_atol",
    "parity_policy",
    "oracle_roots",
    "oracle_repeat_count",
    "stable_mask_mismatch_count",
    "stable_mask_mismatch_coordinates",
    "unobserved_standalone_label_count",
    "unobserved_standalone_label_coordinates",
    "standalone_root",
    "components",
}
_TRANSFORM_REPEATED_ONLY_KEYS = {
    "oracle_repeat_count",
    "stable_mask_mismatch_count",
    "stable_mask_mismatch_coordinates",
    "unobserved_standalone_label_count",
    "unobserved_standalone_label_coordinates",
}
_EVIDENCE_KEYS = {"schema_version", "policy", "run_state", "sources", "inference"}
_SOURCE_KEYS = {"path", "sha256", "snapshot"}
_INFERENCE_METADATA_KEYS = {"parity_policy", "oracle_repeat_count"}


def _load_report(path: Path, report_name: str) -> dict[str, Any]:
    try:
        report = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError(f"invalid JSON {report_name} report {path}: {error}") from error
    if not isinstance(report, dict):
        raise ValueError(f"{report_name} report {path} top level must be a JSON object")
    return report


def _require_nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_non_bool_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _validate_common_report(report: dict[str, Any], report_name: str) -> None:
    if report.get("status") != "passed":
        raise ValueError(f"{report_name} status must be passed")
    if report.get("run_state") != _PENDING_STATE:
        raise ValueError(f"{report_name} run_state must be {_PENDING_STATE}")
    image_atol = report.get("image_atol")
    if (
        isinstance(image_atol, bool)
        or not isinstance(image_atol, (int, float))
        or not math.isfinite(float(image_atol))
        or float(image_atol) != 0.0
    ):
        raise ValueError(f"{report_name} image_atol must be finite 0.0")
    components = report.get("components")
    if not isinstance(components, dict):
        raise ValueError(f"{report_name} components must be an object")
    for name in _COMPONENT_NAMES:
        component = components.get(name)
        if (
            not isinstance(component, dict)
            or component.get("status") != "passed"
            or component.get("diagnostics") != []
        ):
            raise ValueError(f"{report_name} component {name} must be passed with no diagnostics")


def _validate_transform_report(report: dict[str, Any]) -> None:
    _validate_common_report(report, "transform report")
    repeated_only_fields = _TRANSFORM_REPEATED_ONLY_KEYS.intersection(report)
    if repeated_only_fields:
        fields = ", ".join(sorted(repeated_only_fields))
        raise ValueError(f"transform report contains repeated-only fields: {fields}")
    if "parity_policy" in report or "oracle_roots" in report:
        raise ValueError("transform report must use the single-root exact schema")
    _require_nonempty_string(report.get("oracle_root"), "transform oracle_root")
    _require_nonempty_string(report.get("standalone_root"), "transform standalone_root")


def _validate_inference_report(report: dict[str, Any]) -> None:
    _validate_common_report(report, "inference report")
    if report.get("parity_policy") != _INFERENCE_POLICY:
        raise ValueError(
            "inference report parity_policy must be repeat_oracle_stability_v1"
        )
    repeat_count = report.get("oracle_repeat_count")
    if isinstance(repeat_count, bool) or not isinstance(repeat_count, int) or repeat_count < 3:
        raise ValueError("inference oracle_repeat_count must be at least three")
    roots = report.get("oracle_roots")
    if not isinstance(roots, list) or len(roots) != repeat_count:
        raise ValueError("inference oracle_roots length must match oracle_repeat_count")
    for index, root in enumerate(roots):
        _require_nonempty_string(root, f"inference oracle_roots[{index}]")
    if len(set(roots)) != len(roots):
        raise ValueError("inference oracle_roots must be distinct")
    stable_mask_mismatch_count = _require_non_bool_int(
        report.get("stable_mask_mismatch_count"),
        "inference stable_mask_mismatch_count",
    )
    if stable_mask_mismatch_count != 0 or report.get(
        "stable_mask_mismatch_coordinates"
    ) != []:
        raise ValueError(
            "inference stable_mask_mismatch_count and coordinates must both be zero/empty"
        )
    unobserved_standalone_label_count = _require_non_bool_int(
        report.get("unobserved_standalone_label_count"),
        "inference unobserved_standalone_label_count",
    )
    if unobserved_standalone_label_count != 0 or report.get(
        "unobserved_standalone_label_coordinates"
    ) != []:
        raise ValueError(
            "inference unobserved_standalone_label_count and coordinates must both be zero/empty"
        )
    _require_nonempty_string(report.get("standalone_root"), "inference standalone_root")


def _component_snapshot(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "status": report["components"][name]["status"],
            "diagnostics": copy.deepcopy(report["components"][name]["diagnostics"]),
        }
        for name in _COMPONENT_NAMES
    }


def _transform_snapshot(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report["status"],
        "run_state": report["run_state"],
        "image_atol": report["image_atol"],
        "oracle_root": report["oracle_root"],
        "standalone_root": report["standalone_root"],
        "components": _component_snapshot(report),
    }


def _inference_snapshot(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report["status"],
        "run_state": report["run_state"],
        "image_atol": report["image_atol"],
        "parity_policy": report["parity_policy"],
        "oracle_roots": copy.deepcopy(report["oracle_roots"]),
        "oracle_repeat_count": report["oracle_repeat_count"],
        "stable_mask_mismatch_count": report["stable_mask_mismatch_count"],
        "stable_mask_mismatch_coordinates": copy.deepcopy(
            report["stable_mask_mismatch_coordinates"]
        ),
        "unobserved_standalone_label_count": report[
            "unobserved_standalone_label_count"
        ],
        "unobserved_standalone_label_coordinates": copy.deepcopy(
            report["unobserved_standalone_label_coordinates"]
        ),
        "standalone_root": report["standalone_root"],
        "components": _component_snapshot(report),
    }


def _source(path: Path, snapshot: dict[str, Any]) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "snapshot": snapshot,
    }


def build_alignment_evidence(
    transform_report_path: Path, inference_report_path: Path
) -> dict[str, Any]:
    transform_report = _load_report(transform_report_path, "transform")
    inference_report = _load_report(inference_report_path, "inference")
    _validate_transform_report(transform_report)
    _validate_inference_report(inference_report)
    return {
        "schema_version": 1,
        "policy": ALIGNMENT_EVIDENCE_POLICY,
        "run_state": OFFICIAL_ALIGNED,
        "sources": {
            "transform": _source(
                transform_report_path, _transform_snapshot(transform_report)
            ),
            "inference": _source(
                inference_report_path, _inference_snapshot(inference_report)
            ),
        },
        "inference": {
            "parity_policy": inference_report["parity_policy"],
            "oracle_repeat_count": inference_report["oracle_repeat_count"],
        },
    }


def _require_exact_keys(value: dict[str, Any], expected: set[str], field: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{field} schema keys are invalid")


def _validate_evidence_source(
    value: object, field: str, snapshot_keys: set[str]
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    _require_exact_keys(value, _SOURCE_KEYS, field)
    path = _require_nonempty_string(value["path"], f"{field}.path")
    if not Path(path).is_absolute():
        raise ValueError(f"{field}.path must be absolute")
    sha256 = value["sha256"]
    if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-fA-F]{64}", sha256) is None:
        raise ValueError(f"{field}.sha256 must be a 64-character hexadecimal digest")
    snapshot = value["snapshot"]
    if not isinstance(snapshot, dict):
        raise ValueError(f"{field}.snapshot must be an object")
    _require_exact_keys(snapshot, snapshot_keys, f"{field}.snapshot")
    try:
        if snapshot_keys == _TRANSFORM_SNAPSHOT_KEYS:
            _validate_transform_report(snapshot)
        else:
            _validate_inference_report(snapshot)
    except ValueError as error:
        raise ValueError(f"{field} evidence snapshot is invalid: {error}") from error
    return value


def validate_alignment_evidence_record(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("alignment evidence record must be a JSON object")
    _require_exact_keys(value, _EVIDENCE_KEYS, "alignment evidence")
    if value["schema_version"] != 1 or isinstance(value["schema_version"], bool):
        raise ValueError("alignment evidence schema_version must be 1")
    if value["policy"] != ALIGNMENT_EVIDENCE_POLICY:
        raise ValueError("alignment evidence policy is invalid")
    if value["run_state"] != OFFICIAL_ALIGNED:
        raise ValueError("alignment evidence run_state must be official_aligned")

    sources = value["sources"]
    if not isinstance(sources, dict):
        raise ValueError("alignment evidence sources must be an object")
    _require_exact_keys(sources, {"transform", "inference"}, "alignment evidence sources")
    _validate_evidence_source(
        sources["transform"], "alignment evidence transform source", _TRANSFORM_SNAPSHOT_KEYS
    )
    inference_source = _validate_evidence_source(
        sources["inference"], "alignment evidence inference source", _INFERENCE_SNAPSHOT_KEYS
    )

    inference = value["inference"]
    if not isinstance(inference, dict):
        raise ValueError("alignment evidence inference metadata must be an object")
    _require_exact_keys(inference, _INFERENCE_METADATA_KEYS, "alignment evidence inference")
    if inference["parity_policy"] != _INFERENCE_POLICY:
        raise ValueError("alignment evidence inference parity_policy is invalid")
    repeat_count = inference["oracle_repeat_count"]
    if isinstance(repeat_count, bool) or not isinstance(repeat_count, int) or repeat_count < 3:
        raise ValueError("alignment evidence inference oracle_repeat_count must be at least three")
    inference_snapshot = inference_source["snapshot"]
    if (
        inference_snapshot["parity_policy"] != inference["parity_policy"]
        or inference_snapshot["oracle_repeat_count"] != inference["oracle_repeat_count"]
    ):
        raise ValueError("alignment evidence inference metadata does not match snapshot")

    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("alignment evidence must be JSON-safe") from error


def resolve_alignment_state(
    transform_report_path: Path | None, inference_report_path: Path | None
) -> tuple[str, dict[str, Any] | None]:
    if transform_report_path is None and inference_report_path is None:
        return _PENDING_STATE, None
    if transform_report_path is None or inference_report_path is None:
        raise ValueError("both transform and inference report paths are required")
    return OFFICIAL_ALIGNED, build_alignment_evidence(
        transform_report_path, inference_report_path
    )


def validate_checkpoint_alignment_metadata(
    metadata: Mapping[str, Any],
) -> tuple[str, dict[str, Any] | None]:
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint metadata must be a mapping")

    run_type = metadata.get("run_type")
    run_state = metadata.get("run_state")
    if run_type != run_state:
        raise ValueError("checkpoint run_type and run_state must be equal")

    if run_state == _PENDING_STATE:
        if metadata.get("alignment_evidence") is not None:
            raise ValueError("pending checkpoint metadata cannot contain alignment evidence")
        return _PENDING_STATE, None

    if run_state == OFFICIAL_ALIGNED:
        if "alignment_evidence" not in metadata or metadata["alignment_evidence"] is None:
            raise ValueError("official_aligned checkpoint requires alignment evidence")
        return OFFICIAL_ALIGNED, validate_alignment_evidence_record(
            metadata["alignment_evidence"]
        )

    raise ValueError(f"unsupported checkpoint run_state: {run_state!r}")
