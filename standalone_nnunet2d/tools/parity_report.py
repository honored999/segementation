"""Compare portable oracle and standalone artifacts without importing nnU-Net."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


RUN_STATE = "official_alignment_pending"
MANIFEST_NAME = "manifest.json"
REQUIRED_MANIFEST_FIELDS = (
    "artifact_version",
    "nnunetv2_version",
    "plans_hash",
    "seed",
    "case_id",
    "transform_policy",
    "sampling_policy",
    "arrays",
    "nifti_metadata",
)
REQUIRED_ARRAYS = ("image", "label", "mask")


def _load_manifest(root: Path) -> tuple[dict[str, Any] | None, list[str]]:
    path = root / MANIFEST_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, [f"manifest unreadable at {path}: {error}"]
    if not isinstance(payload, dict):
        return None, [f"manifest must contain a JSON object: {path}"]
    missing = [field for field in REQUIRED_MANIFEST_FIELDS if field not in payload]
    if missing:
        return payload, [f"manifest missing mandatory fields: {', '.join(missing)}"]
    if not isinstance(payload["arrays"], Mapping):
        return payload, ["manifest field 'arrays' must be an object"]
    missing_arrays = [name for name in REQUIRED_ARRAYS if name not in payload["arrays"]]
    if missing_arrays:
        return payload, [f"manifest missing mandatory arrays: {', '.join(missing_arrays)}"]
    return payload, []


def _manifest_value_differences(
    oracle: Mapping[str, Any], standalone: Mapping[str, Any]
) -> list[str]:
    differences: list[str] = []
    for field in REQUIRED_MANIFEST_FIELDS:
        if field == "arrays":
            continue
        if oracle.get(field) != standalone.get(field):
            differences.append(f"manifest field differs: {field}")
    oracle_arrays = oracle.get("arrays")
    standalone_arrays = standalone.get("arrays")
    if isinstance(oracle_arrays, Mapping) and isinstance(standalone_arrays, Mapping):
        if set(oracle_arrays) != set(standalone_arrays):
            differences.append("manifest array names differ")
        for name in sorted(set(oracle_arrays) & set(standalone_arrays)):
            if oracle_arrays[name] != standalone_arrays[name]:
                differences.append(f"manifest array description differs: {name}")
    return differences


def _array_path(root: Path, description: Any, name: str) -> Path:
    if not isinstance(description, Mapping) or not isinstance(description.get("file"), str):
        raise ValueError(f"manifest array '{name}' must provide a string file")
    relative = Path(description["file"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"manifest array '{name}' has an unsafe file path")
    return root / relative


def _load_array(root: Path, manifest: Mapping[str, Any], name: str) -> np.ndarray:
    arrays = manifest.get("arrays")
    if not isinstance(arrays, Mapping):
        raise ValueError("manifest field 'arrays' must be an object")
    path = _array_path(root, arrays.get(name), name)
    if not path.is_file():
        raise FileNotFoundError(f"array '{name}' does not exist: {path}")
    array = np.load(path, allow_pickle=False)
    if not isinstance(array, np.ndarray):
        raise ValueError(f"array '{name}' is not a NumPy array")
    return array


def _array_matches(
    name: str,
    oracle: np.ndarray,
    standalone: np.ndarray,
    *,
    image_atol: float,
) -> tuple[bool, str | None]:
    if oracle.shape != standalone.shape:
        return False, f"{name}: shape differs: {oracle.shape} != {standalone.shape}"
    if oracle.dtype != standalone.dtype:
        return False, f"{name}: dtype differs: {oracle.dtype} != {standalone.dtype}"
    if name in ("label", "mask"):
        if not np.issubdtype(oracle.dtype, np.integer):
            return False, f"{name}: expected an integer dtype, got {oracle.dtype}"
        if not np.array_equal(oracle, standalone):
            return False, f"{name}: integer values differ"
        return True, None
    if np.issubdtype(oracle.dtype, np.floating):
        if not np.allclose(oracle, standalone, rtol=0.0, atol=image_atol, equal_nan=False):
            return False, f"{name}: float values exceed atol={image_atol}"
        return True, None
    if not np.array_equal(oracle, standalone):
        return False, f"{name}: values differ"
    return True, None


def compare_artifacts(
    oracle_root: Path,
    standalone_root: Path,
    *,
    image_atol: float = 0.0,
) -> dict[str, Any]:
    """Compare two artifact directories and return a pending-state report.

    Integer labels and masks are compared with exact equality. Floating arrays
    use only the caller-declared absolute tolerance and never an implicit
    relative tolerance. A report cannot claim official alignment by itself.
    """
    if image_atol < 0.0 or not np.isfinite(image_atol):
        raise ValueError(f"image_atol must be a finite non-negative value, got {image_atol}")
    oracle_root = Path(oracle_root).resolve()
    standalone_root = Path(standalone_root).resolve()
    diagnostics: list[str] = []
    components: dict[str, dict[str, Any]] = {}

    oracle_manifest, oracle_errors = _load_manifest(oracle_root)
    standalone_manifest, standalone_errors = _load_manifest(standalone_root)
    manifest_errors = [f"oracle: {error}" for error in oracle_errors]
    manifest_errors.extend(f"standalone: {error}" for error in standalone_errors)
    if oracle_manifest is not None and standalone_manifest is not None:
        manifest_errors.extend(_manifest_value_differences(oracle_manifest, standalone_manifest))
    components["manifest"] = {
        "status": "passed" if not manifest_errors else "failed",
        "diagnostics": manifest_errors,
    }
    diagnostics.extend(manifest_errors)

    array_names: set[str] = set(REQUIRED_ARRAYS)
    if oracle_manifest is not None and isinstance(oracle_manifest.get("arrays"), Mapping):
        array_names.update(str(name) for name in oracle_manifest["arrays"])
    if standalone_manifest is not None and isinstance(standalone_manifest.get("arrays"), Mapping):
        array_names.update(str(name) for name in standalone_manifest["arrays"])

    for name in sorted(array_names):
        if oracle_manifest is None or standalone_manifest is None:
            components[name] = {"status": "failed", "diagnostics": ["manifest validation failed"]}
            continue
        try:
            oracle_array = _load_array(oracle_root, oracle_manifest, name)
            standalone_array = _load_array(standalone_root, standalone_manifest, name)
            passed, diagnostic = _array_matches(
                name,
                oracle_array,
                standalone_array,
                image_atol=image_atol,
            )
            array_diagnostics = [] if diagnostic is None else [diagnostic]
        except (OSError, ValueError) as error:
            passed = False
            array_diagnostics = [f"{name}: {error}"]
        components[name] = {"status": "passed" if passed else "failed", "diagnostics": array_diagnostics}
        diagnostics.extend(array_diagnostics)

    status = "passed" if not diagnostics else "failed"
    return {
        "status": status,
        "run_state": RUN_STATE,
        "oracle_root": str(oracle_root),
        "standalone_root": str(standalone_root),
        "image_atol": float(image_atol),
        "components": components,
        "diagnostics": diagnostics,
    }


def write_report(path: Path, report: Mapping[str, Any]) -> Path:
    """Write a JSON-safe parity report to a caller-selected path."""
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(dict(report), indent=2, sort_keys=True), encoding="utf-8")
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare oracle and standalone parity artifacts")
    parser.add_argument("--oracle-root", required=True, type=Path)
    parser.add_argument("--standalone-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--image-atol", type=float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = compare_artifacts(
        arguments.oracle_root,
        arguments.standalone_root,
        image_atol=arguments.image_atol,
    )
    if arguments.output is not None:
        write_report(arguments.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
