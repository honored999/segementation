"""Compare portable oracle and standalone artifacts without importing nnU-Net."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


RUN_STATE = "official_alignment_pending"
REPEATED_INFERENCE_POLICY = "repeat_oracle_stability_v1"
MINIMUM_ORACLE_REPEATS = 3
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
STRICT_MANIFEST_FIELDS = (
    "artifact_version",
    "plans_hash",
    "seed",
    "case_id",
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
    transform_policy = payload["transform_policy"]
    if not isinstance(transform_policy, Mapping):
        return payload, ["manifest field 'transform_policy' must be an object"]
    if not isinstance(transform_policy.get("mode"), str):
        return payload, ["manifest field 'transform_policy.mode' must be a string"]
    return payload, []


def _manifest_value_differences(
    oracle: Mapping[str, Any], standalone: Mapping[str, Any]
) -> list[str]:
    differences: list[str] = []
    for field in STRICT_MANIFEST_FIELDS:
        if oracle.get(field) != standalone.get(field):
            differences.append(f"manifest field differs: {field}")
    oracle_transform_policy = oracle.get("transform_policy")
    standalone_transform_policy = standalone.get("transform_policy")
    oracle_mode = (
        oracle_transform_policy.get("mode")
        if isinstance(oracle_transform_policy, Mapping)
        else None
    )
    standalone_mode = (
        standalone_transform_policy.get("mode")
        if isinstance(standalone_transform_policy, Mapping)
        else None
    )
    if oracle_mode != standalone_mode:
        differences.append("manifest capture mode differs")
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


def _resolved_distinct_roots(oracle_roots: Sequence[Path]) -> tuple[Path, ...]:
    roots = tuple(Path(root).resolve() for root in oracle_roots)
    if len(roots) < MINIMUM_ORACLE_REPEATS:
        raise ValueError("repeated inference parity requires at least three oracle roots")
    if len(set(roots)) != len(roots):
        raise ValueError("repeated inference parity requires distinct oracle roots")
    return roots


def _coordinates(mask: np.ndarray) -> list[list[int]]:
    return np.argwhere(mask).astype(int, copy=False).tolist()


def _mask_schema_diagnostic(
    name: str,
    reference: np.ndarray,
    candidate: np.ndarray,
) -> str | None:
    if reference.shape != candidate.shape:
        return f"{name}: shape differs: {reference.shape} != {candidate.shape}"
    if reference.dtype != candidate.dtype:
        return f"{name}: dtype differs: {reference.dtype} != {candidate.dtype}"
    if not np.issubdtype(candidate.dtype, np.integer):
        return f"{name}: expected an integer dtype, got {candidate.dtype}"
    return None


def _inference_context_diagnostics(
    manifests: Sequence[dict[str, Any] | None],
    oracle_repeat_count: int,
) -> list[str]:
    diagnostics: list[str] = []
    contexts: list[Mapping[str, Any] | None] = []
    for index, manifest in enumerate(manifests):
        if manifest is None:
            contexts.append(None)
            continue
        value = manifest.get("inference_context")
        if not isinstance(value, Mapping):
            diagnostics.append(f"artifact[{index}]: inference_context must be an object")
            contexts.append(None)
            continue
        fold = value.get("fold")
        source_sha256 = value.get("source_checkpoint_sha256")
        device = value.get("device")
        valid = True
        if isinstance(fold, bool) or not isinstance(fold, int) or fold < 0:
            diagnostics.append(
                f"artifact[{index}]: inference_context.fold must be a non-negative integer"
            )
            valid = False
        if (
            not isinstance(source_sha256, str)
            or len(source_sha256) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in source_sha256)
        ):
            diagnostics.append(
                f"artifact[{index}]: inference_context.source_checkpoint_sha256 must be a 64-character SHA256"
            )
            valid = False
        if device == "cpu":
            pass
        elif isinstance(device, str) and device.startswith("cuda:"):
            index_text = device.removeprefix("cuda:")
            if not index_text.isdigit() or str(int(index_text)) != index_text:
                diagnostics.append(
                    f"artifact[{index}]: inference_context.device must be cpu or canonical cuda:<index>"
                )
                valid = False
        else:
            diagnostics.append(
                f"artifact[{index}]: inference_context.device must be cpu or canonical cuda:<index>"
            )
            valid = False
        contexts.append(value if valid else None)

    if all(context is not None for context in contexts):
        baseline = contexts[0]
        assert baseline is not None
        for index, context in enumerate(contexts[1:], start=1):
            assert context is not None
            if dict(context) != dict(baseline):
                diagnostics.append(f"artifact[{index}]: inference_context differs")

    sampling_keys: list[tuple[int, int] | None] = []
    for index in range(oracle_repeat_count):
        manifest = manifests[index]
        if manifest is None:
            sampling_keys.append(None)
            continue
        policy = manifest.get("sampling_policy")
        if not isinstance(policy, Mapping):
            diagnostics.append(f"artifact[{index}]: sampling_policy must be an object")
            sampling_keys.append(None)
            continue
        fold = policy.get("fold")
        seed = policy.get("seed")
        if isinstance(fold, bool) or not isinstance(fold, int) or fold < 0:
            diagnostics.append(
                f"artifact[{index}]: sampling_policy.fold must be a non-negative integer"
            )
            sampling_keys.append(None)
            continue
        if isinstance(seed, bool) or not isinstance(seed, int):
            diagnostics.append(f"artifact[{index}]: sampling_policy.seed must be an integer")
            sampling_keys.append(None)
            continue
        sampling_keys.append((fold, seed))
    if all(key is not None for key in sampling_keys):
        baseline = sampling_keys[0]
        assert baseline is not None
        for index, key in enumerate(sampling_keys[1:], start=1):
            assert key is not None
            if key != baseline:
                diagnostics.append(f"artifact[{index}]: sampling_policy fold/seed differs")
    return diagnostics


def compare_repeated_oracle_inference(
    oracle_roots: Sequence[Path],
    standalone_root: Path,
    image_atol: float = 0.0,
) -> dict[str, Any]:
    """Compare standalone inference against repeated official mask behavior."""
    if image_atol != 0.0 or not np.isfinite(image_atol):
        raise ValueError("repeated inference parity requires finite image_atol=0.0")
    roots = _resolved_distinct_roots(oracle_roots)
    standalone_root = Path(standalone_root).resolve()
    if any(
        oracle_root == standalone_root
        or oracle_root in standalone_root.parents
        or standalone_root in oracle_root.parents
        for oracle_root in roots
    ):
        raise ValueError(
            "repeated inference parity requires standalone and oracle roots to be independent; "
            "artifact roots must not overlap"
        )

    all_roots = (*roots, standalone_root)
    manifests: list[dict[str, Any] | None] = []
    manifest_diagnostics: list[str] = []
    for index, root in enumerate(all_roots):
        manifest, errors = _load_manifest(root)
        manifests.append(manifest)
        prefix = f"artifact[{index}]"
        manifest_diagnostics.extend(f"{prefix}: {error}" for error in errors)
        if manifest is not None:
            policy = manifest.get("transform_policy")
            mode = policy.get("mode") if isinstance(policy, Mapping) else None
            if mode != "inference":
                raise ValueError(f"{prefix} must use inference mode")

    if all(manifest is not None for manifest in manifests):
        baseline = manifests[0]
        assert baseline is not None
        for index, manifest in enumerate(manifests[1:], start=1):
            assert manifest is not None
            manifest_diagnostics.extend(
                f"artifact[{index}]: {difference}"
                for difference in _manifest_value_differences(baseline, manifest)
            )
        manifest_diagnostics.extend(
            _inference_context_diagnostics(manifests, len(roots))
        )

    components: dict[str, dict[str, Any]] = {
        "manifest": {
            "status": "passed" if not manifest_diagnostics else "failed",
            "diagnostics": manifest_diagnostics,
        },
        "image": {"status": "failed", "diagnostics": []},
        "label": {"status": "failed", "diagnostics": []},
        "mask": {"status": "failed", "diagnostics": []},
    }
    diagnostics = list(manifest_diagnostics)

    arrays_by_name: dict[str, list[np.ndarray | None]] = {
        name: [] for name in REQUIRED_ARRAYS
    }
    for root, manifest in zip(all_roots, manifests):
        for name in REQUIRED_ARRAYS:
            if manifest is None:
                arrays_by_name[name].append(None)
                continue
            try:
                arrays_by_name[name].append(_load_array(root, manifest, name))
            except (OSError, ValueError) as error:
                arrays_by_name[name].append(None)
                message = f"artifact array '{name}' could not be loaded: {error}"
                components[name]["diagnostics"].append(message)
                diagnostics.append(message)

    for name in ("image", "label"):
        arrays = arrays_by_name[name]
        if all(array is not None for array in arrays):
            reference = arrays[0]
            assert reference is not None
            array_diagnostics: list[str] = []
            for index, candidate in enumerate(arrays[1:], start=1):
                assert candidate is not None
                passed, diagnostic = _array_matches(
                    name,
                    reference,
                    candidate,
                    image_atol=0.0,
                )
                if not passed and diagnostic is not None:
                    array_diagnostics.append(f"artifact[{index}]: {diagnostic}")
            components[name]["diagnostics"].extend(array_diagnostics)
            diagnostics.extend(array_diagnostics)
        elif not components[name]["diagnostics"]:
            components[name]["diagnostics"].append("manifest validation failed")
            diagnostics.append(f"{name}: manifest validation failed")
        components[name]["status"] = "passed" if not components[name]["diagnostics"] else "failed"

    oracle_masks = arrays_by_name["mask"][: len(roots)]
    standalone_mask = arrays_by_name["mask"][-1]
    pairwise: list[dict[str, int]] = []
    unstable_coordinates: list[list[int]] = []
    stable_mismatch_coordinates: list[list[int]] = []
    unobserved_coordinates: list[list[int]] = []
    if all(mask is not None for mask in (*oracle_masks, standalone_mask)):
        reference_mask = oracle_masks[0]
        assert reference_mask is not None
        schema_diagnostics: list[str] = []
        for index, candidate in enumerate((*oracle_masks[1:], standalone_mask), start=1):
            assert candidate is not None
            diagnostic = _mask_schema_diagnostic("mask", reference_mask, candidate)
            if diagnostic is not None:
                schema_diagnostics.append(f"artifact[{index}]: {diagnostic}")
        if not np.issubdtype(reference_mask.dtype, np.integer):
            schema_diagnostics.append(
                f"artifact[0]: mask: expected an integer dtype, got {reference_mask.dtype}"
            )
        components["mask"]["diagnostics"].extend(schema_diagnostics)
        diagnostics.extend(schema_diagnostics)
        if not schema_diagnostics:
            stack = np.stack([mask for mask in oracle_masks if mask is not None], axis=0)
            assert standalone_mask is not None
            pairwise = [
                {
                    "left_index": left,
                    "right_index": right,
                    "difference_count": int(np.count_nonzero(stack[left] != stack[right])),
                }
                for left in range(len(roots))
                for right in range(left + 1, len(roots))
            ]
            stable = np.all(stack == stack[0], axis=0)
            unstable = ~stable
            stable_mismatch = stable & (standalone_mask != stack[0])
            observed = np.any(stack == standalone_mask[None], axis=0)
            unobserved = unstable & ~observed
            unstable_coordinates = _coordinates(unstable)
            stable_mismatch_coordinates = _coordinates(stable_mismatch)
            unobserved_coordinates = _coordinates(unobserved)
            components["mask"]["diagnostics"].extend(
                [
                    *(
                        [
                            f"mask: standalone differs on {len(stable_mismatch_coordinates)} stable voxels"
                        ]
                        if stable_mismatch_coordinates
                        else []
                    ),
                    *(
                        [
                            f"mask: standalone uses unobserved labels on {len(unobserved_coordinates)} unstable voxels"
                        ]
                        if unobserved_coordinates
                        else []
                    ),
                ]
            )
    elif not components["mask"]["diagnostics"]:
        components["mask"]["diagnostics"].append("manifest validation failed")
        diagnostics.append("mask: manifest validation failed")

    if stable_mismatch_coordinates:
        diagnostics.append(
            f"mask: standalone differs on {len(stable_mismatch_coordinates)} stable voxels"
        )
    if unobserved_coordinates:
        diagnostics.append(
            f"mask: standalone uses unobserved labels on {len(unobserved_coordinates)} unstable voxels"
        )
    components["mask"]["status"] = "passed" if not components["mask"]["diagnostics"] else "failed"

    return {
        "parity_policy": REPEATED_INFERENCE_POLICY,
        "oracle_roots": [str(root) for root in roots],
        "oracle_repeat_count": len(roots),
        "oracle_pairwise_mask_difference_counts": pairwise,
        "oracle_unstable_voxel_count": len(unstable_coordinates),
        "oracle_unstable_voxel_coordinates": unstable_coordinates,
        "stable_mask_mismatch_count": len(stable_mismatch_coordinates),
        "stable_mask_mismatch_coordinates": stable_mismatch_coordinates,
        "unobserved_standalone_label_count": len(unobserved_coordinates),
        "unobserved_standalone_label_coordinates": unobserved_coordinates,
        "status": "passed" if not diagnostics else "failed",
        "run_state": RUN_STATE,
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
    parser.add_argument("--oracle-root", required=True, action="append", type=Path)
    parser.add_argument("--standalone-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--image-atol", type=float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if len(arguments.oracle_root) == 2:
        parser.error("provide one oracle root or at least three distinct oracle roots")
    if len(arguments.oracle_root) == 1:
        report = compare_artifacts(
            arguments.oracle_root[0],
            arguments.standalone_root,
            image_atol=arguments.image_atol,
        )
    else:
        report = compare_repeated_oracle_inference(
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
