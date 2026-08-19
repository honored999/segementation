"""Generate the prediction-guided Dataset504 coarse-to-fine dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..dataset import EXPECTED_NUM_CASES, build_dataset504
from ..evaluate import _has_verified_formal_provenance


def _read_provenance(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Stage1 provenance file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Stage1 provenance is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("Stage1 provenance must be a JSON object")
    return payload


def _write_roi_manifest(destination: Path, provenance: dict[str, Any]) -> Path:
    manifest_path = destination / "manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Dataset504 builder did not write manifest.json: {manifest_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Dataset504 manifest is not valid JSON: {manifest_path}") from error
    if not isinstance(payload, dict):
        raise ValueError("Dataset504 manifest must be a JSON object")

    payload["stage1_provenance"] = provenance
    payload["formal_eligible"] = _has_verified_formal_provenance(
        provenance, expected_case_count=EXPECTED_NUM_CASES
    )
    serialized = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    manifest_path.write_text(serialized, encoding="utf-8")
    roi_manifest_path = destination / "roi_manifest.json"
    roi_manifest_path.write_text(serialized, encoding="utf-8")
    return roi_manifest_path


def generate_dataset504(
    *,
    dataset501_raw: Path,
    stage1_oof_dir: Path,
    output_root: Path,
    splits: Path,
    margin_px: int,
    min_roi_size: tuple[int, int],
    stage1_provenance: Path,
) -> Path:
    """Call the Dataset504 builder and record its Stage1 provenance."""
    if len(min_roi_size) != 2 or any(size < 1 for size in min_roi_size):
        raise ValueError("min_roi_size must contain two positive integers")
    provenance = _read_provenance(stage1_provenance)
    destination = build_dataset504(
        dataset501_raw,
        stage1_oof_dir,
        output_root,
        splits_path=splits,
        margin=margin_px,
        min_width=min_roi_size[0],
        min_height=min_roi_size[1],
    )
    return _write_roi_manifest(destination, provenance)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset501-raw", type=Path, required=True)
    parser.add_argument("--stage1-oof-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--margin-px", type=int, required=True)
    parser.add_argument(
        "--min-roi-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        required=True,
    )
    parser.add_argument("--stage1-provenance", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        roi_manifest = generate_dataset504(
            dataset501_raw=args.dataset501_raw,
            stage1_oof_dir=args.stage1_oof_dir,
            output_root=args.output_root,
            splits=args.splits,
            margin_px=args.margin_px,
            min_roi_size=tuple(args.min_roi_size),
            stage1_provenance=args.stage1_provenance,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Dataset504 generation failed: {error}")
        return 2
    print(f"roi_manifest={roi_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
