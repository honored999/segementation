"""Fit physical phase-only PSFs for an optoelectronic H2Former initialization."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from standalone_nnunet2d.engine.optoelectronic_psf_fitting import (
    OptoelectronicPSFFitConfig,
    fit_optoelectronic_h2former_psf,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit only active PhaseOnlyPSF theta maps in a physical-mode "
            "optoelectronic H2Former initialization artifact. This does not "
            "run segmentation training or access medical data."
        ),
    )
    parser.add_argument("--initialization-artifact", required=True, type=Path)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--fit-config", required=True, type=Path, help="JSON file using the strict PSF fit config schema")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default=None, help="Override fit-config device (for example cpu or cuda:0)")
    return parser


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if type(key) is not str:
            raise ValueError("JSON object keys must be strings")
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result

def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        with args.fit_config.open("r", encoding="utf-8") as handle:
            raw_config = json.load(
                handle,
                object_pairs_hook=_strict_json_object,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid JSON constant {value}")),
            )
        config = OptoelectronicPSFFitConfig.from_mapping(raw_config)
        if args.device is not None:
            config = replace(config, device=args.device)
        result = fit_optoelectronic_h2former_psf(
            args.initialization_artifact,
            args.source_checkpoint,
            config,
            args.output_root,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        parser.error(str(error))
    report = result.report
    print(json.dumps({
        "artifact_path": str(result.artifact_path),
        "report_path": str(result.report_path),
        "source_initialization_sha256": report["source_initialization_sha256"],
        "source_checkpoint_sha256": report["source_checkpoint_sha256"],
        "route_count": report["route_count"],
        "aggregate_fit_metrics": report["aggregate_metrics"],
        "fit_status": report["fit_status"],
    }, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
