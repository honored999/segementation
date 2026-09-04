"""Explicit read-only server preflight command; never starts training."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from standalone_nnunet2d.tools.server_preflight import inspect_server_readiness


def main(arguments: Sequence[str] | None = None) -> int:
    """Print readiness JSON and return 0 only when all prerequisites exist."""
    parser = argparse.ArgumentParser(description="Read-only standalone nnU-Net server preflight")
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--preprocessed-root", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--device")
    parser.add_argument("--run", action="store_true")
    arguments_namespace = parser.parse_args(arguments)
    if arguments_namespace.run:
        parser.error("--run is not supported; dry_run never starts training")

    report = inspect_server_readiness(
        arguments_namespace.raw_root,
        arguments_namespace.preprocessed_root,
        arguments_namespace.results_root,
        device=arguments_namespace.device,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
