"""CLI for creating a portable optoelectronic H2Former initialization artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from standalone_nnunet2d.engine.optoelectronic_initialization import (
    create_optoelectronic_initialization,
)


def _reject_non_json_constant(value: str) -> Any:
    raise ValueError(f"configuration contains non-finite JSON constant {value!r}")


def _read_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_non_json_constant,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"failed to read JSON configuration: {path}") from error
    if not isinstance(value, dict):
        raise ValueError("configuration JSON root must be an object")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create an isolated optoelectronic H2Former initialization artifact "
            "from a matching ordinary H2Former checkpoint."
        )
    )
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _read_config(args.config)
    result = create_optoelectronic_initialization(
        args.source_checkpoint,
        config,
        args.output_root,
    )
    print(
        json.dumps(
            {
                "artifact_path": str(result.artifact_path),
                "report_path": str(result.report_path),
                "source_checkpoint_sha256": result.source_checkpoint_sha256,
                "evidence_class": "synthetic_engineering_initialization",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
