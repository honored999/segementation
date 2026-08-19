"""Build verified Stage-1 five-fold OOF provenance."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..provenance import build_stage1_provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset501-raw", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--stage1-oof-dir", type=Path, required=True)
    for fold_index in range(5):
        parser.add_argument(f"--fold-{fold_index}-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fold_dirs = {
        fold_index: getattr(args, f"fold_{fold_index}_validation") for fold_index in range(5)
    }
    try:
        output = build_stage1_provenance(
            args.dataset501_raw,
            args.splits,
            args.stage1_oof_dir,
            fold_dirs,
            args.output,
        )
    except (OSError, ValueError) as error:
        print(f"Stage1 provenance build failed: {error}")
        return 2
    print(f"stage1_provenance={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
