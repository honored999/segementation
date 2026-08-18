"""Compare original-volume Stage 1 and restored Stage 2 predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..evaluate import compare_full_volume_predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset501-raw", type=Path, required=True)
    parser.add_argument("--stage1-oof-dir", type=Path, required=True)
    parser.add_argument("--stage2-restored-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-case-count", type=int, default=95)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        csv_path, json_path = compare_full_volume_predictions(
            labels_dir=args.dataset501_raw / "labelsTr",
            stage1_dir=args.stage1_oof_dir,
            stage2_restored_dir=args.stage2_restored_dir,
            output_dir=args.output_dir,
            expected_case_count=args.expected_case_count,
        )
    except (OSError, ValueError) as error:
        print(f"comparison failed: {error}")
        return 2
    print(f"case_metrics_csv={csv_path}")
    print(f"summary_json={json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
