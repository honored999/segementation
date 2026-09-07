"""Train the isolated ADN transform on Dataset501 fold-0 training DWIs."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from standalone_nnunet2d.brain_alignment.diagnostic import train_fold0


DEFAULT_SPLITS = Path(__file__).resolve().parents[1] / "reference" / "splits_final.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--splits-file", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    train_fold0(
        dataset_dir=arguments.dataset_dir,
        splits_file=arguments.splits_file,
        epochs=arguments.epochs,
        learning_rate=arguments.lr,
        device=torch.device(arguments.device),
        output_dir=arguments.output_dir,
        batch_size=arguments.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
