"""Construct the official Trainer and verify its effective sampling setting."""

from __future__ import annotations

import json
import sys
from pathlib import Path


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
if str(EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTENSION_ROOT))

from nnUNetTrainerForeground50 import nnUNetTrainerForeground50


def main(plans_path: str, dataset_json_path: str) -> None:
    with Path(plans_path).open(encoding="utf-8") as plans_file:
        plans = json.load(plans_file)
    # nnUNetv2's official training entry point injects this transient flag
    # before constructing a Trainer. This check models a fresh fold-0 run.
    plans["continue_training"] = False
    with Path(dataset_json_path).open(encoding="utf-8") as dataset_file:
        dataset_json = json.load(dataset_file)

    trainer = nnUNetTrainerForeground50(plans, "2d", 0, dataset_json)
    assert trainer.oversample_foreground_percent == 0.50
    print(
        "EFFECTIVE_OVERSAMPLING_OK "
        f"value={trainer.oversample_foreground_percent:.2f}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: verify_effective_oversampling.py <plans.json> <dataset.json>"
        )
    main(sys.argv[1], sys.argv[2])
