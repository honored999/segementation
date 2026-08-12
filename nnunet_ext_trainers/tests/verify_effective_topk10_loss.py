"""Construct the TopK10 Trainer and verify its effective official loss."""

from __future__ import annotations

import json
import sys
from pathlib import Path


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
if str(EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTENSION_ROOT))

from nnunetv2.training.loss.compound_losses import DC_and_topk_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnUNetTrainerTopK10 import nnUNetTrainerTopK10


def main(plans_path: str, dataset_json_path: str) -> None:
    with Path(plans_path).open(encoding="utf-8") as plans_file:
        plans = json.load(plans_file)
    plans["continue_training"] = False
    with Path(dataset_json_path).open(encoding="utf-8") as dataset_file:
        dataset_json = json.load(dataset_file)

    trainer = nnUNetTrainerTopK10(plans, "2d", 0, dataset_json)
    assert trainer.oversample_foreground_percent == 0.33
    loss = trainer._build_loss()
    assert isinstance(loss, DeepSupervisionWrapper)
    assert isinstance(loss.loss, DC_and_topk_loss)
    assert loss.loss.ce.k == 10
    print(
        "EFFECTIVE_TOPK10_OK "
        f"oversampling={trainer.oversample_foreground_percent:.2f} "
        f"k={loss.loss.ce.k}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: verify_effective_topk10_loss.py <plans.json> <dataset.json>"
        )
    main(sys.argv[1], sys.argv[2])
