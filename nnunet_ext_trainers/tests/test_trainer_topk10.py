"""Server-runtime contract for the TopK10 nnU-Net Trainer variant."""

from __future__ import annotations

from pathlib import Path
import sys


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
if str(EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTENSION_ROOT))

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnUNetTrainerTopK10 import nnUNetTrainerTopK10


def test_topk10_trainer_inherits_and_declares_percent() -> None:
    assert issubclass(nnUNetTrainerTopK10, nnUNetTrainer)
    assert nnUNetTrainerTopK10.TOPK_PERCENT == 10


if __name__ == "__main__":
    test_topk10_trainer_inherits_and_declares_percent()
