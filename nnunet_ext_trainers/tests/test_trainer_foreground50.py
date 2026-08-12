"""Server-runtime contract for the foreground-50 nnU-Net Trainer variant."""

from __future__ import annotations

from pathlib import Path
import sys


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
if str(EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTENSION_ROOT))

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnUNetTrainerForeground50 import nnUNetTrainerForeground50


def test_foreground50_trainer_inherits_and_declares_sampling_rate() -> None:
    assert issubclass(nnUNetTrainerForeground50, nnUNetTrainer)
    assert nnUNetTrainerForeground50.OVERSAMPLE_FOREGROUND_PERCENT == 0.50

if __name__ == "__main__":
    test_foreground50_trainer_inherits_and_declares_sampling_rate()
