"""Isolated check for nnU-Net's external Trainer resolver.

Do not import nnUNetTrainerForeground50 here: doing so would invalidate the
test by preloading the class before the official resolver executes.
"""

from __future__ import annotations

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name


resolved = recursive_find_trainer_class_by_name("nnUNetTrainerForeground50")
assert issubclass(resolved, nnUNetTrainer)
assert resolved.__name__ == "nnUNetTrainerForeground50"
assert resolved.__module__ == "nnUNetTrainerForeground50"
assert resolved.OVERSAMPLE_FOREGROUND_PERCENT == 0.50
print(f"RESOLVER_OK class={resolved.__module__}.{resolved.__name__}")
