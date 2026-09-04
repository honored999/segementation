"""Isolated external-resolver check for nnUNetTrainerTopK10.

Do not import the Trainer before calling the official resolver.
"""

from __future__ import annotations

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name


resolved = recursive_find_trainer_class_by_name("nnUNetTrainerTopK10")
assert issubclass(resolved, nnUNetTrainer)
assert resolved.__name__ == "nnUNetTrainerTopK10"
assert resolved.__module__ == "nnUNetTrainerTopK10"
assert resolved.TOPK_PERCENT == 10
print(f"TOPK10_RESOLVER_OK class={resolved.__module__}.{resolved.__name__}")
