"""Shared fixed-2D network construction for Stage 2 custom Trainers."""

from __future__ import annotations

from typing import Any, ClassVar

from torch import nn

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import (
    nnUNetTrainerNoDeepSupervision,
)


class nnUNetTrainerStage2Base(nnUNetTrainerNoDeepSupervision):
    """Common Stage 2 contract for independent fixed 2D network Trainers."""

    network_class: ClassVar[type[nn.Module]]

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager: Any,
        configuration_manager: Any,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        if len(configuration_manager.patch_size) != 2:
            raise ValueError(f"{cls.__name__} requires a 2D configuration")

        return cls.network_class(
            in_channels=num_input_channels,
            num_classes=num_output_channels,
        )
