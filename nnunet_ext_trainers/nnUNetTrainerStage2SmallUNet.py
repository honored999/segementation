"""Stage 2 nnU-Net Trainer using the fixed small 2D U-Net architecture."""

from __future__ import annotations

from typing import Any

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import (
    nnUNetTrainerNoDeepSupervision,
)

from small_unet_2d import SmallUNet2D


class nnUNetTrainerStage2SmallUNet(nnUNetTrainerNoDeepSupervision):
    @staticmethod
    def build_network_architecture(
        plans_manager: Any,
        configuration_manager: Any,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> SmallUNet2D:
        if len(configuration_manager.patch_size) != 2:
            raise ValueError(
                "nnUNetTrainerStage2SmallUNet requires a 2D configuration"
            )

        return SmallUNet2D(
            in_channels=num_input_channels,
            num_classes=num_output_channels,
        )
