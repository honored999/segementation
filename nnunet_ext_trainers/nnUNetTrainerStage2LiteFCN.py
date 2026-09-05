"""Stage 2 nnU-Net Trainer using the fixed 2D LiteFCN architecture."""

from __future__ import annotations

from typing import Any

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import (
    nnUNetTrainerNoDeepSupervision,
)

from lite_fcn_2d import LiteFCN2D


class nnUNetTrainerStage2LiteFCN(nnUNetTrainerNoDeepSupervision):
    def set_deep_supervision_enabled(self, enabled: bool) -> None:
        pass

    @staticmethod
    def build_network_architecture(
        plans_manager: Any,
        configuration_manager: Any,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> LiteFCN2D:
        if len(configuration_manager.patch_size) != 2:
            raise ValueError(
                "nnUNetTrainerStage2LiteFCN requires a 2D configuration"
            )

        return LiteFCN2D(
            in_channels=num_input_channels,
            num_classes=num_output_channels,
        )
