"""Foreground50 Trainer with full-resolution residual feature refinement."""

from __future__ import annotations

from torch import nn

from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager,
)

from detail_refinement import attach_detail_refinement
from nnUNetTrainerForeground50 import nnUNetTrainerForeground50


class nnUNetTrainerForeground50DetailRefine(nnUNetTrainerForeground50):
    """Change only the final full-resolution feature-to-logit path."""

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        network = nnUNetTrainerForeground50.build_network_architecture(
            plans_manager,
            configuration_manager,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )
        return attach_detail_refinement(network)
