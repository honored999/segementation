"""Configurable-depth official PlainConvUNet Trainer with TopK10 and early stopping."""

from copy import deepcopy

import torch
from torch import nn
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import (
    nnUNetTrainerNoDeepSupervision,
)

from nnUNetTrainerMixins import EarlyStoppingMixin, TopK10LossMixin


class nnUNetTrainerPlainConvDepthTopK10EarlyStopping(
    EarlyStoppingMixin,
    TopK10LossMixin,
    nnUNetTrainerNoDeepSupervision,
):
    """Official encoder and UNetDecoder, retaining stages 0..encoder_last_stage."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        enabled = self.configuration_manager.configuration.get("enable_deep_supervision", False)
        if type(enabled) is not bool:
            raise ValueError("enable_deep_supervision must be a boolean")
        self.enable_deep_supervision = enabled

    def _get_deep_supervision_scales(self):
        scales = super()._get_deep_supervision_scales()
        if scales is None:
            return None
        last_stage = self.configuration_manager.configuration.get(
            "encoder_last_stage",
            self.configuration_manager.network_arch_init_kwargs["n_stages"] - 1,
        )
        return scales[:last_stage]

    def _build_loss(self):
        if self.enable_deep_supervision and len(self._get_deep_supervision_scales()) == 1:
            # One decoder output must retain a nonzero loss weight.
            self.enable_deep_supervision = False
            try:
                loss = super()._build_loss()
            finally:
                self.enable_deep_supervision = True
            return DeepSupervisionWrapper(loss, [1.0])
        return super()._build_loss()

    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = False,
    ) -> nn.Module:
        if len(configuration_manager.patch_size) != 2:
            raise ValueError("PlainConvDepth Trainer requires a 2D configuration")
        official_name = "dynamic_network_architectures.architectures.unet.PlainConvUNet"
        if configuration_manager.network_arch_class_name != official_name:
            raise ValueError("PlainConvDepth Trainer requires the official PlainConvUNet architecture")

        kwargs = deepcopy(configuration_manager.network_arch_init_kwargs)
        original_n_stages = kwargs.get("n_stages")
        if type(original_n_stages) is not int or original_n_stages < 2:
            raise ValueError("PlainConvDepth Trainer requires an integer n_stages of at least 2")

        last_stage = configuration_manager.configuration.get(
            "encoder_last_stage", original_n_stages - 1
        )
        if type(last_stage) is not int or not 1 <= last_stage < original_n_stages:
            raise ValueError(
                f"encoder_last_stage must be an integer in [1, {original_n_stages - 1}]"
            )
        keep = last_stage + 1

        for name in ("features_per_stage", "kernel_sizes", "strides"):
            values = kwargs.get(name)
            if not isinstance(values, (list, tuple)) or len(values) != original_n_stages:
                raise ValueError(
                    f"PlainConvDepth Trainer requires {original_n_stages} explicit {name} entries"
                )
            kwargs[name] = values[:keep]

        encoder_convs = kwargs.get("n_conv_per_stage")
        if type(encoder_convs) is int:
            pass
        elif isinstance(encoder_convs, (list, tuple)) and len(encoder_convs) == original_n_stages:
            kwargs["n_conv_per_stage"] = encoder_convs[:keep]
        else:
            raise ValueError(
                "PlainConvDepth Trainer requires a scalar or complete encoder-convolution sequence"
            )

        decoder_convs = kwargs.get("n_conv_per_stage_decoder")
        if type(decoder_convs) is int:
            pass
        elif isinstance(decoder_convs, (list, tuple)) and len(decoder_convs) == original_n_stages - 1:
            kwargs["n_conv_per_stage_decoder"] = decoder_convs[-(keep - 1) :]
        else:
            raise ValueError(
                "PlainConvDepth Trainer requires a scalar or complete decoder-convolution sequence"
            )
        kwargs["n_stages"] = keep

        network = get_network_from_plans(
            configuration_manager.network_arch_class_name,
            kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )
        if not isinstance(network, PlainConvUNet) or network.encoder.conv_op is not nn.Conv2d:
            raise ValueError("PlainConvDepth Trainer requires an official 2D PlainConvUNet")
        return network
