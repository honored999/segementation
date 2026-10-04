"""Plan-configured encoder selection with a network-level checkpoint identity."""
from __future__ import annotations

from copy import deepcopy
import json
from typing import Union

import numpy as np
import torch
from torch import nn
from torch._dynamo import OptimizedModule
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

from nnUNetTrainerMixins import (
    _PlainConvUNetUPerNet, _as_pair, select_upernet_feature_indices,
    upernet_cumulative_scales, validate_upernet_feature_indices,
)
from nnUNetTrainerUPerNetTopK10EarlyStopping import nnUNetTrainerUPerNetTopK10EarlyStopping


def resolve_upernet_selection(configuration_manager) -> tuple[int, ...]:
    """Read only the resolved configuration, never architecture kwargs or env vars."""
    if len(configuration_manager.patch_size) != 2:
        raise ValueError("UPerNet external Trainer requires a 2D configuration")
    if configuration_manager.network_arch_class_name != (
        "dynamic_network_architectures.architectures.unet.PlainConvUNet"
    ):
        raise ValueError("UPerNet external Trainer requires the official PlainConvUNet architecture")
    kwargs = configuration_manager.network_arch_init_kwargs
    if "upernet_feature_indices" in kwargs:
        raise ValueError("upernet_feature_indices belongs in configuration, not architecture kwargs")
    n_stages = kwargs["n_stages"]
    scales = upernet_cumulative_scales(kwargs["strides"], n_stages)
    configuration = configuration_manager.configuration
    if "upernet_feature_indices" in configuration:
        indices = configuration["upernet_feature_indices"]
    else:
        strides = [((value, value) if isinstance(value, int) else value) for value in kwargs["strides"]]
        indices = select_upernet_feature_indices(strides, n_stages)
    # Validate even the automatic result with the same strict dimensional contract.
    return validate_upernet_feature_indices(indices, kwargs["strides"], len(scales))


class _SelectedStagesUPerNet(_PlainConvUNetUPerNet):
    """Root pre-hook enforces identity before recursive parameter copying.

    This also covers official predictor first-fold and later-fold weight loads,
    and loading this model inside a parent module (the hook receives its prefix).
    Only the new Trainer uses this class; legacy states never gain extra_state.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        scales = upernet_cumulative_scales(self.encoder.strides, len(self.encoder.stages))
        self._selection_identity = {
            "version": 1,
            "indices": list(self.selected_feature_indices),
            "n_levels": len(self.selected_feature_indices),
            "n_encoder_stages": len(self.encoder.stages),
            "scales": [list(scales[i]) for i in self.selected_feature_indices],
            "in_channels": list(self.decoder.in_channels),
            "num_classes": self.decoder.classifier.out_channels,
            "fpn_channels": self.decoder.fpn_channels,
            "pool_scales": list(self.decoder.pool_scales),
        }
        self.register_load_state_dict_pre_hook(self._check_load_identity)

    def get_extra_state(self) -> dict:
        return deepcopy(self._selection_identity)

    def validate_state_identity(self, state_dict, prefix: str = "") -> None:
        identity = state_dict.get(prefix + "_extra_state")
        # Exact primitive type checking rejects equality-spoofing bool/float values.
        def same_types_and_values(expected, actual):
            if type(expected) is not type(actual):
                return False
            if isinstance(expected, dict):
                return expected.keys() == actual.keys() and all(
                    same_types_and_values(value, actual[key]) for key, value in expected.items()
                )
            if isinstance(expected, list):
                return len(expected) == len(actual) and all(
                    same_types_and_values(a, b) for a, b in zip(expected, actual)
                )
            return expected == actual
        if not same_types_and_values(self._selection_identity, identity):
            raise ValueError("UPerNet selected-stage identity missing, corrupted or mismatched; "
                             f"expected {self._selection_identity}, got {identity}")

    def _check_load_identity(self, module, state_dict, prefix, *args) -> None:
        self.validate_state_identity(state_dict, prefix)

    def set_extra_state(self, state) -> None:
        # Never let loading replace the identity derived from current plans.
        self.validate_state_identity({"_extra_state": state})

    def compute_conv_feature_map_size(self, input_size) -> np.int64:
        spatial = _as_pair(input_size, name="input_size")
        total = np.int64(0)
        # Library floor-based encoder accounting is not exact on odd Conv2d inputs.
        # Count all actual encoder convolutions, including unselected stages.
        for stage in self.encoder.stages:
            for conv in stage.modules():
                if isinstance(conv, nn.Conv2d):
                    spatial = tuple(
                        (size + 2*pad - dil*(width-1) - 1)//step + 1
                        for size, pad, dil, width, step in zip(
                            spatial, conv.padding, conv.dilation, conv.kernel_size, conv.stride, strict=True
                        )
                    )
                    if min(spatial) < 1:
                        raise ValueError("input_size produces invalid encoder feature maps")
                    total += np.int64(conv.out_channels) * np.prod(spatial, dtype=np.int64)
        return total + self.decoder.compute_conv_feature_map_size(input_size)


def _unwrapped_network(network):
    if isinstance(network, nn.parallel.DistributedDataParallel):
        network = network.module
    if isinstance(network, OptimizedModule):
        network = network._orig_mod
    return network


class nnUNetTrainerUPerNetSelectedStagesTopK10EarlyStopping(nnUNetTrainerUPerNetTopK10EarlyStopping):
    """Only selection/identity differs from the existing TopK10/early-stop Trainer."""

    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        if enable_deep_supervision:
            raise ValueError("UPerNet external Trainer does not support deep supervision")
        indices = resolve_upernet_selection(configuration_manager)
        official = get_network_from_plans(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels, num_output_channels, allow_init=True, deep_supervision=False,
        )
        if not isinstance(official, PlainConvUNet) or official.encoder.conv_op is not nn.Conv2d:
            raise ValueError("UPerNet requires the official 2D PlainConvUNet encoder")
        validate_upernet_feature_indices(indices, official.encoder.strides, len(official.encoder.stages))
        encoder = official.encoder
        model = _SelectedStagesUPerNet(
            encoder, indices, num_output_channels,
            norm_op=encoder.norm_op, norm_op_kwargs=encoder.norm_op_kwargs, conv_bias=encoder.conv_bias,
        )
        model.decoder.apply(PlainConvUNet.initialize)
        return model

    def initialize(self) -> None:
        first = not self.was_initialized
        super().initialize()
        if first and self.is_ddp:
            wrapper = self.network
            model = _unwrapped_network(wrapper)
            if model.selected_feature_indices[-1] < len(model.encoder.stages) - 1:
                # Official initialize already performed compile/SyncBatchNorm and
                # built the optimizer. Trailing stages execute but do not reach
                # the loss: DDP must detect their genuinely unused parameters.
                # Rebuild only the reducer before any forward, keeping the same
                # module/parameters/device placement/process group and defaults.
                ddp_kwargs = {
                    "device_ids": wrapper.device_ids,
                    "output_device": wrapper.output_device,
                    "process_group": wrapper.process_group,
                    "find_unused_parameters": True,
                }
                self.network = wrapper.module
                del wrapper  # release the old reducer before attaching the new one
                self.network = nn.parallel.DistributedDataParallel(self.network, **ddp_kwargs)
                self.print_to_log_file("UPerNet DDP: find_unused_parameters=True "
                                       "for unselected trailing encoder stages")
        if first:
            self.print_to_log_file("UPerNet effective selected-stage identity: " +
                                   json.dumps(_unwrapped_network(self.network).get_extra_state(), sort_keys=True))

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        checkpoint = (torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)
                      if isinstance(filename_or_checkpoint, str) else filename_or_checkpoint)
        if not self.was_initialized:
            self.initialize()
        model = _unwrapped_network(self.network)
        weights = checkpoint["network_weights"]
        # Match inherited loader's DDP prefix normalization without changing its loop.
        normalized = {key[7:] if key.startswith("module.") else key: value for key, value in weights.items()}
        model.validate_state_identity(normalized)
        # In DDP the inherited loader sees module-prefixed wrapper keys and can
        # retain that prefix before loading the inner model. Pass precisely the
        # validated inner-model state, without mutating the caller's checkpoint.
        checkpoint = dict(checkpoint)
        checkpoint["network_weights"] = normalized
        super().load_checkpoint(checkpoint)
