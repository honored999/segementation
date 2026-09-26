"""Cooperative mixins and the official-plan UPerNet architecture adapter."""

from __future__ import annotations

from numbers import Integral
from typing import Sequence, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch._dynamo import OptimizedModule

from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.simple_conv_blocks import (
    ConvDropoutNormReLU,
    StackedConvBlocks,
)
from nnunetv2.training.loss.compound_losses import DC_and_topk_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans


class EarlyStoppingMixin:
    """Add EMA pseudo-Dice early stopping without changing nnU-Net hooks."""

    MAX_EPOCHS = 1000
    MIN_TRAINING_EPOCHS = 300
    PATIENCE = 100
    MIN_DELTA = 1e-4

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = self.MAX_EPOCHS
        self._early_stopping_best_ema: float | None = None
        self._epochs_without_improvement = 0
        self._early_stopping_triggered = False

    def _update_early_stopping(self) -> None:
        monitored_ema = float(self.logger.get_value("ema_fg_dice", step=-1))
        completed_epochs = self.current_epoch + 1

        if self._early_stopping_best_ema is None:
            self._early_stopping_best_ema = monitored_ema
            return

        if monitored_ema >= self._early_stopping_best_ema + self.MIN_DELTA:
            self._early_stopping_best_ema = monitored_ema
            self._epochs_without_improvement = 0
            return

        if completed_epochs <= self.MIN_TRAINING_EPOCHS:
            return

        self._epochs_without_improvement += 1
        if self._epochs_without_improvement >= self.PATIENCE:
            self._early_stopping_triggered = True
            self.print_to_log_file(
                "Early stopping at epoch "
                f"{completed_epochs}: best monitored EMA Dice="
                f"{self._early_stopping_best_ema:.6f}, patience={self.PATIENCE}, "
                f"min_delta={self.MIN_DELTA}"
            )

    def on_epoch_end(self) -> None:
        self._update_early_stopping()
        super().on_epoch_end()

    def _early_stopping_checkpoint_state(self) -> dict:
        return {
            "best_monitored_ema_dice": self._early_stopping_best_ema,
            "epochs_without_improvement": self._epochs_without_improvement,
            "triggered": self._early_stopping_triggered,
        }

    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                if self.is_ddp:
                    mod = self.network.module
                else:
                    mod = self.network
                if isinstance(mod, OptimizedModule):
                    mod = mod._orig_mod

                checkpoint = {
                    "network_weights": mod.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "grad_scaler_state": self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    "logging": self.logger.get_checkpoint(),
                    "_best_ema": self._best_ema,
                    "current_epoch": self.current_epoch + 1,
                    "init_args": self.my_init_kwargs,
                    "trainer_name": self.__class__.__name__,
                    "inference_allowed_mirroring_axes": self.inference_allowed_mirroring_axes,
                    "early_stopping_state": self._early_stopping_checkpoint_state(),
                }
                torch.save(checkpoint, filename)
            else:
                self.print_to_log_file("No checkpoint written, checkpointing is disabled")

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)
        else:
            checkpoint = filename_or_checkpoint

        new_state_dict = {}
        for key, value in checkpoint["network_weights"].items():
            normalized_key = key
            if normalized_key not in self.network.state_dict().keys() and normalized_key.startswith("module."):
                normalized_key = normalized_key[7:]
            new_state_dict[normalized_key] = value

        self.my_init_kwargs = checkpoint["init_args"]
        self.current_epoch = checkpoint["current_epoch"]
        self.logger.load_checkpoint(checkpoint["logging"])
        self._best_ema = checkpoint["_best_ema"]
        self.inference_allowed_mirroring_axes = (
            checkpoint["inference_allowed_mirroring_axes"]
            if "inference_allowed_mirroring_axes" in checkpoint
            else self.inference_allowed_mirroring_axes
        )

        if self.is_ddp:
            if isinstance(self.network.module, OptimizedModule):
                self.network.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.module.load_state_dict(new_state_dict)
        else:
            if isinstance(self.network, OptimizedModule):
                self.network._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.load_state_dict(new_state_dict)
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.grad_scaler is not None and checkpoint["grad_scaler_state"] is not None:
            self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state"])

        state = checkpoint.get("early_stopping_state")
        if state is None:
            self._early_stopping_best_ema = self._best_ema
            self._epochs_without_improvement = 0
            self._early_stopping_triggered = False
        else:
            self._early_stopping_best_ema = state.get("best_monitored_ema_dice")
            self._epochs_without_improvement = state.get("epochs_without_improvement", 0)
            self._early_stopping_triggered = state.get("triggered", False)

    def run_training(self):
        self.on_train_start()
        if self._early_stopping_triggered:
            self.on_train_end()
            return

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()
            if self._early_stopping_triggered:
                break

        self.on_train_end()


class TopK10LossMixin:
    """Use the official Dice plus TopK CE loss while preserving DS behavior."""

    TOPK_PERCENT = 10

    def _build_loss(self):
        if self.label_manager.has_regions:
            return super()._build_loss()

        loss = DC_and_topk_loss(
            {
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5,
                "do_bg": False,
                "ddp": self.is_ddp,
            },
            {"k": self.TOPK_PERCENT},
            weight_ce=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
        )
        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            weights = np.array(
                [1 / (2**index) for index in range(len(self._get_deep_supervision_scales()))]
            )
            weights[-1] = 1e-6 if self.is_ddp and not self._do_i_compile() else 0
            loss = DeepSupervisionWrapper(loss, weights / weights.sum())
        return loss


def _as_pair(value: Sequence[int], *, name: str) -> tuple[int, int]:
    values = tuple(value)
    if len(values) != 2 or any(
        isinstance(item, bool) or not isinstance(item, Integral) or item <= 0 for item in values
    ):
        raise ValueError(f"{name} must contain two positive integers")
    return int(values[0]), int(values[1])


def select_upernet_feature_indices(
    strides: Sequence[Sequence[int]], n_stages: int | None = None
) -> tuple[int, int, int, int]:
    """Choose four strictly downsampled encoder stages from plan strides.

    The current eight-stage Dataset501 plan yields ``(1, 3, 5, 7)``. The
    selection is derived from the available cumulative 2D resolutions and is
    evenly spread over the available levels for other valid plans.
    """

    if n_stages is None:
        n_stages = len(strides)
    if n_stages != len(strides):
        raise ValueError("n_stages must match the number of stride entries")

    cumulative = (1, 1)
    previous_scale = (1, 1)
    available: list[int] = []
    for index, stride_value in enumerate(strides):
        stride = _as_pair(stride_value, name=f"strides[{index}]")
        cumulative = (cumulative[0] * stride[0], cumulative[1] * stride[1])
        if cumulative[0] > previous_scale[0] and cumulative[1] > previous_scale[1]:
            available.append(index)
            previous_scale = cumulative

    if len(available) < 4:
        raise ValueError("UPerNet requires four strictly downsampled encoder feature stages")

    positions = [round(position * (len(available) - 1) / 3) for position in range(4)]
    selected = tuple(available[position] for position in positions)
    if len(set(selected)) != 4:
        raise ValueError("UPerNet feature-stage selection did not produce four distinct stages")
    return selected  # type: ignore[return-value]


class _ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        norm_factory,
        *,
        bias: bool,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=bias,
            ),
            norm_factory(out_channels),
            nn.ReLU(inplace=True),
        )


def _encoder_spatial_geometry(encoder: PlainConvEncoder) -> tuple[tuple[tuple, ...], ...]:
    """Snapshot actual encoder convolutions in forward order, without registering the encoder twice."""
    stages = []
    for stage in encoder.stages:
        operations = []
        for module in stage.modules():
            if isinstance(module, nn.Conv2d):
                if not isinstance(module.padding, tuple):
                    raise ValueError("UPerNet requires numeric Conv2d padding")
                operations.append((module.kernel_size, module.stride, module.padding, module.dilation))
            elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
                raise ValueError("UPerNet decoder count requires conv-pooling encoder stages")
            elif not isinstance(module, (
                nn.Sequential, StackedConvBlocks, ConvDropoutNormReLU,
                nn.InstanceNorm2d, nn.BatchNorm2d, nn.Dropout2d,
                nn.ReLU, nn.LeakyReLU, nn.Identity,
            )):
                raise ValueError(f"unsupported encoder spatial operation: {type(module).__name__}")
        if not operations:
            raise ValueError("UPerNet encoder stage has no Conv2d")
        stages.append(tuple(operations))
    return tuple(stages)


class UPerNetDecoder(nn.Module):
    """Four-level 2D UPerNet decoder used by the official Trainer variants."""

    def __init__(
        self,
        in_channels: Sequence[int],
        num_classes: int,
        norm_factory,
        *,
        conv_bias: bool,
        fpn_channels: int = 128,
        pool_scales: Sequence[int] = (1, 2, 4),
        encoder_spatial_geometry: tuple[tuple[tuple, ...], ...] | None = None,
        selected_feature_indices: tuple[int, int, int, int] | None = None,
    ) -> None:
        super().__init__()
        channels = tuple(in_channels)
        scales = tuple(pool_scales)
        if len(channels) != 4 or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in channels):
            raise ValueError("in_channels must contain four positive integers")
        if not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")
        if fpn_channels != 128:
            raise ValueError("official UPerNet uses fpn_channels=128")
        if scales != (1, 2, 4):
            raise ValueError("official UPerNet uses pool_scales=(1, 2, 4)")

        self.in_channels = channels
        self.encoder_spatial_geometry = encoder_spatial_geometry
        self.selected_feature_indices = selected_feature_indices
        self.fpn_channels = fpn_channels
        self.pool_scales = scales
        deepest_channels = channels[-1]
        self.ppm_branches = nn.ModuleList(
            nn.Sequential(
                nn.AdaptiveAvgPool2d((scale, scale)),
                nn.Conv2d(deepest_channels, fpn_channels, kernel_size=1, bias=conv_bias),
                nn.ReLU(inplace=True),
            )
            for scale in scales
        )
        self.ppm_bottleneck = _ConvNormAct(
            deepest_channels + len(scales) * fpn_channels,
            fpn_channels,
            kernel_size=3,
            norm_factory=norm_factory,
            bias=conv_bias,
        )
        self.lateral_projections = nn.ModuleList(
            _ConvNormAct(value, fpn_channels, kernel_size=1, norm_factory=norm_factory, bias=conv_bias)
            for value in channels[:-1]
        )
        self.refinements = nn.ModuleList(
            _ConvNormAct(fpn_channels, fpn_channels, kernel_size=3, norm_factory=norm_factory, bias=conv_bias)
            for _ in channels[:-1]
        )
        self.fusion = _ConvNormAct(
            4 * fpn_channels,
            fpn_channels,
            kernel_size=3,
            norm_factory=norm_factory,
            bias=conv_bias,
        )
        self.classifier = nn.Conv2d(fpn_channels, num_classes, kernel_size=1, bias=True)
        self.deep_supervision = False

    def _validate_features(self, features: Sequence[Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        values = tuple(features)
        if len(values) != 4:
            raise ValueError("UPerNetDecoder requires exactly four feature maps")
        batch_size: int | None = None
        previous_spatial: tuple[int, int] | None = None
        for index, (feature, expected_channels) in enumerate(zip(values, self.in_channels, strict=True)):
            if not isinstance(feature, Tensor) or feature.ndim != 4:
                raise ValueError(f"feature {index} must be a BCHW tensor")
            if feature.shape[1] != expected_channels:
                raise ValueError(
                    f"feature {index} channels must be {expected_channels}, got {feature.shape[1]}"
                )
            if batch_size is None:
                batch_size = feature.shape[0]
            elif feature.shape[0] != batch_size:
                raise ValueError("all UPerNet features must have the same batch size")
            spatial = (int(feature.shape[-2]), int(feature.shape[-1]))
            if previous_spatial is not None and not (
                spatial[0] < previous_spatial[0] and spatial[1] < previous_spatial[1]
            ):
                raise ValueError("UPerNet features must have strictly decreasing spatial resolutions")
            previous_spatial = spatial
        return values  # type: ignore[return-value]

    def forward(self, features: Sequence[Tensor], *, output_size: tuple[int, int]) -> Tensor:
        values = self._validate_features(features)
        output_size = _as_pair(output_size, name="output_size")

        deepest = values[-1]
        ppm_values = [deepest]
        for branch in self.ppm_branches:
            pooled = branch(deepest)
            ppm_values.append(
                torch.nn.functional.interpolate(
                    pooled,
                    size=deepest.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            )
        pyramid: list[Tensor | None] = [None, None, None, self.ppm_bottleneck(torch.cat(ppm_values, dim=1))]
        for index in reversed(range(3)):
            lateral = self.lateral_projections[index](values[index])
            top_down = torch.nn.functional.interpolate(
                pyramid[index + 1],
                size=lateral.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            pyramid[index] = self.refinements[index](lateral + top_down)

        refined = [value for value in pyramid if value is not None]
        highest_size = refined[0].shape[-2:]
        fused_inputs = [
            torch.nn.functional.interpolate(value, size=highest_size, mode="bilinear", align_corners=False)
            for value in refined
        ]
        logits = self.classifier(self.fusion(torch.cat(fused_inputs, dim=1)))
        return torch.nn.functional.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)

    def compute_conv_feature_map_size(self, input_size: Sequence[int]) -> np.int64:
        spatial = _as_pair(input_size, name="input_size")
        if self.encoder_spatial_geometry is None or self.selected_feature_indices is None:
            raise ValueError("decoder feature-map count requires encoder spatial geometry and selected indices")

        selected_sizes = []
        for index, operations in enumerate(self.encoder_spatial_geometry):
            for kernel, stride, padding, dilation in operations:
                spatial = tuple(
                    (size + 2 * pad - dil * (width - 1) - 1) // step + 1
                    for size, pad, dil, width, step in zip(spatial, padding, dilation, kernel, stride, strict=True)
                )
            if index in self.selected_feature_indices:
                selected_sizes.append(spatial)
        if len(selected_sizes) != 4 or any(min(size) < 1 for size in selected_sizes):
            raise ValueError("input_size produces invalid selected encoder feature maps")

        def area(size: tuple[int, int]) -> np.int64:
            return np.int64(size[0]) * np.int64(size[1])

        highest, middle_a, middle_b, deepest = selected_sizes
        width = np.int64(self.fpn_channels)
        total = sum((width * np.int64(scale) * np.int64(scale) for scale in self.pool_scales), np.int64(0))
        total += width * area(deepest)  # PPM bottleneck
        for size in (highest, middle_a, middle_b):
            total += width * area(size) * np.int64(2)  # lateral projection and refinement
        total += width * area(highest)  # fusion
        total += np.int64(self.classifier.out_channels) * area(highest)
        return np.int64(total)


class _PlainConvUNetUPerNet(nn.Module):
    def __init__(
        self,
        encoder: PlainConvEncoder,
        selected_feature_indices: tuple[int, int, int, int],
        num_output_channels: int,
        *,
        norm_op,
        norm_op_kwargs: dict | None,
        conv_bias: bool,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.selected_feature_indices = selected_feature_indices
        channels = tuple(int(encoder.output_channels[index]) for index in selected_feature_indices)

        def norm_factory(num_channels: int) -> nn.Module:
            if norm_op is None:
                return nn.Identity()
            return norm_op(num_channels, **dict(norm_op_kwargs or {}))

        self.decoder = UPerNetDecoder(
            channels,
            num_output_channels,
            norm_factory,
            conv_bias=conv_bias,
            encoder_spatial_geometry=_encoder_spatial_geometry(encoder),
            selected_feature_indices=selected_feature_indices,
        )

    def forward(self, image: Tensor) -> Tensor:
        encoder_outputs = self.encoder(image)
        selected = tuple(encoder_outputs[index] for index in self.selected_feature_indices)
        return self.decoder(selected, output_size=(int(image.shape[-2]), int(image.shape[-1])))

    def compute_conv_feature_map_size(self, input_size: Sequence[int]) -> np.int64:
        _as_pair(input_size, name="input_size")
        return np.int64(self.encoder.compute_conv_feature_map_size(list(input_size))) + self.decoder.compute_conv_feature_map_size(
            input_size
        )


class UPerNetArchitectureMixin:
    """Replace only the official PlainConvUNet decoder with UPerNet."""

    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        if len(configuration_manager.patch_size) != 2:
            raise ValueError("UPerNet external Trainer requires a 2D configuration")
        if enable_deep_supervision:
            raise ValueError("UPerNet external Trainer does not support deep supervision")

        official_network = get_network_from_plans(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=False,
        )
        if not isinstance(official_network, PlainConvUNet):
            raise ValueError("UPerNet external Trainer requires the official PlainConvUNet architecture")
        if official_network.encoder.conv_op is not nn.Conv2d:
            raise ValueError("UPerNet external Trainer requires a 2D Conv2d encoder")

        encoder = official_network.encoder
        selected_feature_indices = select_upernet_feature_indices(encoder.strides, len(encoder.stages))
        model = _PlainConvUNetUPerNet(
            encoder,
            selected_feature_indices,
            num_output_channels,
            norm_op=encoder.norm_op,
            norm_op_kwargs=encoder.norm_op_kwargs,
            conv_bias=encoder.conv_bias,
        )
        model.decoder.apply(PlainConvUNet.initialize)
        return model

    def set_deep_supervision_enabled(self, enabled: bool) -> None:
        """Keep UPerNet single-output across train and validation lifecycle hooks."""

        return None


__all__ = [
    "EarlyStoppingMixin",
    "TopK10LossMixin",
    "UPerNetArchitectureMixin",
    "UPerNetDecoder",
    "select_upernet_feature_indices",
]
