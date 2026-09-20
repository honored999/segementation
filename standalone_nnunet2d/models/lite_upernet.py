"""Shared lightweight UPerNet-style decoder for synthetic A/B variants."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


NormFactory = Callable[[int], nn.Module]


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        norm_factory: NormFactory,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            norm_factory(out_channels),
            nn.ReLU(inplace=True),
        )


class LiteUPerDecoder(nn.Module):
    def __init__(
        self,
        in_channels: Sequence[int],
        num_classes: int,
        fpn_channels: int = 64,
        pool_scales: Sequence[int] = (1, 2, 4),
        norm_factory: NormFactory = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        channels = tuple(in_channels)
        scales = tuple(pool_scales)
        if len(channels) != 4 or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in channels
        ):
            raise ValueError("in_channels must contain four positive integers")
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")
        if isinstance(fpn_channels, bool) or not isinstance(fpn_channels, int) or fpn_channels <= 0:
            raise ValueError("fpn_channels must be a positive integer")
        if not scales or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in scales
        ):
            raise ValueError("pool_scales must contain positive integers")
        if not callable(norm_factory):
            raise TypeError("norm_factory must be callable")

        self.in_channels = channels
        self.pool_scales = scales
        deepest_channels = channels[-1]
        self.ppm_branches = nn.ModuleList(
            nn.Sequential(
                nn.AdaptiveAvgPool2d((scale, scale)),
                nn.Conv2d(deepest_channels, fpn_channels, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
            )
            for scale in scales
        )
        self.ppm_bottleneck = ConvNormAct(
            deepest_channels + len(scales) * fpn_channels,
            fpn_channels,
            kernel_size=3,
            norm_factory=norm_factory,
        )
        self.lateral_projections = nn.ModuleList(
            ConvNormAct(value, fpn_channels, kernel_size=1, norm_factory=norm_factory)
            for value in channels[:-1]
        )
        self.refinements = nn.ModuleList(
            ConvNormAct(fpn_channels, fpn_channels, kernel_size=3, norm_factory=norm_factory)
            for _ in channels[:-1]
        )
        self.fusion = ConvNormAct(
            4 * fpn_channels,
            fpn_channels,
            kernel_size=3,
            norm_factory=norm_factory,
        )
        self.classifier = nn.Conv2d(fpn_channels, num_classes, kernel_size=1, bias=True)

    def _validate_features(
        self, features: Sequence[Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        values = tuple(features)
        if len(values) != 4:
            raise ValueError("LiteUPerDecoder requires exactly four feature maps")
        batch_size = None
        previous_spatial = None
        for index, (feature, expected_channels) in enumerate(
            zip(values, self.in_channels, strict=True)
        ):
            if not isinstance(feature, Tensor) or feature.ndim != 4:
                raise ValueError(f"feature {index} must be a BCHW tensor")
            if feature.shape[1] != expected_channels:
                raise ValueError(
                    f"feature {index} channels must be {expected_channels}, got {feature.shape[1]}"
                )
            if batch_size is None:
                batch_size = feature.shape[0]
            elif feature.shape[0] != batch_size:
                raise ValueError("all features must have the same batch size")
            spatial = tuple(feature.shape[-2:])
            if previous_spatial is not None and not (
                spatial[0] < previous_spatial[0] and spatial[1] < previous_spatial[1]
            ):
                raise ValueError("features must be ordered from highest to lowest spatial resolution")
            previous_spatial = spatial
        return values  # type: ignore[return-value]

    def forward(
        self,
        features: Sequence[Tensor],
        *,
        output_size: tuple[int, int],
    ) -> Tensor:
        values = self._validate_features(features)
        if (
            not isinstance(output_size, tuple)
            or len(output_size) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in output_size
            )
        ):
            raise ValueError("output_size must contain two positive integers")

        deepest = values[-1]
        ppm_values = [deepest]
        for branch in self.ppm_branches:
            pooled = branch(deepest)
            ppm_values.append(
                F.interpolate(pooled, size=deepest.shape[-2:], mode="bilinear", align_corners=False)
            )
        pyramid: list[Tensor | None] = [None, None, None, self.ppm_bottleneck(torch.cat(ppm_values, dim=1))]
        for index in reversed(range(3)):
            lateral = self.lateral_projections[index](values[index])
            top_down = F.interpolate(
                pyramid[index + 1],
                size=lateral.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            pyramid[index] = self.refinements[index](lateral + top_down)

        refined = [value for value in pyramid if value is not None]
        highest_size = refined[0].shape[-2:]
        fused_inputs = [
            F.interpolate(value, size=highest_size, mode="bilinear", align_corners=False)
            for value in refined
        ]
        logits = self.classifier(self.fusion(torch.cat(fused_inputs, dim=1)))
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


__all__ = ["LiteUPerDecoder"]
