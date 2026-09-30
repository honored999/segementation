"""H2Former encoder with an isolated lightweight UPerNet decoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from standalone_nnunet2d.models.h2former import H2Former
from standalone_nnunet2d.models.lite_upernet import LiteUPerDecoder


class H2FormerLiteUPerNet(H2Former):
    """Single-output H2Former decoder ablation with the baseline encoder."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        image_size: int = 512,
        fpn_channels: int = 64,
        pool_scales: tuple[int, ...] = (1, 2, 4),
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes, image_size=image_size)
        del self.decode4
        del self.decode3
        del self.decode2
        del self.decode0
        self.decoder = LiteUPerDecoder(
            in_channels=(64, 128, 256, 512),
            num_classes=num_classes,
            fpn_channels=fpn_channels,
            pool_scales=pool_scales,
            norm_factory=nn.BatchNorm2d,
        )
        self.deep_supervision = False
        self.last_feature_shapes: tuple[tuple[int, ...], ...] = ()

    def forward(self, x: Tensor) -> Tensor:
        features = self.forward_features(x)
        self.last_feature_shapes = tuple(tuple(feature.shape) for feature in features)
        return self.decoder(features, output_size=(int(x.shape[-2]), int(x.shape[-1])))


__all__ = ["H2FormerLiteUPerNet"]
