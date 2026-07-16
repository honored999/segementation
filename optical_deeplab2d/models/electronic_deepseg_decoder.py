from __future__ import annotations

import torch
from torch import nn

from .backbone import SpatialLogitHead, build_deepseg_modules


class ModifiedUNetDecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feature: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        feature = nn.functional.interpolate(
            feature,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.refine(torch.cat((feature, skip), dim=1))


class ElectronicDeepSegDecoder(SpatialLogitHead):
    def __init__(
        self,
        encoder_name: str = "mobilenet_v2",
        encoder_weights: str | None = "imagenet",
    ) -> None:
        super().__init__()
        self.encoder, self.aspp, self.resolved_encoder = build_deepseg_modules(
            encoder_name, encoder_weights
        )
        self.resolved_encoder_name = self.resolved_encoder
        channels = self.encoder.out_channels
        self.decoder_stages = nn.ModuleList(
            (
                ModifiedUNetDecoderStage(256, channels[3], 128),
                ModifiedUNetDecoderStage(128, channels[2], 64),
                ModifiedUNetDecoderStage(64, channels[1], 32),
                ModifiedUNetDecoderStage(32, channels[0], 32),
            )
        )
        self.segmentation_head = nn.Conv2d(32, 1, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        batch_size = image.shape[0]
        encoded_image = image.repeat(1, 3, 1, 1)
        if self.training and batch_size == 1:
            encoded_image = encoded_image.repeat(2, 1, 1, 1)
        features = self.encoder(encoded_image)
        feature = self.aspp(features[-1])
        for stage, skip in zip(
            self.decoder_stages,
            (features[3], features[2], features[1], features[0]),
        ):
            feature = stage(feature, skip)
        return self._restore(self.segmentation_head(feature)[:batch_size], image)
