from __future__ import annotations

import torch
from torch import nn

from .backbone import SpatialLogitHead, build_densenet121_deepseg_encoder
from .electronic_deepseg_decoder import ModifiedUNetDecoderStage


class ElectronicDenseNetDeepSegDecoder(SpatialLogitHead):
    context_module = "none"

    def __init__(self, encoder_weights: str | None = "imagenet") -> None:
        super().__init__()
        self.encoder, self.resolved_encoder = build_densenet121_deepseg_encoder(
            encoder_weights
        )
        self.resolved_encoder_name = self.resolved_encoder
        channels = self.encoder.out_channels
        self.decoder_stages = nn.ModuleList(
            (
                ModifiedUNetDecoderStage(channels[4], channels[3], 128),
                ModifiedUNetDecoderStage(128, channels[2], 64),
                ModifiedUNetDecoderStage(64, channels[1], 32),
                ModifiedUNetDecoderStage(32, channels[0], 32),
            )
        )
        self.segmentation_head = nn.Conv2d(32, 1, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.encoder(image.repeat(1, 3, 1, 1))
        feature = features[-1]
        for stage, skip in zip(
            self.decoder_stages,
            (features[3], features[2], features[1], features[0]),
        ):
            feature = stage(feature, skip)
        return self._restore(self.segmentation_head(feature), image)
