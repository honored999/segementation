from __future__ import annotations

import torch
from torch import nn


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
