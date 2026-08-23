"""Lightweight multi-scale 2D FCN for the Stage 2 trainer."""

from __future__ import annotations

import torch
from torch import nn


class LightweightASPP2D(nn.Module):
    """A fixed-resolution four-branch multi-scale 2D segmentation network."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.ReLU(),
        )
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(32, 16, kernel_size=1, stride=1, dilation=1, padding=0),
                nn.Conv2d(32, 16, kernel_size=3, stride=1, dilation=1, padding=1),
                nn.Conv2d(32, 16, kernel_size=3, stride=1, dilation=2, padding=2),
                nn.Conv2d(32, 16, kernel_size=3, stride=1, dilation=4, padding=4),
            ]
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Conv2d(
            32,
            num_classes,
            kernel_size=1,
            stride=1,
            dilation=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.stem(x)
        multi_scale_features = [branch(features) for branch in self.branches]
        fused_features = self.fusion(torch.cat(multi_scale_features, dim=1))
        return self.classifier(fused_features)
