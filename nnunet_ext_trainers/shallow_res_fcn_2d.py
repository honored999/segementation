"""Shallow fixed-resolution residual FCN for the Stage 2 trainer."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Two same-width dilated convolutions with a post-addition ReLU."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.dilation = dilation
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            dilation=dilation,
            padding=dilation,
        )
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            dilation=dilation,
            padding=dilation,
        )
        self.relu2 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu1(self.conv1(x))
        x = self.conv2(x)
        return self.relu2(x + residual)


class ShallowResFCN2D(nn.Module):
    """A 32-channel, no-downsampling residual FCN that returns raw logits."""

    HIDDEN_CHANNELS = 32
    DILATIONS = (1, 2, 4)

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        hidden_channels = self.HIDDEN_CHANNELS
        self.stem = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.stem_relu = nn.ReLU()
        self.residual_blocks = nn.ModuleList(
            ResidualBlock(hidden_channels, dilation=dilation)
            for dilation in self.DILATIONS
        )
        self.classifier = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_relu(self.stem(x))
        for block in self.residual_blocks:
            x = block(x)
        return self.classifier(x)
