"""Minimal 2D fully convolutional network for the Stage 2 trainer."""

from __future__ import annotations

import torch
from torch import nn


class LiteFCN2D(nn.Module):
    """A fixed-resolution-preserving stack of dilated 2D convolutions."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, dilation=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, dilation=4, padding=4),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, dilation=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
