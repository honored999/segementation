"""Small pure-2D U-Net used by the Stage 2 nnU-Net Trainer."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock2D(nn.Module):
    """Two unnormalized 3x3 convolutions with ReLU activations."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.conv1(x))
        return self.relu2(self.conv2(x))


class SmallUNet2D(nn.Module):
    """A fixed small U-Net with two downsampling steps and standard skips."""

    BASE_CHANNELS = 16

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        channels16 = self.BASE_CHANNELS
        channels32 = channels16 * 2
        channels64 = channels32 * 2

        self.encoder16 = ConvBlock2D(in_channels, channels16)
        self.downsample1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder32 = ConvBlock2D(channels16, channels32)
        self.downsample2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder64 = ConvBlock2D(channels32, channels64)

        self.decoder32 = ConvBlock2D(channels64 + channels32, channels32)
        self.decoder16 = ConvBlock2D(channels32 + channels16, channels16)
        self.final_conv = nn.Conv2d(channels16, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder16 = self.encoder16(x)
        encoder32 = self.encoder32(self.downsample1(encoder16))
        encoder64 = self.encoder64(self.downsample2(encoder32))

        decoder32 = F.interpolate(
            encoder64,
            size=encoder32.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoder32 = self.decoder32(torch.cat((decoder32, encoder32), dim=1))

        decoder16 = F.interpolate(
            decoder32,
            size=encoder16.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoder16 = self.decoder16(torch.cat((decoder16, encoder16), dim=1))

        return self.final_conv(decoder16)
