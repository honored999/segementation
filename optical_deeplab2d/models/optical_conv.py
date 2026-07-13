from __future__ import annotations
import torch
from torch import nn

class OpticalConv2d(nn.Module):
    """Ideal signed, trainable 1-to-8 optical convolution."""
    def __init__(self) -> None:
        super().__init__(); self.conv = nn.Conv2d(1, 8, 5, stride=1, padding=2, bias=False); nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
    def forward(self, image: torch.Tensor) -> torch.Tensor: return self.conv(image)

