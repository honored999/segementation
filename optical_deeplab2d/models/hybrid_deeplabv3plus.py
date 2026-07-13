from __future__ import annotations
import torch
from torch import nn
from .backbone import SpatialLogitHead, build_deeplab
from .optical_conv import OpticalConv2d

class HybridOpticalDeepLabV3Plus(SpatialLogitHead):
    """Ideal optical 1-to-8 front end plus three-channel DeepLabV3+."""
    def __init__(self, encoder_name: str = "mobilenet_v2", encoder_weights: str | None = "imagenet") -> None:
        super().__init__(); self.optical = OpticalConv2d(); self.norm = nn.GroupNorm(4, 8); self.adapter = nn.Sequential(nn.Conv2d(8, 3, 1, bias=False), nn.BatchNorm2d(3), nn.ReLU(inplace=True)); self.backbone, self.resolved_encoder = build_deeplab(encoder_name, encoder_weights)
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self._restore(self.backbone(self.adapter(torch.relu(self.norm(self.optical(image))))), image)

