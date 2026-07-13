from __future__ import annotations
import torch
from .backbone import SpatialLogitHead, build_deeplab

class ElectronicDeepLabV3Plus(SpatialLogitHead):
    """Electronic fairness baseline: grayscale repetition plus same backend."""
    def __init__(self, encoder_name: str = "mobilenet_v2", encoder_weights: str | None = "imagenet") -> None:
        super().__init__(); self.backbone, self.resolved_encoder = build_deeplab(encoder_name, encoder_weights)
    def forward(self, image: torch.Tensor) -> torch.Tensor: return self._restore(self.backbone(image.repeat(1, 3, 1, 1)), image)

