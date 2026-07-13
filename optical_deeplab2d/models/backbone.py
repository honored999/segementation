"""SMP DeepLabV3+ factory with explicit, recorded encoder fallback."""
from __future__ import annotations
import warnings
import torch
from torch import nn

def build_deeplab(encoder_name: str = "mobilenet_v2", encoder_weights: str | None = "imagenet") -> tuple[nn.Module, str]:
    try:
        import segmentation_models_pytorch as smp
    except ImportError as error:
        raise RuntimeError("segmentation_models_pytorch is required. Install optical_deeplab2d/requirements.txt on the server.") from error
    kwargs = dict(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=3, classes=1, activation=None, encoder_output_stride=16, decoder_atrous_rates=(12, 24, 36))
    try: return smp.DeepLabV3Plus(**kwargs), encoder_name
    except Exception as error:
        if encoder_name == "resnet18": raise
        warnings.warn(f"DeepLabV3+ encoder '{encoder_name}' unavailable ({error}); falling back to resnet18.", RuntimeWarning)
        kwargs["encoder_name"] = "resnet18"; return smp.DeepLabV3Plus(**kwargs), "resnet18"

class SpatialLogitHead(nn.Module):
    def _restore(self, logits: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(logits, image.shape[-2:], mode="bilinear", align_corners=False) if logits.shape[-2:] != image.shape[-2:] else logits

