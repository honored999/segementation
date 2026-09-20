"""PlainConvUNet encoder with an isolated lightweight UPerNet decoder."""

from __future__ import annotations

from torch import Tensor, nn

from standalone_nnunet2d.config import ModelConfig
from standalone_nnunet2d.models.lite_upernet import LiteUPerDecoder
from standalone_nnunet2d.models.plain_conv_unet import PlainConvUNet2D


class PlainConvUNetLiteUPerNet(PlainConvUNet2D):
    """Single-output PlainConvUNet decoder ablation."""

    _SELECTED_INDICES = (1, 3, 5, 7)

    def __init__(self, config: ModelConfig) -> None:
        if config.n_stages < 8:
            raise ValueError("PlainConvUNetLiteUPerNet requires at least eight encoder stages")
        super().__init__(config, deep_supervision=False)
        del self.transposed_convolutions
        del self.decoder_stages
        del self.segmentation_heads

        norm_factory = lambda channels: nn.InstanceNorm2d(
            channels,
            eps=config.norm_eps,
            affine=config.norm_affine,
        )
        self.decoder = LiteUPerDecoder(
            in_channels=tuple(config.features_per_stage[index] for index in self._SELECTED_INDICES),
            num_classes=config.output_channels,
            fpn_channels=64,
            pool_scales=(1, 2, 4),
            norm_factory=norm_factory,
        )
        self.last_selected_feature_shapes: tuple[tuple[int, ...], ...] = ()

    def forward(self, image: Tensor) -> Tensor:
        encoder_outputs = self.forward_features(image)
        selected = tuple(encoder_outputs[index] for index in self._SELECTED_INDICES)
        self.last_selected_feature_shapes = tuple(tuple(feature.shape) for feature in selected)
        return self.decoder(selected, output_size=(int(image.shape[-2]), int(image.shape[-1])))


__all__ = ["PlainConvUNetLiteUPerNet"]
