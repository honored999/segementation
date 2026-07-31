"""Standalone 2D PlainConvUNet, intentionally independent of nnunetv2."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from standalone_nnunet2d.config import ModelConfig
from standalone_nnunet2d.models.blocks import StackedConvBlocks


class PlainConvUNet2D(nn.Module):
    """Eight-stage 2D PlainConvUNet configured from ``nnUNetPlans.json``.

    With deep supervision enabled, ``forward`` returns a tuple ordered as
    ``(512x512, 256x256, 128x128, 64x64, 32x32, 16x16, 8x8)`` logits.
    Otherwise it returns only the full-resolution logits tensor.
    """

    def __init__(self, config: ModelConfig, *, deep_supervision: bool = False) -> None:
        super().__init__()
        if config.n_stages < 2:
            raise ValueError("PlainConvUNet2D requires at least two encoder stages")
        self.deep_supervision = deep_supervision
        self.config = config
        self.encoder_stages = nn.ModuleList()
        in_channels = config.input_channels
        for stage_index, out_channels in enumerate(config.features_per_stage):
            self.encoder_stages.append(
                self._stage(
                    in_channels,
                    out_channels,
                    config.kernel_sizes[stage_index],
                    config.n_conv_per_stage[stage_index],
                    config.strides[stage_index],
                )
            )
            in_channels = out_channels

        self.transposed_convolutions = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()
        self.segmentation_heads = nn.ModuleList()
        decoder_input_channels = config.features_per_stage[-1]
        for decoder_index in range(config.n_stages - 1):
            skip_index = config.n_stages - 2 - decoder_index
            skip_channels = config.features_per_stage[skip_index]
            up_stride = config.strides[skip_index + 1]
            self.transposed_convolutions.append(
                nn.ConvTranspose2d(
                    decoder_input_channels,
                    skip_channels,
                    kernel_size=up_stride,
                    stride=up_stride,
                    bias=config.conv_bias,
                )
            )
            self.decoder_stages.append(
                self._stage(
                    skip_channels * 2,
                    skip_channels,
                    config.kernel_sizes[skip_index],
                    config.n_conv_per_stage_decoder[decoder_index],
                    (1, 1),
                )
            )
            self.segmentation_heads.append(nn.Conv2d(skip_channels, config.output_channels, kernel_size=1, bias=True))
            decoder_input_channels = skip_channels

        self.last_encoder_shapes: tuple[tuple[int, ...], ...] = ()
        self.last_decoder_shapes: tuple[tuple[int, ...], ...] = ()

    def _stage(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        n_convolutions: int,
        first_stride: tuple[int, int],
    ) -> StackedConvBlocks:
        return StackedConvBlocks(
            in_channels,
            out_channels,
            kernel_size,
            n_convolutions,
            first_stride,
            conv_bias=self.config.conv_bias,
            norm_eps=self.config.norm_eps,
            norm_affine=self.config.norm_affine,
            negative_slope=self.config.leaky_relu_negative_slope,
            inplace=self.config.leaky_relu_inplace,
        )

    def forward(self, image: Tensor) -> Tensor | tuple[Tensor, ...]:
        """Map ``(B, 1, H, W)`` image tensors to raw segmentation logits."""
        encoder_outputs: list[Tensor] = []
        features = image  # (B, 1, 512, 512) for the supplied plans.
        for encoder_stage in self.encoder_stages:
            features = encoder_stage(features)
            encoder_outputs.append(features)
        self.last_encoder_shapes = tuple(tuple(output.shape) for output in encoder_outputs)

        decoder_outputs: list[Tensor] = []
        segmentation_outputs: list[Tensor] = []
        features = encoder_outputs[-1]  # (B, 512, 4, 4) for a 512x512 input.
        for decoder_index, (upsample, decoder_stage, head) in enumerate(
            zip(self.transposed_convolutions, self.decoder_stages, self.segmentation_heads, strict=True)
        ):
            skip = encoder_outputs[-2 - decoder_index]
            upsampled = upsample(features)
            if upsampled.shape[2:] != skip.shape[2:]:
                raise RuntimeError(
                    f"upsampled shape {tuple(upsampled.shape)} does not match skip shape {tuple(skip.shape)}"
                )
            features = decoder_stage(torch.cat((upsampled, skip), dim=1))
            decoder_outputs.append(features)
            segmentation_outputs.append(head(features))
        self.last_decoder_shapes = tuple(tuple(output.shape) for output in decoder_outputs)

        ordered_logits = tuple(reversed(segmentation_outputs))
        return ordered_logits if self.deep_supervision else ordered_logits[0]
