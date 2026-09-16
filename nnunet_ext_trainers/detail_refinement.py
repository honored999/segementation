"""Full-resolution residual feature refinement for official nnU-Net decoders."""

from __future__ import annotations

from copy import deepcopy
from typing import Type

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd


DETAIL_CHANNELS = 16


class DetailRefiner(nn.Module):
    """Predict a zero-initialized residual for a full-resolution feature map."""

    def __init__(
        self,
        input_channels: int,
        conv_op: Type[_ConvNd],
        conv_bias: bool,
        norm_op: type[nn.Module],
        norm_op_kwargs: dict,
        nonlin: type[nn.Module],
        nonlin_kwargs: dict,
    ) -> None:
        super().__init__()
        if norm_op is None or nonlin is None:
            raise TypeError("DetailRefiner requires the baseline normalization and activation types")

        self.input_channels = input_channels
        self.input_projection = conv_op(
            input_channels, DETAIL_CHANNELS, kernel_size=1, stride=1, padding=0, bias=conv_bias
        )
        self.detail_conv1 = conv_op(
            DETAIL_CHANNELS, DETAIL_CHANNELS, kernel_size=3, stride=1, padding=1, bias=conv_bias
        )
        self.detail_norm1 = norm_op(DETAIL_CHANNELS, **deepcopy(norm_op_kwargs))
        self.detail_nonlin1 = nonlin(**deepcopy(nonlin_kwargs))
        self.detail_conv2 = conv_op(
            DETAIL_CHANNELS, DETAIL_CHANNELS, kernel_size=3, stride=1, padding=1, bias=conv_bias
        )
        self.detail_norm2 = norm_op(DETAIL_CHANNELS, **deepcopy(norm_op_kwargs))
        self.detail_nonlin2 = nonlin(**deepcopy(nonlin_kwargs))
        self.output_projection = conv_op(
            DETAIL_CHANNELS, input_channels, kernel_size=1, stride=1, padding=0, bias=conv_bias
        )
        nn.init.zeros_(self.output_projection.weight)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.input_projection(features)
        residual = self.detail_nonlin1(self.detail_norm1(self.detail_conv1(residual)))
        residual = self.detail_nonlin2(self.detail_norm2(self.detail_conv2(residual)))
        return self.output_projection(residual)


class DetailRefinementSegmentationHead(nn.Module):
    """Apply feature refinement immediately before the original segmentation head."""

    def __init__(self, detail_refiner: DetailRefiner, segmentation_head: nn.Module) -> None:
        super().__init__()
        self.detail_refiner = detail_refiner
        self.segmentation_head = segmentation_head

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        refined_features = features + self.detail_refiner(features)
        return self.segmentation_head(refined_features)


def attach_detail_refinement(network: nn.Module) -> nn.Module:
    """Replace only the highest-resolution segmentation head of an nnU-Net."""

    decoder = getattr(network, "decoder", None)
    seg_layers = getattr(decoder, "seg_layers", None)
    encoder = getattr(decoder, "encoder", None)
    if seg_layers is None or encoder is None or len(seg_layers) == 0:
        raise TypeError("Detail refinement requires a network with decoder.seg_layers and decoder.encoder")

    segmentation_head = seg_layers[-1]
    if isinstance(segmentation_head, DetailRefinementSegmentationHead):
        raise ValueError("Detail refinement is already attached")
    input_channels = getattr(segmentation_head, "in_channels", None)
    if not isinstance(input_channels, int):
        raise TypeError("The highest-resolution segmentation head must expose integer in_channels")

    refiner = DetailRefiner(
        input_channels=input_channels,
        conv_op=encoder.conv_op,
        conv_bias=encoder.conv_bias,
        norm_op=encoder.norm_op,
        norm_op_kwargs=encoder.norm_op_kwargs,
        nonlin=encoder.nonlin,
        nonlin_kwargs=encoder.nonlin_kwargs,
    )
    seg_layers[-1] = DetailRefinementSegmentationHead(refiner, segmentation_head)
    return network
