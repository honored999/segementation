"""Standalone 2D H2Former candidate model.

Provenance: adapted from ``third_party/H2Former/models/H2Former.py`` and its
building blocks in ``third_party/H2Former/models/basic_module.py``.  The
reference code is MIT-licensed (Copyright 2022 Along He); see
``third_party/H2Former/LICENSE``.  This adaptation does not modify or import
the reference package and does not include its training or data scripts.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from standalone_nnunet2d.models.h2former_blocks import (
    BasicBlock,
    BasicLayer,
    Decoder,
    PatchEmbed,
    PatchMerging,
    conv1x1,
)


class H2Former(nn.Module):
    """H2Former for fixed-size 2D slices.

    The model preserves the reference ResNet-34 plus multi-scale PatchEmbed,
    four Swin stages, and three decoder stages.  This phase intentionally
    supports only square 512x512 inputs and returns one logits tensor.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        image_size: int = 512,
    ) -> None:
        super().__init__()
        if not isinstance(in_channels, int) or in_channels < 1:
            raise ValueError(f"in_channels must be a positive integer, got {in_channels!r}")
        if not isinstance(num_classes, int) or num_classes < 1:
            raise ValueError(f"num_classes must be a positive integer, got {num_classes!r}")
        if image_size != 512:
            raise ValueError(
                f"H2Former currently supports only image_size=512, got {image_size!r}"
            )

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.image_size = image_size
        self.window_size = image_size // 16
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        self.conv1 = nn.Conv2d(
            in_channels, self.inplanes, kernel_size=7, stride=1, padding=3, bias=False
        )
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, 3)
        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=2)

        embed_dim = 64
        depths = [2, 2, 2, 2]
        num_heads = [2, 4, 8, 16]
        drop_path_rate = 0.1
        drop_path_rates = [
            value.item() for value in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        patch_resolution = image_size // 2

        self.patch_embed = PatchEmbed(
            img_size=image_size,
            patch_size=(2, 4, 8, 16),
            in_chans=in_channels,
            embed_dim=embed_dim,
        )
        self.MS2 = PatchMerging(64)
        self.MS3 = PatchMerging(128)
        self.MS4 = PatchMerging(256)

        self.swin_layers = nn.ModuleList()
        for layer_index in range(4):
            self.swin_layers.append(
                BasicLayer(
                    dim=embed_dim * 2**layer_index,
                    input_resolution=(
                        patch_resolution // 2**layer_index,
                        patch_resolution // 2**layer_index,
                    ),
                    depth=depths[layer_index],
                    num_heads=num_heads[layer_index],
                    window_size=self.window_size,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=0.0,
                    attn_drop=0.0,
                    drop_path=drop_path_rates[
                        sum(depths[:layer_index]) : sum(depths[: layer_index + 1])
                    ],
                    norm_layer=nn.LayerNorm,
                    downsample=None,
                    use_checkpoint=False,
                )
            )

        channels = [64, 128, 256, 512]
        self.decode4 = Decoder(channels[3], channels[2])
        self.decode3 = Decoder(channels[2], channels[1])
        self.decode2 = Decoder(channels[1], channels[0])
        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(channels[0], num_classes, kernel_size=1, bias=False),
        )

    def _make_layer(
        self,
        block: type[BasicBlock],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                self._norm_layer(planes * block.expansion),
            )

        layers = [
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                self._norm_layer,
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=self._norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def _tokens_to_image(self, tokens: Tensor, stage_index: int) -> Tensor:
        if tokens.ndim != 3:
            raise RuntimeError(f"Swin stage {stage_index} returned {tuple(tokens.shape)}, expected 3D")
        batch, token_count, channels = tokens.shape
        expected_side = self.image_size // (2 ** (stage_index + 1))
        expected_channels = 64 * 2**stage_index
        if (token_count, channels) != (expected_side * expected_side, expected_channels):
            raise RuntimeError(
                f"Swin stage {stage_index} returned tokens {(token_count, channels)}, "
                f"expected {(expected_side * expected_side, expected_channels)}"
            )
        return tokens.view(batch, expected_side, expected_side, channels).permute(0, 3, 1, 2)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"H2Former expects BCHW input, got shape {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"H2Former in_channels mismatch: expected {self.in_channels}, got {x.shape[1]}"
            )
        if tuple(x.shape[2:]) != (self.image_size, self.image_size):
            raise ValueError(
                f"H2Former image_size mismatch: expected {(self.image_size, self.image_size)}, "
                f"got {tuple(x.shape[2:])}"
            )

        encoder: list[Tensor] = []
        multi_scale = self.patch_embed(x)
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = x.flatten(2).transpose(1, 2) + multi_scale

        x = self.swin_layers[0](x)
        multi_scale_2 = self.MS2(x)
        x = self._tokens_to_image(x, 0)
        encoder.append(x)

        x = self.layer2(x)
        if x.shape != multi_scale_2.shape:
            raise RuntimeError(f"stage-2 fusion shape mismatch: {tuple(x.shape)} vs {tuple(multi_scale_2.shape)}")
        x = x + multi_scale_2
        x = self.swin_layers[1](x.flatten(2).transpose(1, 2))
        multi_scale_3 = self.MS3(x)
        x = self._tokens_to_image(x, 1)
        encoder.append(x)

        x = self.layer3(x)
        if x.shape != multi_scale_3.shape:
            raise RuntimeError(f"stage-3 fusion shape mismatch: {tuple(x.shape)} vs {tuple(multi_scale_3.shape)}")
        x = x + multi_scale_3
        x = self.swin_layers[2](x.flatten(2).transpose(1, 2))
        multi_scale_4 = self.MS4(x)
        x = self._tokens_to_image(x, 2)
        encoder.append(x)

        x = self.layer4(x)
        if x.shape != multi_scale_4.shape:
            raise RuntimeError(f"stage-4 fusion shape mismatch: {tuple(x.shape)} vs {tuple(multi_scale_4.shape)}")
        x = x + multi_scale_4
        x = self.swin_layers[3](x.flatten(2).transpose(1, 2))
        encoder.append(self._tokens_to_image(x, 3))

        x = self.decode4(encoder[3], encoder[2])
        x = self.decode3(x, encoder[1])
        x = self.decode2(x, encoder[0])
        return self.decode0(x)


__all__ = ["H2Former"]
