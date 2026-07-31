"""Pure PyTorch building blocks used by the standalone PlainConvUNet."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn


class ConvNormNonlin(nn.Sequential):
    """Conv2d -> InstanceNorm2d -> LeakyReLU, with same-size padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        *,
        conv_bias: bool,
        norm_eps: float,
        norm_affine: bool,
        negative_slope: float,
        inplace: bool,
    ) -> None:
        padding = tuple(size // 2 for size in kernel_size)
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=conv_bias),
            nn.InstanceNorm2d(out_channels, eps=norm_eps, affine=norm_affine),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace),
        )


class StackedConvBlocks(nn.Module):
    """A stage whose first convolution may perform stride-based downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        n_convolutions: int,
        first_stride: tuple[int, int],
        *,
        conv_bias: bool,
        norm_eps: float,
        norm_affine: bool,
        negative_slope: float,
        inplace: bool,
    ) -> None:
        super().__init__()
        if n_convolutions < 1:
            raise ValueError("a convolution stage requires at least one convolution")
        blocks: list[nn.Module] = []
        for index in range(n_convolutions):
            blocks.append(
                ConvNormNonlin(
                    in_channels if index == 0 else out_channels,
                    out_channels,
                    kernel_size,
                    first_stride if index == 0 else (1, 1),
                    conv_bias=conv_bias,
                    norm_eps=norm_eps,
                    norm_affine=norm_affine,
                    negative_slope=negative_slope,
                    inplace=inplace,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.blocks(inputs)
