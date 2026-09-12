"""Self-contained H2Former building blocks.

Provenance: adapted from ``third_party/H2Former/models/basic_module.py``.
The reference code is MIT-licensed (Copyright 2022 Along He); see
``third_party/H2Former/LICENSE``.  This module intentionally excludes the
reference training and data scripts and does not import the reference package.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
from torch import Tensor, nn


def to_2tuple(value: int | tuple[int, int]) -> tuple[int, int]:
    """Return an integer as a square tuple, matching the timm helper used upstream."""

    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"expected a 2-tuple, got {value!r}")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def trunc_normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
    """Initialize with PyTorch's truncated normal implementation."""

    return nn.init.trunc_normal_(tensor, mean=mean, std=std)


class DropPath(nn.Module):
    """Per-sample stochastic depth, equivalent to timm's DropPath."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError(f"drop_prob must be in [0, 1), got {drop_prob}")
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep_prob) * random_tensor.floor()


def _square_side(token_count: int, context: str) -> int:
    side = math.isqrt(token_count)
    if side * side != token_count:
        raise ValueError(f"{context} requires a square token grid, got {token_count} tokens")
    return side


class eca_layer(nn.Module):
    """Efficient channel attention used by the reference H2Former blocks."""

    def __init__(self, channel: int, k_size: int = 3) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x) + x


class Mlp(nn.Module):
    """Swin feed-forward block with the reference ECA channel refinement."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.eca = eca_layer(out_features, 3)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Mlp expects [B, tokens, channels], got {tuple(x.shape)}")
        batch, tokens, channels = x.shape
        height = width = _square_side(tokens, "H2Former Mlp")
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = x.view(batch, channels, height, width)
        x = self.eca(x)
        x = x.flatten(2).transpose(1, 2)
        return self.drop(x)


def window_reverse(windows: Tensor, window_size: int, height: int, width: int) -> Tensor:
    if height % window_size != 0 or width % window_size != 0:
        raise ValueError(
            f"window_size={window_size} must divide feature resolution {(height, width)}"
        )
    windows_per_image = (height // window_size) * (width // window_size)
    if windows.shape[0] % windows_per_image != 0:
        raise ValueError("window batch does not match the feature resolution")
    batch = windows.shape[0] // windows_per_image
    x = windows.view(batch, height // window_size, width // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, height, width, -1)
    return x


def window_partition(x: Tensor, window_size: int) -> Tensor:
    batch, height, width, channels = x.shape
    if height % window_size != 0 or width % window_size != 0:
        raise ValueError(
            f"window_size={window_size} must divide feature resolution {(height, width)}"
        )
    x = x.view(batch, height // window_size, window_size, width // window_size, window_size, channels)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, channels)


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_windows, tokens, channels = x.shape
        if channels != self.dim:
            raise ValueError(f"WindowAttention expected {self.dim} channels, got {channels}")
        qkv = self.qkv(x).reshape(
            batch_windows, tokens, 3, self.num_heads, channels // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        query = query * self.scale
        attention = query @ key.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attention = attention + relative_position_bias.unsqueeze(0)

        if mask is not None:
            number_windows = mask.shape[0]
            attention = attention.view(
                batch_windows // number_windows, number_windows, self.num_heads, tokens, tokens
            ) + mask.unsqueeze(1).unsqueeze(0)
            attention = attention.view(-1, self.num_heads, tokens, tokens)
        attention = self.softmax(attention)
        attention = self.attn_drop(attention)
        x = (attention @ value).transpose(1, 2).reshape(batch_windows, tokens, channels)
        return self.proj_drop(self.proj(x))


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(input_resolution)
        if not 0 <= self.shift_size < self.window_size:
            raise ValueError("shift_size must be in [0, window_size)")

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

        if self.shift_size > 0:
            height, width = input_resolution
            image_mask = torch.zeros((1, height, width, 1))
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            counter = 0
            for h_slice in h_slices:
                for w_slice in w_slices:
                    image_mask[:, h_slice, w_slice, :] = counter
                    counter += 1
            mask_windows = window_partition(image_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attention_mask = attention_mask.masked_fill(
                attention_mask != 0, float(-100.0)
            ).masked_fill(attention_mask == 0, float(0.0))
        else:
            attention_mask = None
        self.register_buffer("attn_mask", attention_mask)

    def forward(self, x: Tensor) -> Tensor:
        height, width = self.input_resolution
        batch, tokens, channels = x.shape
        if tokens != height * width:
            raise ValueError(
                f"SwinTransformerBlock expected {height * width} tokens, got {tokens}"
            )
        if channels != self.dim:
            raise ValueError(f"SwinTransformerBlock expected {self.dim} channels, got {channels}")

        shortcut = x
        x = self.norm1(x).view(batch, height, width, channels)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size).view(
            -1, self.window_size * self.window_size, channels
        )
        attention_windows = self.attn(x_windows, mask=self.attn_mask)
        attention_windows = attention_windows.view(-1, self.window_size, self.window_size, channels)
        shifted_x = window_reverse(attention_windows, self.window_size, height, width)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(batch, height * width, channels)
        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        downsample: Optional[type[nn.Module]] = None,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if i % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(2 * out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self.up(x1)
        if x1.shape[2:] != x2.shape[2:]:
            raise RuntimeError(
                f"decoder upsample shape {tuple(x1.shape)} does not match skip shape {tuple(x2.shape)}"
            )
        return self.conv_bn_relu(torch.cat((x1, x2), dim=1))


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: tuple[int, ...] = (4,),
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: Optional[type[nn.Module]] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.patches_resolution = [self.img_size[0] // 4, self.img_size[1] // 4]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.eca = eca_layer(embed_dim, 3)

        self.projs = nn.ModuleList()
        for index, patch in enumerate(patch_size):
            if index == len(patch_size) - 1:
                channels = embed_dim // 2**index
            else:
                channels = embed_dim // 2 ** (index + 1)
            stride = 2
            padding = (patch - stride) // 2
            self.projs.append(
                nn.Conv2d(in_chans, channels, kernel_size=patch, stride=stride, padding=padding)
            )
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"PatchEmbed expects BCHW input, got {tuple(x.shape)}")
        _, _, height, width = x.shape
        if (height, width) != self.img_size:
            raise ValueError(
                f"PatchEmbed expects spatial size {self.img_size}, got {(height, width)}"
            )
        features = [projection(x) for projection in self.projs]
        x = torch.cat(features, dim=1)
        x = self.eca(x).flatten(2).transpose(1, 2)
        return self.norm(x) if self.norm is not None else x


class PatchMerging(nn.Module):
    def __init__(
        self,
        dim: int,
        patch_size: tuple[int, int] = (2, 4),
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = norm_layer(dim)
        self.eca = eca_layer(dim, 3)
        self.reductions = nn.ModuleList()
        for index, patch in enumerate(patch_size):
            if index == len(patch_size) - 1:
                out_dim = 2 * dim // 2**index
            else:
                out_dim = 2 * dim // 2 ** (index + 1)
            stride = 2
            padding = (patch - stride) // 2
            self.reductions.append(
                nn.Sequential(
                    nn.Conv2d(dim, out_dim, kernel_size=patch, stride=stride, padding=padding)
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"PatchMerging expects [B, tokens, channels], got {tuple(x.shape)}")
        batch, tokens, channels = x.shape
        if channels != self.dim:
            raise ValueError(f"PatchMerging expected {self.dim} channels, got {channels}")
        x = self.norm(x)
        height = width = _square_side(tokens, "H2Former PatchMerging")
        x = x.view(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()
        x = torch.cat([reduction(x) for reduction in self.reductions], dim=1)
        return self.eca(x)


__all__ = [
    "BasicBlock",
    "BasicLayer",
    "Decoder",
    "DropPath",
    "PatchEmbed",
    "PatchMerging",
    "WindowAttention",
    "conv1x1",
    "conv3x3",
    "to_2tuple",
    "trunc_normal_",
]
