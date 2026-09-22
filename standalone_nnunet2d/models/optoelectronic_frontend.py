"""Independent synthetic optoelectronic front-end operators.

This module deliberately stops at reusable operators and synthetic tests.  It
does not modify or import the H2Former model.  The referenced OEADNet paper
supports PCA initialization, signed-kernel splitting, differentiable optical
responses, and joint optical/electronic optimization.  The centered PCA
contract, single-channel dual rails, 4f sampling constants, global exposure
ledger, and finite-support accounting below are project assumptions or
deliberate extensions; they are not claimed to be the paper's original
implementation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
import hashlib
import math
from typing import Any, Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


_DEFAULT_WAVELENGTH_M = 520e-9
_DEFAULT_PHASE_GRID_SIZE = 256
_DEFAULT_PHASE_PITCH_M = 8e-6
_DEFAULT_FOCAL_LENGTH_M = 20e-3
_DEFAULT_APERTURE_DIAMETER_M = 2.048e-3
_DEFAULT_MAGNIFICATION = 1.0


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _require_positive(value: object, name: str) -> float:
    if not _is_finite_number(value) or float(value) <= 0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return float(value)


def _require_unit_interval(value: object, name: str) -> float:
    if not _is_finite_number(value) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1], got {value!r}")
    return float(value)


def _as_pair(value: int | tuple[int, int], name: str) -> int | tuple[int, int]:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer or a pair of integers")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value!r}")
        return value
    if isinstance(value, tuple) and len(value) == 2 and all(isinstance(item, int) for item in value):
        if any(item < 0 for item in value):
            raise ValueError(f"{name} must be non-negative, got {value!r}")
        return value
    raise ValueError(f"{name} must be an integer or a pair of integers, got {value!r}")


def _scalar_value(value: Tensor | float | int) -> float:
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError(f"expected a scalar tensor, got shape {tuple(value.shape)}")
        return float(value.detach().item())
    return float(value)


def _tensor_version(tensor: Tensor) -> int:
    version = getattr(tensor, "_version", None)
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("runtime identity tensor has no valid version counter")
    return int(version)


@dataclass(frozen=True)
class PhasePSFConfig:
    """Synthetic engineering constants for the ideal phase-only 4f model."""

    wavelength_m: float = _DEFAULT_WAVELENGTH_M
    phase_grid_size: int = _DEFAULT_PHASE_GRID_SIZE
    phase_pitch_m: float = _DEFAULT_PHASE_PITCH_M
    focal_length_m: float = _DEFAULT_FOCAL_LENGTH_M
    aperture_diameter_m: float = _DEFAULT_APERTURE_DIAMETER_M
    magnification: float = _DEFAULT_MAGNIFICATION

    def __post_init__(self) -> None:
        _require_positive(self.wavelength_m, "wavelength_m")
        if isinstance(self.phase_grid_size, bool) or not isinstance(self.phase_grid_size, int):
            raise ValueError(f"phase_grid_size must be a positive integer, got {self.phase_grid_size!r}")
        if self.phase_grid_size < 2:
            raise ValueError(f"phase_grid_size must be at least 2, got {self.phase_grid_size!r}")
        _require_positive(self.phase_pitch_m, "phase_pitch_m")
        _require_positive(self.focal_length_m, "focal_length_m")
        _require_positive(self.aperture_diameter_m, "aperture_diameter_m")
        _require_positive(self.magnification, "magnification")

    @property
    def assumption_status(self) -> str:
        return "synthetic engineering assumptions"

    @property
    def detector_pitch_m(self) -> float:
        return (
            self.wavelength_m
            * self.focal_length_m
            / (self.phase_grid_size * self.phase_pitch_m)
        )

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["detector_pitch_m"] = self.detector_pitch_m
        result["assumption_status"] = self.assumption_status
        return result


@dataclass(frozen=True)
class OptoelectronicConfig:
    """Runtime constraints for synthetic physical-path calculations."""

    phase: PhasePSFConfig = field(default_factory=PhasePSFConfig)
    dual_rail_mode: str = "sequential_dual_rail"
    gain_update_mode: str = "calibrated_detached"
    rho: float = 1.0
    min_throughput: float = 1e-6
    max_electronic_gain: float = 1e6
    min_split_fraction: float = 1e-6
    max_support_response_error: float = 1e-3
    support_error_mode: str = "report_only"

    def __post_init__(self) -> None:
        if self.dual_rail_mode != "sequential_dual_rail":
            raise ValueError(
                "dual_rail_mode must be 'sequential_dual_rail', "
                f"got {self.dual_rail_mode!r}"
            )
        if self.gain_update_mode != "calibrated_detached":
            raise ValueError(
                "gain_update_mode must be 'calibrated_detached', "
                f"got {self.gain_update_mode!r}"
            )
        _require_positive(self.rho, "rho")
        if not _is_finite_number(self.min_throughput) or not 0.0 <= float(self.min_throughput) <= 1.0:
            raise ValueError("min_throughput must be finite and in [0, 1]")
        _require_positive(self.max_electronic_gain, "max_electronic_gain")
        if not _is_finite_number(self.min_split_fraction) or not 0.0 <= float(self.min_split_fraction) <= 1.0:
            raise ValueError("min_split_fraction must be finite and in [0, 1]")
        if not _is_finite_number(self.max_support_response_error) or float(self.max_support_response_error) < 0:
            raise ValueError("max_support_response_error must be finite and non-negative")
        if self.support_error_mode not in {"report_only", "fail_closed", "report-only", "fail-closed"}:
            raise ValueError(
                "support_error_mode must be report_only or fail_closed, "
                f"got {self.support_error_mode!r}"
            )

    @property
    def normalized_support_error_mode(self) -> str:
        return self.support_error_mode.replace("-", "_")

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["phase"] = self.phase.as_dict()
        result["support_error_mode"] = self.normalized_support_error_mode
        result["assumption_status"] = self.phase.assumption_status
        return result


@dataclass(frozen=True)
class DetectorSamplingMetadata:
    wavelength_m: float
    phase_grid_size: int
    phase_pitch_m: float
    focal_length_m: float
    aperture_diameter_m: float
    magnification: float
    detector_pitch_m: float
    assumption_status: str
    input_pixel_to_object_sample: str
    output_position_to_detector_sample: str
    detector_sample_to_fft_cell: str
    nifti_spacing_is_unrelated: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def detector_sampling_metadata(config: PhasePSFConfig | None = None) -> DetectorSamplingMetadata:
    config = config or PhasePSFConfig()
    return DetectorSamplingMetadata(
        wavelength_m=config.wavelength_m,
        phase_grid_size=config.phase_grid_size,
        phase_pitch_m=config.phase_pitch_m,
        focal_length_m=config.focal_length_m,
        aperture_diameter_m=config.aperture_diameter_m,
        magnification=config.magnification,
        detector_pitch_m=config.detector_pitch_m,
        assumption_status=config.assumption_status,
        input_pixel_to_object_sample=(
            "one digital input pixel is one object-plane intensity sample"
        ),
        output_position_to_detector_sample=(
            "unit magnification maps one digital output position to one detector sample"
        ),
        detector_sample_to_fft_cell=(
            "one detector sample maps to one FFT PSF grid cell as a discrete area-integral approximation"
        ),
        nifti_spacing_is_unrelated=(
            "optical micrometre coordinates are independent of NIfTI spacing"
        ),
    )


@dataclass(frozen=True)
class PCAReport:
    resolved_rank: int
    structural_rank_upper_bound: int
    explained_variance_ratio: float
    absolute_frobenius_error: float
    relative_frobenius_error: float
    maximum_absolute_error: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CenteredPCAResult:
    """A centered row-wise PCA decomposition of convolution kernels."""

    mean_kernel: Tensor
    basis: Tensor
    coefficients: Tensor
    component_bank: Tensor
    original_weight: Tensor
    original_bias: Tensor | None
    stride: int | tuple[int, int]
    padding: int | tuple[int, int]
    kernel_size: tuple[int, int]
    report: PCAReport

    @property
    def mean(self) -> Tensor:
        return self.mean_kernel

    @property
    def resolved_rank(self) -> int:
        return self.report.resolved_rank

    @property
    def structural_rank_upper_bound(self) -> int:
        return self.report.structural_rank_upper_bound

    def default_mixing(self) -> Tensor:
        ones = torch.ones(
            (self.coefficients.shape[0], 1),
            dtype=self.coefficients.dtype,
            device=self.coefficients.device,
        )
        return torch.cat((ones, self.coefficients), dim=1)

    def reconstruct_weight(self, mixing: Tensor | None = None) -> Tensor:
        if mixing is None:
            mixing = self.default_mixing()
        expected = (self.original_weight.shape[0], self.component_bank.shape[0])
        if tuple(mixing.shape) != expected:
            raise ValueError(f"mixing must have shape {expected}, got {tuple(mixing.shape)}")
        flat = mixing @ self.component_bank.flatten(1)
        return flat.view_as(self.original_weight)

    def as_dict(self) -> dict[str, Any]:
        return {
            "resolved_rank": self.resolved_rank,
            "structural_rank_upper_bound": self.structural_rank_upper_bound,
            "report": self.report.as_dict(),
            "stride": self.stride,
            "padding": self.padding,
            "kernel_size": self.kernel_size,
            "mean_kernel_shape": tuple(self.mean_kernel.shape),
            "basis_shape": tuple(self.basis.shape),
            "coefficients_shape": tuple(self.coefficients.shape),
            "component_bank_shape": tuple(self.component_bank.shape),
            "pca_target_trainable": False,
        }


def centered_pca_decompose(
    weight: Tensor,
    *,
    rank: int | None = None,
    variance_threshold: float | None = None,
    bias: Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
) -> CenteredPCAResult:
    """Decompose output kernels as ``W = 1 mu^T + C B``.

    The output channels are samples.  ``basis`` stores rows of ``B`` and
    ``coefficients`` stores ``C``.  ``rank=0`` intentionally retains only the
    centered sample mean.
    """

    if not isinstance(weight, Tensor) or weight.ndim != 4:
        raise ValueError(f"weight must be a four-dimensional tensor, got {getattr(weight, 'shape', None)}")
    if not torch.is_floating_point(weight):
        raise ValueError("weight must use a floating-point dtype")
    if not torch.isfinite(weight).all():
        raise ValueError("weight must be finite")
    if rank is not None and variance_threshold is not None:
        raise ValueError("specify exactly one of rank or variance_threshold")
    if rank is not None:
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise ValueError(f"rank must be an integer, got {rank!r}")
    else:
        variance_threshold = 0.99 if variance_threshold is None else variance_threshold
        if not _is_finite_number(variance_threshold) or not 0.0 <= float(variance_threshold) <= 1.0:
            raise ValueError("variance_threshold must be in [0, 1]")

    stride = _as_pair(stride, "stride")
    padding = _as_pair(padding, "padding")
    if bias is not None:
        if bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
            raise ValueError(
                f"bias must have shape {(weight.shape[0],)}, got {tuple(bias.shape)}"
            )
        if not torch.is_floating_point(bias) or not torch.isfinite(bias).all():
            raise ValueError("bias must be finite and floating-point")

    out_channels = weight.shape[0]
    flattened_dimension = weight[0].numel()
    structural_upper = min(out_channels - 1, flattened_dimension)
    if rank is not None:
        if rank < 0 or rank > structural_upper:
            raise ValueError(
                f"rank must satisfy 0 <= rank <= {structural_upper}, got {rank}"
            )
        resolved_rank = rank
    else:
        assert variance_threshold is not None
        resolved_rank = 0

    flat = weight.detach().reshape(out_channels, flattened_dimension)
    mean_flat = flat.mean(dim=0)
    centered = flat - mean_flat
    if structural_upper > 0:
        left_vectors, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    else:
        left_vectors = weight.new_empty((out_channels, 0))
        singular_values = weight.new_empty((0,))
        vh = weight.new_empty((0, flattened_dimension))
    variance = singular_values.square()
    total_variance = float(variance.sum().item())

    if rank is None:
        if total_variance > 0 and float(variance_threshold) > 0:
            cumulative = torch.cumsum(variance[:structural_upper], dim=0)
            target = float(variance_threshold) * total_variance
            resolved_rank = int(torch.count_nonzero(cumulative < target).item()) + 1
            resolved_rank = min(resolved_rank, structural_upper)
        else:
            resolved_rank = 0

    if resolved_rank:
        basis_flat = vh[:resolved_rank]
        coefficients = left_vectors[:, :resolved_rank] * singular_values[:resolved_rank]
    else:
        basis_flat = weight.new_empty((0, flattened_dimension))
        coefficients = weight.new_empty((out_channels, 0))

    reconstructed_flat = mean_flat.unsqueeze(0) + coefficients @ basis_flat
    difference = reconstructed_flat - flat
    absolute_error = float(torch.linalg.vector_norm(difference).item())
    weight_norm = float(torch.linalg.vector_norm(flat).item())
    relative_error = absolute_error / weight_norm if weight_norm > 0 else 0.0
    maximum_error = float(difference.abs().max().item()) if difference.numel() else 0.0
    if total_variance == 0:
        explained_ratio = 1.0
    else:
        explained_ratio = float(variance[:resolved_rank].sum().item()) / total_variance

    report = PCAReport(
        resolved_rank=resolved_rank,
        structural_rank_upper_bound=structural_upper,
        explained_variance_ratio=explained_ratio,
        absolute_frobenius_error=absolute_error,
        relative_frobenius_error=relative_error,
        maximum_absolute_error=maximum_error,
    )
    mean_kernel = mean_flat.view(1, *weight.shape[1:])
    basis = basis_flat.view(resolved_rank, *weight.shape[1:])
    component_bank = torch.cat((mean_kernel, basis), dim=0)
    return CenteredPCAResult(
        mean_kernel=mean_kernel,
        basis=basis,
        coefficients=coefficients,
        component_bank=component_bank,
        original_weight=weight.detach().clone(),
        original_bias=None if bias is None else bias.detach().clone(),
        stride=stride,
        padding=padding,
        kernel_size=(weight.shape[-2], weight.shape[-1]),
        report=report,
    )


def decompose_conv2d(
    convolution: nn.Conv2d,
    *,
    rank: int | None = None,
    variance_threshold: float | None = None,
) -> CenteredPCAResult:
    """Create a centered PCA result from an existing Conv2d without changing it."""

    return centered_pca_decompose(
        convolution.weight,
        rank=rank,
        variance_threshold=variance_threshold,
        bias=convolution.bias,
        stride=convolution.stride,
        padding=convolution.padding,
    )


class PCAConv2d(nn.Module):
    """Ideal electronic Conv2d reconstructed from a centered PCA component bank."""

    def __init__(
        self,
        decomposition: CenteredPCAResult,
        *,
        trainable_mixing: bool = False,
        trainable_bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = decomposition.original_weight.shape[1]
        self.out_channels = decomposition.original_weight.shape[0]
        self.kernel_size = decomposition.kernel_size
        self.stride = decomposition.stride
        self.padding = decomposition.padding
        self.resolved_rank = decomposition.resolved_rank
        self.register_buffer("mean_kernel", decomposition.mean_kernel.detach().clone())
        self.register_buffer("basis", decomposition.basis.detach().clone())
        self.register_buffer("coefficients", decomposition.coefficients.detach().clone())
        self.register_buffer("component_bank", decomposition.component_bank.detach().clone())
        self.register_buffer("pca_target_weight", decomposition.original_weight.detach().clone())
        self.register_buffer(
            "original_bias",
            None if decomposition.original_bias is None else decomposition.original_bias.detach().clone(),
        )
        mixing = decomposition.default_mixing().detach().clone()
        if trainable_mixing:
            self.mixing = nn.Parameter(mixing)
        else:
            self.register_buffer("mixing", mixing)
        if decomposition.original_bias is None:
            self.register_parameter("bias", None)
        elif trainable_bias:
            self.bias = nn.Parameter(decomposition.original_bias.detach().clone())
        else:
            self.register_buffer("bias", decomposition.original_bias.detach().clone())
        self.mixing_trainable = bool(trainable_mixing)
        self.bias_trainable = bool(trainable_bias and decomposition.original_bias is not None)

    def reconstruct_weight(self) -> Tensor:
        flat = self.mixing @ self.component_bank.flatten(1)
        return flat.view(self.out_channels, self.in_channels, *self.kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(
            x,
            self.reconstruct_weight(),
            self.bias,
            stride=self.stride,
            padding=self.padding,
        )


@dataclass(frozen=True)
class SignedKernelLobes:
    p_pos: Tensor
    p_neg: Tensor
    alpha_pos: float
    alpha_neg: float

    def reconstruct(self) -> Tensor:
        return self.alpha_pos * self.p_pos - self.alpha_neg * self.p_neg


def signed_input_rails(x: Tensor) -> tuple[Tensor, Tensor]:
    if not isinstance(x, Tensor):
        raise ValueError("x must be a tensor")
    return x.clamp_min(0), (-x).clamp_min(0)


def signed_kernel_lobes(kernel: Tensor) -> SignedKernelLobes:
    if not isinstance(kernel, Tensor) or kernel.ndim < 2:
        raise ValueError("kernel must be a tensor with at least two dimensions")
    if not torch.is_floating_point(kernel) or not torch.isfinite(kernel).all():
        raise ValueError("kernel must be finite and floating-point")
    k_pos = kernel.clamp_min(0)
    k_neg = (-kernel).clamp_min(0)
    alpha_pos = float(k_pos.sum().item())
    alpha_neg = float(k_neg.sum().item())
    p_pos = k_pos / alpha_pos if alpha_pos > 0 else torch.zeros_like(k_pos)
    p_neg = k_neg / alpha_neg if alpha_neg > 0 else torch.zeros_like(k_neg)
    return SignedKernelLobes(p_pos=p_pos, p_neg=p_neg, alpha_pos=alpha_pos, alpha_neg=alpha_neg)


def ideal_signed_conv2d(
    x: Tensor,
    kernel: Tensor,
    bias: Tensor | None = None,
    *,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
) -> Tensor:
    """Evaluate the ideal four-response signed dual-rail operator.

    ``F.conv2d`` is PyTorch's cross-correlation operator, so each of the four
    response terms uses the corresponding non-negative input/kernel rail
    without an additional spatial flip.  The ideal path intentionally keeps
    this algebra explicit; tests may use one independent direct convolution
    as the reference identity.
    """

    if kernel.ndim != 4:
        raise ValueError(f"kernel must be [out,in,H,W], got {tuple(kernel.shape)}")
    output_dtype = x.dtype
    if x.dtype == torch.float32 and kernel.dtype == torch.float32:
        response_dtype = torch.float64
    elif x.dtype in {torch.float16, torch.bfloat16}:
        response_dtype = torch.float32
    else:
        response_dtype = x.dtype
    x_response = x.to(dtype=response_dtype)
    kernel_response = kernel.to(dtype=response_dtype)
    x_pos, x_neg = signed_input_rails(x_response)
    k_pos = kernel_response.clamp_min(0)
    k_neg = (-kernel_response).clamp_min(0)

    response_pos_pos = F.conv2d(
        x_pos,
        k_pos,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    response_neg_pos = F.conv2d(
        x_neg,
        k_pos,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    response_pos_neg = F.conv2d(
        x_pos,
        k_neg,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    response_neg_neg = F.conv2d(
        x_neg,
        k_neg,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    output = response_pos_pos - response_neg_pos - response_pos_neg + response_neg_neg
    if bias is not None:
        output = output + bias.to(dtype=response_dtype).view(1, -1, 1, 1)
    return output.to(dtype=output_dtype)


def _conv_output_shape(
    spatial: tuple[int, int],
    kernel: tuple[int, int],
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
    dilation: int | tuple[int, int],
) -> tuple[int, int]:
    def pair(value: int | tuple[int, int]) -> tuple[int, int]:
        return (value, value) if isinstance(value, int) else value

    sh, sw = pair(stride)
    ph, pw = pair(padding)
    dh, dw = pair(dilation)
    kh, kw = kernel
    return (
        (spatial[0] + 2 * ph - dh * (kh - 1) - 1) // sh + 1,
        (spatial[1] + 2 * pw - dw * (kw - 1) - 1) // sw + 1,
    )


def _same_padding_extents(
    kernel_size: int | tuple[int, int],
) -> tuple[int, int, int, int]:
    """Return ``(top, bottom, left, right)`` for physical same padding."""

    if isinstance(kernel_size, bool):
        raise ValueError("kernel_size must be a positive integer or pair")
    if isinstance(kernel_size, int):
        kernel_height = kernel_width = kernel_size
    elif (
        isinstance(kernel_size, tuple)
        and len(kernel_size) == 2
        and all(isinstance(value, int) and not isinstance(value, bool) for value in kernel_size)
    ):
        kernel_height, kernel_width = kernel_size
    else:
        raise ValueError("kernel_size must be a positive integer or pair")
    if kernel_height <= 0 or kernel_width <= 0:
        raise ValueError("kernel_size must be positive")
    top = kernel_height // 2
    bottom = kernel_height - 1 - top
    left = kernel_width // 2
    right = kernel_width - 1 - left
    return top, bottom, left, right


def _same_internal_mask(
    spatial_shape: tuple[int, int],
    kernel_size: int | tuple[int, int],
    *,
    device: torch.device | None = None,
) -> Tensor:
    """Return the internal output region for a same-padded kernel.

    The right and bottom bounds are exclusive and use ``None`` when their
    extent is zero, avoiding Python's ``:-0`` empty-slice trap.
    """

    if (
        not isinstance(spatial_shape, tuple)
        or len(spatial_shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in spatial_shape)
    ):
        raise ValueError("spatial_shape must be a pair of non-negative integers")
    height, width = spatial_shape
    top, bottom, left, right = _same_padding_extents(kernel_size)
    mask = torch.zeros((height, width), dtype=torch.bool, device=device)
    row_stop = height - bottom if bottom else height
    col_stop = width - right if right else width
    if top < row_stop and left < col_stop:
        mask[top:row_stop, left:col_stop] = True
    return mask


def crop_centered(image: Tensor, support: int | tuple[int, int]) -> Tensor:
    """Crop around the ``fftshift`` discrete origin on each spatial axis.

    After ``fftshift`` the origin is at ``N // 2``.  A support of length ``K``
    therefore starts at ``N // 2 - K // 2`` and places the origin at index
    ``K // 2`` in the cropped array, including for even ``N`` and/or ``K``.
    """

    if not isinstance(image, Tensor) or image.ndim < 2:
        raise ValueError("image must have at least two dimensions")
    if isinstance(support, int):
        kh = kw = support
    elif isinstance(support, tuple) and len(support) == 2:
        kh, kw = support
    else:
        raise ValueError(f"support must be an integer or pair, got {support!r}")
    if isinstance(kh, bool) or isinstance(kw, bool) or not isinstance(kh, int) or not isinstance(kw, int):
        raise ValueError("support dimensions must be integers")
    height, width = image.shape[-2:]
    if kh <= 0 or kw <= 0 or kh > height or kw > width:
        raise ValueError(
            f"support {(kh, kw)} must be positive and fit image spatial shape {(height, width)}"
        )
    top = height // 2 - kh // 2
    left = width // 2 - kw // 2
    bottom = top + kh
    right = left + kw
    if top < 0 or left < 0 or bottom > height or right > width:
        raise ValueError(
            f"support {(kh, kw)} produces invalid crop "
            f"[(top={top}, bottom={bottom}), (left={left}, right={right})] "
            f"for image spatial shape {(height, width)}"
        )
    return image[..., top:bottom, left:right]


def centered_fft(field: Tensor) -> Tensor:
    if field.ndim < 2:
        raise ValueError("field must have at least two dimensions")
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(field, dim=(-2, -1)), dim=(-2, -1)),
        dim=(-2, -1),
    )


def _circular_aperture(config: PhasePSFConfig, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    coordinates = (
        torch.arange(config.phase_grid_size, device=device, dtype=dtype)
        - (config.phase_grid_size - 1) / 2
    ) * config.phase_pitch_m
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    return ((xx.square() + yy.square()) <= (config.aperture_diameter_m / 2) ** 2).to(dtype)


def phase_to_full_psf(
    theta: Tensor,
    *,
    config: PhasePSFConfig | None = None,
    aperture: Tensor | None = None,
) -> Tensor:
    """Convert a trainable phase map to one full-plane normalized intensity PSF."""

    config = config or PhasePSFConfig()
    if theta.ndim != 2 or tuple(theta.shape) != (config.phase_grid_size, config.phase_grid_size):
        raise ValueError(
            "theta must be a square phase map with shape "
            f"{(config.phase_grid_size, config.phase_grid_size)}, got {tuple(theta.shape)}"
        )
    if not torch.is_floating_point(theta):
        theta = theta.float()
    if not torch.isfinite(theta).all():
        raise ValueError("theta must be finite")
    if aperture is None:
        aperture = _circular_aperture(config, device=theta.device, dtype=theta.dtype)
    elif tuple(aperture.shape) != tuple(theta.shape):
        raise ValueError(f"aperture must match theta shape, got {tuple(aperture.shape)}")
    else:
        aperture = aperture.to(device=theta.device, dtype=theta.dtype)
    if torch.count_nonzero(aperture).item() == 0:
        raise ValueError("aperture must contain at least one active cell")
    field = aperture * torch.exp(1j * theta)
    spectrum = centered_fft(field)
    psf = spectrum.real.square() + spectrum.imag.square()
    if not torch.isfinite(psf).all() or torch.any(psf < 0):
        raise ValueError("full PSF is not finite and non-negative")
    energy = psf.sum()
    if not torch.isfinite(energy) or float(energy.detach().item()) <= 0:
        raise ValueError("full PSF has no finite positive energy")
    return psf / energy


class PhaseOnlyPSF(nn.Module):
    """Trainable phase map with an explicit circular aperture and full PSF output."""

    def __init__(
        self,
        config: PhasePSFConfig | None = None,
        *,
        theta: Tensor | None = None,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.config = config or PhasePSFConfig()
        if theta is None:
            theta = torch.zeros(self.config.phase_grid_size, self.config.phase_grid_size)
        if tuple(theta.shape) != (self.config.phase_grid_size, self.config.phase_grid_size):
            raise ValueError("theta has the wrong phase-grid shape")
        if not torch.is_floating_point(theta):
            theta = theta.float()
        self.theta = nn.Parameter(theta.detach().clone(), requires_grad=trainable)
        self.register_buffer(
            "aperture",
            _circular_aperture(self.config, device=theta.device, dtype=theta.dtype),
        )

    def forward(self) -> Tensor:
        return phase_to_full_psf(self.theta, config=self.config, aperture=self.aperture)

    def sampling_metadata(self) -> DetectorSamplingMetadata:
        return detector_sampling_metadata(self.config)


def _normalize_mode(mode: str) -> str:
    return mode.replace("-", "_")


@dataclass(frozen=True)
class SupportResponseReport:
    support: int
    tau: float
    leakage: float
    relative_response_l2: float
    max_absolute_response_error: float
    impulse_response_error: float
    random_nonnegative_response_error: float
    random_signed_dual_rail_response_error: float
    internal_region_error: float
    boundary_region_error: float
    full_image_response_error: float = 0.0

    @property
    def response_error(self) -> float:
        return max(
            self.relative_response_l2,
            self.max_absolute_response_error,
            self.impulse_response_error,
            self.random_nonnegative_response_error,
            self.random_signed_dual_rail_response_error,
            self.internal_region_error,
            self.boundary_region_error,
            self.full_image_response_error,
        )

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["response_error"] = self.response_error
        return result


@dataclass(frozen=True)
class SupportSweepResult:
    reports: tuple[SupportResponseReport, ...]
    reference_support: int
    mode: str
    max_support_response_error: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "reference_support": self.reference_support,
            "mode": self.mode,
            "max_support_response_error": self.max_support_response_error,
            "reports": [report.as_dict() for report in self.reports],
        }


class SupportResponseError(ValueError):
    def __init__(self, message: str, report: SupportResponseReport) -> None:
        super().__init__(message)
        self.report = report


def _same_physical_response(x: Tensor, psf: Tensor) -> Tensor:
    top, bottom, left, right = _same_padding_extents(tuple(psf.shape[-2:]))
    padded = F.pad(x, (left, right, top, bottom))
    return convolve_psf(padded, psf, padding=0)


def _relative_l2(actual: Tensor, reference: Tensor) -> float:
    denominator = float(torch.linalg.vector_norm(reference).detach().item())
    numerator = float(torch.linalg.vector_norm(actual - reference).detach().item())
    return numerator / denominator if denominator > 0 else (0.0 if numerator == 0 else math.inf)


def support_sweep(
    full_psf: Tensor,
    supports: Iterable[int],
    *,
    seed: int = 0,
    max_support_response_error: float | None = None,
    mode: str = "report_only",
) -> SupportSweepResult:
    """Compare raw finite-support responses with the complete PSF reference."""

    if full_psf.ndim == 4 and full_psf.shape[:2] == (1, 1):
        full_psf_2d = full_psf[0, 0]
    elif full_psf.ndim == 2:
        full_psf_2d = full_psf
    else:
        raise ValueError("full_psf must be [H,W] or [1,1,H,W]")
    if not torch.is_floating_point(full_psf_2d) or not torch.isfinite(full_psf_2d).all():
        raise ValueError("full_psf must be finite and floating-point")
    if full_psf_2d.shape[-2] != full_psf_2d.shape[-1]:
        raise ValueError("full_psf must be square")
    support_values = list(supports)
    if not support_values or any(isinstance(value, bool) or not isinstance(value, int) for value in support_values):
        raise ValueError("supports must be a non-empty sequence of integers")
    if any(value <= 0 or value > min(full_psf_2d.shape) for value in support_values):
        raise ValueError("every support must be positive and fit full_psf")
    if len(set(support_values)) != len(support_values):
        raise ValueError("supports must be unique")
    normalized_mode = _normalize_mode(mode)
    if normalized_mode not in {"report_only", "fail_closed"}:
        raise ValueError("mode must be report_only or fail_closed")
    threshold = math.inf if max_support_response_error is None else max_support_response_error
    if max_support_response_error is not None and (
        not _is_finite_number(threshold) or float(threshold) < 0
    ):
        raise ValueError("max_support_response_error must be finite and non-negative")

    reference_support = int(full_psf_2d.shape[-1])
    reference_kernel = full_psf_2d
    input_size = max(2 * reference_support + 5, 17)
    impulse = torch.zeros(1, 1, input_size, input_size, dtype=full_psf_2d.dtype, device=full_psf_2d.device)
    impulse[:, :, input_size // 2, input_size // 2] = 1
    generator = torch.Generator(device=full_psf_2d.device)
    generator.manual_seed(seed)
    random_nonnegative = torch.rand(
        (1, 1, input_size, input_size), generator=generator, dtype=full_psf_2d.dtype, device=full_psf_2d.device
    )
    random_signed = torch.randn(
        (1, 1, input_size, input_size), generator=generator, dtype=full_psf_2d.dtype, device=full_psf_2d.device
    )
    full_image = torch.linspace(
        -1.0,
        1.0,
        input_size * input_size,
        dtype=full_psf_2d.dtype,
        device=full_psf_2d.device,
    ).view(1, 1, input_size, input_size)
    reference_outputs = (
        _same_physical_response(impulse, reference_kernel),
        _same_physical_response(random_nonnegative, reference_kernel),
        _same_physical_response(random_signed.clamp_min(0), reference_kernel)
        - _same_physical_response((-random_signed).clamp_min(0), reference_kernel),
        _same_physical_response(full_image, reference_kernel),
    )
    reports: list[SupportResponseReport] = []
    for support in support_values:
        kernel = crop_centered(full_psf_2d, support)
        output_impulse = _same_physical_response(impulse, kernel)
        output_nonnegative = _same_physical_response(random_nonnegative, kernel)
        output_signed = (
            _same_physical_response(random_signed.clamp_min(0), kernel)
            - _same_physical_response((-random_signed).clamp_min(0), kernel)
        )
        output_full_image = _same_physical_response(full_image, kernel)
        errors = (
            output_impulse - reference_outputs[0],
            output_nonnegative - reference_outputs[1],
            output_signed - reference_outputs[2],
            output_full_image - reference_outputs[3],
        )
        all_error = torch.cat([error.reshape(-1) for error in errors])
        internal_mask_2d = _same_internal_mask(
            tuple(output_nonnegative.shape[-2:]),
            tuple(reference_kernel.shape[-2:]),
            device=output_nonnegative.device,
        )
        internal_mask = internal_mask_2d.unsqueeze(0).unsqueeze(0).expand_as(output_nonnegative)
        boundary_mask = ~internal_mask
        internal_actual = output_nonnegative[internal_mask]
        internal_reference = reference_outputs[1][internal_mask]
        boundary_actual = output_nonnegative[boundary_mask]
        boundary_reference = reference_outputs[1][boundary_mask]
        report = SupportResponseReport(
            support=support,
            tau=float(kernel.sum().detach().item()),
            leakage=float(1.0 - kernel.sum().detach().item()),
            relative_response_l2=_relative_l2(output_nonnegative, reference_outputs[1]),
            max_absolute_response_error=float(all_error.abs().max().detach().item()),
            impulse_response_error=_relative_l2(output_impulse, reference_outputs[0]),
            random_nonnegative_response_error=_relative_l2(output_nonnegative, reference_outputs[1]),
            random_signed_dual_rail_response_error=_relative_l2(output_signed, reference_outputs[2]),
            internal_region_error=_relative_l2(internal_actual, internal_reference),
            boundary_region_error=_relative_l2(boundary_actual, boundary_reference),
            full_image_response_error=_relative_l2(output_full_image, reference_outputs[3]),
        )
        reports.append(report)
        if normalized_mode == "fail_closed" and report.response_error > threshold:
            raise SupportResponseError(
                f"support {support} response error {report.response_error:.6g} exceeds "
                f"fail-closed threshold {threshold:.6g}",
                report,
            )
    return SupportSweepResult(
        reports=tuple(reports),
        reference_support=reference_support,
        mode=normalized_mode,
        max_support_response_error=threshold,
    )


@dataclass(frozen=True)
class ActiveRoute:
    route_id: str
    group_id: str
    component_index: int
    sign: str
    lobe: Tensor
    alpha: float


@dataclass(frozen=True)
class _ActiveRouteIdentitySnapshot:
    route_id: str
    group_id: str
    component_index: int
    sign: str
    alpha: float
    alpha_hex: str
    lobe_shape: tuple[int, ...]
    lobe_dtype: str
    lobe_digest: str
    digest: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "group_id": self.group_id,
            "component_index": self.component_index,
            "sign": self.sign,
            "alpha": self.alpha,
            "alpha_hex": self.alpha_hex,
            "lobe_shape": self.lobe_shape,
            "lobe_dtype": self.lobe_dtype,
            "lobe_digest": self.lobe_digest,
            "digest": self.digest,
        }


@dataclass(frozen=True, eq=False)
class _RouteRuntimeIdentity:
    route_ref: ActiveRoute
    route_id: str
    group_id: str
    component_index: int
    sign: str
    alpha: float
    lobe_ref: Tensor
    lobe_version: int
    lobe_shape: tuple[int, ...]
    lobe_dtype: str


@dataclass(frozen=True, eq=False)
class _ExposureRuntimeIdentity:
    exposure_ref: Exposure
    exposure_id: int
    input_rail: str
    route_ids: tuple[str, ...]
    beta_by_route_type: type
    beta_by_route: tuple[tuple[Any, Any], ...]
    sum_beta: float
    unused_budget: float


@dataclass(frozen=True, eq=False)
class _ExposureLedgerRuntimeIdentity:
    policy: str
    active_group_count: int
    active_route_count: int
    exposures_ref: tuple[Exposure, ...]
    exposures_length: int
    canonical_routes_ref: tuple[ActiveRoute, ...]
    canonical_routes_length: int
    routes: tuple[_RouteRuntimeIdentity, ...]
    exposures: tuple[_ExposureRuntimeIdentity, ...]


def _validate_active_route(route: ActiveRoute, *, name: str = "route") -> None:
    if not isinstance(route, ActiveRoute):
        raise ValueError(f"{name} must be an ActiveRoute")
    if not isinstance(route.route_id, str) or not route.route_id:
        raise ValueError(f"{name} route_id must be a non-empty string")
    if not isinstance(route.group_id, str) or not route.group_id:
        raise ValueError(f"{name} group_id must be a non-empty string")
    if route.sign not in {"positive", "negative"}:
        raise ValueError(f"{name} sign must be 'positive' or 'negative'")
    if (
        isinstance(route.component_index, bool)
        or not isinstance(route.component_index, int)
        or route.component_index < 0
    ):
        raise ValueError(f"{name} component_index must be a non-negative integer")
    expected_route_id = f"{route.group_id}:component{route.component_index}:{route.sign}"
    if route.route_id != expected_route_id:
        raise ValueError(
            f"{name} route_id {route.route_id!r} does not match its group/component/sign identity"
        )
    if not _is_finite_number(route.alpha) or float(route.alpha) <= 0:
        raise ValueError(f"{name} alpha must be finite and strictly positive")
    if not isinstance(route.lobe, Tensor) or route.lobe.ndim < 1:
        raise ValueError(f"{name} lobe must be a non-empty tensor")
    if not torch.is_floating_point(route.lobe) or route.lobe.numel() == 0:
        raise ValueError(f"{name} lobe must be a non-empty floating-point tensor")
    if not torch.isfinite(route.lobe).all():
        raise ValueError(f"{name} lobe must be finite")
    if torch.any(route.lobe < 0):
        raise ValueError(f"{name} lobe must be non-negative")
    lobe_sum = float(route.lobe.detach().sum().item())
    if lobe_sum <= 0 or not math.isclose(lobe_sum, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{name} lobe must be normalized to unit energy")


def _identity_field(name: str, value: bytes) -> bytes:
    name_bytes = name.encode("utf-8")
    return (
        str(len(name_bytes)).encode("ascii")
        + b":"
        + name_bytes
        + str(len(value)).encode("ascii")
        + b":"
        + value
        + b";"
    )


def _active_route_identity_snapshot(route: ActiveRoute) -> _ActiveRouteIdentitySnapshot:
    lobe_cpu = route.lobe.detach().to(device="cpu").contiguous()
    lobe_bytes = lobe_cpu.view(torch.uint8).numpy().tobytes()
    route_id = route.route_id.encode("utf-8")
    group_id = route.group_id.encode("utf-8")
    component_index = str(route.component_index).encode("ascii")
    sign = route.sign.encode("utf-8")
    alpha = float(route.alpha)
    alpha_hex = alpha.hex()
    lobe_shape = tuple(int(size) for size in route.lobe.shape)
    lobe_dtype = str(route.lobe.dtype)
    payload = b"ActiveRouteIdentity:v1;" + b"".join(
        (
            _identity_field("route_id", route_id),
            _identity_field("group_id", group_id),
            _identity_field("component_index", component_index),
            _identity_field("sign", sign),
            _identity_field("alpha_hex", alpha_hex.encode("ascii")),
            _identity_field("lobe_shape", repr(lobe_shape).encode("ascii")),
            _identity_field("lobe_dtype", lobe_dtype.encode("ascii")),
            _identity_field("lobe_bytes", lobe_bytes),
        )
    )
    return _ActiveRouteIdentitySnapshot(
        route_id=route.route_id,
        group_id=route.group_id,
        component_index=route.component_index,
        sign=route.sign,
        alpha=alpha,
        alpha_hex=alpha_hex,
        lobe_shape=lobe_shape,
        lobe_dtype=lobe_dtype,
        lobe_digest=hashlib.sha256(lobe_bytes).hexdigest(),
        digest=hashlib.sha256(payload).hexdigest(),
    )


def _route_identity_matches(canonical: ActiveRoute, candidate: ActiveRoute) -> bool:
    if canonical is candidate:
        return True
    return (
        canonical.route_id == candidate.route_id
        and canonical.group_id == candidate.group_id
        and canonical.component_index == candidate.component_index
        and canonical.sign == candidate.sign
        and canonical.alpha == candidate.alpha
        and canonical.lobe.shape == candidate.lobe.shape
        and canonical.lobe.dtype == candidate.lobe.dtype
        and (
            canonical.lobe is candidate.lobe
            or torch.equal(canonical.lobe.detach().cpu(), candidate.lobe.detach().cpu())
        )
    )


def _active_route_as_dict(route: ActiveRoute) -> dict[str, Any]:
    return {
        "route_id": route.route_id,
        "group_id": route.group_id,
        "component_index": route.component_index,
        "sign": route.sign,
        "alpha": route.alpha,
        "lobe_shape": tuple(route.lobe.shape),
        "lobe_dtype": str(route.lobe.dtype),
        "lobe": route.lobe.detach().cpu().tolist(),
    }


def _group_items(groups: Mapping[str, Any] | Sequence[Any]) -> list[tuple[str, Any]]:
    if isinstance(groups, Mapping):
        return [(str(key), value) for key, value in groups.items()]
    result: list[tuple[str, Any]] = []
    for index, value in enumerate(groups):
        if isinstance(value, tuple) and len(value) == 2:
            result.append((str(value[0]), value[1]))
        else:
            result.append((f"group{index}", value))
    return result


def _components(value: Any) -> Tensor:
    if isinstance(value, CenteredPCAResult):
        value = value.component_bank
    if not isinstance(value, Tensor):
        raise ValueError("each group must contain a tensor or CenteredPCAResult")
    if value.ndim == 3:
        value = value.unsqueeze(0)
    if value.ndim != 4:
        raise ValueError("group components must have shape [components, channels, H, W]")
    return value


def build_active_routes(groups: Mapping[str, Any] | Sequence[Any]) -> tuple[ActiveRoute, ...]:
    routes: list[ActiveRoute] = []
    for group_id, value in _group_items(groups):
        for component_index, component in enumerate(_components(value)):
            lobes = signed_kernel_lobes(component)
            if lobes.alpha_pos > 0:
                routes.append(
                    ActiveRoute(
                        route_id=f"{group_id}:component{component_index}:positive",
                        group_id=group_id,
                        component_index=component_index,
                        sign="positive",
                        lobe=lobes.p_pos,
                        alpha=lobes.alpha_pos,
                    )
                )
            if lobes.alpha_neg > 0:
                routes.append(
                    ActiveRoute(
                        route_id=f"{group_id}:component{component_index}:negative",
                        group_id=group_id,
                        component_index=component_index,
                        sign="negative",
                        lobe=lobes.p_neg,
                        alpha=lobes.alpha_neg,
                    )
                )
    return tuple(routes)


@dataclass(frozen=True)
class Exposure:
    exposure_id: int
    input_rail: str
    route_ids: tuple[str, ...]
    beta_by_route: dict[str, float]
    sum_beta: float
    unused_budget: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.input_rail not in {"pos", "neg"}:
            raise ValueError(f"input_rail must be 'pos' or 'neg', got {self.input_rail!r}")
        if not self.route_ids:
            raise ValueError("an exposure must contain at least one route")
        if len(set(self.route_ids)) != len(self.route_ids):
            raise ValueError("duplicate route in one exposure")
        if set(self.beta_by_route) != set(self.route_ids):
            raise ValueError("route IDs and beta allocation keys must match")
        for route_id, beta in self.beta_by_route.items():
            if not isinstance(route_id, str) or not route_id:
                raise ValueError("route IDs must be non-empty strings")
            _require_unit_interval(beta, f"beta for route {route_id!r}")
        computed_sum = float(sum(float(beta) for beta in self.beta_by_route.values()))
        if not _is_finite_number(self.sum_beta) or not math.isclose(
            float(self.sum_beta), computed_sum, rel_tol=0.0, abs_tol=1e-7
        ):
            raise ValueError("sum_beta must equal the route beta allocation sum")
        if computed_sum > 1.0 + 1e-7:
            raise ValueError("sum_beta exceeds the one-exposure budget")
        expected_unused = 1.0 - computed_sum
        if not _is_finite_number(self.unused_budget) or not math.isclose(
            float(self.unused_budget), expected_unused, rel_tol=0.0, abs_tol=1e-7
        ):
            raise ValueError("unused budget does not match sum_beta")

    def as_dict(self) -> dict[str, Any]:
        return {
            "exposure_id": self.exposure_id,
            "input_rail": self.input_rail,
            "route_ids": list(self.route_ids),
            "beta_by_route": dict(self.beta_by_route),
            "sum_beta": self.sum_beta,
            "unused_budget": self.unused_budget,
        }


@dataclass(frozen=True)
class ExposureLedger:
    policy: str
    active_group_count: int
    active_route_count: int
    exposures: tuple[Exposure, ...]
    canonical_routes: tuple[ActiveRoute, ...] = ()
    _canonical_route_identity_snapshots: tuple[_ActiveRouteIdentitySnapshot, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _runtime_identity_snapshot: _ExposureLedgerRuntimeIdentity = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        self._validate_canonical_routes()
        object.__setattr__(
            self,
            "_canonical_route_identity_snapshots",
            tuple(_active_route_identity_snapshot(route) for route in self.canonical_routes),
        )
        self.validate()
        self._capture_runtime_identity()

    @property
    def exposure_count(self) -> int:
        return len(self.exposures)

    def _validate_canonical_routes(self) -> tuple[set[str], set[str]]:
        if not isinstance(self.canonical_routes, tuple) or not self.canonical_routes:
            raise ValueError("exposure ledger must contain canonical routes")
        canonical_ids: set[str] = set()
        canonical_group_ids: set[str] = set()
        for route in self.canonical_routes:
            _validate_active_route(route, name="canonical route")
            if route.route_id in canonical_ids:
                raise ValueError(f"duplicate canonical route ID {route.route_id!r}")
            canonical_ids.add(route.route_id)
            canonical_group_ids.add(route.group_id)
        return canonical_ids, canonical_group_ids

    def validate(self) -> None:
        if not self.exposures:
            raise ValueError("exposure ledger must contain at least one exposure")
        if len(self.exposures) % 2:
            raise ValueError("sequential dual-rail ledger requires paired exposures")
        canonical_ids, canonical_group_ids = self._validate_canonical_routes()
        current_snapshots = tuple(
            _active_route_identity_snapshot(route) for route in self.canonical_routes
        )
        expected_snapshots = getattr(self, "_canonical_route_identity_snapshots", None)
        if expected_snapshots is None or len(expected_snapshots) != len(current_snapshots):
            raise ValueError("exposure ledger is missing canonical route creation identities")
        for route, expected, current in zip(
            self.canonical_routes,
            expected_snapshots,
            current_snapshots,
        ):
            if current != expected:
                raise ValueError(
                    f"canonical route {route.route_id!r} identity changed since ledger creation"
                )
        if (
            isinstance(self.active_route_count, bool)
            or not isinstance(self.active_route_count, int)
            or self.active_route_count != len(canonical_ids)
        ):
            raise ValueError("active_route_count does not match canonical routes")
        if (
            isinstance(self.active_group_count, bool)
            or not isinstance(self.active_group_count, int)
            or self.active_group_count != len(canonical_group_ids)
            or self.active_group_count <= 0
        ):
            raise ValueError("active_group_count does not match canonical groups")
        exposure_ids = [exposure.exposure_id for exposure in self.exposures]
        if len(set(exposure_ids)) != len(exposure_ids):
            raise ValueError("exposure IDs must be unique")
        for exposure in self.exposures:
            exposure.validate()

        routes_by_rail: dict[str, set[str]] = {"pos": set(), "neg": set()}
        for first, second in zip(self.exposures[::2], self.exposures[1::2]):
            if {first.input_rail, second.input_rail} != {"pos", "neg"}:
                raise ValueError("each allocation must contain positive and negative input-rail exposures")
            if set(first.route_ids) != set(second.route_ids):
                raise ValueError("paired dual-rail exposures must allocate the same routes")
            for route_id in first.route_ids:
                if not math.isclose(
                    first.beta_by_route[route_id],
                    second.beta_by_route[route_id],
                    rel_tol=0.0,
                    abs_tol=1e-7,
                ):
                    raise ValueError(
                        f"paired dual-rail beta mismatch for route {route_id!r}"
                    )
            for exposure in (first, second):
                routes_by_rail[exposure.input_rail].update(exposure.route_ids)

        for rail, route_ids in routes_by_rail.items():
            seen: set[str] = set()
            for exposure in self.exposures:
                if exposure.input_rail != rail:
                    continue
                duplicates = seen.intersection(exposure.route_ids)
                if duplicates:
                    raise ValueError(
                        f"duplicate route allocation for input rail {rail!r}: {sorted(duplicates)!r}"
                    )
                seen.update(exposure.route_ids)
            if seen != route_ids or seen != canonical_ids:
                raise ValueError(f"invalid route allocation for input rail {rail!r}")

    @staticmethod
    def _runtime_beta_snapshot(exposure: Exposure) -> tuple[tuple[Any, Any], ...]:
        try:
            if any(
                not isinstance(route_id, str) or not _is_finite_number(beta)
                for route_id, beta in exposure.beta_by_route.items()
            ):
                raise ValueError("beta allocation contains a non-scalar key or value")
            return tuple(sorted(exposure.beta_by_route.items()))
        except (TypeError, ValueError) as error:
            raise ValueError("exposure ledger runtime identity drift: beta allocation is malformed") from error

    @staticmethod
    def _runtime_route_identity(route: ActiveRoute) -> _RouteRuntimeIdentity:
        try:
            return _RouteRuntimeIdentity(
                route_ref=route,
                route_id=route.route_id,
                group_id=route.group_id,
                component_index=route.component_index,
                sign=route.sign,
                alpha=route.alpha,
                lobe_ref=route.lobe,
                lobe_version=_tensor_version(route.lobe),
                lobe_shape=tuple(int(size) for size in route.lobe.shape),
                lobe_dtype=str(route.lobe.dtype),
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("exposure ledger runtime identity drift: route is malformed") from error

    @classmethod
    def _runtime_exposure_identity(cls, exposure: Exposure) -> _ExposureRuntimeIdentity:
        try:
            route_ids = exposure.route_ids
            if type(route_ids) is not tuple or any(not isinstance(route_id, str) for route_id in route_ids):
                raise ValueError("route IDs are malformed")
            return _ExposureRuntimeIdentity(
                exposure_ref=exposure,
                exposure_id=exposure.exposure_id,
                input_rail=exposure.input_rail,
                route_ids=route_ids,
                beta_by_route_type=type(exposure.beta_by_route),
                beta_by_route=cls._runtime_beta_snapshot(exposure),
                sum_beta=exposure.sum_beta,
                unused_budget=exposure.unused_budget,
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("exposure ledger runtime identity drift: exposure is malformed") from error

    def _capture_runtime_identity(self) -> None:
        object.__setattr__(
            self,
            "_runtime_identity_snapshot",
            _ExposureLedgerRuntimeIdentity(
                policy=self.policy,
                active_group_count=self.active_group_count,
                active_route_count=self.active_route_count,
                exposures_ref=self.exposures,
                exposures_length=len(self.exposures),
                canonical_routes_ref=self.canonical_routes,
                canonical_routes_length=len(self.canonical_routes),
                routes=tuple(self._runtime_route_identity(route) for route in self.canonical_routes),
                exposures=tuple(self._runtime_exposure_identity(exposure) for exposure in self.exposures),
            ),
        )

    @staticmethod
    def _same_scalar(expected: Any, current: Any) -> bool:
        return type(current) is type(expected) and current == expected

    def validate_runtime(self) -> None:
        """Validate creation-time ledger bindings without scanning tensor contents."""

        expected = getattr(self, "_runtime_identity_snapshot", None)
        if not isinstance(expected, _ExposureLedgerRuntimeIdentity):
            raise ValueError("exposure ledger runtime identity baseline is unavailable")
        if not self._same_scalar(expected.policy, self.policy):
            raise ValueError("exposure ledger runtime identity drift: policy changed")
        if not self._same_scalar(expected.active_group_count, self.active_group_count):
            raise ValueError("exposure ledger runtime identity drift: active group count changed")
        if not self._same_scalar(expected.active_route_count, self.active_route_count):
            raise ValueError("exposure ledger runtime identity drift: active route count changed")
        if self.exposures is not expected.exposures_ref or len(self.exposures) != expected.exposures_length:
            raise ValueError("exposure ledger runtime identity drift: exposures tuple changed")
        if (
            self.canonical_routes is not expected.canonical_routes_ref
            or len(self.canonical_routes) != expected.canonical_routes_length
        ):
            raise ValueError("exposure ledger runtime identity drift: canonical routes tuple changed")
        if len(self.canonical_routes) != len(expected.routes):
            raise ValueError("exposure ledger runtime identity drift: route count changed")
        for route, route_expected in zip(self.canonical_routes, expected.routes):
            if route is not route_expected.route_ref:
                raise ValueError("exposure ledger runtime identity drift: canonical route object changed")
            if not self._same_scalar(route_expected.route_id, route.route_id):
                raise ValueError("exposure ledger runtime identity drift: route ID changed")
            if not self._same_scalar(route_expected.group_id, route.group_id):
                raise ValueError("exposure ledger runtime identity drift: route group changed")
            if not self._same_scalar(route_expected.component_index, route.component_index):
                raise ValueError("exposure ledger runtime identity drift: route component changed")
            if not self._same_scalar(route_expected.sign, route.sign):
                raise ValueError("exposure ledger runtime identity drift: route sign changed")
            if not self._same_scalar(route_expected.alpha, route.alpha):
                raise ValueError("exposure ledger runtime identity drift: route alpha changed")
            if route.lobe is not route_expected.lobe_ref:
                raise ValueError("exposure ledger runtime identity drift: route lobe object changed")
            if _tensor_version(route.lobe) != route_expected.lobe_version:
                raise ValueError("exposure ledger runtime identity drift: route lobe version changed")
            if tuple(int(size) for size in route.lobe.shape) != route_expected.lobe_shape:
                raise ValueError("exposure ledger runtime identity drift: route lobe shape changed")
            if str(route.lobe.dtype) != route_expected.lobe_dtype:
                raise ValueError("exposure ledger runtime identity drift: route lobe dtype changed")
        if len(self.exposures) != len(expected.exposures):
            raise ValueError("exposure ledger runtime identity drift: exposure count changed")
        for exposure, exposure_expected in zip(self.exposures, expected.exposures):
            if exposure is not exposure_expected.exposure_ref:
                raise ValueError("exposure ledger runtime identity drift: exposure object changed")
            if not self._same_scalar(exposure_expected.exposure_id, exposure.exposure_id):
                raise ValueError("exposure ledger runtime identity drift: exposure ID changed")
            if not self._same_scalar(exposure_expected.input_rail, exposure.input_rail):
                raise ValueError("exposure ledger runtime identity drift: input rail changed")
            if type(exposure.route_ids) is not tuple or any(
                not isinstance(route_id, str) for route_id in exposure.route_ids
            ):
                raise ValueError("exposure ledger runtime identity drift: route IDs are malformed")
            if exposure.route_ids != exposure_expected.route_ids:
                raise ValueError("exposure ledger runtime identity drift: route IDs changed")
            if type(exposure.beta_by_route) is not exposure_expected.beta_by_route_type:
                raise ValueError("exposure ledger runtime identity drift: beta mapping type changed")
            if self._runtime_beta_snapshot(exposure) != exposure_expected.beta_by_route:
                raise ValueError("exposure ledger runtime identity drift: beta allocation changed")
            if not self._same_scalar(exposure_expected.sum_beta, exposure.sum_beta):
                raise ValueError("exposure ledger runtime identity drift: sum_beta changed")
            if not self._same_scalar(exposure_expected.unused_budget, exposure.unused_budget):
                raise ValueError("exposure ledger runtime identity drift: unused budget changed")

    def beta_for_route(self, route_id: str, *, input_rail: str) -> float:
        """Return the validated beta assigned to one route and input rail."""

        self.validate_runtime()
        if input_rail not in {"pos", "neg"}:
            raise ValueError(f"input_rail must be 'pos' or 'neg', got {input_rail!r}")
        matches = [
            exposure
            for exposure in self.exposures
            if exposure.input_rail == input_rail and route_id in exposure.route_ids
        ]
        if not matches:
            raise ValueError(
                f"route {route_id!r} is not allocated on input rail {input_rail!r}"
            )
        if len(matches) != 1:
            raise ValueError(
                f"duplicate route allocation for route {route_id!r} on input rail {input_rail!r}"
            )
        return float(matches[0].beta_by_route[route_id])

    def resolve_dual_rail_beta(self, route_id: str) -> float:
        """Resolve one route's beta only after both sequential rails are present."""

        positive_beta = self.beta_for_route(route_id, input_rail="pos")
        negative_beta = self.beta_for_route(route_id, input_rail="neg")
        if not math.isclose(positive_beta, negative_beta, rel_tol=0.0, abs_tol=1e-7):
            raise ValueError(f"paired dual-rail beta mismatch for route {route_id!r}")
        return positive_beta

    def route_for_id(self, route_id: str) -> ActiveRoute:
        """Return a canonical route after low-cost runtime identity validation."""

        self.validate_runtime()
        if not isinstance(route_id, str) or not route_id:
            raise ValueError("route_id must be a non-empty string")
        for route in self.canonical_routes:
            if route.route_id == route_id:
                return route
        raise ValueError(f"route {route_id!r} is not present in canonical routes")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "policy": self.policy,
            "active_group_count": self.active_group_count,
            "active_route_count": self.active_route_count,
            "exposure_count": self.exposure_count,
            "canonical_routes": [_active_route_as_dict(route) for route in self.canonical_routes],
            "canonical_route_identity_digests": [
                snapshot.digest for snapshot in self._canonical_route_identity_snapshots
            ],
            "canonical_route_identities": [
                snapshot.as_dict() for snapshot in self._canonical_route_identity_snapshots
            ],
            "exposures": [exposure.as_dict() for exposure in self.exposures],
        }


def build_exposure_ledger(
    groups_or_routes: Mapping[str, Any] | Sequence[Any],
    *,
    policy: str = "shared_input_global_parallel_equal_split",
    route_batch_capacity: int | None = None,
) -> ExposureLedger:
    """Build a pure logical exposure ledger with one shared budget per rail."""

    normalized_policy = _normalize_mode(policy)
    aliases = {
        "shared_input_global_parallel_equal_split": "shared_input_global_parallel_equal_split",
        "global_parallel": "global_parallel",
        "group_sequential": "group_sequential",
        "route_sequential": "route_sequential",
        "bounded_parallel": "bounded_parallel",
    }
    if normalized_policy not in aliases:
        raise ValueError(f"unsupported exposure policy {policy!r}")
    if normalized_policy == "bounded_parallel":
        if isinstance(route_batch_capacity, bool) or not isinstance(route_batch_capacity, int) or route_batch_capacity <= 0:
            raise ValueError("route_batch_capacity must be a positive integer for bounded_parallel")

    if isinstance(groups_or_routes, Sequence) and all(isinstance(item, ActiveRoute) for item in groups_or_routes):
        routes = list(groups_or_routes)
    else:
        routes = list(build_active_routes(groups_or_routes))
    if not routes:
        raise ValueError("at least one active route is required")
    for route in routes:
        _validate_active_route(route)
    route_ids = [route.route_id for route in routes]
    if len(set(route_ids)) != len(route_ids):
        raise ValueError("route IDs must be unique")
    group_ids: list[str] = []
    for route in routes:
        if route.group_id not in group_ids:
            group_ids.append(route.group_id)

    chunks: list[list[ActiveRoute]]
    if normalized_policy in {"global_parallel", "shared_input_global_parallel_equal_split"}:
        chunks = [routes]
    elif normalized_policy == "group_sequential":
        chunks = [[route for route in routes if route.group_id == group_id] for group_id in group_ids]
    elif normalized_policy == "route_sequential":
        chunks = [[route] for route in routes]
    else:
        assert route_batch_capacity is not None
        chunks = [routes[index : index + route_batch_capacity] for index in range(0, len(routes), route_batch_capacity)]

    exposures: list[Exposure] = []
    exposure_id = 0
    for chunk in chunks:
        if normalized_policy == "bounded_parallel":
            assert route_batch_capacity is not None
            beta = 1.0 / route_batch_capacity
        else:
            beta = 1.0 / len(chunk)
        route_ids_for_chunk = tuple(route.route_id for route in chunk)
        beta_by_route = {route_id: beta for route_id in route_ids_for_chunk}
        sum_beta = float(sum(beta_by_route.values()))
        for input_rail in ("pos", "neg"):
            exposures.append(
                Exposure(
                    exposure_id=exposure_id,
                    input_rail=input_rail,
                    route_ids=route_ids_for_chunk,
                    beta_by_route=beta_by_route,
                    sum_beta=sum_beta,
                    unused_budget=1.0 - sum_beta,
                )
            )
            exposure_id += 1
    return ExposureLedger(
        policy=(
            "shared_input_global_parallel_equal_split"
            if normalized_policy == "shared_input_global_parallel_equal_split"
            else normalized_policy
        ),
        active_group_count=len(group_ids),
        active_route_count=len(routes),
        exposures=tuple(exposures),
        canonical_routes=tuple(routes),
    )


def convolve_psf(
    x: Tensor,
    psf: Tensor,
    *,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
) -> Tensor:
    """Apply physical convolution through one PSF-to-correlation conversion.

    ``psf`` is a physical convolution impulse response.  PyTorch's
    :func:`torch.nn.functional.conv2d` performs cross-correlation, so the
    physical PSF is flipped exactly once at this propagation boundary.
    """

    if psf.ndim == 2:
        psf = psf.unsqueeze(0).unsqueeze(0)
    elif psf.ndim == 3:
        psf = psf.unsqueeze(0)
    if psf.ndim != 4:
        raise ValueError("psf must be [H,W], [C,H,W], or [1,C,H,W]")
    if psf.shape[0] != 1:
        raise ValueError("a single PSF response expects one output kernel")
    return F.conv2d(
        x,
        physical_psf_to_correlation_kernel(psf),
        stride=stride,
        padding=padding,
    )


def physical_psf_from_digital_lobe(digital_lobe: Tensor) -> Tensor:
    """Construct a physical PSF from a non-negative digital correlation lobe.

    PCA and digital convolution use the correlation-kernel orientation.  The
    corresponding physical impulse response is its 180-degree spatial flip;
    propagation flips this physical PSF back once in :func:`convolve_psf`.
    """

    if not isinstance(digital_lobe, Tensor) or digital_lobe.ndim < 2:
        raise ValueError("digital_lobe must be a tensor with at least two dimensions")
    if not torch.is_floating_point(digital_lobe) or not torch.isfinite(digital_lobe).all():
        raise ValueError("digital_lobe must be finite and floating-point")
    if torch.any(digital_lobe < 0):
        raise ValueError("digital_lobe must be non-negative")
    return torch.flip(digital_lobe, dims=(-2, -1))


def physical_psf_to_correlation_kernel(psf: Tensor) -> Tensor:
    """Convert a physical PSF to PyTorch's cross-correlation orientation once."""

    if not isinstance(psf, Tensor) or psf.ndim < 2:
        raise ValueError("psf must be a tensor with at least two dimensions")
    if not torch.is_floating_point(psf) or not torch.isfinite(psf).all():
        raise ValueError("psf must be finite and floating-point")
    return torch.flip(psf, dims=(-2, -1))


def detector_response(
    x_rail: Tensor,
    h_support: Tensor,
    *,
    rho: float,
    beta: float,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
) -> Tensor:
    """Return ``D = rho * beta * convolution(x_rail, h_support)``."""

    _require_positive(rho, "rho")
    beta_value = _require_unit_interval(beta, "beta")
    response = convolve_psf(x_rail, h_support, stride=stride, padding=padding)
    return float(rho) * beta_value * response


@dataclass(frozen=True)
class RouteCalibrationSnapshot:
    alpha: float
    rho: float
    beta: float
    tau: float
    gain: Tensor | float
    active: bool
    near_threshold: dict[str, bool]
    triggered_constraints: tuple[str, ...]
    gain_update_mode: str = "calibrated_detached"

    def as_dict(self) -> dict[str, Any]:
        gain = float(self.gain.detach().item()) if isinstance(self.gain, Tensor) else float(self.gain)
        return {
            "alpha": self.alpha,
            "rho": self.rho,
            "beta": self.beta,
            "tau": self.tau,
            "gain": gain,
            "active": self.active,
            "near_threshold": dict(self.near_threshold),
            "triggered_constraints": list(self.triggered_constraints),
            "gain_update_mode": self.gain_update_mode,
        }


def _near(value: float, threshold: float, fraction: float = 0.05) -> bool:
    return threshold > 0 and abs(value - threshold) <= fraction * threshold


def compute_calibrated_gain(
    *,
    alpha: float,
    rho: float,
    beta: float,
    tau: Tensor | float,
    min_throughput: float = 0.0,
    max_electronic_gain: float = math.inf,
    min_split_fraction: float = 0.0,
) -> Tensor | float:
    """Compute detached ``g = alpha / (rho * beta * tau)`` exactly once."""

    if not _is_finite_number(alpha) or float(alpha) < 0:
        raise ValueError("alpha must be finite and non-negative")
    _require_positive(rho, "rho")
    beta_value = _require_unit_interval(beta, "beta")
    if not _is_finite_number(min_throughput) or float(min_throughput) < 0:
        raise ValueError("min_throughput must be finite and non-negative")
    if max_electronic_gain != math.inf:
        _require_positive(max_electronic_gain, "max_electronic_gain")
    if not _is_finite_number(min_split_fraction) or float(min_split_fraction) < 0:
        raise ValueError("min_split_fraction must be finite and non-negative")

    tau_value = _scalar_value(tau)
    if float(alpha) == 0:
        if isinstance(tau, Tensor):
            return torch.zeros((), dtype=tau.dtype, device=tau.device)
        return 0.0
    if not math.isfinite(tau_value) or tau_value < float(min_throughput):
        raise ValueError(
            f"throughput constraint failed: tau={tau_value!r} < min_throughput={min_throughput!r}"
        )
    if beta_value < float(min_split_fraction):
        raise ValueError(
            f"split fraction constraint failed: beta={beta!r} < min_split_fraction={min_split_fraction!r}"
        )
    if tau_value <= 0 or beta_value <= 0:
        raise ValueError("active routes require positive tau and beta")
    if isinstance(tau, Tensor):
        gain = (
            torch.as_tensor(alpha, dtype=tau.dtype, device=tau.device)
            / (float(rho) * beta_value * tau.detach())
        ).detach()
        gain_value = float(gain.item())
    else:
        gain_value = float(alpha) / (float(rho) * beta_value * tau_value)
        gain = gain_value
    if abs(gain_value) > float(max_electronic_gain):
        raise ValueError(
            f"electronic gain constraint failed: abs(gain)={abs(gain_value)!r} > "
            f"max_electronic_gain={max_electronic_gain!r}"
        )
    return gain


def _snapshot(
    *,
    alpha: float,
    rho: float,
    beta: float,
    tau: Tensor | float,
    gain: Tensor | float,
    min_throughput: float,
    max_electronic_gain: float,
    min_split_fraction: float,
) -> RouteCalibrationSnapshot:
    tau_value = _scalar_value(tau)
    gain_value = _scalar_value(gain)
    return RouteCalibrationSnapshot(
        alpha=float(alpha),
        rho=float(rho),
        beta=float(beta),
        tau=tau_value,
        gain=gain,
        active=float(alpha) > 0,
        near_threshold={
            "throughput": _near(tau_value, float(min_throughput)),
            "gain": _near(abs(gain_value), float(max_electronic_gain))
            if math.isfinite(float(max_electronic_gain))
            else False,
            "split_fraction": _near(float(beta), float(min_split_fraction)),
        },
        triggered_constraints=(),
    )


@dataclass(frozen=True)
class PhysicalForwardResult:
    output: Tensor
    positive: RouteCalibrationSnapshot
    negative: RouteCalibrationSnapshot
    support_error: float | None = None

    @property
    def snapshots(self) -> tuple[RouteCalibrationSnapshot, RouteCalibrationSnapshot]:
        return self.positive, self.negative

    @property
    def global_max_gain(self) -> float:
        return max(abs(_scalar_value(snapshot.gain)) for snapshot in self.snapshots)

    @property
    def minimum_throughput(self) -> float:
        active_taus = [snapshot.tau for snapshot in self.snapshots if snapshot.active]
        return min(active_taus) if active_taus else 0.0

    @property
    def gain_update_mode(self) -> str:
        return "calibrated_detached"

    def as_dict(self) -> dict[str, Any]:
        return {
            "positive": self.positive.as_dict(),
            "negative": self.negative.as_dict(),
            "support_error": self.support_error,
            "global_max_gain": self.global_max_gain,
            "minimum_throughput": self.minimum_throughput,
            "gain_update_mode": self.gain_update_mode,
        }


def _inactive_snapshot(alpha: float, rho: float, beta: float) -> RouteCalibrationSnapshot:
    return RouteCalibrationSnapshot(
        alpha=float(alpha),
        rho=float(rho),
        beta=float(beta),
        tau=0.0,
        gain=0.0,
        active=False,
        near_threshold={"throughput": False, "gain": False, "split_fraction": False},
        triggered_constraints=(),
    )


def physical_dual_rail_forward(
    x: Tensor,
    *,
    positive_psf: Tensor | None,
    negative_psf: Tensor | None,
    positive_route: ActiveRoute | None = None,
    negative_route: ActiveRoute | None = None,
    rho: float | None = None,
    support: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    bias: Tensor | None = None,
    config: OptoelectronicConfig | None = None,
    support_error: float | None = None,
    support_error_mode: str | None = None,
    ledger: ExposureLedger | None = None,
) -> PhysicalForwardResult:
    """Apply the physical detector path with one detached throughput gain.

    Each active lobe uses raw ``h_K`` in the detector response.  Only the
    electronic gain divides by ``rho * beta * tau``.  The ideal Conv2d path
    is intentionally separate in :func:`ideal_signed_conv2d`.
    """

    config = config or OptoelectronicConfig()
    rho = config.rho if rho is None else rho
    _require_positive(rho, "rho")
    if ledger is not None:
        if not isinstance(ledger, ExposureLedger):
            raise ValueError("ledger must be an ExposureLedger")
        ledger.validate_runtime()
    elif positive_route is not None or negative_route is not None:
        raise ValueError(
            "physical_dual_rail_forward requires a validated ExposureLedger for every active route"
        )

    def resolve_route(
        route: ActiveRoute | None,
        *,
        expected_sign: str,
        name: str,
    ) -> ActiveRoute | None:
        if route is None:
            return None
        if ledger is None:
            raise ValueError("an active physical route requires an ExposureLedger")
        if not isinstance(route, ActiveRoute):
            raise ValueError(f"{name} must be an ActiveRoute")
        canonical = ledger.route_for_id(route.route_id)
        if route is not canonical:
            _validate_active_route(route, name=name)
        if not _route_identity_matches(canonical, route):
            raise ValueError(f"{name} does not match the ledger canonical route identity")
        if canonical.sign != expected_sign:
            raise ValueError(
                f"{name} must have sign={expected_sign!r}, got {canonical.sign!r}"
            )
        return canonical

    canonical_positive_route = resolve_route(
        positive_route,
        expected_sign="positive",
        name="positive_route",
    )
    canonical_negative_route = resolve_route(
        negative_route,
        expected_sign="negative",
        name="negative_route",
    )
    if canonical_positive_route is None and positive_psf is not None:
        raise ValueError("positive_psf requires a positive_route")
    if canonical_negative_route is None and negative_psf is not None:
        raise ValueError("negative_psf requires a negative_route")
    if canonical_positive_route is not None and positive_psf is None:
        raise ValueError("an active positive_route requires a positive_psf")
    if canonical_negative_route is not None and negative_psf is None:
        raise ValueError("an active negative_route requires a negative_psf")
    resolved_beta_pos = (
        0.0
        if canonical_positive_route is None
        else ledger.resolve_dual_rail_beta(canonical_positive_route.route_id)
    )
    resolved_beta_neg = (
        0.0
        if canonical_negative_route is None
        else ledger.resolve_dual_rail_beta(canonical_negative_route.route_id)
    )
    mode = config.normalized_support_error_mode if support_error_mode is None else _normalize_mode(support_error_mode)
    if mode not in {"report_only", "fail_closed"}:
        raise ValueError("support_error_mode must be report_only or fail_closed")
    if support_error is not None:
        if not _is_finite_number(support_error) or float(support_error) < 0:
            raise ValueError("support_error must be finite and non-negative")
        if mode == "fail_closed" and float(support_error) > config.max_support_response_error:
            raise SupportResponseError(
                f"support response error {support_error:.6g} exceeds fail-closed threshold "
                f"{config.max_support_response_error:.6g}",
                SupportResponseReport(
                    support=int(support) if isinstance(support, int) else support[0],
                    tau=math.nan,
                    leakage=math.nan,
                    relative_response_l2=float(support_error),
                    max_absolute_response_error=float(support_error),
                    impulse_response_error=float(support_error),
                    random_nonnegative_response_error=float(support_error),
                    random_signed_dual_rail_response_error=float(support_error),
                    internal_region_error=float(support_error),
                    boundary_region_error=float(support_error),
                ),
            )

    x_pos, x_neg = signed_input_rails(x)
    contributions: list[Tensor] = []
    snapshots: list[RouteCalibrationSnapshot] = []
    output_shape: tuple[int, int] | None = None

    def route_contribution(
        full_psf: Tensor | None,
        route: ActiveRoute | None,
        beta: float,
        sign: float,
    ) -> tuple[Tensor, RouteCalibrationSnapshot]:
        nonlocal output_shape
        if route is None:
            return torch.zeros((), dtype=x.dtype, device=x.device), _inactive_snapshot(0.0, rho, beta)
        alpha = route.alpha
        if full_psf is None:
            raise ValueError("an active route requires a full PSF")
        if full_psf.ndim == 4 and full_psf.shape[:2] == (1, 1):
            full_psf_2d = full_psf[0, 0]
        elif full_psf.ndim == 2:
            full_psf_2d = full_psf
        else:
            raise ValueError("a route PSF must be [H,W] or [1,1,H,W]")
        h_support = crop_centered(full_psf_2d, support)
        tau = h_support.sum()
        gain = compute_calibrated_gain(
            alpha=alpha,
            rho=rho,
            beta=beta,
            tau=tau,
            min_throughput=config.min_throughput,
            max_electronic_gain=config.max_electronic_gain,
            min_split_fraction=config.min_split_fraction,
        )
        positive_detector = detector_response(
            x_pos, h_support, rho=rho, beta=beta, stride=stride, padding=padding
        )
        negative_detector = detector_response(
            x_neg, h_support, rho=rho, beta=beta, stride=stride, padding=padding
        )
        output = sign * gain * (positive_detector - negative_detector)
        output_shape = tuple(output.shape[-2:])
        return output, _snapshot(
            alpha=alpha,
            rho=rho,
            beta=beta,
            tau=tau,
            gain=gain,
            min_throughput=config.min_throughput,
            max_electronic_gain=config.max_electronic_gain,
            min_split_fraction=config.min_split_fraction,
        )

    positive_output, positive_snapshot = route_contribution(
        positive_psf, canonical_positive_route, resolved_beta_pos, 1.0
    )
    if positive_snapshot.active:
        contributions.append(positive_output)
    snapshots.append(positive_snapshot)
    negative_output, negative_snapshot = route_contribution(
        negative_psf, canonical_negative_route, resolved_beta_neg, -1.0
    )
    if negative_snapshot.active:
        contributions.append(negative_output)
    snapshots.append(negative_snapshot)

    if contributions:
        output = contributions[0]
        for contribution in contributions[1:]:
            output = output + contribution
    else:
        if isinstance(support, int):
            support_pair = (support, support)
        else:
            support_pair = support
        output_spatial = _conv_output_shape(x.shape[-2:], support_pair, stride, padding, 1)
        output = torch.zeros(
            (x.shape[0], 1, *output_spatial), dtype=x.dtype, device=x.device
        )
    if bias is not None:
        if bias.ndim != 1 or bias.shape[0] != output.shape[1]:
            raise ValueError("bias has incompatible shape")
        output = output + bias.view(1, -1, 1, 1)
    return PhysicalForwardResult(
        output=output,
        positive=snapshots[0],
        negative=snapshots[1],
        support_error=None if support_error is None else float(support_error),
    )


@dataclass(frozen=True)
class PSFFitMetrics:
    target_lobe_normalized: Tensor
    predicted_shape: Tensor
    normalized_mse: float
    cosine_similarity: float
    throughput: float
    support_leakage: float
    response_error: float
    low_throughput_penalty: float
    excessive_gain_penalty: float
    predicted_gain: float | None

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("target_lobe_normalized")
        result.pop("predicted_shape")
        return result


def fit_psf_shape(
    target_lobe: Tensor,
    predicted_full_psf: Tensor,
    *,
    support: int | tuple[int, int],
    min_throughput: float = 0.0,
    alpha: float | None = None,
    rho: float = 1.0,
    beta: float = 1.0,
    max_electronic_gain: float = math.inf,
) -> PSFFitMetrics:
    """Report shape metrics without feeding normalized shape into physical forward."""

    if target_lobe.ndim != 2 or not torch.is_floating_point(target_lobe):
        raise ValueError("target_lobe must be a floating-point [H,W] tensor")
    if torch.any(target_lobe < 0) or not torch.isfinite(target_lobe).all():
        raise ValueError("target_lobe must be finite and non-negative")
    target_sum = target_lobe.sum()
    if float(target_sum.item()) <= 0:
        raise ValueError("target_lobe must have positive energy")
    target_normalized = target_lobe / target_sum
    full = predicted_full_psf[0, 0] if predicted_full_psf.ndim == 4 else predicted_full_psf
    cropped = crop_centered(full, support)
    tau = float(cropped.sum().detach().item())
    if tau <= 0:
        predicted_shape = torch.zeros_like(cropped)
    else:
        predicted_shape = cropped / cropped.sum()
    if tuple(predicted_shape.shape) != tuple(target_normalized.shape):
        raise ValueError(
            f"target and predicted support shapes differ: {tuple(target_normalized.shape)} vs "
            f"{tuple(predicted_shape.shape)}"
        )
    difference = predicted_shape - target_normalized
    normalized_mse = float(difference.square().mean().item())
    target_norm = float(torch.linalg.vector_norm(target_normalized).item())
    prediction_norm = float(torch.linalg.vector_norm(predicted_shape).item())
    dot = float((target_normalized * predicted_shape).sum().item())
    cosine = dot / (target_norm * prediction_norm) if target_norm > 0 and prediction_norm > 0 else 0.0
    response_error = _relative_l2(predicted_shape, target_normalized)
    low_penalty = max(0.0, float(min_throughput) - tau) ** 2
    predicted_gain: float | None = None
    excessive_penalty = 0.0
    if alpha is not None:
        _require_positive(rho, "rho")
        if not _is_finite_number(beta) or float(beta) <= 0:
            raise ValueError("beta must be positive when alpha is supplied")
        predicted_gain = float(alpha) / (float(rho) * float(beta) * tau) if tau > 0 else math.inf
        if math.isfinite(float(max_electronic_gain)):
            excessive_penalty = max(0.0, abs(predicted_gain) - float(max_electronic_gain)) ** 2
    return PSFFitMetrics(
        target_lobe_normalized=target_normalized,
        predicted_shape=predicted_shape,
        normalized_mse=normalized_mse,
        cosine_similarity=cosine,
        throughput=tau,
        support_leakage=float(1.0 - tau),
        response_error=response_error,
        low_throughput_penalty=low_penalty,
        excessive_gain_penalty=excessive_penalty,
        predicted_gain=predicted_gain,
    )


__all__ = [
    "ActiveRoute",
    "CenteredPCAResult",
    "DetectorSamplingMetadata",
    "Exposure",
    "ExposureLedger",
    "OptoelectronicConfig",
    "PCAConv2d",
    "PCAReport",
    "PSFFitMetrics",
    "PhaseOnlyPSF",
    "PhasePSFConfig",
    "PhysicalForwardResult",
    "RouteCalibrationSnapshot",
    "SignedKernelLobes",
    "SupportResponseError",
    "SupportResponseReport",
    "SupportSweepResult",
    "build_active_routes",
    "build_exposure_ledger",
    "centered_fft",
    "centered_pca_decompose",
    "compute_calibrated_gain",
    "convolve_psf",
    "crop_centered",
    "decompose_conv2d",
    "detector_response",
    "detector_sampling_metadata",
    "fit_psf_shape",
    "ideal_signed_conv2d",
    "phase_to_full_psf",
    "physical_psf_from_digital_lobe",
    "physical_psf_to_correlation_kernel",
    "physical_dual_rail_forward",
    "signed_input_rails",
    "signed_kernel_lobes",
    "support_sweep",
]
