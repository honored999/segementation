"""Numerical core for the 450/532 nm tilted-plane-wave isoplanatism validation.

This module is a read-only consumer of the existing fitting / propagation code:

* the transmission field is rebuilt exactly like the fitting entry point,
  ``exp(1j * compute_phase_fitted(width_nm, artifact.wavelength_nm))`` with uniform unit
  amplitude (``fit_psf_gray450.py:62-78``);
* propagation always goes through
  :func:`optics.propagation.create_propagation_layer`; no ASM physics is
  copied or modified here.

Conventions
-----------
* Fields are 2-D ``(n, n)`` ``complex64`` arrays, ``array[row, col] = [y, x]``.
* Pixel-centre coordinates are ``(arange(n) - (n - 1) / 2) * dx`` so that for
  even ``n`` the optical axis lies between the two central samples.  This is
  the convention of the fitting entry point, which never used explicit
  coordinates.
* Physical ``+y`` is "down" on screen (``origin="upper"``), physical ``+x`` is
  "right"; both are documented in the README and the run manifest.
* A single-axis tilted plane wave along ``+x`` multiplies the field by
  ``exp(1j * 2*pi/lambda * x * sin(theta_x))`` (same for ``+y``).
* The propagation engine's extended plane is built by embedding the same
  ``n x n`` field into a larger zero field and calling the real ASM with
  ``padding=0``; the outer region is zero amplitude, never extra metasurface.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from optics.fitted_phase_parameters_p400 import compute_phase_fitted
from optics.propagation import create_propagation_layer

WAVELENGTH_NM = 450.0  # backward-compatible default only
SUPPORTED_WAVELENGTHS_NM = (450.0, 532.0)
DX_NM = 400.0
WIDTH_MIN_NM = 80.0
WIDTH_MAX_NM = 320.0

#: Task book section 3.3: small-angle single-axis scan, per axis.
DEFAULT_ANGLES_DEG: Tuple[float, ...] = (-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5)

#: Locked inputs from the task book (section 2).  Any change is a hard stop.
LOCKED_SHA256: Dict[str, str] = {
    "fit_psf_gray450.py": "D3034D83EF8BF216BE431C162C2AAF654DC3E158734C83FA91E04DC2C8F4B53F",
    "fit_psf.py": "E251CDA52251F3EB415E116AC2FA6ED5DDC476B3F7A9DF24E088EE55AD29C1E8",
    "optics/propagation.py": "B9D7CF9C302814EB232D13D74C26220C5D371CCD628B4DB353225E87193FCE1E",
    "optics/fitted_phase_parameters_p400.py": "A71E0B7DCCB4F0E9FCCD094A059EDC0B0ED7195EA88549EFD9C324A1F3D3BE20",
}


class ValidationError(RuntimeError):
    """Raised when an input violates the documented validation contract."""


# ---------------------------------------------------------------------------
# hashing / provenance
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def verify_locked_sources(base_dir: Path) -> Dict[str, Dict[str, str]]:
    """Verify the four locked files; raise if any changed."""
    base_dir = Path(base_dir)
    report: Dict[str, Dict[str, str]] = {}
    problems: List[str] = []
    for relative, expected in LOCKED_SHA256.items():
        path = base_dir / relative
        if not path.is_file():
            problems.append(f"{relative}: missing ({path})")
            continue
        actual = sha256_file(path)
        report[relative] = {"expected": expected, "actual": actual, "ok": actual == expected}
        if actual != expected:
            problems.append(f"{relative}: sha256 {actual} != locked {expected}")
    if problems:
        raise ValidationError("locked source files changed:\n  " + "\n  ".join(problems))
    return report


# ---------------------------------------------------------------------------
# artifact / config loading
# ---------------------------------------------------------------------------


def _cpu_array(value):
    """Postprocessing is CPU-only; propagation may run on the selected device."""
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _as_real_array(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == object:
        raise ValidationError(f"{name}: object arrays are not accepted")
    if arr.dtype.kind not in "biuf":
        raise ValidationError(f"{name}: expected a real numeric array, got dtype {arr.dtype}")
    out = arr.astype(np.float64, copy=False)
    if not np.isfinite(out).all():
        raise ValidationError(f"{name}: contains NaN or Inf")
    return out


def _as_scalar_float(value: Any, name: str) -> float:
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValidationError(f"{name}: expected a scalar, got shape {arr.shape}")
    out = float(arr.reshape(-1)[0])
    if not math.isfinite(out):
        raise ValidationError(f"{name}: not finite")
    return out


@dataclass
class Artifact:
    """A loaded optimisation artifact (``kernel_XXX_pos/neg.npz``)."""

    path: Path
    sha256: str
    size_bytes: int
    width_nm: Optional[np.ndarray]
    psf_raw: Optional[np.ndarray]
    gain_gray: Optional[float]
    active_gray: Optional[bool]
    wavelength_nm: float
    skipped: bool
    keys: Tuple[str, ...]

    @property
    def branch(self) -> str:
        return self.path.stem


def load_artifact(path: Path) -> Artifact:
    """Load and validate an optimization artifact (``.npz`` only, no pickle)."""
    path = Path(path)
    if not path.is_file():
        raise ValidationError(f"artifact not found: {path}")
    if path.suffix.lower() != ".npz":
        raise ValidationError(f"artifact must be an .npz file, got {path.name}")
    try:
        loaded = np.load(path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001 - re-raise with context
        raise ValidationError(f"{path.name}: cannot read as a plain .npz ({exc})") from exc
    if isinstance(loaded, np.ndarray):
        raise ValidationError(f"{path.name}: expected a named .npz archive, got a bare array")
    try:
        arrays = {key: np.array(loaded[key]) for key in loaded.files}
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(f"{path.name}: contains unsupported (e.g. pickled) data ({exc})") from exc
    finally:
        loaded.close()

    if "width_nm" not in arrays:
        raise ValidationError(f"{path.name}: missing required array 'width_nm' (found {sorted(arrays)})")
    raw_width = arrays["width_nm"]
    if raw_width.dtype == object:
        raise ValidationError(f"{path.name}: width_nm is an object array")

    skipped = raw_width.size == 0
    if skipped:
        width = raw_width
    else:
        width = _as_real_array(raw_width, "width_nm")
        if width.ndim != 2:
            raise ValidationError(f"{path.name}: width_nm must be 2-D, got shape {width.shape}")
        if float(width.min()) >= -1.0 and float(width.max()) <= 1.0:
            raise ValidationError(
                f"{path.name}: width_nm is confined to [-1, 1]; this looks like raw sigmoid "
                "parameters, not physical widths in nm"
            )
        if float(width.min()) < WIDTH_MIN_NM - 1e-6 or float(width.max()) > WIDTH_MAX_NM + 1e-6:
            raise ValidationError(
                f"{path.name}: width_nm range [{width.min():.3f}, {width.max():.3f}] nm is outside "
                f"the fitted library range [{WIDTH_MIN_NM}, {WIDTH_MAX_NM}] nm"
            )

    psf: Optional[np.ndarray] = None
    if "psf_raw" in arrays:
        psf = _as_real_array(arrays["psf_raw"], "psf_raw")
        if psf.ndim == 3 and psf.shape[0] == 1:
            psf = psf[0]
        if psf.ndim != 2:
            raise ValidationError(f"{path.name}: psf_raw must be [1, n, n] or [n, n], got {psf.shape}")
        if float(psf.min()) < -1e-6 * max(1.0, float(psf.max())):
            raise ValidationError(f"{path.name}: psf_raw has significant negative values")

    if "wavelength_nm" not in arrays:
        raise ValidationError(f"{path.name}: missing wavelength_nm")
    wavelength = _as_scalar_float(arrays["wavelength_nm"], "wavelength_nm")
    if wavelength not in SUPPORTED_WAVELENGTHS_NM:
        raise ValidationError(f"{path.name}: unsupported wavelength_nm={wavelength}; supported={SUPPORTED_WAVELENGTHS_NM}")

    gain = None
    if "gain_gray" in arrays:
        gain = _as_scalar_float(arrays["gain_gray"], "gain_gray")
    active = None
    if "active_gray" in arrays:
        active = bool(np.asarray(arrays["active_gray"]).reshape(-1)[0])

    if skipped:
        if psf is not None and float(np.abs(psf).max()) > 0.0:
            raise ValidationError(f"{path.name}: empty width_nm but psf_raw is non-zero; refusing a fake PSF")
        if active:
            raise ValidationError(f"{path.name}: empty width_nm but active_gray=True")

    return Artifact(
        path=path,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        width_nm=width,
        psf_raw=psf,
        gain_gray=gain,
        active_gray=active,
        wavelength_nm=wavelength,
        skipped=skipped,
        keys=tuple(sorted(arrays)),
    )


@dataclass
class SourceConfig:
    """Parsed source ``config.json`` of the fitting run."""

    path: Path
    sha256: str
    n: int
    dx_nm: float
    distance_um: float
    wavelength_nm: float
    padding: int
    padding_source: str
    patch_meta: Optional[int]
    grayscale: Optional[bool]
    raw: Dict[str, Any]


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{name}: expected an integer, got {value!r}")
    return value


def load_source_config(path: Path) -> SourceConfig:
    """Load the fitting run config and validate the physical parameters."""
    path = Path(path)
    if not path.is_file():
        raise ValidationError(f"config not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{path.name}: invalid JSON ({exc})") from exc
    if not isinstance(raw, dict):
        raise ValidationError(f"{path.name}: top-level JSON must be an object")

    data: Dict[str, Any] = {}
    nested = raw.get("args")
    if isinstance(nested, dict):
        conflicts = [k for k in nested if k in raw and nested[k] != raw[k]]
        if conflicts:
            raise ValidationError(f"conflicting nested config keys: {conflicts}")
        data.update(nested)
    data.update({key: value for key, value in raw.items() if key != "args"})

    missing = [key for key in ("n", "dx_nm", "distance_um", "wavelength_nm", "camera_pixel_um", "patch_camera", "patch_meta") if key not in data]
    if missing:
        raise ValidationError(f"{path.name}: missing required config keys {missing}")

    n = _as_int(data["n"], "n")
    if n < 16 or n > 8192:
        raise ValidationError(f"{path.name}: n={n} outside supported range [16, 8192]")
    dx_nm = _as_scalar_float(data["dx_nm"], "dx_nm")
    distance_um = _as_scalar_float(data["distance_um"], "distance_um")
    wavelength_nm = _as_scalar_float(data["wavelength_nm"], "wavelength_nm")
    if abs(dx_nm - DX_NM) > 1e-9:
        raise ValidationError(f"{path.name}: dx_nm={dx_nm} differs from the fitted library pitch {DX_NM}")
    if wavelength_nm not in SUPPORTED_WAVELENGTHS_NM:
        raise ValidationError(f"{path.name}: unsupported wavelength_nm={wavelength_nm}; supported={SUPPORTED_WAVELENGTHS_NM}")
    if distance_um <= 0:
        raise ValidationError(f"{path.name}: distance_um must be positive")

    grayscale = data.get("grayscale")
    if grayscale is not None and grayscale is not True:
        raise ValidationError(f"{path.name}: grayscale must be true for this grayscale pipeline")

    patch = None
    if "patch_meta" in data:
        patch = _as_int(data["patch_meta"], "patch_meta")
        if patch <= 0:
            raise ValidationError(f"{path.name}: patch_meta must be positive")

    camera = _as_scalar_float(data['camera_pixel_um'], 'camera_pixel_um')
    patch_camera = _as_scalar_float(data['patch_camera'], 'patch_camera')
    if camera <= 0 or patch_camera <= 0 or patch != int(patch_camera*camera*1000/dx_nm) or patch > n:
        raise ValidationError('camera/patch fields conflict with source target construction')

    if "padding" in data:
        padding = _as_int(data["padding"], "padding")
        if padding < 0:
            raise ValidationError(f"{path.name}: padding must be >= 0")
        if "fft_size" in data:
            fft_size = _as_int(data["fft_size"], "fft_size")
            if fft_size != n + 2 * padding:
                raise ValidationError("fft_size conflicts with n + 2*padding")
        elif padding != n:
            raise ValidationError("legacy config without fft_size requires padding=n")
        padding_source = "config fft_size=n+2*padding" if "fft_size" in data else "legacy config padding=n"
    else:
        if "fft_size" in data:
            raise ValidationError("fft_size requires explicit padding")
        padding = n
        padding_source = (
            "derived: source config has no padding field; fit_psf_gray450.py:65-69 passes "
            "padding=args.n, i.e. one full image width per side"
        )

    return SourceConfig(
        path=path,
        sha256=sha256_file(path),
        n=n,
        dx_nm=dx_nm,
        distance_um=distance_um,
        wavelength_nm=wavelength_nm,
        padding=padding,
        padding_source=padding_source,
        patch_meta=patch,
        grayscale=grayscale,
        raw=data,
    )


def cross_check_config_artifact(config: SourceConfig, artifact: Artifact) -> None:
    """Fail loudly on config/artifact conflicts; never silently pick a default."""
    problems: List[str] = []
    if abs(artifact.wavelength_nm - config.wavelength_nm) > 1e-6:
        problems.append(f"wavelength mismatch: artifact {artifact.wavelength_nm} vs config {config.wavelength_nm}")
    if artifact.skipped:
        if artifact.psf_raw is not None and float(np.abs(artifact.psf_raw).max()) > 0.0:
            problems.append("skipped artifact carries a non-zero psf_raw")
        if artifact.active_gray:
            problems.append("skipped artifact has active_gray=True")
    else:
        if artifact.width_nm is None or artifact.width_nm.shape != (config.n, config.n):
            problems.append(f"width shape {None if artifact.width_nm is None else artifact.width_nm.shape} != ({config.n}, {config.n})")
    if artifact.psf_raw is not None and not artifact.skipped and artifact.psf_raw.shape != (config.n, config.n):
        problems.append(f"psf_raw shape {artifact.psf_raw.shape} != ({config.n}, {config.n})")
    if problems:
        raise ValidationError("artifact/config conflict: " + "; ".join(problems))


# ---------------------------------------------------------------------------
# geometry, fields
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Geometry:
    n: int
    dx_m: float
    wavelength_m: float
    distance_m: float

    @classmethod
    def from_config(cls, config: SourceConfig) -> "Geometry":
        return cls(
            n=config.n,
            dx_m=config.dx_nm * 1e-9,
            wavelength_m=config.wavelength_nm * 1e-9,
            distance_m=config.distance_um * 1e-6,
        )

    @property
    def nyquist_per_m(self) -> float:
        return 0.5 / self.dx_m

    @property
    def extent_m(self) -> float:
        return self.n * self.dx_m

    @property
    def lateral_extent_m(self) -> float:
        return (self.n - 1) * self.dx_m

    @property
    def max_sin_theta(self) -> float:
        """Largest transverse sine the sampled field can represent."""
        return self.wavelength_m * self.nyquist_per_m


def axis_coordinates(n: int, dx_m: float) -> torch.Tensor:
    idx = torch.arange(n, dtype=torch.float64)
    return (idx - (n - 1) / 2.0) * float(dx_m)


def carrier_phase(
    n: int, dx_m: float, wavelength_m: float, theta_x_deg: float, theta_y_deg: float
) -> torch.Tensor:
    """Phase of a single-axis (or combined) tilted plane wave, float64 [n, n]."""
    coords = axis_coordinates(n, dx_m)
    k0 = 2.0 * math.pi / wavelength_m
    kx = k0 * math.sin(math.radians(float(theta_x_deg)))
    ky = k0 * math.sin(math.radians(float(theta_y_deg)))
    return kx * coords[None, :] + ky * coords[:, None]


def build_transmission(width_nm: np.ndarray, wavelength_nm: float = WAVELENGTH_NM) -> torch.Tensor:
    """Rebuild the fitting-entry transmission field: uniform amplitude * exp(i*phase)."""
    if wavelength_nm not in SUPPORTED_WAVELENGTHS_NM:
        raise ValidationError(f"Unsupported grayscale wavelength: {wavelength_nm}")
    width = torch.as_tensor(np.asarray(width_nm), dtype=torch.float64)
    phase = compute_phase_fitted(width, float(wavelength_nm))
    phase = torch.as_tensor(phase, dtype=torch.float64)
    if not torch.isfinite(phase).all():
        raise ValidationError("compute_phase_fitted returned non-finite values")
    return torch.exp(1j * phase).to(torch.complex64)


def apply_carrier(
    field: torch.Tensor, geometry: Geometry, theta_x_deg: float, theta_y_deg: float
) -> torch.Tensor:
    """Multiply a field by the tilted plane-wave carrier (no renormalisation)."""
    if theta_x_deg == 0.0 and theta_y_deg == 0.0:
        return field.clone()
    phase = carrier_phase(geometry.n, geometry.dx_m, geometry.wavelength_m, theta_x_deg, theta_y_deg)
    carrier = torch.exp(1j * phase).to(torch.complex64)
    return field * carrier.to(field.device)


def lateral_shift_theory(geometry: Geometry, theta_x_deg: float, theta_y_deg: float) -> Tuple[float, float]:
    """Geometric reference displacement ``z * tan(theta)`` in pixels: (dy, dx).

    Positive angles move the focus toward +x / +y (down/right on screen).
    """
    dx_px = geometry.distance_m * math.tan(math.radians(float(theta_x_deg))) / geometry.dx_m
    dy_px = geometry.distance_m * math.tan(math.radians(float(theta_y_deg))) / geometry.dx_m
    return dy_px, dx_px


def intensity(field: torch.Tensor) -> torch.Tensor:
    return field.abs().square()


# ---------------------------------------------------------------------------
# real-ASM propagation engine
# ---------------------------------------------------------------------------


class PropagationEngine:
    """Thin cache around the existing ``create_propagation_layer`` implementation."""

    def __init__(self, geometry: Geometry, device: str = "cpu") -> None:
        self.geometry = geometry
        self.device = torch.device(device)
        self._layers: Dict[Tuple[int, int], Any] = {}

    def _layer(self, padding: int, size: int):
        key = (int(padding), int(size))
        layer = self._layers.get(key)
        if layer is None:
            layer = create_propagation_layer(
                distance=self.geometry.distance_m,
                wavelength=self.geometry.wavelength_m,
                pixel_size=self.geometry.dx_m,
                device=str(self.device),
                padding=int(padding),
            )
            self._layers[key] = layer
        return layer

    def direct(self, field: torch.Tensor, padding: int) -> torch.Tensor:
        """Original entry-point style call: ASM pads the n x n field by ``padding``."""
        layer = self._layer(padding, self.geometry.n)
        with torch.no_grad():
            out = layer(field.to(self.device).to(torch.complex64)[None])[0]
        return out

    def extended(self, field: torch.Tensor, padding: int) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """Embed the n x n field in a larger zero plane and propagate with padding=0.

        Returns ``(full, crop, layer)`` where ``full`` is the ``(n + 2p)^2`` plane
        and ``crop`` is its central ``n x n`` window.
        """
        n = self.geometry.n
        size = n + 2 * int(padding)
        big = torch.zeros((size, size), dtype=torch.complex64, device=self.device)
        big[padding : padding + n, padding : padding + n] = field.to(self.device).to(torch.complex64)
        layer = self._layer(0, size)
        with torch.no_grad():
            full = layer(big[None])[0]
        crop = full[padding : padding + n, padding : padding + n]
        return full, crop, layer

    def transfer_function(self, size: int) -> torch.Tensor:
        layer = self._layer(0, size)
        if layer.transfer_function is None or tuple(layer.cached_shape or ()) != (size, size):
            layer.precompute_transfer_function(size, size)
        return layer.transfer_function


# ---------------------------------------------------------------------------
# shift estimation and alignment
# ---------------------------------------------------------------------------


@dataclass
class ShiftEstimate:
    dy_px: Optional[float]
    dx_px: Optional[float]
    peak: Optional[float]
    psr: Optional[float]
    valid: bool
    reason: Optional[str]
    at_boundary: bool


def _parabolic_delta(previous: float, center: float, following: float) -> float:
    denominator = previous - 2.0 * center + following
    if denominator == 0.0 or not math.isfinite(denominator):
        return 0.0
    delta = 0.5 * (previous - following) / denominator
    return float(delta) if math.isfinite(delta) else 0.0


def estimate_shift(
    reference: torch.Tensor,
    moved: torch.Tensor,
    *,
    max_shift: Optional[float] = None,
) -> ShiftEstimate:
    """Linear (non-circular) cross-correlation shift with parabolic sub-pixel refinement.

    Both images are zero-extended to ``2n x 2n`` before the FFT, so a displacement
    never wraps around the border.  The measured value is the translation ``d``
    of the off-axis PSF relative to the reference. The image translation to
    align it is -d; sample_shifted instead samples coordinates at +d.
    """
    ref = torch.as_tensor(_cpu_array(reference), dtype=torch.float64)
    mov = torch.as_tensor(_cpu_array(moved), dtype=torch.float64)
    if ref.ndim != 2 or mov.ndim != 2:
        raise ValidationError("estimate_shift expects 2-D images")
    if ref.shape != mov.shape:
        raise ValidationError(f"estimate_shift shape mismatch {tuple(ref.shape)} vs {tuple(mov.shape)}")
    n = int(ref.shape[0])
    if ref.shape[1] != n:
        raise ValidationError("estimate_shift expects square images")
    if n < 4:
        raise ValidationError("estimate_shift needs at least 4x4 pixels")
    if not torch.isfinite(ref).all() or not torch.isfinite(mov).all():
        raise ValidationError("estimate_shift received non-finite pixels")

    ref = ref - ref.mean()
    mov = mov - mov.mean()
    norm_ref = float(ref.norm())
    norm_mov = float(mov.norm())
    if norm_ref <= 0.0 or norm_mov <= 0.0:
        return ShiftEstimate(None, None, None, None, False, "zero_variance", False)

    size = 2 * n
    spectrum_ref = torch.fft.rfft2(ref, s=(size, size))
    spectrum_mov = torch.fft.rfft2(mov, s=(size, size))
    corr = torch.fft.irfft2(spectrum_mov * spectrum_ref.conj(), s=(size, size)) / (norm_ref * norm_mov)

    signed = torch.arange(size, dtype=torch.float64)
    signed = torch.where(signed > size // 2, signed - size, signed)
    search = torch.ones((size, size), dtype=torch.bool)
    if max_shift is not None:
        limit = float(max_shift)
        search = (signed.abs()[:, None] <= limit) & (signed.abs()[None, :] <= limit)
        if not bool(search.any()):
            return ShiftEstimate(None, None, None, None, False, "empty_search_window", False)

    masked = torch.where(search, corr, torch.full_like(corr, float("-inf")))
    flat_index = int(torch.argmax(masked))
    peak_y, peak_x = divmod(flat_index, size)
    peak = float(corr[peak_y, peak_x])

    dy = float(signed[peak_y]) + _parabolic_delta(
        float(corr[(peak_y - 1) % size, peak_x]), peak, float(corr[(peak_y + 1) % size, peak_x])
    )
    dx = float(signed[peak_x]) + _parabolic_delta(
        float(corr[peak_y, (peak_x - 1) % size]), peak, float(corr[peak_y, (peak_x + 1) % size])
    )
    if not math.isfinite(dy) or not math.isfinite(dx):
        return ShiftEstimate(None, None, peak, None, False, "non_finite_subpixel", False)

    radius = 3
    y0, y1 = max(0, peak_y - radius), min(size, peak_y + radius + 1)
    x0, x1 = max(0, peak_x - radius), min(size, peak_x + radius + 1)
    residual = corr.clone()
    # Circular FFT indices represent signed linear lags; exclude neighbors
    # across index zero as well, without wrapping any image pixels.
    for oy in range(-radius, radius+1):
        for ox in range(-radius, radius+1):
            residual[(peak_y+oy) % size, (peak_x+ox) % size] = float("-inf")
    second = float(residual.max())
    psr = peak / second if second > 0.0 else None

    at_boundary = False
    if max_shift is not None:
        at_boundary = max(abs(dy), abs(dx)) >= float(max_shift) - 1e-9

    valid = math.isfinite(peak) and peak > 0.0
    reason = None if valid else "no_positive_correlation_peak"
    return ShiftEstimate(dy, dx, peak, psr, valid, reason, at_boundary)


def sample_shifted(image: torch.Tensor, dy: float, dx: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bilinearly resample ``image`` at ``(y + dy, x + dx)`` with zero extension.

    Returns ``(resampled, valid_mask)``; the mask is False wherever the 2x2
    interpolation footprint leaves the original image.
    """
    image = torch.as_tensor(_cpu_array(image), dtype=torch.float64)
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValidationError("sample_shifted expects a square 2-D image")
    n = int(image.shape[0])
    ys = torch.arange(n, dtype=torch.float64) + float(dy)
    xs = torch.arange(n, dtype=torch.float64) + float(dx)
    grid_y = 2.0 * ys / (n - 1) - 1.0
    grid_x = 2.0 * xs / (n - 1) - 1.0
    grid = torch.stack(
        (grid_x[None, :].expand(n, n), grid_y[:, None].expand(n, n)), dim=-1
    )[None]
    batch = image[None, None]
    resampled = F.grid_sample(batch, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0]
    ones = torch.ones_like(batch)
    coverage = F.grid_sample(ones, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0]
    valid = coverage > 1.0 - 1e-9
    return resampled, valid


# ---------------------------------------------------------------------------
# shape metrics, energy, ROI and window diagnostics
# ---------------------------------------------------------------------------


@dataclass
class ShapeComparison:
    valid: bool
    reason: Optional[str]
    support_px: int
    reference_support_px: int
    overlap_fraction: float
    shape_error: Optional[float]
    shape_error_squared: Optional[float]
    pearson: Optional[float]
    pearson_reason: Optional[str]


def compare_shapes(
    reference: torch.Tensor,
    aligned: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    rel_threshold: float = 0.0,
    min_support_frac: float = 0.25,
    min_overlap: float = 0.5,
    min_support_px: int = 9,
) -> ShapeComparison:
    """L2-normalised shape error and Pearson correlation on a common support.

    ``shape_error = ||P/||P|| - R/||R||`` with both norms taken over the same
    support; ``shape_error_squared`` matches the single-channel squared
    normalized distance before the training loss floor; this validator has no floor.
    """
    ref = torch.as_tensor(_cpu_array(reference), dtype=torch.float64)
    img = torch.as_tensor(_cpu_array(aligned), dtype=torch.float64)
    mask = torch.as_tensor(_cpu_array(valid_mask)).to(torch.bool)
    if ref.shape != img.shape or ref.shape != mask.shape or ref.ndim != 2:
        raise ValidationError("compare_shapes expects three equally shaped 2-D arrays")
    finite_ref = torch.isfinite(ref)
    finite_img = torch.isfinite(img)
    if not bool(finite_ref.any()):
        return ShapeComparison(False, "non_finite_reference", 0, 0, 0.0, None, None, None, "non_finite_reference")

    ref_max = float(ref[finite_ref].max())
    if ref_max <= 0.0:
        return ShapeComparison(False, "zero_reference_energy", 0, 0, 0.0, None, None, None, "zero_reference_energy")

    reference_support = finite_ref & (ref > rel_threshold * ref_max)
    support = reference_support & mask & finite_img
    reference_support_px = int(reference_support.sum())
    support_px = int(support.sum())
    if reference_support_px < min_support_px:
        return ShapeComparison(False, "reference_support_too_small", support_px, reference_support_px, 0.0, None, None, None, "reference_support_too_small")

    overlap = support_px / float(reference_support_px)
    required = max(min_support_px, int(math.ceil(min_support_frac * reference_support_px)))
    if overlap < min_overlap or support_px < required:
        return ShapeComparison(
            False,
            f"insufficient_valid_overlap (overlap={overlap:.4f} < {min_overlap}, support={support_px} < {required})",
            support_px,
            reference_support_px,
            overlap,
            None,
            None,
            None,
            "insufficient_valid_overlap",
        )

    ref_values = ref[support]
    img_values = img[support]
    ref_norm = float(ref_values.norm())
    img_norm = float(img_values.norm())
    shape_error = None
    shape_error_squared = None
    shape_reason = None
    if ref_norm <= 0.0 or img_norm <= 0.0:
        shape_reason = "zero_energy_on_support"
    else:
        difference = img_values / img_norm - ref_values / ref_norm
        shape_error = float(difference.norm())
        shape_error_squared = shape_error * shape_error

    scale = max(1.0, float(ref_values.abs().max()))
    ref_centered = ref_values - ref_values.mean()
    img_centered = img_values - img_values.mean()
    ref_centered_norm = float(ref_centered.norm())
    img_centered_norm = float(img_centered.norm())
    pearson = None
    pearson_reason = None
    if ref_centered_norm <= 1e-12 * scale or img_centered_norm <= 1e-12 * scale:
        pearson_reason = "zero_variance_on_support"
    else:
        pearson = float((ref_centered * img_centered).sum() / (ref_centered_norm * img_centered_norm))

    valid = shape_error is not None and pearson is not None
    reason = None
    if shape_error is None:
        reason = shape_reason
    elif pearson is None:
        reason = pearson_reason
    return ShapeComparison(
        valid=valid,
        reason=reason,
        support_px=support_px,
        reference_support_px=reference_support_px,
        overlap_fraction=overlap,
        shape_error=shape_error,
        shape_error_squared=shape_error_squared,
        pearson=pearson,
        pearson_reason=pearson_reason,
    )


def energy(intensity_image: torch.Tensor, dx_m: float) -> float:
    """Raw intensity integral ``sum(I) * dx^2`` in arbitrary (field^2 * m^2) units."""
    return float(torch.as_tensor(intensity_image, dtype=torch.float64).sum()) * float(dx_m) ** 2


def roi_window(n: int, size: int, offset_y: float = 0.0, offset_x: float = 0.0) -> Dict[str, Any]:
    """Centred square ROI, optionally translated by an integer pixel offset."""
    size = int(size)
    if size < 1 or size > n:
        raise ValidationError(f"ROI size {size} outside [1, {n}]")
    base = n // 2 - size // 2
    y0 = base + int(round(float(offset_y)))
    x0 = base + int(round(float(offset_x)))
    ry, rx = y0, x0
    y0, y1 = max(0, min(n, ry)), max(0, min(n, ry + size))
    x0, x1 = max(0, min(n, rx)), max(0, min(n, rx + size))
    coverage = (y1-y0)*(x1-x0)/size**2
    return dict(y0=y0, y1=y1, x0=x0, x1=x1, size=size,
                requested_y0=ry, requested_x0=rx, requested_y1=ry+size,
                requested_x1=rx+size, requested_offset_y=offset_y,
                requested_offset_x=offset_x, rounding='nearest integer, ties to even',
                coverage_fraction=coverage, clipped=coverage < 1,
                valid=coverage == 1, reason=None if coverage == 1 else 'roi_outside_output')



def roi_sum(intensity_image: torch.Tensor, window: Dict[str, Any]) -> float:
    image = torch.as_tensor(_cpu_array(intensity_image), dtype=torch.float64)
    return float(image[window["y0"] : window["y1"], window["x0"] : window["x1"]].sum())


@dataclass
class BoundaryStats:
    band_px: int
    edge_energy_fraction: float
    peak_y: int
    peak_x: int
    peak_value: float
    peak_distance_to_border: int


def boundary_stats(intensity_image: torch.Tensor, band_px: Optional[int] = None) -> BoundaryStats:
    image = torch.as_tensor(_cpu_array(intensity_image), dtype=torch.float64)
    n = int(image.shape[0])
    band = int(band_px) if band_px is not None else max(1, min(8, n // 8))
    band = max(1, min(band, n // 2))
    total = float(image.sum())
    edge_mask = torch.ones_like(image, dtype=torch.bool)
    edge_mask[band:-band, band:-band] = False
    edge = float(image[edge_mask].sum())
    edge_fraction = edge / total if total > 0.0 else 0.0
    flat = int(torch.argmax(image))
    peak_y, peak_x = divmod(flat, n)
    distance = min(peak_y, peak_x, n - 1 - peak_y, n - 1 - peak_x)
    return BoundaryStats(
        band_px=band,
        edge_energy_fraction=edge_fraction,
        peak_y=peak_y,
        peak_x=peak_x,
        peak_value=float(image[peak_y, peak_x]),
        peak_distance_to_border=int(distance),
    )


@dataclass
class SamplingDiagnostics:
    valid: bool
    reason: Optional[str]
    band_fraction: float
    spectral_edge_fraction: float
    evanescent_fraction: float
    centroid_fx: Optional[float]
    centroid_fy: Optional[float]
    carrier_fx: Optional[float]
    carrier_fy: Optional[float]
    carrier_over_nyquist_x: Optional[float]
    carrier_over_nyquist_y: Optional[float]
    tf_step_p99_over_pi: Optional[float]
    tf_step_max_over_pi: Optional[float]
    tf_near_pi_fraction: Optional[float]


def _linear_p99_cpu(values: torch.Tensor) -> float:
    """Global h=(N-1)*0.99 linear quantile of all unchanged CPU values."""
    return float(np.quantile(values.detach().numpy(), 0.99, method='linear'))


def sampling_diagnostics(
    field_in: torch.Tensor,
    layer: Any,
    geometry: Geometry,
    theta_x_deg: float,
    theta_y_deg: float,
    *,
    band_fraction: float = 0.99,
    edge_frequency_fraction: float = 0.9,
) -> SamplingDiagnostics:
    """Nyquist / transfer-function sampling diagnostics for one incident field.

    The transfer-function phase step is evaluated only inside the spectral band
    that carries ``band_fraction`` of the field energy; outside that band the
    step is meaningless (it is ~pi almost everywhere at high frequencies).
    """
    if layer.transfer_function is None:
        raise ValidationError("sampling_diagnostics needs a precomputed transfer function")
    tf = layer.transfer_function.detach().cpu()
    size = int(tf.shape[0])
    if tuple(tf.shape) != (size, size):
        raise ValidationError("sampling_diagnostics expects a square transfer function")

    field_2d = torch.as_tensor(_cpu_array(field_in), dtype=torch.complex128)
    spectrum = torch.fft.fftshift(torch.fft.fft2(field_2d, s=(size, size)))
    power = spectrum.abs().square()
    total = float(power.sum())
    if total <= 0.0:
        return SamplingDiagnostics(False, "zero_spectral_energy", band_fraction, 0.0, 0.0, None, None, None, None, None, None, None, None, None)

    frequencies = torch.fft.fftshift(torch.fft.fftfreq(size, geometry.dx_m, dtype=torch.float64))
    nyquist = geometry.nyquist_per_m
    edge_mask = (frequencies.abs()[:, None] >= edge_frequency_fraction * nyquist) | (
        frequencies.abs()[None, :] >= edge_frequency_fraction * nyquist
    )
    spectral_edge_fraction = float(power[edge_mask].sum() / total)

    normalized = (geometry.wavelength_m * frequencies[None, :]) ** 2 + (geometry.wavelength_m * frequencies[:, None]) ** 2
    evanescent_fraction = float(power[normalized >= 1.0].sum() / total)

    centroid_fx = float((power.sum(dim=0) * frequencies).sum() / total)
    centroid_fy = float((power.sum(dim=1) * frequencies).sum() / total)

    sorted_power, _ = torch.sort(power.flatten(), descending=True)
    cumulative = torch.cumsum(sorted_power, dim=0)
    target = torch.tensor(band_fraction * total, dtype=cumulative.dtype)
    target_index = int(torch.searchsorted(cumulative, target).item())
    target_index = min(target_index, int(sorted_power.numel()) - 1)
    threshold = float(sorted_power[target_index])
    band_mask = power >= threshold

    step_x = torch.angle(tf[:, 1:] * tf[:, :-1].conj()).abs()
    step_y = torch.angle(tf[1:, :] * tf[:-1, :].conj()).abs()
    mask_x = band_mask[:, :-1] & band_mask[:, 1:]
    mask_y = band_mask[:-1, :] & band_mask[1:, :]
    steps = torch.cat([step_x[mask_x].flatten(), step_y[mask_y].flatten()])
    if steps.numel() == 0:
        tf_p99 = tf_max = tf_near_pi = None
    else:
        tf_p99 = _linear_p99_cpu(steps) / math.pi
        tf_max = float(steps.max()) / math.pi
        tf_near_pi = float((steps > 0.9 * math.pi).to(torch.float64).mean())

    carrier_fx = math.sin(math.radians(float(theta_x_deg))) / geometry.wavelength_m
    carrier_fy = math.sin(math.radians(float(theta_y_deg))) / geometry.wavelength_m
    return SamplingDiagnostics(
        valid=True,
        reason=None,
        band_fraction=band_fraction,
        spectral_edge_fraction=spectral_edge_fraction,
        evanescent_fraction=evanescent_fraction,
        centroid_fx=centroid_fx,
        centroid_fy=centroid_fy,
        carrier_fx=carrier_fx,
        carrier_fy=carrier_fy,
        carrier_over_nyquist_x=abs(carrier_fx) / nyquist,
        carrier_over_nyquist_y=abs(carrier_fy) / nyquist,
        tf_step_p99_over_pi=tf_p99,
        tf_step_max_over_pi=tf_max,
        tf_near_pi_fraction=tf_near_pi,
    )


# ---------------------------------------------------------------------------
# thresholds, padding stability, status classification
# ---------------------------------------------------------------------------


@dataclass
class Thresholds:
    """Engineering self-check tolerances; serialised with rationale in the manifest."""

    min_peak_border_px: int = 8
    max_edge_energy_fraction: float = 0.01
    min_output_extended_ratio: float = 0.60
    min_overlap_fraction: float = 0.50
    min_support_fraction: float = 0.25
    padding_shape_abs: float = 0.01
    padding_energy_rel: float = 0.01
    padding_shift_px: float = 0.10
    sampling_tf_step_over_pi: float = 0.90
    sampling_edge_fraction: float = 0.01

    def as_dict(self) -> Dict[str, float]:
        return {
            "min_peak_border_px": self.min_peak_border_px,
            "max_edge_energy_fraction": self.max_edge_energy_fraction,
            "min_output_extended_ratio": self.min_output_extended_ratio,
            "min_overlap_fraction": self.min_overlap_fraction,
            "min_support_fraction": self.min_support_fraction,
            "padding_shape_abs": self.padding_shape_abs,
            "padding_energy_rel": self.padding_energy_rel,
            "padding_shift_px": self.padding_shift_px,
            "sampling_tf_step_over_pi": self.sampling_tf_step_over_pi,
            "sampling_edge_fraction": self.sampling_edge_fraction,
        }


THRESHOLD_RATIONALE = {
    'min_peak_border_px': 'Prospective conservative 8-pixel proximity flag, not proof of loss.',
    'max_edge_energy_fraction': 'Prospective 1% edge-band energy flag for finite-window sensitivity.',
    'min_output_extended_ratio': 'Prospective gross loss flag if more than 40% leaves original window.',
    'min_overlap_fraction': 'Require at least half the reference support to avoid low-overlap normalization.',
    'min_support_fraction': 'Require at least a quarter of support and 9 absolute pixels; engineering guard.',
    'padding_shape_abs': 'Prospective direct cross-padding full-window normalized L2 PSF distance budget 0.01; not a scalar-error difference or isoplanatism threshold.',
    'padding_energy_rel': 'Prospective 1% raw output-energy padding stability budget.',
    'padding_shift_px': 'Prospective tenth-pixel padding stability budget for displacement.',
    'sampling_tf_step_over_pi': 'Flag wrapped steps near pi (0.9*pi); cannot detect unwrapped advances >pi.',
    'sampling_edge_fraction': 'Prospective 1% energy near Nyquist flag, not an anti-aliasing certificate.',
}
SAMPLING_LIMITATION = ('Wrapped adjacent transfer phase differences cannot detect phase advances '
                       'exceeding pi; absent warnings do not exclude aliasing. Padding does not '
                       'repair input sampling. Existing ASM is not a band-limited ASM.')



def padding_stability(
    baseline: Mapping[str, Any],
    extended: Mapping[str, Any],
    thresholds: Thresholds,
) -> Dict[str, Any]:
    """Compare raw center-crop PSFs at fixed coordinates, without registration.

    The full n*n window includes candidate energy absent in the baseline.
    ``padding_shape_abs`` now bounds this direct normalized L2 distance,
    rather than a difference of separately computed shape-error scalars.
    """
    result = dict(baseline_padding=baseline.get('padding'),
                  candidate_padding=extended.get('padding'),
                  baseline_npz_file=baseline.get('npz_file'),
                  candidate_npz_file=extended.get('npz_file'),
                  shape_definition='||candidate/||candidate||2 - baseline/||baseline||2||2; full fixed n*n crop, no shift',
                  status='indeterminate', reasons=[], checks={})
    reasons = result['reasons']
    a, b = baseline.get('psf'), extended.get('psf')
    shape_error = None
    if a is None or b is None:
        reasons.append('missing_psf')
    else:
        a, b = np.asarray(_cpu_array(a), dtype=np.float64), np.asarray(_cpu_array(b), dtype=np.float64)
        if a.ndim != 2 or a.shape != b.shape or not a.size:
            reasons.append('invalid_psf_support')
        elif not np.isfinite(a).all() or not np.isfinite(b).all() or (a < 0).any() or (b < 0).any():
            reasons.append('nonfinite_or_negative_psf')
        else:
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if not np.isfinite(na) or not np.isfinite(nb) or min(na, nb) <= 0:
                reasons.append('zero_or_invalid_psf_norm')
            else:
                shape_error = float(np.linalg.norm(a/na - b/nb))

    def finite_value(record, key):
        value = record.get(key)
        return float(value) if value is not None and math.isfinite(float(value)) else None

    ea = finite_value(baseline, 'energy_output_au')
    eb = finite_value(extended, 'energy_output_au')
    energy_rel = None if ea is None or eb is None or min(ea,eb) <= 0 else abs(ea-eb)/max(abs(ea),abs(eb))
    shifts = [finite_value(record,key) for record in (baseline,extended)
              for key in ('shift_measured_dy_px','shift_measured_dx_px')]
    shift_delta = None if any(v is None for v in shifts) else max(abs(shifts[0]-shifts[2]),abs(shifts[1]-shifts[3]))
    result['checks'] = dict(direct_psf_shape_error=shape_error,
                            energy_output_rel_diff=energy_rel,shift_abs_diff_px=shift_delta)
    if energy_rel is None:
        reasons.append('invalid_output_energy')
    if shift_delta is None:
        reasons.append('missing_or_invalid_measured_shift')
    if baseline.get('padding') == extended.get('padding'):
        reasons.append('baseline_self_comparison_is_not_convergence')
    if reasons:
        return result
    for key, value, limit in [('direct_psf_shape_error',shape_error,thresholds.padding_shape_abs),
                              ('energy_output_rel_diff',energy_rel,thresholds.padding_energy_rel),
                              ('shift_abs_diff_px',shift_delta,thresholds.padding_shift_px)]:
        if value > limit:
            reasons.append(f'{key}={value} > {limit}')
    result['status'] = 'inconsistent' if reasons else 'consistent'
    return result



def classify_truncation(
    boundary: BoundaryStats,
    energy_output_over_extended: Optional[float],
    padding_status: str,
    thresholds: Thresholds,
) -> Tuple[str, List[str]]:
    """Return ``(truncated|not_detected|indeterminate, evidence)``."""
    evidence: List[str] = []
    truncated = False
    if boundary.edge_energy_fraction > thresholds.max_edge_energy_fraction:
        evidence.append(
            f"edge_energy_fraction={boundary.edge_energy_fraction:.4f} > {thresholds.max_edge_energy_fraction}"
        )
        truncated = True
    if boundary.peak_distance_to_border < thresholds.min_peak_border_px:
        evidence.append(
            f"peak_distance_to_border={boundary.peak_distance_to_border} < {thresholds.min_peak_border_px} px"
        )
        truncated = True
    if energy_output_over_extended is not None and energy_output_over_extended < thresholds.min_output_extended_ratio:
        evidence.append(
            f"output/extended={energy_output_over_extended:.4f} < {thresholds.min_output_extended_ratio}"
        )
        truncated = True
    if truncated:
        return "truncated", evidence
    if padding_status != "consistent":
        evidence.append(f"padding_stability={padding_status}; window wrap-around not excluded")
        return "indeterminate", evidence
    evidence.append(
        f"edge_energy_fraction={boundary.edge_energy_fraction:.4f}, peak_distance={boundary.peak_distance_to_border} px, "
        f"output/extended={energy_output_over_extended if energy_output_over_extended is None else round(energy_output_over_extended, 4)}, "
        "two paddings consistent"
    )
    return "not_detected", evidence


def classify_sampling(diagnostics: SamplingDiagnostics, thresholds: Thresholds) -> Tuple[bool, List[str]]:
    """Return ``(warning, reasons)`` for Nyquist / TF sampling risk."""
    if not diagnostics.valid:
        return True, [f"spectral diagnostics invalid: {diagnostics.reason}"]
    reasons: List[str] = []
    if diagnostics.tf_step_p99_over_pi is not None and diagnostics.tf_step_p99_over_pi > thresholds.sampling_tf_step_over_pi:
        reasons.append(
            f"tf_step_p99/pi={diagnostics.tf_step_p99_over_pi:.3f} > {thresholds.sampling_tf_step_over_pi}"
        )
    for axis in ("x", "y"):
        ratio = getattr(diagnostics, f"carrier_over_nyquist_{axis}")
        if ratio is not None and ratio >= 1.0:
            reasons.append(f"carrier_{axis}/Nyquist={ratio:.3f} >= 1")
    if diagnostics.spectral_edge_fraction > thresholds.sampling_edge_fraction:
        reasons.append(
            f"spectral_edge_fraction={diagnostics.spectral_edge_fraction:.4f} > {thresholds.sampling_edge_fraction}"
        )
    return bool(reasons), reasons


def classify_validity(
    comparison_measured: Optional[ShapeComparison],
    comparison_theory: Optional[ShapeComparison],
    shift: ShiftEstimate,
    thresholds: Thresholds,
) -> Tuple[str, List[str]]:
    """Numeric validity of a single row (truncation/padding handled separately)."""
    reasons: List[str] = []
    invalid = False
    if not shift.valid:
        invalid = True
        reasons.append(f"shift_estimation_failed: {shift.reason}")
    if shift.valid and shift.at_boundary:
        invalid = True
        reasons.append("measured_shift_at_search_boundary")
    for name, comparison in (("measured", comparison_measured), ("theory", comparison_theory)):
        if comparison is None:
            continue
        if not comparison.valid:
            invalid = True
            reasons.append(f"{name}_alignment_invalid: {comparison.reason}")
            continue
    return ("invalid" if invalid else "ok"), reasons


# ---------------------------------------------------------------------------
# detection-plane metrics: centroid, ROI energy budget, camera-pixel binning
# ---------------------------------------------------------------------------
#
# Additive only: the tilted-plane validation and the FFT-window convergence
# scan share these definitions so that "same ROI, same units" is checkable
# instead of implied.  Every function is CPU/float64 on postprocessing values.


@dataclass
class Centroid:
    """Intensity centroid inside an optional window, relative to array centre."""

    valid: bool
    reason: Optional[str]
    total_au: float
    y_px: Optional[float]
    x_px: Optional[float]
    y_um: Optional[float]
    x_um: Optional[float]
    window: Optional[Dict[str, Any]]
    definition: str = (
        "sum(I*r)/sum(I) over the requested window only; r=(index-(N-1)/2)*dx; "
        "+y down. Centred crops of every FFT size share the same physical centre."
    )


def centroid(
    intensity_image: torch.Tensor,
    dx_m: float,
    window: Optional[Mapping[str, Any]] = None,
) -> Centroid:
    """Centroid of a non-negative intensity image, optionally inside ``window``."""
    image = torch.as_tensor(intensity_image)
    if image.ndim != 2 or image.numel() == 0:
        return Centroid(False, "needs_2d_intensity", 0.0, None, None, None, None, window)
    ny, nx = int(image.shape[0]), int(image.shape[1])
    y0, x0 = 0, 0
    sub = image
    if window is not None:
        if not window.get("valid", False):
            return Centroid(False, "roi_outside_output", 0.0, None, None, None, None, window)
        y0, x0 = int(window["y0"]), int(window["x0"])
        sub = image[y0 : int(window["y1"]), x0 : int(window["x1"])]
    if not sub.is_floating_point():
        sub = sub.to(torch.float64)
    # Accumulate in float64 via sum(dtype=...); a full float64 copy of an
    # extended plane would cost hundreds of MB.
    total = float(sub.sum(dtype=torch.float64))
    if not math.isfinite(total):
        return Centroid(False, "nonfinite_intensity", 0.0, None, None, None, None, window)
    if total <= 0.0:
        return Centroid(False, "zero_total_intensity", 0.0, None, None, None, None, window)
    if float(sub.min()) < 0.0:
        return Centroid(False, "negative_intensity", 0.0, None, None, None, None, window)
    ys = torch.arange(y0, y0 + sub.shape[0], dtype=torch.float64, device=sub.device) - (ny - 1) / 2.0
    xs = torch.arange(x0, x0 + sub.shape[1], dtype=torch.float64, device=sub.device) - (nx - 1) / 2.0
    y_px = float((sub.sum(dim=1, dtype=torch.float64) * ys).sum() / total)
    x_px = float((sub.sum(dim=0, dtype=torch.float64) * xs).sum() / total)
    return Centroid(
        True, None, total, y_px, x_px, y_px * dx_m * 1e6, x_px * dx_m * 1e6, window
    )


@dataclass
class RoiEnergy:
    """Energy budget of a square detection ROI inside a (possibly extended) plane."""

    valid: bool
    reason: Optional[str]
    total_au: float
    inside_au: float
    outside_au: float
    inside_fraction: Optional[float]
    outside_fraction: Optional[float]
    window: Optional[Dict[str, Any]]


def roi_energy(
    intensity_image: torch.Tensor,
    window: Mapping[str, Any],
) -> RoiEnergy:
    """Raw (un-normalised) ROI energy and the fraction that falls outside it."""
    image = torch.as_tensor(intensity_image)
    if image.ndim != 2 or image.numel() == 0:
        return RoiEnergy(False, "needs_2d_intensity", 0.0, 0.0, 0.0, None, None, dict(window))
    if not window.get("valid", False):
        return RoiEnergy(False, "roi_outside_output", 0.0, 0.0, 0.0, None, None, dict(window))
    slice_y = slice(int(window["y0"]), int(window["y1"]))
    slice_x = slice(int(window["x0"]), int(window["x1"]))
    # Promote the accumulators, not the array: extended planes are hundreds of MB.
    inside = float(image[slice_y, slice_x].sum(dtype=torch.float64))
    total = float(image.sum(dtype=torch.float64))
    if not (math.isfinite(total) and math.isfinite(inside)):
        return RoiEnergy(False, "nonfinite_intensity", 0.0, 0.0, 0.0, None, None, dict(window))
    if total <= 0.0:
        return RoiEnergy(False, "zero_total_intensity", 0.0, inside, 0.0, None, None, dict(window))
    return RoiEnergy(
        True, None, total, inside, total - inside, inside / total, 1.0 - inside / total, dict(window)
    )


def pixel_bin_matrix(
    n: int,
    dx_m: float,
    pixel_m: float,
    *,
    origin_px: int = 0,
) -> Dict[str, Any]:
    """Area-overlap weights binning ``n`` simulation pixels onto camera pixels.

    Bin ``i`` covers ``[origin_px + i*p, origin_px + (i+1)*p)`` simulation pixels
    with ``p = pixel_m/dx_m``.  Partial bins at both ends are retained with
    their true width and area; nothing is redistributed or renormalised.
    """
    n = int(n)
    if n < 1:
        raise ValidationError("pixel_bin_matrix needs n >= 1")
    if not (math.isfinite(dx_m) and dx_m > 0.0 and math.isfinite(pixel_m) and pixel_m > 0.0):
        raise ValidationError("pixel_bin_matrix needs positive finite dx and pixel pitch")
    p = float(pixel_m) / float(dx_m)
    if p < 1.0:
        raise ValidationError(f"camera pixel {pixel_m} is smaller than simulation pixel {dx_m}")
    lo = float(origin_px)
    bins = int(math.ceil((n - lo) / p))
    if bins < 1:
        raise ValidationError("empty camera-pixel binning")
    edges = lo + np.arange(bins + 1, dtype=np.float64) * p
    index = np.arange(n, dtype=np.float64)[None, :]
    lower = np.maximum(edges[:-1, None], index)
    upper = np.minimum(np.minimum(edges[1:, None], index + 1.0), float(n))
    matrix = np.clip(upper - lower, 0.0, None)
    widths = np.diff(edges)
    covered = matrix.sum(axis=1)
    return dict(
        matrix=matrix,
        bins=bins,
        pixel_px=p,
        edges_px=edges,
        origin_px=int(origin_px),
        bin_width_px=widths,
        bin_area_m2=widths * float(dx_m),
        coverage_px=covered,
        partial_bins=int((covered + 1e-12 < p).sum()),
        definition=(
            "exact area overlap of simulation pixels [j,j+1)*dx with camera bins "
            "[origin + i*p, origin + (i+1)*p) in simulation-pixel units; "
            "edges kept at their true partial width"
        ),
    )


def pixel_integrate(
    intensity_image: torch.Tensor,
    rows: Mapping[str, Any],
    cols: Mapping[str, Any],
    dx_m: float,
) -> np.ndarray:
    """Integrate ``I`` over each camera pixel: ``(W_y @ I @ W_x^T) * dx^2``."""
    image = np.asarray(_cpu_array(intensity_image), dtype=np.float64)
    if image.ndim != 2:
        raise ValidationError("pixel_integrate needs a 2D intensity image")
    if rows["matrix"].shape[1] != image.shape[0] or cols["matrix"].shape[1] != image.shape[1]:
        raise ValidationError("pixel bin matrices do not match the image shape")
    return rows["matrix"] @ image @ cols["matrix"].T * float(dx_m) ** 2
