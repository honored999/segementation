"""FFT-window convergence scan for one fitted PSF pair; metrics only by default.

The device grid, ``dx``, propagation distance and the physical detection ROI are
held fixed; only the FFT side length grows.  Every window is propagated serially
and its large arrays are released before the next window starts, so the peak
allocation is one window, not the whole scan.

Four groups are measured (two artifacts x centre/corner).  A candidate window is
reported as ``supported`` only when at least ``support_windows`` strictly larger
windows are also measured and every metric deviates by less than the explicitly
configured tolerance, without a rebound.  Missing levels, failed or non-finite
samples never pass.  Without a tolerance configuration nothing is marked passed:
the scan then reports deltas only.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

import isoplanatism_core as c
from validate_isoplanatism import dump_json, guard, prepare_output, resource_preflight


DEFAULT_FFT_SIZES = (2400, 3200, 3600, 4000, 4400, 4800, 5600, 6400, 7200)
SIGNS = ("pos", "neg")
# One window is alive at a time: complex64 field + crop + two intensity planes +
# transfer function + one FFT workspace is ~56 B/px; 64 B/px keeps headroom.
BYTES_PER_PIXEL = 64


# ---------------------------------------------------------------------------
# explicit configuration
# ---------------------------------------------------------------------------


@dataclass
class Tolerances:
    """Explicit engineering budgets for the cross-window comparison.

    These are numerical-convergence budgets, not isoplanatism or device-
    accuracy thresholds; the manifest serialises this class with the rationale.
    """

    energy_roi_rel: float = 0.01
    centroid_px: float = 0.05
    shape_pixel_integrated: float = 0.01
    outside_roi_fraction: float = 0.005
    edge_fraction: float = 0.005
    edge_fraction_max: float = 0.01
    differential_energy_rel: float = 0.01
    differential_shape: float = 0.01
    rebound_ratio: float = 1.5
    rebound_floor_fraction: float = 0.25
    support_windows: int = 2

    def as_dict(self) -> Dict[str, float]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any], where: str = "<tolerances>") -> "Tolerances":
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise c.ValidationError(f"{where}: unknown tolerance keys {unknown}")
        values = {}
        for name, value in mapping.items():
            if name == "support_windows":
                if int(value) != value or int(value) < 1:
                    raise c.ValidationError(f"{where}: support_windows must be an integer >= 1")
                values[name] = int(value)
            else:
                number = float(value)
                if not math.isfinite(number) or number <= 0.0:
                    raise c.ValidationError(f"{where}: {name} must be positive and finite, got {value}")
                values[name] = number
        return cls(**values)


TOLERANCE_RATIONALE = {
    "energy_roi_rel": "Raw ROI-integrated intensity must agree to 1% across windows; conservatively loose for a detector-level budget.",
    "centroid_px": "Centroid may move by at most 0.05 pixel (20 nm at dx=400 nm); below this the registration-insensitive position is window-stable.",
    "shape_pixel_integrated": "Normalised L2 distance between camera-pixel integrated ROI patches must stay under 0.01; same scale as the existing padding shape budget.",
    "outside_roi_fraction": "The fraction of plane energy outside the fixed ROI may move by at most 0.005; larger moves mean the window still redistributes light.",
    "edge_fraction": "Edge-band energy fraction may change by at most 0.005 between neighbouring windows.",
    "edge_fraction_max": "Absolute cap on the edge-band energy fraction of any window admitted as support; a window with 1% of its energy in the border band is not a support.",
    "differential_energy_rel": "POS-NEG differential response at fixed gain must agree to 1% across windows.",
    "differential_shape": "Normalised L2 distance of the camera-pixel integrated POS-NEG differential patch must stay under 0.01.",
    "rebound_ratio": "A residual that grows by more than 1.5x against the next larger window is a rebound and forbids a pass.",
    "rebound_floor_fraction": "Rebound is only judged above 0.25 of the tolerance; below that the ordering is numerical noise, not a trend.",
    "support_windows": "Each candidate needs two strictly larger measured windows; a single successor cannot distinguish convergence from a plateau.",
}

CONVERGENCE_METRICS: Tuple[Dict[str, str], ...] = (
    dict(
        key="energy_roi_rel",
        tolerance="energy_roi_rel",
        definition="|E_a-E_b|/max(|E_a|,|E_b|) of the raw ROI integral sum(I)*dx^2",
    ),
    dict(
        key="centroid_px",
        tolerance="centroid_px",
        definition="max(|dy|,|dx|) of the ROI intensity centroid, in pixels",
    ),
    dict(
        key="shape_pixel_integrated",
        tolerance="shape_pixel_integrated",
        definition="||p_a/||p_a|| - p_b/||p_b||||_2 over the camera-pixel integrated ROI patch",
    ),
    dict(
        key="outside_roi_fraction",
        tolerance="outside_roi_fraction",
        definition="|o_a-o_b| of the extended-plane energy fraction outside the fixed ROI",
    ),
    dict(
        key="edge_fraction",
        tolerance="edge_fraction",
        definition="|e_a-e_b| of the extended-plane border-band energy fraction",
    ),
)

DIFFERENTIAL_METRICS: Tuple[Dict[str, str], ...] = (
    dict(
        key="differential_energy_rel",
        tolerance="differential_energy_rel",
        definition="|D_a-D_b|/max(|D_a|,|D_b|), D = gain_pos*E_pos - gain_neg*E_neg at the ROI",
    ),
    dict(
        key="differential_shape",
        tolerance="differential_shape",
        definition="||D̂_a-D̂_b||_2 of the normalised camera-pixel integrated differential patch",
    ),
)


# ---------------------------------------------------------------------------
# windows, points, coordinates
# ---------------------------------------------------------------------------


def padding_for(n: int, fft_size: int) -> int:
    """Half padding for an FFT side length, validated against the device grid."""
    n, fft_size = int(n), int(fft_size)
    if n < 1:
        raise c.ValidationError(f"device grid must be positive, got {n}")
    if fft_size < n:
        raise c.ValidationError(
            f"FFT size {fft_size} is smaller than the {n}px device grid; no padding is possible"
        )
    if (fft_size - n) % 2:
        raise c.ValidationError(
            f"FFT size {fft_size} minus device grid {n} is odd; the device window would not stay centred"
        )
    return (fft_size - n) // 2


def window_plan(n: int, sizes: Sequence[int], dx_m: float) -> List[Dict[str, Any]]:
    """Deterministic per-window bookkeeping, ascending in FFT size."""
    unique = sorted({int(size) for size in sizes})
    if not unique:
        raise c.ValidationError("at least one FFT size is required")
    plan = []
    for size in unique:
        padding = padding_for(n, size)
        plan.append(
            dict(
                fft_size=size,
                device_px=int(n),
                padding_px=padding,
                padding_um=padding * dx_m * 1e6,
                device_extent_um=n * dx_m * 1e6,
                fft_extent_um=size * dx_m * 1e6,
                extra_extent_um=2 * padding * dx_m * 1e6,
                check="fft_size = device_px + 2*padding_px, equal on both sides",
            )
        )
    return plan


@dataclass(frozen=True)
class Point:
    """One object-plane angle pair; the centre and one explicit far corner."""

    name: str
    theta_x_deg: float
    theta_y_deg: float

    def as_dict(self) -> Dict[str, Any]:
        return dict(name=self.name, theta_x_deg=self.theta_x_deg, theta_y_deg=self.theta_y_deg)


def check_angles(theta_x_deg: float, theta_y_deg: float) -> Tuple[float, float]:
    out = []
    for value in (theta_x_deg, theta_y_deg):
        number = float(value)
        if not math.isfinite(number) or abs(number) >= 90.0:
            raise c.ValidationError(f"angle must be finite and strictly inside (-90, 90), got {value}")
        out.append(number)
    return out[0], out[1]


def points_for_corner(theta_x_deg: float, theta_y_deg: float) -> List[Point]:
    """Centre plus the pre-specified farthest corner; never derived from data."""
    tx, ty = check_angles(theta_x_deg, theta_y_deg)
    if tx == 0.0 or ty == 0.0:
        raise c.ValidationError(
            "the corner point needs both angles non-zero; a single-axis point is not a corner"
        )
    return [Point("center", 0.0, 0.0), Point("corner", tx, ty)]


def _relative(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    if not (math.isfinite(a) and math.isfinite(b)):
        return None
    scale = max(abs(a), abs(b))
    return None if scale <= 0.0 else abs(a - b) / scale


def _normalise_patch(patch: Any) -> Optional[np.ndarray]:
    array = np.asarray(patch, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        return None
    norm = float(np.linalg.norm(array))
    return None if norm <= 0.0 else array / norm


def _patch_distance(a: Any, b: Any) -> Optional[float]:
    left, right = _normalise_patch(a), _normalise_patch(b)
    if left is None or right is None or left.shape != right.shape:
        return None
    return float(np.linalg.norm(left - right))


# ---------------------------------------------------------------------------
# per-window measurement
# ---------------------------------------------------------------------------


def measure_level(
    psf: torch.Tensor,
    extended_intensity: torch.Tensor,
    geometry: c.Geometry,
    roi_size: int,
    *,
    band_px: Optional[int] = None,
    bins_y: Optional[Mapping[str, Any]] = None,
    bins_x: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """All convergence observables for one propagated window.

    The detection ROI is the same physical box in the n*n crop and in the
    extended plane; the crop is only ever a centred sub-array of the plane.
    """
    n, dx_m = geometry.n, geometry.dx_m
    size = int(extended_intensity.shape[0])
    roi = c.roi_window(n, roi_size)
    roi_extended = c.roi_window(size, roi_size)
    offset = (size - n) // 2
    aligned = (roi["y0"] + offset, roi["x0"] + offset) == (roi_extended["y0"], roi_extended["x0"])
    if not roi["valid"] or roi["size"] != roi_size:
        raise c.ValidationError(
            f"ROI {roi_size}px is clipped by the {n}px detector crop; "
            "the fixed detection window must fit entirely")
    if not roi_extended["valid"] or roi_extended["size"] != roi_size:
        raise c.ValidationError(f"ROI {roi_size}px is clipped by the {size}px extended plane")

    centroid = c.centroid(psf, dx_m, roi)
    budget_extended = c.roi_energy(extended_intensity, roi_extended)
    boundary_window = c.boundary_stats(psf, band_px)
    boundary_extended = c.boundary_stats(extended_intensity, band_px)

    psf_cpu = psf.detach().cpu().numpy()
    patch = None
    if bins_y is not None and bins_x is not None:
        # Bins describe the fixed ROI, so integrate exactly that sub-array.
        patch = c.pixel_integrate(
            psf_cpu[roi["y0"]:roi["y1"], roi["x0"]:roi["x1"]], bins_y, bins_x, dx_m)

    energy_roi = c.roi_sum(psf, roi) * dx_m**2
    energy_window = float(torch.as_tensor(psf).sum(dtype=torch.float64)) * dx_m**2
    energy_extended = float(torch.as_tensor(extended_intensity).sum(dtype=torch.float64)) * dx_m**2
    return dict(
        roi_window=roi,
        roi_window_extended=roi_extended,
        roi_alignment=dict(
            aligned=bool(aligned),
            extended_offset_px=int(offset),
            rule="extended-plane ROI = crop ROI shifted by (fft_size-device_px)/2",
        ),
        centroid=asdict(centroid),
        centroid_y_px=centroid.y_px,
        centroid_x_px=centroid.x_px,
        centroid_y_um=centroid.y_um,
        centroid_x_um=centroid.x_um,
        energy_roi_au=energy_roi,
        energy_window_au=energy_window,
        energy_extended_au=energy_extended,
        energy_output_over_extended=(energy_window / energy_extended) if energy_extended else None,
        energy_roi_over_extended=(energy_roi / energy_extended) if energy_extended else None,
        outside_roi_fraction=budget_extended.outside_fraction,
        outside_roi_au=budget_extended.outside_au * dx_m**2,
        roi_inside_fraction=budget_extended.inside_fraction,
        edge_fraction_extended=boundary_extended.edge_energy_fraction,
        edge_fraction_window=boundary_window.edge_energy_fraction,
        boundary_extended=asdict(boundary_extended),
        boundary_window=asdict(boundary_window),
        pixel_patch=None if patch is None else np.asarray(patch, dtype=np.float64).tolist(),
        pixel_patch_norm=None if patch is None else float(np.linalg.norm(patch)),
    )


def differential_rows(level_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """POS-NEG differential response at fixed gain, one entry per level/point."""
    by_key = {(row["artifact"], row["point"]): row for row in level_rows}
    labels = {}
    for row in level_rows:
        labels.setdefault(row["point"], set()).add((row["artifact"], row["sign"]))
    out: List[Dict[str, Any]] = []
    for point, pairs in sorted(labels.items()):
        pos = next((label for label, sign in pairs if sign == "pos"), None)
        neg = next((label for label, sign in pairs if sign == "neg"), None)
        reasons: List[str] = []
        energy = patch = None
        gains = [None, None]
        if pos is None or neg is None:
            reasons.append("missing_pos_or_neg_branch")
        else:
            first, second = by_key.get((pos, point)), by_key.get((neg, point))
            if first is None or second is None:
                reasons.append("missing_level_for_one_sign")
            else:
                gains = [first.get("gain_gray"), second.get("gain_gray")]
                if not all(isinstance(g, (int, float)) and math.isfinite(float(g)) for g in gains):
                    reasons.append("missing_finite_gain_gray")
                elif first["status"] != "measured" or second["status"] != "measured":
                    reasons.append("non_measured_level")
                else:
                    energy = float(gains[0]) * first["energy_roi_au"] - float(gains[1]) * second["energy_roi_au"]
                    left = _normalise_patch(first["pixel_patch"])
                    right = _normalise_patch(second["pixel_patch"])
                    if left is None or right is None:
                        reasons.append("missing_pixel_patch")
                    else:
                        patch = (float(gains[0]) * left - float(gains[1]) * right).tolist()
        fft_size = level_rows[0]["fft_size"] if level_rows else None
        out.append(
            dict(
                point=point,
                fft_size=fft_size,
                status="measured" if not reasons else "undefined",
                reasons=reasons,
                gain_pos=gains[0],
                gain_neg=gains[1],
                differential_energy_au=energy,
                differential_patch=patch,
                definition="D = gain_pos*E_pos - gain_neg*E_neg with both gains held fixed at their artifact values",
            )
        )
    return out


# ---------------------------------------------------------------------------
# convergence evaluation
# ---------------------------------------------------------------------------


def _metric_deltas(
    candidate: Mapping[str, Any],
    other: Mapping[str, Any],
    tolerance_map: Optional[Mapping[str, float]] = None,
) -> Dict[str, Optional[float]]:
    return dict(
        energy_roi_rel=_relative(candidate.get("energy_roi_au"), other.get("energy_roi_au")),
        centroid_px=(
            None
            if candidate.get("centroid_y_px") is None or other.get("centroid_y_px") is None
            else max(
                abs(candidate["centroid_y_px"] - other["centroid_y_px"]),
                abs(candidate["centroid_x_px"] - other["centroid_x_px"]),
            )
        ),
        shape_pixel_integrated=_patch_distance(candidate.get("pixel_patch"), other.get("pixel_patch")),
        outside_roi_fraction=(
            None
            if candidate.get("outside_roi_fraction") is None or other.get("outside_roi_fraction") is None
            else abs(candidate["outside_roi_fraction"] - other["outside_roi_fraction"])
        ),
        edge_fraction=(
            None
            if candidate.get("edge_fraction_extended") is None
            or other.get("edge_fraction_extended") is None
            else abs(candidate["edge_fraction_extended"] - other["edge_fraction_extended"])
        ),
    )


def _differential_deltas(candidate: Mapping[str, Any], other: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    return dict(
        differential_energy_rel=_relative(
            candidate.get("differential_energy_au"), other.get("differential_energy_au")
        ),
        differential_shape=_patch_distance(
            candidate.get("differential_patch"), other.get("differential_patch")
        ),
    )


def _judge(
    deltas: Sequence[Mapping[str, Any]],
    metrics: Iterable[Mapping[str, str]],
    tolerances: Tolerances,
    label: str,
) -> Tuple[List[str], Dict[str, Any]]:
    """Tolerance and rebound verdict for one candidate against its supporters."""
    reasons: List[str] = []
    detail: Dict[str, Any] = {}
    for metric in metrics:
        key, limit = metric["key"], float(getattr(tolerances, metric["tolerance"]))
        values = [entry["deltas"].get(key) for entry in deltas]
        missing = [i for i, value in enumerate(values) if value is None]
        if missing:
            reasons.append(f"{key}: undefined against supporter(s) {missing}")
            detail[key] = dict(values=values, tolerance=limit, exceeded=None, rebound=None)
            continue
        worst = max(values)
        exceeded = worst > limit
        rebound = False
        for index in range(1, len(values)):
            previous, following = values[index - 1], values[index]
            if following <= tolerances.rebound_floor_fraction * limit:
                continue
            if previous <= tolerances.rebound_floor_fraction * limit:
                rebound = True
            elif following > tolerances.rebound_ratio * previous:
                rebound = True
        if exceeded:
            reasons.append(f"{key}: {worst:.6g} > tolerance {limit:.6g}")
        if rebound:
            reasons.append(f"{key}: residual rebounds across larger windows {values}")
        detail[key] = dict(values=values, tolerance=limit, exceeded=bool(exceeded), rebound=bool(rebound))
    if label:
        detail["label"] = label
    return reasons, detail


def evaluate_group(
    rows: Sequence[Mapping[str, Any]],
    tolerances: Optional[Tolerances],
    *,
    group: str = "",
    metrics: Sequence[Mapping[str, str]] = CONVERGENCE_METRICS,
    differential: bool = False,
) -> Dict[str, Any]:
    """Evaluate one measured family (artifact x point, or the POS-NEG pair)."""
    ordered = sorted(rows, key=lambda row: row["fft_size"])
    support = tolerances.support_windows if tolerances else 2
    candidates: List[Dict[str, Any]] = []
    for index, row in enumerate(ordered):
        entry: Dict[str, Any] = dict(
            fft_size=row["fft_size"],
            padding_px=row.get("padding_px"),
            status="metrics_only" if tolerances is None else "not_evaluated",
            reasons=[],
            supporters=[],
            deltas={},
        )
        if row.get("status") != "measured":
            entry["status"] = row.get("status", "not_measured")
            entry["reasons"].append(f"level_not_measured: {row.get('status_reason')}")
            candidates.append(entry)
            continue
        edge = row.get("edge_fraction_extended")
        capped = False
        if (
            tolerances is not None
            and not differential
            and (edge is None or edge > tolerances.edge_fraction_max)
        ):
            capped = True
            entry["status"] = "exceeds_tolerance"
            entry["reasons"].append(
                f"edge_fraction_extended={edge} > cap {tolerances.edge_fraction_max}"
            )
        successors = ordered[index + 1 : index + 1 + support]
        entry["supporters"] = [item["fft_size"] for item in successors]
        if len(successors) < support:
            if tolerances is not None and not capped:
                entry["status"] = "insufficient_support"
            entry["reasons"].append(
                f"needs {support} larger window(s), found {len(successors)}"
            )
            candidates.append(entry)
            continue
        if any(item.get("status") != "measured" for item in successors):
            if tolerances is not None and not capped:
                entry["status"] = "missing_level"
            entry["reasons"].append(
                "larger window not measured: "
                + ", ".join(str(item["fft_size"]) for item in successors if item.get("status") != "measured")
            )
            candidates.append(entry)
            continue
        deltas = [
            dict(
                fft_size=item["fft_size"],
                deltas=(
                    _differential_deltas(row, item) if differential else _metric_deltas(row, item)
                ),
            )
            for item in successors
        ]
        entry["deltas"] = {str(item["fft_size"]): item["deltas"] for item in deltas}
        if tolerances is None:
            entry["status"] = "metrics_only"
        else:
            reasons, detail = _judge(deltas, metrics, tolerances, group)
            entry["detail"] = detail
            if reasons:
                entry["reasons"].extend(reasons)
                if any("undefined" in reason for reason in reasons):
                    entry["status"] = "undefined_metric"
                elif any(">" in reason for reason in reasons):
                    entry["status"] = "exceeds_tolerance"
                else:
                    entry["status"] = "rebound"
            elif not capped:
                entry["status"] = "supported"
        candidates.append(entry)
    supported = [entry["fft_size"] for entry in candidates if entry["status"] == "supported"]
    if tolerances is None:
        status = "metrics_only"
    elif supported:
        status = "converged"
    else:
        status = "not_converged"
    return dict(
        group=group,
        status=status,
        first_supported_fft_size=min(supported) if supported else None,
        support_windows=support,
        metrics=[metric["key"] for metric in metrics],
        levels=candidates,
        fft_sizes=[row["fft_size"] for row in ordered],
    )


def evaluate(
    rows: Sequence[Mapping[str, Any]],
    differentials: Sequence[Mapping[str, Any]],
    tolerances: Optional[Tolerances],
) -> Dict[str, Any]:
    """All four families plus the POS-NEG differential, with one run verdict."""
    groups = {}
    for row in rows:
        key = f"{row['artifact']}|{row['sign']}|{row['point']}"
        groups.setdefault(key, []).append(row)
    evaluated = {
        key: evaluate_group(value, tolerances, group=key) for key, value in sorted(groups.items())
    }
    differential_groups = {}
    for row in differentials:
        differential_groups.setdefault(row["point"], []).append(row)
    evaluated_differential = {
        point: evaluate_group(
            value,
            tolerances,
            group=f"differential|{point}",
            metrics=DIFFERENTIAL_METRICS,
            differential=True,
        )
        for point, value in sorted(differential_groups.items())
    }
    first_supported = [
        item["first_supported_fft_size"]
        for item in list(evaluated.values()) + list(evaluated_differential.values())
        if item["first_supported_fft_size"] is not None
    ]
    total = len(evaluated) + len(evaluated_differential)
    if tolerances is None:
        status = "METRICS_ONLY"
        converged_from = None
    elif len(first_supported) == total and total:
        status = "CONVERGED"
        converged_from = max(first_supported)
    else:
        status = "NOT_CONVERGED"
        converged_from = None
    return dict(
        status=status,
        scope="all four artifact/point families plus the POS-NEG differential at fixed gain",
        converged_from_fft_size=converged_from,
        groups=evaluated,
        differential_groups=evaluated_differential,
        note=(
            "No earlier window is claimed converged: the reported window is the smallest one "
            "supported by two strictly larger windows in every family. Missing, non-finite or "
            "rebounding levels never yield a pass, and no threshold means no verdict."
            if tolerances is not None
            else "No tolerance configuration supplied: deltas are reported without any pass/fail."
        ),
    )


# ---------------------------------------------------------------------------
# dx re-check (reported separately, never folded into the window verdict)
# ---------------------------------------------------------------------------


def dx_review(
    config: c.SourceConfig,
    geometry: c.Geometry,
    *,
    roi_size: int,
    camera_pixel_m: float,
    plan: Sequence[Mapping[str, Any]],
    points: Sequence[Point],
) -> Dict[str, Any]:
    """Independent re-check of dx, the FFT mapping and the ROI physical size."""
    checks: List[Dict[str, Any]] = []

    def add(name: str, expected: Any, actual: Any, ok: bool, note: str = "") -> None:
        checks.append(dict(name=name, expected=expected, actual=actual, ok=bool(ok), note=note))

    dx_nm = geometry.dx_m * 1e9
    add("dx_nm_matches_core_constant", c.DX_NM, dx_nm, abs(dx_nm - c.DX_NM) < 1e-6,
        "dx is the fixed 400 nm library grid; a different dx needs a different locked library.")
    add("dx_nm_matches_source_config", config.dx_nm, dx_nm, abs(dx_nm - float(config.dx_nm)) < 1e-6,
        "Only the config dx reaches every metric through Geometry.")
    add("wavelength_nm_matches_config", config.wavelength_nm, geometry.wavelength_m * 1e9,
        abs(config.wavelength_nm - geometry.wavelength_m * 1e9) < 1e-6)
    add("distance_um_matches_config", config.distance_um, geometry.distance_m * 1e6,
        abs(float(config.distance_um) - geometry.distance_m * 1e6) < 1e-6)
    patch_camera = float(config.raw.get("patch_camera", config.patch_meta))
    patch_meta_from_pitch = int(patch_camera * (camera_pixel_m * 1e6) * 1000.0 / dx_nm)
    add("patch_meta_from_camera_pixel", config.patch_meta, patch_meta_from_pitch,
        patch_meta_from_pitch == config.patch_meta,
        "patch_meta = int(patch_camera*camera_pixel_um*1000/dx_nm); re-derived from the pitch used here.")
    roi_um = roi_size * geometry.dx_m * 1e6
    camera_um = patch_camera * camera_pixel_m * 1e6
    add("roi_physical_width_um", camera_um, roi_um, abs(roi_um - camera_um) <= geometry.dx_m * 1e6,
        "The fixed ROI must cover the same physical support as the fitted camera patch.")
    for item in plan:
        add(
            f"fft_mapping_{item['fft_size']}",
            item["fft_size"],
            item["device_px"] + 2 * item["padding_px"],
            item["device_px"] + 2 * item["padding_px"] == item["fft_size"],
            f"padding {item['padding_px']}px = {item['padding_um']:.3f} um per side",
        )
    for point in points:
        dy_px, dx_px = c.lateral_shift_theory(geometry, point.theta_x_deg, point.theta_y_deg)
        theory_um = (
            geometry.distance_m
            * math.tan(math.radians(point.theta_y_deg)),
            geometry.distance_m * math.tan(math.radians(point.theta_x_deg)),
        )
        add(
            f"theory_shift_um_from_geometry_{point.name}",
            [value * 1e6 for value in theory_um],
            [dy_px * geometry.dx_m * 1e6, dx_px * geometry.dx_m * 1e6],
            abs(theory_um[0] * 1e6 - dy_px * geometry.dx_m * 1e6) < 1e-6
            and abs(theory_um[1] * 1e6 - dx_px * geometry.dx_m * 1e6) < 1e-6,
            "z*tan(theta) reported in both um and px through the same dx.",
        )
    bins = c.pixel_bin_matrix(roi_size, geometry.dx_m, camera_pixel_m)
    add(
        "camera_pixel_binning",
        "integer pixel pitch >= dx",
        dict(pixel_px=bins["pixel_px"], bins=bins["bins"], partial_bins=bins["partial_bins"]),
        bins["pixel_px"] >= 1.0,
        "Partial edge bins keep their true width; nothing is renormalised.",
    )
    return dict(
        status="PASS" if all(check["ok"] for check in checks) else "FAIL",
        checks=checks,
        dx_nm=dx_nm,
        roi_size_px=int(roi_size),
        roi_width_um=roi_um,
        camera_pixel_um=camera_pixel_m * 1e6,
        note="Reported separately from the window verdict; a FAIL blocks any convergence conclusion.",
    )


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _release(device: str) -> None:
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(
    branches: Sequence[Mapping[str, Any]],
    geometry: c.Geometry,
    points: Sequence[Point],
    sizes: Sequence[int],
    *,
    roi_size: int,
    device: str,
    out: Path,
    tolerances: Optional[Tolerances],
    camera_pixel_m: float,
    band_px: Optional[int] = None,
    plan_only: bool = False,
    sampling: bool = False,
    engine_factory=None,
) -> Dict[str, Any]:
    """Serial scan over FFT windows; one window alive at a time."""
    if {branch["sign"] for branch in branches} != set(SIGNS):
        raise c.ValidationError("exactly one pos and one neg artifact are required for the differential response")
    if len({branch["label"] for branch in branches}) != len(branches):
        raise c.ValidationError("artifact labels must be distinct")
    n, dx_m = geometry.n, geometry.dx_m
    plan = window_plan(n, sizes, dx_m)
    roi = c.roi_window(n, roi_size)
    if not roi["valid"]:
        raise c.ValidationError(f"ROI size {roi_size} is outside the {n}px device window")
    bins_y = c.pixel_bin_matrix(roi_size, dx_m, camera_pixel_m)
    bins_x = c.pixel_bin_matrix(roi_size, dx_m, camera_pixel_m)
    out = Path(out)
    samples = out / "samples"
    samples.mkdir(exist_ok=False)

    rows: List[Dict[str, Any]] = []
    differentials: List[Dict[str, Any]] = []
    reference: Dict[Tuple[str, str], Dict[str, Any]] = {}
    engine_factory = engine_factory or c.PropagationEngine
    for item in plan:
        padding = item["padding_px"]
        started = time.time()
        engine = engine_factory(geometry, device)
        level_rows: List[Dict[str, Any]] = []
        for branch in branches:
            for point in points:
                row = _measure_sample(
                    branch,
                    point,
                    item,
                    geometry,
                    engine,
                    padding,
                    roi_size,
                    band_px,
                    bins_y,
                    bins_x,
                    samples,
                    len(rows),
                    sampling,
                )
                key = (branch["label"], point.name)
                if row["status"] == "measured" and key not in reference:
                    reference[key] = dict(
                        energy_roi_au=row["energy_roi_au"],
                        pixel_patch=row["pixel_patch"],
                        fft_size=item["fft_size"],
                    )
                row["reference_fft_size"] = reference.get(key, {}).get("fft_size")
                row["energy_roi_rel_to_reference"] = _relative(
                    row.get("energy_roi_au"), reference.get(key, {}).get("energy_roi_au")
                )
                row["shape_pixel_integrated_rel_to_reference"] = _patch_distance(
                    row.get("pixel_patch"), reference.get(key, {}).get("pixel_patch")
                )
                row["elapsed_level_s"] = None
                level_rows.append(row)
                rows.append(row)
        differentials.extend(differential_rows(level_rows))
        elapsed = time.time() - started
        for row in level_rows:
            row["elapsed_level_s"] = elapsed
        engine = None
        _release(device)
        print(
            json.dumps(
                dict(
                    fft_size=item["fft_size"],
                    padding_px=padding,
                    rows=len(level_rows),
                    elapsed_s=round(elapsed, 3),
                )
            ),
            flush=True,
        )
        del level_rows
    conclusion = evaluate(rows, differentials, tolerances)
    return dict(rows=rows, differentials=differentials, plan=plan, conclusion=conclusion)


def _measure_sample(
    branch: Mapping[str, Any],
    point: Point,
    item: Mapping[str, Any],
    geometry: c.Geometry,
    engine: Any,
    padding: int,
    roi_size: int,
    band_px: Optional[int],
    bins_y: Mapping[str, Any],
    bins_x: Mapping[str, Any],
    samples: Path,
    index: int,
    sampling: bool,
) -> Dict[str, Any]:
    sample_id = f"sample_{index:06d}"
    row: Dict[str, Any] = dict(
        sample_id=sample_id,
        artifact=branch["label"],
        sign=branch["sign"],
        point=point.name,
        theta_x_deg=point.theta_x_deg,
        theta_y_deg=point.theta_y_deg,
        fft_size=item["fft_size"],
        padding_px=padding,
        padding_um=item["padding_um"],
        fft_extent_um=item["fft_extent_um"],
        gain_gray=branch.get("gain_gray"),
        artifact_sha256=branch.get("sha256"),
        status="failed",
        status_reason=None,
    )
    shift_y, shift_x = c.lateral_shift_theory(geometry, point.theta_x_deg, point.theta_y_deg)
    row.update(
        shift_theory_dy_px=shift_y,
        shift_theory_dx_px=shift_x,
        shift_theory_dy_um=shift_y * geometry.dx_m * 1e6,
        shift_theory_dx_um=shift_x * geometry.dx_m * 1e6,
    )
    try:
        tilted = c.apply_carrier(branch["field"], geometry, point.theta_x_deg, point.theta_y_deg)
        full, propagated, layer = engine.extended(tilted, padding)
        psf = c.intensity(propagated)
        extended = c.intensity(full)
        del full, propagated
        finite = bool(torch.isfinite(psf).all() and torch.isfinite(extended).all())
        if not finite:
            row["status"] = "nonfinite"
            row["status_reason"] = "propagated intensity contains non-finite values"
            return row, None
        measured = measure_level(
            psf,
            extended,
            geometry,
            roi_size,
            band_px=band_px,
            bins_y=bins_y,
            bins_x=bins_x,
        )
        row.update(measured)
        if sampling:
            row["sampling"] = asdict(
                c.sampling_diagnostics(tilted, layer, geometry, point.theta_x_deg, point.theta_y_deg)
            )
        row["status"] = "measured"
        npz_file = f"{sample_id}_{branch['label']}_{point.name}_p{padding}.npz"
        roi = row["roi_window"]
        wide = row["roi_window_extended"]
        with (samples / npz_file).open("xb") as handle:
            np.savez_compressed(
                handle,
                roi_intensity=psf[roi["y0"] : roi["y1"], roi["x0"] : roi["x1"]]
                .detach()
                .cpu()
                .numpy(),
                pixel_patch=np.asarray(
                    row["pixel_patch"] if row["pixel_patch"] is not None else [], dtype=np.float64
                ),
                extended_roi_intensity=extended[wide["y0"] : wide["y1"], wide["x0"] : wide["x1"]]
                .detach()
                .cpu()
                .numpy(),
                sample_id=np.array(sample_id),
                fft_size=np.array(item["fft_size"]),
                padding=np.array(padding),
                angle_deg=np.array([point.theta_x_deg, point.theta_y_deg]),
            )
        row["npz_file"] = f"samples/{npz_file}"
        del psf, extended, tilted
    except Exception as exc:  # noqa: BLE001 - a failed window must be recorded, not fatal
        row["status"] = "failed"
        row["status_reason"] = f"{type(exc).__name__}: {exc}"
        print(f"{sample_id} ({item['fft_size']}px) failed: {row['status_reason']}", file=sys.stderr, flush=True)
    return row


# ---------------------------------------------------------------------------
# run configuration and CLI
# ---------------------------------------------------------------------------


RUN_CONFIG_KEYS = {
    "schema",
    "description",
    "fft_sizes",
    "corner_theta_deg",
    "corner_source",
    "tolerances",
    "edge_band_px",
    "roi_size_px",
    "camera_pixel_um",
    "notes",
}


def load_run_config(path: Path) -> Dict[str, Any]:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise c.ValidationError(f"{path}: cannot read run configuration ({exc})") from exc
    if not isinstance(data, dict):
        raise c.ValidationError(f"{path}: run configuration must be a JSON object")
    unknown = sorted(set(data) - RUN_CONFIG_KEYS)
    if unknown:
        raise c.ValidationError(f"{path}: unknown run configuration keys {unknown}")
    return data


def build_document(
    *,
    args: argparse.Namespace,
    config: c.SourceConfig,
    geometry: c.Geometry,
    artifacts: Sequence[c.Artifact],
    points: Sequence[Point],
    plan: Sequence[Mapping[str, Any]],
    resources: Mapping[str, Any],
    locks: Mapping[str, Any],
    tolerances: Optional[Tolerances],
    camera_pixel_m: float,
    dx: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    differentials: Sequence[Mapping[str, Any]],
    conclusion: Optional[Mapping[str, Any]],
    run_config: Mapping[str, Any],
) -> Dict[str, Any]:
    rois = c.roi_window(geometry.n, int(args.roi_size))
    return dict(
        status="RUNNING",
        run_class="fft_window_convergence",
        arguments={
            key: ([str(item) for item in value] if isinstance(value, list) else str(value))
            for key, value in vars(args).items()
        },
        run_config=dict(run_config),
        source_config=config.raw,
        artifacts=[
            dict(
                label=artifact.branch,
                path=str(artifact.path),
                sha256=artifact.sha256,
                size_bytes=artifact.size_bytes,
                wavelength_nm=artifact.wavelength_nm,
                gain_gray=artifact.gain_gray,
                active_gray=artifact.active_gray,
                skipped=artifact.skipped,
                width_range_nm=(
                    None
                    if artifact.width_nm is None or artifact.width_nm.size == 0
                    else [float(artifact.width_nm.min()), float(artifact.width_nm.max())]
                ),
            )
            for artifact in artifacts
        ],
        geometry=asdict(geometry),
        points=[point.as_dict() for point in points],
        detection=dict(
            roi_window=rois,
            roi_size_px=int(args.roi_size),
            roi_width_um=int(args.roi_size) * geometry.dx_m * 1e6,
            camera_pixel_um=camera_pixel_m * 1e6,
            camera_pixel_over_dx=camera_pixel_m / geometry.dx_m,
            device_px=geometry.n,
            rule=(
                "One physical ROI, fixed for every FFT size: centred on the device grid in the "
                "crop and shifted by (fft-device)/2 in the extended plane. It is never moved to "
                "follow a PSF and never resized between windows."
            ),
        ),
        windows=list(plan),
        tolerances=None if tolerances is None else tolerances.as_dict(),
        tolerance_rationale=None if tolerances is None else TOLERANCE_RATIONALE,
        tolerances_source="explicit run configuration" if tolerances is not None else "none supplied",
        metric_definitions=list(CONVERGENCE_METRICS) + list(DIFFERENTIAL_METRICS),
        dx_review=dict(dx),
        convergence=dict(conclusion) if conclusion is not None else None,
        resources=dict(resources),
        locked_sources=dict(locks),
        input_hashes={str(artifact.path.resolve()): artifact.sha256 for artifact in artifacts}
        | {str(Path(args.config).resolve()): config.sha256},
        python=sys.version,
        torch=torch.__version__,
        numpy=np.__version__,
        device=args.device,
        coordinates="[row,col]=[y,x], pixel centres (index-(N-1)/2)*dx; origin upper, +y down",
        limitations=[
            "phase-only normal-incidence library; no angular electromagnetic response",
            "window convergence is a numerical statement about this FFT grid, not device accuracy",
            "the extended plane is also finite: convergence does not exclude wrap-around or aliasing",
            "gains are held fixed at their artifact values; no refit is performed per window",
            "no isoplanatism or hardware PASS/FAIL is derived here",
        ],
        notes=list(run_config.get("notes", [])),
    )


def write_rows(out: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (Path(out) / "window_convergence_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, allow_nan=False) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, action="append", required=True,
                        help="one kernel_000_pos.npz and one kernel_000_neg.npz")
    parser.add_argument("--config", type=Path, required=True, help="fit config.json for the artifacts")
    parser.add_argument("--out", required=True, type=Path, help="fresh output directory outside the checkout")
    parser.add_argument("--run-config", type=Path, help="JSON with fft_sizes, corner_theta_deg, tolerances")
    parser.add_argument("--fft-sizes", type=int, nargs="+")
    parser.add_argument("--corner-deg", type=float, nargs=2, metavar=("TX", "TY"))
    parser.add_argument("--roi-size", type=int)
    parser.add_argument("--camera-pixel-um", type=float)
    parser.add_argument("--edge-band-px", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-thresholds", action="store_true",
                        help="report deltas only; never mark a window passed")
    parser.add_argument("--plan-only", action="store_true",
                        help="resources, window mapping and dx review only; no propagation")
    parser.add_argument("--sampling-diagnostics", action="store_true",
                        help="also record Nyquist/transfer-function sampling flags per window")
    args = parser.parse_args(argv)

    run_config = load_run_config(args.run_config) if args.run_config else {}
    sizes = args.fft_sizes or run_config.get("fft_sizes") or list(DEFAULT_FFT_SIZES)
    corner = args.corner_deg or run_config.get("corner_theta_deg")
    if corner is None:
        parser.error("--corner-deg TX TY or run_config.corner_theta_deg is required: the far corner is pre-specified")
    points = points_for_corner(*corner)
    camera_pixel_um = args.camera_pixel_um or run_config.get("camera_pixel_um")
    if camera_pixel_um is None:
        parser.error("--camera-pixel-um or run_config.camera_pixel_um is required for pixel integration")
    camera_pixel_m = float(camera_pixel_um) * 1e-6
    raw_tolerances = run_config.get("tolerances")
    if args.no_thresholds:
        tolerances = None
    elif raw_tolerances is not None:
        tolerances = Tolerances.from_mapping(raw_tolerances, where=str(args.run_config))
    else:
        tolerances = None
    torch.set_num_threads(1)
    source = Path(__file__).resolve().parent
    locks = c.verify_locked_sources(source)

    config = c.load_source_config(args.config)
    geometry = c.Geometry.from_config(config)
    artifacts = [c.load_artifact(path) for path in args.artifact]
    for artifact in artifacts:
        c.cross_check_config_artifact(config, artifact)
        if artifact.skipped:
            raise c.ValidationError(
                f"{artifact.path.name}: empty width (zero branch); a convergence scan needs a physical PSF"
            )
    branches = []
    for artifact in artifacts:
        sign = artifact.branch.rsplit("_", 1)[-1].lower()
        if sign not in SIGNS:
            raise c.ValidationError(
                f"{artifact.path.name}: branch name must end in _pos or _neg for the differential response"
            )
        branches.append(
            dict(
                label=artifact.branch,
                sign=sign,
                sha256=artifact.sha256,
                gain_gray=artifact.gain_gray,
                field=c.build_transmission(artifact.width_nm, artifact.wavelength_nm),
            )
        )
    if sorted(branch["sign"] for branch in branches) != sorted(SIGNS):
        raise c.ValidationError("supply exactly one _pos and one _neg artifact")
    roi_size = args.roi_size or run_config.get("roi_size_px") or config.patch_meta
    args.roi_size = int(roi_size)
    plan = window_plan(geometry.n, sizes, geometry.dx_m)
    dx = dx_review(
        config,
        geometry,
        roi_size=int(roi_size),
        camera_pixel_m=camera_pixel_m,
        plan=plan,
        points=points,
    )
    if dx["status"] != "PASS":
        failed = [check["name"] for check in dx["checks"] if not check["ok"]]
        raise c.ValidationError(f"dx re-check failed: {failed}")

    paddings = [item["padding_px"] for item in plan]
    resources = resource_preflight()
    guard(resources, geometry.n, paddings, args.device, bytes_per_px=BYTES_PER_PIXEL)
    inputs = [args.config] + list(args.artifact)
    out = prepare_output(args.out, inputs)
    document = build_document(
        args=args,
        config=config,
        geometry=geometry,
        artifacts=artifacts,
        points=points,
        plan=plan,
        resources=resources,
        locks=locks,
        tolerances=tolerances,
        camera_pixel_m=camera_pixel_m,
        dx=dx,
        rows=[],
        differentials=[],
        conclusion=None,
        run_config=run_config,
    )
    document["status"] = "PLANNED" if args.plan_only else "RUNNING"
    dump_json(out / "window_convergence.json", document)
    if args.plan_only:
        print(json.dumps(dict(status="PLANNED", windows=[item["fft_size"] for item in plan], out=str(out))))
        return document
    try:
        with torch.no_grad():
            result = run(
                branches,
                geometry,
                points,
                sizes,
                roi_size=int(roi_size),
                device=args.device,
                out=out,
                tolerances=tolerances,
                camera_pixel_m=camera_pixel_m,
                band_px=args.edge_band_px or run_config.get("edge_band_px"),
                sampling=args.sampling_diagnostics,
            )
        document.update(
            rows=result["rows"],
            differential=result["differentials"],
            convergence=result["conclusion"],
            status="COMPLETED",
        )
        write_rows(out, result["rows"])
    except Exception as exc:
        document["status"] = "FAILED"
        document["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        dump_json(out / "window_convergence.json", document)
    print(
        json.dumps(
            dict(
                status=document["status"],
                rows=len(document.get("rows") or []),
                convergence=document["convergence"]["status"],
                converged_from_fft_size=document["convergence"]["converged_from_fft_size"],
                out=str(out),
            )
        )
    )
    return document


if __name__ == "__main__":
    try:
        main()
    except c.ValidationError as exc:
        raise SystemExit(str(exc))
