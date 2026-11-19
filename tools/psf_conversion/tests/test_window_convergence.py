"""Unit tests for the FFT-window convergence scan.

Only pure bookkeeping is exercised here: window bookkeeping, the single fixed
detection ROI, the convergence verdict and the dx re-check. No propagation and
no large window is ever started; the real scan lives on the GPU server.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isoplanatism_core as c
import validate_isoplanatism as cli
import window_convergence as w


# ---------------------------------------------------------------------------
# window bookkeeping
# ---------------------------------------------------------------------------

def test_padding_maps_each_fft_side_to_a_centred_padding():
    assert w.padding_for(800, 2400) == 800
    assert w.padding_for(800, 800) == 0
    with pytest.raises(c.ValidationError):
        w.padding_for(800, 700)
    with pytest.raises(c.ValidationError):
        w.padding_for(800, 2401)


def test_window_plan_is_ordered_deduplicated_and_self_checked():
    plan = w.window_plan(800, (3200, 2400, 2400), 400e-9)
    assert [item["fft_size"] for item in plan] == [2400, 3200]
    first = plan[0]
    assert first["device_px"] == 800
    assert first["padding_px"] == 800
    assert first["padding_um"] == pytest.approx(800 * 400e-9 * 1e6)
    assert first["device_extent_um"] == pytest.approx(800 * 400e-9 * 1e6)
    assert first["fft_extent_um"] == pytest.approx(2400 * 400e-9 * 1e6)
    assert first["extra_extent_um"] == pytest.approx(1600 * 400e-9 * 1e6)
    assert first["fft_size"] == first["device_px"] + 2 * first["padding_px"]


def test_default_windows_start_at_the_device_size_and_grow():
    assert w.DEFAULT_FFT_SIZES[0] == 2400
    assert list(w.DEFAULT_FFT_SIZES) == sorted(w.DEFAULT_FFT_SIZES)
    assert len(set(w.DEFAULT_FFT_SIZES)) == len(w.DEFAULT_FFT_SIZES)


def test_tolerances_parse_from_a_mapping_and_reject_unknown_keys():
    default = w.Tolerances()
    assert default.support_windows == 2
    assert default.rebound_ratio > 1
    overridden = w.Tolerances.from_mapping({"energy_roi_rel": 0.02})
    assert overridden.energy_roi_rel == 0.02
    assert overridden.centroid_px == default.centroid_px
    with pytest.raises(c.ValidationError):
        w.Tolerances.from_mapping({"not_a_tolerance": 1.0})
    with pytest.raises(c.ValidationError):
        w.Tolerances.from_mapping({"support_windows": 0})


def test_corner_point_needs_two_tilted_axes():
    points = w.points_for_corner(0.5, -0.5)
    assert [p.name for p in points] == ["center", "corner"]
    assert points[0].theta_x_deg == 0 and points[0].theta_y_deg == 0
    assert points[1].theta_x_deg == 0.5 and points[1].theta_y_deg == -0.5
    with pytest.raises(c.ValidationError):
        w.points_for_corner(0.0, 0.5)


# ---------------------------------------------------------------------------
# one fixed physical ROI across every window
# ---------------------------------------------------------------------------

def test_measure_level_keeps_one_fixed_physical_roi():
    g = c.Geometry(33, 400e-9, 450e-9, 2400e-6)
    y, x = torch.meshgrid(torch.arange(33), torch.arange(33), indexing="ij")
    psf = torch.exp(-((y - 16.0) ** 2 + (x - 16.0) ** 2) / 8.0)
    extended = torch.nn.functional.pad(psf, (32, 32, 32, 32))
    bins = c.pixel_bin_matrix(9, g.dx_m, 3.45e-6)

    level = w.measure_level(psf, extended, g, 9, bins_y=bins, bins_x=bins)

    roi, roi_extended = level["roi_window"], level["roi_window_extended"]
    assert level["roi_alignment"]["aligned"] is True
    assert level["roi_alignment"]["extended_offset_px"] == 32
    assert (roi_extended["y0"], roi_extended["x0"]) == (roi["y0"] + 32, roi["x0"] + 32)
    assert level["energy_roi_au"] == pytest.approx(c.roi_sum(psf, roi) * g.dx_m ** 2)
    assert level["energy_extended_au"] == pytest.approx(
        float(extended.sum(dtype=torch.float64)) * g.dx_m ** 2)
    assert 0.0 <= level["outside_roi_fraction"] < 1.0
    # Centroid is reported about the array centre, so a centred PSF reads zero.
    assert level["centroid_y_px"] == pytest.approx(0.0, abs=0.05)
    assert level["centroid_x_px"] == pytest.approx(0.0, abs=0.05)
    patch = np.asarray(level["pixel_patch"], dtype=float)
    crop = psf[roi["y0"]:roi["y1"], roi["x0"]:roi["x1"]].numpy()
    assert patch.shape == (bins["bins"], bins["bins"])
    # pixel_integrate returns integrated intensity (AU * m^2), so dx^2 is expected.
    assert float(patch.sum()) == pytest.approx(float(crop.sum()) * g.dx_m ** 2)


def test_measure_level_roi_is_independent_of_the_window_size():
    g = c.Geometry(33, 400e-9, 450e-9, 2400e-6)
    y, x = torch.meshgrid(torch.arange(33), torch.arange(33), indexing="ij")
    psf = torch.exp(-((y - 16.0) ** 2 + (x - 18.0) ** 2) / 6.0)
    energies = []
    for padding in (16, 32):
        extended = torch.nn.functional.pad(psf, (padding,) * 4)
        level = w.measure_level(psf, extended, g, 9)
        energies.append(level["energy_roi_au"])
        assert level["roi_alignment"]["aligned"] is True
        assert level["roi_window"]["y0"] == c.roi_window(33, 9)["y0"]
        # The fixed ROI clips the displaced tail, so the centroid reads slightly short.
        assert level["centroid_x_px"] == pytest.approx(2.0, abs=0.3)
    # Same physical box, same crop: the ROI value may not drift with padding.
    assert energies[0] == pytest.approx(energies[1], rel=1e-12)


# ---------------------------------------------------------------------------
# convergence verdict
# ---------------------------------------------------------------------------

def level(fft_size, *, energy=1000.0, centroid=16.0, patch=(1.0, 0.0, 0.0),
          outside=0.01, edge=0.001, status="measured", padding=None):
    return dict(fft_size=fft_size, padding_px=padding, status=status,
                energy_roi_au=energy, centroid_y_px=centroid, centroid_x_px=centroid,
                pixel_patch=list(patch), outside_roi_fraction=outside,
                edge_fraction_extended=edge)


def test_stable_levels_converge_only_where_two_larger_windows_agree():
    rows = [level(size) for size in (2400, 3200, 3600, 4000)]
    result = w.evaluate_group(rows, w.Tolerances())
    assert result["status"] == "converged"
    assert result["first_supported_fft_size"] == 2400
    assert result["levels"][0]["status"] == "supported"
    assert result["levels"][1]["status"] == "supported"
    assert result["levels"][2]["status"] == "insufficient_support"
    assert result["levels"][3]["status"] == "insufficient_support"


def test_energy_drift_across_windows_is_never_a_pass():
    rows = [level(size, energy=e)
            for size, e in ((2400, 1000.0), (3200, 1200.0), (3600, 1400.0))]
    result = w.evaluate_group(rows, w.Tolerances())
    assert result["status"] == "not_converged"
    first = result["levels"][0]
    assert first["status"] == "exceeds_tolerance"
    assert any(">" in reason for reason in first["reasons"])
    # Relative difference is normalised by the larger value: 200/1200.
    assert first["deltas"]["3200"]["energy_roi_rel"] == pytest.approx(200 / 1200)


def test_residual_rebound_blocks_the_pass_even_below_tolerance():
    # Steps agree well at first and then drift again: no tolerance is exceeded
    # yet the residual grows faster than the rebound ratio, so no pass.
    rows = [level(size, energy=e) for size, e in
            ((2400, 1000.0), (3200, 1003.0), (3600, 1007.0))]
    result = w.evaluate_group(rows, w.Tolerances())
    assert result["status"] == "not_converged"
    first = result["levels"][0]
    assert first["status"] == "rebound"
    assert first["deltas"]["3600"]["energy_roi_rel"] == pytest.approx(7 / 1007)
    assert any("rebounds" in reason for reason in first["reasons"])


def test_missing_larger_window_blocks_the_pass():
    rows = [level(2400), level(3200, status="failed"), level(3600), level(4000)]
    result = w.evaluate_group(rows, w.Tolerances())
    assert result["status"] == "not_converged"
    assert result["levels"][0]["status"] == "missing_level"
    assert result["levels"][1]["status"] == "failed"


def test_undefined_metric_is_reported_as_undefined():
    rows = [level(2400), level(3200, centroid=None), level(3600), level(4000)]
    result = w.evaluate_group(rows, w.Tolerances())
    assert result["levels"][0]["status"] == "undefined_metric"


def test_without_tolerances_only_metrics_are_reported():
    rows = [level(size, energy=e) for size, e in
            ((2400, 1000.0), (3200, 1200.0), (3600, 900.0))]
    result = w.evaluate_group(rows, None)
    assert result["status"] == "metrics_only"
    assert all(entry["status"] == "metrics_only" for entry in result["levels"])
    # The deltas are still computed and stored: no threshold, no verdict.
    assert result["levels"][0]["deltas"]["3200"]["energy_roi_rel"] == pytest.approx(200 / 1200)


def test_edge_cap_applies_to_levels_but_not_to_the_differential():
    hot = [level(size, edge=0.5) for size in (2400, 3200, 3600)]
    capped = w.evaluate_group(hot, w.Tolerances())
    assert capped["levels"][0]["status"] == "exceeds_tolerance"
    assert capped["status"] == "not_converged"

    differential = [dict(fft_size=size, status="measured", edge_fraction_extended=None,
                         differential_energy_au=1.0, differential_patch=[1.0, 0.0])
                    for size in (2400, 3200, 3600)]
    passed = w.evaluate_group(differential, w.Tolerances(),
                              metrics=w.DIFFERENTIAL_METRICS, differential=True)
    assert passed["levels"][0]["status"] == "supported"
    assert passed["status"] == "converged"


def test_evaluate_aggregates_four_families_plus_the_differential():
    rows, differentials = [], []
    for artifact, sign in (("kernel_000_pos", "pos"), ("kernel_000_neg", "neg")):
        for point in ("center", "corner"):
            for size in (2400, 3200, 3600, 4000):
                rows.append(dict(level(size), artifact=artifact, sign=sign, point=point))
            differentials.append(dict(level(2400), point=point, differential_energy_au=2.0,
                                      differential_patch=[1.0, 0.0]))
            differentials.append(dict(level(3200), point=point, differential_energy_au=2.0,
                                      differential_patch=[1.0, 0.0]))
            differentials.append(dict(level(3600), point=point, differential_energy_au=2.0,
                                      differential_patch=[1.0, 0.0]))
    result = w.evaluate(rows, differentials, w.Tolerances())
    assert len(result["groups"]) == 4
    assert set(result["differential_groups"]) == {"center", "corner"}
    assert result["status"] == "CONVERGED"
    assert result["converged_from_fft_size"] == 2400

    # A family whose two largest windows were never measured cannot be converged.
    for row in rows:
        if row["artifact"] == "kernel_000_neg" and row["point"] == "corner" and row["fft_size"] >= 3600:
            row["status"] = "failed"
    assert w.evaluate(rows, differentials, w.Tolerances())["status"] == "NOT_CONVERGED"
    assert w.evaluate(rows, differentials, None)["status"] == "METRICS_ONLY"


# ---------------------------------------------------------------------------
# dx re-check and the resource guard
# ---------------------------------------------------------------------------

def source_config(tmp_path, **updates):
    data = dict(n=16, dx_nm=400, distance_um=2400, wavelength_nm=450,
                camera_pixel_um=3.45, patch_camera=1, patch_meta=8, grayscale=True)
    data.update(updates)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return c.load_source_config(path)


def test_dx_review_passes_only_for_a_consistent_detector(tmp_path):
    config = source_config(tmp_path)
    g = c.Geometry.from_config(config)
    plan = w.window_plan(g.n, (16, 32), g.dx_m)
    review = w.dx_review(config, g, roi_size=config.patch_meta,
                         camera_pixel_m=3.45e-6, plan=plan,
                         points=w.points_for_corner(0.5, 0.5))
    assert review["status"] == "PASS"
    assert all(check["ok"] for check in review["checks"])
    names = {check["name"] for check in review["checks"]}
    assert {"dx_nm_matches_core_constant", "dx_nm_matches_source_config",
            "patch_meta_from_camera_pixel", "roi_physical_width_um",
            "camera_pixel_binning", "fft_mapping_16"} <= names
    assert review["dx_nm"] == pytest.approx(c.DX_NM)
    assert review["roi_width_um"] == pytest.approx(8 * 400e-9 * 1e6)

    # A ROI or camera pitch that contradicts the fitted detector patch fails.
    broken = w.dx_review(config, g, roi_size=4, camera_pixel_m=3.45e-6, plan=plan,
                         points=w.points_for_corner(0.5, 0.5))
    assert broken["status"] == "FAIL"
    failed = {check["name"] for check in broken["checks"] if not check["ok"]}
    assert "roi_physical_width_um" in failed
    assert "patch_meta_from_camera_pixel" not in failed

    wrong_pitch = w.dx_review(config, g, roi_size=config.patch_meta,
                              camera_pixel_m=2.0e-6, plan=plan,
                              points=w.points_for_corner(0.5, 0.5))
    assert wrong_pitch["status"] == "FAIL"


def test_guard_accepts_a_measured_bytes_per_pixel_budget():
    generous = dict(cpu_percent=10.0, ram_percent=20.0,
                    ram_available_bytes=64 * 2 ** 30, gpus=[])
    cli.guard(generous, 800, [800], "cpu", bytes_per_px=64)
    assert generous["allocation_budget_bytes"] == (800 + 1600) ** 2 * 64
    assert generous["allocation_bytes_per_px"] == 64
    cli.guard(generous, 800, [800], "cpu")
    assert generous["allocation_budget_bytes"] == (800 + 1600) ** 2 * 512

    tight = dict(cpu_percent=10.0, ram_percent=20.0,
                 ram_available_bytes=8 * 2 ** 30, gpus=[])
    with pytest.raises(c.ValidationError):
        cli.guard(dict(tight), 800, [1600], "cpu")
    cli.guard(dict(tight), 800, [1600], "cpu", bytes_per_px=64)
