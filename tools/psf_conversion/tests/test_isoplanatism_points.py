"""Unit tests for the 2D point isoplanatism entry and its convergence gate.

The point scan is driven through a monkeypatched propagation engine, so no
large window and no optimizer is ever started; the numbers only have to prove
that bookkeeping, pairing and the gate behave as documented.
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


def config_file(tmp_path, **updates):
    data = dict(n=64, dx_nm=400, distance_um=80, wavelength_nm=450, camera_pixel_um=3.45,
                patch_camera=1, patch_meta=8, grayscale=True)
    data.update(updates)
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    return path


def artifact_file(directory, stem, n=64, gain=1.0):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{stem}.npz'
    width = np.full((n, n), 200.0)
    psf = np.exp(-((np.arange(n) - (n - 1) / 2) ** 2)[:, None] / 4.0) * np.ones((1, n))
    np.savez(path, width_nm=width, psf_raw=psf[None], wavelength_nm=450,
             active_gray=True, gain_gray=gain)
    return c.load_artifact(path)


def convergence_file(tmp_path, *, status='CONVERGED', fft_sizes=(64, 128, 192),
                     converged_from=64, dx_status='PASS', run_class='fft_window_convergence'):
    manifest = dict(run_class=run_class,
                    convergence=dict(status=status, converged_from_fft_size=converged_from),
                    windows=[dict(fft_size=size) for size in fft_sizes],
                    tolerances=dict(energy_roi_rel=0.01),
                    tolerances_source='explicit',
                    dx_review=dict(status=dx_status))
    path = tmp_path / 'window_convergence_manifest.json'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    return path


# ---------------------------------------------------------------------------
# point grid and the object-plane -> input mapping
# ---------------------------------------------------------------------------

def test_formal_points_cover_centre_edges_and_four_corners():
    points = cli.formal_points(0.5)
    assert [p['name'] for p in points] == list(cli.POINT_NAMES)
    assert len(points) == 9
    centre = points[0]
    assert (centre['theta_x_deg'], centre['theta_y_deg']) == (0.0, 0.0)
    by_name = {p['name']: p for p in points}
    assert by_name['edge_x_pos']['theta_x_deg'] == 0.5
    assert by_name['edge_y_neg']['theta_y_deg'] == -0.5
    assert (by_name['corner_pn']['theta_x_deg'], by_name['corner_pn']['theta_y_deg']) == (0.5, -0.5)
    assert (by_name['corner_nn']['theta_x_deg'], by_name['corner_nn']['theta_y_deg']) == (-0.5, -0.5)
    assert set(cli.CORNER_NAMES) <= set(by_name)
    for bad in (0.0, 90.0, float('nan')):
        with pytest.raises(c.ValidationError):
            cli.formal_points(bad)


def test_object_to_input_records_the_forecast_shift():
    g = c.Geometry(64, 400e-9, 450e-9, 2400e-6)
    point = {'name': 'corner_pp', 'theta_x_deg': 0.5, 'theta_y_deg': -0.25}
    mapping = cli.object_to_input(point, g, 128)
    dy, dx = c.lateral_shift_theory(g, 0.5, -0.25)
    assert mapping['detector_shift_theory_px'] == {'dy': dy, 'dx': dx}
    assert mapping['detector_shift_theory_um']['dx'] == pytest.approx(dx * g.dx_m * 1e6)
    assert mapping['sin_theta_x'] == pytest.approx(np.sin(np.radians(0.5)))
    assert mapping['sin_theta_over_max'] == pytest.approx(
        max(abs(np.sin(np.radians(0.5))), abs(np.sin(np.radians(0.25)))) / g.max_sin_theta)
    assert '128' in mapping['input_grid']
    assert mapping['detector_shift_rule'].startswith('z*tan(theta)/dx')
    assert 'no lateral object-plane translation' in mapping['limitation']


# ---------------------------------------------------------------------------
# convergence manifest and identity gate
# ---------------------------------------------------------------------------

def test_load_convergence_rejects_foreign_manifests(tmp_path):
    with pytest.raises(c.ValidationError):
        cli.load_convergence(convergence_file(tmp_path, run_class='something_else'))
    loaded = cli.load_convergence(convergence_file(tmp_path))
    assert loaded['run_class'] == 'fft_window_convergence'
    assert loaded['status'] == 'CONVERGED'
    assert loaded['fft_sizes'] == [64, 128, 192]
    assert loaded['converged_from_fft_size'] == 64
    assert loaded['dx_review_status'] == 'PASS'
    assert len(loaded['sha256']) == 64


def test_identity_checks_gate_a_consistent_kernel_pair(tmp_path):
    config = c.load_source_config(config_file(tmp_path))
    g = c.Geometry.from_config(config)
    directory = tmp_path / 'fit_all'
    pos = artifact_file(directory, 'kernel_000_pos', gain=1.0)
    neg = artifact_file(directory, 'kernel_000_neg', gain=1.1)
    loaded = cli.load_convergence(convergence_file(tmp_path))

    checks = cli.identity_checks(config, [pos, neg], g, config.patch_meta, [0, 32], loaded)
    assert checks['status'] == 'PASS'
    assert checks['failed'] == []
    assert checks['formal_conclusion_allowed'] is True

    without = cli.identity_checks(config, [pos, neg], g, config.patch_meta, [0, 32])
    assert without['status'] == 'FAIL'
    assert 'convergence_manifest_supplied' in without['failed']


def test_identity_checks_fail_on_a_split_or_mismatched_pair(tmp_path):
    config = c.load_source_config(config_file(tmp_path))
    g = c.Geometry.from_config(config)
    pos = artifact_file(tmp_path / 'fit_a', 'kernel_000_pos')
    neg = artifact_file(tmp_path / 'fit_b', 'kernel_000_neg')
    checks = cli.identity_checks(config, [pos, neg], g, config.patch_meta, [0, 32],
                                 cli.load_convergence(convergence_file(tmp_path)))
    assert checks['status'] == 'FAIL'
    assert 'same_kernel_directory' in checks['failed']

    narrow = c.load_source_config(config_file(tmp_path, n=32))
    narrow_geometry = c.Geometry.from_config(narrow)
    pair = [artifact_file(tmp_path / 'fit_all', 'kernel_000_pos'),
            artifact_file(tmp_path / 'fit_all', 'kernel_000_neg')]
    mismatch = cli.identity_checks(narrow, pair, narrow_geometry, narrow.patch_meta, [0, 16],
                                   cli.load_convergence(convergence_file(tmp_path)))
    assert mismatch['status'] == 'FAIL'
    assert 'artifact_grid_matches_config_n' in mismatch['failed']

    wrong_roi = cli.identity_checks(config, pair, g, config.patch_meta + 1, [0, 32],
                                    cli.load_convergence(convergence_file(tmp_path)))
    assert 'detector_roi_matches_patch_meta' in wrong_roi['failed']

    not_converged = cli.identity_checks(
        config, pair, g, config.patch_meta, [0, 32],
        cli.load_convergence(convergence_file(tmp_path, status='NOT_CONVERGED',
                                              converged_from=None)))
    assert 'convergence_status_converged' in not_converged['failed']


def test_formal_conclusion_never_claims_a_physical_verdict(tmp_path):
    config = c.load_source_config(config_file(tmp_path))
    g = c.Geometry.from_config(config)
    directory = tmp_path / 'fit_all'
    pair = [artifact_file(directory, 'kernel_000_pos'), artifact_file(directory, 'kernel_000_neg')]
    loaded = cli.load_convergence(convergence_file(tmp_path))

    blocked = cli.formal_conclusion(cli.identity_checks(config, pair, g, config.patch_meta, [0, 32]),
                                    None)
    assert blocked['verdict'] == 'BLOCKED'
    assert blocked['acceptance'] == 'NOT_SPECIFIED'

    passed = cli.formal_conclusion(
        cli.identity_checks(config, pair, g, config.patch_meta, [0, 32], loaded), loaded)
    assert passed['verdict'] == 'METRICS_ONLY'
    assert passed['acceptance'] == 'NOT_SPECIFIED'
    assert 'no isoplanatism tolerance is defined' in passed['reasons'][-1]
    assert 'measured points only' in passed['scope']


# ---------------------------------------------------------------------------
# the point scan itself
# ---------------------------------------------------------------------------

def controlled_points(monkeypatch, tmp_path, label, paddings, points, gain=1.0):
    g = c.Geometry(16, 400e-9, 450e-9, 8e-6)
    original = c.PropagationEngine

    class ControlledEngine(original):
        def extended(self, field, padding):
            y, x = torch.meshgrid(torch.arange(16), torch.arange(16), indexing='ij')
            psf = torch.exp(-((y - 7.5) ** 2 + (x - 7.5) ** 2) / (7. if padding == 16 else 3.))
            psf = psf / psf.sum()
            crop = torch.sqrt(psf).to(torch.complex64)
            full = torch.nn.functional.pad(crop, (padding,) * 4)
            self.transfer_function(16 + 2 * padding)
            return full, crop, self._layer(0, 16 + 2 * padding)

        def direct(self, field, padding):
            return self.extended(field, padding)[1]

    monkeypatch.setattr(c, 'PropagationEngine', ControlledEngine)
    out = tmp_path / label
    out.mkdir()
    rows = cli.scan_points(cli.ideal_field(g), g, paddings, points, 9, 'cpu', out, label,
                           3.45e-6, baseline_padding=paddings[-1])
    return out, rows


def test_scan_points_pairs_every_point_with_the_centre(monkeypatch, tmp_path):
    points = cli.formal_points(0.5)[:3]
    out, rows = controlled_points(monkeypatch, tmp_path, 'kernel_000_pos', [8, 16], points)
    assert len(rows) == len(points) * 2
    assert {row['point'] for row in rows} == {p['name'] for p in points}
    assert all(row['axis'] == 'point' for row in rows)
    assert all(row['isoplanatism_acceptance'] == 'NOT_SPECIFIED' for row in rows)
    assert len({row['sample_id'] for row in rows}) == len(rows)
    for row in rows:
        assert (out / row['npz_file']).is_file()
        assert (out / row['png_file']).is_file()
        assert row['npz_file'].endswith(f"_{row['point']}_p{row['padding']}.npz")
        assert row['padding_stability']['status'] in ('consistent', 'inconsistent', 'indeterminate')
        assert row['physical_interpretation'].startswith('indeterminate')
        with np.load(out / row['npz_file']) as saved:
            assert saved['psf'].shape == (16, 16)
            assert saved['point'].item() == row['point']
    centre = next(row for row in rows if row['point'] == 'center')
    assert centre['padding_stability']['scope'].startswith('centre control')

    corner_rows = []
    edge_points = [p for p in cli.formal_points(0.5) if p['name'].startswith('corner')][:2]
    out2, corner_rows = controlled_points(monkeypatch, tmp_path, 'kernel_000_neg', [8, 16],
                                          edge_points)
    assert all(row['padding_stability']['scope'] == 'corner point re-checked with larger window'
               for row in corner_rows if row['point'].startswith('corner'))


def test_scan_points_requires_the_baseline_window(monkeypatch, tmp_path):
    g = c.Geometry(16, 400e-9, 450e-9, 8e-6)
    out = tmp_path / 'missing'
    out.mkdir()
    with pytest.raises(c.ValidationError):
        cli.scan_points(cli.ideal_field(g), g, [8, 16], cli.formal_points(0.5)[:1], 9, 'cpu',
                        out, 'branch', 3.45e-6, baseline_padding=99)


def test_pos_neg_rows_aggregate_both_branches_at_fixed_gain(monkeypatch, tmp_path):
    points = cli.formal_points(0.5)[:2]
    _, pos = controlled_points(monkeypatch, tmp_path, 'kernel_000_pos', [8, 16], points)
    _, neg = controlled_points(monkeypatch, tmp_path, 'kernel_000_neg', [8, 16], points)
    gains = {'kernel_000_pos': 1.0, 'kernel_000_neg': 1.2}
    rows = cli.pos_neg_rows({'kernel_000_pos': pos, 'kernel_000_neg': neg}, gains)
    assert len(rows) == len(points) * 2
    assert all(row['status'] == 'metrics_only' for row in rows)
    assert all(row['threshold'] == 'NOT_SPECIFIED' for row in rows)
    assert all('no differential isoplanatism tolerance' in row['reason'] for row in rows)
    first = rows[0]
    key = (first['point'], first['padding'])
    pos_row = next(r for r in pos if (r['point'], r['padding']) == key)
    neg_row = next(r for r in neg if (r['point'], r['padding']) == key)
    assert first['pos_artifact'] == 'kernel_000_neg'
    assert first['neg_artifact'] == 'kernel_000_pos'
    assert first['energy_au'] == pytest.approx(
        1.2 * neg_row['energy_roi_fixed_au'] - 1.0 * pos_row['energy_roi_fixed_au'])
    # Both branches share the synthetic PSF, so the differential shape vanishes.
    assert first['shape'] == pytest.approx(0.0, abs=1e-9)
    assert first['energy_definition'].startswith('gain_pos*ROI_integrated_pos')

    with pytest.raises(c.ValidationError):
        cli.pos_neg_rows({'only': pos}, gains)


def test_point_mode_rejects_the_engineering_fixture(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(['--points', '--self-check', '--out', str(tmp_path / 'points')])


def test_identity_checks_use_the_explicit_padding_baseline(tmp_path):
    config = c.load_source_config(config_file(tmp_path))
    g = c.Geometry.from_config(config)
    directory = tmp_path / 'fit'
    artifacts = [artifact_file(directory, 'kernel_000_pos'),
                 artifact_file(directory, 'kernel_000_neg')]
    convergence = cli.load_convergence(
        convergence_file(tmp_path, fft_sizes=(192, 320, 448), converged_from=320))

    # The source config padding (64 -> FFT 192) is below the converged FFT, so the
    # default reference must be refused and only the explicit 128 may pass.
    default = cli.identity_checks(config, artifacts, g, config.patch_meta, [64, 128],
                                 convergence)
    assert 'baseline_window_at_or_above_convergence' in default['failed']
    raised = cli.identity_checks(config, artifacts, g, config.patch_meta, [64, 128],
                                 convergence, baseline_padding=128)
    assert raised['status'] == 'PASS'
    assert raised['formal_conclusion_allowed'] is True


def test_point_mode_needs_a_padding_baseline_at_the_converged_window(tmp_path):
    config = config_file(tmp_path)
    directory = tmp_path / 'fit'
    artifacts = [artifact_file(directory, 'kernel_000_pos'),
                 artifact_file(directory, 'kernel_000_neg')]
    with pytest.raises(SystemExit):
        # Source padding 64 is absent, and no explicit baseline was given.
        cli.main(['--artifact', str(artifacts[0].path), '--artifact', str(artifacts[1].path),
                  '--config', str(config), '--points', '--paddings', '128',
                  '--out', str(tmp_path / 'points')])
    with pytest.raises(SystemExit):
        cli.main(['--artifact', str(artifacts[0].path), '--artifact', str(artifacts[1].path),
                  '--config', str(config), '--points', '--paddings', '128',
                  '--baseline-padding', '64', '--out', str(tmp_path / 'points2')])


# ---------------------------------------------------------------------------
# baseline padding follows the converged window
# ---------------------------------------------------------------------------

def _geometry(n=64):
    return c.Geometry(n, 400e-9, 450e-9, 80e-6)


def _convergence(**overrides):
    data = dict(status='CONVERGED', converged_from_fft_size=128, fft_sizes=[64, 128, 192])
    data.update(overrides)
    return data


def test_baseline_padding_is_derived_from_the_converged_window():
    geometry = _geometry()
    padding, source = cli.resolve_baseline_padding(geometry, _convergence(), [32, 64])
    assert (padding, 'FFT 128' in source) == (32, True)
    # A scan that converged later moves the baseline with it; nothing is hard-coded.
    padding, source = cli.resolve_baseline_padding(
        geometry, _convergence(converged_from_fft_size=192), [64, 96])
    assert (padding, 'FFT 192' in source) == (64, True)
    padding, source = cli.resolve_baseline_padding(geometry, None, [64, 128], config_padding=64)
    assert (padding, source) == (64, 'source config padding')


def test_baseline_padding_stops_when_the_manifest_cannot_support_it():
    geometry = _geometry()
    with pytest.raises(c.ValidationError):  # not converged
        cli.resolve_baseline_padding(geometry, _convergence(status='NOT_CONVERGED'), [32, 64])
    with pytest.raises(c.ValidationError):  # converged FFT absent from the window list
        cli.resolve_baseline_padding(geometry, _convergence(fft_sizes=[64, 96]), [32, 64])
    with pytest.raises(c.ValidationError):  # odd delta cannot be two equal paddings
        cli.resolve_baseline_padding(geometry, _convergence(converged_from_fft_size=127), [32, 64])
    with pytest.raises(c.ValidationError):  # FFT below the device grid
        cli.resolve_baseline_padding(geometry, _convergence(converged_from_fft_size=32), [32, 64])
    with pytest.raises(c.ValidationError):  # derived padding missing from --paddings
        cli.resolve_baseline_padding(geometry, _convergence(), [64, 96])
    with pytest.raises(c.ValidationError):  # explicit baseline contradicts the manifest
        cli.resolve_baseline_padding(geometry, _convergence(), [32, 64], explicit=64)
    assert cli.resolve_baseline_padding(geometry, _convergence(), [32, 64], explicit=32)[0] == 32
    with pytest.raises(c.ValidationError):  # no manifest and no source padding
        cli.resolve_baseline_padding(geometry, None, [32, 64])
    with pytest.raises(c.ValidationError):  # source padding not requested
        cli.resolve_baseline_padding(geometry, None, [96], config_padding=64)


def test_baseline_padding_reads_a_synthetic_convergence_manifest(tmp_path):
    geometry = _geometry()
    path = convergence_file(tmp_path, fft_sizes=(64, 128, 192), converged_from=192)
    padding, source = cli.resolve_baseline_padding(geometry, cli.load_convergence(path), [64, 96])
    assert (padding, 'FFT 192' in source) == (64, True)
    blocked = convergence_file(tmp_path, status='NOT_CONVERGED')
    assert cli.load_convergence(blocked)['status'] == 'NOT_CONVERGED'
    with pytest.raises(c.ValidationError):
        cli.resolve_baseline_padding(geometry, cli.load_convergence(blocked), [64, 96])


def test_point_mode_stops_on_an_unconverged_manifest(tmp_path):
    config = config_file(tmp_path)
    directory = tmp_path / 'fit'
    artifacts = [artifact_file(directory, 'kernel_000_pos'),
                 artifact_file(directory, 'kernel_000_neg')]
    blocked = convergence_file(tmp_path, status='NOT_CONVERGED')
    with pytest.raises(SystemExit):
        cli.main(['--artifact', str(artifacts[0].path), '--artifact', str(artifacts[1].path),
                  '--config', str(config), '--points', '--paddings', '32', '64',
                  '--convergence', str(blocked), '--out', str(tmp_path / 'points')])

