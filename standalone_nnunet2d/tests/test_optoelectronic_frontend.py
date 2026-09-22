from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import standalone_nnunet2d.models.optoelectronic_frontend as frontend
from standalone_nnunet2d.models.optoelectronic_frontend import (
    ActiveRoute,
    CenteredPCAResult,
    Exposure,
    ExposureLedger,
    OptoelectronicConfig,
    PCAConv2d,
    PhaseOnlyPSF,
    PhasePSFConfig,
    SupportResponseError,
    build_active_routes,
    build_exposure_ledger,
    centered_pca_decompose,
    compute_calibrated_gain,
    convolve_psf,
    crop_centered,
    detector_response,
    detector_sampling_metadata,
    fit_psf_shape,
    ideal_signed_conv2d,
    phase_to_full_psf,
    physical_dual_rail_forward,
    signed_kernel_lobes,
    signed_input_rails,
    support_sweep,
)


def _small_phase_config(**overrides: object) -> PhasePSFConfig:
    values: dict[str, object] = {
        "wavelength_m": 520e-9,
        "phase_grid_size": 16,
        "phase_pitch_m": 8e-6,
        "focal_length_m": 20e-3,
        "aperture_diameter_m": 96e-6,
        "magnification": 1.0,
    }
    values.update(overrides)
    return PhasePSFConfig(**values)


def _same_conv(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    k = kernel.shape[-1]
    left = (k - 1) // 2
    right = k - 1 - left
    padded = F.pad(x, (left, right, left, right))
    return F.conv2d(padded, kernel)


def _single_positive_route_ledger() -> tuple[ActiveRoute, ExposureLedger]:
    route = build_active_routes({"g": torch.tensor([[[[1.0, 3.0]]]])})[0]
    return route, build_exposure_ledger((route,))


def test_five_future_kernel_shapes_have_expected_pca_dimensions_and_rank_bounds() -> None:
    specs = ((64, 7), (32, 2), (16, 4), (8, 8), (8, 16))
    for out_channels, kernel_size in specs:
        weight = torch.randn(out_channels, 1, kernel_size, kernel_size)
        result = centered_pca_decompose(weight, rank=0)
        upper = min(out_channels - 1, kernel_size * kernel_size)
        assert result.structural_rank_upper_bound == upper
        assert result.resolved_rank == 0
        assert result.mean_kernel.shape == (1, 1, kernel_size, kernel_size)
        assert result.basis.shape == (0, 1, kernel_size, kernel_size)
        assert result.coefficients.shape == (out_channels, 0)
        assert result.component_bank.shape == (1, 1, kernel_size, kernel_size)


def test_rank_zero_is_mean_only_and_full_centered_rank_reconstructs() -> None:
    weight = torch.tensor(
        [
            [[[1.0, 2.0], [3.0, 4.0]]],
            [[[2.0, 0.0], [4.0, 1.0]]],
            [[[-1.0, 3.0], [2.0, 5.0]]],
        ]
    )
    mean_only = centered_pca_decompose(weight, rank=0)
    assert torch.allclose(mean_only.reconstruct_weight(), weight.mean(dim=0, keepdim=True).expand_as(weight))
    full = centered_pca_decompose(weight, rank=weight.shape[0] - 1)
    assert full.resolved_rank == full.structural_rank_upper_bound
    assert full.report.absolute_frobenius_error < 1e-5
    assert full.report.relative_frobenius_error < 1e-5
    assert full.report.maximum_absolute_error < 1e-5


def test_variance_threshold_is_resolved_independently_and_reported() -> None:
    weight = torch.zeros(5, 1, 2, 2)
    weight[:, 0, 0, 0] = torch.arange(5, dtype=torch.float32)
    weight[:, 0, 0, 1] = 0.01 * torch.arange(5, dtype=torch.float32)
    result = centered_pca_decompose(weight, variance_threshold=0.99)
    assert 0 <= result.resolved_rank <= result.structural_rank_upper_bound
    assert result.report.explained_variance_ratio >= 0.99
    assert result.report.resolved_rank == result.resolved_rank
    with pytest.raises(ValueError, match="exactly one|rank|variance"):
        centered_pca_decompose(weight, rank=1, variance_threshold=0.9)


def test_zero_centered_variance_is_finite_and_defined() -> None:
    weight = torch.full((4, 1, 3, 3), 2.5)
    result = centered_pca_decompose(weight, rank=0)
    assert result.report.explained_variance_ratio == pytest.approx(1.0)
    assert result.report.absolute_frobenius_error == pytest.approx(0.0)
    assert torch.isfinite(result.component_bank).all()
    assert torch.isfinite(result.coefficients).all()


def test_pca_reconstruction_preserves_mean_mixing_and_bias_contract() -> None:
    torch.manual_seed(11)
    conv = nn.Conv2d(1, 5, kernel_size=3, stride=2, padding=1, bias=True)
    result = centered_pca_decompose(
        conv.weight.detach(), rank=conv.out_channels - 1, bias=conv.bias.detach(), stride=2, padding=1
    )
    assert isinstance(result, CenteredPCAResult)
    assert torch.allclose(result.original_bias, conv.bias)
    module = PCAConv2d(result, trainable_mixing=False, trainable_bias=False)
    assert not isinstance(module.mixing, nn.Parameter)
    assert not isinstance(module.bias, nn.Parameter)
    x = torch.randn(2, 1, 9, 11)
    assert torch.allclose(module(x), conv(x), atol=1e-5, rtol=1e-5)
    assert torch.allclose(module.mixing[:, 0], torch.ones(conv.out_channels))


def test_signed_input_and_kernel_dual_rail_identity_matches_conv2d() -> None:
    torch.manual_seed(12)
    x = torch.randn(2, 1, 9, 10)
    kernel = torch.randn(3, 1, 3, 2)
    bias = torch.randn(3)
    expected = F.conv2d(x, kernel, bias=bias, stride=2, padding=(1, 0))
    actual = ideal_signed_conv2d(x, kernel, bias=bias, stride=2, padding=(1, 0))
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    x_pos, x_neg = signed_input_rails(x)
    assert torch.equal(x, x_pos - x_neg)
    assert torch.equal(x_pos, x.clamp_min(0))
    assert torch.equal(x_neg, (-x).clamp_min(0))


def test_ideal_signed_conv2d_uses_four_response_rails_and_adds_bias_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(120)
    x = torch.randn(1, 1, 8, 9)
    kernel = torch.randn(2, 1, 3, 2)
    bias = torch.tensor([0.25, -0.75])
    reference_conv2d = F.conv2d
    expected = reference_conv2d(x, kernel, bias=bias, stride=2, padding=(1, 0))
    calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = []

    def recording_conv2d(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        calls.append((input, weight, bias))
        return reference_conv2d(input, weight, bias=bias, **kwargs)

    monkeypatch.setattr(frontend.F, "conv2d", recording_conv2d)
    actual = frontend.ideal_signed_conv2d(x, kernel, bias=bias, stride=2, padding=(1, 0))

    assert len(calls) == 4
    assert all(call[2] is None for call in calls)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_zero_kernel_lobes_have_exact_zero_contribution_without_division() -> None:
    positive = torch.zeros(1, 1, 3, 3)
    negative = torch.tensor([[[[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]])
    lobes = signed_kernel_lobes(negative)
    assert lobes.alpha_pos == pytest.approx(0.0)
    assert torch.count_nonzero(lobes.p_pos) == 0
    assert lobes.alpha_neg == pytest.approx(1.0)
    assert torch.equal(lobes.p_neg, -negative)
    zero_lobes = signed_kernel_lobes(positive)
    assert zero_lobes.alpha_pos == pytest.approx(0.0)
    assert zero_lobes.alpha_neg == pytest.approx(0.0)
    assert torch.count_nonzero(zero_lobes.p_pos) == 0
    assert torch.count_nonzero(zero_lobes.p_neg) == 0


def test_asymmetric_psf_direction_is_flipped_once_for_cross_correlation() -> None:
    x = torch.zeros(1, 1, 7, 7)
    x[:, :, 3, 3] = 1.0
    psf = torch.tensor([[[[1.0, 2.0], [4.0, 8.0]]]])
    actual = convolve_psf(x, psf, padding=0)
    expected = F.conv2d(x, torch.flip(psf, (-2, -1)), padding=0)
    assert torch.equal(actual, expected)
    assert torch.count_nonzero(actual) == 4


def test_digital_kernel_lobe_to_physical_psf_flips_once_and_is_not_symmetric_by_accident() -> None:
    digital_lobe = torch.tensor([[[[1.0, 2.0, 3.0], [5.0, 7.0, 11.0]]]])
    physical_psf = frontend.physical_psf_from_digital_lobe(digital_lobe)
    assert torch.equal(physical_psf, torch.flip(digital_lobe, dims=(-2, -1)))
    x = torch.zeros(1, 1, 5, 6)
    x[:, :, 2, 3] = 1.0
    actual = convolve_psf(x, physical_psf)
    expected = F.conv2d(x, digital_lobe)
    assert torch.equal(actual, expected)
    assert not torch.equal(actual, F.conv2d(x, physical_psf))


@pytest.mark.parametrize("kernel_size", [2, 4, 7, 8, 16])
def test_ideal_kernel_supports_future_stride_padding_and_output_shape(kernel_size: int) -> None:
    x = torch.randn(2, 1, 31, 29)
    kernel = torch.randn(4, 1, kernel_size, kernel_size)
    stride = 2
    padding = kernel_size // 2
    actual = ideal_signed_conv2d(x, kernel, stride=stride, padding=padding)
    expected = F.conv2d(x, kernel, stride=stride, padding=padding)
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_full_phase_psf_is_finite_nonnegative_and_unit_energy() -> None:
    config = _small_phase_config()
    theta = torch.zeros(config.phase_grid_size, config.phase_grid_size)
    psf = phase_to_full_psf(theta, config=config)
    assert psf.shape == (config.phase_grid_size, config.phase_grid_size)
    assert torch.isfinite(psf).all()
    assert torch.all(psf >= 0)
    assert psf.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert config.assumption_status == "synthetic engineering assumptions"


def test_detector_pitch_and_one_to_one_sampling_metadata_are_explicit() -> None:
    config = _small_phase_config()
    metadata = detector_sampling_metadata(config)
    expected = config.wavelength_m * config.focal_length_m / (
        config.phase_grid_size * config.phase_pitch_m
    )
    assert metadata.detector_pitch_m == pytest.approx(expected)
    assert metadata.input_pixel_to_object_sample
    assert metadata.output_position_to_detector_sample
    assert metadata.detector_sample_to_fft_cell
    assert metadata.nifti_spacing_is_unrelated


def test_center_crop_has_explicit_even_and_odd_rules() -> None:
    image = torch.arange(8 * 10, dtype=torch.float32).view(8, 10)
    odd = crop_centered(image, 3)
    even = crop_centered(image, 4)
    assert torch.equal(odd, image[3:6, 4:7])
    assert torch.equal(even, image[2:6, 3:7])
    assert crop_centered(image, (2, 4)).shape == (2, 4)


@pytest.mark.parametrize("full_size,support", [(8, 3), (8, 4), (9, 3), (9, 4)])
def test_center_crop_places_fftshifted_discrete_origin_at_support_half(
    full_size: int,
    support: int,
) -> None:
    image = torch.zeros(full_size, full_size)
    image[full_size // 2, full_size // 2] = 1.0
    cropped = crop_centered(image, support)
    assert cropped[support // 2, support // 2].item() == 1.0
    assert torch.count_nonzero(cropped) == 1


def test_cropped_psf_is_raw_and_throughput_is_separate_from_shape_normalization() -> None:
    psf = torch.zeros(8, 8)
    psf[3:5, 3:5] = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    psf = psf / psf.sum()
    cropped = crop_centered(psf, 2)
    assert cropped.sum().item() == pytest.approx(1.0)
    larger = crop_centered(psf, 4)
    assert larger.sum().item() == pytest.approx(1.0)
    assert not torch.allclose(cropped, cropped / cropped.sum()) if cropped.sum() != 1 else True
    metrics = fit_psf_shape(torch.ones(2, 2), psf, support=2)
    assert metrics.throughput == pytest.approx(cropped.sum().item())
    assert metrics.predicted_shape.sum().item() == pytest.approx(1.0)


def test_support_sweep_reports_tau_leakage_and_response_error() -> None:
    psf = torch.zeros(16, 16)
    psf[8, 8] = 0.7
    psf[8, 10] = 0.3
    reports = support_sweep(psf, supports=(3, 7, 15), seed=3, mode="report_only")
    assert [report.support for report in reports.reports] == [3, 7, 15]
    assert reports.reports[0].tau < reports.reports[-1].tau
    assert reports.reports[-1].leakage == pytest.approx(0.0)
    for report in reports.reports:
        assert report.relative_response_l2 >= 0
        assert report.max_absolute_response_error >= 0
        assert report.impulse_response_error >= 0
        assert report.random_nonnegative_response_error >= 0
        assert report.random_signed_dual_rail_response_error >= 0
        assert report.internal_region_error >= 0
        assert report.boundary_region_error >= 0


def test_support_sweep_uses_full_psf_reference_and_keeps_outside_energy_error() -> None:
    full_psf = torch.zeros(9, 9)
    full_psf[4, 4] = 0.5
    full_psf[4, 0] = 0.5
    result = support_sweep(full_psf, supports=(3, 7), seed=31, mode="report_only")

    assert result.reference_support == 9
    assert result.reports[-1].relative_response_l2 > 0
    assert result.reports[-1].full_image_response_error > 0


def test_support_sweep_equal_full_support_has_zero_internal_and_boundary_error() -> None:
    full_psf = torch.arange(81, dtype=torch.float32).view(9, 9)
    full_psf = full_psf / full_psf.sum()
    result = support_sweep(
        full_psf,
        supports=(9,),
        seed=32,
        max_support_response_error=1e-8,
        mode="fail_closed",
    )
    report = result.reports[0]
    assert report.relative_response_l2 == pytest.approx(0.0, abs=1e-8)
    assert report.full_image_response_error == pytest.approx(0.0, abs=1e-8)
    assert report.internal_region_error == pytest.approx(0.0, abs=1e-8)
    assert report.boundary_region_error == pytest.approx(0.0, abs=1e-8)
    assert report.response_error == pytest.approx(0.0, abs=1e-8)


def test_even_reference_region_uses_asymmetric_same_padding_and_exact_complement() -> None:
    assert frontend._same_padding_extents((8, 8)) == (4, 3, 4, 3)
    internal = frontend._same_internal_mask((21, 21), (8, 8))

    assert internal.shape == (21, 21)
    assert internal[4, 4]
    assert internal[17, 17]
    assert not internal[18, 17]
    assert not internal[17, 18]
    boundary = ~internal
    assert torch.equal(internal | boundary, torch.ones_like(internal))
    assert not torch.any(internal & boundary)


def test_same_region_masks_handle_empty_internal_and_zero_bottom_right() -> None:
    assert frontend._same_padding_extents((1, 1)) == (0, 0, 0, 0)
    full_internal = frontend._same_internal_mask((4, 5), (1, 1))
    assert torch.all(full_internal)

    empty_internal = frontend._same_internal_mask((4, 5), (8, 8))
    assert not torch.any(empty_internal)
    assert torch.all(~empty_internal)
    assert math.isfinite(frontend._relative_l2(torch.empty(0), torch.empty(0)))


def test_even_full_support_has_zero_error_and_does_not_fail_closed() -> None:
    full_psf = torch.arange(1, 65, dtype=torch.float32).view(8, 8)
    full_psf = full_psf / full_psf.sum()
    result = support_sweep(
        full_psf,
        supports=(8,),
        seed=33,
        max_support_response_error=1e-8,
        mode="fail_closed",
    )
    report = result.reports[0]
    assert report.internal_region_error == pytest.approx(0.0, abs=1e-8)
    assert report.boundary_region_error == pytest.approx(0.0, abs=1e-8)
    assert report.response_error == pytest.approx(0.0, abs=1e-8)


def test_default_grid_even_reference_extent_is_a_lightweight_mask_check() -> None:
    top, bottom, left, right = frontend._same_padding_extents((256, 256))
    assert (top, bottom, left, right) == (128, 127, 128, 127)
    internal = frontend._same_internal_mask((517, 517), (256, 256))
    assert internal[128, 128]
    assert internal[389, 389]
    assert not internal[390, 389]
    assert not internal[389, 390]


@pytest.mark.parametrize(
    "supports",
    [(7, 15, 31), (2, 4, 8), (4, 8, 16), (8, 16, 32), (16, 32, 64)],
)
def test_support_sweep_covers_all_future_support_families(supports: tuple[int, int, int]) -> None:
    size = supports[-1]
    psf = torch.zeros(size, size)
    center = size // 2
    psf[center, center] = 0.8
    psf[center, min(size - 1, center + supports[0] // 2)] = 0.2
    result = support_sweep(psf, supports=supports, seed=5, mode="report_only")
    assert tuple(report.support for report in result.reports) == supports
    assert result.reports[-1].leakage == pytest.approx(0.0)


def test_exposure_ledger_excludes_zero_lobes_and_respects_global_budget() -> None:
    groups = {
        "main": torch.tensor([[[[1.0, -1.0], [0.0, 0.0]]]]),
        "patch2": torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
        "patch4": torch.tensor([[[[2.0, 0.0], [0.0, 0.0]]]]),
    }
    routes = build_active_routes(groups)
    assert len(routes) == 3
    ledger = build_exposure_ledger(groups)
    assert isinstance(ledger, ExposureLedger)
    assert ledger.policy == "shared_input_global_parallel_equal_split"
    assert ledger.active_group_count == 2
    assert ledger.active_route_count == 3
    assert ledger.exposure_count == 2
    assert all(exposure.sum_beta <= 1.0 + 1e-7 for exposure in ledger.exposures)
    assert all(exposure.unused_budget == pytest.approx(0.0) for exposure in ledger.exposures)


@pytest.mark.parametrize(
    ("policy", "expected"),
    [("global_parallel", 2), ("group_sequential", 4), ("route_sequential", 6), ("bounded_parallel", 4)],
)
def test_all_exposure_policies_have_expected_counts(policy: str, expected: int) -> None:
    groups = {"g0": torch.ones(2, 1, 2, 2), "g1": torch.ones(1, 1, 2, 2)}
    kwargs = {"route_batch_capacity": 2} if policy == "bounded_parallel" else {}
    ledger = build_exposure_ledger(groups, policy=policy, **kwargs)
    assert ledger.exposure_count == expected
    assert all(exposure.sum_beta <= 1.0 + 1e-7 for exposure in ledger.exposures)
    assert all(exposure.route_ids for exposure in ledger.exposures)
    assert all(set(exposure.beta_by_route) == set(exposure.route_ids) for exposure in ledger.exposures)
    if policy == "bounded_parallel":
        assert any(exposure.unused_budget > 0 for exposure in ledger.exposures)


def test_detector_response_uses_raw_cropped_psf_and_rho_beta_once() -> None:
    x = torch.randn(1, 1, 7, 7)
    h = torch.tensor([[[[0.25, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.25]]]])
    rho = 2.0
    beta = 0.25
    actual = detector_response(x, h, rho=rho, beta=beta, padding=1)
    expected = rho * beta * convolve_psf(x, h, padding=1)
    assert torch.allclose(actual, expected)
    assert not torch.allclose(actual, convolve_psf(x, h, padding=1))


def test_detector_response_rejects_beta_outside_unit_interval() -> None:
    x = torch.ones(1, 1, 5, 5)
    psf = torch.ones(3, 3) / 9
    with pytest.raises(ValueError, match="beta"):
        detector_response(x, psf, rho=1.0, beta=2.0, padding=1)
    with pytest.raises(ValueError, match="beta"):
        detector_response(x, psf, rho=1.0, beta=-0.1, padding=1)


def test_physical_gain_compensates_rho_beta_tau_once_without_renormalizing_h() -> None:
    full_psf = torch.zeros(5, 5)
    full_psf[2, 2] = 0.25
    full_psf[2, 3] = 0.25
    full_psf[0, 0] = 0.5
    x = torch.randn(1, 1, 7, 7)
    config = OptoelectronicConfig(
        phase=_small_phase_config(phase_grid_size=5, aperture_diameter_m=40e-6),
        min_throughput=1e-8,
        max_electronic_gain=1e6,
        min_split_fraction=1e-8,
    )
    ledger_group = {"gain": torch.tensor([[[[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]])}
    gain_route = build_active_routes(ledger_group)[0]
    gain_ledger = build_exposure_ledger(ledger_group, policy="bounded_parallel", route_batch_capacity=4)
    result = physical_dual_rail_forward(
        x,
        positive_psf=full_psf,
        negative_psf=None,
        positive_route=gain_route,
        negative_route=None,
        rho=3.0,
        ledger=gain_ledger,
        support=3,
        padding=1,
        config=config,
    )
    raw_support = crop_centered(full_psf, 3)
    expected = 2.0 * convolve_psf(x, raw_support, padding=1) / raw_support.sum()
    assert torch.allclose(result.output, expected, atol=1e-6, rtol=1e-6)


def test_calibrated_gain_formula_and_snapshot_are_detached() -> None:
    tau = torch.tensor(0.25, requires_grad=True)
    gain = compute_calibrated_gain(alpha=0.5, rho=2.0, beta=0.25, tau=tau)
    assert gain.item() == pytest.approx(4.0)
    assert not gain.requires_grad
    loss = (tau * 3.0) + gain * 0.0
    loss.backward()
    assert tau.grad.item() == pytest.approx(3.0)


def test_physical_dual_rail_forward_returns_snapshot_and_gain_is_not_parameter() -> None:
    config = OptoelectronicConfig(
        phase=_small_phase_config(),
        min_throughput=1e-8,
        max_electronic_gain=1e6,
        min_split_fraction=1e-8,
    )
    theta = torch.zeros(16, 16, requires_grad=True)
    full_psf = phase_to_full_psf(theta, config=config.phase)
    x = torch.randn(1, 1, 9, 9)
    dual_group = {"dual": torch.tensor([[[[0.75, -0.25], [0.0, 0.0]]]])}
    dual_routes = build_active_routes(dual_group)
    dual_positive_route = next(route for route in dual_routes if route.sign == "positive")
    dual_negative_route = next(route for route in dual_routes if route.sign == "negative")
    dual_ledger = build_exposure_ledger(dual_group, policy="global_parallel")
    result = physical_dual_rail_forward(
        x,
        positive_psf=full_psf,
        negative_psf=full_psf,
        positive_route=dual_positive_route,
        negative_route=dual_negative_route,
        rho=config.rho,
        ledger=dual_ledger,
        support=3,
        padding=1,
        config=config,
    )
    assert result.output.shape == x.shape
    assert config.dual_rail_mode == "sequential_dual_rail"
    assert config.gain_update_mode == "calibrated_detached"
    assert not isinstance(result.positive.gain, nn.Parameter)
    assert not isinstance(result.negative.gain, nn.Parameter)
    assert result.positive.gain.item() == pytest.approx(
        0.75 / (config.rho * 0.5 * result.positive.tau)
    )
    assert result.global_max_gain == pytest.approx(abs(result.positive.gain.item()))
    assert result.minimum_throughput == pytest.approx(result.positive.tau)
    assert result.gain_update_mode == "calibrated_detached"
    result.output.square().mean().backward()
    assert theta.grad is not None
    assert torch.isfinite(theta.grad).all()


def test_physical_forward_consumes_matching_positive_and_negative_ledger_allocations() -> None:
    group = {"g": torch.tensor([[[[1.0, -2.0], [0.0, 0.0]]]])}
    routes = build_active_routes(group)
    positive_route = next(route for route in routes if route.sign == "positive")
    negative_route = next(route for route in routes if route.sign == "negative")
    ledger = build_exposure_ledger(group, policy="global_parallel")
    psf = torch.ones(3, 3) / 9
    config = OptoelectronicConfig(min_throughput=1e-8, min_split_fraction=0.0)
    result = physical_dual_rail_forward(
        torch.randn(1, 1, 7, 7),
        positive_psf=psf,
        negative_psf=psf,
        positive_route=positive_route,
        negative_route=negative_route,
        ledger=ledger,
        support=3,
        padding=1,
        config=config,
    )

    assert result.positive.beta == pytest.approx(
        ledger.beta_for_route(positive_route.route_id, input_rail="pos")
    )
    assert result.negative.beta == pytest.approx(
        ledger.beta_for_route(negative_route.route_id, input_rail="pos")
    )


def test_physical_forward_accepts_route_sequential_ledger() -> None:
    group = {"g": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])}
    route = build_active_routes(group)[0]
    ledger = build_exposure_ledger(group, policy="route_sequential")
    result = physical_dual_rail_forward(
        torch.randn(1, 1, 5, 5),
        positive_psf=torch.ones(3, 3) / 9,
        negative_psf=None,
        positive_route=route,
        negative_route=None,
        ledger=ledger,
        support=3,
        padding=1,
        config=OptoelectronicConfig(min_throughput=1e-8, min_split_fraction=0.0),
    )
    assert result.positive.beta == pytest.approx(1.0)


def test_physical_flipped_lobes_match_original_correlation_kernel() -> None:
    kernel = torch.tensor(
        [[[[1.0, -2.0, 0.5], [3.0, 0.0, -1.0], [0.25, 2.0, -0.75]]]]
    )
    lobes = signed_kernel_lobes(kernel)
    group = {"kernel": kernel}
    routes = build_active_routes(group)
    positive_route = next(route for route in routes if route.sign == "positive")
    negative_route = next(route for route in routes if route.sign == "negative")
    ledger = build_exposure_ledger(group, policy="global_parallel")
    x = torch.randn(1, 1, 7, 8)
    result = physical_dual_rail_forward(
        x,
        positive_psf=frontend.physical_psf_from_digital_lobe(lobes.p_pos),
        negative_psf=frontend.physical_psf_from_digital_lobe(lobes.p_neg),
        positive_route=positive_route,
        negative_route=negative_route,
        ledger=ledger,
        support=3,
        padding=1,
        config=OptoelectronicConfig(min_throughput=1e-8, min_split_fraction=0.0),
    )

    assert torch.allclose(result.output, F.conv2d(x, kernel, padding=1), atol=1e-5, rtol=1e-5)


def test_physical_forward_rejects_route_not_in_ledger() -> None:
    group = {"g": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])}
    ledger = build_exposure_ledger(group)
    missing_route = ActiveRoute(
        route_id="missing:component0:positive",
        group_id="missing",
        component_index=0,
        sign="positive",
        lobe=torch.ones(1, 1, 1),
        alpha=1.0,
    )
    with pytest.raises(ValueError, match="not allocated|route"):
        physical_dual_rail_forward(
            torch.ones(1, 1, 5, 5),
            positive_psf=torch.ones(3, 3) / 9,
            negative_psf=None,
            positive_route=missing_route,
            negative_route=None,
            ledger=ledger,
            support=3,
            padding=1,
            config=OptoelectronicConfig(min_throughput=1e-8, min_split_fraction=0.0),
        )


def test_physical_forward_does_not_accept_independent_beta_override() -> None:
    group = {"g": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])}
    route = build_active_routes(group)[0]
    ledger = build_exposure_ledger(group)
    with pytest.raises(TypeError, match="beta_pos"):
        physical_dual_rail_forward(
            torch.ones(1, 1, 5, 5),
            positive_psf=torch.ones(3, 3) / 9,
            negative_psf=None,
            positive_route=route,
            negative_route=None,
            beta_pos=0.5,
            ledger=ledger,
            support=3,
            padding=1,
            config=OptoelectronicConfig(min_throughput=1e-8, min_split_fraction=0.0),
        )


def test_exposure_ledger_rejects_out_of_range_beta_and_over_budget_sum() -> None:
    with pytest.raises(ValueError, match="beta"):
        Exposure(
            exposure_id=0,
            input_rail="pos",
            route_ids=("r",),
            beta_by_route={"r": 2.0},
            sum_beta=2.0,
            unused_budget=-1.0,
        )
    with pytest.raises(ValueError, match="sum_beta|budget"):
        Exposure(
            exposure_id=0,
            input_rail="pos",
            route_ids=("r1", "r2"),
            beta_by_route={"r1": 0.75, "r2": 0.75},
            sum_beta=1.5,
            unused_budget=-0.5,
        )
    with pytest.raises(ValueError, match="duplicate route"):
        Exposure(
            exposure_id=0,
            input_rail="pos",
            route_ids=("r", "r"),
            beta_by_route={"r": 0.5},
            sum_beta=0.5,
            unused_budget=0.5,
        )


def test_exposure_ledger_rejects_route_missing_from_one_input_rail() -> None:
    group = {"g": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])}
    ledger = build_exposure_ledger(group)
    positive_exposure, negative_exposure = ledger.exposures
    mismatched_negative = Exposure(
        exposure_id=negative_exposure.exposure_id,
        input_rail=negative_exposure.input_rail,
        route_ids=("other-route",),
        beta_by_route={"other-route": 1.0},
        sum_beta=1.0,
        unused_budget=0.0,
    )
    with pytest.raises(ValueError, match="dual-rail|route"):
        ExposureLedger(
            policy=ledger.policy,
            active_group_count=ledger.active_group_count,
            active_route_count=ledger.active_route_count,
            exposures=(positive_exposure, mismatched_negative),
            canonical_routes=ledger.canonical_routes,
        )


def test_physical_forward_rejects_unbound_scalar_beta_bypass() -> None:
    group = {"g": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])}
    route = build_active_routes(group)[0]
    ledger = build_exposure_ledger(group)
    with pytest.raises(TypeError, match="beta_pos"):
        physical_dual_rail_forward(
            torch.ones(1, 1, 5, 5),
            positive_psf=torch.ones(3, 3) / 9,
            negative_psf=None,
            positive_route=route,
            negative_route=None,
            beta_pos=2.0,
            ledger=ledger,
            support=3,
            padding=1,
            config=OptoelectronicConfig(min_throughput=1e-8, min_split_fraction=0.0),
        )


def test_physical_forward_rejects_unbound_psf_without_typed_route() -> None:
    with pytest.raises(ValueError, match="route"):
        physical_dual_rail_forward(
            torch.ones(1, 1, 5, 5),
            positive_psf=torch.ones(3, 3) / 9,
            negative_psf=None,
            support=3,
            padding=1,
            config=OptoelectronicConfig(min_throughput=1e-8, min_split_fraction=0.0),
        )


def test_exposure_ledger_keeps_and_resolves_canonical_routes() -> None:
    group = {"g": torch.tensor([[[[1.0, -2.0], [0.0, 0.0]]]])}
    routes = build_active_routes(group)
    ledger = build_exposure_ledger(group, policy="global_parallel")

    assert tuple(route.route_id for route in ledger.canonical_routes) == tuple(
        route.route_id for route in routes
    )
    for route in routes:
        resolved = ledger.route_for_id(route.route_id)
        assert resolved.route_id == route.route_id
        assert resolved.group_id == route.group_id
        assert resolved.component_index == route.component_index
        assert resolved.sign == route.sign
        assert resolved.alpha == route.alpha
        assert resolved.lobe.dtype == route.lobe.dtype
        assert torch.equal(resolved.lobe, route.lobe)


def test_exposure_ledger_rejects_unit_energy_canonical_lobe_mutation() -> None:
    route, ledger = _single_positive_route_ledger()
    with torch.no_grad():
        route.lobe.copy_(torch.tensor([[[0.35, 0.65]]], dtype=route.lobe.dtype))
    assert torch.all(route.lobe >= 0)
    assert torch.isfinite(route.lobe).all()
    assert torch.sum(route.lobe).item() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="identity|creation"):
        ledger.validate()


def test_exposure_ledger_route_lookup_and_physical_forward_reject_mutated_canonical_lobe() -> None:
    route, ledger = _single_positive_route_ledger()
    with torch.no_grad():
        route.lobe.copy_(torch.tensor([[[0.35, 0.65]]], dtype=route.lobe.dtype))
    with pytest.raises(ValueError, match="identity|creation"):
        ledger.route_for_id(route.route_id)
    with pytest.raises(ValueError, match="identity|creation"):
        ledger.beta_for_route(route.route_id, input_rail="pos")
    with pytest.raises(ValueError, match="identity|creation"):
        physical_dual_rail_forward(
            torch.ones(1, 1, 5, 5),
            positive_psf=torch.ones(3, 3) / 9,
            negative_psf=None,
            positive_route=route,
            negative_route=None,
            ledger=ledger,
            support=3,
            padding=1,
            config=OptoelectronicConfig(min_throughput=1e-8, min_split_fraction=0.0),
        )


@pytest.mark.parametrize(
    "replacement",
    [
        torch.tensor([[[-0.1, 1.1]]]),
        torch.tensor([[[float("nan"), 0.75]]]),
        torch.tensor([[[float("inf"), 0.75]]]),
    ],
)
def test_exposure_ledger_rejects_invalid_canonical_lobe_mutations(
    replacement: torch.Tensor,
) -> None:
    route, ledger = _single_positive_route_ledger()
    with torch.no_grad():
        route.lobe.copy_(replacement.to(dtype=route.lobe.dtype))
    with pytest.raises(ValueError, match="lobe|identity|creation"):
        ledger.validate()


def test_exposure_ledger_rejects_canonical_lobe_shape_mutation() -> None:
    route, ledger = _single_positive_route_ledger()
    with torch.no_grad():
        route.lobe.resize_(1, 2)
    with pytest.raises(ValueError, match="identity|creation"):
        ledger.validate()


def test_exposure_ledger_repeated_validation_and_physical_lookup_remain_valid_without_mutation() -> None:
    route, ledger = _single_positive_route_ledger()
    for _ in range(3):
        ledger.validate()
        assert ledger.route_for_id(route.route_id) is route
    result = physical_dual_rail_forward(
        torch.ones(1, 1, 5, 5),
        positive_psf=torch.ones(3, 3) / 9,
        negative_psf=None,
        positive_route=route,
        negative_route=None,
        ledger=ledger,
        support=3,
        padding=1,
        config=OptoelectronicConfig(min_throughput=1e-8, min_split_fraction=0.0),
    )
    assert torch.isfinite(result.output).all()


def _identity_probe_route(
    *,
    group_id: str = "g",
    component_index: int = 0,
    sign: str = "positive",
    alpha: float = 4.0,
    lobe: torch.Tensor | None = None,
) -> ActiveRoute:
    return ActiveRoute(
        route_id=f"{group_id}:component{component_index}:{sign}",
        group_id=group_id,
        component_index=component_index,
        sign=sign,
        lobe=torch.tensor([0.25, 0.75]) if lobe is None else lobe,
        alpha=alpha,
    )


def test_canonical_route_identity_digest_is_stable_and_covers_route_fields() -> None:
    base = _identity_probe_route()

    def digest(route: ActiveRoute) -> str:
        ledger = build_exposure_ledger((route,))
        return ledger.as_dict()["canonical_route_identity_digests"][0]

    base_digest = digest(base)
    assert base_digest == digest(replace(base, lobe=base.lobe.clone()))
    variants = (
        replace(base, lobe=torch.tensor([0.3, 0.7])),
        replace(base, lobe=torch.tensor([[0.25, 0.75]])),
        replace(base, lobe=base.lobe.double()),
        replace(base, alpha=5.0),
        _identity_probe_route(group_id="other"),
        _identity_probe_route(component_index=1),
        _identity_probe_route(sign="negative"),
    )
    assert all(digest(variant) != base_digest for variant in variants)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available for the device-independent digest probe")
def test_canonical_route_identity_digest_is_device_independent() -> None:
    cpu_route = _identity_probe_route()
    gpu_route = replace(cpu_route, lobe=cpu_route.lobe.cuda())
    assert build_exposure_ledger((cpu_route,)).as_dict()["canonical_route_identity_digests"] == (
        build_exposure_ledger((gpu_route,)).as_dict()["canonical_route_identity_digests"]
    )


def test_exposure_ledger_as_dict_reports_stable_creation_identity_digest() -> None:
    _, ledger = _single_positive_route_ledger()
    first = ledger.as_dict()
    second = ledger.as_dict()
    assert first == second
    assert first["canonical_routes"]
    assert first["canonical_route_identity_digests"]
    assert first["canonical_route_identities"]
    assert first["exposures"]
    assert all("0x" not in digest for digest in first["canonical_route_identity_digests"])


@pytest.mark.parametrize(
    "replacement",
    [
        pytest.param(torch.tensor([[[0.35, 0.65]]]), id="unit-energy-content"),
        pytest.param(torch.tensor([[[-0.1, 1.1]]]), id="negative"),
        pytest.param(torch.tensor([[[float("nan"), 0.75]]]), id="nan"),
        pytest.param(torch.tensor([[[float("inf"), 0.75]]]), id="inf"),
    ],
)
def test_exposure_ledger_as_dict_rejects_canonical_lobe_mutation(
    replacement: torch.Tensor,
) -> None:
    route, ledger = _single_positive_route_ledger()
    with torch.no_grad():
        route.lobe.copy_(replacement.to(dtype=route.lobe.dtype))
    with pytest.raises(ValueError, match="identity|creation|canonical"):
        ledger.as_dict()


def test_exposure_ledger_as_dict_rejects_canonical_lobe_shape_mutation() -> None:
    route, ledger = _single_positive_route_ledger()
    with torch.no_grad():
        route.lobe.resize_(1, 2)
    with pytest.raises(ValueError, match="identity|creation|canonical"):
        ledger.as_dict()


def test_physical_forward_rejects_negative_route_on_positive_branch() -> None:
    group = {"g": torch.tensor([[[[1.0, -2.0], [0.0, 0.0]]]])}
    routes = build_active_routes(group)
    negative_route = next(route for route in routes if route.sign == "negative")
    ledger = build_exposure_ledger(group, policy="global_parallel")
    with pytest.raises(ValueError, match="positive|sign|canonical"):
        physical_dual_rail_forward(
            torch.ones(1, 1, 5, 5),
            positive_psf=torch.ones(3, 3) / 9,
            negative_psf=None,
            positive_route=negative_route,
            negative_route=None,
            ledger=ledger,
            support=3,
            padding=1,
            config=OptoelectronicConfig(min_throughput=1e-8, min_split_fraction=0.0),
        )


def test_physical_forward_rejects_positive_route_on_negative_branch() -> None:
    group = {"g": torch.tensor([[[[1.0, -2.0], [0.0, 0.0]]]])}
    routes = build_active_routes(group)
    positive_route = next(route for route in routes if route.sign == "positive")
    ledger = build_exposure_ledger(group, policy="global_parallel")
    with pytest.raises(ValueError, match="negative|sign|canonical"):
        physical_dual_rail_forward(
            torch.ones(1, 1, 5, 5),
            positive_psf=None,
            negative_psf=torch.ones(3, 3) / 9,
            positive_route=None,
            negative_route=positive_route,
            ledger=ledger,
            support=3,
            padding=1,
            config=OptoelectronicConfig(min_throughput=1e-8, min_split_fraction=0.0),
        )


def test_physical_forward_does_not_accept_independent_alpha_override() -> None:
    group = {"g": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])}
    route = build_active_routes(group)[0]
    ledger = build_exposure_ledger(group)
    with pytest.raises(TypeError, match="alpha_pos"):
        physical_dual_rail_forward(
            torch.ones(1, 1, 5, 5),
            positive_psf=torch.ones(3, 3) / 9,
            negative_psf=None,
            positive_route=route,
            negative_route=None,
            alpha_pos=9.0,
            ledger=ledger,
            support=3,
            padding=1,
            config=OptoelectronicConfig(min_throughput=1e-8, min_split_fraction=0.0),
        )


def test_physical_forward_rejects_same_id_with_tampered_route_identity() -> None:
    group = {"g": torch.tensor([[[[1.0, 2.0], [0.0, 0.0]]]])}
    route = build_active_routes(group)[0]
    ledger = build_exposure_ledger(group)
    tampered_routes = (
        replace(route, group_id="other"),
        replace(route, component_index=route.component_index + 1),
        replace(route, sign="negative"),
        replace(route, alpha=route.alpha + 1.0),
        replace(route, lobe=torch.flip(route.lobe, dims=(-1,))),
    )
    for tampered in tampered_routes:
        with pytest.raises(ValueError, match="identity|canonical|route|sign"):
            physical_dual_rail_forward(
                torch.ones(1, 1, 5, 5),
                positive_psf=torch.ones(3, 3) / 9,
                negative_psf=None,
                positive_route=tampered,
                negative_route=None,
                ledger=ledger,
                support=3,
                padding=1,
                config=OptoelectronicConfig(min_throughput=1e-8, min_split_fraction=0.0),
            )


def test_exposure_ledger_rejects_canonical_route_set_mismatch() -> None:
    group = {"g": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])}
    ledger = build_exposure_ledger(group)
    positive_exposure, negative_exposure = ledger.exposures
    extra_positive = Exposure(
        exposure_id=positive_exposure.exposure_id,
        input_rail=positive_exposure.input_rail,
        route_ids=("other-route",),
        beta_by_route={"other-route": 1.0},
        sum_beta=1.0,
        unused_budget=0.0,
    )
    extra_negative = Exposure(
        exposure_id=negative_exposure.exposure_id,
        input_rail=negative_exposure.input_rail,
        route_ids=("other-route",),
        beta_by_route={"other-route": 1.0},
        sum_beta=1.0,
        unused_budget=0.0,
    )
    with pytest.raises(ValueError, match="canonical|route"):
        ExposureLedger(
            policy=ledger.policy,
            active_group_count=ledger.active_group_count,
            active_route_count=ledger.active_route_count,
            exposures=(extra_positive, extra_negative),
            canonical_routes=ledger.canonical_routes,
        )


def test_exposure_ledger_rejects_inconsistent_active_group_count() -> None:
    group = {
        "g0": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]]),
        "g1": torch.tensor([[[[2.0, 0.0], [0.0, 0.0]]]]),
    }
    ledger = build_exposure_ledger(group)
    with pytest.raises(ValueError, match="active_group_count|group"):
        ExposureLedger(
            policy=ledger.policy,
            active_group_count=ledger.active_group_count + 1,
            active_route_count=ledger.active_route_count,
            exposures=ledger.exposures,
            canonical_routes=ledger.canonical_routes,
        )


@pytest.mark.parametrize(
    "policy_kwargs",
    [
        ("global_parallel", {}),
        ("group_sequential", {}),
        ("route_sequential", {}),
        ("bounded_parallel", {"route_batch_capacity": 2}),
    ],
)
def test_physical_forward_accepts_all_ledger_policies_with_typed_routes(
    policy_kwargs: tuple[str, dict[str, int]],
) -> None:
    policy, kwargs = policy_kwargs
    group = {
        "g0": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]]),
        "g1": torch.tensor([[[[2.0, 0.0], [0.0, 0.0]]]]),
    }
    route = build_active_routes(group)[0]
    ledger = build_exposure_ledger(group, policy=policy, **kwargs)
    result = physical_dual_rail_forward(
        torch.ones(1, 1, 5, 5),
        positive_psf=torch.ones(3, 3) / 9,
        negative_psf=None,
        positive_route=route,
        negative_route=None,
        ledger=ledger,
        support=3,
        padding=1,
        config=OptoelectronicConfig(min_throughput=1e-8, min_split_fraction=0.0),
    )
    assert torch.isfinite(result.output).all()


def test_physical_forward_accepts_negative_only_typed_route() -> None:
    group = {"g": torch.tensor([[[[-1.0, 0.0], [0.0, 0.0]]]])}
    route = build_active_routes(group)[0]
    assert route.sign == "negative"
    ledger = build_exposure_ledger(group, policy="global_parallel")
    result = physical_dual_rail_forward(
        torch.ones(1, 1, 5, 5),
        positive_psf=None,
        negative_psf=torch.ones(3, 3) / 9,
        positive_route=None,
        negative_route=route,
        ledger=ledger,
        support=3,
        padding=1,
        config=OptoelectronicConfig(min_throughput=1e-8, min_split_fraction=0.0),
    )
    assert result.negative.alpha == route.alpha
    assert torch.isfinite(result.output).all()


def test_physical_forward_accepts_perturbed_psf_and_preserves_psf_gradient() -> None:
    group = {"g": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])}
    route = build_active_routes(group)[0]
    ledger = build_exposure_ledger(group)
    psf = torch.ones(3, 3, requires_grad=True)
    result = physical_dual_rail_forward(
        torch.ones(1, 1, 5, 5),
        positive_psf=psf,
        negative_psf=None,
        positive_route=route,
        negative_route=None,
        ledger=ledger,
        support=3,
        padding=1,
        config=OptoelectronicConfig(min_throughput=1e-8, min_split_fraction=0.0),
    )
    result.output.square().mean().backward()
    assert psf.grad is not None
    assert torch.isfinite(psf.grad).all()


def test_constraint_failures_are_explicit_and_fail_closed() -> None:
    tau = torch.tensor(0.01)
    with pytest.raises(ValueError, match="throughput"):
        compute_calibrated_gain(alpha=1.0, rho=1.0, beta=1.0, tau=tau, min_throughput=0.1)
    with pytest.raises(ValueError, match="gain"):
        compute_calibrated_gain(alpha=1.0, rho=1.0, beta=1.0, tau=tau, max_electronic_gain=10.0)
    with pytest.raises(ValueError, match="split fraction|beta"):
        compute_calibrated_gain(alpha=1.0, rho=1.0, beta=0.001, tau=torch.tensor(1.0), min_split_fraction=0.01)
    psf = torch.zeros(8, 8)
    psf[4, 4] = 0.8
    psf[4, 5] = 0.2
    with pytest.raises(SupportResponseError, match="support"):
        support_sweep(psf, supports=(1, 3), max_support_response_error=0.0, mode="fail_closed")


def test_fit_metrics_separate_shape_normalization_from_physical_throughput() -> None:
    target = torch.tensor([[1.0, 2.0], [0.0, 1.0]])
    full = torch.zeros(8, 8)
    full[3:5, 3:5] = target / target.sum()
    metrics = fit_psf_shape(target, full, support=2, alpha=2.0, rho=2.0, beta=0.5)
    assert metrics.normalized_mse == pytest.approx(0.0, abs=1e-8)
    assert metrics.cosine_similarity == pytest.approx(1.0, abs=1e-7)
    assert metrics.response_error == pytest.approx(0.0, abs=1e-7)
    assert metrics.low_throughput_penalty == 0.0
    assert metrics.excessive_gain_penalty == 0.0


def test_phase_module_has_trainable_theta_and_phase_gradient() -> None:
    module = PhaseOnlyPSF(_small_phase_config())
    assert isinstance(module.theta, nn.Parameter)
    psf = module()
    loss = (psf * torch.arange(psf.numel(), dtype=psf.dtype).view_as(psf)).sum()
    loss.backward()
    assert module.theta.grad is not None
    assert torch.isfinite(module.theta.grad).all()


def test_ideal_and_physical_paths_are_explicitly_distinct() -> None:
    assert ideal_signed_conv2d is not physical_dual_rail_forward
    assert ideal_signed_conv2d.__name__ != physical_dual_rail_forward.__name__
    config = _small_phase_config()
    theta = torch.zeros(16, 16)
    psf = phase_to_full_psf(theta, config=config)
    x = torch.randn(1, 1, 7, 7)
    path_group = {"path": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])}
    path_route = build_active_routes(path_group)[0]
    path_ledger = build_exposure_ledger(path_group, policy="global_parallel")
    physical = physical_dual_rail_forward(
        x,
        positive_psf=psf,
        negative_psf=None,
        positive_route=path_route,
        negative_route=None,
        rho=1.0,
        ledger=path_ledger,
        support=3,
        padding=1,
        config=OptoelectronicConfig(phase=config, min_split_fraction=0.0),
    )
    assert hasattr(physical, "snapshots")
    assert hasattr(physical, "output")


def test_cpu_phase_and_ledger_are_deterministic() -> None:
    config = _small_phase_config()
    theta = torch.linspace(-1.0, 1.0, config.phase_grid_size**2).view(
        config.phase_grid_size, config.phase_grid_size
    )
    assert torch.equal(phase_to_full_psf(theta, config=config), phase_to_full_psf(theta, config=config))
    groups = {"g": torch.tensor([[[[1.0, -2.0], [0.0, 0.0]]]])}
    first = build_exposure_ledger(groups).as_dict()
    second = build_exposure_ledger(groups).as_dict()
    assert first == second


def test_malformed_configs_and_inputs_fail_explicitly() -> None:
    with pytest.raises(ValueError, match="wavelength|positive"):
        PhasePSFConfig(wavelength_m=0.0)
    with pytest.raises(ValueError, match="magnification|positive"):
        PhasePSFConfig(magnification=0.0)
    with pytest.raises(ValueError, match="square|shape"):
        phase_to_full_psf(torch.zeros(4, 5), config=_small_phase_config())
    with pytest.raises(ValueError, match="support"):
        crop_centered(torch.ones(4, 4), 5)
    with pytest.raises(ValueError, match="route_batch_capacity"):
        build_exposure_ledger({"g": torch.ones(1, 1, 2, 2)}, policy="bounded_parallel", route_batch_capacity=0)


def test_config_reports_default_physical_assumptions_without_claiming_paper_fidelity() -> None:
    config = PhasePSFConfig()
    metadata = detector_sampling_metadata(config).as_dict()
    assert metadata["assumption_status"] == "synthetic engineering assumptions"
    assert metadata["wavelength_m"] == pytest.approx(520e-9)
    assert metadata["phase_grid_size"] == 256
    assert metadata["phase_pitch_m"] == pytest.approx(8e-6)
    assert metadata["focal_length_m"] == pytest.approx(20e-3)
    assert metadata["aperture_diameter_m"] == pytest.approx(2.048e-3)
    assert metadata["magnification"] == pytest.approx(1.0)


def _run_small_physical_forward(route: ActiveRoute, ledger: ExposureLedger) -> frontend.PhysicalForwardResult:
    return physical_dual_rail_forward(
        torch.ones(1, 1, 7, 7),
        positive_psf=torch.ones(5, 5) / 25,
        negative_psf=None,
        positive_route=route,
        negative_route=None,
        ledger=ledger,
        support=3,
        padding=1,
        config=OptoelectronicConfig(min_throughput=1e-8, min_split_fraction=0.0),
    )


def test_physical_forward_hot_path_does_not_rebuild_route_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route, ledger = _single_positive_route_ledger()
    calls = 0
    original = frontend._active_route_identity_snapshot

    def recording(route_value: ActiveRoute) -> frontend._ActiveRouteIdentitySnapshot:
        nonlocal calls
        calls += 1
        return original(route_value)

    monkeypatch.setattr(frontend, "_active_route_identity_snapshot", recording)
    result = _run_small_physical_forward(route, ledger)

    assert torch.isfinite(result.output).all()
    assert calls == 0


def test_physical_forward_hot_path_never_enters_route_cpu_hash_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route, ledger = _single_positive_route_ledger()

    def fail_if_called(_: ActiveRoute) -> frontend._ActiveRouteIdentitySnapshot:
        raise AssertionError("route digest helper entered the physical hot path")

    monkeypatch.setattr(frontend, "_active_route_identity_snapshot", fail_if_called)
    result = _run_small_physical_forward(route, ledger)

    assert torch.isfinite(result.output).all()


def test_runtime_lobe_version_drift_rejects_physical_forward_without_full_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route, ledger = _single_positive_route_ledger()
    with torch.no_grad():
        route.lobe.copy_(torch.tensor([[[0.35, 0.65]]]))

    def fail_if_called(_: ActiveRoute) -> frontend._ActiveRouteIdentitySnapshot:
        raise AssertionError("full route digest should not be needed for runtime drift")

    monkeypatch.setattr(frontend, "_active_route_identity_snapshot", fail_if_called)
    with pytest.raises(ValueError, match="runtime|version|mutation|identity|creation"):
        _run_small_physical_forward(route, ledger)


@pytest.mark.parametrize(
    "mutation",
    ["beta", "route_ids", "sum_beta", "unused_budget", "policy", "counts"],
)
def test_runtime_exposure_drift_rejects_physical_forward_without_full_digest(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    route, ledger = _single_positive_route_ledger()
    exposure = ledger.exposures[0]
    if mutation == "beta":
        exposure.beta_by_route[route.route_id] = 0.5
    elif mutation == "route_ids":
        object.__setattr__(exposure, "route_ids", ("changed-route",))
    elif mutation == "sum_beta":
        object.__setattr__(exposure, "sum_beta", 0.5)
    elif mutation == "unused_budget":
        object.__setattr__(exposure, "unused_budget", 0.5)
    elif mutation == "policy":
        object.__setattr__(ledger, "policy", "route_sequential")
    else:
        object.__setattr__(ledger, "active_route_count", ledger.active_route_count + 1)

    def fail_if_called(_: ActiveRoute) -> frontend._ActiveRouteIdentitySnapshot:
        raise AssertionError("full route digest should not be needed for runtime exposure drift")

    monkeypatch.setattr(frontend, "_active_route_identity_snapshot", fail_if_called)
    with pytest.raises(ValueError, match="runtime|exposure|drift|mutation|identity|creation|count"):
        _run_small_physical_forward(route, ledger)


def test_runtime_and_full_validation_keep_distinct_lobe_mutation_guards() -> None:
    route, ledger = _single_positive_route_ledger()
    with torch.no_grad():
        route.lobe.copy_(torch.tensor([[[0.35, 0.65]]]))

    with pytest.raises(ValueError, match="runtime|version|mutation|identity|creation"):
        ledger.validate_runtime()
    with pytest.raises(ValueError, match="identity|creation"):
        ledger.validate()
    with pytest.raises(ValueError, match="identity|creation"):
        ledger.as_dict()
