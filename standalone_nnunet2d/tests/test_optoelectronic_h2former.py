from __future__ import annotations

import gc
import copy
import json
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from standalone_nnunet2d.models.h2former import H2Former
from standalone_nnunet2d.models.optoelectronic_frontend import (
    OptoelectronicConfig,
    PhasePSFConfig,
    build_exposure_ledger,
    signed_kernel_lobes,
)
from standalone_nnunet2d.models.optoelectronic_h2former import (
    ENTRY_GROUPS,
    OptoelectronicH2Former,
    OptoelectronicH2FormerConfig,
    OptoelectronicPCAConv2d,
)


GROUP_IDS = tuple(ENTRY_GROUPS)
ENTRY_SPECS = (
    ("stem7", "conv1", 64, 7, 1, 3, False),
    ("patch2", "patch_embed.projs.0", 32, 2, 2, 0, True),
    ("patch4", "patch_embed.projs.1", 16, 4, 2, 1, True),
    ("patch8", "patch_embed.projs.2", 8, 8, 2, 3, True),
    ("patch16", "patch_embed.projs.3", 8, 16, 2, 7, True),
)


def _all_ranks(rank: int = 0) -> dict[str, int]:
    return {group_id: rank for group_id in GROUP_IDS}


def _config(**overrides: object) -> OptoelectronicH2FormerConfig:
    values: dict[str, object] = {
        "rank_by_group": _all_ranks(),
        "variance_threshold": None,
    }
    values.update(overrides)
    return OptoelectronicH2FormerConfig(**values)


def _clone_state_dict(state: dict[str, object]) -> dict[str, object]:
    return {
        name: value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
        for name, value in state.items()
    }


def _assert_state_dict_equal(expected: dict[str, object], actual: dict[str, object]) -> None:
    assert set(expected) == set(actual)
    for name, expected_value in expected.items():
        actual_value = actual[name]
        if isinstance(expected_value, torch.Tensor):
            assert isinstance(actual_value, torch.Tensor)
            assert torch.equal(expected_value, actual_value), name
        else:
            assert actual_value == expected_value, name


def _model_atomic_snapshot(model: OptoelectronicH2Former) -> dict[str, object]:
    return {
        "state": _clone_state_dict(model.state_dict()),
        "metadata": copy.deepcopy(model.identity_metadata()),
        "ledger": copy.deepcopy(model.exposure_ledger.as_dict()),
        "config": copy.deepcopy(model.opto_config.as_dict()),
    }


def _assert_model_unchanged(
    before: dict[str, object],
    model: OptoelectronicH2Former,
) -> None:
    _assert_state_dict_equal(before["state"], model.state_dict())  # type: ignore[arg-type]
    assert before["metadata"] == model.identity_metadata()
    assert before["ledger"] == model.exposure_ledger.as_dict()
    assert before["config"] == model.opto_config.as_dict()


def _small_phase_config(**overrides: object) -> PhasePSFConfig:
    values: dict[str, object] = {
        "phase_grid_size": 16,
        "phase_pitch_m": 8e-6,
        "aperture_diameter_m": 96e-6,
    }
    values.update(overrides)
    return PhasePSFConfig(**values)


@pytest.fixture(scope="module")
def source_h2former() -> H2Former:
    torch.manual_seed(7001)
    model = H2Former(in_channels=1, num_classes=2, image_size=512).eval()
    yield model
    del model
    gc.collect()


def test_model_replaces_exactly_five_entry_convolutions_and_preserves_source(
    source_h2former: H2Former,
) -> None:
    source_entry_weights = {
        "conv1": source_h2former.conv1.weight.detach().clone(),
        **{
            f"patch_embed.projs.{index}": projection.weight.detach().clone()
            for index, projection in enumerate(source_h2former.patch_embed.projs)
        },
    }
    model = OptoelectronicH2Former.from_h2former(source_h2former, _config())

    assert isinstance(model.conv1, OptoelectronicPCAConv2d)
    assert all(isinstance(projection, OptoelectronicPCAConv2d) for projection in model.patch_embed.projs)
    assert isinstance(source_h2former.conv1, nn.Conv2d)
    assert all(isinstance(projection, nn.Conv2d) for projection in source_h2former.patch_embed.projs)
    for group_id, path, out_channels, kernel_size, stride, padding, has_bias in ENTRY_SPECS:
        module = model.conv1 if path == "conv1" else model.patch_embed.projs[int(path.rsplit(".", 1)[1])]
        assert module.group_id == group_id
        assert module.in_channels == 1
        assert module.out_channels == out_channels
        assert module.kernel_size == (kernel_size, kernel_size)
        assert module.stride == (stride, stride)
        assert module.padding == (padding, padding)
        assert (module.bias is not None) is has_bias
    assert torch.equal(source_h2former.conv1.weight, source_entry_weights["conv1"])
    for index, projection in enumerate(source_h2former.patch_embed.projs):
        assert torch.equal(projection.weight, source_entry_weights[f"patch_embed.projs.{index}"])


def test_non_entry_electronic_state_is_copied_strictly(
    source_h2former: H2Former,
) -> None:
    model = OptoelectronicH2Former.from_h2former(source_h2former, _config())
    source_state = source_h2former.state_dict()
    model_state = model.state_dict()
    entry_prefixes = ("conv1.", "patch_embed.projs.")

    for name, value in source_state.items():
        if name.startswith(entry_prefixes):
            continue
        assert name in model_state
        assert torch.equal(model_state[name], value), name


def test_inherited_h2former_forward_keeps_the_original_electronic_data_flow(
    source_h2former: H2Former,
) -> None:
    model = OptoelectronicH2Former.from_h2former(source_h2former, _config())

    assert OptoelectronicH2Former.forward_features is H2Former.forward_features
    assert isinstance(model.bn1, nn.BatchNorm2d)
    assert isinstance(model.relu, nn.ReLU)
    assert isinstance(model.maxpool, nn.MaxPool2d)
    assert isinstance(model.layer1, nn.Sequential)
    assert isinstance(model.patch_embed.eca, nn.Module)
    assert isinstance(model.patch_embed.norm, nn.LayerNorm)
    assert model.MS2 is not None
    assert model.MS3 is not None
    assert model.MS4 is not None
    assert model.decode4 is not None
    assert model.decode3 is not None
    assert model.decode2 is not None
    assert model.decode0 is not None


@pytest.mark.parametrize(
    ("out_channels", "kernel_size", "stride", "padding", "bias"),
    [
        (64, 7, 1, 3, False),
        (32, 2, 2, 0, True),
        (16, 4, 2, 1, True),
        (8, 8, 2, 3, True),
        (8, 16, 2, 7, True),
    ],
)
def test_full_rank_ideal_entry_matches_original_conv_with_signed_input(
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    bias: bool,
) -> None:
    torch.manual_seed(7002 + kernel_size)
    convolution = nn.Conv2d(
        1,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    ).eval()
    rank = min(out_channels - 1, kernel_size * kernel_size)
    optical = OptoelectronicPCAConv2d(
        convolution,
        group_id=f"test{kernel_size}",
        mode="ideal",
        rank=rank,
        variance_threshold=None,
    ).eval()
    image_size = max(31, kernel_size + 9)
    image = torch.randn(2, 1, image_size, image_size)

    with torch.inference_mode():
        expected = convolution(image)
        actual = optical(image)

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_ideal_path_calls_signed_dual_rail_for_each_component_and_adds_bias_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(7003)
    convolution = nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=True).eval()
    optical = OptoelectronicPCAConv2d(
        convolution,
        group_id="bias_once",
        mode="ideal",
        rank=2,
        variance_threshold=None,
    ).eval()
    calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = []
    original = __import__(
        "standalone_nnunet2d.models.optoelectronic_frontend",
        fromlist=["ideal_signed_conv2d"],
    ).ideal_signed_conv2d

    def recording(
        image: torch.Tensor,
        kernel: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        calls.append((image, kernel, bias))
        return original(image, kernel, bias=bias, **kwargs)

    monkeypatch.setattr(
        __import__(
            "standalone_nnunet2d.models.optoelectronic_h2former",
            fromlist=["ideal_signed_conv2d"],
        ),
        "ideal_signed_conv2d",
        recording,
    )
    image = torch.randn(1, 1, 9, 10)
    actual = optical(image)

    assert len(calls) == optical.component_bank.shape[0]
    assert all(bias is None for _, _, bias in calls)
    assert all(kernel.shape == (1, 1, 3, 3) for _, kernel, _ in calls)
    assert torch.isfinite(actual).all()


def test_pca_rank_selection_and_reports_are_explicit() -> None:
    convolution = nn.Conv2d(1, 5, kernel_size=3, bias=True).eval()
    truncated = OptoelectronicPCAConv2d(
        convolution,
        group_id="truncated",
        mode="ideal",
        rank=0,
        variance_threshold=None,
    )
    assert truncated.resolved_rank == 0
    assert truncated.structural_rank_upper_bound == 5 - 1
    assert truncated.pca_report.resolved_rank == 0
    assert truncated.pca_report.absolute_frobenius_error >= 0
    assert truncated.pca_report.maximum_absolute_error >= 0
    assert torch.isfinite(truncated.component_bank).all()

    with pytest.raises(ValueError, match="exactly one|rank|variance"):
        OptoelectronicPCAConv2d(
            convolution,
            group_id="invalid-both",
            mode="ideal",
            rank=1,
            variance_threshold=0.9,
        )
    with pytest.raises(ValueError, match="rank"):
        OptoelectronicPCAConv2d(
            convolution,
            group_id="invalid-rank",
            mode="ideal",
            rank=5,
            variance_threshold=None,
        )


def test_default_variance_threshold_is_configurable_per_all_five_groups(
    source_h2former: H2Former,
) -> None:
    config = OptoelectronicH2FormerConfig(variance_threshold=0.99)
    model = OptoelectronicH2Former.from_h2former(source_h2former, config)
    metadata = model.identity_metadata()

    assert metadata["mode"] == "ideal"
    assert set(metadata["resolved_ranks"]) == set(GROUP_IDS)
    assert all(0 <= rank <= metadata["pca"][group_id]["structural_rank_upper_bound"] for group_id, rank in metadata["resolved_ranks"].items())
    assert all("explained_variance_ratio" in metadata["pca"][group_id]["report"] for group_id in GROUP_IDS)

    with pytest.raises(ValueError, match="exactly one|rank|variance"):
        OptoelectronicH2FormerConfig(
            rank_by_group=_all_ranks(0),
            variance_threshold=0.99,
        )


def test_model_config_rejects_unknown_or_incomplete_explicit_rank_groups() -> None:
    with pytest.raises(ValueError, match="group"):
        OptoelectronicH2FormerConfig(
            rank_by_group={"unknown": 0},
            variance_threshold=None,
        )


def test_physical_entry_is_finite_has_nonnegative_psf_and_phase_gradient() -> None:
    torch.manual_seed(7004)
    convolution = nn.Conv2d(1, 3, kernel_size=3, padding=1, bias=True).eval()
    physical_config = OptoelectronicConfig(
        phase=_small_phase_config(),
        min_throughput=1e-8,
        max_electronic_gain=1e8,
        min_split_fraction=0.0,
    )
    optical = OptoelectronicPCAConv2d(
        convolution,
        group_id="physical",
        mode="physical",
        rank=0,
        variance_threshold=None,
        physical_config=physical_config,
        trainable_phase=True,
    ).eval()
    image = torch.randn(1, 1, 9, 9)
    output = optical(image)

    assert torch.isfinite(output).all()
    assert optical.physical_fit_status == "unfitted_synthetic_initialization"
    assert optical.phase_psfs
    for phase_psf in optical.phase_psfs.values():
        full_psf = phase_psf()
        assert torch.isfinite(full_psf).all()
        assert torch.all(full_psf >= 0)
        assert full_psf.sum().item() == pytest.approx(1.0, abs=1e-5)

    output.square().mean().backward()
    gradients = [phase_psf.theta.grad for phase_psf in optical.phase_psfs.values()]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


def test_global_ledger_covers_all_five_groups_and_reports_exposures(
    source_h2former: H2Former,
) -> None:
    model = OptoelectronicH2Former.from_h2former(source_h2former, _config())
    ledger = model.exposure_ledger
    routes = ledger.canonical_routes

    assert ledger.active_group_count == 5
    assert {route.group_id for route in routes} == set(GROUP_IDS)
    assert ledger.active_route_count == len(routes)
    assert ledger.exposure_count == len(ledger.exposures)
    assert all(exposure.sum_beta <= 1.0 + 1e-7 for exposure in ledger.exposures)
    assert all(exposure.unused_budget == pytest.approx(0.0) for exposure in ledger.exposures)
    assert model.conv1.exposure_ledger is ledger
    assert all(projection.exposure_ledger is ledger for projection in model.patch_embed.projs)

    metadata = model.identity_metadata()
    assert metadata["global_exposure"]["active_route_count"] == ledger.active_route_count
    assert metadata["global_exposure"]["exposure_count"] == ledger.exposure_count
    assert metadata["global_exposure"]["policy"] == ledger.policy
    json.dumps(metadata)


def test_trainable_and_frozen_configuration_matches_requires_grad(
    source_h2former: H2Former,
) -> None:
    model = OptoelectronicH2Former.from_h2former(
        source_h2former,
        _config(
            trainable_electronic_backend=False,
            trainable_mixing=True,
            trainable_bias=True,
        ),
    )

    assert model.bn1.weight.requires_grad is False
    assert model.swin_layers[0].blocks[0].attn.qkv.weight.requires_grad is False
    assert isinstance(model.conv1.mixing, nn.Parameter)
    assert model.conv1.mixing.requires_grad is True
    assert isinstance(model.patch_embed.projs[0].bias, nn.Parameter)
    assert model.patch_embed.projs[0].bias.requires_grad is True

    frozen = OptoelectronicPCAConv2d(
        nn.Conv2d(1, 2, 3, padding=1),
        group_id="frozen_phase",
        mode="physical",
        rank=0,
        variance_threshold=None,
        physical_config=OptoelectronicConfig(phase=_small_phase_config()),
        trainable_phase=False,
    )
    assert all(not phase_psf.theta.requires_grad for phase_psf in frozen.phase_psfs.values())


def test_model_state_dict_round_trip_is_strict_for_same_resolved_architecture(
    source_h2former: H2Former,
) -> None:
    config = _config(trainable_mixing=True, trainable_bias=True)
    first = OptoelectronicH2Former.from_h2former(source_h2former, config)
    second = OptoelectronicH2Former.from_h2former(source_h2former, config)
    state = first.state_dict()
    result = second.load_state_dict(state, strict=True)

    assert not result.missing_keys
    assert not result.unexpected_keys
    assert set(first.state_dict()) == set(second.state_dict())
    _assert_state_dict_equal(first.state_dict(), second.state_dict())


def test_identity_extra_state_is_versioned_json_safe_and_isolated_from_mutation(
    source_h2former: H2Former,
) -> None:
    model = OptoelectronicH2Former.from_h2former(source_h2former, _config())

    extra = model.get_extra_state()
    assert isinstance(extra, dict)
    assert extra["schema"]
    assert extra["version"] >= 1
    assert isinstance(extra["snapshot"], dict)
    assert isinstance(extra["snapshot_sha256"], str)
    json.dumps(extra)

    original = copy.deepcopy(extra)
    extra["snapshot"]["resolved_ranks"]["stem7"] = 999
    extra["snapshot_sha256"] = "tampered"
    assert model.get_extra_state() == original
    assert model.state_dict()["_extra_state"] == original


def test_cross_source_strict_load_rejects_before_copy_and_preserves_target_state(
    source_h2former: H2Former,
) -> None:
    torch.manual_seed(7102)
    other_source = H2Former(in_channels=1, num_classes=2, image_size=512).eval()
    config = _config()
    source_model = OptoelectronicH2Former.from_h2former(source_h2former, config)
    target_model = OptoelectronicH2Former.from_h2former(other_source, config)
    incoming = source_model.state_dict()
    assert "_extra_state" in incoming
    before = _model_atomic_snapshot(target_model)

    with pytest.raises((ValueError, RuntimeError), match="identity|source|creation"):
        target_model.load_state_dict(_clone_state_dict(incoming), strict=True)

    _assert_model_unchanged(before, target_model)


def test_same_source_round_trip_restores_trained_state_but_keeps_creation_identity(
    source_h2former: H2Former,
) -> None:
    config = _config(
        mode="physical",
        trainable_mixing=True,
        trainable_bias=True,
        trainable_phase=True,
        physical_config=OptoelectronicConfig(phase=_small_phase_config()),
    )
    first = OptoelectronicH2Former.from_h2former(source_h2former, config)
    second = OptoelectronicH2Former.from_h2former(source_h2former, config)

    with torch.no_grad():
        first.conv1.mixing.add_(0.125)
        first.patch_embed.projs[0].bias.add_(0.25)
        next(iter(first.conv1.phase_psfs.values())).theta.add_(0.05)
        first.bn1.weight.add_(0.375)

    state = first.state_dict()
    assert "_extra_state" in state
    first_identity = first.get_extra_state()
    second.load_state_dict(_clone_state_dict(state), strict=True)

    _assert_state_dict_equal(state, second.state_dict())
    assert second.get_extra_state() == first_identity
    assert second.identity_metadata()["identity_snapshot_sha256"] == first.identity_metadata()[
        "identity_snapshot_sha256"
    ]
    assert second.exposure_ledger.as_dict() == first.exposure_ledger.as_dict()


@pytest.mark.parametrize(
    "buffer_name",
    ["mean_kernel", "basis", "coefficients", "component_bank", "pca_target_weight", "original_bias"],
)
def test_each_immutable_pca_buffer_tamper_is_rejected_atomically(
    source_h2former: H2Former,
    buffer_name: str,
) -> None:
    config = _config(rank_by_group=_all_ranks(1), variance_threshold=None)
    source_model = OptoelectronicH2Former.from_h2former(source_h2former, config)
    target_model = OptoelectronicH2Former.from_h2former(source_h2former, config)
    incoming = _clone_state_dict(source_model.state_dict())
    key = f"patch_embed.projs.0.{buffer_name}"
    assert isinstance(incoming[key], torch.Tensor)
    tampered = incoming[key].clone()  # type: ignore[union-attr]
    tampered.view(-1)[0] += 0.125
    incoming[key] = tampered
    before = _model_atomic_snapshot(target_model)

    with pytest.raises((ValueError, RuntimeError), match="immutable|digest|identity|PCA"):
        target_model.load_state_dict(incoming, strict=True)  # type: ignore[arg-type]

    _assert_model_unchanged(before, target_model)


@pytest.mark.parametrize("tamper", ["snapshot_digest", "snapshot_field", "missing_extra_state"])
def test_extra_state_tamper_is_rejected_atomically(
    source_h2former: H2Former,
    tamper: str,
) -> None:
    source_model = OptoelectronicH2Former.from_h2former(source_h2former, _config())
    target_model = OptoelectronicH2Former.from_h2former(source_h2former, _config())
    incoming = _clone_state_dict(source_model.state_dict())
    assert "_extra_state" in incoming
    assert isinstance(incoming["_extra_state"], dict)
    if tamper == "snapshot_digest":
        incoming["_extra_state"]["snapshot_sha256"] = "tampered"  # type: ignore[index]
    elif tamper == "snapshot_field":
        incoming["_extra_state"]["snapshot"].pop("config")  # type: ignore[index]
    else:
        incoming.pop("_extra_state")
    before = _model_atomic_snapshot(target_model)

    with pytest.raises((ValueError, RuntimeError), match="extra|snapshot|identity|schema"):
        target_model.load_state_dict(incoming, strict=True)  # type: ignore[arg-type]

    _assert_model_unchanged(before, target_model)


@pytest.mark.parametrize("relation", ["mean_kernel", "basis"])
def test_component_bank_redundancy_relation_is_checked_before_load(
    source_h2former: H2Former,
    relation: str,
) -> None:
    config = _config(rank_by_group=_all_ranks(1), variance_threshold=None)
    source_model = OptoelectronicH2Former.from_h2former(source_h2former, config)
    target_model = OptoelectronicH2Former.from_h2former(source_h2former, config)
    incoming = _clone_state_dict(source_model.state_dict())
    component_key = "patch_embed.projs.0.component_bank"
    assert isinstance(incoming[component_key], torch.Tensor)
    component_bank = incoming[component_key].clone()  # type: ignore[union-attr]
    component_bank[0 if relation == "mean_kernel" else 1].add_(0.25)
    incoming[component_key] = component_bank
    before = _model_atomic_snapshot(target_model)

    with pytest.raises((ValueError, RuntimeError), match="component_bank|digest|identity|PCA"):
        target_model.load_state_dict(incoming, strict=True)  # type: ignore[arg-type]

    _assert_model_unchanged(before, target_model)


def test_rank_configuration_is_immutable_and_metadata_uses_resolved_ranks(
    source_h2former: H2Former,
) -> None:
    ranks = _all_ranks(1)
    config = _config(rank_by_group=ranks, variance_threshold=None)
    ranks["stem7"] = 999

    assert config.rank_by_group["stem7"] == 1
    with pytest.raises(TypeError):
        config.rank_by_group["stem7"] = 999  # type: ignore[index]

    exported = config.as_dict()
    exported["rank_by_group"]["stem7"] = 999
    assert config.rank_by_group["stem7"] == 1

    model = OptoelectronicH2Former.from_h2former(source_h2former, config)
    metadata = model.identity_metadata()
    assert metadata["config"]["rank_by_group"] == metadata["resolved_ranks"]
    assert metadata["identity_snapshot"]["resolved_ranks"] == metadata["resolved_ranks"]


def test_ledger_pca_routes_and_metadata_share_one_creation_identity(
    source_h2former: H2Former,
) -> None:
    model = OptoelectronicH2Former.from_h2former(source_h2former, _config())

    for module in model._entry_modules():
        for route in model.exposure_ledger.canonical_routes:
            if route.group_id != module.group_id:
                continue
            lobes = signed_kernel_lobes(module.component_bank[route.component_index])
            expected_lobe = lobes.p_pos if route.sign == "positive" else lobes.p_neg
            expected_alpha = lobes.alpha_pos if route.sign == "positive" else lobes.alpha_neg
            assert torch.equal(route.lobe, expected_lobe)
            assert route.alpha == pytest.approx(expected_alpha)

    metadata = model.identity_metadata()
    extra = model.get_extra_state()
    assert metadata["identity_snapshot_sha256"] == extra["snapshot_sha256"]
    assert metadata["identity_snapshot"]["exposure_ledger"][
        "canonical_route_identity_digests"
    ] == metadata["global_exposure"]["canonical_route_identity_digests"]


def test_live_immutable_pca_mutation_fails_closed_without_refreshing_creation_identity(
    source_h2former: H2Former,
) -> None:
    model = OptoelectronicH2Former.from_h2former(source_h2former, _config())
    original_extra = copy.deepcopy(model.get_extra_state())

    with torch.no_grad():
        model.conv1.mean_kernel.add_(0.125)

    with pytest.raises((ValueError, RuntimeError), match="immutable|identity|creation|drift"):
        model.state_dict()
    with pytest.raises((ValueError, RuntimeError), match="immutable|identity|creation|drift"):
        model.identity_metadata()
    assert model._creation_identity_digest == original_extra["snapshot_sha256"]
    assert model._creation_identity_snapshot == original_extra["snapshot"]


def test_live_component_bank_mutation_fails_before_ideal_full_model_forward(
    source_h2former: H2Former,
) -> None:
    model = OptoelectronicH2Former.from_h2former(source_h2former, _config())
    original_extra = copy.deepcopy(model.get_extra_state())

    with torch.no_grad():
        model.conv1.component_bank.add_(0.125)

    with pytest.raises((ValueError, RuntimeError), match="immutable|identity|creation|drift"):
        model(torch.zeros(1, 1, 512, 512))
    with pytest.raises((ValueError, RuntimeError), match="immutable|identity|creation|drift"):
        model.state_dict()
    with pytest.raises((ValueError, RuntimeError), match="immutable|identity|creation|drift"):
        model.identity_metadata()

    assert model._creation_identity_digest == original_extra["snapshot_sha256"]
    assert model._creation_identity_snapshot == original_extra["snapshot"]


def test_live_phase_aperture_mutation_fails_closed_for_physical_model(
    source_h2former: H2Former,
) -> None:
    config = _config(
        mode="physical",
        physical_config=OptoelectronicConfig(phase=_small_phase_config()),
    )
    model = OptoelectronicH2Former.from_h2former(source_h2former, config)
    original_extra = copy.deepcopy(model.get_extra_state())
    aperture = next(iter(model.conv1.phase_psfs.values())).aperture

    with torch.no_grad():
        aperture.add_(0.125)

    with pytest.raises((ValueError, RuntimeError), match="aperture|immutable|identity|creation|drift"):
        model(torch.zeros(1, 1, 512, 512))
    with pytest.raises((ValueError, RuntimeError), match="aperture|immutable|identity|creation|drift"):
        model.state_dict()
    with pytest.raises((ValueError, RuntimeError), match="aperture|immutable|identity|creation|drift"):
        model.identity_metadata()

    assert model._creation_identity_digest == original_extra["snapshot_sha256"]
    assert model._creation_identity_snapshot == original_extra["snapshot"]


def test_physical_apertures_are_explicit_creation_identity_tensors(
    source_h2former: H2Former,
) -> None:
    model = OptoelectronicH2Former.from_h2former(
        source_h2former,
        _config(
            mode="physical",
            physical_config=OptoelectronicConfig(phase=_small_phase_config()),
        ),
    )
    state = model.state_dict()
    aperture_keys = sorted(name for name in state if name.endswith(".aperture"))
    assert aperture_keys
    assert set(aperture_keys) == set(model._creation_immutable_state_identity).intersection(aperture_keys)
    for group in model.get_extra_state()["snapshot"]["groups"]:
        assert group["phase_apertures"]
        for aperture in group["phase_apertures"]:
            assert aperture["state_key"] in aperture_keys
            assert aperture["group_id"] == group["group_id"]
            assert aperture["route_id"]
            assert aperture["component_index"] >= 0
            assert aperture["sign"] in {"positive", "negative"}


def test_incoming_phase_aperture_tamper_is_rejected_before_copy_atomically(
    source_h2former: H2Former,
) -> None:
    config = _config(
        mode="physical",
        physical_config=OptoelectronicConfig(phase=_small_phase_config()),
    )
    source_model = OptoelectronicH2Former.from_h2former(source_h2former, config)
    target_model = OptoelectronicH2Former.from_h2former(source_h2former, config)
    incoming = _clone_state_dict(source_model.state_dict())
    aperture_key = next(name for name in incoming if name.endswith(".aperture"))
    assert isinstance(incoming[aperture_key], torch.Tensor)
    tampered = incoming[aperture_key].clone()  # type: ignore[union-attr]
    tampered.view(-1)[0] += 0.125
    incoming[aperture_key] = tampered
    before = _model_atomic_snapshot(target_model)

    with pytest.raises((ValueError, RuntimeError), match="aperture|immutable|digest|identity"):
        target_model.load_state_dict(incoming, strict=True)  # type: ignore[arg-type]

    _assert_model_unchanged(before, target_model)


@pytest.mark.parametrize("tamper", ["extra_wrapper_field", "missing_wrapper_field"])
def test_identity_extra_state_wrapper_schema_is_exact_for_set_and_load(
    source_h2former: H2Former,
    tamper: str,
) -> None:
    source_model = OptoelectronicH2Former.from_h2former(source_h2former, _config())
    target_model = OptoelectronicH2Former.from_h2former(source_h2former, _config())
    incoming = _clone_state_dict(source_model.state_dict())
    assert isinstance(incoming["_extra_state"], dict)
    extra = incoming["_extra_state"]
    if tamper == "extra_wrapper_field":
        extra["unrecognized_wrapper_field"] = True  # type: ignore[index]
    else:
        extra.pop("snapshot_sha256")  # type: ignore[union-attr]
    before = _model_atomic_snapshot(target_model)

    with pytest.raises((ValueError, RuntimeError), match="extra|schema|field|snapshot"):
        target_model.set_extra_state(copy.deepcopy(extra))  # type: ignore[arg-type]
    with pytest.raises((ValueError, RuntimeError), match="extra|schema|field|snapshot"):
        target_model.load_state_dict(incoming, strict=True)  # type: ignore[arg-type]

    _assert_model_unchanged(before, target_model)


@pytest.mark.parametrize(
    ("load_kwargs", "message"),
    [
        ({"assign": True}, "assign"),
        ({"strict": False}, "strict"),
    ],
)
def test_unsupported_load_modes_are_rejected_before_copy_atomically(
    source_h2former: H2Former,
    load_kwargs: dict[str, object],
    message: str,
) -> None:
    source_model = OptoelectronicH2Former.from_h2former(source_h2former, _config())
    target_model = OptoelectronicH2Former.from_h2former(source_h2former, _config())
    incoming = _clone_state_dict(source_model.state_dict())
    before = _model_atomic_snapshot(target_model)

    with pytest.raises((ValueError, RuntimeError), match=message):
        target_model.load_state_dict(incoming, **load_kwargs)  # type: ignore[arg-type]

    _assert_model_unchanged(before, target_model)
    assert not hasattr(target_model, "_identity_rebind_allowed")


@pytest.mark.parametrize("layout_kind", ["float64", "meta", "sparse", "quantized", "shape"])
def test_incoming_tensor_layout_is_rejected_before_copy_atomically(
    source_h2former: H2Former,
    layout_kind: str,
) -> None:
    source_model = OptoelectronicH2Former.from_h2former(source_h2former, _config())
    target_model = OptoelectronicH2Former.from_h2former(source_h2former, _config())
    incoming = _clone_state_dict(source_model.state_dict())
    key = "conv1.mixing"
    value = incoming[key]
    assert isinstance(value, torch.Tensor)
    if layout_kind == "float64":
        incoming[key] = value.double()
    elif layout_kind == "meta":
        incoming[key] = torch.empty_like(value, device="meta")
    elif layout_kind == "sparse":
        incoming[key] = value.to_sparse()
    elif layout_kind == "quantized":
        incoming[key] = torch.quantize_per_tensor(
            torch.ones_like(value), scale=0.1, zero_point=0, dtype=torch.qint8
        )
    else:
        incoming[key] = value.reshape(-1)
    before = _model_atomic_snapshot(target_model)

    with pytest.raises((ValueError, RuntimeError), match="tensor|dtype|meta|sparse|quantized|layout|shape"):
        target_model.load_state_dict(incoming, strict=True)  # type: ignore[arg-type]

    _assert_model_unchanged(before, target_model)


def test_legal_load_updates_only_live_baseline_versions_and_keeps_creation_identity(
    source_h2former: H2Former,
) -> None:
    config = _config(
        mode="physical",
        trainable_mixing=True,
        trainable_bias=True,
        physical_config=OptoelectronicConfig(phase=_small_phase_config()),
    )
    source_model = OptoelectronicH2Former.from_h2former(source_h2former, config)
    target_model = OptoelectronicH2Former.from_h2former(source_h2former, config)
    before_snapshot = copy.deepcopy(target_model._creation_identity_snapshot)
    before_digest = target_model._creation_identity_digest
    before_refs = {
        key: id(record[2])
        for key, record in target_model._creation_immutable_tensor_refs.items()
    }
    before_versions = {
        key: record[3]
        for key, record in target_model._creation_immutable_tensor_refs.items()
    }

    target_model.load_state_dict(_clone_state_dict(source_model.state_dict()), strict=True)

    after_records = target_model._creation_immutable_tensor_refs
    assert all(isinstance(record[3], int) for record in after_records.values())
    assert any(after_records[key][3] != before_versions[key] for key in before_versions)
    assert {key: id(record[2]) for key, record in after_records.items()} == before_refs
    assert target_model._creation_identity_snapshot == before_snapshot
    assert target_model._creation_identity_digest == before_digest
    target_model(torch.zeros(1, 1, 512, 512))
    target_model.state_dict()
    target_model.identity_metadata()


def test_full_rank_ideal_h2former_matches_source_once_in_eval_mode(
    source_h2former: H2Former,
) -> None:
    full_rank_config = OptoelectronicH2FormerConfig(
        rank_by_group={
            "stem7": 49,
            "patch2": 4,
            "patch4": 15,
            "patch8": 7,
            "patch16": 7,
        },
        variance_threshold=None,
        mode="ideal",
    )
    model = OptoelectronicH2Former.from_h2former(source_h2former, full_rank_config).eval()
    image = torch.randn(1, 1, 512, 512)

    with torch.inference_mode():
        expected = source_h2former(image)
        actual = model(image)

    assert isinstance(actual, torch.Tensor)
    assert actual.shape == (1, source_h2former.num_classes, 512, 512)
    assert torch.isfinite(actual).all()
    assert torch.allclose(actual, expected, atol=3e-4, rtol=3e-4)


def _small_physical_model(source_h2former: H2Former) -> OptoelectronicH2Former:
    return OptoelectronicH2Former.from_h2former(
        source_h2former,
        _config(
            mode="physical",
            physical_config=OptoelectronicConfig(
                phase=_small_phase_config(),
                min_throughput=1e-8,
                max_electronic_gain=1e8,
                min_split_fraction=0.0,
            ),
        ),
    ).eval()


def test_physical_entry_hot_path_uses_no_route_digest_and_keeps_phase_gradient(
    monkeypatch: pytest.MonkeyPatch,
    source_h2former: H2Former,
) -> None:
    model = _small_physical_model(source_h2former)
    calls = 0
    frontend = __import__(
        "standalone_nnunet2d.models.optoelectronic_frontend",
        fromlist=["optoelectronic_frontend"],
    )
    original = frontend._active_route_identity_snapshot

    def recording(route: object) -> object:
        nonlocal calls
        calls += 1
        return original(route)

    monkeypatch.setattr(frontend, "_active_route_identity_snapshot", recording)
    output = model.conv1(torch.randn(1, 1, 32, 32))
    output.square().mean().backward()

    assert torch.isfinite(output).all()
    assert calls == 0
    phase_gradients = [phase_psf.theta.grad for phase_psf in model.conv1.phase_psfs.values()]
    assert phase_gradients
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in phase_gradients)


def test_phase_theta_update_is_legal_and_round_trips_without_hot_path_hashing(
    monkeypatch: pytest.MonkeyPatch,
    source_h2former: H2Former,
) -> None:
    model = _small_physical_model(source_h2former)
    phase_psf = next(iter(model.conv1.phase_psfs.values()))
    with torch.no_grad():
        phase_psf.theta.add_(0.05)

    calls = 0
    frontend = __import__(
        "standalone_nnunet2d.models.optoelectronic_frontend",
        fromlist=["optoelectronic_frontend"],
    )
    original = frontend._active_route_identity_snapshot

    def recording(route: object) -> object:
        nonlocal calls
        calls += 1
        return original(route)

    monkeypatch.setattr(frontend, "_active_route_identity_snapshot", recording)
    output = model.conv1(torch.randn(1, 1, 32, 32))
    output.square().mean().backward()

    assert torch.isfinite(output).all()
    assert phase_psf.theta.grad is not None
    assert torch.isfinite(phase_psf.theta.grad).all()
    assert calls == 0

    state = _clone_state_dict(model.state_dict())
    restored = _small_physical_model(source_h2former)
    restored.load_state_dict(state)
    restored.state_dict()
    restored.identity_metadata()


@pytest.mark.parametrize(
    ("entry_index", "drift"),
    [
        (0, "replace"),
        (1, "max_electronic_gain"),
        (2, "rho"),
        (3, "replace"),
        (4, "max_electronic_gain"),
    ],
)
def test_entry_physical_config_drift_fails_closed_on_all_model_surfaces(
    source_h2former: H2Former,
    entry_index: int,
    drift: str,
) -> None:
    model = _small_physical_model(source_h2former)
    entry = model._entry_modules()[entry_index]
    if drift == "replace":
        object.__setattr__(
            entry,
            "physical_config",
            replace(entry.physical_config, max_electronic_gain=entry.physical_config.max_electronic_gain * 2),
        )
    elif drift == "max_electronic_gain":
        object.__setattr__(
            entry.physical_config,
            "max_electronic_gain",
            entry.physical_config.max_electronic_gain * 2,
        )
    else:
        object.__setattr__(entry.physical_config, "rho", entry.physical_config.rho + 0.25)

    with pytest.raises((ValueError, RuntimeError), match="config|identity|drift|creation"):
        entry(torch.zeros(1, 1, 32, 32))
    with pytest.raises((ValueError, RuntimeError), match="config|identity|drift|creation"):
        model(torch.zeros(1, 1, 512, 512))
    with pytest.raises((ValueError, RuntimeError), match="config|identity|drift|creation"):
        model.state_dict()
    with pytest.raises((ValueError, RuntimeError), match="config|identity|drift|creation"):
        model.identity_metadata()


def test_entry_pca_metadata_rejects_physical_config_drift_without_payload(
    source_h2former: H2Former,
) -> None:
    model = _small_physical_model(source_h2former)
    entry = model.conv1
    object.__setattr__(entry.physical_config, "rho", entry.physical_config.rho + 0.25)

    with pytest.raises((ValueError, RuntimeError), match="config|identity|drift|creation"):
        entry.pca_metadata()


def test_entry_ledger_replacement_fails_closed_on_direct_and_model_surfaces(
    source_h2former: H2Former,
) -> None:
    model = _small_physical_model(source_h2former)
    entry = model.conv1
    alternate = build_exposure_ledger(
        {entry.group_id: entry.component_bank.detach()},
        policy="route_sequential",
    )
    entry.exposure_ledger = alternate

    with pytest.raises((ValueError, RuntimeError), match="ledger|identity|drift|creation|binding"):
        entry(torch.zeros(1, 1, 32, 32))
    with pytest.raises((ValueError, RuntimeError), match="ledger|identity|drift|creation|binding"):
        model(torch.zeros(1, 1, 512, 512))
    with pytest.raises((ValueError, RuntimeError), match="ledger|identity|drift|creation|binding"):
        model.state_dict()
    with pytest.raises((ValueError, RuntimeError), match="ledger|identity|drift|creation|binding"):
        model.identity_metadata()


def test_full_identity_surfaces_reject_replaced_canonical_routes_tuple(
    source_h2former: H2Former,
) -> None:
    model = _small_physical_model(source_h2former)
    entry = model.conv1
    assert torch.isfinite(entry(torch.zeros(1, 1, 32, 32))).all()
    model.state_dict()
    model.identity_metadata()

    ledger = model.exposure_ledger
    original_routes = ledger.canonical_routes
    replacement_routes = tuple(route for route in original_routes)
    assert replacement_routes is not original_routes
    object.__setattr__(ledger, "canonical_routes", replacement_routes)

    with pytest.raises((ValueError, RuntimeError), match="ledger|route|identity|drift|binding"):
        entry(torch.zeros(1, 1, 32, 32))
    with pytest.raises((ValueError, RuntimeError), match="ledger|route|identity|drift|binding"):
        model.state_dict()
    with pytest.raises((ValueError, RuntimeError), match="ledger|route|identity|drift|binding"):
        model.identity_metadata()


def test_phase_psf_module_replacement_fails_closed_on_all_identity_surfaces(
    source_h2former: H2Former,
) -> None:
    model = _small_physical_model(source_h2former)
    entry = model.conv1
    phase_key = next(iter(entry.phase_psfs))
    entry.phase_psfs[phase_key] = copy.deepcopy(entry.phase_psfs[phase_key])

    with pytest.raises((ValueError, RuntimeError), match="phase|PSF|module|identity|drift|binding"):
        entry(torch.zeros(1, 1, 32, 32))
    with pytest.raises((ValueError, RuntimeError), match="phase|PSF|module|identity|drift|binding"):
        model(torch.zeros(1, 1, 512, 512))
    with pytest.raises((ValueError, RuntimeError), match="phase|PSF|module|identity|drift|binding"):
        entry.state_dict()
    with pytest.raises((ValueError, RuntimeError), match="phase|PSF|module|identity|drift|binding"):
        model.state_dict()
    with pytest.raises((ValueError, RuntimeError), match="phase|PSF|module|identity|drift|binding"):
        entry.pca_metadata()
    with pytest.raises((ValueError, RuntimeError), match="phase|PSF|module|identity|drift|binding"):
        model.identity_metadata()


def test_entry_route_object_replacement_fails_closed_without_content_hash_bypass(
    source_h2former: H2Former,
) -> None:
    model = _small_physical_model(source_h2former)
    entry = model.conv1
    component_index, routes = next(iter(entry._routes_by_component.items()))
    sign, route = next(iter(routes.items()))
    entry._routes_by_component[component_index][sign] = replace(
        route,
        lobe=route.lobe.detach().clone(),
    )

    with pytest.raises((ValueError, RuntimeError), match="route|binding|identity|drift|creation"):
        entry(torch.zeros(1, 1, 32, 32))
    with pytest.raises((ValueError, RuntimeError), match="route|binding|identity|drift|creation"):
        model(torch.zeros(1, 1, 512, 512))
    with pytest.raises((ValueError, RuntimeError), match="route|binding|identity|drift|creation"):
        model.state_dict()
    with pytest.raises((ValueError, RuntimeError), match="route|binding|identity|drift|creation"):
        model.identity_metadata()
