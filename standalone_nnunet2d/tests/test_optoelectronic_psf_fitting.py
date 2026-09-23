from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from standalone_nnunet2d.engine.checkpoint import save_checkpoint
from standalone_nnunet2d.engine.optoelectronic_initialization import (
    INITIALIZATION_FILENAME,
    build_optoelectronic_config,
    create_optoelectronic_initialization,
    load_optoelectronic_initialization,
)
from standalone_nnunet2d.engine import optoelectronic_psf_fitting as fitting
from standalone_nnunet2d.models.h2former import H2Former
from standalone_nnunet2d.models.optoelectronic_frontend import (
    ActiveRoute,
    OptoelectronicConfig,
    PhaseOnlyPSF,
    PhasePSFConfig,
)

GROUPS = ("stem7", "patch2", "patch4", "patch8", "patch16")


def _init_config(mode: str = "physical") -> dict[str, object]:
    return {
        "mode": mode,
        "rank_by_group": {group: 1 for group in GROUPS},
        "variance_threshold": None,
        "trainable_phase": False,
        "trainable_mixing": True,
        "trainable_bias": True,
        "physical_config": {
            "phase": {
                "phase_grid_size": 16,
                "phase_pitch_m": 8e-6,
                "aperture_diameter_m": 96e-6,
            },
            "min_throughput": 1e-8,
            "max_electronic_gain": 1e8,
            "min_split_fraction": 0.0,
            "support_error_mode": "report_only",
        },
    }


def _fit_config(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "steps": 4,
        "learning_rate": 0.03,
        "seed": 41,
        "phase_init_std": 0.0,
        "shape_weight": 1.0,
        "throughput_weight": 0.05,
        "gain_weight": 0.001,
        "device": "cpu",
        "map_location": "cpu",
    }
    values.update(overrides)
    return values


@pytest.fixture(scope="module")
def synthetic_artifacts(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    root = tmp_path_factory.mktemp("synthetic-optoelectronic-psf-fit")
    source_path = root / "synthetic-h2former.pth"
    init_root = root / "initialization"
    rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(3817)
        source = H2Former(in_channels=1, num_classes=2, image_size=512).eval()
        save_checkpoint(
            source,
            None,
            source_path,
            {"model_name": "h2former", "supervision_mode": "single_output"},
            allowed_root=root,
        )
        del source
    finally:
        torch.random.set_rng_state(rng_state)
    initialized = create_optoelectronic_initialization(
        source_path,
        _init_config(),
        init_root,
    )
    return initialized.artifact_path, source_path, root


@pytest.fixture(scope="module")
def fitted_artifacts(synthetic_artifacts, tmp_path_factory: pytest.TempPathFactory):
    initialization_path, source_path, _ = synthetic_artifacts
    output_root = tmp_path_factory.mktemp("synthetic-optoelectronic-psf-fit-output")
    result = fitting.fit_optoelectronic_h2former_psf(
        initialization_path,
        source_path,
        fitting.OptoelectronicPSFFitConfig.from_mapping(_fit_config()),
        output_root,
    )
    return result, initialization_path, source_path


def _load_payload(path: Path) -> dict[str, object]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _copy_payload(source: Path, destination: Path, edit) -> Path:
    payload = _load_payload(source)
    edit(payload)
    torch.save(payload, destination)
    return destination


def test_config_requires_strict_values_and_at_least_one_loss_weight() -> None:
    with pytest.raises(ValueError):
        fitting.OptoelectronicPSFFitConfig.from_mapping(_fit_config(steps=True))
    with pytest.raises(ValueError):
        fitting.OptoelectronicPSFFitConfig.from_mapping(_fit_config(learning_rate=float("inf")))
    with pytest.raises(ValueError):
        fitting.OptoelectronicPSFFitConfig.from_mapping(_fit_config(seed=-1))
    with pytest.raises(ValueError):
        fitting.OptoelectronicPSFFitConfig.from_mapping(
            _fit_config(shape_weight=0.0, throughput_weight=0.0, gain_weight=0.0)
        )
    with pytest.raises(ValueError):
        fitting.OptoelectronicPSFFitConfig.from_mapping(_fit_config(unexpected=True))


def test_digital_lobe_conversion_flips_once_and_normalizes_target(monkeypatch) -> None:
    asymmetric = torch.arange(1, 10, dtype=torch.float32).reshape(1, 3, 3)
    route = ActiveRoute("stem7:component0:positive", "stem7", 0, "positive", asymmetric, 0.7)
    original = fitting.physical_psf_from_digital_lobe
    calls: list[torch.Tensor] = []

    def counted(value: torch.Tensor) -> torch.Tensor:
        calls.append(value)
        return original(value)

    monkeypatch.setattr(fitting, "physical_psf_from_digital_lobe", counted)
    target = fitting._physical_target_for_route(route)
    assert len(calls) == 1
    assert calls[0] is asymmetric
    assert torch.equal(target, asymmetric.flip((-2, -1)).squeeze(0) / asymmetric.sum())
    assert target.sum().item() == pytest.approx(1.0)


def test_route_objective_is_finite_tensor_and_differentiates_to_theta() -> None:
    phase = PhaseOnlyPSF(PhasePSFConfig(phase_grid_size=8, aperture_diameter_m=40e-6))
    target = torch.tensor(
        [[0.0, 0.1, 0.0], [0.2, 0.4, 0.1], [0.0, 0.1, 0.1]],
        dtype=torch.float32,
    )
    target = target / target.sum()
    route = ActiveRoute("stem7:component0:positive", "stem7", 0, "positive", target, 0.5)
    config = fitting.OptoelectronicPSFFitConfig.from_mapping(_fit_config())
    objective, terms = fitting._differentiable_route_objective(
        phase(), target, support=3, route=route, rho=1.2, beta=0.25,
        physical_config=OptoelectronicConfig(
            phase=phase.config, min_throughput=1e-5, max_electronic_gain=1e5
        ),
        fit_config=config,
    )
    assert isinstance(objective, torch.Tensor)
    assert objective.ndim == 0 and torch.isfinite(objective)
    assert all(isinstance(value, torch.Tensor) and torch.isfinite(value) for value in terms.values())
    objective.backward()
    assert phase.theta.grad is not None
    assert torch.isfinite(phase.theta.grad).all()
    assert phase.theta.grad.abs().sum().item() > 0


def test_support_sizes_are_deduplicated_and_keep_outside_energy() -> None:
    assert fitting._support_sizes(7, 16) == (7, 14, 16)
    assert fitting._support_sizes(8, 16) == (8, 16)
    assert fitting._support_sizes(16, 16) == (16,)
    phase = PhaseOnlyPSF(PhasePSFConfig(phase_grid_size=16, aperture_diameter_m=96e-6))
    sweep = fitting._support_sweep_for_psf(
        phase(), 7, seed=9, physical_config=OptoelectronicConfig(phase=phase.config)
    )
    reports = sweep.as_dict()["reports"]
    assert tuple(item["support"] for item in reports) == (7, 14, 16)
    assert reports[0]["leakage"] > 0
    assert reports[-1]["tau"] == pytest.approx(1.0, abs=1e-6)
    assert "random_signed_dual_rail_response_error" in reports[0]
    assert "internal_region_error" in reports[0]
    assert "boundary_region_error" in reports[0]
    assert "full_image_response_error" in reports[0]


def test_quality_and_final_physical_constraints_fail_closed() -> None:
    config = fitting.OptoelectronicPSFFitConfig.from_mapping(
        _fit_config(max_normalized_mse=0.01, quality_threshold_mode="fail_closed")
    )
    with pytest.raises(ValueError, match="quality"):
        fitting._check_quality_thresholds({"normalized_mse": 0.1, "cosine_similarity": 0.9}, config)
    physical = OptoelectronicConfig(
        phase=PhasePSFConfig(phase_grid_size=16, aperture_diameter_m=96e-6),
        min_throughput=0.8,
        max_electronic_gain=1.5,
    )
    with pytest.raises(ValueError, match="throughput"):
        fitting._check_final_physical_limits({"tau": 0.7, "gain": 1.0}, physical)
    with pytest.raises(ValueError, match="gain"):
        fitting._check_final_physical_limits({"tau": 0.9, "gain": 2.0}, physical)
    report_only = fitting.OptoelectronicPSFFitConfig.from_mapping(
        _fit_config(max_normalized_mse=0.01, quality_threshold_mode="report_only")
    )
    fitting._check_quality_thresholds(
        {"normalized_mse": 0.1, "cosine_similarity": 0.9}, report_only
    )


def test_fit_covers_all_groups_only_changes_theta_and_reports_raw_gain(
    fitted_artifacts,
) -> None:
    result, initialization_path, _ = fitted_artifacts
    report = result.report
    artifact = _load_payload(result.artifact_path)
    initialization = _load_payload(initialization_path)
    assert tuple(report["group_order"]) == GROUPS
    assert report["route_count"] == len(report["route_order"])
    routes = report["routes"]
    assert tuple(routes) == tuple(report["route_order"])
    assert {entry["group_id"] for entry in routes.values()} == set(GROUPS)
    assert artifact["optimizer_state_dict"] is None
    assert artifact["metadata"]["resume_eligible"] is False
    assert artifact["metadata"]["fit_only"] is True
    assert artifact["metadata"]["evidence_class"] == "synthetic_engineering_psf_fit"
    assert artifact["metadata"]["source_initialization_sha256"] == hashlib.sha256(
        initialization_path.read_bytes()
    ).hexdigest()
    before = initialization["model_state_dict"]
    after = artifact["model_state_dict"]
    assert set(before) == set(after)
    changed = {key for key in before if (not torch.equal(before[key], after[key]) if isinstance(before[key], torch.Tensor) else before[key] != after[key])}
    assert changed
    assert all(".phase_psfs." in key and key.endswith(".theta") for key in changed)
    assert changed <= set(artifact["metadata"]["theta_digests"])
    for route in routes.values():
        assert set(("initial", "best", "final")) <= set(route)
        final = route["final"]
        tau = final["tau"]
        expected_gain = route["alpha"] / (route["rho"] * route["beta"] * tau)
        assert final["gain"] == pytest.approx(expected_gain, rel=1e-6)
        assert final["leakage"] == pytest.approx(1.0 - tau, abs=1e-6)
        assert final["full_psf_sum"] == pytest.approx(1.0, abs=1e-6)
        assert final["full_psf_min"] >= 0.0
        assert route["target_conversion"] == "digital_correlation_lobe_to_physical_psf_once"
        assert route["support_sweep"]["reference_support"] == 16
    assert any("support-external energy remains" in item for item in report["limitations"])
    assert report["fit_status"] == "synthetic_engineering_psf_fit"
    assert report["claims"]["hardware_validated"] is False
    assert report["claims"]["medical_performance"] is False


def test_seed_reproducibility_and_zero_std_preserve_initial_theta(
    fitted_artifacts, synthetic_artifacts, tmp_path_factory
) -> None:
    result, initialization_path, source_path = fitted_artifacts
    metadata = _load_payload(result.artifact_path)["metadata"]
    init_state = _load_payload(initialization_path)["model_state_dict"]
    for key, digest in metadata["theta_digests"].items():
        assert digest["initial_sha256"] == fitting._tensor_sha256(init_state[key])
    rng_before = torch.random.get_rng_state().clone()
    second = fitting.fit_optoelectronic_h2former_psf(
        initialization_path,
        source_path,
        fitting.OptoelectronicPSFFitConfig.from_mapping(_fit_config()),
        tmp_path_factory.mktemp("same-seed-fit"),
    )
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert _load_payload(result.artifact_path)["metadata"]["theta_digests"] == _load_payload(
        second.artifact_path
    )["metadata"]["theta_digests"]


def test_nonzero_phase_perturbation_is_seeded_and_distinguishable() -> None:
    base = torch.zeros(8, 8)
    a = fitting._perturb_initial_theta(base, std=0.2, seed=31)
    b = fitting._perturb_initial_theta(base, std=0.2, seed=32)
    again = fitting._perturb_initial_theta(base, std=0.2, seed=31)
    assert torch.equal(a, again)
    assert not torch.equal(a, b)
    assert torch.equal(base, torch.zeros_like(base))


def test_fit_artifact_strict_round_trip_and_source_identity(
    fitted_artifacts,
) -> None:
    result, initialization_path, source_path = fitted_artifacts
    model = fitting.load_optoelectronic_psf_fit(
        result.artifact_path, initialization_path, source_path
    )
    assert model.opto_config.mode == "physical"
    payload = _load_payload(result.artifact_path)
    metadata = payload["metadata"]
    assert payload["artifact_type"] == fitting.ARTIFACT_TYPE
    assert payload["artifact_version"] == fitting.ARTIFACT_VERSION
    assert metadata["source_checkpoint_sha256"] == hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert metadata["creation_identity_digest"]
    assert metadata["initialization_metadata_sha256"]
    assert metadata["global_ledger_identity"]["sha256"]
    assert result.report["report_schema"] == fitting.REPORT_SCHEMA
    assert result.report["report_version"] == fitting.REPORT_VERSION


def test_ideal_initialization_is_rejected(synthetic_artifacts, tmp_path: Path) -> None:
    _, source_path, _ = synthetic_artifacts
    ideal = create_optoelectronic_initialization(
        source_path, _init_config(mode="ideal"), tmp_path / "ideal-init"
    )
    with pytest.raises(ValueError, match="physical"):
        fitting.fit_optoelectronic_h2former_psf(
            ideal.artifact_path,
            source_path,
            fitting.OptoelectronicPSFFitConfig.from_mapping(_fit_config()),
            tmp_path / "rejected-fit",
        )


def test_loader_rejects_non_theta_tamper_before_target_copy(
    fitted_artifacts, monkeypatch, tmp_path: Path
) -> None:
    result, initialization_path, source_path = fitted_artifacts
    target = load_optoelectronic_initialization(initialization_path, source_path)
    before = {key: value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
              for key, value in target.state_dict().items()}
    calls: list[object] = []
    original = target.load_state_dict

    def tracked(*args, **kwargs):
        calls.append(args[0] if args else None)
        return original(*args, **kwargs)

    target.load_state_dict = tracked

    def corrupt(payload):
        state = payload["model_state_dict"]
        key = next(name for name in state if name.endswith(".mixing"))
        state[key] = state[key].clone()
        state[key].view(-1)[0] += 1

    bad = _copy_payload(result.artifact_path, tmp_path / "tampered.pth", corrupt)
    with pytest.raises(ValueError, match="non-theta|initialization"):
        fitting.load_optoelectronic_psf_fit(
            bad, initialization_path, source_path, target_model=target
        )
    assert calls == []
    for key, value in target.state_dict().items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, before[key])
        else:
            assert value == before[key]


def test_loader_rejects_malformed_theta_and_metadata(
    fitted_artifacts, tmp_path: Path
) -> None:
    result, initialization_path, source_path = fitted_artifacts

    def bad_theta(payload):
        state = payload["model_state_dict"]
        key = next(name for name in state if name.endswith(".theta"))
        state[key] = torch.full_like(state[key], float("nan"))

    bad = _copy_payload(result.artifact_path, tmp_path / "bad-theta.pth", bad_theta)
    with pytest.raises(ValueError, match="theta|digest|finite"):
        fitting.load_optoelectronic_psf_fit(bad, initialization_path, source_path)

    def bad_metadata(payload):
        payload["metadata"]["fit_config"]["steps"] += 1

    bad = _copy_payload(result.artifact_path, tmp_path / "bad-metadata.pth", bad_metadata)
    with pytest.raises(ValueError, match="metadata|digest|integrity"):
        fitting.load_optoelectronic_psf_fit(bad, initialization_path, source_path)


def test_loader_binds_initialization_and_source_content_hashes(
    fitted_artifacts, tmp_path: Path
) -> None:
    result, initialization_path, source_path = fitted_artifacts
    changed_init = tmp_path / "changed-init.pth"
    changed_init.write_bytes(initialization_path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="initialization.*SHA|SHA256"):
        fitting.load_optoelectronic_psf_fit(result.artifact_path, changed_init, source_path)
    changed_source = tmp_path / "changed-source.pth"
    changed_source.write_bytes(source_path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="source.*SHA|SHA256"):
        fitting.load_optoelectronic_psf_fit(result.artifact_path, initialization_path, changed_source)


def test_report_tamper_and_atomic_no_overwrite_rollback(
    fitted_artifacts, monkeypatch, tmp_path: Path
) -> None:
    result, initialization_path, source_path = fitted_artifacts
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    report["routes"][report["route_order"][0]]["final"]["tau"] += 0.01
    bad_report = tmp_path / "tampered-report.json"
    bad_report.write_text(json.dumps(report), encoding="utf-8")
    bad_artifact = tmp_path / "fit-copy.pth"
    bad_artifact.write_bytes(result.artifact_path.read_bytes())
    with pytest.raises(ValueError, match="report|metric|metadata"):
        fitting.load_optoelectronic_psf_fit(
            bad_artifact, initialization_path, source_path, report_path=bad_report
        )

    root = tmp_path / "rollback"
    root.mkdir()
    from standalone_nnunet2d.engine import optoelectronic_initialization as initialization
    original_publish = initialization._publish_without_overwrite
    calls = 0

    def fail_second(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected report publication failure")
        original_publish(source, destination)

    monkeypatch.setattr(initialization, "_publish_without_overwrite", fail_second)
    with pytest.raises(OSError, match="injected"):
        fitting._publish_fit_artifacts(
            {"payload": True}, {"report": True}, output_root=root
        )
    assert list(root.iterdir()) == []


def test_output_files_are_not_overwritten(fitted_artifacts) -> None:
    result, initialization_path, source_path = fitted_artifacts
    with pytest.raises(FileExistsError):
        fitting.fit_optoelectronic_h2former_psf(
            initialization_path,
            source_path,
            fitting.OptoelectronicPSFFitConfig.from_mapping(_fit_config()),
            result.artifact_path.parent,
        )



class _DictSubclass(dict):
    pass


class _ListSubclass(list):
    pass


class _StringSubclass(str):
    pass


class _IntegerSubclass(int):
    pass


class _MaskedExtraState(dict):
    items_calls = 0
    get_calls = 0

    def __init__(self, backing: dict[str, object], visible: dict[str, object]) -> None:
        super().__init__(backing)
        self.visible = visible

    def items(self):
        type(self).items_calls += 1
        return self.visible.items()

    def get(self, key, default=None):
        type(self).get_calls += 1
        return self.visible.get(key, default)


class _EqualityTrapDict(dict):
    equality_calls = 0

    def __eq__(self, other):
        type(self).equality_calls += 1
        return True


class _ObservedMetadataDict(dict):
    method_calls = 0

    def __iter__(self):
        type(self).method_calls += 1
        return dict.__iter__(self)

    def items(self):
        type(self).method_calls += 1
        return dict.items(self)

    def get(self, key, default=None):
        type(self).method_calls += 1
        return dict.get(self, key, default)

    def __eq__(self, other):
        type(self).method_calls += 1
        return True


def test_loader_rejects_hidden_extra_state_backing_mutation_before_target_copy(
    fitted_artifacts, monkeypatch, tmp_path: Path
) -> None:
    result, initialization_path, source_path = fitted_artifacts
    target = load_optoelectronic_initialization(initialization_path, source_path)
    before_state = {
        key: value.detach().clone() if type(value) is torch.Tensor else copy.deepcopy(value)
        for key, value in target.state_dict().items()
    }
    before_ledger = copy.deepcopy(target.exposure_ledger.as_dict())
    before_identity = copy.deepcopy(target.identity_metadata())
    load_calls: list[object] = []
    original_load = target.load_state_dict

    def tracked_load(*args, **kwargs):
        load_calls.append(args[0] if args else None)
        return original_load(*args, **kwargs)

    target.load_state_dict = tracked_load

    def corrupt(payload):
        state = payload["model_state_dict"]
        key = next(name for name in state if name == "_extra_state" or name.endswith("._extra_state"))
        backing = copy.deepcopy(state[key])
        visible = copy.deepcopy(state[key])
        assert type(backing) is dict
        assert type(visible) is dict
        assert type(backing["version"]) is int
        backing["version"] = "invalid-version"
        state[key] = _MaskedExtraState(backing, visible)

    bad = _copy_payload(result.artifact_path, tmp_path / "hidden-extra-state.pth", corrupt)
    _MaskedExtraState.items_calls = 0
    _MaskedExtraState.get_calls = 0

    with pytest.raises(ValueError, match="non-theta|state|JSON"):
        fitting.load_optoelectronic_psf_fit(
            bad, initialization_path, source_path, target_model=target, report_path=result.report_path
        )

    assert load_calls == []
    assert _MaskedExtraState.items_calls == 0
    assert _MaskedExtraState.get_calls == 0
    after_state = target.state_dict()
    assert set(after_state) == set(before_state)
    for key, before in before_state.items():
        assert fitting._state_value_equal(before, after_state[key])
    assert target.exposure_ledger.as_dict() == before_ledger
    assert target.identity_metadata() == before_identity


def test_state_json_comparison_and_digest_reject_exact_type_subclasses() -> None:
    cases = (
        ({"value": 1}, _DictSubclass({"value": 1})),
        ([1, "value"], _ListSubclass([1, "value"])),
        ("value", _StringSubclass("value")),
        (7, _IntegerSubclass(7)),
    )
    for expected, actual in cases:
        assert fitting._state_value_equal(expected, actual) is False
        with pytest.raises(ValueError, match="JSON tree"):
            fitting._state_digest({"extra": actual}, exclude_theta=False)


def test_state_equality_never_calls_custom_json_equality() -> None:
    value = _EqualityTrapDict({"version": 1})
    _EqualityTrapDict.equality_calls = 0

    assert fitting._state_value_equal({"version": 1}, value) is False
    assert _EqualityTrapDict.equality_calls == 0
    with pytest.raises(ValueError, match="JSON tree"):
        fitting._state_digest({"extra": value}, exclude_theta=False)
    assert _EqualityTrapDict.equality_calls == 0


def test_state_json_preserves_bool_int_and_finite_float_rules() -> None:
    assert fitting._state_value_equal(True, 1) is False
    assert fitting._state_value_equal(1, True) is False
    assert fitting._state_value_equal(1.5, 1.5) is True
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="JSON tree"):
            fitting._state_digest({"extra": value}, exclude_theta=False)


def test_fit_metadata_rejects_dict_subclasses_before_iteration(
    fitted_artifacts,
) -> None:
    result, _, _ = fitted_artifacts
    metadata = _load_payload(result.artifact_path)["metadata"]
    untrusted = _ObservedMetadataDict(metadata)
    _ObservedMetadataDict.method_calls = 0

    with pytest.raises(ValueError, match="metadata|dict|JSON"):
        fitting._validate_fit_metadata(untrusted)

    assert _ObservedMetadataDict.method_calls == 0


def test_fit_metadata_rejects_nested_scalar_subclasses(fitted_artifacts) -> None:
    result, _, _ = fitted_artifacts
    metadata = copy.deepcopy(_load_payload(result.artifact_path)["metadata"])
    metadata["seed"] = _IntegerSubclass(metadata["seed"])

    with pytest.raises(ValueError, match="JSON tree"):
        fitting._validate_fit_metadata(metadata)


import argparse


@pytest.mark.parametrize(
    ("config_text", "duplicate_key"),
    (
        ('{"steps": 1, "steps": 2}', "steps"),
        ('{"outer": {"steps": 1, "steps": 2}}', "steps"),
    ),
)
def test_cli_rejects_duplicate_json_keys_before_config_or_writes(
    config_text: str,
    duplicate_key: str,
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    from standalone_nnunet2d.tools import fit_optoelectronic_h2former_psf as cli

    config_path = tmp_path / "duplicate-config.json"
    config_path.write_text(config_text, encoding="utf-8")
    output_root = tmp_path / "must-not-be-created"
    fit_calls: list[object] = []
    mapping_calls: list[object] = []
    original_from_mapping = cli.OptoelectronicPSFFitConfig.from_mapping

    def tracked_from_mapping(cls, raw):
        mapping_calls.append(raw)
        return original_from_mapping(raw)

    monkeypatch.setattr(
        cli.OptoelectronicPSFFitConfig,
        "from_mapping",
        classmethod(tracked_from_mapping),
    )
    monkeypatch.setattr(
        cli,
        "fit_optoelectronic_h2former_psf",
        lambda *args, **kwargs: fit_calls.append(args),
    )
    argv = [
        "--initialization-artifact", str(tmp_path / "init.pth"),
        "--source-checkpoint", str(tmp_path / "source.pth"),
        "--fit-config", str(config_path),
        "--output-root", str(output_root),
    ]

    with pytest.raises(SystemExit) as error:
        cli.main(argv)

    assert error.value.code == 2
    assert mapping_calls == []
    assert fit_calls == []
    assert not output_root.exists()
    stderr = capsys.readouterr().err
    assert "duplicate" in stderr.lower()
    assert duplicate_key in stderr


def test_cli_accepts_strict_json_and_device_override(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    from types import SimpleNamespace
    from standalone_nnunet2d.tools import fit_optoelectronic_h2former_psf as cli

    config_path = tmp_path / "valid-config.json"
    config_path.write_text(
        json.dumps(_fit_config(device="cuda:0"), allow_nan=False),
        encoding="utf-8",
    )
    output_root = tmp_path / "valid-output"
    fit_calls: list[tuple[object, ...]] = []
    report = {
        "source_initialization_sha256": "i" * 64,
        "source_checkpoint_sha256": "s" * 64,
        "route_count": 1,
        "aggregate_metrics": {},
        "fit_status": "synthetic_engineering_psf_fit",
    }

    def fake_fit(initialization, source, config, destination):
        fit_calls.append((initialization, source, config, destination))
        return SimpleNamespace(
            artifact_path=tmp_path / "fit.pth",
            report_path=tmp_path / "report.json",
            report=report,
        )

    monkeypatch.setattr(cli, "fit_optoelectronic_h2former_psf", fake_fit)
    argv = [
        "--initialization-artifact", str(tmp_path / "init.pth"),
        "--source-checkpoint", str(tmp_path / "source.pth"),
        "--fit-config", str(config_path),
        "--output-root", str(output_root),
        "--device", "cpu",
    ]

    assert cli.main(argv) == 0
    assert len(fit_calls) == 1
    assert fit_calls[0][2].device == "cpu"
    assert fit_calls[0][2].steps == _fit_config()["steps"]
    assert fit_calls[0][3] == output_root
    assert "synthetic_engineering_psf_fit" in capsys.readouterr().out


@pytest.mark.parametrize("constant", ("NaN", "Infinity", "-Infinity"))
def test_cli_continues_to_reject_nonfinite_json_constants(
    constant: str, monkeypatch, tmp_path: Path
) -> None:
    from standalone_nnunet2d.tools import fit_optoelectronic_h2former_psf as cli

    config_path = tmp_path / "nonfinite-config.json"
    config_path.write_text('{"steps": ' + constant + '}', encoding="utf-8")
    fit_calls: list[object] = []
    monkeypatch.setattr(
        cli,
        "fit_optoelectronic_h2former_psf",
        lambda *args, **kwargs: fit_calls.append(args),
    )

    with pytest.raises(SystemExit) as error:
        cli.main([
            "--initialization-artifact", str(tmp_path / "init.pth"),
            "--source-checkpoint", str(tmp_path / "source.pth"),
            "--fit-config", str(config_path),
            "--output-root", str(tmp_path / "output"),
        ])

    assert error.value.code == 2
    assert fit_calls == []
