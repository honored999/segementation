from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from standalone_nnunet2d.engine.checkpoint import save_checkpoint
from standalone_nnunet2d.engine.optoelectronic_initialization import (
    ARTIFACT_TYPE,
    ARTIFACT_VERSION,
    INITIALIZATION_FILENAME,
    REPORT_FILENAME,
    build_optoelectronic_config,
    create_optoelectronic_initialization,
    load_optoelectronic_initialization,
)
from standalone_nnunet2d.models.h2former import H2Former


GROUPS = ("stem7", "patch2", "patch4", "patch8", "patch16")
class _AlwaysEqualString(str):
    equality_calls = 0

    def __eq__(self, other: object) -> bool:
        type(self).equality_calls += 1
        return True

    __hash__ = str.__hash__


class _StringSubclass(str):
    pass


class _DictSubclass(dict):
    pass


class _ListSubclass(list):
    pass


def _small_physical_config() -> dict[str, object]:
    return {
        "phase": {
            "phase_grid_size": 16,
            "phase_pitch_m": 8e-6,
            "aperture_diameter_m": 96e-6,
        },
        "min_throughput": 1e-8,
        "max_electronic_gain": 1e8,
        "min_split_fraction": 0.0,
    }


def _config(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {"physical_config": _small_physical_config()}
    values.update(overrides)
    return values


def _explicit_rank_config(rank: int = 0, **overrides: object) -> dict[str, object]:
    values = _config(
        rank_by_group={group_id: rank for group_id in GROUPS},
        variance_threshold=None,
    )
    values.update(overrides)
    return values


def _write_source_payload(path: Path, metadata: object, state: object | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_state_dict": {} if state is None else state,
            "optimizer_state_dict": None,
            "metadata": metadata,
        },
        path,
    )
    return path


def _load_payload(path: Path) -> dict[str, object]:
    return torch.load(path, map_location="cpu", weights_only=False)
def _refresh_extra_state_snapshot_digest(extra_state: dict[str, object]) -> None:
    snapshot = extra_state["snapshot"]
    unsigned_snapshot = {
        key: value for key, value in snapshot.items() if key != "snapshot_sha256"
    }
    digest = hashlib.sha256(
        json.dumps(
            unsigned_snapshot,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    snapshot["snapshot_sha256"] = digest
    extra_state["snapshot_sha256"] = digest


def _track_optoelectronic_state_load(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    from standalone_nnunet2d.models.optoelectronic_h2former import OptoelectronicH2Former

    original_load_state_dict = OptoelectronicH2Former.load_state_dict
    load_calls: list[str] = []

    def tracked_load_state_dict(model, *args, **kwargs):
        load_calls.append("called")
        return original_load_state_dict(model, *args, **kwargs)

    monkeypatch.setattr(OptoelectronicH2Former, "load_state_dict", tracked_load_state_dict)
    return load_calls


@pytest.fixture(scope="module")
def source_bundle(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, H2Former]:
    root = tmp_path_factory.mktemp("optoelectronic-initialization-source")
    torch.manual_seed(9022)
    source = H2Former(in_channels=1, num_classes=2, image_size=512).eval()
    path = root / "h2former-source.pth"
    save_checkpoint(
        source,
        None,
        path,
        {"model_name": "h2former", "supervision_mode": "single_output"},
        allowed_root=root,
    )
    return path, source


@pytest.fixture(scope="module")
def ideal_initialization(
    source_bundle: tuple[Path, H2Former],
    tmp_path_factory: pytest.TempPathFactory,
):
    source_path, _ = source_bundle
    return create_optoelectronic_initialization(
        source_path,
        _config(),
        tmp_path_factory.mktemp("optoelectronic-initialization-ideal"),
    )



@pytest.fixture(scope="module")
def physical_initialization(
    source_bundle: tuple[Path, H2Former],
    tmp_path_factory: pytest.TempPathFactory,
):
    source_path, _ = source_bundle
    return create_optoelectronic_initialization(
        source_path,
        _explicit_rank_config(1, mode="physical"),
        tmp_path_factory.mktemp("optoelectronic-initialization-physical"),
    )



def test_valid_h2former_checkpoint_creates_ideal_artifact_and_complete_report(
    ideal_initialization,
) -> None:
    artifact = _load_payload(ideal_initialization.artifact_path)
    report = json.loads(ideal_initialization.report_path.read_text(encoding="utf-8"))
    metadata = artifact["metadata"]

    assert set(artifact) == {
        "format_version",
        "artifact_type",
        "artifact_version",
        "model_state_dict",
        "optimizer_state_dict",
        "metadata",
    }
    assert artifact["format_version"] == 1
    assert artifact["artifact_type"] == ARTIFACT_TYPE
    assert artifact["artifact_version"] == ARTIFACT_VERSION
    assert artifact["optimizer_state_dict"] is None
    assert metadata["model_name"] == "optoelectronic_h2former"
    assert metadata["supervision_mode"] == "single_output"
    assert metadata["initialization_only"] is True
    assert metadata["resume_eligible"] is False
    assert metadata["evidence_class"] == "synthetic_engineering_initialization"
    assert metadata["pca_rank_selection"]["source"] == "project_default"
    assert metadata["pca_rank_selection"]["variance_threshold"] == pytest.approx(0.99)
    assert report["evidence_class"] == "synthetic_engineering_initialization"
    assert report["resolved_config"]["variance_threshold"] == pytest.approx(0.99)
    assert report["global"]["total_entry_kernel_count"] == 128
    assert report["global"]["active_route_count"] == metadata["global_exposure"]["active_route_count"]
    assert report["global"]["exposure_count"] == metadata["global_exposure"]["exposure_count"]
    assert report["claims"]["is_medical_result"] is False
    assert report["claims"]["is_speed_result"] is False
    assert report["claims"]["is_hardware_manufacturability_result"] is False
    assert report["claims"]["physical_psf_fit_completed"] is False

    assert {group["group_id"] for group in report["groups"]} == set(GROUPS)
    for group in report["groups"]:
        for field in (
            "original_conv_shape",
            "stride",
            "padding",
            "bias",
            "pca_sample_matrix_shape",
            "legal_maximum_rank",
            "resolved_rank",
            "explained_variance_ratio",
            "frobenius_reconstruction_error",
            "relative_reconstruction_error",
            "maximum_absolute_reconstruction_error",
            "centered_mean_contribution",
            "original_bias_retention",
        ):
            assert field in group, field
        assert group["centered_mean_contribution"]["retained"] is True
        assert group["original_bias_retention"]["retained"] is True

    json.dumps(metadata, allow_nan=False)


def test_physical_initialization_is_explicitly_unfitted(
    source_bundle: tuple[Path, H2Former], tmp_path: Path
) -> None:
    source_path, _ = source_bundle
    result = create_optoelectronic_initialization(
        source_path,
        _explicit_rank_config(mode="physical"),
        tmp_path,
    )
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    artifact = _load_payload(result.artifact_path)

    assert report["physical_fit_status"] == "unfitted_synthetic_initialization"
    assert artifact["metadata"]["physical_fit_status"] == "unfitted_synthetic_initialization"
    assert report["global"]["physical_assumptions"]["phase"]["wavelength_m"] == pytest.approx(520e-9)
    assert report["global"]["support_truncation"]["physical_truncation_claim"] is False


def test_explicit_five_group_ranks_are_resolved_and_strict_round_trip_loads(
    source_bundle: tuple[Path, H2Former], tmp_path: Path
) -> None:
    source_path, _ = source_bundle
    result = create_optoelectronic_initialization(
        source_path,
        _explicit_rank_config(1),
        tmp_path,
    )
    artifact = _load_payload(result.artifact_path)
    model = load_optoelectronic_initialization(
        result.artifact_path,
        source_path,
        map_location="cpu",
    )

    assert artifact["metadata"]["resolved_ranks"] == {group_id: 1 for group_id in GROUPS}
    assert model.identity_metadata()["resolved_ranks"] == {group_id: 1 for group_id in GROUPS}
    assert set(model.state_dict()) == set(artifact["model_state_dict"])


def test_source_sha_is_content_based_and_rejects_changed_content(
    source_bundle: tuple[Path, H2Former], ideal_initialization, tmp_path: Path
) -> None:
    source_path, _ = source_bundle
    copied_source = tmp_path / "renamed-source.pth"
    copied_source.write_bytes(source_path.read_bytes())
    loaded = load_optoelectronic_initialization(
        ideal_initialization.artifact_path,
        copied_source,
        map_location="cpu",
    )
    assert loaded.identity_metadata()["architecture_identity"] == "optoelectronic_h2former_from_h2former"

    changed_source = tmp_path / "changed-source.pth"
    changed_source.write_bytes(source_path.read_bytes() + b"content-change")
    with pytest.raises(ValueError, match="SHA256|sha256|source checkpoint"):
        load_optoelectronic_initialization(
            ideal_initialization.artifact_path,
            changed_source,
            map_location="cpu",
        )

    expected_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    artifact = _load_payload(ideal_initialization.artifact_path)
    assert artifact["metadata"]["source_checkpoint"]["sha256"] == expected_sha


@pytest.mark.parametrize(
    "metadata",
    [
        {"model_name": "plain_conv_unet", "supervision_mode": "single_output"},
        {"model_name": "h2former_lite_upernet", "supervision_mode": "single_output"},
        {"model_name": "h2former", "supervision_mode": "deep_supervision"},
        {"model_name": "h2former"},
        {
            "model_name": "h2former",
            "supervision_mode": "single_output",
            "config": {"model": {"name": "plain_conv_unet", "supervision_mode": "single_output"}},
        },
    ],
)
def test_invalid_source_identity_rejects_before_model_construction(
    monkeypatch: pytest.MonkeyPatch, metadata: dict[str, object], tmp_path: Path
) -> None:
    import standalone_nnunet2d.engine.optoelectronic_initialization as initialization

    source_path = _write_source_payload(tmp_path / "invalid-identity.pth", metadata)
    monkeypatch.setattr(
        initialization,
        "H2Former",
        lambda *args, **kwargs: pytest.fail("H2Former must not be constructed for invalid metadata"),
    )

    with pytest.raises((ValueError, TypeError), match="identity|model|supervision|metadata"):
        initialization.create_optoelectronic_initialization(
            source_path,
            _config(),
            tmp_path / "output",
        )


def test_malformed_source_checkpoint_rejects_before_model_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import standalone_nnunet2d.engine.optoelectronic_initialization as initialization

    path = tmp_path / "malformed.pth"
    torch.save({"format_version": 1, "metadata": {}}, path)
    monkeypatch.setattr(
        initialization,
        "H2Former",
        lambda *args, **kwargs: pytest.fail("H2Former must not be constructed for malformed payload"),
    )

    with pytest.raises(ValueError, match="checkpoint|payload|metadata"):
        initialization.create_optoelectronic_initialization(path, _config(), tmp_path / "output")


@pytest.mark.parametrize(
    "bad_config",
    [
        {"physical_config": {"phase": {}}, "unknown": 1},
        {"physical_config": {"unknown": 1, "phase": {}}},
        {"physical_config": {"phase": {"unknown": 1}}},
        {"physical_config": {}},
        {"physical_config": {"phase": {}}, "rank_by_group": {"stem7": 0}},
        {
            "physical_config": {"phase": {}},
            "rank_by_group": {group_id: 0 for group_id in GROUPS},
            "variance_threshold": 0.9,
        },
        {
            "physical_config": {"phase": {}},
            "rank_by_group": {"stem7": 50, "patch2": 0, "patch4": 0, "patch8": 0, "patch16": 0},
        },
        {"physical_config": {"phase": {"phase_grid_size": True}}},
        {"physical_config": {"phase": {"wavelength_m": float("nan")}}},
    ],
)
def test_config_parser_rejects_unknown_nested_invalid_or_nonfinite_values(
    bad_config: dict[str, object],
) -> None:
    with pytest.raises((ValueError, TypeError), match="config|key|rank|variance|finite|phase|physical"):
        build_optoelectronic_config(bad_config)


def test_artifact_schema_and_metadata_tampering_are_rejected(
    ideal_initialization, source_bundle: tuple[Path, H2Former], tmp_path: Path
) -> None:
    source_path, _ = source_bundle
    original = _load_payload(ideal_initialization.artifact_path)

    missing_key = copy.deepcopy(original)
    missing_key.pop("artifact_version")
    missing_path = tmp_path / "missing-key.pth"
    torch.save(missing_key, missing_path)
    with pytest.raises(ValueError, match="artifact|key|version"):
        load_optoelectronic_initialization(missing_path, source_path, map_location="cpu")

    metadata_tampered = copy.deepcopy(original)
    metadata_tampered["metadata"]["identity_digest"] = "tampered"
    metadata_path = tmp_path / "metadata-tampered.pth"
    torch.save(metadata_tampered, metadata_path)
    with pytest.raises(ValueError, match="identity|digest|metadata"):
        load_optoelectronic_initialization(metadata_path, source_path, map_location="cpu")


def test_artifact_tensor_and_extra_state_tampering_are_rejected(
    ideal_initialization, source_bundle: tuple[Path, H2Former], tmp_path: Path
) -> None:
    source_path, _ = source_bundle
    original = _load_payload(ideal_initialization.artifact_path)

    tensor_tampered = copy.deepcopy(original)
    key = "patch_embed.projs.0.component_bank"
    tensor = tensor_tampered["model_state_dict"][key].clone()
    tensor.view(-1)[0] += 0.125
    tensor_tampered["model_state_dict"][key] = tensor
    tensor_path = tmp_path / "tensor-tampered.pth"
    torch.save(tensor_tampered, tensor_path)
    with pytest.raises((ValueError, RuntimeError), match="identity|digest|PCA|tensor|state"):
        load_optoelectronic_initialization(tensor_path, source_path, map_location="cpu")

    extra_tampered = copy.deepcopy(original)
    extra_tampered["model_state_dict"]["_extra_state"]["snapshot_sha256"] = "tampered"
    extra_path = tmp_path / "extra-state-tampered.pth"
    torch.save(extra_tampered, extra_path)
    with pytest.raises((ValueError, RuntimeError), match="extra|snapshot|identity|digest"):
        load_optoelectronic_initialization(extra_path, source_path, map_location="cpu")




def test_loader_rejects_equality_spoofed_extra_state_before_state_load(
    ideal_initialization,
    source_bundle: tuple[Path, H2Former],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path, _ = source_bundle
    payload = copy.deepcopy(_load_payload(ideal_initialization.artifact_path))
    extra_state = payload["model_state_dict"]["_extra_state"]
    extra_state["snapshot"]["model_class"] = _AlwaysEqualString("ForgedModelClass")
    _refresh_extra_state_snapshot_digest(extra_state)
    tampered_path = tmp_path / "equality-spoofed-extra-state.pth"
    torch.save(payload, tampered_path)

    load_calls = _track_optoelectronic_state_load(monkeypatch)
    _AlwaysEqualString.equality_calls = 0
    try:
        load_optoelectronic_initialization(
            tampered_path, source_path, map_location="cpu"
        )
    except ValueError:
        pass
    else:
        pytest.fail(
            "loader accepted equality-spoofed _extra_state; "
            f"target load_state_dict called {len(load_calls)} time(s)"
        )

    assert load_calls == []
    assert _AlwaysEqualString.equality_calls == 0


def test_loader_rejects_same_content_string_subclass_in_extra_state_before_load(
    ideal_initialization,
    source_bundle: tuple[Path, H2Former],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path, _ = source_bundle
    payload = copy.deepcopy(_load_payload(ideal_initialization.artifact_path))
    extra_state = payload["model_state_dict"]["_extra_state"]
    extra_state["snapshot"]["model_class"] = _AlwaysEqualString("OptoelectronicH2Former")
    _refresh_extra_state_snapshot_digest(extra_state)
    tampered_path = tmp_path / "same-content-string-subclass.pth"
    torch.save(payload, tampered_path)

    load_calls = _track_optoelectronic_state_load(monkeypatch)
    _AlwaysEqualString.equality_calls = 0
    try:
        load_optoelectronic_initialization(
            tampered_path, source_path, map_location="cpu"
        )
    except ValueError:
        pass
    else:
        pytest.fail(
            "loader accepted same-content string subclass; "
            f"target load_state_dict called {len(load_calls)} time(s)"
        )
    assert load_calls == []
    assert _AlwaysEqualString.equality_calls == 0


@pytest.mark.parametrize("mutation", ["dict-subclass", "list-subclass", "string-key-subclass"])
def test_loader_rejects_extra_state_container_subclasses_before_load(
    mutation: str,
    ideal_initialization,
    source_bundle: tuple[Path, H2Former],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path, _ = source_bundle
    payload = copy.deepcopy(_load_payload(ideal_initialization.artifact_path))
    extra_state = payload["model_state_dict"]["_extra_state"]
    snapshot = extra_state["snapshot"]
    if mutation == "dict-subclass":
        extra_state["snapshot"] = _DictSubclass(snapshot)
    elif mutation == "list-subclass":
        snapshot["entry_convolutions"] = _ListSubclass(snapshot["entry_convolutions"])
    elif mutation == "string-key-subclass":
        value = snapshot.pop("model_class")
        snapshot[_StringSubclass("model_class")] = value
    else:
        raise AssertionError(f"unknown mutation: {mutation}")
    _refresh_extra_state_snapshot_digest(extra_state)
    tampered_path = tmp_path / f"container-subclass-{mutation}.pth"
    torch.save(payload, tampered_path)

    load_calls = _track_optoelectronic_state_load(monkeypatch)
    with pytest.raises(ValueError, match="state|JSON|type|key"):
        load_optoelectronic_initialization(
            tampered_path, source_path, map_location="cpu"
        )
    assert load_calls == []


def test_loader_rejects_bool_integer_equality_in_extra_state_before_load(
    ideal_initialization,
    source_bundle: tuple[Path, H2Former],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path, _ = source_bundle
    payload = copy.deepcopy(_load_payload(ideal_initialization.artifact_path))
    extra_state = payload["model_state_dict"]["_extra_state"]
    snapshot = extra_state["snapshot"]
    snapshot["config"]["trainable_phase"] = int(
        snapshot["config"]["trainable_phase"]
    )
    _refresh_extra_state_snapshot_digest(extra_state)
    tampered_path = tmp_path / "bool-integer-extra-state.pth"
    torch.save(payload, tampered_path)

    load_calls = _track_optoelectronic_state_load(monkeypatch)
    with pytest.raises(ValueError, match="state|JSON|type"):
        load_optoelectronic_initialization(
            tampered_path, source_path, map_location="cpu"
        )
    assert load_calls == []


def test_loader_rejects_non_finite_extra_state_float_before_load(
    ideal_initialization,
    source_bundle: tuple[Path, H2Former],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path, _ = source_bundle
    payload = copy.deepcopy(_load_payload(ideal_initialization.artifact_path))
    extra_state = payload["model_state_dict"]["_extra_state"]
    extra_state["snapshot"]["config"]["variance_threshold"] = float("nan")
    tampered_path = tmp_path / "non-finite-extra-state.pth"
    torch.save(payload, tampered_path)

    load_calls = _track_optoelectronic_state_load(monkeypatch)
    with pytest.raises(ValueError, match="state|JSON|finite|non-finite"):
        load_optoelectronic_initialization(
            tampered_path, source_path, map_location="cpu"
        )
    assert load_calls == []


def test_physical_initialization_round_trip_preserves_complete_state(
    physical_initialization, source_bundle: tuple[Path, H2Former]
) -> None:
    source_path, _ = source_bundle
    artifact = _load_payload(physical_initialization.artifact_path)
    model = load_optoelectronic_initialization(
        physical_initialization.artifact_path,
        source_path,
        map_location="cpu",
    )

    loaded_state = model.state_dict()
    artifact_state = artifact["model_state_dict"]
    assert set(loaded_state) == set(artifact_state)
    for key, expected in artifact_state.items():
        actual = loaded_state[key]
        if type(expected) is torch.Tensor:
            assert torch.equal(actual.cpu(), expected.cpu())
        else:
            assert actual == expected


@pytest.mark.parametrize(
    ("fixture_name", "state_key"),
    [
        ("ideal_initialization", "conv1.mixing"),
        ("ideal_initialization", "patch_embed.projs.0.bias"),
        ("ideal_initialization", "bn1.weight"),
        ("physical_initialization", "phase_theta"),
    ],
)
def test_loader_rejects_trainable_initialization_state_tampering_before_load(
    fixture_name: str,
    state_key: str,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import standalone_nnunet2d.engine.optoelectronic_initialization as initialization
    from standalone_nnunet2d.models.optoelectronic_h2former import OptoelectronicH2Former

    result = request.getfixturevalue(fixture_name)
    source_path = request.getfixturevalue("source_bundle")[0]
    payload = copy.deepcopy(_load_payload(result.artifact_path))
    state = payload["model_state_dict"]
    if state_key == "phase_theta":
        state_key = next(
            key for key in state if ".phase_psfs." in key and key.endswith(".theta")
        )
    tensor = state[state_key].clone()
    tensor.view(-1)[0] += 0.125
    state[state_key] = tensor
    tampered_path = tmp_path / f"tampered-{fixture_name}-{state_key.rsplit('.', 1)[-1]}.pth"
    torch.save(payload, tampered_path)

    original_load_state_dict = OptoelectronicH2Former.load_state_dict
    load_calls: list[str] = []

    def tracked_load_state_dict(model, *args, **kwargs):
        load_calls.append("called")
        return original_load_state_dict(model, *args, **kwargs)

    monkeypatch.setattr(OptoelectronicH2Former, "load_state_dict", tracked_load_state_dict)
    with pytest.raises(ValueError, match="state|tensor|initialization"):
        initialization.load_optoelectronic_initialization(
            tampered_path, source_path, map_location="cpu"
        )
    assert load_calls == []


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-key",
        "unexpected-key",
        "non-tensor",
        "tensor-subclass",
        "shape",
        "dtype",
        "quantized",
        "non-strided",
        "meta",
    ],
)
def test_loader_rejects_malformed_complete_state_before_load(
    mutation: str,
    ideal_initialization,
    source_bundle: tuple[Path, H2Former],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import standalone_nnunet2d.engine.optoelectronic_initialization as initialization
    from standalone_nnunet2d.models.optoelectronic_h2former import OptoelectronicH2Former

    source_path, _ = source_bundle
    payload = copy.deepcopy(_load_payload(ideal_initialization.artifact_path))
    state = payload["model_state_dict"]
    key = "bn1.weight"
    tensor = state[key]
    if mutation == "missing-key":
        state.pop(key)
    elif mutation == "unexpected-key":
        state["unexpected"] = torch.zeros(1)
    elif mutation == "non-tensor":
        state[key] = "not a tensor"
    elif mutation == "tensor-subclass":
        state[key] = torch.nn.Parameter(tensor.clone())
    elif mutation == "shape":
        state[key] = tensor[:1]
    elif mutation == "dtype":
        state[key] = tensor.to(dtype=torch.float64)
    elif mutation == "quantized":
        state[key] = torch.quantize_per_tensor(tensor.cpu(), scale=0.1, zero_point=0, dtype=torch.qint8)
    elif mutation == "non-strided":
        state[key] = tensor.to_sparse()
    elif mutation == "meta":
        state[key] = torch.empty_like(tensor, device="meta")
    else:
        raise AssertionError(f"unknown mutation: {mutation}")
    tampered_path = tmp_path / f"malformed-{mutation}.pth"
    torch.save(payload, tampered_path)

    original_load_state_dict = OptoelectronicH2Former.load_state_dict
    load_calls: list[str] = []

    def tracked_load_state_dict(model, *args, **kwargs):
        load_calls.append("called")
        return original_load_state_dict(model, *args, **kwargs)

    monkeypatch.setattr(OptoelectronicH2Former, "load_state_dict", tracked_load_state_dict)
    with pytest.raises((ValueError, TypeError), match="state|tensor|initialization|key"):
        initialization.load_optoelectronic_initialization(
            tampered_path, source_path, map_location="cpu"
        )
    assert load_calls == []


def test_source_unlink_failure_rolls_back_first_published_artifact(
    source_bundle: tuple[Path, H2Former],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import standalone_nnunet2d.engine.optoelectronic_initialization as initialization

    source_path, _ = source_bundle
    output_root = tmp_path / "source-unlink-failure"
    original_unlink = Path.unlink
    injected_paths: set[Path] = set()

    def fail_source_unlink_once(path: Path, *args, **kwargs):
        if (
            path.parent == output_root
            and path.name.startswith(".optoelectronic-initialization-")
            and path not in injected_paths
        ):
            injected_paths.add(path)
            raise OSError("injected source temp unlink failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_source_unlink_once)
    with pytest.raises(OSError, match="injected source temp unlink failure"):
        initialization.create_optoelectronic_initialization(
            source_path, _config(), output_root
        )

    assert len(injected_paths) == 1
    assert not (output_root / INITIALIZATION_FILENAME).exists()
    assert not (output_root / REPORT_FILENAME).exists()
    assert list(output_root.iterdir()) == []


@pytest.mark.parametrize("competing_target", [False, True], ids=["unsupported", "race-target"])
def test_hard_link_failure_fails_closed_without_rename_or_overwrite(
    competing_target: bool,
    source_bundle: tuple[Path, H2Former],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import standalone_nnunet2d.engine.optoelectronic_initialization as initialization

    source_path, _ = source_bundle
    output_root = tmp_path / f"hard-link-failure-{competing_target}"
    artifact_path = output_root / INITIALIZATION_FILENAME
    competitor_bytes = b"competing process output"

    def unsupported_link(source: Path, destination: Path) -> None:
        if competing_target:
            destination.write_bytes(competitor_bytes)
        raise OSError("simulated hard-link unsupported")

    rename_calls: list[str] = []

    def forbidden_rename(*args, **kwargs):
        rename_calls.append("called")
        raise AssertionError("os.rename must not be used as a no-replace fallback")

    monkeypatch.setattr(initialization.os, "link", unsupported_link)
    monkeypatch.setattr(initialization.os, "rename", forbidden_rename)
    with pytest.raises(OSError):
        initialization.create_optoelectronic_initialization(
            source_path, _config(), output_root
        )

    assert rename_calls == []
    if competing_target:
        assert artifact_path.read_bytes() == competitor_bytes
    else:
        assert not artifact_path.exists()
    assert not (output_root / REPORT_FILENAME).exists()
    expected_entries = {artifact_path} if competing_target else set()
    assert set(output_root.iterdir()) == expected_entries



def test_report_publish_failure_removes_first_published_artifact(
    source_bundle: tuple[Path, H2Former],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import standalone_nnunet2d.engine.optoelectronic_initialization as initialization

    source_path, _ = source_bundle
    output_root = tmp_path / "report-publish-failure"
    original_link = initialization.os.link

    def fail_report_publish(source: Path, destination: Path) -> None:
        if destination.name == INITIALIZATION_FILENAME:
            original_link(source, destination)
            return
        if destination.name == REPORT_FILENAME:
            raise OSError("injected report link failure")
        raise AssertionError(f"unexpected publication destination: {destination}")

    monkeypatch.setattr(initialization.os, "link", fail_report_publish)
    with pytest.raises(OSError, match="injected report link failure"):
        initialization.create_optoelectronic_initialization(
            source_path, _config(), output_root
        )

    assert not (output_root / INITIALIZATION_FILENAME).exists()
    assert not (output_root / REPORT_FILENAME).exists()
    assert list(output_root.iterdir()) == []



def test_second_mkstemp_failure_cleans_first_temporary_file(
    source_bundle: tuple[Path, H2Former],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import standalone_nnunet2d.engine.optoelectronic_initialization as initialization

    source_path, _ = source_bundle
    output_root = tmp_path / "second-mkstemp-failure"
    original_mkstemp = initialization.tempfile.mkstemp
    calls = 0

    def fail_second_mkstemp(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second mkstemp failure")
        return original_mkstemp(*args, **kwargs)

    monkeypatch.setattr(initialization.tempfile, "mkstemp", fail_second_mkstemp)
    with pytest.raises(OSError, match="injected second mkstemp failure"):
        initialization.create_optoelectronic_initialization(
            source_path, _config(), output_root
        )

    assert calls == 2
    assert list(output_root.iterdir()) == []



def test_existing_outputs_are_not_overwritten_and_failed_initialization_leaves_no_final_files(
    source_bundle: tuple[Path, H2Former], ideal_initialization, tmp_path: Path
) -> None:
    source_path, _ = source_bundle
    output_root = ideal_initialization.artifact_path.parent
    artifact_before = ideal_initialization.artifact_path.read_bytes()
    report_before = ideal_initialization.report_path.read_bytes()

    with pytest.raises(FileExistsError, match="exists|overwrite"):
        create_optoelectronic_initialization(source_path, _config(), output_root)
    assert ideal_initialization.artifact_path.read_bytes() == artifact_before
    assert ideal_initialization.report_path.read_bytes() == report_before

    failed_root = tmp_path / "failed-output"
    with pytest.raises(ValueError, match="unknown|key|config"):
        create_optoelectronic_initialization(
            source_path,
            {"physical_config": {"phase": {}}, "unexpected": True},
            failed_root,
        )
    assert not (failed_root / INITIALIZATION_FILENAME).exists()
    assert not (failed_root / REPORT_FILENAME).exists()
