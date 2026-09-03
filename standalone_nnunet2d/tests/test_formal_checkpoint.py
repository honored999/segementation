from __future__ import annotations
import json
import random
from uuid import uuid4
from copy import deepcopy
from pathlib import Path

import pytest
import numpy as np
import torch
from torch import nn
from standalone_nnunet2d.engine.checkpoint import PROJECT_OUTPUTS_DIRECTORY
from standalone_nnunet2d.alignment_evidence import build_alignment_evidence
from standalone_nnunet2d.training.formal_checkpoint import (
    FormalTrainerState,
    checkpoint_input_channels,
    checkpoint_input_mode,
    load_formal_checkpoint,
    save_formal_checkpoint,
)
from standalone_nnunet2d.data.input_mode import InputMode, input_spec
from standalone_nnunet2d.training.official_config import PolyLRScheduler


def _checkpoint_path() -> object:
    return PROJECT_OUTPUTS_DIRECTORY / f"pytest-formal-{uuid4().hex}.pth"


def _components() -> dict[str, dict[str, object]]:
    return {
        name: {"status": "passed", "diagnostics": []}
        for name in ("image", "label", "manifest", "mask")
    }


def _alignment_evidence(tmp_path: Path, *, suffix: str = "") -> dict[str, object]:
    transform_path = tmp_path / f"transform{suffix}.json"
    inference_path = tmp_path / f"inference{suffix}.json"
    transform_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "run_state": "official_alignment_pending",
                "oracle_root": f"/oracle/transform/{suffix}",
                "standalone_root": f"/standalone/transform/{suffix}",
                "image_atol": 0.0,
                "components": _components(),
                "diagnostics": [],
            }
        ),
        encoding="utf-8",
    )
    inference_path.write_text(
        json.dumps(
            {
                "parity_policy": "repeat_oracle_stability_v1",
                "oracle_roots": [
                    f"/oracle/inference/{suffix}/0",
                    f"/oracle/inference/{suffix}/1",
                    f"/oracle/inference/{suffix}/2",
                ],
                "oracle_repeat_count": 3,
                "stable_mask_mismatch_count": 0,
                "stable_mask_mismatch_coordinates": [],
                "unobserved_standalone_label_count": 0,
                "unobserved_standalone_label_coordinates": [],
                "status": "passed",
                "run_state": "official_alignment_pending",
                "standalone_root": f"/standalone/inference/{suffix}",
                "image_atol": 0.0,
                "components": _components(),
                "diagnostics": [],
            }
        ),
        encoding="utf-8",
    )
    return build_alignment_evidence(transform_path, inference_path)


def _aligned_config(evidence: dict[str, object]) -> dict[str, object]:
    return {
        "run_type": "official_aligned",
        "run_state": "official_aligned",
        "alignment_evidence": deepcopy(evidence),
        "resolved": True,
    }


def test_formal_checkpoint_restores_scheduler_and_rng_state() -> None:
    torch.manual_seed(3)
    model = nn.Conv2d(1, 2, 1)
    optimizer = torch.optim.SGD(model.parameters(), .01, momentum=.9)
    scheduler = PolyLRScheduler(optimizer, .01, 1000)
    scheduler.step(17)
    state = FormalTrainerState(epoch=18, global_step=4500, best_validation_dice=.4, fold=0)
    config = {"seed": 0, "resolved": True}
    policies = {"scheduler": {"name": "poly", "exponent": .9}, "sampling": {"foreground": .33}}

    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    path = _checkpoint_path()
    save_formal_checkpoint(
        model,
        optimizer,
        scheduler,
        path,
        state,
        config,
        plan_hash="plan-sha256",
        policies=policies,
    )
    expected_rng = (random.random(), float(np.random.random()), torch.rand(3))
    random.seed(91)
    np.random.seed(91)
    torch.manual_seed(91)

    restored_model = nn.Conv2d(1, 2, 1)
    restored_optimizer = torch.optim.SGD(restored_model.parameters(), .01, momentum=.9)
    restored_scheduler = PolyLRScheduler(restored_optimizer, .01, 1000)
    restored = load_formal_checkpoint(
        restored_model,
        restored_optimizer,
        restored_scheduler,
        path,
        fold=0,
        plan_hash="plan-sha256",
        policies=policies,
    )

    assert restored.state == state
    assert restored.scheduler_step == 17
    assert restored.config == config
    assert restored.policies == policies
    assert restored.run_state == "official_alignment_pending"
    assert random.random() == expected_rng[0]
    assert np.random.random() == expected_rng[1]
    assert torch.equal(torch.rand(3), expected_rng[2])

    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    assert metadata["scheduler_state"]["step"] == 17
    assert metadata["plan_hash"] == "plan-sha256"
    assert metadata["policies"] == policies
    assert metadata["run_state"] == "official_alignment_pending"
    assert metadata["resolved_config"] == config


def test_formal_checkpoint_records_input_channels_in_metadata() -> None:
    model = nn.Conv2d(3, 2, 1)
    optimizer = torch.optim.SGD(model.parameters(), .01)
    scheduler = PolyLRScheduler(optimizer, .01, 1000)
    state = FormalTrainerState(epoch=1, global_step=1, best_validation_dice=.1, fold=0)
    path = _checkpoint_path()

    save_formal_checkpoint(
        model,
        optimizer,
        scheduler,
        path,
        state,
        {"input_channels": 3, "run_state": "official_alignment_pending"},
    )

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["metadata"]["input_channels"] == 3


def test_checkpoint_metadata_resolves_dwi_adc_bilateral_and_legacy_bilateral_modes() -> None:
    c4_formal_config = {
        "input_mode": "dwi_adc_bilateral",
        "input_channels": 4,
        "physical_input_channels": 2,
        "effective_model_input_channels": 4,
        "run_state": "official_alignment_pending",
    }
    c4_metadata = {
        "input_channels": 4,
        "resolved_config": c4_formal_config,
    }

    resolved_mode = checkpoint_input_mode(c4_metadata)
    assert resolved_mode is InputMode.DWI_ADC_BILATERAL
    resolved_spec = input_spec(resolved_mode)
    assert resolved_spec.physical_input_channels == 2
    assert resolved_spec.effective_input_channels == 4

    legacy_metadata = {
        "input_channels": 2,
        "resolved_config": {
            "bilateral_asymmetry_channel": True,
            "physical_input_channels": 1,
            "effective_model_input_channels": 2,
        },
    }
    assert checkpoint_input_mode(legacy_metadata) is InputMode.DWI_BILATERAL


@pytest.mark.parametrize(
    "legacy_metadata",
    [
        {"bilateral_asymmetry_channel": True},
        {"resolved_config": {"bilateral_asymmetry_channel": True}},
    ],
)
def test_legacy_bilateral_metadata_without_counts_resolves_to_effective_two_channels(
    legacy_metadata: dict[str, object],
) -> None:
    assert checkpoint_input_mode(legacy_metadata) is InputMode.DWI_BILATERAL
    assert checkpoint_input_channels(legacy_metadata) == 2


@pytest.mark.parametrize(
    ("conflicting_field", "conflicting_value"),
    [
        ("input_channels", 1),
        ("physical_input_channels", 2),
        ("effective_model_input_channels", 1),
    ],
)
def test_legacy_bilateral_metadata_rejects_explicit_contract_conflicts(
    conflicting_field: str,
    conflicting_value: int,
) -> None:
    metadata = {
        "bilateral_asymmetry_channel": True,
        conflicting_field: conflicting_value,
    }

    with pytest.raises(ValueError):
        checkpoint_input_mode(metadata)


def test_modern_bilateral_metadata_without_counts_remains_strict() -> None:
    with pytest.raises(ValueError, match="input_channels=2"):
        checkpoint_input_mode({"input_mode": "dwi_bilateral"})


@pytest.mark.parametrize(
    ("input_mode", "input_channels", "physical_channels", "effective_channels", "field", "bad_value"),
    [
        ("dwi_bilateral", 2, 1, 2, "physical_input_channels", 1.0),
        ("dwi_bilateral", 2, 1, 2, "effective_model_input_channels", 2.0),
        ("dwi_bilateral", 2, 1, 2, "physical_input_channels", True),
        ("dwi", 1, 1, 1, "effective_model_input_channels", True),
    ],
)
def test_checkpoint_input_mode_rejects_non_integer_channel_metadata(
    input_mode: str,
    input_channels: int,
    physical_channels: int,
    effective_channels: int,
    field: str,
    bad_value: object,
) -> None:
    resolved_config = {
        "input_mode": input_mode,
        "input_channels": input_channels,
        "physical_input_channels": physical_channels,
        "effective_model_input_channels": effective_channels,
    }
    metadata = {
        "input_channels": input_channels,
        field: bad_value,
        "resolved_config": resolved_config,
    }

    with pytest.raises(ValueError, match=field):
        checkpoint_input_mode(metadata)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("physical_input_channels", 2.0),
        ("physical_input_channels", True),
        ("effective_model_input_channels", 4.0),
        ("effective_model_input_channels", True),
    ],
)
def test_checkpoint_input_channels_rejects_non_integer_contract_metadata(
    field: str, bad_value: object
) -> None:
    metadata = {
        "input_mode": "dwi_adc_bilateral",
        "input_channels": 4,
        "physical_input_channels": 2,
        "effective_model_input_channels": 4,
        field: bad_value,
    }

    with pytest.raises(ValueError, match=field):
        checkpoint_input_channels(metadata)


@pytest.mark.parametrize(
    ("field", "metadata_value", "resolved_value"),
    [
        ("physical_input_channels", 2, 1),
        ("effective_model_input_channels", 4, 2),
    ],
)
def test_checkpoint_input_channels_rejects_cross_source_contract_conflicts(
    field: str, metadata_value: int, resolved_value: int
) -> None:
    metadata = {
        "input_mode": "dwi_adc_bilateral",
        "input_channels": 4,
        "physical_input_channels": metadata_value if field == "physical_input_channels" else 2,
        "effective_model_input_channels": metadata_value if field == "effective_model_input_channels" else 4,
        "resolved_config": {
            "input_mode": "dwi_adc_bilateral",
            "input_channels": 4,
            "physical_input_channels": resolved_value if field == "physical_input_channels" else 2,
            "effective_model_input_channels": resolved_value if field == "effective_model_input_channels" else 4,
        },
    }

    with pytest.raises(ValueError, match=f"{field} declarations conflict"):
        checkpoint_input_channels(metadata)


@pytest.mark.parametrize(
    ("input_mode", "input_channels", "physical_channels", "effective_channels"),
    [
        ("dwi", 1, 1, 1),
        ("dwi_adc", 2, 2, 2),
        ("dwi_bilateral", 2, 1, 2),
        ("dwi_adc_bilateral", 4, 2, 4),
    ],
)
def test_checkpoint_input_channels_accepts_consistent_integer_contract_metadata(
    input_mode: str,
    input_channels: int,
    physical_channels: int,
    effective_channels: int,
) -> None:
    declaration = {
        "input_mode": input_mode,
        "input_channels": input_channels,
        "physical_input_channels": physical_channels,
        "effective_model_input_channels": effective_channels,
    }
    metadata = {
        **declaration,
        "resolved_config": dict(declaration),
        "config": dict(declaration),
    }

    assert checkpoint_input_channels(metadata) == input_channels


def test_formal_checkpoint_rejects_official_aligned_local_state() -> None:
    model = nn.Conv2d(1, 2, 1)
    optimizer = torch.optim.SGD(model.parameters(), .01)
    scheduler = PolyLRScheduler(optimizer, .01, 1000)
    state = FormalTrainerState(epoch=1, global_step=1, best_validation_dice=.1, fold=0)

    try:
        save_formal_checkpoint(
            model,
            optimizer,
            scheduler,
            _checkpoint_path(),
            state,
            {"run_state": "official_aligned"},
            run_state="official_aligned",
        )
    except ValueError as error:
        assert "official_alignment_pending" in str(error)
    else:
        raise AssertionError("official_aligned must not be persisted locally")


def test_aligned_checkpoint_saves_loads_and_restores_evidence(tmp_path: Path) -> None:
    evidence = _alignment_evidence(tmp_path)
    config = _aligned_config(evidence)
    model = nn.Conv2d(1, 2, 1)
    optimizer = torch.optim.SGD(model.parameters(), .01)
    scheduler = PolyLRScheduler(optimizer, .01, 1000)
    state = FormalTrainerState(epoch=2, global_step=3, best_validation_dice=.2, fold=0)
    path = _checkpoint_path()

    save_formal_checkpoint(
        model,
        optimizer,
        scheduler,
        path,
        state,
        config,
        run_state="official_aligned",
        alignment_evidence=evidence,
    )

    restored = load_formal_checkpoint(
        model,
        optimizer,
        scheduler,
        path,
        fold=0,
        run_state="official_aligned",
        alignment_evidence=evidence,
    )

    assert restored.run_state == "official_aligned"
    assert restored.alignment_evidence == evidence
    assert restored.alignment_evidence is not evidence
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["metadata"]["run_type"] == "official_aligned"
    assert payload["metadata"]["run_state"] == "official_aligned"
    assert payload["metadata"]["alignment_evidence"] == evidence


@pytest.mark.parametrize(
    ("config", "run_state", "alignment_evidence"),
    [
        (
            {"run_type": "official_aligned", "run_state": "official_aligned"},
            "official_aligned",
            None,
        ),
        (
            {"run_type": "official_alignment_pending", "run_state": "official_alignment_pending"},
            "official_alignment_pending",
            {"tampered": True},
        ),
        (
            {"run_type": "official_alignment_pending", "run_state": "official_alignment_pending"},
            "official_aligned",
            {"tampered": True},
        ),
        (
            {"run_type": "official_aligned", "run_state": "official_aligned"},
            "official_alignment_pending",
            None,
        ),
    ],
)
def test_save_rejects_inconsistent_alignment_state(
    tmp_path: Path,
    config: dict[str, object],
    run_state: str,
    alignment_evidence: dict[str, object] | None,
) -> None:
    model = nn.Conv2d(1, 2, 1)
    optimizer = torch.optim.SGD(model.parameters(), .01)
    scheduler = PolyLRScheduler(optimizer, .01, 1000)
    state = FormalTrainerState(epoch=1, global_step=1, best_validation_dice=.1, fold=0)

    with pytest.raises(ValueError):
        save_formal_checkpoint(
            model,
            optimizer,
            scheduler,
            _checkpoint_path(),
            state,
            config,
            run_state=run_state,
            alignment_evidence=alignment_evidence,
        )


def test_aligned_checkpoint_load_rejects_different_evidence(tmp_path: Path) -> None:
    evidence = _alignment_evidence(tmp_path, suffix="_one")
    different_evidence = _alignment_evidence(tmp_path, suffix="_two")
    config = _aligned_config(evidence)
    model = nn.Conv2d(1, 2, 1)
    optimizer = torch.optim.SGD(model.parameters(), .01)
    scheduler = PolyLRScheduler(optimizer, .01, 1000)
    state = FormalTrainerState(epoch=1, global_step=1, best_validation_dice=.1, fold=0)
    path = _checkpoint_path()
    save_formal_checkpoint(
        model,
        optimizer,
        scheduler,
        path,
        state,
        config,
        run_state="official_aligned",
        alignment_evidence=evidence,
    )

    with pytest.raises(ValueError):
        load_formal_checkpoint(
            model,
            optimizer,
            scheduler,
            path,
            fold=0,
            run_state="official_aligned",
            alignment_evidence=different_evidence,
        )


def test_aligned_checkpoint_load_rejects_tampered_embedded_evidence(tmp_path: Path) -> None:
    evidence = _alignment_evidence(tmp_path)
    config = _aligned_config(evidence)
    model = nn.Conv2d(1, 2, 1)
    optimizer = torch.optim.SGD(model.parameters(), .01)
    scheduler = PolyLRScheduler(optimizer, .01, 1000)
    state = FormalTrainerState(epoch=1, global_step=1, best_validation_dice=.1, fold=0)
    path = _checkpoint_path()
    save_formal_checkpoint(
        model,
        optimizer,
        scheduler,
        path,
        state,
        config,
        run_state="official_aligned",
        alignment_evidence=evidence,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["metadata"]["alignment_evidence"]["sources"]["transform"]["sha256"] = "0" * 64
    torch.save(payload, path)

    with pytest.raises(ValueError):
        load_formal_checkpoint(
            model,
            optimizer,
            scheduler,
            path,
            fold=0,
            run_state="official_aligned",
            alignment_evidence=evidence,
        )
