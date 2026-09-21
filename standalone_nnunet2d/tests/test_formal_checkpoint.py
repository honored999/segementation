from __future__ import annotations
import json
import random
from unittest.mock import Mock
from uuid import uuid4
from copy import deepcopy
from pathlib import Path

import pytest
import numpy as np
import torch
from torch import nn
from standalone_nnunet2d.engine.checkpoint import PROJECT_OUTPUTS_DIRECTORY, load_checkpoint
from standalone_nnunet2d.engine import checkpoint as checkpoint_module
from standalone_nnunet2d.alignment_evidence import build_alignment_evidence
from standalone_nnunet2d.training.formal_checkpoint import (
    FormalTrainerState,
    load_formal_checkpoint,
    save_formal_checkpoint,
)
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


def test_formal_checkpoint_round_trip_preserves_selection_continuation_state(tmp_path: Path) -> None:
    model = nn.Conv2d(1, 2, 1)
    optimizer = torch.optim.SGD(model.parameters(), .01)
    scheduler = PolyLRScheduler(optimizer, 1000)
    state = FormalTrainerState(
        epoch=110,
        global_step=7,
        best_validation_dice=.4,
        fold=0,
        best_selection_dice=.73,
        best_selection_epoch=100,
        early_stop_reference_dice=.72,
        checks_without_improvement=2,
    )
    config = {
        "run_type": "official_alignment_pending",
        "run_state": "official_alignment_pending",
        "plan_hash": "selection-plan",
    }
    path = tmp_path / "checkpoint_latest.pth"
    save_formal_checkpoint(
        model,
        optimizer,
        scheduler,
        path,
        state,
        config,
        plan_hash="selection-plan",
        checkpoint_root=tmp_path,
    )

    restored_model = nn.Conv2d(1, 2, 1)
    restored_optimizer = torch.optim.SGD(restored_model.parameters(), .01)
    restored_scheduler = PolyLRScheduler(restored_optimizer, 1000)
    restored = load_formal_checkpoint(
        restored_model,
        restored_optimizer,
        restored_scheduler,
        path,
        fold=0,
        plan_hash="selection-plan",
        checkpoint_root=tmp_path,
    )
    assert restored.state == state
    metadata = torch.load(path, map_location="cpu", weights_only=False)["metadata"]
    assert metadata["best_selection_dice"] == pytest.approx(.73)
    assert metadata["best_selection_epoch"] == 100
    assert metadata["early_stop_reference_dice"] == pytest.approx(.72)
    assert metadata["checks_without_improvement"] == 2


def test_formal_checkpoint_save_and_load_use_explicit_checkpoint_root(tmp_path: Path) -> None:
    root = tmp_path / "formal-run"
    path = root / "checkpoint_latest.pth"
    model = nn.Conv2d(1, 2, 1)
    optimizer = torch.optim.SGD(model.parameters(), .01)
    state = FormalTrainerState(epoch=1, global_step=2, best_validation_dice=.3, fold=0)
    config = {"run_type": "official_alignment_pending", "run_state": "official_alignment_pending"}

    save_formal_checkpoint(
        model,
        optimizer,
        path,
        state,
        config,
        checkpoint_root=root,
    )

    restored_model = nn.Conv2d(1, 2, 1)
    restored = load_formal_checkpoint(
        restored_model,
        torch.optim.SGD(restored_model.parameters(), .01),
        path,
        fold=0,
        checkpoint_root=root,
    )

    assert restored.state == state
    assert torch.equal(model.weight, restored_model.weight)


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


def _model_identity(name: str, supervision_mode: str) -> dict[str, object]:
    return {
        "name": name,
        "in_channels": 1,
        "num_classes": 2,
        "image_size": 512 if name in {"h2former", "h2former_lite_upernet"} else None,
        "supervision_mode": supervision_mode,
    }


@pytest.mark.parametrize(
    ("saved_name", "saved_mode", "expected_name", "expected_mode"),
    [
        ("plain_conv_unet", "deep_supervision", "h2former", "single_output"),
        ("h2former", "single_output", "plain_conv_unet", "deep_supervision"),
        ("h2former", "single_output", "h2former_lite_upernet", "single_output"),
        ("plain_conv_unet", "deep_supervision", "plain_conv_unet_lite_upernet", "single_output"),
        ("h2former_lite_upernet", "single_output", "plain_conv_unet_lite_upernet", "single_output"),
    ],
)
def test_formal_checkpoint_rejects_cross_model_before_state_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    saved_name: str,
    saved_mode: str,
    expected_name: str,
    expected_mode: str,
) -> None:
    monkeypatch.setattr(checkpoint_module, "PROJECT_OUTPUTS_DIRECTORY", tmp_path.resolve())
    model = nn.Conv2d(1, 2, 1)
    optimizer = torch.optim.SGD(model.parameters(), 0.01)
    state = FormalTrainerState(epoch=1, global_step=1, best_validation_dice=0.1, fold=0)
    config = {
        "run_type": "official_alignment_pending",
        "run_state": "official_alignment_pending",
        "model": _model_identity(saved_name, saved_mode),
    }
    path = tmp_path / f"cross-model-{saved_name}.pth"
    save_formal_checkpoint(model, optimizer, path, state, config)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["metadata"]["model_name"] == saved_name
    assert payload["metadata"]["supervision_mode"] == saved_mode

    restored = nn.Conv2d(1, 2, 1)
    load_called = False

    def fail_if_loaded(*_args: object, **_kwargs: object) -> None:
        nonlocal load_called
        load_called = True
        raise AssertionError("cross-model checkpoint must fail before model.load_state_dict")

    restored.load_state_dict = fail_if_loaded  # type: ignore[method-assign]
    restored_optimizer = torch.optim.SGD(restored.parameters(), 0.01)
    with pytest.raises(ValueError, match="model_name"):
        load_formal_checkpoint(
            restored,
            restored_optimizer,
            path,
            fold=0,
            model_name=expected_name,
            supervision_mode=expected_mode,
        )
    assert not load_called


def test_formal_checkpoint_round_trip_preserves_model_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(checkpoint_module, "PROJECT_OUTPUTS_DIRECTORY", tmp_path.resolve())
    model = nn.Conv2d(1, 2, 1)
    optimizer = torch.optim.SGD(model.parameters(), 0.01)
    state = FormalTrainerState(epoch=1, global_step=1, best_validation_dice=0.1, fold=0)
    config = {
        "run_type": "official_alignment_pending",
        "run_state": "official_alignment_pending",
        "model": _model_identity("h2former", "single_output"),
    }
    path = tmp_path / "same-model.pth"
    save_formal_checkpoint(model, optimizer, path, state, config)

    restored = nn.Conv2d(1, 2, 1)
    restored_optimizer = torch.optim.SGD(restored.parameters(), 0.01)
    result = load_formal_checkpoint(
        restored,
        restored_optimizer,
        path,
        fold=0,
        model_name="h2former",
        supervision_mode="single_output",
    )

    assert result.config["model"]["name"] == "h2former"
    assert result.config["model"]["supervision_mode"] == "single_output"


@pytest.mark.parametrize(
    ("saved_mode", "expected_mode"),
    [("deep_supervision", "single_output"), ("single_output", "deep_supervision")],
)
def test_plain_stage3_supervision_modes_are_isolated_before_state_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    saved_mode: str,
    expected_mode: str,
) -> None:
    monkeypatch.setattr(checkpoint_module, "PROJECT_OUTPUTS_DIRECTORY", tmp_path.resolve())
    model = nn.Conv2d(1, 2, 1)
    optimizer = torch.optim.SGD(model.parameters(), 0.01)
    state = FormalTrainerState(epoch=1, global_step=1, best_validation_dice=0.1, fold=0)
    config = {
        "run_type": "official_alignment_pending",
        "run_state": "official_alignment_pending",
        "model": {
            "name": "plain_conv_unet",
            "in_channels": 1,
            "num_classes": 2,
            "image_size": None,
            "supervision_mode": saved_mode,
            "deep_supervision": saved_mode == "deep_supervision",
            "loss_name": "DeepSupervisionLoss" if saved_mode == "deep_supervision" else "DiceCrossEntropyLoss",
        },
    }
    path = tmp_path / f"plain-{saved_mode}.pth"
    save_formal_checkpoint(model, optimizer, path, state, config)

    restored = nn.Conv2d(1, 2, 1)
    load_state_dict = Mock(wraps=restored.load_state_dict)
    restored.load_state_dict = load_state_dict  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="model_name|supervision"):
        load_formal_checkpoint(
            restored,
            torch.optim.SGD(restored.parameters(), 0.01),
            path,
            fold=0,
            model_name="plain_conv_unet",
            supervision_mode=expected_mode,
        )
    assert load_state_dict.call_count == 0


@pytest.mark.parametrize(
    ("model_name", "supervision_mode", "message"),
    [
        ("unknown_model", "single_output", "unsupported model_name"),
        ("h2former", "deep_supervision", "supervision_mode"),
    ],
)
def test_formal_checkpoint_validates_explicit_identity_without_nested_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_name: str,
    supervision_mode: str,
    message: str,
) -> None:
    monkeypatch.setattr(checkpoint_module, "PROJECT_OUTPUTS_DIRECTORY", tmp_path.resolve())
    model = nn.Conv2d(1, 2, 1)
    optimizer = torch.optim.SGD(model.parameters(), 0.01)
    state = FormalTrainerState(epoch=1, global_step=1, best_validation_dice=0.1, fold=0)

    with pytest.raises(ValueError, match=message):
        save_formal_checkpoint(
            model,
            optimizer,
            tmp_path / f"invalid-{model_name}.pth",
            state,
            {"run_type": "official_alignment_pending", "run_state": "official_alignment_pending"},
            model_name=model_name,
            supervision_mode=supervision_mode,
        )


def _write_minimal_checkpoint(path: Path, metadata: dict[str, object]) -> None:
    model = nn.Conv2d(1, 2, 1)
    torch.save(
        {
            "format_version": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": None,
            "metadata": metadata,
        },
        path,
    )


@pytest.mark.parametrize(
    ("case", "metadata", "expected", "should_load"),
    [
        (
            "top_plain_config_h2",
            {
                "model_name": "plain_conv_unet",
                "supervision_mode": "deep_supervision",
                "config": {"model": _model_identity("h2former", "single_output")},
            },
            {"model_name": "plain_conv_unet", "supervision_mode": "deep_supervision"},
            False,
        ),
        (
            "top_plain_resolved_h2",
            {
                "model_name": "plain_conv_unet",
                "supervision_mode": "deep_supervision",
                "resolved_config": {"model": _model_identity("h2former", "single_output")},
            },
            {"model_name": "plain_conv_unet", "supervision_mode": "deep_supervision"},
            False,
        ),
        (
            "top_h2_config_plain",
            {
                "model_name": "h2former",
                "supervision_mode": "single_output",
                "config": {"model": _model_identity("plain_conv_unet", "deep_supervision")},
            },
            {"model_name": "h2former", "supervision_mode": "single_output"},
            False,
        ),
        (
            "missing_top_config_h2_expected_plain",
            {"config": {"model": _model_identity("h2former", "single_output")}},
            {"model_name": "plain_conv_unet", "supervision_mode": "deep_supervision"},
            False,
        ),
        (
            "missing_top_config_h2_expected_h2",
            {"config": {"model": _model_identity("h2former", "single_output")}},
            {"model_name": "h2former", "supervision_mode": "single_output"},
            True,
        ),
        (
            "missing_top_resolved_h2_expected_h2",
            {"resolved_config": {"model": _model_identity("h2former", "single_output")}},
            {"model_name": "h2former", "supervision_mode": "single_output"},
            True,
        ),
        (
            "config_resolved_conflict",
            {
                "config": {"model": _model_identity("plain_conv_unet", "deep_supervision")},
                "resolved_config": {"model": _model_identity("h2former", "single_output")},
            },
            {"model_name": "h2former", "supervision_mode": "single_output"},
            False,
        ),
        (
            "true_legacy_expected_plain",
            {"run_state": "official_alignment_pending"},
            {"model_name": "plain_conv_unet", "supervision_mode": "deep_supervision"},
            True,
        ),
        (
            "true_legacy_expected_h2",
            {"run_state": "official_alignment_pending"},
            {"model_name": "h2former", "supervision_mode": "single_output"},
            False,
        ),
        (
            "partial_nested_identity_does_not_fallback",
            {"config": {"model": {"name": "h2former"}}},
            {"model_name": "plain_conv_unet", "supervision_mode": "deep_supervision"},
            False,
        ),
    ],
)
def test_checkpoint_identity_is_canonical_and_preload_safe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: str,
    metadata: dict[str, object],
    expected: dict[str, str],
    should_load: bool,
) -> None:
    monkeypatch.setattr(checkpoint_module, "PROJECT_OUTPUTS_DIRECTORY", tmp_path.resolve())
    path = tmp_path / f"identity-{case}.pth"
    _write_minimal_checkpoint(path, metadata)

    restored = nn.Conv2d(1, 2, 1)
    original_load = restored.load_state_dict
    load_state_dict = Mock(wraps=original_load)
    restored.load_state_dict = load_state_dict  # type: ignore[method-assign]

    if should_load:
        load_checkpoint(restored, None, path, expected)
        assert load_state_dict.call_count == 1
    else:
        with pytest.raises(ValueError):
            load_checkpoint(restored, None, path, expected)
        assert load_state_dict.call_count == 0
