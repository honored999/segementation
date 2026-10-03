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


@pytest.fixture(autouse=True)
def _isolate_default_checkpoint_output(tmp_path, monkeypatch):
    # Preserve the default-root guard while isolating synthetic artifacts on D:.
    from standalone_nnunet2d.engine import checkpoint as engine_checkpoint
    root = tmp_path / "default-checkpoint-output"
    monkeypatch.setattr(engine_checkpoint, "PROJECT_OUTPUTS_DIRECTORY", root)
    monkeypatch.setitem(globals(), "PROJECT_OUTPUTS_DIRECTORY", root)


class _OptimizerContractModel(nn.Module):
    def __init__(self, value=1.0, scalar=False):
        super().__init__()
        self.weight = nn.Parameter(torch.full(() if scalar else (2, 3), value))
        self.other = nn.Parameter(torch.full((2,), value))
        self.register_buffer('running', torch.full((2,), value))


def _optimizer_contract_objects(kind, value=1.0, *, scalar=False, initialized=True, momentum=.9, amsgrad=False):
    model = _OptimizerContractModel(value, scalar)
    groups = [{'params': [model.weight], 'lr': .01}, {'params': [model.other], 'lr': .02}]
    optimizer = (torch.optim.AdamW(groups, amsgrad=amsgrad) if kind == 'adamw'
                 else torch.optim.SGD(groups, momentum=momentum, nesterov=momentum > 0))
    if initialized:
        (model.weight.sum() + model.other.sum()).backward()
        optimizer.step()
        optimizer.zero_grad()
    scheduler = PolyLRScheduler(optimizer, 1000)
    scheduler.step(7)
    return model, optimizer, scheduler


def _optimizer_contract_snapshot(model, optimizer, scheduler):
    from standalone_nnunet2d.training import formal_checkpoint as ck
    return deepcopy((model.state_dict(), optimizer.state_dict(),
                     {k: v for k, v in scheduler.__dict__.items() if k != 'optimizer'}, ck.capture_rng_state()))


def _assert_optimizer_contract_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert actual.dtype == expected.dtype and torch.equal(actual, expected)
    elif isinstance(expected, np.ndarray):
        assert np.array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_optimizer_contract_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            _assert_optimizer_contract_equal(a, e)
    else:
        assert actual == expected


@pytest.mark.parametrize('kind,bad', [
    ('adamw', 'missing_exp_avg'), ('adamw', 'scalar_exp_avg'), ('adamw', 'bad_betas'),
    ('adamw', 'missing_step'), ('adamw', 'bad_step'), ('adamw', 'bad_eps'),
    ('adamw', 'missing_max'), ('adamw', 'bad_max'), ('adamw', 'bad_sq'),
    ('adamw', 'bad_flag'), ('adamw', 'bad_combo'),
    ('sgd', 'bad_buffer'), ('sgd', 'bad_momentum'), ('sgd', 'bad_dampening'),
    ('sgd', 'bad_nesterov'), ('sgd', 'bad_nesterov_combo'),
    ('sgd', 'bad_lr'), ('sgd', 'bad_weight_decay'),
    ('adamw', 'duplicate_ref'), ('sgd', 'unknown_ref'), ('adamw', 'unknown_field'),
    ('sgd', 'missing_lr'), ('adamw', 'wrong_state_type'),
    ('adamw', 'beta_range'), ('adamw', 'beta_length'), ('adamw', 'eps_nan'),
    ('adamw', 'step_vector'), ('adamw', 'step_negative'), ('adamw', 'moment_dtype'),
    ('adamw', 'missing_sq'), ('adamw', 'missing_betas'), ('adamw', 'bad_foreach'),
    ('adamw', 'capturable'), ('adamw', 'differentiable'),
    ('sgd', 'momentum_negative'), ('sgd', 'dampening_negative'), ('sgd', 'lr_boolean'),
    ('sgd', 'bad_maximize'), ('sgd', 'bad_fused'), ('sgd', 'unknown_sgd_field'),
    ('sgd', 'group_length'), ('adamw', 'state_boolean_ref'), ('adamw', 'group_boolean_ref'),
])
def test_optimizer_schema_rejects_before_any_restore(tmp_path, kind, bad):
    from standalone_nnunet2d.training import formal_checkpoint as ck
    source, opt, sched = _optimizer_contract_objects(kind, 7.0, amsgrad=bad in ('missing_max', 'bad_max'))
    path = tmp_path / 'optimizer-schema.pth'
    save_formal_checkpoint(source, opt, sched, path, FormalTrainerState(8, 2000, .8, 0), {}, checkpoint_root=tmp_path)
    payload, _ = ck.read_formal_payload(path)
    state = payload['optimizer_state_dict']
    values = state['state'][0]
    group = state['param_groups'][0]
    if bad == 'missing_exp_avg': del values['exp_avg']
    elif bad == 'scalar_exp_avg': values['exp_avg'] = torch.tensor(0.)
    elif bad == 'bad_betas': group['betas'] = ('invalid', .999)
    elif bad == 'missing_step': del values['step']
    elif bad == 'bad_step': values['step'] = torch.tensor(float('nan'))
    elif bad == 'bad_eps': group['eps'] = -1.
    elif bad == 'missing_max': del values['max_exp_avg_sq']
    elif bad == 'bad_max': values['max_exp_avg_sq'] = torch.tensor(0.)
    elif bad == 'bad_sq': values['exp_avg_sq'].fill_(-1)
    elif bad == 'bad_flag': group['amsgrad'] = 'false'
    elif bad == 'bad_combo': group.update(foreach=True, fused=True)
    elif bad == 'bad_buffer': values['momentum_buffer'] = torch.tensor(0.)
    elif bad == 'bad_momentum': group['momentum'] = 'invalid'
    elif bad == 'bad_dampening': group['dampening'] = float('inf')
    elif bad == 'bad_nesterov': group['nesterov'] = 'false'
    elif bad == 'bad_nesterov_combo': group['momentum'] = 0.
    elif bad == 'bad_lr': group['lr'] = float('nan')
    elif bad == 'bad_weight_decay': group['weight_decay'] = -1.
    elif bad == 'duplicate_ref': state['param_groups'][1]['params'] = [0]
    elif bad == 'unknown_ref': state['state'][99] = {}
    elif bad == 'unknown_field': values['unexpected'] = torch.tensor(0.)
    elif bad == 'missing_lr': del group['lr']
    elif bad == 'wrong_state_type': state['state'][0] = []
    elif bad == 'beta_range': group['betas'] = (1., .999)
    elif bad == 'beta_length': group['betas'] = (.9,)
    elif bad == 'eps_nan': group['eps'] = float('nan')
    elif bad == 'step_vector': values['step'] = torch.ones(2)
    elif bad == 'step_negative': values['step'] = -1.
    elif bad == 'moment_dtype': values['exp_avg'] = values['exp_avg'].to(torch.int64)
    elif bad == 'missing_sq': del values['exp_avg_sq']
    elif bad == 'missing_betas': del group['betas']
    elif bad == 'bad_foreach': group['foreach'] = 'invalid'
    elif bad == 'capturable': group['capturable'] = True
    elif bad == 'differentiable': group['differentiable'] = True
    elif bad == 'momentum_negative': group['momentum'] = -1.
    elif bad == 'dampening_negative': group['dampening'] = -1.
    elif bad == 'lr_boolean': group['lr'] = True
    elif bad == 'bad_maximize': group['maximize'] = 1
    elif bad == 'bad_fused': group['fused'] = 'invalid'
    elif bad == 'unknown_sgd_field': values['unexpected'] = torch.zeros(())
    elif bad == 'group_length': group['params'] = []
    elif bad == 'state_boolean_ref': state['state'] = {False: values}
    elif bad == 'group_boolean_ref': group['params'] = [False]
    torch.save(payload, path)
    model, optimizer, scheduler = _optimizer_contract_objects(kind)
    before = _optimizer_contract_snapshot(model, optimizer, scheduler)
    refs = [list(g['params']) for g in optimizer.param_groups]
    calls = []
    handle = model.register_load_state_dict_pre_hook(lambda *args: calls.append('model'))
    try:
        with pytest.raises(ValueError, match='optimizer'):
            load_formal_checkpoint(model, optimizer, scheduler, path, fold=0, checkpoint_root=tmp_path)
    finally:
        handle.remove()
    assert calls == []
    assert scheduler.optimizer is optimizer
    assert all(a is b for g, old in zip(optimizer.param_groups, refs) for a, b in zip(g['params'], old))
    _assert_optimizer_contract_equal(_optimizer_contract_snapshot(model, optimizer, scheduler), before)


@pytest.mark.parametrize('kind,scalar,initialized,historical,momentum,amsgrad', [
    ('adamw', False, True, False, .9, False), ('adamw', True, True, False, .9, False),
    ('adamw', False, False, False, .9, False), ('adamw', False, True, True, .9, False),
    ('adamw', False, True, False, .9, True),
    ('sgd', False, True, False, .9, False), ('sgd', True, True, False, .9, False),
    ('sgd', False, False, False, .9, False), ('sgd', False, True, True, .9, False),
    ('sgd', False, True, False, 0., False), ('sgd', False, True, True, 0., False),
])
def test_optimizer_schema_valid_resume_and_synthetic_step(tmp_path, kind, scalar, initialized, historical, momentum, amsgrad):
    from standalone_nnunet2d.training import formal_checkpoint as ck
    source, opt, sched = _optimizer_contract_objects(kind, 7.0, scalar=scalar, initialized=initialized, momentum=momentum, amsgrad=amsgrad)
    path = tmp_path / 'optimizer-valid.pth'
    save_formal_checkpoint(source, opt, sched, path, FormalTrainerState(8, 2000, .8, 0), {}, checkpoint_root=tmp_path)
    if historical:
        payload, _ = ck.read_formal_payload(path)
        for group in payload['optimizer_state_dict']['param_groups']:
            for key in ('maximize', 'foreach', 'capturable', 'differentiable', 'fused', 'decoupled_weight_decay'):
                group.pop(key, None)
            if kind == 'adamw': group.pop('amsgrad', None)
            elif momentum == 0: group.pop('nesterov', None)
        if kind == 'adamw':
            for values in payload['optimizer_state_dict']['state'].values(): values['step'] = values['step'].item()
        torch.save(payload, path)
    model, optimizer, scheduler = _optimizer_contract_objects(kind, scalar=scalar, momentum=momentum, amsgrad=amsgrad)
    load_formal_checkpoint(model, optimizer, scheduler, path, fold=0, checkpoint_root=tmp_path)
    _assert_optimizer_contract_equal(model.state_dict(), source.state_dict())
    assert scheduler.optimizer is optimizer and scheduler.ctr == 8
    assert optimizer.param_groups[0]['lr'] != optimizer.param_groups[1]['lr']
    (model.weight.sum() + model.other.sum()).backward()
    (source.weight.sum() + source.other.sum()).backward()
    optimizer.step(); opt.step()
    _assert_optimizer_contract_equal(model.state_dict(), source.state_dict())
    for a, b in zip(optimizer.state.values(), opt.state.values()):
        for key in b: _assert_optimizer_contract_equal(a[key], b[key])


@pytest.mark.parametrize('kind', ['adamw', 'sgd'])
@pytest.mark.parametrize('empty_entry', [False, True])
def test_optimizer_schema_partial_lazy_state(tmp_path, kind, empty_entry, monkeypatch):
    from standalone_nnunet2d.training import formal_checkpoint as ck
    source, opt, sched = _optimizer_contract_objects(kind, 7.0)
    # Only the first parameter has been initialized in this valid checkpoint.
    if empty_entry: opt.state[source.other] = {}
    else: del opt.state[source.other]
    path = tmp_path / 'partial.pth'
    save_formal_checkpoint(source, opt, sched, path, FormalTrainerState(8, 2000, .8, 0), {}, checkpoint_root=tmp_path)
    model, optimizer, scheduler = _optimizer_contract_objects(kind)
    def forbidden(*args, **kwargs):
        raise AssertionError('restore preflight must never execute optimizer.step')
    with monkeypatch.context() as scoped:
        scoped.setattr(type(optimizer), 'step', forbidden)
        load_formal_checkpoint(model, optimizer, scheduler, path, fold=0, checkpoint_root=tmp_path)
    (model.weight.sum() + model.other.sum()).backward()
    (source.weight.sum() + source.other.sum()).backward()
    optimizer.step(); opt.step()
    _assert_optimizer_contract_equal(model.state_dict(), source.state_dict())
    _assert_optimizer_contract_equal(optimizer.state_dict(), opt.state_dict())


@pytest.mark.parametrize('kind,field', [
    ('adamw', 'exp_avg'), ('adamw', 'exp_avg_sq'),
    ('adamw', 'max_exp_avg_sq'), ('sgd', 'momentum_buffer'),
])
@pytest.mark.parametrize('layout', ['expanded', 'positive_overlap'])
def test_optimizer_internal_overlap_rejected_atomically(tmp_path, monkeypatch, kind, field, layout):
    from standalone_nnunet2d.training import formal_checkpoint as ck
    source, opt, sched = _optimizer_contract_objects(kind, 7.0, amsgrad=field == 'max_exp_avg_sq')
    path = tmp_path / 'overlap.pth'
    save_formal_checkpoint(source, opt, sched, path, FormalTrainerState(8, 2000, .8, 0), {}, checkpoint_root=tmp_path)
    payload, _ = ck.read_formal_payload(path)
    damaged = (torch.tensor(1.).expand(2, 3) if layout == 'expanded'
               else torch.ones(4).as_strided((2, 3), (1, 1)))
    payload['optimizer_state_dict']['state'][0][field] = damaged
    torch.save(payload, path)
    loaded, _ = ck.read_formal_payload(path)
    assert loaded['optimizer_state_dict']['state'][0][field].stride() == damaged.stride()
    model, optimizer, scheduler = _optimizer_contract_objects(kind, amsgrad=field == 'max_exp_avg_sq')
    before = _optimizer_contract_snapshot(model, optimizer, scheduler)
    refs = [list(g['params']) for g in optimizer.param_groups]
    calls = []
    handle = model.register_load_state_dict_pre_hook(lambda *args: calls.append('model'))
    def forbidden(*args, **kwargs):
        raise AssertionError('preflight must not step')
    try:
        with monkeypatch.context() as scoped:
            scoped.setattr(type(optimizer), 'step', forbidden)
            with pytest.raises(ValueError, match='optimizer.*overlap'):
                load_formal_checkpoint(model, optimizer, scheduler, path, fold=0, checkpoint_root=tmp_path)
    finally:
        handle.remove()
    assert calls == []
    assert scheduler.optimizer is optimizer
    assert all(a is b for g, old in zip(optimizer.param_groups, refs) for a, b in zip(g['params'], old))
    _assert_optimizer_contract_equal(_optimizer_contract_snapshot(model, optimizer, scheduler), before)


@pytest.mark.parametrize('kind,amsgrad', [('adamw', False), ('adamw', True), ('sgd', False)])
@pytest.mark.parametrize('layout', ['transpose', 'slice', 'interleaved'])
def test_optimizer_noncontiguous_resume_and_step(tmp_path, monkeypatch, kind, amsgrad, layout):
    from standalone_nnunet2d.training import formal_checkpoint as ck
    source, opt, sched = _optimizer_contract_objects(kind, 7.0, amsgrad=amsgrad)
    path = tmp_path / 'noncontiguous.pth'
    save_formal_checkpoint(source, opt, sched, path, FormalTrainerState(8, 2000, .8, 0), {}, checkpoint_root=tmp_path)
    payload, _ = ck.read_formal_payload(path)
    values = payload['optimizer_state_dict']['state'][0]
    for field, value in list(values.items()):
        if field == 'step': continue
        if layout == 'transpose': replacement = torch.empty(3, 2).t()
        elif layout == 'slice': replacement = torch.empty(2, 6)[:, ::2]
        else: replacement = torch.empty(8).as_strided((2, 3), (3, 2))
        replacement.copy_(value)
        values[field] = replacement
    torch.save(payload, path)
    loaded, _ = ck.read_formal_payload(path)
    for field, value in values.items():
        if field != 'step':
            assert not value.is_contiguous()
            assert loaded['optimizer_state_dict']['state'][0][field].stride() == value.stride()
    model, optimizer, scheduler = _optimizer_contract_objects(kind, amsgrad=amsgrad)
    def forbidden(*args, **kwargs):
        raise AssertionError('preflight must not step')
    with monkeypatch.context() as scoped:
        scoped.setattr(type(optimizer), 'step', forbidden)
        load_formal_checkpoint(model, optimizer, scheduler, path, fold=0, checkpoint_root=tmp_path)
    for field, value in values.items():
        if field != 'step': assert optimizer.state[model.weight][field].stride() == value.stride()
    _assert_optimizer_contract_equal(model.state_dict(), source.state_dict())
    (model.weight.sum() + model.other.sum()).backward()
    (source.weight.sum() + source.other.sum()).backward()
    optimizer.step()
    opt.step()
    _assert_optimizer_contract_equal(model.state_dict(), source.state_dict())
    _assert_optimizer_contract_equal(optimizer.state_dict(), opt.state_dict())
