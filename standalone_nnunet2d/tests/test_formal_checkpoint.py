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
