from __future__ import annotations

from copy import deepcopy
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from standalone_nnunet2d.models.factory import (
    H2FORMER_LITE_UPERNET_W128 as A,
    H2FORMER_LITE_UPERNET_W128_PPM1236 as B,
    PPM1236_ARCHITECTURE,
    build_model,
    get_model_contract,
    resolve_checkpoint_model_identity,
)
from standalone_nnunet2d.training.official_config import OfficialTrainerSchedule, resolve_optimizer_config


def count(module):
    return sum(p.numel() for p in module.parameters())


def test_structure_parameters_and_config_identity():
    from standalone_nnunet2d.formal_train import build_formal_config, build_parser

    a = build_model(A)
    b = build_model(B)
    assert a.decoder.pool_scales == (1, 2, 4)
    assert b.decoder.pool_scales == (1, 2, 3, 6)
    assert [tuple(branch[0].output_size) for branch in b.decoder.ppm_branches] == [(1, 1), (2, 2), (3, 3), (6, 6)]
    assert all(branch[1].in_channels == 512 and branch[1].out_channels == 128
               and branch[1].bias is None and isinstance(branch[2], nn.ReLU)
               and len(branch) == 3 for branch in b.decoder.ppm_branches)
    assert b.decoder.ppm_bottleneck[0].in_channels == 1024
    assert b.decoder.ppm_bottleneck[0].out_channels == 128
    assert b.decoder.fusion[0].out_channels == 128
    assert count(a) - count(a.decoder) == count(b) - count(b.decoder)
    assert count(b) - count(a) == 212992
    schedule = OfficialTrainerSchedule(num_iterations_per_epoch=1, num_val_iterations_per_epoch=1)
    ac = build_formal_config(fold=0, epochs=1, schedule=schedule, model_name=A)
    bc = build_formal_config(fold=0, epochs=1, schedule=schedule, model_name=B)
    assert set(PPM1236_ARCHITECTURE).isdisjoint(ac["model"])
    assert all(bc["model"][key] == value for key, value in PPM1236_ARCHITECTURE.items())
    assert ac["plan_hash"] != bc["plan_hash"]
    assert resolve_optimizer_config(model_name=B, optimizer_name="adamw")["name"] == "AdamW"
    args = build_parser().parse_args(["--raw-root", "raw", "--plans", "plans.json", "--output-root", "out", "--model", B, "--resume", "out/checkpoint_latest.pth"])
    assert args.model == B and args.resume.name == "checkpoint_latest.pth"


@pytest.mark.parametrize("field", list(PPM1236_ARCHITECTURE))
def test_missing_and_conflicting_b_architecture_rejected(field):
    model = get_model_contract(B).as_dict()
    metadata = {"model_name": B, "supervision_mode": "single_output", "architecture": dict(PPM1236_ARCHITECTURE), "resolved_config": {"model": model}}
    assert resolve_checkpoint_model_identity(metadata) == (B, "single_output")
    absent = deepcopy(metadata)
    del absent["resolved_config"]["model"][field]
    with pytest.raises(ValueError, match="architecture"):
        resolve_checkpoint_model_identity(absent)
    wrong = deepcopy(metadata)
    wrong["resolved_config"]["model"][field] = (1, 2, 4, 6) if field == "ppm_scales" else 64
    with pytest.raises(ValueError, match="conflicts"):
        resolve_checkpoint_model_identity(wrong)


def test_same_shape_different_scales_rejects_before_target_load(tmp_path):
    from standalone_nnunet2d.engine.checkpoint import load_checkpoint, save_checkpoint
    source = nn.Conv2d(1, 2, 1)
    target = nn.Conv2d(1, 2, 1)
    metadata = {"model_name": B, "supervision_mode": "single_output", "architecture": dict(PPM1236_ARCHITECTURE), "resolved_config": {"model": get_model_contract(B).as_dict()}}
    path = tmp_path / "same-shape.pth"
    save_checkpoint(source, None, path, metadata, allowed_root=tmp_path)
    assert load_checkpoint(target, None, path, {"model_name": B, "supervision_mode": "single_output"}, allowed_root=tmp_path) == metadata
    assert all(torch.equal(source.state_dict()[k], v) for k, v in target.state_dict().items())
    bad = deepcopy(metadata)
    bad["resolved_config"]["model"]["ppm_scales"] = (1, 2, 4, 6)
    save_checkpoint(source, None, path, bad, allowed_root=tmp_path)
    spy = Mock(wraps=target.load_state_dict)
    target.load_state_dict = spy
    with pytest.raises(ValueError, match="conflicts"):
        load_checkpoint(target, None, path, {"model_name": B, "supervision_mode": "single_output"}, allowed_root=tmp_path)
    assert spy.call_count == 0


def test_prediction_semantic_rejection_before_model_build(tmp_path, monkeypatch):
    from standalone_nnunet2d import predict
    path = tmp_path / "wrong-scales.pth"
    metadata = {"model_name": B, "supervision_mode": "single_output", "architecture": dict(PPM1236_ARCHITECTURE), "resolved_config": {"model": get_model_contract(B).as_dict()}}
    metadata["resolved_config"]["model"]["ppm_scales"] = (1, 2, 4, 6)
    torch.save({"format_version": 1, "model_state_dict": nn.Conv2d(1, 2, 1).state_dict(), "metadata": metadata}, path)
    spy = Mock(side_effect=AssertionError("model built"))
    monkeypatch.setattr(predict, "build_model", spy)
    with pytest.raises(ValueError, match="conflicts"):
        predict._load_model(path, torch.device("cpu"))
    assert spy.call_count == 0


def test_train_mode_decoder_logits_backward_all_ppm_branches():
    from standalone_nnunet2d.models.lite_upernet import LiteUPerDecoder
    decoder = LiteUPerDecoder((64, 128, 256, 512), 2, fpn_channels=128, pool_scales=(1, 2, 3, 6))
    decoder.train()
    features = [torch.randn(shape, requires_grad=True) for shape in
                ((2, 64, 32, 32), (2, 128, 16, 16), (2, 256, 8, 8), (2, 512, 4, 4))]
    logits = decoder(features, output_size=(64, 64))
    assert logits.shape == (2, 2, 64, 64) and torch.isfinite(logits).all()
    logits.square().mean().backward()
    for branch in decoder.ppm_branches:
        grad = branch[1].weight.grad
        assert grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0
    assert features[-1].grad is not None and torch.count_nonzero(features[-1].grad) > 0


def test_full_b_finite_logits_at_512():
    model = build_model(B).eval()
    with torch.no_grad():
        logits = model(torch.zeros((1, 1, 512, 512)))
    assert logits.shape == (1, 2, 512, 512) and torch.isfinite(logits).all()
    assert tuple(shape[1] for shape in model.last_feature_shapes) == (64, 128, 256, 512)


def test_formal_b_checkpoint_round_trip_and_semantic_resume_guard(tmp_path):
    from standalone_nnunet2d.training.formal_checkpoint import FormalTrainerState, load_formal_checkpoint, save_formal_checkpoint
    source, target = nn.Conv2d(1, 2, 1), nn.Conv2d(1, 2, 1)
    optimizer = torch.optim.SGD(source.parameters(), lr=0.01)
    target_optimizer = torch.optim.SGD(target.parameters(), lr=0.01)
    state = FormalTrainerState(epoch=1, global_step=1, best_validation_dice=0.2, fold=0)
    config = {"run_type": "official_alignment_pending", "run_state": "official_alignment_pending",
              "model": get_model_contract(B).as_dict()}
    path = tmp_path / "b-formal.pth"
    save_formal_checkpoint(source, optimizer, path, state, config, model_name=B,
                           supervision_mode="single_output", checkpoint_root=tmp_path)
    result = load_formal_checkpoint(target, target_optimizer, path, fold=0, model_name=B,
                                    supervision_mode="single_output", checkpoint_root=tmp_path)
    assert result.state == state and result.config == config
    assert torch.equal(source.weight, target.weight)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["metadata"]["architecture"] == PPM1236_ARCHITECTURE
    payload["metadata"]["resolved_config"]["model"]["ppm_scales"] = (1, 2, 4, 6)
    torch.save(payload, path)
    spy = Mock(wraps=target.load_state_dict)
    target.load_state_dict = spy
    with pytest.raises(ValueError, match="conflicts"):
        load_formal_checkpoint(target, target_optimizer, path, fold=0, model_name=B,
                               supervision_mode="single_output", checkpoint_root=tmp_path)
    assert spy.call_count == 0


def test_prediction_loads_same_configuration_without_disk_checkpoint(monkeypatch, tmp_path):
    from standalone_nnunet2d import predict
    source = build_model(B)
    metadata = {"model_name": B, "supervision_mode": "single_output",
                "architecture": dict(PPM1236_ARCHITECTURE),
                "resolved_config": {"model": get_model_contract(B).as_dict()}}
    monkeypatch.setattr(predict, "_read_checkpoint", lambda path: (source.state_dict(), metadata))
    restored, loaded_metadata = predict._load_model(tmp_path / "unused.pth", torch.device("cpu"))
    assert restored.decoder.pool_scales == (1, 2, 3, 6)
    assert torch.equal(restored.decoder.classifier.weight, source.decoder.classifier.weight)
    assert loaded_metadata == metadata


def test_real_b_formal_checkpoint_prediction_and_same_shape_scale_guard(tmp_path, monkeypatch):
    from standalone_nnunet2d import predict
    from standalone_nnunet2d.training.formal_checkpoint import (
        FormalTrainerState, load_formal_checkpoint, save_formal_checkpoint,
    )
    from standalone_nnunet2d.training.official_config import PolyLRScheduler

    source = build_model(B)
    source_state = source.state_dict()
    representative_keys = [
        next(key for key in source_state if not key.startswith("decoder.")),
        *(f"decoder.ppm_branches.{index}.1.weight" for index in range(4)),
        "decoder.ppm_bottleneck.0.weight",
        "decoder.classifier.weight",
    ]
    assert all(key in source_state for key in representative_keys)
    optimizer = torch.optim.SGD(source.parameters(), lr=0.01, momentum=0.9)
    scheduler = PolyLRScheduler(optimizer, 0.01, 10)
    scheduler.step(2)
    scheduler._formal_last_step = 2
    state = FormalTrainerState(epoch=3, global_step=7, best_validation_dice=0.2, fold=0)
    config = {
        "run_type": "official_alignment_pending",
        "run_state": "official_alignment_pending",
        "model": get_model_contract(B).as_dict(),
    }
    path = tmp_path / "real-b-formal.pth"
    save_formal_checkpoint(
        source, optimizer, scheduler, path, state, config, model_name=B,
        supervision_mode="single_output", checkpoint_root=tmp_path,
    )

    target = build_model(B)
    target_optimizer = torch.optim.SGD(target.parameters(), lr=0.01, momentum=0.9)
    target_scheduler = PolyLRScheduler(target_optimizer, 0.01, 10)
    restored = load_formal_checkpoint(
        target, target_optimizer, target_scheduler, path, fold=0, model_name=B,
        supervision_mode="single_output", checkpoint_root=tmp_path,
    )
    assert restored.state == state
    assert restored.config == config
    assert restored.config["model"] == get_model_contract(B).as_dict()
    assert restored.scheduler_step == 2
    assert target_scheduler.ctr == 3
    assert target_scheduler.get_last_lr() == scheduler.get_last_lr()
    assert target_optimizer.state_dict() == optimizer.state_dict()
    assert target.decoder.pool_scales == (1, 2, 3, 6)
    for key in representative_keys:
        assert torch.equal(target.state_dict()[key], source_state[key]), key

    predicted, metadata = predict._load_model(path, torch.device("cpu"))
    assert metadata["model_name"] == B
    assert metadata["supervision_mode"] == "single_output"
    assert metadata["architecture"] == PPM1236_ARCHITECTURE
    assert metadata["resolved_config"] == config
    assert predicted.decoder.pool_scales == (1, 2, 3, 6)
    for key in representative_keys:
        assert torch.equal(predicted.state_dict()[key], source_state[key]), key

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert set(payload["model_state_dict"]) == set(target.state_dict())
    assert all(
        payload["model_state_dict"][key].shape == value.shape
        for key, value in target.state_dict().items()
    )
    payload["metadata"]["resolved_config"]["model"]["ppm_scales"] = (1, 2, 4, 6)
    assert payload["metadata"]["architecture"]["ppm_scales"] == (1, 2, 3, 6)
    torch.save(payload, path)

    load_spy = Mock(wraps=target.load_state_dict)
    monkeypatch.setattr(target, "load_state_dict", load_spy)
    with pytest.raises(ValueError, match="conflicts"):
        load_formal_checkpoint(
            target, target_optimizer, target_scheduler, path, fold=0, model_name=B,
            supervision_mode="single_output", checkpoint_root=tmp_path,
        )
    assert load_spy.call_count == 0
    build_spy = Mock(side_effect=AssertionError("model built before metadata rejection"))
    monkeypatch.setattr(predict, "build_model", build_spy)
    with pytest.raises(ValueError, match="conflicts"):
        predict._load_model(path, torch.device("cpu"))
    assert build_spy.call_count == 0
