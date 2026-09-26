from __future__ import annotations

import pytest
import torch
from torch import nn

from standalone_nnunet2d.models.factory import (
    H2FORMER_LITE_UPERNET,
    H2FORMER_LITE_UPERNET_W128,
    build_model,
    get_model_contract,
)
from standalone_nnunet2d.models.h2former_lite_upernet import H2FormerLiteUPerNet
from standalone_nnunet2d.training.official_config import resolve_optimizer_config


def _count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def test_width_128_changes_only_decoder_capacity() -> None:
    a = build_model(H2FORMER_LITE_UPERNET)
    b = build_model(H2FORMER_LITE_UPERNET_W128)
    assert isinstance(a, H2FormerLiteUPerNet)
    assert isinstance(b, H2FormerLiteUPerNet)
    assert a.decoder.ppm_branches[0][1].out_channels == 64
    assert b.decoder.ppm_branches[0][1].out_channels == 128
    assert a.decoder.classifier.in_channels == 64
    assert b.decoder.classifier.in_channels == 128
    assert all(branch[1].out_channels == 128 for branch in b.decoder.ppm_branches)
    assert b.decoder.ppm_bottleneck[0].out_channels == 128
    assert all(layer[0].out_channels == 128 for layer in b.decoder.lateral_projections)
    assert all(layer[0].out_channels == 128 for layer in b.decoder.refinements)
    assert b.decoder.fusion[0].out_channels == 128
    assert all(isinstance(layer[1], nn.BatchNorm2d) and layer[1].num_features == 128 for layer in (*b.decoder.lateral_projections, *b.decoder.refinements, b.decoder.ppm_bottleneck, b.decoder.fusion))
    assert {k: v.shape for k, v in a.state_dict().items() if not k.startswith('decoder.')} == {k: v.shape for k, v in b.state_dict().items() if not k.startswith('decoder.')}
    assert _count(a) - _count(a.decoder) == _count(b) - _count(b.decoder)
    assert _count(a.decoder) < _count(b.decoder)
    assert _count(a) < _count(b)
    assert get_model_contract(H2FORMER_LITE_UPERNET_W128).supervision_mode == 'single_output'
    assert resolve_optimizer_config(model_name=H2FORMER_LITE_UPERNET_W128, optimizer_name='adamw')['name'] == 'AdamW'


@pytest.mark.parametrize('name', [H2FORMER_LITE_UPERNET, H2FORMER_LITE_UPERNET_W128])
def test_width_variant_512_logits_and_gradients(name: str) -> None:
    model = build_model(name).eval()
    x = torch.zeros((1, 1, 512, 512))
    x[:, :, 256, 256] = 1
    logits = model(x)
    assert logits.shape == (1, 2, 512, 512)
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert model.conv1.weight.grad is not None
    assert model.decoder.classifier.weight.grad is not None
    assert torch.isfinite(model.decoder.classifier.weight.grad).all()

@pytest.mark.parametrize('saved_name,target_name', [
    (H2FORMER_LITE_UPERNET, H2FORMER_LITE_UPERNET_W128),
    (H2FORMER_LITE_UPERNET_W128, H2FORMER_LITE_UPERNET),
])
def test_cross_width_resume_rejects_before_load(tmp_path, saved_name: str, target_name: str) -> None:
    from standalone_nnunet2d.engine.checkpoint import load_checkpoint, save_checkpoint

    source = nn.Conv2d(1, 2, 1)
    target = nn.Conv2d(1, 2, 1)
    path = tmp_path / 'checkpoint.pth'
    save_checkpoint(source, None, path, {'model_name': saved_name, 'supervision_mode': 'single_output'}, allowed_root=tmp_path)
    called = False

    def fail_if_loaded(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError('target load_state_dict called')

    target.load_state_dict = fail_if_loaded
    with pytest.raises(ValueError, match='identity'):
        load_checkpoint(target, None, path, {'model_name': target_name, 'supervision_mode': 'single_output'}, allowed_root=tmp_path)
    assert not called


@pytest.mark.parametrize('saved_name,target_name', [
    (H2FORMER_LITE_UPERNET, H2FORMER_LITE_UPERNET_W128),
    (H2FORMER_LITE_UPERNET_W128, H2FORMER_LITE_UPERNET),
])
def test_cross_width_prediction_metadata_conflict_rejects_before_load(tmp_path, saved_name: str, target_name: str, monkeypatch) -> None:
    from standalone_nnunet2d import predict

    path = tmp_path / 'checkpoint.pth'
    torch.save({'format_version': 1, 'model_state_dict': nn.Conv2d(1, 2, 1).state_dict(),
                'metadata': {'model_name': saved_name, 'supervision_mode': 'single_output',
                             'resolved_config': {'model': {'name': target_name, 'supervision_mode': 'single_output'}}}}, path)
    called = False

    def fail_if_built(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError('target model built')

    monkeypatch.setattr(predict, 'build_model', fail_if_built)
    with pytest.raises(ValueError, match='identity'):
        predict._load_model(path, torch.device('cpu'))
    assert not called


def test_state_shape_mismatch_rejects_before_target_load(tmp_path) -> None:
    from standalone_nnunet2d.engine.checkpoint import load_checkpoint, save_checkpoint

    source = nn.Conv2d(1, 2, 1)
    target = nn.Conv2d(1, 3, 1)
    path = tmp_path / 'checkpoint.pth'
    metadata = {'model_name': H2FORMER_LITE_UPERNET_W128, 'supervision_mode': 'single_output'}
    save_checkpoint(source, None, path, metadata, allowed_root=tmp_path)
    called = False

    def fail_if_loaded(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError('target load_state_dict called')

    target.load_state_dict = fail_if_loaded
    with pytest.raises(ValueError, match='shape/type mismatch'):
        load_checkpoint(target, None, path, metadata, allowed_root=tmp_path)
    assert not called


def test_same_variant_checkpoint_restores_weights(tmp_path) -> None:
    from standalone_nnunet2d.engine.checkpoint import load_checkpoint, save_checkpoint

    source = nn.Conv2d(1, 2, 1)
    target = nn.Conv2d(1, 2, 1)
    metadata = {'model_name': H2FORMER_LITE_UPERNET_W128, 'supervision_mode': 'single_output'}
    path = tmp_path / 'checkpoint.pth'
    save_checkpoint(source, None, path, metadata, allowed_root=tmp_path)
    assert load_checkpoint(target, None, path, metadata, allowed_root=tmp_path) == metadata
    assert all(torch.equal(source.state_dict()[key], value) for key, value in target.state_dict().items())


def test_resolved_config_and_parser_distinguish_width_variants() -> None:
    from standalone_nnunet2d.formal_train import build_formal_config, build_parser
    from standalone_nnunet2d.training.official_config import OfficialTrainerSchedule

    schedule = OfficialTrainerSchedule(num_iterations_per_epoch=1, num_val_iterations_per_epoch=1)
    a = build_formal_config(fold=0, epochs=1, schedule=schedule, model_name=H2FORMER_LITE_UPERNET)
    b = build_formal_config(fold=0, epochs=1, schedule=schedule, model_name=H2FORMER_LITE_UPERNET_W128)
    assert a['model']['name'] == H2FORMER_LITE_UPERNET
    assert b['model']['name'] == H2FORMER_LITE_UPERNET_W128
    assert a['plan_hash'] != b['plan_hash']
    parser = build_parser()
    args = parser.parse_args(['--raw-root', 'raw', '--plans', 'plans.json', '--output-root', 'out', '--model', H2FORMER_LITE_UPERNET_W128, '--resume', 'out/checkpoint_latest.pth'])
    assert args.model == H2FORMER_LITE_UPERNET_W128
    assert args.resume.name == 'checkpoint_latest.pth'

@pytest.fixture(scope="module")
def real_width_checkpoint(tmp_path_factory):
    """One real formal B artifact, shared by the save/restore and prediction checks."""
    from standalone_nnunet2d.training.formal_checkpoint import FormalTrainerState, save_formal_checkpoint
    from standalone_nnunet2d.training.official_config import PolyLRScheduler

    root = tmp_path_factory.mktemp("width128-real")
    model = build_model(H2FORMER_LITE_UPERNET_W128)
    optimizer = torch.optim.SGD(model.parameters(), 0.01)
    scheduler = PolyLRScheduler(optimizer, 0.01, 10)
    scheduler.step(2)
    state = FormalTrainerState(epoch=3, global_step=2, best_validation_dice=0.25, fold=0)
    config = {"run_type": "official_alignment_pending", "run_state": "official_alignment_pending",
              "model": get_model_contract(H2FORMER_LITE_UPERNET_W128).as_dict()}
    path = root / "b-formal.pth"
    save_formal_checkpoint(model, optimizer, scheduler, path, state, config,
                           checkpoint_root=root)
    yield path, model, state, config


def test_real_b_formal_round_trip_and_prediction_load(real_width_checkpoint):
    from standalone_nnunet2d import predict
    from standalone_nnunet2d.training.formal_checkpoint import load_formal_checkpoint
    from standalone_nnunet2d.training.official_config import PolyLRScheduler

    path, source, state, config = real_width_checkpoint
    restored = build_model(H2FORMER_LITE_UPERNET_W128)
    optimizer = torch.optim.SGD(restored.parameters(), 0.01)
    scheduler = PolyLRScheduler(optimizer, 0.01, 10)
    result = load_formal_checkpoint(restored, optimizer, scheduler, path, fold=0,
                                    model_name=H2FORMER_LITE_UPERNET_W128,
                                    supervision_mode="single_output", checkpoint_root=path.parent)
    assert result.state == state
    assert result.config == config
    assert result.scheduler_step == 2
    assert torch.equal(restored.decoder.classifier.weight, source.decoder.classifier.weight)
    predicted, metadata = predict._load_model(path, torch.device("cpu"))
    assert isinstance(predicted, H2FormerLiteUPerNet)
    assert metadata["model_name"] == H2FORMER_LITE_UPERNET_W128
    assert torch.equal(predicted.decoder.classifier.weight, source.decoder.classifier.weight)


@pytest.mark.parametrize("source_name,target_name", [
    (H2FORMER_LITE_UPERNET, H2FORMER_LITE_UPERNET_W128),
    (H2FORMER_LITE_UPERNET_W128, H2FORMER_LITE_UPERNET),
])
def test_real_width_weights_rejected_before_formal_target_load(
    real_width_checkpoint, monkeypatch, source_name, target_name
):
    from unittest.mock import Mock
    from standalone_nnunet2d.engine import checkpoint
    from standalone_nnunet2d.training.formal_checkpoint import load_formal_checkpoint

    path, b_model, _, _ = real_width_checkpoint
    source = b_model if source_name == H2FORMER_LITE_UPERNET_W128 else build_model(source_name)
    target = build_model(target_name)
    mismatch = next(key for key, value in source.state_dict().items()
                    if key.startswith("decoder.") and value.shape != target.state_dict()[key].shape)
    assert mismatch.startswith("decoder.")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["model_state_dict"] = source.state_dict()
    payload["metadata"]["model_name"] = target_name
    payload["metadata"]["resolved_config"]["model"] = get_model_contract(target_name).as_dict()
    payload["metadata"]["config"]["model"] = get_model_contract(target_name).as_dict()
    monkeypatch.setattr(checkpoint.torch, "load", lambda *a, **k: payload)
    spy = Mock(wraps=target.load_state_dict)
    monkeypatch.setattr(target, "load_state_dict", spy)
    with pytest.raises(ValueError, match="shape/type mismatch"):
        load_formal_checkpoint(target, torch.optim.SGD(target.parameters(), 0.01), path,
                               fold=0, model_name=target_name, supervision_mode="single_output",
                               checkpoint_root=path.parent)
    assert spy.call_count == 0


@pytest.mark.parametrize("source_name,target_name", [
    (H2FORMER_LITE_UPERNET, H2FORMER_LITE_UPERNET_W128),
    (H2FORMER_LITE_UPERNET_W128, H2FORMER_LITE_UPERNET),
])
def test_real_width_weights_rejected_before_prediction_target_load(
    real_width_checkpoint, monkeypatch, source_name, target_name
):
    from unittest.mock import Mock
    from standalone_nnunet2d import predict

    path, b_model, _, _ = real_width_checkpoint
    source = b_model if source_name == H2FORMER_LITE_UPERNET_W128 else build_model(source_name)
    target_probe = build_model(target_name)
    assert any(key.startswith("decoder.") and value.shape != target_probe.state_dict()[key].shape
               for key, value in source.state_dict().items())
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["model_state_dict"] = source.state_dict()
    payload["metadata"]["model_name"] = target_name
    payload["metadata"]["resolved_config"]["model"] = get_model_contract(target_name).as_dict()
    payload["metadata"]["config"]["model"] = get_model_contract(target_name).as_dict()
    monkeypatch.setattr(predict.torch, "load", lambda *a, **k: payload)
    real_build = predict.build_model
    calls = []

    def recording_build(*args, **kwargs):
        model = real_build(*args, **kwargs)
        spy = Mock(wraps=model.load_state_dict)
        monkeypatch.setattr(model, "load_state_dict", spy)
        calls.append(spy)
        return model

    monkeypatch.setattr(predict, "build_model", recording_build)
    with pytest.raises(ValueError, match="shape/type mismatch"):
        predict._load_model(path, torch.device("cpu"))
    assert len(calls) == 1
    assert calls[0].call_count == 0
