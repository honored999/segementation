"""Tiny CPU contracts; never construct a production network."""
from pathlib import Path
from types import SimpleNamespace
import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nnUNetTrainerTopK10EarlyStopping import nnUNetTrainerTopK10EarlyStopping as Trainer
from nnUNetTrainerEarlyStopping import nnUNetTrainerEarlyStopping as Early
from nnUNetTrainerTopK10 import nnUNetTrainerTopK10 as TopK
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer as Base
from nnunetv2.training.logging.nnunet_logger import MetaLogger
from nnunetv2.training.loss.compound_losses import DC_and_topk_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper


def state(tmp_path):
    t = object.__new__(Trainer)
    t.current_epoch = 0
    t.logger = MetaLogger(str(tmp_path), False)
    t._early_stopping_best_ema = None
    t._epochs_without_improvement = 0
    t._early_stopping_triggered = False
    t.print_to_log_file = lambda *a, **k: None
    return t


def test_cooperative_init_and_inherited_contract(monkeypatch):
    calls = []
    def init(self, *args):
        calls.append(args)
        self.enable_deep_supervision = True
    monkeypatch.setattr(Base, '__init__', init)
    t = Trainer({}, '2d', 0, {}, torch.device('cpu'))
    assert len(calls) == 1
    assert t.enable_deep_supervision and t.num_epochs == 1000
    assert t._early_stopping_checkpoint_state() == {
        'best_monitored_ema_dice': None, 'epochs_without_improvement': 0, 'triggered': False}
    assert Trainer.__mro__[:4] == (Trainer, Early, TopK, Base)
    for name in ('build_network_architecture', 'configure_optimizers', 'get_dataloaders',
                 'get_training_transforms', 'configure_rotation_dummyDA_mirroring_and_inital_patch_size'):
        assert getattr(Trainer, name) == getattr(TopK, name)
    for name in ('run_training', 'save_checkpoint', 'load_checkpoint', 'on_epoch_end'):
        assert getattr(Trainer, name) == getattr(Early, name)
    assert Trainer._build_loss is TopK._build_loss


def test_actual_topk10_and_deep_supervision(tmp_path):
    t = state(tmp_path)
    t.label_manager = SimpleNamespace(has_regions=False, ignore_label=None)
    t.configuration_manager = SimpleNamespace(batch_dice=False)
    t.is_ddp = False
    t.enable_deep_supervision = True
    t._do_i_compile = lambda: False
    t._get_deep_supervision_scales = lambda: [1, .5, .25]
    loss = t._build_loss()
    assert isinstance(loss, DeepSupervisionWrapper)
    assert isinstance(loss.loss, DC_and_topk_loss)
    assert loss.loss.ce.k == 10
    assert loss.loss.weight_ce == loss.loss.weight_dice == 1
    assert list(loss.weight_factors) == pytest.approx([2/3, 1/3, 0])
    x = torch.arange(40, dtype=torch.float32).reshape(1, 2, 4, 5) / 10
    x[:, 1] = torch.linspace(-2, 2, 20).reshape(4, 5)
    y = (torch.arange(20).reshape(1, 1, 4, 5) % 2).long()
    ce = F.cross_entropy(x, y[:, 0], reduction='none').flatten()
    expected_ce = ce.topk(2).values.mean()
    assert torch.allclose(loss.loss.ce(x, y), expected_ce)
    assert not torch.allclose(expected_ce, ce.mean())
    expected = expected_ce + loss.loss.dc(x, y)
    assert torch.allclose(loss.loss(x, y), expected)
    second = x.flip(-1)
    actual = loss([x, second, x * 100], [y, y, y])
    assert torch.isfinite(actual)
    assert torch.allclose(actual, (2 * expected + loss.loss(second, y)) / 3)
    assert torch.allclose(actual, loss([x, second, x * -100], [y, y, y]))


def test_real_logger_ema(tmp_path):
    t = state(tmp_path)
    t.logger.log('mean_fg_dice', .5, 0)
    t.logger.log('mean_fg_dice', .6, 1)
    t.current_epoch = 1
    t._update_early_stopping()
    assert t._early_stopping_best_ema == pytest.approx(.51)


def test_warmup_and_patience_boundary(tmp_path):
    t = state(tmp_path)
    for epoch in range(400):
        t.current_epoch = epoch
        t.logger.log('ema_fg_dice', .5, epoch)
        t._update_early_stopping()
        assert t._epochs_without_improvement == max(0, epoch + 1 - 300)
        assert t._early_stopping_triggered == (epoch == 399)


@pytest.mark.parametrize('delta,expected', [(0.000099, 18), (0.0001, 0), (.0002, 0)])
def test_improvement_reset(tmp_path, delta, expected):
    t = state(tmp_path)
    t.current_epoch = 300
    t._early_stopping_best_ema = .5
    t._epochs_without_improvement = 17
    t.logger.log('ema_fg_dice', .5 + delta, 0)
    t._update_early_stopping()
    assert t._epochs_without_improvement == expected
    assert t._early_stopping_best_ema == pytest.approx(.5 if expected else .5 + delta)


def test_official_best_selection_independent_of_min_delta(tmp_path):
    t = state(tmp_path)
    t.current_epoch = 300
    t._early_stopping_best_ema = t._best_ema = .5
    for key, value in [('ema_fg_dice', .50005), ('train_losses', 1.), ('val_losses', 1.),
                       ('dice_per_class_or_region', [.5]), ('epoch_start_timestamps', 0.)]:
        t.logger.log(key, value, 0)
    t.num_epochs = 1000
    t.save_every = 50
    t.output_folder = str(tmp_path)
    t.local_rank = 1  # omit plotting only
    saved = []
    t.save_checkpoint = lambda name: saved.append(Path(name).name)
    t.on_epoch_end()
    assert saved == ['checkpoint_best.pth']
    assert t._best_ema == pytest.approx(.50005)
    assert t._early_stopping_best_ema == .5
    assert t._epochs_without_improvement == 1
    assert t.current_epoch == 301


class TinyState:
    def __init__(self, value):
        self.value = value
    def state_dict(self):
        return self.value
    def load_state_dict(self, value):
        self.value = value


def checkpoint_fixture(tmp_path):
    t = state(tmp_path)
    t.device = torch.device('cpu')
    t.local_rank = 0
    t.disable_checkpointing = False
    t.is_ddp = False
    t.was_initialized = True
    t.network = TinyState({'weight': torch.tensor([1.])})
    t.optimizer = TinyState({'state': {0: {'momentum_buffer': torch.tensor([.25])}},
                             'param_groups': [{'lr': .01, 'params': [0]}]})
    t.grad_scaler = None
    t._best_ema = .7
    t.my_init_kwargs = {'configuration': '2d', 'fold': 0}
    t.inference_allowed_mirroring_axes = (0, 1)
    t.initialize = lambda: pytest.fail('production initialization forbidden')
    return t


@pytest.mark.parametrize('triggered', [False, True])
def test_tiny_checkpoint_roundtrip_and_resume_control(tmp_path, triggered):
    source = checkpoint_fixture(tmp_path)
    source.current_epoch = 399 if triggered else 350
    source._early_stopping_best_ema = .69
    source._epochs_without_improvement = 100 if triggered else 42
    source._early_stopping_triggered = triggered
    source.logger.log('mean_fg_dice', .7, 0)
    path = tmp_path / 'tiny.pth'
    source.save_checkpoint(str(path))
    assert path.stat().st_size < 16384
    payload = torch.load(path, map_location='cpu', weights_only=False)
    assert payload['trainer_name'] == Trainer.__name__
    assert payload['current_epoch'] == source.current_epoch + 1
    restored = checkpoint_fixture(tmp_path)
    restored.load_checkpoint(str(path))
    assert restored._early_stopping_checkpoint_state() == source._early_stopping_checkpoint_state()
    assert restored.current_epoch == source.current_epoch + 1
    assert restored._best_ema == .7
    assert restored.logger.get_value('ema_fg_dice', -1) == .7
    assert restored.optimizer.state_dict()['param_groups'] == payload['optimizer_state']['param_groups']
    assert torch.equal(restored.optimizer.state_dict()['state'][0]['momentum_buffer'], torch.tensor([.25]))
    assert torch.equal(restored.network.state_dict()['weight'], torch.tensor([1.]))
    calls = []
    restored.on_train_start = lambda: calls.append('start')
    restored.on_train_end = lambda: calls.append('end')
    restored.on_epoch_start = lambda: pytest.fail('unexpected additional epoch')
    restored.train_step = lambda *a: pytest.fail('unexpected training step')
    restored.num_epochs = 1000 if triggered else restored.current_epoch
    restored.run_training()
    assert calls == ['start', 'end']


@pytest.mark.parametrize('outside', [False, True])
def test_fresh_official_discovery(tmp_path, outside):
    env = os.environ.copy()
    env['nnUNet_extTrainer'] = str(ROOT)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, '-B', str(ROOT / 'tests' /
                            'verify_external_topk10_early_stopping_discovery.py')],
                            cwd=tmp_path if outside else ROOT.parent, env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'TOPK10_EARLY_STOPPING_RESOLVER_OK' in result.stdout
