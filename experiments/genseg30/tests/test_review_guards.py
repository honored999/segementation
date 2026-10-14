"""Review regressions: constructor side effects, standalone val, and resume binding."""
from pathlib import Path
import importlib
import json
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'nnunet_ext_trainers'))
from nnUNetTrainerGenSeg30EarlyStopping import nnUNetTrainerGenSeg30EarlyStopping as Trainer
from nnUNetTrainerEarlyStopping import nnUNetTrainerEarlyStopping as Base
from experiments.genseg30.contract import read_json, sha256, write_json
from test_trainer import trainer_for

NN_MODULE = importlib.import_module('nnunetv2.training.nnUNetTrainer.nnUNetTrainer')


@pytest.mark.parametrize('destination', ['repo', 'raw', 'source', 'pool', 'generator'])
def test_constructor_rejects_results_before_parent_side_effects(source_factory, tmp_path, monkeypatch, destination):
    source = source_factory('b2nd')
    raw = source.raw_root
    pool, run = tmp_path / 'pool_boundary', tmp_path / 'run_boundary'
    pool.mkdir()
    run.mkdir()
    write_json(pool / 'manifest.json', {'generator_checkpoint': str(run / 'generator.pt')})
    protected = {'repo': ROOT / 'forbidden_results', 'raw': raw / 'results',
                 'source': source.folder / 'results', 'pool': pool / 'results', 'generator': run / 'results'}
    results = protected[destination]
    monkeypatch.setattr(NN_MODULE, 'nnUNet_results', str(results))
    monkeypatch.setattr(NN_MODULE, 'nnUNet_preprocessed', str(source.base.parent))
    monkeypatch.setenv('nnUNet_raw', str(raw))
    monkeypatch.setenv('GENSEG30_POOL', str(pool))
    def forbidden(*args, **kwargs):
        raise AssertionError('Parent constructor can create folders/logs before validation')
    monkeypatch.setattr(Base, '__init__', forbidden)
    plans = dict(source.plans, plans_name='nnUNetPlans', continue_training=False)
    with pytest.raises(ValueError, match='Output'):
        Trainer(plans, '2d', 0, source.dataset_json, device=torch.device('cpu'))
    assert not results.exists()


@pytest.mark.parametrize('invalid', ['missing_split', 'fold', 'configuration', 'cascade'])
def test_standalone_validation_split_guard_never_regenerates(source_factory, monkeypatch, invalid):
    source = source_factory('b2nd')
    monkeypatch.setenv('nnUNet_raw', str(source.raw_root))
    trainer = trainer_for(source)
    del trainer.do_split  # exercise production method, not the older assembly-only fixture override
    if invalid == 'missing_split':
        source.splits.unlink()
    elif invalid == 'fold':
        trainer.fold = 1
    elif invalid == 'configuration':
        trainer.configuration_name = '3d_fullres'
    else:
        trainer.is_cascaded = True
    with pytest.raises(ValueError):
        trainer.do_split()
    if invalid == 'missing_split':
        assert not source.splits.exists()


def test_standalone_validation_checks_before_parent_validation(source_factory, monkeypatch):
    source = source_factory('b2nd')
    monkeypatch.setenv('nnUNet_raw', str(source.raw_root))
    trainer = trainer_for(source)
    del trainer.do_split
    source.splits.unlink()
    monkeypatch.setattr(Base, 'perform_actual_validation', lambda *a, **k: (_ for _ in ()).throw(
        AssertionError('Parent validation entered before strict guard')))
    with pytest.raises(ValueError):
        trainer.perform_actual_validation()
    assert not source.splits.exists()


def test_missing_receipt_checkpoint_load_fails_before_parent_load(source_factory, tmp_path, monkeypatch):
    source = source_factory('b2nd')
    monkeypatch.setenv('nnUNet_raw', str(source.raw_root))
    trainer = trainer_for(source)
    Path(trainer.output_folder).mkdir(parents=True)
    copied = Path(trainer.output_folder) / 'checkpoint_latest.pth'
    copied.write_bytes(b'copied checkpoint must not even be deserialized without receipt')
    monkeypatch.setattr(Base, 'load_checkpoint', lambda *a, **k: (_ for _ in ()).throw(
        AssertionError('Parent checkpoint load entered without required receipt')))
    with pytest.raises(ValueError, match='EXISTING'):
        trainer.load_checkpoint(str(copied))
    assert not (Path(trainer.output_folder) / 'genseg30_pool_identity.json').exists()

class TinyLogger:
    def __init__(self):
        self.state = {'ema': __import__('numpy').float64(.4), 'fixture': __import__('numpy').arange(2)}
    def get_checkpoint(self):
        return self.state
    def load_checkpoint(self, state):
        self.state = state


def receipt_for(source, output):
    return {'format': 1, 'trainer_name': Trainer.__name__, 'manifest_sha256': 'a' * 64,
            'pool': str(source.base.parent.parent / 'approved_pool'),
            'generator_checkpoint': str(source.base.parent.parent / 'generator_run' / 'generator.pt'),
            'generator_sha256': 'b' * 64, 'source': source.identity, 'synthetic_slices': 24}


def test_inherited_checkpoint_save_and_guarded_load_roundtrip(source_factory, monkeypatch):
    from nnUNetTrainerGenSeg30EarlyStopping import PROVENANCE_KEY
    source = source_factory('b2nd')
    trainer = trainer_for(source)
    output = Path(trainer.output_folder)
    output.mkdir(parents=True)
    receipt = receipt_for(source, output)
    write_json(output / 'genseg30_pool_identity.json', receipt)
    trainer.my_init_kwargs[PROVENANCE_KEY] = receipt
    trainer.local_rank = 0
    trainer.disable_checkpointing = False
    trainer.is_ddp = False
    trainer.device = torch.device('cpu')
    trainer.network = torch.nn.Linear(1, 1)
    trainer.optimizer = torch.optim.SGD(trainer.network.parameters(), lr=.01)
    trainer.grad_scaler = None
    trainer.logger = TinyLogger()
    trainer._best_ema = .3
    trainer.current_epoch = 7
    trainer.inference_allowed_mirroring_axes = (0, 1)
    trainer._early_stopping_best_ema = .4
    trainer._epochs_without_improvement = 2
    trainer._early_stopping_triggered = False
    trainer.was_initialized = True
    before = trainer.network.weight.detach().clone()
    path = output / 'checkpoint_latest.pth'
    trainer.save_checkpoint(str(path))  # actual inherited serializer; tiny weights, ordinary NumPy logger metadata
    trainer.network.weight.data.fill_(123)
    trainer.load_checkpoint(str(path))
    assert torch.equal(trainer.network.weight, before)
    assert trainer.current_epoch == 8 and trainer._epochs_without_improvement == 2
    assert trainer._genseg30_checkpoint_loaded is True
    # Deleting the receipt after load must not allow get_tr... to establish a new identity.
    (output / 'genseg30_pool_identity.json').unlink()
    with pytest.raises(ValueError, match='EXISTING'):
        trainer.load_checkpoint(str(path))


@pytest.mark.parametrize('mutation', ['trainer', 'pool', 'missing_provenance', 'fold_bool'])
def test_checkpoint_foreign_identity_rejected_before_parent_restore(source_factory, monkeypatch, mutation):
    from nnUNetTrainerGenSeg30EarlyStopping import PROVENANCE_KEY
    source = source_factory('b2nd')
    trainer = trainer_for(source)
    output = Path(trainer.output_folder)
    output.mkdir(parents=True)
    receipt = receipt_for(source, output)
    write_json(output / 'genseg30_pool_identity.json', receipt)
    args = dict(trainer.my_init_kwargs, **{PROVENANCE_KEY: dict(receipt)})
    checkpoint = {'trainer_name': Trainer.__name__, 'init_args': args}
    if mutation == 'trainer':
        checkpoint['trainer_name'] = 'nnUNetTrainerEarlyStopping'
    elif mutation == 'pool':
        args[PROVENANCE_KEY]['manifest_sha256'] = 'c' * 64
    elif mutation == 'missing_provenance':
        del args[PROVENANCE_KEY]
    else:
        args['fold'] = False
    monkeypatch.setattr(Base, 'load_checkpoint', lambda *a, **k: (_ for _ in ()).throw(AssertionError('restore')))
    with pytest.raises(ValueError):
        trainer.load_checkpoint(checkpoint)


def test_actual_constructor_safe_results_and_fixed_split(source_factory, tmp_path, monkeypatch):
    source = source_factory('b2nd')
    monkeypatch.delenv('GENSEG30_POOL', raising=False)
    monkeypatch.setenv('nnUNet_results', str(tmp_path / 'valid_results'))
    plans = dict(source.plans, continue_training=False)
    trainer = Trainer(plans, '2d', 0, source.dataset_json, torch.device('cpu'))
    assert Path(trainer.output_folder).is_dir()
    train, val = trainer.do_split()
    assert train == source.train and val == source.val
    assert trainer.dataset_class is source.dataset_class
    assert not (Path(trainer.output_folder) / 'genseg30_pool_identity.json').exists()


def test_continuation_constructor_requires_receipt_before_logs(source_factory, tmp_path, monkeypatch):
    source = source_factory('b2nd')
    monkeypatch.setattr(NN_MODULE, 'nnUNet_results', str(tmp_path / 'safe_results'))
    monkeypatch.delenv('GENSEG30_POOL', raising=False)
    def forbidden(*args, **kwargs):
        raise AssertionError('Constructor entered despite receiptless continuation')
    monkeypatch.setattr(Base, '__init__', forbidden)
    with pytest.raises(ValueError, match='EXISTING'):
        Trainer(dict(source.plans, continue_training=True), '2d', 0, source.dataset_json, torch.device('cpu'))
    assert not (tmp_path / 'safe_results').exists()


def test_validation_guard_preserves_parent_call_and_original_ids(source_factory, monkeypatch):
    source = source_factory('b2nd')
    trainer = trainer_for(source)
    del trainer.do_split
    seen = []
    def parent_validation(self, save_probabilities):
        seen.append((self.dataset_class, self.do_split(), save_probabilities))
        return 'original-validation'
    monkeypatch.setattr(Base, 'perform_actual_validation', parent_validation)
    assert trainer.perform_actual_validation(True) == 'original-validation'
    assert seen == [(source.dataset_class, (source.train, source.val), True)]

@pytest.mark.parametrize('state', ['loaded', 'copied'])
def test_receiptless_resume_refused_before_pool_audit(source_factory, monkeypatch, state):
    import nnUNetTrainerGenSeg30EarlyStopping as module
    source = source_factory('b2nd')
    trainer = trainer_for(source)
    output = Path(trainer.output_folder)
    output.mkdir(parents=True)
    (output / 'checkpoint_latest.pth').write_bytes(b'existing checkpoint')
    trainer._genseg30_checkpoint_loaded = state == 'loaded'
    monkeypatch.setenv('GENSEG30_POOL', str(source.base.parent.parent / 'pool_missing'))
    monkeypatch.setenv('GENSEG30_MANIFEST_SHA256', 'a' * 64)
    monkeypatch.setattr(module, 'SlicePoolDataset', lambda *a: (_ for _ in ()).throw(
        AssertionError('Pool audit reached before receiptless resume refusal')))
    with pytest.raises(ValueError, match='EXISTING|Existing checkpoint'):
        trainer.get_tr_and_val_datasets()
    assert not (output / 'genseg30_pool_identity.json').exists()

@pytest.mark.parametrize('flag', [False, True])
def test_actual_cli_constructor_transient_continue_flag_only(source_factory, tmp_path, monkeypatch, flag):
    import nnunetv2.run.run_training as run_module
    source = source_factory('b2nd')
    monkeypatch.setenv('nnUNet_results', str(tmp_path / 'cli_results'))
    monkeypatch.delenv('GENSEG30_POOL', raising=False)
    monkeypatch.delenv('GENSEG30_MANIFEST_SHA256', raising=False)
    # Substitute class discovery only; actual get_trainer_from_args reads plans and inserts the flag.
    monkeypatch.setattr(run_module, 'recursive_find_trainer_class_by_name', lambda name: Trainer)
    fresh = run_module.get_trainer_from_args('Dataset501_StrokeLesion', '2d', 0,
        Trainer.__name__, 'nnUNetPlans', False, torch.device('cpu'))
    assert fresh.do_split() == (source.train, source.val)
    if flag:
        output = Path(fresh.output_folder)
        write_json(output / 'genseg30_pool_identity.json', receipt_for(source, output))
        (output / 'checkpoint_latest.pth').write_bytes(b'metadata-only constructor fixture')
        trainer = run_module.get_trainer_from_args('Dataset501_StrokeLesion', '2d', 0,
            Trainer.__name__, 'nnUNetPlans', True, torch.device('cpu'))
    else:
        trainer = fresh
    # Actual installed parent pops only this transient CLI flag before PlansManager.
    assert 'continue_training' not in trainer.plans_manager.plans
    assert trainer.do_split() == (source.train, source.val)
    trainer.plans_manager.plans['configurations']['2d']['patch_size'] = [17, 16]
    with pytest.raises(ValueError, match='identity'):
        trainer.do_split()
