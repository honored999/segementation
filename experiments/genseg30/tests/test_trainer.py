import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'nnunet_ext_trainers'))
from nnUNetTrainerGenSeg30EarlyStopping import nnUNetTrainerGenSeg30EarlyStopping
from nnUNetTrainerEarlyStopping import nnUNetTrainerEarlyStopping
from experiments.genseg30.contract import sha256
from experiments.genseg30.pipeline import train_generator, export_pool, approve_pool, ACKNOWLEDGEMENT


def trainer_for(source):
    trainer = object.__new__(nnUNetTrainerGenSeg30EarlyStopping)
    trainer.fold = 0
    trainer.configuration_name = '2d'
    trainer.configuration_manager = SimpleNamespace(patch_size=[16, 16])
    trainer.plans_manager = SimpleNamespace(plans=source.plans)
    trainer.dataset_json = source.dataset_json
    trainer.preprocessed_dataset_folder = str(source.folder)
    trainer.preprocessed_dataset_folder_base = str(source.base)
    trainer.print_to_log_file = lambda *a, **k: None
    trainer.my_init_kwargs = {'plans': source.plans, 'configuration': '2d', 'fold': 0,
                              'dataset_json': source.dataset_json, 'device': torch.device('cpu')}
    trainer._genseg30_checkpoint_loaded = False
    trainer._genseg30_continue_requested = False
    trainer.output_folder_base = str(source.base.parent.parent / 'training_results' / 'genseg30')
    trainer.output_folder = str(Path(trainer.output_folder_base) / 'fold_0')
    trainer.folder_with_segs_from_previous_stage = None
    trainer.is_cascaded = False
    trainer.dataset_class = source.dataset_class
    trainer.do_split = lambda: (source.train.copy(), source.val.copy())
    return trainer


def test_minimal_trainer_original_validation_and_actual_loader(source_factory, tmp_path, monkeypatch):
    source = source_factory('b2nd')
    monkeypatch.setenv('nnUNet_raw', str(source.raw_root))
    run, pool = tmp_path / 'run', tmp_path / 'pool'
    train_generator(source, run, epochs=1, max_steps=1, width=1, downs=5, discriminator_width=1,
                    seed=3, device='cpu')
    export_pool(source, run, pool, seed=4, device='cpu')
    approve_pool(source, pool, 'fixture reviewer', ACKNOWLEDGEMENT)
    monkeypatch.setenv('GENSEG30_POOL', str(pool))
    monkeypatch.setenv('GENSEG30_MANIFEST_SHA256', sha256(pool / 'manifest.json'))
    trainer = trainer_for(source)
    before = {p.name: sha256(p) for p in source.folder.iterdir()}
    train, val = trainer.get_tr_and_val_datasets()
    assert trainer.dataset_class is source.dataset_class
    assert type(val) is source.dataset_class and val.identifiers == source.val
    assert train.identifiers == source.train
    trainer.dataset_class.unpack_dataset(str(source.folder), overwrite_existing=False, verify=True)
    assert before == {p.name: sha256(p) for p in source.folder.iterdir()}
    original = source.training_dataset()
    source_data = np.asarray(original.load_case(source.train[0])[0][:])
    from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
    from nnunetv2.utilities.label_handling.label_handling import LabelManager
    manager = LabelManager(source.dataset_json['labels'], None)
    loader = nnUNetDataLoader(train, 1, [9, 11], [9, 11], manager, oversample_foreground_percent=1.)
    assert loader.indices == source.train and loader.sampling_probabilities is None
    # Target only the first real source case, while exercising the official crop/FG loader unchanged.
    loader.get_indices = lambda: [source.train[0]]
    np.random.seed(7)
    saw_synthetic = False
    for _ in range(25):
        batch = loader.generate_train_batch()
        data, target = batch['data'].numpy(), batch['target'].numpy()
        assert data.shape == target.shape == (1, 1, 9, 11)
        assert 1 in target and np.isin(target, [-1, 0, 1]).all()
        saw_synthetic |= not any(np.array_equal(data[0, 0], plane) for plane in source_data[0])
    assert saw_synthetic
    # No changes to model/loss/schedule/checkpoint/augmentation or validation implementation.
    for name in ('initialize', '_build_loss', 'configure_optimizers', 'get_dataloaders',
                 'on_train_start', 'run_training', 'save_checkpoint'):
        assert getattr(nnUNetTrainerGenSeg30EarlyStopping, name) is getattr(nnUNetTrainerEarlyStopping, name)


@pytest.mark.parametrize('change', ['npz', 'fold', '3d', 'cascade', 'missing_pool', 'pin', 'plans'])
def test_trainer_rejects_unsafe_before_official_split(source_factory, monkeypatch, change):
    source = source_factory('npz' if change == 'npz' else 'b2nd')
    monkeypatch.setenv('nnUNet_raw', str(source.raw_root))
    monkeypatch.setenv('GENSEG30_POOL', '/unavailable/pool')
    monkeypatch.setenv('GENSEG30_MANIFEST_SHA256', 'not-a-hash')
    trainer = trainer_for(source)
    if change == 'fold':
        trainer.fold = 1
    elif change == '3d':
        trainer.configuration_name = '3d_fullres'
    elif change == 'cascade':
        trainer.is_cascaded = True
    elif change == 'missing_pool':
        monkeypatch.delenv('GENSEG30_POOL')
    elif change == 'pin':
        monkeypatch.delenv('GENSEG30_MANIFEST_SHA256')
    elif change == 'plans':
        trainer.plans_manager.plans = {}
    def forbidden():
        raise AssertionError('Official split reached before checks')
    trainer.do_split = forbidden
    with pytest.raises(ValueError):
        trainer.get_tr_and_val_datasets()

