import json
from pathlib import Path

import numpy as np
import pytest

from experiments.genseg30.contract import sha256
from experiments.genseg30.pipeline import train_generator, export_pool, audit_pool, approve_pool, ACKNOWLEDGEMENT
from experiments.genseg30.dataset import SlicePoolDataset


def make_pool(source, root):
    run = root / 'run'
    train_generator(source, run, epochs=2, max_steps=2, width=1, downs=5, discriminator_width=1,
                    seed=11, device='cpu')
    pool = root / 'pool'
    export_pool(source, run, pool, seed=12, device='cpu')
    return run, pool


def test_training_export_audit_quality_and_wrapper(source_factory, tmp_path, monkeypatch):
    source = source_factory()
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetNumpy
    original_load = nnUNetDatasetNumpy.load_case
    reads = []
    def tracked(self, identifier):
        reads.append(identifier)
        assert identifier in source.train  # any validation read fails
        return original_load(self, identifier)
    monkeypatch.setattr(nnUNetDatasetNumpy, 'load_case', tracked)
    run, pool = make_pool(source, tmp_path)
    manifest = audit_pool(source, pool)
    assert reads and not set(reads) & set(source.val)
    assert manifest['synthetic_slices'] == 24  # ceil(.30 * 78)
    assert sum(case['synthetic_slices'] for case in manifest['cases'].values()) == 24
    assert manifest['generator']['actual_steps'] == 2
    assert len(manifest['previews']) == 24
    assert (run / 'generator.pt').is_file()
    with pytest.raises(ValueError, match='quality'):
        audit_pool(source, pool, require_quality=True)
    with pytest.raises(ValueError):
        approve_pool(source, pool, 'reviewer', 'auto-approve')
    approve_pool(source, pool, 'reviewer', ACKNOWLEDGEMENT)
    wrapped = SlicePoolDataset(source.training_dataset(), source, pool, sha256(pool / 'manifest.json'))
    assert wrapped.identifiers == source.train
    data, seg, previous, props = wrapped.load_case(source.train[0])
    assert data.shape == (1, 4, 9, 11) and seg.shape == data.shape and previous is None
    assert props['genseg30_sampling_only'] is True
    coords = props['class_locations'][1]
    assert np.all(seg[tuple(coords.T)] == 1)
    assert 3 in coords[:, 1]
    # A source with one z and one generated z must retain 4D, with no singleton squeeze.
    single = next(p for p, case in manifest['cases'].items()
                  if case['original_slices'] == case['synthetic_slices'] == 1)
    assert wrapped.load_case(single)[0].shape == (1, 2, 9, 11)
    with pytest.raises(ValueError):
        wrapped.load_case(source.val[0])
    with pytest.raises(ValueError):
        export_pool(source, run, pool, seed=12, device='cpu')
    with pytest.raises(ValueError):
        train_generator(source, run, epochs=1, max_steps=1, width=1, downs=5,
                        discriminator_width=1, seed=11, device='cpu')


@pytest.mark.parametrize('mutation', ['foreign', 'incomplete', 'hash', 'escape', 'counts', 'source'])
def test_fail_closed_before_source_load(source_factory, tmp_path, monkeypatch, mutation):
    source = source_factory()
    _, pool = make_pool(source, tmp_path)
    manifest_path = pool / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if mutation == 'foreign':
        manifest['source']['train_ids'][0] = source.val[0]
    elif mutation == 'incomplete':
        manifest['complete'] = False
    elif mutation == 'hash':
        name = next(iter(manifest['cases'].values()))['file']
        (pool / name).write_bytes(b'tampered')
    elif mutation == 'escape':
        next(iter(manifest['cases'].values()))['file'] = '../patient.npz'
    elif mutation == 'counts':
        manifest['synthetic_slices'] += 1
    else:
        (source.folder / (source.train[0] + '.npz')).write_bytes(b'changed')
    manifest_path.write_text(json.dumps(manifest))
    def forbidden(*args, **kwargs):
        raise AssertionError('Unsafe read reached source dataset')
    monkeypatch.setattr(source.dataset_class, 'load_case', forbidden)
    with pytest.raises(ValueError):
        audit_pool(source, pool)


def test_source_rejects_changed_reference_missing_split_and_foreign_ids(source_factory, monkeypatch):
    source = source_factory()
    from experiments.genseg30.contract import Source
    source.splits.unlink()  # disposable synthetic fixture only
    with pytest.raises(ValueError):
        Source(source.folder, source.raw_root)


def test_npz_source_unpack_rejected_without_writes(source_factory):
    source = source_factory()
    before = sorted(p.name for p in source.folder.iterdir())
    with pytest.raises(ValueError, match='unpack'):
        source.require_no_unpack_writes()
    assert before == sorted(p.name for p in source.folder.iterdir())

def test_manifest_changed_after_wrapper_creation_refused_before_source_load(source_factory, tmp_path, monkeypatch):
    source = source_factory()
    _, pool = make_pool(source, tmp_path)
    approve_pool(source, pool, 'reviewer', ACKNOWLEDGEMENT)
    wrapped = SlicePoolDataset(source.training_dataset(), source, pool, sha256(pool / 'manifest.json'))
    (pool / 'manifest.json').write_text('{}')
    monkeypatch.setattr(source.dataset_class, 'load_case', lambda *a: (_ for _ in ()).throw(
        AssertionError('Source load happened after manifest tampering')))
    with pytest.raises(ValueError, match='manifest'):
        wrapped.load_case(source.train[0])


@pytest.mark.parametrize('mutation', ['foreign', 'mismatch', 'channels', 'cascade', 'reference'])
def test_source_metadata_rejection_before_dataset_reads(source_factory, monkeypatch, mutation):
    source = source_factory()
    from experiments.genseg30 import contract
    monkeypatch.setattr(source.dataset_class, 'load_case', lambda *a: (_ for _ in ()).throw(AssertionError('read')))
    if mutation == 'foreign':
        (source.folder / 'foreign.npz').write_bytes(b'foreign metadata')
    elif mutation == 'mismatch':
        splits = json.loads(source.splits.read_text())
        splits[0]['train'][0] = 'foreign'
        source.splits.write_text(json.dumps(splits))
    elif mutation == 'channels':
        doc = json.loads(source.dataset_path.read_text())
        doc['channel_names']['1'] = 'ADC'
        source.dataset_path.write_text(json.dumps(doc))
    elif mutation == 'cascade':
        doc = json.loads(source.plans_path.read_text())
        doc['configurations']['2d']['previous_stage'] = '3d_lowres'
        source.plans_path.write_text(json.dumps(doc))
    else:
        contract.REFERENCE.write_text('[]')
    with pytest.raises(ValueError):
        contract.Source(source.folder, source.raw_root)

@pytest.mark.parametrize('artifact', ['quality_approval.json', 'manifest.json'])
def test_quality_or_manifest_changed_after_wrapper_creation_is_rejected(source_factory, tmp_path, monkeypatch, artifact):
    source = source_factory()
    _, pool = make_pool(source, tmp_path)
    approve_pool(source, pool, 'reviewer', ACKNOWLEDGEMENT)
    wrapped = SlicePoolDataset(source.training_dataset(), source, pool, sha256(pool / 'manifest.json'))
    (pool / artifact).write_text('{}')
    monkeypatch.setattr(source.dataset_class, 'load_case', lambda *a: (_ for _ in ()).throw(AssertionError('read')))
    with pytest.raises(ValueError):
        wrapped.load_case(source.train[0])

@pytest.mark.parametrize('artifact', ['source', 'pool'])
def test_data_mutation_after_startup_refused_before_source_read(source_factory, tmp_path, monkeypatch, artifact):
    source = source_factory()
    _, pool = make_pool(source, tmp_path)
    approve_pool(source, pool, 'reviewer', ACKNOWLEDGEMENT)
    wrapped = SlicePoolDataset(source.training_dataset(), source, pool, sha256(pool / 'manifest.json'))
    path = (source.folder / (source.train[0] + '.npz')) if artifact == 'source' else (
        pool / wrapped.manifest['cases'][source.train[0]]['file'])
    path.write_bytes(b'mutation')
    monkeypatch.setattr(source.dataset_class, 'load_case', lambda *a: (_ for _ in ()).throw(AssertionError('read')))
    with pytest.raises(ValueError):
        wrapped.load_case(source.train[0])


def test_unchanged_wrapper_avoids_content_rehash_and_detects_during_read(source_factory, tmp_path, monkeypatch):
    from experiments.genseg30 import contract
    source = source_factory()
    _, pool = make_pool(source, tmp_path)
    approve_pool(source, pool, 'reviewer', ACKNOWLEDGEMENT)
    wrapped = SlicePoolDataset(source.training_dataset(), source, pool, sha256(pool / 'manifest.json'))
    monkeypatch.setattr(contract, 'sha256', lambda *a: (_ for _ in ()).throw(AssertionError('Full file rehash')))
    wrapped.load_case(source.train[0])
    original_load = source.dataset_class.load_case
    def changed_during_read(dataset, identifier):
        result = original_load(dataset, identifier)
        path = pool / wrapped.manifest['cases'][identifier]['file']
        path.write_bytes(b'during-read mutation')
        return result
    monkeypatch.setattr(source.dataset_class, 'load_case', changed_during_read)
    with pytest.raises(ValueError):
        wrapped.load_case(source.train[0])
