"""Tiny CPU checks using the installed official loader and real DA/worker path.
Run in the server's existing nnunet5090 environment too; no private data needed.
"""
import argparse
import inspect
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import numpy as np
import torch


class FixtureOriginal:
    def __init__(self, folder):
        self.identifiers = [f'fixture{i:03d}' for i in range(100)]
        self.source_folder = str(folder)
        self.folder_with_segs_from_previous_stage = None

    def load_case(self, identifier):
        data = np.ones((1, 2, 16, 16), dtype=np.float32)
        seg = np.zeros_like(data, dtype=np.int8)
        seg[:, :, 5:11, 5:11] = 1
        loc = np.argwhere(seg == 1)
        return data, seg, None, {'class_locations': {1: loc}}


def seed(value=123):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def raises(call):
    try:
        call()
    except (ValueError, KeyError):
        return
    raise AssertionError('expected explicit rejection')


def fixture(folder):
    from quality import sha, write_new
    original = FixtureOriginal(folder)
    cases, sidecar = {}, {}
    for i, cid in enumerate(original.identifiers):
        count = 5 if i < 78 else 4
        data = np.broadcast_to(np.arange(2, 2 + count, dtype=np.float32)[None, :, None, None], (1, count, 16, 16)).copy()
        seg = np.zeros_like(data, dtype=np.int8)
        seg[:, :, 5:11, 5:11] = 1
        path = folder / (cid + '.npz')
        np.savez(path, data=data, seg=seg)
        cases[cid] = dict(file=path.name, sha256=sha(path), count=count, source_shape=[1, 2, 16, 16])
        sidecar[cid] = [dict(weight=.2 if k % 2 == 0 else .8, lesion_pixels=36) for k in range(count)]
    write_new(folder / 'manifest.json', dict(cases=cases))
    pool_pin = sha(folder / 'manifest.json')
    return original, pool_pin, dict(cases=sidecar, pool_manifest_sha256=pool_pin)


def check_quality():
    from quality import contrast, quality_weight
    image = np.indices((32, 32)).sum(0).astype(np.float32) / 100 + 1
    label = np.zeros((32, 32), dtype=np.int8)
    label[12:20, 12:20] = 1
    image[label == 1] += 2
    value, size = contrast(image, label)
    assert value > 0 and size == 64
    assert quality_weight(2., [1., 2., 3.]) == 1
    assert quality_weight(20., [1., 2., 3.]) == .1
    assert quality_weight(None, [1., 2., 3.]) == 0
    assert contrast(image, np.zeros_like(label))[0] is None
    assert contrast(image, np.ones_like(label))[0] is None
    label[:] = 0
    label[:8, :8] = 1
    assert contrast(image, label)[0] is not None
    image[0, 0] = np.nan
    raises(lambda: contrast(image, label))


def check_loss():
    from loss import SourceLoss
    from nnunetv2.training.loss.compound_losses import DC_and_topk_loss
    from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
    base = DC_and_topk_loss(dict(batch_dice=True, smooth=1e-5, do_bg=False, ddp=False), dict(k=10))
    seed()
    x = torch.randn(2, 2, 8, 8, requires_grad=True)
    y = torch.randint(0, 2, (2, 1, 8, 8))
    loss = SourceLoss(base)
    # Validation delegates bit-for-bit, without metadata or RNG draws.
    assert torch.equal(loss(x, y), base(x, y))
    loss.training_context = True
    raises(lambda: loss(x, y))
    loss.metadata = ([0, 1], [1., 1.])
    full = loss(x, y)
    r = loss.level(x[:1], y[:1], [0], [1.])
    g = loss.level(x[1:], y[1:], [1], [1.])
    assert torch.allclose(full, r + g)
    loss.metadata = ([0, 1], [1., .2])
    low = loss(x, y)
    assert torch.allclose(low, r + .2 * g)
    low.backward()
    assert torch.isfinite(x.grad).all() and (x.grad.abs().flatten(1).sum(1) > 0).all()
    # Synthetic hardness cannot alter real budget or its gradient.
    a = x.detach().clone().requires_grad_()
    b = x.detach().clone()
    b[1] *= 100
    b.requires_grad_()
    da = torch.autograd.grad(loss(a, y), a)[0][0]
    db = torch.autograd.grad(loss(b, y), b)[0][0]
    assert torch.equal(da, db)
    loss.metadata = ([1, 1], [0., 0.])
    zero = loss(x, y)
    assert zero.item() == 0
    assert torch.autograd.grad(zero, x)[0].abs().sum() == 0
    for kind in (0, 1):
        assert torch.isfinite(loss.level(x, y, [kind, kind], [1., 1.]))
    ignore = SourceLoss(base, ignore_label=2)
    ignore.training_context = True
    ignore.metadata = ([0, 1], [1., 1.])
    empty = torch.full_like(y, 2)
    assert ignore(x, empty).item() == 0
    mixed = y.clone()
    mixed[0] = 2
    assert torch.allclose(ignore(x, mixed), ignore.level(x[1:], y[1:], [1], [1.]))
    partial = y.clone()
    partial[:, :, :4] = 2
    assert torch.isfinite(ignore(x, partial))
    dsbase = DeepSupervisionWrapper(base, [2/3, 1/3, 0])
    ds = SourceLoss(dsbase, [2/3, 1/3, 0])
    xs = [x, torch.randn(2, 2, 4, 4, requires_grad=True), torch.randn(2, 2, 2, 2, requires_grad=True)]
    ys = [y, y[:, :, ::2, ::2], y[:, :, ::4, ::4]]
    assert torch.equal(ds(xs, ys), dsbase(xs, ys))
    ds.training_context = True
    ds.metadata = ([0, 1], [1., .2])
    result = ds(xs, ys)
    result.backward()
    assert torch.isfinite(result) and xs[1].grad.abs().sum() > 0 and xs[2].grad is None
    raises(lambda: ds(xs[:1], ys))
    raises(lambda: loss.level(x, y, [0], [1.]))
    raises(lambda: loss.level(x, y, [0, 2], [1., 1.]))
    raises(lambda: loss.level(x, y, [0, 1], [.5, 1.]))
    raises(lambda: loss.level(x, y, [0, 1], [1., float('nan')]))


def make_loader(dataset, cls, transforms=None, force=0., patch=(16, 16), batch=12):
    from nnunetv2.utilities.label_handling.label_handling import LabelManager
    return cls(dataset, batch, patch, patch, LabelManager({'background': 0, 'lesion': 1}, None),
               oversample_foreground_percent=force, transforms=transforms)


def check_loader(folder, worker):
    from data import QualityDataset, SourceLoader
    from lesiongen.segmentation512 import AugmentedDataset
    from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    original, pool_pin, sidecar = fixture(folder)
    dataset = QualityDataset(original, folder, pool_pin, sidecar, 'q')
    # Full unaugmented plane data encodes source; no per-case inference can pass.
    for fg in (0., 1.):
        seed()
        batch = make_loader(dataset, SourceLoader, force=fg).generate_train_batch()
        assert torch.equal((batch['data'][:, 0, 0, 0] >= 2).long(), batch['q_source'])
        assert set(batch['q_source'].tolist()) == {0, 1}
        assert (batch['q_quality'][batch['q_source'] == 0] == 1).all()
        plane = batch['data'][:, 0, 0, 0]
        expected = torch.where(plane == 1, 1., torch.where((plane.long() - 2) % 2 == 0, .2, .8))
        assert torch.allclose(batch['q_quality'], expected)
    transform = nnUNetTrainer.get_training_transforms(
        (16, 16), (-.2, .2), [[1., 1.], [.5, .5]], (0, 1), False,
        use_mask_for_norm=[False], is_cascaded=False, foreground_labels=[1], regions=None, ignore_label=None)
    augmented = AugmentedDataset(original, folder, pool_pin)
    seed(321)
    tagged = make_loader(dataset, SourceLoader, transforms=transform).generate_train_batch()
    rng_tagged = (np.random.get_state(), torch.get_rng_state())
    seed(321)
    ordinary = make_loader(augmented, nnUNetDataLoader, transforms=transform).generate_train_batch()
    rng_ordinary = (np.random.get_state(), torch.get_rng_state())
    assert torch.equal(tagged['data'], ordinary['data'])
    assert all(torch.equal(a, b) for a, b in zip(tagged['target'], ordinary['target']))
    assert torch.equal(rng_tagged[1], rng_ordinary[1])
    assert np.array_equal(rng_tagged[0][1], rng_ordinary[0][1]) and rng_tagged[0][2:] == rng_ordinary[0][2:]
    assert tagged['data'].shape[1] == 1 and tagged['target'][0].shape[1] == 1
    # Padding keeps upstream -1 -> background convention after the actual DA path.
    padded = make_loader(dataset, SourceLoader, transforms=transform, patch=(24, 24)).generate_train_batch()
    assert (padded['target'][0] >= 0).all()
    filtered = QualityDataset(original, folder, pool_pin, sidecar, 'quality_filter', .5)
    data, _, _, _ = filtered.load_case(original.identifiers[0])
    assert data.shape[1] == 4  # two real plus two retained .8 generated planes
    assert (filtered.current['quality'] == 1).all()
    filtered_all = QualityDataset(original, folder, pool_pin, sidecar, 'quality_filter', 1.)
    assert filtered_all.load_case(original.identifiers[0])[0].shape[1] == 2
    assert (filtered_all.current['source'] == 0).all()
    if worker:
        from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
        generator = NonDetMultiThreadedAugmenter(make_loader(dataset, SourceLoader, transforms=transform, batch=2),
                                                None, num_processes=1, num_cached=1, seeds=[123], pin_memory=False)
        try:
            batch = next(generator)
            assert batch['q_source'].shape == batch['q_quality'].shape == (2,)
            assert len(batch['target']) == 2
            # Loss consumes metadata delivered through a spawned worker and deep supervision.
            from loss import SourceLoss
            from torch import nn
            loss = SourceLoss(nn.Identity(), [.75, .25])
            loss.training_context = True
            loss.metadata = (batch['q_source'], batch['q_quality'])
            outputs = [torch.randn(2, 2, *t.shape[2:], requires_grad=True) for t in batch['target']]
            value = loss(outputs, batch['target'])
            value.backward()
            assert torch.isfinite(value) and outputs[0].grad.abs().sum() > 0
        finally:
            generator._finish()


def check_adapter():
    from trainer import trainer_class, Baseline, nnUNetTrainerDataset509Quality
    from nnUNetTrainerLesionGen512TopK10EarlyStopping import nnUNetTrainerLesionGen512TopK10EarlyStopping as Original
    assert trainer_class('baseline') is Baseline
    assert Baseline.get_dataloaders is Original.get_dataloaders
    assert Baseline.train_step is Original.train_step
    assert Baseline._build_loss is Original._build_loss
    assert Baseline.validation_step is Original.validation_step
    assert inspect.signature(Baseline.__init__) == inspect.signature(nnUNetTrainerDataset509Quality.__init__)
    raises(lambda: trainer_class('q+c'))


def check_constructor(folder):
    from quality import sha, write_new, digest, SCORING
    from trainer import nnUNetTrainerDataset509Quality
    original, pool_pin, sidecar = fixture(folder)
    base = folder / 'prepared' / 'Dataset514_CI1DWILesionGen30'
    base.mkdir(parents=True)
    split = base / 'splits_final.json'
    write_new(split, [{'train': original.identifiers, 'val': [f'validation{i}' for i in range(28)]}])
    sidecar.update(scoring=SCORING, scoring_sha256=digest(SCORING), split_sha256=sha(split))
    side_path = folder / 'quality.json'
    write_new(side_path, sidecar)
    config = folder / 'config.json'
    write_new(config, dict(mode='q', sidecar=dict(path=str(side_path), sha256=sha(side_path)),
                           synthetic_coefficient=1., filter_threshold=.5))
    changes = dict(nnUNet_preprocessed=str(base.parent), nnUNet_results=str(folder / 'results'),
                   LESIONGEN512_POOL=str(folder), LESIONGEN512_POOL_SHA256=pool_pin,
                   DATASET509_QUALITY_CONFIG=str(config), nnUNet_compile='false')
    old = {k: os.environ.get(k) for k in changes}
    os.environ.update(changes)
    try:
        plans = dict(dataset_name='Dataset514_CI1DWILesionGen30', plans_name='fixture',
                     configurations={'2d': {'data_identifier': 'fixture_2d', 'batch_dice': True, 'architecture': {}}})
        dataset = dict(channel_names={'0': 'DWI'}, labels={'background': 0, 'lesion': 1})
        trainer = nnUNetTrainerDataset509Quality(plans, '2d', 0, dataset, torch.device('cpu'))
        assert trainer.my_init_kwargs['quality_identity']['sidecar_sha256'] == sha(side_path)
        trainer.enable_deep_supervision = False
        from loss import SourceLoss
        assert isinstance(trainer._build_loss(), SourceLoss)
        trainer.quality_config['mode'] = 'quality_filter'
        assert not isinstance(trainer._build_loss(), SourceLoss)
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lesiongen-repo', required=True)
    p.add_argument('--skip-worker', action='store_true')
    a = p.parse_args()
    repo = Path(a.lesiongen_repo)
    sys.path[:0] = [str(repo / 'src'), str(repo / 'trainers')]
    torch.set_num_threads(1)
    from api_contract import verify
    verify()
    check_quality()
    print('PASS native quality signal edge cases', flush=True)
    check_loss()
    print('PASS source-separated TopK/weight/gradient/deep-supervision/ignore boundaries', flush=True)
    check_adapter()
    print('PASS baseline original class/method path and constructor compatibility', flush=True)
    with tempfile.TemporaryDirectory(prefix='dataset509_q_cpu_') as tmp:
        check_loader(Path(tmp), not a.skip_worker)
    with tempfile.TemporaryDirectory(prefix='dataset509_q_init_') as tmp:
        check_constructor(Path(tmp))
    print('PASS real trainer CPU construction, Q/filter loss selection; no network initialization', flush=True)
    print('PASS actual loader/foreground/DA/RNG/filter/padding' + ('/spawned-worker' if not a.skip_worker else ''), flush=True)
    print('ENGINEERING_CHECKS_PASS; real pool distribution/preflight/training NOT RUN', flush=True)


if __name__ == '__main__':
    main()
