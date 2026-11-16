"""C-only CPU checks. Does not rerun unchanged Q tests or initialize/train a model."""
import argparse
import json
import os
import pickle
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import numpy as np
import torch
from quality import sha, read, write_new, digest, contrast, SCORING
from curriculum import (POLICY, GROUPS, physical_area, reference_statistics, phase, group,
                        probabilities, case_current, policy_audit, build, validate_policy)


def reject(fn):
    try:
        fn()
    except (ValueError, KeyError):
        return
    raise AssertionError('Expected explicit rejection')


def geometry(depth=16):
    return dict(size_xyz=[16, 16, depth], spacing_xyz_mm=[2., 3., 4.],
                affine_xyz_to_lps_mm=np.diag([2., 3., 4., 1.]).tolist())


def make_fixture(root):
    package, base = root / 'package', root / 'prepared' / 'Dataset514_CI1DWILesionGen30'
    cache, pool = base / 'fixture_2d', base / 'synthetic_pool'
    for path in (package, cache, pool):
        path.mkdir(parents=True)
    ids = [f'fixture{i:03d}' for i in range(100)]
    val = [f'validation{i:03d}' for i in range(28)]
    split = base / 'splits_final.json'
    write_new(split, [dict(train=ids, val=val)])
    plans = dict(dataset_name=base.name, plans_name='fixture', transpose_forward=[0, 1, 2],
                 configurations={'2d': dict(data_identifier='fixture_2d', batch_dice=True, architecture={},
                                             patch_size=[16, 16], use_mask_for_norm=[False])})
    plan_path = base / 'plans.json'
    write_new(plan_path, plans)
    rows, samples, records, pool_cases, q_cases, real_hashes = [], [], [], {}, {}, []
    index = 0
    for ci, cid in enumerate(ids):
        depth, ns = (16 if ci < 93 else 15), (5 if ci < 78 else 4)
        records.append(dict(case_id=cid, split='train', geometry=geometry(depth), hashes=dict(dwi='fixture', mask='fixture')))
        native_masks, generated_masks = [], []
        q_cases[cid] = []
        for kind, count in (('REAL', depth), ('SYNTHETIC', ns)):
            for z in range(count):
                image = 1 + (np.indices((16, 16)).sum(0).astype(np.float32) / 50)
                mask = np.zeros((16, 16), dtype=np.int8)
                if kind == 'SYNTHETIC' or z % 4:
                    width = 2 if z % 2 else 4
                    mask[5:5+width, 5:5+width] = 1
                    image[mask == 1] += [0.1, 0.5, 2.][(z + ci) % 3]
                name = f'sample{index:05d}'
                index += 1
                file = package / (name + '.npz')
                np.savez(file, image=image, label=mask)
                pin = sha(file)
                rows.append(dict(sample=name, path=file.name, sha256=pin, kind=kind))
                samples.append(dict(sample=name, case_id=cid, split='train', center_z=z))
                if kind == 'REAL':
                    native_masks.append(mask.T)
                    real_hashes.append(pin)
                else:
                    generated_masks.append(mask.T)
                    score, size = contrast(image, mask)
                    q_cases[cid].append(dict(contrast=score, lesion_pixels=size, weight=.8,
                                              native_pair_sha256=pin))
        seg = np.stack(native_masks[1:-1])[None]
        data = np.broadcast_to(np.arange(11, depth+9, dtype=np.float32)[None, :, None, None], seg.shape).copy()
        np.save(cache / (cid + '.npy'), data)
        np.save(cache / (cid + '_seg.npy'), seg)
        props = dict(bbox_used_for_cropping=[[1, depth-1], [0, 16], [0, 16]],
                     class_locations={1: np.argwhere(seg == 1)})
        with (cache / (cid + '.pkl')).open('wb') as f:
            pickle.dump(props, f)
        synth_seg = np.stack(generated_masks)[None]
        synth_data = np.broadcast_to(np.arange(100, 100+ns, dtype=np.float32)[None, :, None, None], synth_seg.shape).copy()
        path = pool / (cid + '.npz')
        np.savez(path, data=synth_data, seg=synth_seg)
        pool_cases[cid] = dict(file=path.name, count=ns, source_shape=list(data.shape), sha256=sha(path))
    write_new(package / 'train_manifest.json', dict(samples=rows))
    write_new(package / 'provenance.private.json', dict(samples=samples, cases=records))
    write_new(pool / 'manifest.json', dict(cases=pool_cases, plans_sha256=sha(plan_path), split_sha256=sha(split)))
    q = dict(pool_manifest_sha256=sha(pool / 'manifest.json'), split_sha256=sha(split),
             package_manifest_sha256=sha(package / 'train_manifest.json'),
             provenance_sha256=sha(package / 'provenance.private.json'),
             training_reference_sha256=digest(dict(split=sha(split), pairs=real_hashes)),
             scoring=SCORING, scoring_sha256=digest(SCORING), cases=q_cases)
    q_path = root / 'quality.json'
    write_new(q_path, q)
    return package, base, plan_path, q_path


def test_math():
    assert physical_area(10, geometry()) == 60.  # pixels != square mm
    reject(lambda: physical_area(10, dict(spacing_xyz_mm=[1, 1, 1])))
    bad = geometry(); bad['spacing_xyz_mm'] = [1, 1, 1]
    reject(lambda: physical_area(10, bad))
    assert phase(0, POLICY) == phase(19, POLICY) == phase(20, POLICY) == 0
    assert phase(60, POLICY) == .5 and phase(100, POLICY) == phase(1000, POLICY) == 1
    reject(lambda: phase(-1, POLICY))
    reject(lambda: validate_policy(dict(POLICY, transition_epochs=0)))
    reject(lambda: validate_policy(dict(POLICY, synthetic_end=.75)))
    entries = [dict(area_mm2=6., contrast=.1)] * 10 + [dict(area_mm2=60., contrast=2.)] * 100
    thresholds, counts, gains = reference_statistics(entries, POLICY)
    assert thresholds['area_mm2'] == 60. and all(1 <= g <= 2 for g in gains)
    assert group(0, None, thresholds) == 5 and group(3, None, thresholds) == 4
    current = dict(source=np.array([0, 0, 1, 1, 1, 1]), group=np.array([5, 1, 0, 1, 2, 3]),
                   quality=np.array([1., 1., .8, .8, .8, .8]), foreground_pixels=np.array([0, 4, 4, 4, 4, 4]))
    gains = [2., 1., 2., 1.]
    gains += [1., 1.]
    p0, _ = probabilities(current, 0, POLICY, gains)
    p1, state = probabilities(current, 100, POLICY, gains)
    assert np.isclose(p0.sum(), 1) and np.isclose(p1.sum(), 1)
    assert p0[2] == p0[4] == 0 and p0[0] > 0
    assert np.isclose(p0[2:].sum(), .1) and np.isclose(p1[2:].sum(), .25)
    assert p1[2] == 2 * p1[3]  # group gain survives normalization
    np.random.seed(12345)
    observed = np.bincount(np.random.choice(len(p1), size=40000, p=p1), minlength=len(p1))
    assert 1.8 < observed[2] / observed[3] < 2.2
    assert abs(observed[2:].sum() / 40000 - .25) < .01
    forced, _ = probabilities(current, 100, POLICY, gains, True)
    assert forced[0] == 0
    current['quality'][2:] = 0
    fallback, state = probabilities(current, 100, POLICY, gains)
    assert state['fallback'] == 'no_trusted_candidate' and fallback[2:].sum() == 0
    current['quality'][2:] = .8
    current['group'][2:] = 0
    waiting, state = probabilities(current, 0, POLICY, gains)
    assert state['fallback'] == 'waiting_for_difficult_phase' and waiting[2:].sum() == 0
    current['foreground_pixels'][:] = 0
    empty, state = probabilities(current, 100, POLICY, gains, True)
    assert state['real_foreground_unavailable'] and empty[2:].sum() == 0


def test_real_path(root):
    from curriculum import case_current
    from data_c import CurriculumDataset, CurriculumLoader
    from trainer import trainer_class, Baseline, nnUNetTrainerDataset509Quality
    from trainer_c import nnUNetTrainerDataset509Curriculum as C
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetNumpy
    from nnunetv2.utilities.label_handling.label_handling import LabelManager
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
    from run_layout import job_layout
    package, base, plans, q_path = make_fixture(root)
    q_before, pool_before = sha(q_path), sha(base / 'synthetic_pool' / 'manifest.json')
    output = root / 'curriculum.json'
    audit = build(package, base, plans, q_path, output)
    side = read(output)
    assert sum(side['real_reference_group_counts']) == 1593
    assert side['thresholds']['area_mm2'] in (24., 96.)
    assert set(side['cases']) == set(read(base / 'splits_final.json')[0]['train'])
    assert side['audit']['native_real_area_mm2']['quantiles'][-1] == 96.
    assert sha(q_path) == q_before and sha(base / 'synthetic_pool' / 'manifest.json') == pool_before
    reject(lambda: build(package, base, plans, q_path, output))
    original_split = (base / 'splits_final.json').read_bytes()
    altered = read(base / 'splits_final.json'); altered[0]['train'][0] = 'validation000'
    (base / 'splits_final.json').write_text(json.dumps(altered), encoding='utf-8')
    reject(lambda: build(package, base, plans, q_path, root / 'leak.json'))
    (base / 'splits_final.json').write_bytes(original_split)
    pool = base / 'synthetic_pool'
    original = nnUNetDatasetNumpy(str(base / 'fixture_2d'), sorted(side['cases']))
    dataset = CurriculumDataset(original, pool, sha(pool / 'manifest.json'), side)
    lm = LabelManager({'background': 0, 'lesion': 1}, None)
    loader = CurriculumLoader(dataset, 12, (8, 8), (8, 8), lm, oversample_foreground_percent=.33,
                              policy=POLICY, gains=side['gains'])
    np.random.seed(12345); torch.manual_seed(12345)
    loader.set_epoch(100)
    batch = next(loader)
    for i, cid in enumerate(batch['keys']):
        entry = side['cases'][cid]
        z = batch['c_z'][i].item()
        source = int(z >= len(entry['real']))
        plane = (entry['real'] + entry['synthetic'])[z]
        assert batch['c_source'][i] == source and batch['c_group'][i] == plane['group']
        marker = z + 11 if source == 0 else 100 + z - len(entry['real'])
        assert (batch['data'][i] == marker).all()
        if loader.get_do_oversample(i) and plane['foreground_pixels']:
            assert (batch['target'][i] == 1).any()  # FG matches chosen Z, never another plane
    early, late = [], []
    for epoch, counts_out in ((0, early), (100, late)):
        loader.set_epoch(epoch)
        for _ in range(25):
            selected = next(loader)
            counts_out.extend(selected['c_source'].tolist())
    assert np.mean(late) > np.mean(early) + .05
    transform = nnUNetTrainer.get_training_transforms((8, 8), (-.1, .1), [[1., 1.], [.5, .5]],
                         (0, 1), False, use_mask_for_norm=[False], is_cascaded=False, foreground_labels=[1])
    augmented_loader = CurriculumLoader(dataset, 12, (12, 12), (8, 8), lm, transforms=transform,
                                       oversample_foreground_percent=.33, policy=POLICY, gains=side['gains'])
    wrapper = SingleThreadedAugmenter(augmented_loader, None)
    b0 = next(wrapper); augmented_loader.set_epoch(100); b1 = next(wrapper)
    assert b0['c_epoch'] == 0 and b1['c_epoch'] == 100 and len(b1['target']) == 2
    assert b1['data'].shape == (12, 1, 8, 8) and b1['target'][0].shape[1] == 1
    os.environ['nnUNet_n_proc_DA'] = '1'
    reject(lambda: CurriculumLoader(dataset, 2, (8, 8), (8, 8), lm, policy=POLICY, gains=side['gains']))
    os.environ['nnUNet_n_proc_DA'] = '0'
    assert trainer_class('baseline') is Baseline and trainer_class('q') is nnUNetTrainerDataset509Quality
    assert trainer_class('quality_filter') is nnUNetTrainerDataset509Quality and trainer_class('c') is C
    assert C._build_loss is Baseline._build_loss and C.validation_step is Baseline.validation_step
    config = read(Path(__file__).with_name('c.json'))
    config['curriculum_sidecar'] = dict(path=str(output), sha256=sha(output))
    config_path = root / 'config.json'; write_new(config_path, config)
    os.environ.update(nnUNet_preprocessed=str(base.parent), nnUNet_results=str(root / 'results'),
                      LESIONGEN512_POOL=str(pool), LESIONGEN512_POOL_SHA256=sha(pool / 'manifest.json'),
                      DATASET509_QUALITY_CONFIG=str(config_path), nnUNet_compile='false')
    trainer = C(read(plans), '2d', 0, dict(channel_names={'0': 'DWI'}, labels={'background': 0, 'lesion': 1}),
                torch.device('cpu'))
    trainer.enable_deep_supervision = False
    loss = trainer._build_loss()
    from loss import SourceLoss
    assert not isinstance(loss, SourceLoss)
    x = torch.randn(2, 2, 8, 8, requires_grad=True)
    y = torch.randint(0, 2, (2, 1, 8, 8))
    value = loss(x, y); value.backward()
    assert torch.isfinite(value) and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    steps = []
    trainer.network = torch.nn.Identity()
    trainer.optimizer = SimpleNamespace(param_groups=[{'lr': .01}])
    trainer.lr_scheduler = SimpleNamespace(step=lambda epoch: steps.append(epoch))
    trainer.logger = SimpleNamespace(log=lambda *args: None)
    trainer.print_to_log_file = lambda *args, **kwargs: None
    trainer.dataloader_train = wrapper
    trainer.current_epoch = 100
    trainer.on_train_epoch_start()
    assert steps == [100] and trainer.current_epoch == 100 and next(wrapper)['c_epoch'] == 100
    legacy = job_layout(root, {})
    assert legacy['training'].name == 'q_training' and legacy['config'].name == 'q.formal.json'
    assert job_layout(root, dict(mode='c'))['metrics'].name == 'c_metrics'
    reject(lambda: job_layout(root, dict(mode='c', training_name='../q_training')))
    # A held-out string cannot appear in reference cases; metadata builder never opens validation files.
    assert all(cid.startswith('fixture') for cid in side['spacing_provenance'])
    print('Observed synthetic fractions early/late:', float(np.mean(early)), float(np.mean(late)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lesiongen-repo', required=True)
    args = parser.parse_args()
    repo = Path(args.lesiongen_repo)
    sys.path[:0] = [str(repo / 'src'), str(repo / 'trainers')]
    os.environ['nnUNet_n_proc_DA'] = '0'
    torch.set_num_threads(1)
    from api_contract import verify
    verify()
    test_math()
    print('PASS C physical units/reference groups/stages/gain normalization/empty-candidate boundaries', flush=True)
    with tempfile.TemporaryDirectory(prefix='dataset509_c_cpu_') as tmp:
        test_real_path(Path(tmp))
    print('PASS C complete offline fixture/real loader/Z/foreground/DA/live epoch/original loss/gradient/layout compatibility', flush=True)
    print('C_ENGINEERING_CHECKS_PASS; real sidecar/distribution/GPU preflight/training NOT RUN', flush=True)


if __name__ == '__main__':
    main()
