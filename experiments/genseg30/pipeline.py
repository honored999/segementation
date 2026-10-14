"""Fresh, offline train-only GAN preparation and an explicit human QA gate."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np
import torch

from .contract import (CLASSIFICATION, allocate_counts, check_output, checked_file, read_json,
                       require, sha256, validate_arrays, write_json, capture_identities, assert_identities)
from .models import FILES, UPSTREAM_COMMIT, Generator, pad_plane, train_step

ACKNOWLEDGEMENT = 'I reviewed the training-only generated image-mask previews and approve this exploratory pool.'


def now():
    return datetime.now(timezone.utc).isoformat()


def train_generator(source, output, *, epochs=200, max_steps=5000, width=64, downs=8,
                    discriminator_width=64, seed=20261014, device='cpu', lr=2e-5, l1_weight=100.):
    require(type(epochs) is int and epochs > 0 and type(max_steps) is int and max_steps > 0,
            'Explicit positive epochs/max_steps required; budget applies ONLY to generator')
    require(discriminator_width > 0 and np.isfinite(lr) and lr > 0 and
            np.isfinite(l1_weight) and l1_weight >= 0, 'Invalid GAN settings')
    output = check_output(output, source.protected)
    # Inventory precedes any source load. Rechecked after training, detecting source changes.
    inventory = {p: source.inventory(p) for p in source.train}
    dataset = source.training_dataset()
    cases, lo, hi = {}, float('inf'), float('-inf')
    for patient in source.train:
        image, seg, _ = source.load_train(dataset, patient)
        require(np.any(seg >= 0), 'Patient has no valid pixels')
        cases[patient] = {'shape': list(image.shape), 'original_slices': image.shape[1]}
        lo, hi = min(lo, float(image.min())), max(hi, float(image.max()))
    require(hi > lo and np.isfinite(hi - lo), 'Nondegenerate train-only min/max scale required')
    output.mkdir(parents=True, exist_ok=False)
    settings = {'epochs': epochs, 'max_steps': max_steps, 'width': width, 'downs': downs,
                'discriminator_width': discriminator_width, 'seed': seed, 'device': str(device),
                'lr': lr, 'betas': [.5, .999], 'l1_weight': l1_weight, 'batch_size': 1,
                'gan_loss': 'vanilla', 'norm': 'batch', 'dropout': False,
                'architecture': 'fixed_seeded_upstream_initial_logits_no_arch_optimizer',
                'pad': 'bottom_right_constant_to_multiple_2**downs_no_resampling',
                'training_order': 'seeded_patient_then_all_z_shuffle_each_epoch',
                'generator_augmentation': 'none', 'scheduler': 'constant_lr',
                'checkpoint_selection': 'last_completed_update_no_validation_tuning'}
    write_json(output / 'started.json', {'classification': CLASSIFICATION, 'complete': False,
                'source': source.identity, 'settings': settings, 'started_utc': now()})
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    generator = Generator(width, downs, seed, device)
    discriminator = generator.discriminator(discriminator_width)
    optimizer_g = torch.optim.Adam(generator.net.parameters(), lr=lr, betas=(.5, .999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(.5, .999))
    steps, finished_epochs, losses = 0, 0, {}
    for epoch in range(epochs):
        for patient in rng.permutation(source.train):
            image, seg, _ = source.load_train(dataset, str(patient))
            for z in rng.permutation(image.shape[1]):
                plane = image[0, z]
                label = seg[0, z]
                multiple = 2**downs
                condition = pad_plane((label == 1).astype(np.float32), multiple, 0)
                target = pad_plane((2 * (plane - lo) / (hi - lo) - 1).astype(np.float32), multiple, 0)
                valid = pad_plane((label >= 0).astype(np.float32), multiple, 0)
                args = [torch.from_numpy(arr[None, None]).to(generator.device) for arr in (condition, target, valid)]
                losses = train_step(generator, discriminator, optimizer_g, optimizer_d, args[0], args[1],
                                    l1_weight, args[2])
                steps += 1
                if steps == 1 or steps % 100 == 0:
                    print(f'GENERATOR TRAINING: update={steps}/{max_steps}, epoch={epoch + 1}/{epochs}', flush=True)
                if steps >= max_steps:
                    break
            if steps >= max_steps:
                break
        if steps >= max_steps:
            # Exact last epoch completeness, independent of stop reason.
            finished_epochs = steps // sum(case['original_slices'] for case in cases.values())
            break
        finished_epochs = epoch + 1
    source.verify_inventory(inventory)
    metadata = {'format': 1, 'classification': CLASSIFICATION, 'complete': True, 'source': source.identity,
                'inventory': inventory, 'cases': cases, 'scaling': {'method': 'global_train_minmax',
                'min': lo, 'max': hi}, 'settings': settings, 'actual_steps': steps,
                'completed_epochs': finished_epochs, 'last_losses': losses, 'completed_utc': now(),
                'upstream_commit': UPSTREAM_COMMIT, 'upstream_files': FILES,
                'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__),
                            'numpy': str(np.__version__)}}
    generator.save(output / 'generator.pt', metadata, discriminator)
    write_json(output / 'generator.json', {'metadata': metadata, 'checkpoint_sha256': sha256(output / 'generator.pt')})
    print(f'GENERATOR TRAINING COMPLETE: actual_updates={steps}, complete_epochs={finished_epochs}; '
          'train-only offline exploration; effect unknown.', flush=True)
    return metadata


def load_run(source, run):
    run = check_output(run, source.protected, fresh=False)
    record = read_json(checked_file(run, 'generator.json'))
    metadata = record['metadata']
    require(metadata.get('complete') is True and metadata.get('classification') == CLASSIFICATION and
            metadata.get('format') == 1, 'Incomplete/foreign generator run')
    source.verify_identity(metadata['source'])
    require(metadata.get('upstream_commit') == UPSTREAM_COMMIT and metadata.get('upstream_files') == FILES,
            'Generator upstream identity mismatch')
    require(set(metadata['cases']) == set(source.train), 'Foreign generator case metadata')
    require(metadata['settings']['max_steps'] >= metadata['actual_steps'] > 0 and
            metadata['settings']['epochs'] > 0, 'Invalid generator update budget')
    require(metadata['scaling']['method'] == 'global_train_minmax' and
            np.isfinite(metadata['scaling']['min']) and np.isfinite(metadata['scaling']['max']) and
            metadata['scaling']['max'] > metadata['scaling']['min'], 'Invalid recorded train scaling')
    checkpoint = checked_file(run, 'generator.pt')
    require(sha256(checkpoint) == record['checkpoint_sha256'], 'Generator checkpoint hash mismatch')
    return record, checkpoint


def records_for_cases(cases, seed):
    rng = np.random.default_rng(seed)
    allocation = allocate_counts({p: cases[p]['original_slices'] for p in cases})
    records = {}
    for patient in sorted(cases):
        records[patient] = [{'source_patient': patient, 'source_z': int(rng.integers(cases[patient]['original_slices'])),
                             'flip_y': bool(rng.integers(2)), 'flip_x': bool(rng.integers(2))}
                            for _ in range(allocation[patient])]
    return allocation, records


def flip_plane(plane, record):
    if record['flip_y']:
        plane = np.flip(plane, axis=-2)
    if record['flip_x']:
        plane = np.flip(plane, axis=-1)
    return np.ascontiguousarray(plane)


def save_preview(path, original, generated, label, scaling):
    from PIL import Image, ImageDraw
    lo, hi = scaling['min'], scaling['max']
    def rgb(plane):
        pixels = np.clip((plane - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        return np.repeat(pixels[:, :, None], 3, axis=2)
    a, b = rgb(original), rgb(generated)
    overlay = b.copy()
    overlay[label == 1] = (.4 * overlay[label == 1] + .6 * np.array([255, 0, 0])).astype(np.uint8)
    overlay[label == -1] = [255, 255, 0]
    canvas = Image.fromarray(np.concatenate([a, b, overlay], axis=1))
    labelled = Image.new('RGB', (canvas.width, canvas.height + 20), 'black')
    labelled.paste(canvas, (0, 20))
    ImageDraw.Draw(labelled).text((2, 2), 'source | generated | red=lesion yellow=-1', fill='white')
    with path.open('xb') as stream:
        labelled.save(stream, format='PNG')  # visualization only; never fed to a generator/nnU-Net


def export_pool(source, run, output, *, seed=20261015, device='cpu'):
    require(type(seed) is int and 0 <= seed < 2**32, 'Seed must be uint32')
    output = check_output(output, source.protected + [Path(run)])
    record, checkpoint = load_run(source, run)
    metadata = record['metadata']
    source.verify_inventory(metadata['inventory'])
    generator, restored_meta = Generator.load(checkpoint, device)
    require(restored_meta == metadata, 'Generator metadata mismatch between checkpoint and sidecar')
    generator.net.eval()
    allocation, records = records_for_cases(metadata['cases'], seed)
    output.mkdir(parents=True, exist_ok=False)
    dataset, cases, previews = source.training_dataset(), {}, {}
    scaling = metadata['scaling']
    for patient in source.train:
        image, seg, _ = source.load_train(dataset, patient)
        require(list(image.shape) == metadata['cases'][patient]['shape'], 'Source native shape changed')
        shape = (1, allocation[patient], *image.shape[2:])
        extra, targets = np.empty(shape, np.float32), np.empty(shape, np.int16)
        for index, entry in enumerate(records[patient]):
            label = flip_plane(seg[0, entry['source_z']], entry)
            original = flip_plane(image[0, entry['source_z']], entry)
            condition = pad_plane((label == 1).astype(np.float32), 2**generator.config['downs'], 0)
            with torch.no_grad():
                fake = generator.net(torch.from_numpy(condition[None, None]).to(generator.device)).cpu().numpy()[0, 0]
            require(fake.shape == condition.shape and np.isfinite(fake).all(), 'Generated shape/finiteness failure')
            generated = (fake[:label.shape[0], :label.shape[1]] + 1) * (scaling['max'] - scaling['min']) / 2 + scaling['min']
            generated[label == -1] = original[label == -1]  # retain source sentinel-image support
            extra[0, index], targets[0, index] = generated, label
            preview_name = f'{patient}_{index:06d}_preview.png'
            save_preview(output / preview_name, original, generated, label, scaling)
            previews[preview_name] = sha256(output / preview_name)
        validate_arrays(extra, targets, allow_empty=True)
        name = patient + '_synthetic.npz'
        with (output / name).open('xb') as stream:
            np.savez_compressed(stream, data=extra, seg=targets)
        cases[patient] = {'file': name, 'sha256': sha256(output / name),
                          'original_slices': image.shape[1], 'synthetic_slices': allocation[patient],
                          'source_shape': list(image.shape), 'records': records[patient]}
    source.verify_inventory(metadata['inventory'])
    manifest = {'format': 1, 'classification': CLASSIFICATION, 'complete': True, 'created_utc': now(),
                'source': source.identity, 'generator': metadata, 'generator_checkpoint': str(checkpoint),
                'generator_sha256': record['checkpoint_sha256'], 'seed': seed,
                'count_unit': 'all_train_preprocessed_2d_z_slices',
                'original_slices': sum(c['original_slices'] for c in cases.values()),
                'synthetic_slices': sum(allocation.values()), 'cases': cases, 'previews': previews,
                'transforms': 'seeded_independent_inplane_flips_no_resize_no_affine',
                'sampling_only': True, 'physical_volume': False}
    write_json(output / 'manifest.json', manifest)  # completion marker is created LAST
    print('POOL COMPLETE, NOT QUALITY APPROVED: manifest_sha256=' + sha256(output / 'manifest.json'), flush=True)
    return manifest


def audit_pool(source, pool, *, require_quality=False, expected_sha256=None, return_identities=False):
    pool = check_output(pool, source.protected, fresh=False)
    manifest_path = checked_file(pool, 'manifest.json')
    digest = sha256(manifest_path)
    if expected_sha256 is not None:
        require(digest == expected_sha256, 'Pinned pool manifest hash mismatch')
    manifest = read_json(manifest_path)
    require(manifest.get('format') == 1 and manifest.get('complete') is True and
            manifest.get('classification') == CLASSIFICATION, 'Incomplete/foreign pool')
    source.verify_identity(manifest['source'])
    require(manifest.get('count_unit') == 'all_train_preprocessed_2d_z_slices' and
            manifest.get('sampling_only') is True and manifest.get('physical_volume') is False and
            manifest.get('transforms') == 'seeded_independent_inplane_flips_no_resize_no_affine', 'Pool protocol changed')
    require(set(manifest['cases']) == set(source.train), 'Foreign/incomplete pool patients')
    checkpoint = Path(manifest['generator_checkpoint']).resolve()
    check_output(checkpoint.parent, source.protected, fresh=False)
    require(checkpoint.name == 'generator.pt', 'Foreign generator checkpoint filename')
    if require_quality:
        require((pool / 'quality_approval.json').is_file(), 'Explicit human quality approval required')
    # Capture identities BEFORE the content audit and check them again afterwards.
    # This binds the hash audit to the same immutable files later used by data workers.
    from . import contract
    paths = [source.folder, source.base, pool, checkpoint.parent, manifest_path,
             source.dataset_path, source.plans_path, source.splits, contract.REFERENCE,
             checked_file(checkpoint.parent, 'generator.pt'), checked_file(checkpoint.parent, 'generator.json')]
    paths += [path for patient in source.train for path in source.case_files(patient)]
    paths += [checked_file(pool, patient + '_synthetic.npz') for patient in source.train]
    paths += [checked_file(pool, name) for name in manifest['previews']]
    if require_quality:
        paths.append(checked_file(pool, 'quality_approval.json'))
    identities = capture_identities(paths)
    require(sha256(manifest_path) == digest, 'Manifest changed during audit')
    record, expected_checkpoint = load_run(source, checkpoint.parent)
    require(checkpoint == expected_checkpoint and manifest['generator'] == record['metadata'] and
            manifest['generator_sha256'] == record['checkpoint_sha256'], 'Pool/generator provenance mismatch')
    counts, records = records_for_cases(record['metadata']['cases'], manifest['seed'])
    total = sum(c['original_slices'] for c in record['metadata']['cases'].values())
    require(manifest['original_slices'] == total and manifest['synthetic_slices'] == (3 * total + 9) // 10,
            'Pool does not contain exactly ceil30% train slices')
    expected_previews = set()
    for patient in source.train:
        case = manifest['cases'][patient]
        source_case = record['metadata']['cases'][patient]
        require(case['source_shape'] == source_case['shape'] and
                case['original_slices'] == source_case['original_slices'] and
                case['synthetic_slices'] == counts[patient] and case['records'] == records[patient],
                'Case counts/shape/source-z/transforms mismatch')
        require(case['file'] == patient + '_synthetic.npz', 'Foreign pool case filename')
        path = checked_file(pool, case['file'])
        require(sha256(path) == case['sha256'], 'Synthetic slice file hash mismatch')
        expected_previews.update(f'{patient}_{i:06d}_preview.png' for i in range(counts[patient]))
    require(set(manifest['previews']) == expected_previews, 'Missing/foreign quality previews')
    for name, expected in manifest['previews'].items():
        require(sha256(checked_file(pool, name)) == expected, 'Preview hash mismatch')
    if require_quality:
        require((pool / 'quality_approval.json').is_file(), 'Explicit human quality approval required')
        approval = read_json(checked_file(pool, 'quality_approval.json'))
        require(approval.get('acknowledgement') == ACKNOWLEDGEMENT and
                approval.get('manifest_sha256') == digest and bool(approval.get('reviewer', '').strip()) and
                approval.get('previews') == manifest['previews'], 'Invalid explicit quality approval')
    # No source dataset load can occur until every provenance and artifact check above succeeds.
    source.verify_inventory(record['metadata']['inventory'])
    for patient in source.train:
        case = manifest['cases'][patient]
        with np.load(checked_file(pool, case['file']), allow_pickle=False) as arrays:
            image, seg = arrays['data'], arrays['seg']
        validate_arrays(image, seg, allow_empty=True)
        require(list(image.shape) == [1, counts[patient], *case['source_shape'][2:]], 'Pool native shape mismatch')
    assert_identities(identities)
    return (manifest, identities) if return_identities else manifest


def approve_pool(source, pool, reviewer, acknowledgement):
    require(acknowledgement == ACKNOWLEDGEMENT and isinstance(reviewer, str) and reviewer.strip(),
            'Explicit quality acknowledgement and reviewer name required; no auto-approval')
    manifest = audit_pool(source, pool)
    pool = Path(pool).resolve()
    write_json(pool / 'quality_approval.json', {'manifest_sha256': sha256(pool / 'manifest.json'),
                'reviewer': reviewer.strip(), 'acknowledgement': acknowledgement, 'approved_utc': now(),
                'previews': manifest['previews'], 'meaning': 'Human quality acceptance, not a claim of metric benefit'})
    return sha256(pool / 'manifest.json')
