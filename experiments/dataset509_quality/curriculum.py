"""Training-only 2D physical-area/contrast curriculum metadata and probabilities."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from quality import need, read, sha, digest, write_new, contrast, summary

GROUPS = ['small_low', 'small_nonlow', 'large_low', 'large_nonlow', 'unknown_contrast', 'negative']
POLICY = dict(schema='dataset509-curriculum-policy/v1', warmup_epochs=20, transition_epochs=80,
              synthetic_start=.10, synthetic_end=.25, trusted_quality_min=.5,
              gain_cap=2., area_quantile=.5, contrast_quantile=.25)


def validate_policy(policy):
    need(set(policy) == set(POLICY) and policy['schema'] == POLICY['schema'], 'C_POLICY_SCHEMA')
    need(type(policy['warmup_epochs']) is int and policy['warmup_epochs'] >= 0 and
         type(policy['transition_epochs']) is int and policy['transition_epochs'] > 0, 'C_EPOCH_POLICY')
    values = [policy[k] for k in ('synthetic_start', 'synthetic_end', 'trusted_quality_min',
                                  'gain_cap', 'area_quantile', 'contrast_quantile')]
    need(np.isfinite(values).all() and 0 <= policy['synthetic_start'] <= policy['synthetic_end'] <= .5 and
         0 < policy['trusted_quality_min'] <= 1 and 1 <= policy['gain_cap'] <= 2 and
         0 < policy['area_quantile'] < 1 and 0 < policy['contrast_quantile'] < 1, 'C_POLICY_RANGE')


def phase(epoch, policy):
    validate_policy(policy)
    need(type(epoch) is int and epoch >= 0, 'C_INVALID_EPOCH')
    return float(np.clip((epoch - policy['warmup_epochs']) / policy['transition_epochs'], 0, 1))


def physical_area(pixels, geometry):
    # geometry is the pinned provenance from audit.read_volume; no guessed pixel units.
    spacing = np.asarray(geometry.get('spacing_xyz_mm', []), dtype=float)
    affine = np.asarray(geometry.get('affine_xyz_to_lps_mm', []), dtype=float)
    need(spacing.shape == (3,) and np.isfinite(spacing).all() and (spacing > 0).all() and
         affine.shape == (4, 4) and np.isfinite(affine).all(), 'C_TRUSTED_MM_SPACING_REQUIRED')
    basis = affine[:3, :3]
    need(np.allclose(np.linalg.norm(basis, axis=0), spacing, atol=1e-4, rtol=0), 'C_SPACING_AFFINE_MISMATCH')
    directions = basis / spacing
    need(np.allclose(directions.T @ directions, np.eye(3), atol=1e-4), 'C_SHEARED_GEOMETRY_UNSUPPORTED')
    return float(pixels * spacing[0] * spacing[1])


def group(area, score, thresholds):
    if area == 0:
        return 5
    if score is None:
        return 4
    return int(area > thresholds['area_mm2']) * 2 + int(score >= thresholds['contrast'])


def reference_statistics(entries, policy):
    validate_policy(policy)
    positive = [e for e in entries if e['area_mm2'] > 0]
    valid = [e['contrast'] for e in positive if e['contrast'] is not None]
    need(len(positive) > 0 and len(valid) > 0, 'C_NO_POSITIVE_TRAINING_REFERENCE')
    thresholds = dict(area_mm2=float(np.quantile([e['area_mm2'] for e in positive], policy['area_quantile'])),
                      contrast=float(np.quantile(valid, policy['contrast_quantile'])))
    counts = np.bincount([group(e['area_mm2'], e['contrast'], thresholds) for e in entries], minlength=6)
    maximum = int(counts[:4].max())
    # Groups absent in real training receive the bounded cap, never an infinity.
    gains = [float(min(policy['gain_cap'], np.sqrt(maximum / max(int(n), 1)))) for n in counts[:4]] + [1., 1.]
    return thresholds, counts.tolist(), gains


def probabilities(current, epoch, policy, gains, force_fg=False):
    """Joint per-plane probabilities for a uniformly selected case, conditional on FG slot."""
    alpha = phase(epoch, policy)
    source = np.asarray(current['source'])
    groups = np.asarray(current['group'])
    quality = np.asarray(current['quality'])
    foreground = np.asarray(current['foreground_pixels'])
    need(source.ndim == 1 and groups.shape == quality.shape == foreground.shape == source.shape and
         ((source == 0) | (source == 1)).all() and ((groups >= 0) & (groups < 6)).all() and
         np.isfinite(quality).all() and ((quality >= 0) & (quality <= 1)).all(), 'C_CURRENT_METADATA_INVALID')
    gain = np.asarray(gains, dtype=float)
    need(gain.shape == (6,) and np.isfinite(gain).all() and (gain >= 1).all() and
         (gain <= policy['gain_cap']).all(), 'C_GAIN_INVALID')
    real = source == 0
    need(real.any(), 'C_REAL_PLANES_REQUIRED')
    # Real foreground slots retain voxel-mass proportional Z sampling; random slots
    # retain uniform real Z, including negative slices.
    real_weights = real * (foreground if force_fg else np.ones(len(source)))
    real_fg_fallback = not real_weights.any()
    if real_fg_fallback:
        real_weights = real.astype(float)
    real_weights = real_weights / real_weights.sum()
    trusted = (source == 1) & (quality >= policy['trusted_quality_min']) & (groups < 4)
    if force_fg:
        trusted &= foreground > 0
    # Difficult = low contrast OR a genuinely under-covered training group.
    difficult = (groups % 2 == 0) | (gain[groups] > 1 + 1e-12)
    synthetic_weights = trusted * gain[groups] * np.where(difficult, alpha, 1.)
    available = float(synthetic_weights.sum())
    nominal = policy['synthetic_start'] + alpha * (policy['synthetic_end'] - policy['synthetic_start'])
    syn_probability = float(nominal if available > 0 else 0.)
    syn = synthetic_weights / available if available > 0 else np.zeros(len(source))
    joint = (1 - syn_probability) * real_weights + syn_probability * syn
    joint = joint / joint.sum()
    status = 'available' if available > 0 else ('waiting_for_difficult_phase' if trusted.any() else 'no_trusted_candidate')
    return joint, dict(alpha=alpha, synthetic_probability=syn_probability, nominal_synthetic_probability=float(nominal),
                       fallback=status, real_foreground_unavailable=bool(force_fg and real_fg_fallback))


def policy_audit(cases, epoch, policy, gains):
    result = {'epoch': epoch, 'alpha': phase(epoch, policy), 'conditional': {}}
    for force_fg in (False, True):
        joint = np.zeros((2, 6), dtype=float)
        fallbacks = {}
        for entry in cases.values():
            metadata = case_current(entry)
            p, state = probabilities(metadata, epoch, policy, gains, force_fg)
            np.add.at(joint, (metadata['source'], metadata['group']), p)
            fallbacks[state['fallback']] = fallbacks.get(state['fallback'], 0) + 1
        joint /= len(cases)
        result['conditional']['foreground_slot' if force_fg else 'random_slot'] = dict(
            source_group_probabilities=joint.tolist(), source_probabilities=joint.sum(1).tolist(),
            fallback_cases=fallbacks)
    return result


def case_current(entry):
    real, synthetic = entry['real'], entry['synthetic']
    planes = real + synthetic
    return dict(source=np.array([0] * len(real) + [1] * len(synthetic), dtype=np.int64),
                group=np.array([p['group'] for p in planes], dtype=np.int64),
                quality=np.array([1.] * len(real) + [p['quality'] for p in synthetic], dtype=float),
                foreground_pixels=np.array([p['foreground_pixels'] for p in planes], dtype=np.int64))


def build(package, base, plans_path, quality_path, output, policy=POLICY):
    validate_policy(policy)
    package, base, output = Path(package).resolve(), Path(base).resolve(), Path(output).resolve()
    need(not output.exists(), 'C_SIDECAR_EXISTS')
    need(base not in output.parents and package not in output.parents and output not in base.parents and
         output not in package.parents and output not in (base, package), 'C_OUTPUT_INPUT_OVERLAP')
    pool = base / 'synthetic_pool'
    q = read(quality_path)
    pm = read(pool / 'manifest.json')
    plans = read(plans_path)
    need(sha(plans_path) == pm['plans_sha256'] and plans['transpose_forward'] == [0, 1, 2], 'C_PLANS_AXES_CHANGED')
    split_path = base / 'splits_final.json'
    split = read(split_path)[0]
    train, val = set(split['train']), set(split['val'])
    need(len(train) == 100 and len(val) == 28 and not train & val, 'C_TRAIN100_SPLIT')
    need(q['split_sha256'] == pm['split_sha256'] == sha(split_path) and
         q['pool_manifest_sha256'] == sha(pool / 'manifest.json'), 'C_Q_POOL_SPLIT_CHANGED')
    need(q['package_manifest_sha256'] == sha(package / 'train_manifest.json') and
         q['provenance_sha256'] == sha(package / 'provenance.private.json'), 'C_NATIVE_PACKAGE_CHANGED')
    need(q['scoring_sha256'] == digest(q['scoring']), 'C_QUALITY_SCORING_CHANGED')
    provenance = read(package / 'provenance.private.json')
    rows = read(package / 'train_manifest.json')['samples']
    need(len(rows) == 2071 and len({r['path'] for r in rows}) == len(rows), 'C_PACKAGE_COUNTS')
    origins = {r['sample']: r for r in provenance['samples']}
    records = {r['case_id']: r for r in provenance['cases']}
    need(set(records) == train and all(r['split'] == 'train' for r in records.values()), 'C_GEOMETRY_NOT_TRAIN_ONLY')
    native = {cid: dict(real={}, synthetic=[]) for cid in train}
    all_real, real_hashes = [], []
    for row in rows:
        need(row['kind'] in ('REAL', 'SYNTHETIC'), 'C_SOURCE_KIND')
        origin = origins[row['sample']]
        cid = origin['case_id']
        need(cid in train and origin['split'] == 'train', 'C_REFERENCE_LEAKAGE')
        path = (package / row['path']).resolve()
        need(package in path.parents and sha(path) == row['sha256'], 'C_NATIVE_PAIR_CHANGED')
        geometry = records[cid]['geometry']
        with np.load(path, allow_pickle=False) as pair:
            score, size = contrast(pair['image'], pair['label'], q['scoring'])
            need(list(pair['image'].shape) == geometry['size_xyz'][:2], 'C_NATIVE_XY_SHAPE')
        e = dict(area_mm2=physical_area(size, geometry), contrast=score, native_z=int(origin['center_z']),
                 native_pair_sha256=row['sha256'])
        need(0 <= e['native_z'] < geometry['size_xyz'][2], 'C_NATIVE_Z_RANGE')
        if row['kind'] == 'REAL':
            need(e['native_z'] not in native[cid]['real'], 'C_DUPLICATE_NATIVE_REAL_Z')
            native[cid]['real'][e['native_z']] = e
            all_real.append(e)
            real_hashes.append(row['sha256'])
        else:
            native[cid]['synthetic'].append(e)
    need(len(real_hashes) == 1593 and sum(len(v['synthetic']) for v in native.values()) == 478, 'C_POOL_COUNTS')
    need(digest(dict(split=sha(split_path), pairs=real_hashes)) == q['training_reference_sha256'], 'C_TRAINING_REFERENCE_CHANGED')
    thresholds, counts, gains = reference_statistics(all_real, policy)
    cache = base / plans['configurations']['2d']['data_identifier']
    cases, spacing_refs = {}, {}
    for cid in sorted(train):
        prop_path = cache / (cid + '.pkl')
        # Existing, trusted nnU-Net-generated local properties only (not arbitrary downloads).
        with prop_path.open('rb') as f:
            props = pickle.load(f)
        z0, z1 = props['bbox_used_for_cropping'][0]
        geometry = records[cid]['geometry']
        need(set(native[cid]['real']) == set(range(geometry['size_xyz'][2])), 'C_NATIVE_REAL_Z_COVERAGE')
        data_path, seg_path = cache / (cid + '.npy'), cache / (cid + '_seg.npy')
        real_data = np.load(data_path, mmap_mode='r', allow_pickle=False)
        real_seg = np.load(seg_path, mmap_mode='r', allow_pickle=False)
        need(real_data.shape == real_seg.shape and real_data.shape[0] == 1 and
             real_data.shape[1] == z1-z0 and np.isin(real_seg, [0, 1]).all(), 'C_PREPARED_Z_MAPPING_INVALID')
        real = [dict(native[cid]['real'][z]) for z in range(z0, z1)]
        for z, entry in enumerate(real):
            entry.update(group=group(entry['area_mm2'], entry['contrast'], thresholds),
                         foreground_pixels=int((real_seg[0, z] == 1).sum()))
        synthetic = native[cid]['synthetic']
        if synthetic:
            record = pm['cases'][cid]
            path = (pool / record['file']).resolve()
            need(pool in path.parents and sha(path) == record['sha256'], 'C_PREPARED_POOL_CHANGED')
            q_entries = q['cases'][cid]
            with np.load(path, allow_pickle=False) as pair:
                seg = pair['seg']
                need(seg.shape == (1, len(synthetic), *real_data.shape[2:]) and
                     len(synthetic) == record['count'] == len(q_entries), 'C_SYNTHETIC_Z_MAPPING')
                for i, (entry, quality) in enumerate(zip(synthetic, q_entries)):
                    need(entry['native_pair_sha256'] == quality['native_pair_sha256'] and
                         entry['contrast'] == quality['contrast'], 'C_Q_PLANE_ORDER_CHANGED')
                    entry.update(group=group(entry['area_mm2'], entry['contrast'], thresholds),
                                 quality=quality['weight'], foreground_pixels=int((seg[0, i] == 1).sum()))
        cases[cid] = dict(real=real, synthetic=synthetic, properties_sha256=sha(prop_path),
                          real_seg_sha256=sha(seg_path), real_shape=list(real_data.shape))
        spacing_refs[cid] = dict(geometry_sha256=digest(geometry), raw_hashes=records[cid]['hashes'],
                                spacing_xyz_mm=geometry['spacing_xyz_mm'])
        del real_data, real_seg
    need(set(pm['cases']) == {cid for cid, e in cases.items() if e['synthetic']}, 'C_POOL_CASES')
    result = dict(schema='dataset509-curriculum/v1', policy=policy, policy_sha256=digest(policy),
                  group_names=GROUPS, thresholds=thresholds, real_reference_group_counts=counts, gains=gains,
                  pool_manifest_sha256=sha(pool / 'manifest.json'), split_sha256=sha(split_path),
                  plans_sha256=sha(plans_path), quality_sidecar_sha256=sha(quality_path),
                  package_manifest_sha256=q['package_manifest_sha256'], provenance_sha256=q['provenance_sha256'],
                  training_reference_sha256=digest(dict(real_reference=q['training_reference_sha256'],
                                                        spacing=spacing_refs, thresholds=thresholds)),
                  spacing_provenance=spacing_refs, cases=cases,
                  area_definition='native XY binary-lesion pixel count * spacing_x_mm * spacing_y_mm; 2D area, not 3D volume',
                  evidence='TRAINING_ONLY_CANDIDATE_QUALITY_GATE_NOT_VALIDATED')
    result['audit'] = dict(native_real_area_mm2=summary([e['area_mm2'] for e in all_real if e['area_mm2'] > 0]),
                           native_real_positive_contrast=summary([e['contrast'] for e in all_real
                                                                 if e['area_mm2'] > 0 and e['contrast'] is not None]),
                           missing_real_groups=[GROUPS[i] for i in range(4) if counts[i] == 0],
                           stages=[policy_audit(cases, epoch, policy, gains) for epoch in
                                   (0, policy['warmup_epochs'], policy['warmup_epochs'] + policy['transition_epochs'])])
    write_new(output, result)
    return dict(group_names=GROUPS, thresholds=thresholds, real_reference_group_counts=counts,
                gains=gains, audit=result['audit'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('package', 'base', 'plans', 'quality-sidecar', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--config', help='Read only the curriculum policy field; defaults are fixed in code')
    args = parser.parse_args()
    policy = read(args.config)['curriculum'] if args.config else POLICY.copy()
    print(json.dumps(build(args.package, args.base, args.plans, args.quality_sidecar, args.output, policy), allow_nan=False))
