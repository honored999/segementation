"""Read-only preview of the completed bounded diagnostic; never a training handoff."""
import hashlib
import inspect
import math
import re
import subprocess
from pathlib import Path

import numpy as np
from experiments.genseg30.contract import (require, read_json, write_json, sha256,
    checked_file, check_output, capture_identities, assert_identities, validate_arrays)
from . import convergence, diagnostic, preview
from .data import Cohort, inner_split
from .models import Generator, source_identity
from .runtime import audit, gpu_guard, implementation_identity

CLASSIFICATION = 'exploratory_diagnostic_snapshot_fixed_multiplane_preview_not_formal_not_training'
EXPECTED_SHA256 = 'c025ddf4fcc300d95789d959984563d486071af25832fe465b4dab98b16a22b3'
EXPECTED_BYTES = 1100514439
GENERATION_COMMIT = '830642cc9ad07c84211188369d5315ad59b6dfcc'
MODEL_CONFIG = {'G': {'width': 64, 'downs': 8, 'seed': 20261018}, 'D_width': 64}


def compatible_code(protocol):
    """Historical producer and current reader are distinct; never compare the whole list."""
    saved = protocol.get('implementation_files_sha256')
    require(isinstance(saved, dict) and bool(saved) and all(isinstance(k, str) and
        isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v) for k, v in saved.items()),
        'Missing generation implementation identities')
    root = Path(__file__).parent
    # These unchanged modules define G, its functional eval and the input/statistics contract.
    for name in ('models.py', 'core.py', 'data.py', 'diagnostic.py'):
        raw = (root/name).read_bytes()
        require(saved.get(name) in (hashlib.sha256(raw).hexdigest(),
            hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest()),
            'Incompatible generation implementation: '+name)
    # restore_arch is inherited from the older adapter, outside the saved joint inventory.
    base = root.parent/'genseg30/models.py'
    original = subprocess.run(['git', 'show', GENERATION_COMMIT+':experiments/genseg30/models.py'],
        cwd=root, check=True, capture_output=True).stdout
    require(base.read_bytes().replace(b'\r\n', b'\n') == original.replace(b'\r\n', b'\n'),
        'Incompatible inherited generator implementation')
    return {'generation_commit': protocol['repository_commit'], 'generation_files_sha256': saved,
        'reader_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, check=True,
            capture_output=True, text=True).stdout.strip(),
        'reader_files_sha256': implementation_identity(),
        'inherited_models_sha256': sha256(base),
        'boundary': 'pinned upstream source plus unchanged models/core/data/diagnostic and inherited restore_arch; reader CLI may differ; strict G/alpha keys shapes dtypes finite audit'}


def ready_protocol(protocol):
    # Producer adds the final allocation ceiling after immutable readiness.
    return {k: v for k, v in protocol.items() if k != 'applied_allocator_fraction'}


def audit_diagnostic(target, run, digest):
    require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest),
        'Explicit lowercase snapshot SHA256 required')
    require(digest == EXPECTED_SHA256, 'Only the approved diagnostic snapshot is supported')
    run = Path(run).resolve()
    require(not (run/'failed.json').exists() and not (run/'complete.json').exists(),
        'Failed or training run refused')
    names = ('diagnostic.json', 'ready.json', convergence.SNAPSHOT_FILE)
    paths = [checked_file(run, name) for name in names]
    identities = capture_identities(paths)
    receipt, ready = (read_json(p) for p in paths[:2])
    require(receipt.get('format') == 1 and receipt.get('classification') == diagnostic.CLASSIFICATION
        and receipt.get('diagnostic_only') is True and receipt.get('complete') is True
        and receipt.get('completed_steps') == 500 and receipt.get('weights_saved') is True
        and receipt.get('reusable_checkpoint_saved') is False, 'Completed 500 diagnostic receipt required')
    p = receipt.get('protocol')
    require(isinstance(p, dict), 'Missing diagnostic protocol')
    expected = {'format': 1, 'classification': diagnostic.CLASSIFICATION, 'diagnostic_only': True,
        'seed': 20261018, 'steps': 500, 'strategy': 'adam_trainbn_epoch_v1',
        'convergence_audit': True, 'convergence_plan': convergence.convergence_plan(500),
        'convergence_policy': convergence.CONVERGENCE_POLICY,
        'initialization': 'identical_seed_cold_G_D_alpha_buffers; fresh_Adam_zero_moments; no_resume',
        'g_objective': 'gan_l1', 'architecture': 'both_alpha_frozen',
        'repeat_output_policy': diagnostic.REPEAT_OUTPUT_POLICY,
        'probe_backend_policy': diagnostic.PROBE_BACKEND_POLICY,
        'source': source_identity(), 'target_identity': target.identity,
        'target_manifest_sha256': target.manifest_sha256, 'repository_commit': GENERATION_COMMIT}
    for key, value in expected.items():
        require(p.get(key) == value, 'Diagnostic protocol mismatch: '+key)
    split = inner_split(target.train, target.val, 20261018)
    scale = p.get('scaling')
    require(isinstance(scale, dict) and scale.get('method') ==
        'inner61_valid_pixel_global_minmax_to_minus1_plus1_no_clip' and scale.get('fit_ids') == split['train']
        and all(type(scale.get(k)) in (int, float) and math.isfinite(scale[k]) for k in ('min', 'max'))
        and scale['max'] > scale['min'], 'Invalid saved inner61 scaling')
    selection = p.get('selection')
    require(isinstance(selection, dict) and selection.get('patient') in split['train']
        and selection.get('role') == 'train' and selection.get('category') == 'lesion'
        and type(selection.get('z')) is int and selection['z'] >= 0, 'Invalid anchor selection')
    require(ready.get('format') == 1 and ready.get('classification') == diagnostic.CLASSIFICATION
        and ready.get('diagnostic_only') is True and ready.get('complete') is False
        and ready.get('stage') == 'ready_after_fixed_plane_before_models_or_updates'
        and ready_protocol(ready.get('protocol', {})) == ready_protocol(p) and ready.get('policy') == diagnostic.READY_POLICY,
        'Ready protocol mismatch')
    c = receipt.get('convergence', {})
    require(c.get('plan') == convergence.convergence_plan(500) and c.get('policy') == convergence.CONVERGENCE_POLICY
        and c.get('scheduled_points_complete') is True, 'Incomplete convergence plan')
    s = c.get('snapshot', {})
    for key, value in {'format': 1, 'file': convergence.SNAPSHOT_FILE, 'sha256': digest,
        'bytes': EXPECTED_BYTES, 'steps': 500, 'not_pretraining': True, 'not_joint_handoff': True,
        'no_resume': True, 'optimizer_moments': False, 'rng_state': False,
        'weights_only_load': True, 'readback_exact': True}.items():
        require(s.get(key) == value, 'Snapshot receipt mismatch: '+key)
    require(paths[2].stat().st_size == EXPECTED_BYTES and sha256(paths[2]) == digest,
        'Snapshot bytes/hash mismatch')
    logs = receipt.get('output_logs_sha256', {})
    require(logs.get('ready.json') == sha256(paths[1]) and logs.get(convergence.SNAPSHOT_FILE) == digest,
        'Ready/snapshot receipt binding mismatch')
    for name in ('started.json', 'steps.jsonl', 'guard_events.jsonl', 'trajectory.json'):
        path = checked_file(run, name)
        require(logs.get(name) == sha256(path), 'Source log binding mismatch: '+name)
        identities.update(capture_identities([path]))
    require(c.get('trajectory_file') == 'trajectory.json' and c.get('trajectory_sha256') == logs['trajectory.json'],
        'Trajectory binding mismatch')
    trajectory = read_json(checked_file(run, 'trajectory.json'))
    require(trajectory.get('classification') == convergence.CLASSIFICATION and trajectory.get('diagnostic_only') is True
        and trajectory.get('plan') == c['plan'] and trajectory.get('scheduled_points_complete') is True,
        'Trajectory plan mismatch')
    provenance = compatible_code(p)
    target.verify_identity(target.identity)
    assert_identities(identities)
    return receipt, ready, identities, provenance


def load_diagnostic(run, digest, receipt, ready):
    payload = preview.load_checkpoint(run, convergence.SNAPSHOT_FILE, digest)
    require(isinstance(payload, dict) and set(payload) == {'format', 'classification', 'diagnostic_only',
        'not_pretraining', 'not_joint_handoff', 'no_resume', 'training_receipt', 'checkpoint_name',
        'steps', 'seed', 'models', 'model_config', 'source', 'protocol', 'input'}, 'Snapshot schema mismatch')
    for key, value in {'format': 1, 'classification': convergence.CLASSIFICATION, 'diagnostic_only': True,
        'not_pretraining': True, 'not_joint_handoff': True, 'no_resume': True,
        'training_receipt': False, 'checkpoint_name': 'not_gan_not_joint', 'steps': 500, 'seed': 20261018,
        'model_config': MODEL_CONFIG, 'source': receipt['protocol']['source'], 'protocol': receipt['protocol'],
        'input': {'ready_file': 'ready.json', 'ready_sha256': sha256(checked_file(run, 'ready.json')),
            'selection': receipt['protocol']['selection'], 'scaling': receipt['protocol']['scaling'],
            'condition': 'label==1', 'metric_region': 'label>=0'}}.items():
        require(payload.get(key) == value, 'Snapshot payload mismatch: '+key)
    require(ready_protocol(ready['protocol']) == ready_protocol(payload['protocol']), 'Snapshot ready mismatch')
    models = payload.pop('models')
    require(isinstance(models, dict) and set(models) == {'G', 'D', 'alpha'}, 'Snapshot model schema mismatch')
    # Drop D before constructing any model. No D/S/optimizer is instantiated.
    models.pop('D')
    return {'G': models['G'], 'arch': models['alpha'], 'model_config': payload['model_config'],
        'protocol': payload['protocol']}


def coverage(cohort, scale, selection, ready):
    """Predeclare all candidates and hashes before the first model forward."""
    patients, records, arrays = [], [], []
    chosen = sorted(cohort.train)[:4]
    anchor = (selection['patient'], selection['z'])
    for patient in chosen + ([] if selection['patient'] in chosen else [selection['patient']]):
        image, label, _ = cohort.load('train', patient)
        validate_arrays(image, label)
        planes, missing = preview.select_planes(label[0])
        patients.append({'patient': patient, 'missing_categories': missing,
            'all_ignore_planes': np.flatnonzero(~(label[0] >= 0).any(axis=(1, 2))).tolist(),
            'candidates': [{'z': z, 'category': cat} for z, cat in planes]})
        if patient not in chosen: planes = []
        if patient == anchor[0] and anchor[1] not in [z for z, _ in planes]:
            planes.append((anchor[1], 'anchor'))
        for z, category in planes:
            x, y, meta = cohort.plane('train', patient, z, scale, arrays=(image, label))
            hashes = diagnostic.ready_arrays(x, y)
            is_anchor = (patient, z) == anchor
            if is_anchor:
                require(hashes == ready['arrays'], 'Anchor CPU condition/label/target hash mismatch')
                require(meta['native_shape'] == selection['native_shape'] and
                    list(x.shape[2:]) == selection['padded_shape'], 'Anchor shape mismatch')
            records.append({'patient': patient, 'z': z, 'category': category, 'anchor': is_anchor,
                'native_shape': meta['native_shape'], 'padded_shape': list(x.shape[2:]), 'arrays': hashes})
            arrays.append((x, y, image[0, z].copy(), label[0, z].copy()))
    require(len(records) <= 13 and sum(r['anchor'] for r in records) == 1, 'Coverage/anchor bound failed')
    return {'rule': 'sorted(inner61)[:4]; first lesion + first two valid empty; missing never replaced; append anchor only',
        'seed': 20261018, 'patients': patients, 'planes': records}, arrays


def frozen_state(g):
    """Hash state without retaining another full model copy."""
    import torch
    tensors = dict(g.net.state_dict()) | {k: getattr(g.networks, k) for k in ('conv_arch', 'upconv_arch')}
    result = {}
    for key, value in tensors.items():
        a = value.detach().cpu().contiguous()
        result[key] = (str(a.dtype), list(a.shape), hashlib.sha256(a.numpy().tobytes()).hexdigest())
    result['modes'] = [m.training for m in g.net.modules()]
    result['requires_grad'] = [p.requires_grad for p in g.net.parameters()]
    require(not any(result['modes']) and not any(result['requires_grad']), 'Frozen eval G required')
    require(not g.networks.conv_arch.requires_grad and not g.networks.upconv_arch.requires_grad, 'Frozen alpha required')
    return result


def evaluate(g, x, y, shape, scale):
    """Match historical functional, grad-enabled eval before detach; no backward."""
    import torch
    from .core import FunctionalModel
    before = frozen_state(g)
    backend = diagnostic._probe_backend_snapshot()
    options = dict(enabled=None, benchmark=False, deterministic=True, allow_tf32=None)
    options.update({k: None for k in ('benchmark_limit', 'fp32_precision')
        if k in inspect.signature(torch.backends.cudnn.flags).parameters})
    try:
        with torch.backends.cudnn.flags(**options), torch.enable_grad():
            active = diagnostic._probe_backend_snapshot()
            require(active == dict(backend, deterministic=True, benchmark=False), 'Unexpected probe backend')
            f = FunctionalModel(g.net)
            parameters = f.parameters()
            label = torch.from_numpy(y).to(g.device)
            real = torch.from_numpy(x).to(g.device)
            condition = (label == 1).to(real.dtype)
            raw = f(parameters, condition)
            repeated = f(parameters, condition)
            require(raw.shape == real.shape and raw.dtype == real.dtype and raw.device == real.device
                and bool(torch.isfinite(raw).all()) and bool(torch.isfinite(repeated).all())
                and torch.allclose(raw, repeated, atol=diagnostic.REPEAT_ATOL, rtol=0), 'Eval repeat mismatch')
            statistics, native = diagnostic.plane_statistics(real, label, raw, shape, scale)
            return raw.detach().cpu().numpy().copy(), statistics, native, active
    finally:
        require(diagnostic._probe_backend_snapshot() == backend, 'Backend not restored')
        require(frozen_state(g) == before, 'G/BN/alpha changed during preview')


def compare_anchor(actual, expected):
    """Compare historical scalar summaries, not unavailable historical float output."""
    differences = {}
    for group in ('valid', 'foreground', 'background'):
        require((actual.get(group) is None) == (expected.get(group) is None), 'Anchor region mismatch')
        if actual[group] is None: continue
        for key in ('normalized_l1', 'optimal_constant_l1', 'valid_median_constant_l1', 'generated_spatial_std'):
            a, b = actual[group][key], expected[group][key]
            delta = abs(a-b)
            differences[group+'.'+key] = delta
            if not (math.isfinite(a) and math.isfinite(b) and delta <= diagnostic.REPEAT_ATOL):
                error = ValueError('Anchor scalar mismatch: '+group+'.'+key+' actual='+str(a)+
                    ' expected='+str(b)+' delta='+str(delta)+' atol=1e-7 rtol=0')
                error.anchor_comparison = {'region': group, 'statistic': key, 'actual': a,
                    'expected': b, 'absolute_difference': delta, 'atol': diagnostic.REPEAT_ATOL, 'rtol': 0}
                raise error
    return differences


def condition_groups(records, arrays, outputs):
    groups = {}
    for i, record in enumerate(records):
        condition = record['arrays']['condition']
        groups.setdefault((tuple(condition['shape']), condition['sha256']), []).append(i)
    report = []
    for (shape, digest), indices in groups.items():
        if len(indices) < 2: continue
        first = indices[0]; comparisons = []
        for other in indices[1:]:
            a, b = outputs[first], outputs[other]
            require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
                and np.allclose(a, b, atol=diagnostic.REPEAT_ATOL, rtol=0),
                'Identical conditions produced inconsistent raw outputs')
            x, y = arrays[first][:2]; xx, yy = arrays[other][:2]
            common = (y >= 0) & (yy >= 0)
            comparisons.append({'indices': [first, other], 'raw_output_max_abs_diff': float(np.max(np.abs(a-b))),
                'common_valid_pixels': int(common.sum()),
                'target_common_valid_l1': float(np.abs(x[common]-xx[common]).mean()) if common.any() else None,
                'target_full_padded_equal': bool(np.array_equal(x, xx))})
        report.append({'condition_shape': list(shape), 'condition_sha256': digest,
            'indices': indices, 'comparisons': comparisons})
    return {'applicable': bool(report), 'groups': report,
        'interpretation': 'Same condition cannot identify patient/slice-specific different targets; raw outputs compared before crop/ignore. Same-condition same-output is not evidence of mode collapse.'
            if report else 'Not applicable: no repeated same-shape condition in fixed coverage.'}


def run_preview_diagnostic(target, run, checkpoint_sha256, output, *, gpu):
    run = Path(run).resolve()
    output = check_output(output, list(target.protected)+[run])
    target.require_no_unpack_writes()
    receipt, ready, identities, provenance = audit_diagnostic(target, run, checkpoint_sha256)
    inventory = {p: target.inventory(p) for p in target.train}
    p = receipt['protocol']; scale = p['scaling']
    cohort = Cohort(target, 20261018)
    manifest, arrays = coverage(cohort, scale, p['selection'], ready)
    def final_audit():
        assert_identities(identities)
        audit(target, inventory)
        require(implementation_identity() == provenance['reader_files_sha256'], 'Reader implementation changed')
        require(source_identity() == p['source'], 'Pinned source changed')
    require(inventory == p.get('input_inventory'), 'Generation input inventory mismatch')
    final_audit()
    resources = gpu_guard(gpu)  # guard CPU payload headroom before deserialization too
    payload = load_diagnostic(run, checkpoint_sha256, receipt, ready)
    final_audit()
    resources = gpu_guard(gpu, expected_uuid=resources['gpu_uuid'])
    import torch
    require(not torch.cuda.is_initialized() and torch.cuda.is_available(), 'Fresh CUDA binding required')
    torch.cuda.set_per_process_memory_fraction(resources['allocator_fraction'], 0)
    g = Generator(**payload['model_config']['G'], device='cuda:0')
    preview.restore_generator(g, payload)
    del payload
    final_audit()
    output.mkdir(parents=True, exist_ok=False)
    try:
        write_json(output/'manifest.json', manifest)
        outputs, records, anchor_comparison = [], [], None
        initial_state = frozen_state(g)
        for record, (x, y, real, label) in zip(manifest['planes'], arrays):
            final_audit()
            gpu_guard(gpu, expected_uuid=resources['gpu_uuid'])
            raw, stats, native, active = evaluate(g, x, y, record['native_shape'], scale)
            if record['anchor']:
                require(active == receipt['observations']['after']['probe_backend']['active'],
                    'Anchor probe backend differs from historical eval')
                anchor_comparison = compare_anchor(stats, receipt['observations']['after']['statistics'])
            name = '%s_z%04d_%s.png' % (record['patient'], record['z'], record['category'])
            display = preview.render(output/name, real, label, native, scale)
            records.append(dict(record, file=name, sha256=sha256(output/name), statistics=stats,
                missing_metric_reasons={k: 'no pixels in this region' for k, v in stats.items() if v is None}, display=display))
            outputs.append(raw)
        ambiguity = condition_groups(manifest['planes'], arrays, outputs)
        require(frozen_state(g) == initial_state, 'G state changed across coverage')
        final_audit()
        hashes = {path.name: sha256(path) for path in output.iterdir() if path.is_file()}
        write_json(output/'preview_diagnostic.json', {'format': 1, 'classification': CLASSIFICATION,
            'complete': True, 'diagnostic_only': True, 'run': str(run), 'checkpoint_sha256': checkpoint_sha256,
            'generation_protocol': p, 'provenance': provenance, 'initial_resources': resources,
            'manifest': manifest, 'previews': records, 'condition_groups': ambiguity,
            'anchor_scalar_differences': anchor_comparison, 'comparison_atol': diagnostic.REPEAT_ATOL,
            'historical_pixelwise_reproduction': False, 'G_BN_alpha_unchanged': True,
            'D_S_optimizer_constructed': False, 'files_sha256': hashes,
            'interpretation': 'Single-slice memorization model on fixed conditions; no multi-slice training, generalization or augmentation benefit claim; original quality gate unchanged'})
        # Separate inventory includes the receipt; an inventory cannot hash itself.
        write_json(output/'files_sha256.json', {path.name: sha256(path) for path in output.iterdir() if path.is_file()})
    except BaseException as exc:
        # Never leave a success marker if final inventory writing failed.
        (output/'preview_diagnostic.json').unlink(missing_ok=True)
        write_json(output/'failed.json', {'format': 1, 'complete': False, 'classification': CLASSIFICATION,
            'error': type(exc).__name__+': '+str(exc), 'provenance': provenance,
            'anchor_comparison': getattr(exc, 'anchor_comparison', None),
            'original_run_unchanged': True})
        raise
    return output
