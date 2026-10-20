"""Bounded single-slice convergence plan: fixed points, private trajectory, one state snapshot.

Diagnostic only: the snapshot is never a pretraining checkpoint, a joint handoff
or a resume state, and intermediate observations are never the final result.
"""
from pathlib import Path
from experiments.genseg30.contract import require, sha256

CONVERGENCE_STEPS = 500
CONVERGENCE_POINTS = (100, 250, 500)
SNAPSHOT_FILE = 'diagnostic_state.pt'
SNAPSHOT_FORMAT = 1
TRAJECTORY_FILE = 'trajectory.json'
CLASSIFICATION = 'exploratory_bounded_convergence_single_slice_diagnostic_not_formal'
MODES = ('eval', 'batchstats', 'calibrated_eval')

CONVERGENCE_POLICY = {
    'scope': 'one fixed inner61 training slice, one cold start, one fixed upper bound; 500 updates is a candidate bound, not paper-specified and not authorized as a formal budget',
    'schedule': 'observations before the first update and at every planned completed step; the final planned step is the single after observation, with no duplicate forward and no observation after it',
    'budget': 'exactly the declared steps; no early stop for image quality, no automatic rerun, no seed switch and no expansion to 1000/5000',
    'modes': {'before': 'eval and batchstats only, identical to the historical before observation',
              'points': 'eval, batchstats and one private calibrated eval; the compared modes are localization only'},
    'state': 'each point is a read-only probe; live parameters, BN buffers/counters, Adam, RNG, hooks and backend flags must be unchanged around it',
    'authority': 'the quality assessment still uses only before and the final original eval; intermediate points and compared modes never replace the gate or select a best step',
    'trajectory': 'per point and mode: valid/foreground/background normalized L1, optimal-constant and valid-median-constant baselines, native range and generated std; training loss is never mixed with eval',
    'snapshot': 'one diagnostic_state.pt written only after the whole fixed budget and the final state audit; G/D state_dict incl. real BN, alpha, model config, source/code identity, seed, actual steps and input hashes; CPU detached tensors and basic types only; never named gan.pt/joint.pt',
    'reuse': 'not_pretraining, not_joint_handoff, no_resume; no optimizer moments, no RNG state, no training complete.json and no path into formal training',
    'weights_saved_semantics': 'receipt weights_saved True means exactly one diagnostic_state.pt exists; protocol weights_saved False means no reusable training checkpoint',
    'failure': 'the run stops at the first state/numeric/resource failure; completed steps and the failure are recorded and a partially written snapshot is referenced as partial, never as a success marker',
    'interpretation': '100 to 250 to 500 same-mode errors describe this one slice and budget only; not capacity, generalization or clinical evidence'}


def stage_name(step, steps):
    """Stable stage/file stem: the final planned step is the single after."""
    require(type(step) is int and type(steps) is int and 0 < step <= steps, 'Invalid convergence step')
    return 'after' if step == steps else 'step%06d' % step


def preview_filename(stage, mode):
    """'before'/'after' keep the historical names; planned points carry both step and mode."""
    require(type(stage) is str and stage and all(c.isalnum() or c in '_-' for c in stage),
            'Unsafe convergence stage name')
    require(mode in MODES, 'Unknown convergence mode')
    if mode == 'eval' and stage in ('before', 'after'):
        return stage+'.png'
    return stage+'_'+mode+'.png'


def convergence_plan(steps, schedule=None):
    """Fixed observation plan computed before any data read or output creation."""
    require(type(steps) is int and 0 < steps <= CONVERGENCE_STEPS,
            'Bounded convergence budget must be 1..500 updates')
    if schedule is None:
        require(steps == CONVERGENCE_STEPS, 'Bounded convergence requires exactly 500 updates')
        points = list(CONVERGENCE_POINTS)
    else:
        require(isinstance(schedule, (tuple, list)) and bool(schedule) and
                all(type(point) is int and point > 0 for point in schedule),
                'Convergence schedule must be explicit positive integer points')
        points = list(schedule)
        require(points == sorted(set(points)), 'Convergence schedule must be sorted and unique')
        require(points[-1] == steps, 'Final convergence point must equal the actual update budget')
    return {'format': 1, 'steps': steps, 'points': points, 'before': True,
            'final_point': points[-1], 'final_point_is_after': True,
            'observation_stages': ['before']+[stage_name(step, steps) for step in points],
            'intermediate_points': points[:-1],
            'modes': {'before': ['eval', 'batchstats'], 'points': list(MODES)},
            'policy_version': 1}


def trajectory(plan, records):
    """Independent per-point, per-mode trajectory; not a completion receipt."""
    points = []
    for index, stage in enumerate(plan['observation_stages']):
        record = records.get(stage)
        require(record is not None, 'Missing scheduled convergence observation: '+stage)
        modes = {}
        candidates = (('eval', record.get('statistics')),
                      ('batchstats', record.get('bn_mode_comparison', {}).get('batchstats_statistics')),
                      ('calibrated_eval', record.get('bn_calibration', {}).get('calibrated_eval_statistics')))
        for mode, statistics in candidates:
            if statistics is None:
                continue
            modes[mode] = {}
            for group, values in statistics.items():
                if values is None:
                    continue
                row = {key: values[key] for key in ('normalized_l1', 'optimal_constant_l1',
                       'valid_median_constant_l1', 'generated_spatial_std')}
                row['target_native_range'] = values['target_native']['range']
                row['generated_native_range'] = values['generated_native']['range']
                modes[mode][group] = row
        points.append({'step': 0 if stage == 'before' else plan['points'][index-1],
                       'stage': stage, 'scheduled': True, 'final_after': stage == 'after',
                       'modes': modes, 'png': record['png'],
                       'state': {'eval_repeat_within_tolerance': record['eval_repeat_within_tolerance'],
                                 'eval_repeat_max_abs_diff': record['eval_repeat_metrics']['max_abs_diff'],
                                 'bn_counts': record['bn_counts'], 'adam_steps': record['adam_steps'],
                                 'probe_backend_restored': record['probe_backend']['restored']}})
    return {'format': 1, 'classification': CLASSIFICATION, 'diagnostic_only': True,
            'complete': False, 'intermediate_points_are_final_result': False,
            'scheduled_points_complete': len(points) == len(plan['observation_stages']),
            'authority': 'diagnostic.json remains the only completion receipt',
            'policy': dict(CONVERGENCE_POLICY), 'plan': plan, 'points': points}


def save_diagnostic_snapshot(output, engine, protocol, *, ready, source, seed):
    """One diagnostic-only final snapshot; CPU detached copies, safe readback, no live model."""
    import torch
    models = {'G': {name: value.detach().to('cpu').clone() for name, value in engine.g.net.state_dict().items()},
              'D': {name: value.detach().to('cpu').clone() for name, value in engine.d.state_dict().items()},
              'alpha': {key: getattr(engine.g.networks, key).detach().to('cpu').clone()
                        for key in ('conv_arch', 'upconv_arch')}}
    payload = {'format': SNAPSHOT_FORMAT, 'classification': CLASSIFICATION, 'diagnostic_only': True,
               'not_pretraining': True, 'not_joint_handoff': True, 'no_resume': True,
               'training_receipt': False, 'checkpoint_name': 'not_gan_not_joint',
               'steps': int(engine.steps), 'seed': int(seed), 'models': models,
               'model_config': {'G': dict(engine.g.config), 'D_width': 64},
               'source': dict(source), 'protocol': protocol,
               'input': {'ready_file': ready['file'], 'ready_sha256': ready['sha256'],
                         'selection': protocol.get('selection'), 'scaling': protocol.get('scaling'),
                         'condition': 'label==1', 'metric_region': 'label>=0'}}
    path = Path(output)/SNAPSHOT_FILE
    require(not path.exists(), 'Fresh diagnostic state snapshot required')
    created = False
    try:
        with path.open('xb') as stream:
            created = True
            torch.save(payload, stream)
        readback = torch.load(path, map_location='cpu', weights_only=True)
        _same(payload, readback)
        require(readback['steps'] == engine.steps and readback['seed'] == seed,
                'Snapshot readback lost the actual steps/seed')
        info = {'format': SNAPSHOT_FORMAT, 'file': SNAPSHOT_FILE, 'sha256': sha256(path),
                'bytes': path.stat().st_size, 'steps': int(engine.steps),
                'tensor_count': sum(len(values) for values in models.values()),
                'weights_only_load': True, 'readback_exact': True,
                'not_pretraining': True, 'not_joint_handoff': True, 'no_resume': True,
                'optimizer_moments': False, 'rng_state': False}
    except BaseException as exc:
        if created:
            partial = {'file': SNAPSHOT_FILE, 'validated': False, 'partial': True,
                       'steps': int(engine.steps), 'bytes': None, 'sha256': None}
            for key, obtain in (('bytes', lambda: path.stat().st_size),
                                ('sha256', lambda: sha256(path))):
                try:
                    partial[key] = obtain()
                except BaseException as metadata_error:
                    partial[key+'_error'] = type(metadata_error).__name__+': '+str(metadata_error)
            exc.snapshot_partial = partial
        raise
    return info


def _same(left, right, path='payload'):
    """Exact structural/tensor comparison of the just-written snapshot."""
    import torch
    require(type(left) is type(right), 'Snapshot readback type mismatch at '+path)
    if isinstance(left, dict):
        require(left.keys() == right.keys(), 'Snapshot readback key mismatch at '+path)
        for key in left:
            _same(left[key], right[key], path+'.'+str(key))
    elif isinstance(left, (list, tuple)):
        require(len(left) == len(right), 'Snapshot readback length mismatch at '+path)
        for index, (first, second) in enumerate(zip(left, right)):
            _same(first, second, path+'['+str(index)+']')
    elif isinstance(left, torch.Tensor):
        require(left.dtype == right.dtype and left.device == right.device and
                bool(torch.equal(left, right)), 'Snapshot readback tensor mismatch at '+path)
    else:
        require(left == right, 'Snapshot readback value mismatch at '+path)
