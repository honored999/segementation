"""Future CPU contract tests. LOCAL STATIC ONLY: not executed by the implementer."""
from copy import deepcopy
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from experiments.genseg30.contract import REPO, sha256, write_json, read_json
from experiments.genseg_joint import diagnostic as diag, runtime
from experiments.genseg_joint.core import PretrainEngine
from experiments.genseg_joint.data import Cohort, inner_split
from experiments.genseg_joint.pretrain import policy, save_pretrain


class TinyGenerator:
    """Only hardware/model size is substituted; production engine/loader still execute."""
    def __init__(self, *, seed=diag.SEED, device='cpu'):
        self.config = {'width': 64, 'downs': 8, 'seed': seed}
        self.device = torch.device('cpu')
        self.net = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 1), torch.nn.BatchNorm2d(2),
            torch.nn.ConvTranspose2d(2, 1, 1), torch.nn.Tanh())
        self.networks = SimpleNamespace(conv_arch=torch.zeros(1), upconv_arch=torch.zeros(1))
        self.restore_arch({k: getattr(self.networks, k) for k in ('conv_arch', 'upconv_arch')})

    def restore_arch(self, values):
        for key, value in values.items():
            setattr(self.networks, key, value.detach().clone().requires_grad_(False))
        self.networks._conv_arch_parameters = [self.networks.conv_arch]
        self.networks._upconv_arch_parameters = [self.networks.upconv_arch]
        self.networks._arch_parameters = [self.networks.upconv_arch, self.networks.conv_arch]

    def discriminator(self):
        return torch.nn.Sequential(torch.nn.Conv2d(2, 2, 1), torch.nn.BatchNorm2d(2),
            torch.nn.LeakyReLU(), torch.nn.Conv2d(2, 1, 1))


class Target:
    train = [f'case{i:03}' for i in range(76)]
    val = [f'outer{i:03}' for i in range(19)]
    identity = {'source': 'synthetic'}
    manifest_sha256 = 'a'*64
    _identities = {}

    def __init__(self, root):
        self.protected = [REPO, root/'source']
        self.reads = []; self.changed = False

    def require_no_unpack_writes(self): pass
    def training_dataset(self): return SimpleNamespace(identifiers=self.train)
    def inventory(self, patient): return {'synthetic': patient}
    def verify_inventory(self, inventory):
        assert set(inventory) == set(self.train)
        self.verify_identity(self.identity)
    def verify_identity(self, identity):
        if self.changed or identity != self.identity:
            raise ValueError('target input changed')
    def load_train(self, dataset, patient):
        self.reads.append(patient)
        x = np.arange(4*9*11, dtype=np.float32).reshape(1, 4, 9, 11)
        y = np.zeros_like(x, dtype=np.int16); y[:, 0] = -1; y[0, 2, 3:5, 4:6] = 1
        return x, y, {}


def protocol(target):
    split = inner_split(target.train, target.val, diag.SEED)
    return dict(policy(), seed=diag.SEED, steps=500, split=split,
        target_identity=target.identity, target_manifest_sha256=target.manifest_sha256,
        model_config={'G': {'width': 64, 'downs': 8, 'seed': diag.SEED}, 'D_width': 64},
        scaling={'method': 'inner61_valid_pixel_global_minmax_to_minus1_plus1_no_clip',
                 'min': -5., 'max': 500., 'fit_ids': split['train']},
        implementation_files_sha256={'historical.py': 'b'*64})


def metadata_run(tmp_path, target, monkeypatch, *, weights=False):
    run = tmp_path/'GD500'; run.mkdir()
    p = protocol(target)
    if weights:
        g = TinyGenerator(); e = PretrainEngine(g, g.discriminator()); e.steps = 500
        save_pretrain(run, e, p, {})
    else:
        (run/'gan.pt').write_bytes(b'metadata-only fixture')
        write_json(run/'complete.json', dict(format=1, phase='gan_pretrain', classification=p['classification'],
            complete=True, steps=500, checkpoint='gan.pt', checkpoint_sha256=sha256(run/'gan.pt'), protocol=p))
    digest = sha256(run/'gan.pt')
    # Fixture bytes cannot have the real server checkpoint hash. Hash validation remains active.
    monkeypatch.setattr(diag, 'GD500_SHA256', digest)
    return run, digest


def test_cli_requires_explicit_inputs_and_has_no_tuning_knobs():
    from experiments.genseg_joint.cli import parser
    base = ['diagnose-single-slice', '--source', 's', '--raw-root', 'r', '--targets', 't',
            '--baseline-run', 'b', '--output', 'o', '--pretrained-run', 'p',
            '--pretrained-sha256', diag.GD500_SHA256, '--gpu', '0', '--steps', '3']
    args = parser().parse_args(base)
    assert args.steps == 3 and args.seed == diag.SEED
    for flag in ('--g-lr', '--d-lr', '--l1-weight', '--arch-lr', '--pretrain-steps', '--min-free-vram-gib'):
        with pytest.raises(SystemExit): parser().parse_args(base+[flag, '1'])
    for flag, value in (('--steps', '0'), ('--steps', '-1'), ('--seed', '1'), ('--pretrained-sha256', 'bad')):
        changed = list(base); changed[changed.index(flag)+1] = value
        with pytest.raises(SystemExit): parser().parse_args(changed)
    for flag in ('--pretrained-run', '--pretrained-sha256', '--gpu', '--steps'):
        changed = list(base); i = changed.index(flag); del changed[i:i+2]
        with pytest.raises(SystemExit): parser().parse_args(changed)


@pytest.mark.parametrize('damage', ['rates', 'l1_weight', 'steps', 'split', 'source', 'scaling', 'hash', 'complete'])
def test_initialization_rejects_bad_or_custom_handoff_before_arrays(tmp_path, monkeypatch, damage):
    target = Target(tmp_path); run, digest = metadata_run(tmp_path, target, monkeypatch)
    r = read_json(run/'complete.json')
    if damage == 'rates': r['protocol']['rates']['g'] = 1e-5
    elif damage == 'l1_weight': r['protocol']['l1_weight'] = 1.
    elif damage == 'steps': r['steps'] = r['protocol']['steps'] = 499
    elif damage == 'split': r['protocol']['split']['train'][0] = target.val[0]
    elif damage == 'source': r['protocol']['source'] = {}
    elif damage == 'scaling': r['protocol']['scaling']['fit_ids'] = []
    elif damage == 'complete': r['complete'] = False
    else: (run/'gan.pt').write_bytes(b'changed')
    (run/'complete.json').write_text(json.dumps(r), encoding='utf-8')
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('metadata audit deserialized weights'))
    with pytest.raises(ValueError): diag.initialization(target, run, digest)
    assert not target.reads


def test_fixed_plane_role_gate_first_lesion_and_exact_padding(tmp_path, monkeypatch):
    target = Target(tmp_path); cohort = Cohort(target, diag.SEED); scale = protocol(target)['scaling']
    monkeypatch.setattr(Cohort, 'sample', lambda *a: pytest.fail('sampled'))
    monkeypatch.setattr(Cohort, 'fit_scale', lambda *a: pytest.fail('refit'))
    x, y, native, label, record = diag.fixed_plane(cohort, scale)
    assert target.reads == [sorted(cohort.train)[0]] and record['z'] == 2
    assert x.shape == y.shape == (1, 1, 256, 256)
    np.testing.assert_array_equal(x[0, 0, :9, :11], (2*(native+5)/505-1).astype(np.float32))
    assert (x[0, 0, 9:] == 0).all() and (y[0, 0, 9:] == -1).all()
    target.reads.clear(); cohort.train = [cohort.val[0]]
    with pytest.raises(ValueError, match='Forbidden'): diag.fixed_plane(cohort, scale)
    assert target.reads == []
    cohort = Cohort(target, diag.SEED)
    original = target.load_train
    def no_lesion(*args):
        a, b, c = original(*args); b[b == 1] = 0; return a, b, c
    monkeypatch.setattr(target, 'load_train', no_lesion)
    with pytest.raises(ValueError, match='no valid lesion'): diag.fixed_plane(cohort, scale)
    assert target.reads == [sorted(cohort.train)[0]]  # no search for an easier patient


def state(engine):
    return [{k: v.clone() for k, v in n.state_dict().items()} for n in (engine.g.net, engine.d)]


def test_probe_exact_state_rng_grad_hooks_and_next_step_unchanged(monkeypatch):
    torch.manual_seed(7)
    g = TinyGenerator(); engine = PretrainEngine(g, g.discriminator())
    control = deepcopy(engine)
    real = torch.linspace(-1, 1, 72).reshape(1, 1, 8, 9)
    label = torch.zeros_like(real); label[:, :, 2:4, 3:6] = 1; label[:, :, 0] = -1
    for net in (g.net, engine.d):
        for p in net.parameters(): p.grad = torch.ones_like(p)
    before = state(engine); gradients = [p.grad.clone() for n in (g.net, engine.d) for p in n.parameters()]
    rng = torch.get_rng_state().clone(); py_rng = random.getstate(); np_rng = np.random.get_state()
    observation, generated = diag.probe(engine, real, label, (8, 9), {'min': 0., 'max': 10.})
    assert generated.shape == (8, 9) and (generated[0] == 0).all()
    assert observation['statistics']['valid']['target_normalized']['count'] == 63
    assert observation['statistics']['foreground']['target_normalized']['count'] == 6
    assert set(observation['parameter_gradient_norms']['G']) == dict(g.net.named_parameters()).keys()
    assert observation['activations'] and observation['purity']['RNG'] == 'exact_unchanged'
    assert torch.equal(rng, torch.get_rng_state()) and py_rng == random.getstate()
    after_np = np.random.get_state()
    assert np_rng[0] == after_np[0] and np.array_equal(np_rng[1], after_np[1]) and np_rng[2:] == after_np[2:]
    for net, saved in zip((g.net, engine.d), before):
        assert all(torch.equal(v, saved[k]) for k, v in net.state_dict().items())
        assert all(not m._forward_hooks for m in net.modules())
    assert all(torch.equal(p.grad, old) for p, old in zip((p for n in (g.net, engine.d) for p in n.parameters()), gradients))
    assert engine.steps == 0
    engine.step(real, label); control.step(real, label)
    assert all(all(torch.equal(v, other[k]) for k, v in n.state_dict().items())
               for n, other in zip((g.net, engine.d), state(control)))
    def fail(*a, **k): raise ValueError('hook observation failure')
    monkeypatch.setattr(diag, 'tensor_statistics', fail)
    before = state(engine)
    with pytest.raises(ValueError, match='hook observation'): diag.probe(engine, real, label, (8, 9), {'min': 0., 'max': 10.})
    assert all(not m._forward_hooks for n in (g.net, engine.d) for m in n.modules())
    assert all(all(torch.equal(v, old[k]) for k, v in n.state_dict().items()) for n, old in zip((g.net, engine.d), before))


def cpu_transport(monkeypatch):
    guards = []; waits = []; order = []
    def guard(index, **kwargs):
        assert index == 0 and kwargs.get('expected_uuid', 'CPU-fixture') == 'CPU-fixture'
        guards.append(kwargs); order.append('guard')
        return {'gpu_uuid': 'CPU-fixture', 'allocator_fraction': .5}
    monkeypatch.setattr(diag, 'gpu_guard', guard); monkeypatch.setattr(runtime, 'gpu_guard', guard)
    monkeypatch.setattr(runtime.time, 'sleep', waits.append)
    monkeypatch.setattr(diag, 'Generator', TinyGenerator)
    for name, result in (('is_initialized', False), ('is_available', True), ('max_memory_allocated', 0)):
        monkeypatch.setattr(torch.cuda, name, lambda v=result: v)
    for name in ('set_per_process_memory_fraction', 'synchronize', 'empty_cache', 'manual_seed_all'):
        monkeypatch.setattr(torch.cuda, name, lambda *a, event=name: order.append(event)
                            if event in ('synchronize', 'empty_cache') else None)
    original = torch.Tensor.to
    def to_cpu(self, *args, **kwargs):
        if args and args[0] == 'cuda:0': args = ('cpu',)+args[1:]
        return original(self, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'to', to_cpu)
    monkeypatch.setattr(Cohort, 'fit_scale', lambda *a: pytest.fail('refit'))
    monkeypatch.setattr(Cohort, 'sample', lambda *a: pytest.fail('sample'))
    return guards, waits, order


@pytest.mark.parametrize('tamper', [None, 'input', 'implementation', 'png', 'pretrain', 'receipt_write'])
def test_run_fixed_updates_final_audits_and_no_checkpoint(tmp_path, monkeypatch, tamper):
    target = Target(tmp_path); run, digest = metadata_run(tmp_path, target, monkeypatch, weights=True)
    guards, waits, order = cpu_transport(monkeypatch)
    original_audit, original_probe, original_step = diag.audit, diag.probe, PretrainEngine.step
    def audit(*args):
        result = original_audit(*args); order.append('audit'); return result
    def probe(*args):
        order.append('probe'); return original_probe(*args)
    def step(self, *args):
        order.append('step'); return original_step(self, *args)
    monkeypatch.setattr(diag, 'audit', audit)
    monkeypatch.setattr(diag, 'probe', probe)
    monkeypatch.setattr(PretrainEngine, 'step', step)
    output = tmp_path/'diagnostic'
    original_render = diag.render; original_identity = diag.implementation_identity
    changed = {'implementation': False}; renders = []
    monkeypatch.setattr(diag, 'implementation_identity', lambda: {'diagnostic.py': 'changed'} if changed['implementation'] else original_identity())
    def render(*args):
        result = original_render(*args); renders.append(args[0])
        if len(renders) == 2:
            if tamper == 'input': target.changed = True
            elif tamper == 'implementation': changed['implementation'] = True
            elif tamper == 'png': renders[0].write_bytes(renders[0].read_bytes()+b'changed')
            elif tamper == 'pretrain': (run/'complete.json').write_bytes((run/'complete.json').read_bytes()+b'\n')
        return result
    monkeypatch.setattr(diag, 'render', render)
    original_write = diag.write_json
    def write(path, data):
        original_write(path, data)
        if tamper == 'receipt_write' and path.name == 'diagnostic.json': raise OSError('receipt write failed')
    monkeypatch.setattr(diag, 'write_json', write)
    options = dict(pretrained_run=run, pretrained_sha256=digest, gpu=0, steps=3)
    if tamper:
        with pytest.raises((ValueError, OSError)): diag.run_single_slice(target, output, **options)
        assert read_json(output/'failed.json')['complete'] is False
        assert not (output/'diagnostic.json').exists()
    else:
        diag.run_single_slice(target, output, **options)
        r = read_json(output/'diagnostic.json')
        assert r['complete'] and r['diagnostic_only'] and r['completed_steps'] == 3 and r['weights_saved'] is False
        assert set(r['observations']) == {'before', 'after'} and len(renders) == 2
        assert r['protocol']['rates'] == {'g': 2e-5, 'd': 2e-5} and r['protocol']['l1_weight'] == 100.
        entries = [json.loads(line) for line in (output/'steps.jsonl').read_text().splitlines()]
        assert [e['step'] for e in entries] == [1, 2, 3]
        assert all(e['selection'] == r['protocol']['selection'] for e in entries)
        events = [json.loads(line) for line in (output/'guard_events.jsonl').read_text().splitlines()]
        assert [e['unit'] for e in events] == ['initial', 'before_patient_arrays', 'before_models',
            'update_1', 'update_2', 'update_3', 'after_probe']
        assert len(guards) == 7 and not waits
        first_probe = order.index('probe')
        # Actual audit completes before sync/cache/telemetry; no audit or second
        # guard separates the before probe from its first production update.
        assert first_probe >= 4 and order[first_probe-4:] == [
            'audit', 'synchronize', 'empty_cache', 'guard', 'probe', 'step', 'synchronize',
            'audit', 'synchronize', 'empty_cache', 'guard', 'step', 'synchronize',
            'audit', 'synchronize', 'empty_cache', 'guard', 'step', 'synchronize',
            'audit', 'synchronize', 'empty_cache', 'guard', 'probe', 'audit']
        assert sha256(run/'gan.pt') == digest
    assert target.reads == [sorted(inner_split(target.train, target.val, diag.SEED)['train'])[0]]
    assert not list(output.glob('*.pt')) and not (output/'complete.json').exists()


def test_initial_resource_refusal_before_arrays_or_models_and_protected_output(tmp_path, monkeypatch):
    target = Target(tmp_path); run, digest = metadata_run(tmp_path, target, monkeypatch)
    def refuse(*a, **k): raise ValueError('Resource guard refused')
    monkeypatch.setattr(diag, 'gpu_guard', refuse)
    monkeypatch.setattr(diag, 'Generator', lambda **k: pytest.fail('built model'))
    options = dict(pretrained_run=run, pretrained_sha256=digest, gpu=0, steps=3)
    for protected in (run, run/'nested', tmp_path, target.protected[1]/'nested'):
        with pytest.raises(ValueError, match='outside'): diag.run_single_slice(target, protected, **options)
    output = tmp_path/'refused'
    with pytest.raises(ValueError, match='Resource guard refused'): diag.run_single_slice(target, output, **options)
    assert read_json(output/'failed.json')['completed_steps'] == 0 and not target.reads
    assert not (output/'diagnostic.json').exists()
    with pytest.raises(ValueError, match='Fresh'): diag.run_single_slice(target, output, **options)
