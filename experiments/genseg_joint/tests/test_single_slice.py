"""Synthetic CPU diagnostic contracts; no patient/CUDA execution."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import inspect
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


class CountingGenerator(TinyGenerator):
    """Same tiny fixture, plus G/D forward counts: previews must add none."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.forwards = {'g': 0, 'd': 0}
        self.net.register_forward_pre_hook(self._count('g'))

    def _count(self, key):
        def hook(module, inputs): self.forwards[key] += 1
        return hook

    def discriminator(self):
        net = super().discriminator()
        net.register_forward_pre_hook(self._count('d'))
        return net


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
    assert args.compare_bn_modes is False and args.compare_bn_calibration is False
    assert args.g_objective == 'gan_l1'
    # l1_only is explicit, default-off, and only valid for the repair cold diagnostic.
    cold = ['diagnose-single-slice', '--source', 's', '--raw-root', 'r', '--targets', 't',
            '--baseline-run', 'b', '--output', 'o', '--gpu', '0', '--steps', '3',
            '--strategy', 'adam_trainbn_epoch_v1', '--initialization', 'cold']
    assert parser().parse_args(cold).g_objective == 'gan_l1'
    assert parser().parse_args(cold+['--g-objective', 'l1_only']).g_objective == 'l1_only'
    with pytest.raises(SystemExit): parser().parse_args(cold+['--g-objective', 'bogus'])
    for rejected in (base, base+['--strategy', 'adam_trainbn_epoch_v1']):
        with pytest.raises(SystemExit): parser().parse_args(rejected+['--g-objective', 'l1_only'])
    assert parser().parse_args(base+['--compare-bn-calibration']).compare_bn_calibration is True
    assert parser().parse_args(base+['--compare-bn-modes']).compare_bn_modes is True
    # Mode previews are default-off and require an explicit BN comparison switch.
    assert parser().parse_args(base).save_mode_previews is False
    assert parser().parse_args(base+['--compare-bn-modes','--save-mode-previews']).save_mode_previews is True
    assert parser().parse_args(base+['--compare-bn-calibration','--save-mode-previews']).save_mode_previews is True
    with pytest.raises(SystemExit): parser().parse_args(base+['--save-mode-previews'])
    preview = ['preview', '--source', 's', '--raw-root', 'r', '--targets', 't', '--baseline-run',
               'b', '--output', 'o', '--run', 'x', '--checkpoint-sha256', 'a'*64, '--gpu', '0']
    with pytest.raises(SystemExit): parser().parse_args(preview+['--save-mode-previews'])
    for flag in ('--g-lr', '--d-lr', '--l1-weight', '--arch-lr', '--pretrain-steps', '--min-free-vram-gib'):
        with pytest.raises(SystemExit): parser().parse_args(base+[flag, '1'])
    for flag, value in (('--steps', '0'), ('--steps', '-1'), ('--seed', '1'), ('--pretrained-sha256', 'bad')):
        changed = list(base)
        if flag in changed: changed[changed.index(flag)+1] = value
        else: changed += [flag,value]
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


@pytest.mark.parametrize('strategy',['legacy','adam_trainbn_epoch_v1'])
def test_probe_exact_state_rng_grad_hooks_and_next_step_unchanged(monkeypatch,strategy):
    # Test opposite caller flags, including both enabled/TF32 values across strategies.
    for key,value in dict(enabled=strategy=='legacy',benchmark=True,benchmark_limit=7,
                          deterministic=False,allow_tf32=strategy!='legacy').items():
        monkeypatch.setattr(torch.backends.cudnn,key,value)
    torch.manual_seed(7)
    g = TinyGenerator(); engine = PretrainEngine(g, g.discriminator(),strategy=strategy)
    control = deepcopy(engine)
    real = torch.linspace(-1, 1, 72).reshape(1, 1, 8, 9)
    label = torch.zeros_like(real); label[:, :, 2:4, 3:6] = 1; label[:, :, 0] = -1
    for net in (g.net, engine.d):
        for p in net.parameters(): p.grad = torch.ones_like(p)
    before = state(engine); gradients = [p.grad.clone() for n in (g.net, engine.d) for p in n.parameters()]
    rng = torch.get_rng_state().clone(); py_rng = random.getstate(); np_rng = np.random.get_state()
    caller_backend={name:getattr(torch.backends.cudnn,name) for name in
        ('enabled','benchmark','benchmark_limit','deterministic','allow_tf32')}
    caller_global=(torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled())
    forward_grad_contexts=[]
    def check_probe_backend(module,inputs):
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        for key in ('enabled','allow_tf32','benchmark_limit'):
            assert getattr(torch.backends.cudnn,key)==caller_backend[key]
        assert (torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled())==caller_global
        forward_grad_contexts.append(torch.is_grad_enabled())
    handle=g.net.register_forward_pre_hook(check_probe_backend)
    try:
        observation, generated = diag.probe(engine, real, label, (8, 9), {'min': 0., 'max': 10.})
    finally:
        handle.remove()
    assert {name:getattr(torch.backends.cudnn,name) for name in caller_backend}==caller_backend
    assert forward_grad_contexts==[True,True], 'Repeated eval forwards must use the same enabled grad context'
    assert generated.shape == (8, 9) and (generated[0] == 0).all()
    assert observation['statistics']['valid']['target_normalized']['count'] == 63
    assert observation['statistics']['foreground']['target_normalized']['count'] == 6
    valid=real[label>=0]; median=valid.median()
    assert observation['statistics']['valid']['optimal_constant_l1']==float((valid-median).abs().mean())
    assert observation['eval_repeat_exact'] and observation['eval_repeat_within_tolerance']
    assert observation['eval_repeat_metrics']['max_abs_diff']==0.
    assert observation['g_objective']=='gan_l1' and observation['g_adversarial_coefficient']==1.
    assert observation['g_l1_coefficient']==100.
    assert observation['g_optimized_objective'].startswith('adversarial+')
    assert observation['g_loss']==pytest.approx(observation['g_adversarial_loss']+observation['g_l1_loss'],rel=1e-6)
    trace=observation['probe_backend']
    assert trace['restored'] and trace['entry']==trace['exit']==diag._probe_backend_snapshot()
    assert trace['active']==dict(trace['entry'],deterministic=True,benchmark=False)
    # Exercise an older flags signature with no fp32_precision/benchmark_limit,
    # and no cudnn.conv precision object; real local context still restores flags.
    original_flags=torch.backends.cudnn.flags
    @contextmanager
    def old_flags(enabled=False,benchmark=False,deterministic=False,allow_tf32=True):
        options=dict(enabled=enabled,benchmark=benchmark,deterministic=deterministic,allow_tf32=allow_tf32)
        options.update({key:None for key in ('benchmark_limit','fp32_precision')
                        if key in inspect.signature(original_flags).parameters})
        with original_flags(**options):
            yield
    with monkeypatch.context() as compat:
        compat.setattr(torch.backends.cudnn,'flags',old_flags)
        compat.setattr(torch.backends.cudnn,'conv',None,raising=False)
        old_observation,_=diag.probe(engine,real,label,(8,9),{'min':0.,'max':10.})
        old_trace=old_observation['probe_backend']
        assert old_trace['restored'] and old_trace['entry']==old_trace['exit']==diag._probe_backend_snapshot()
        assert 'conv_fp32_precision' not in old_trace['entry']
        assert old_trace['active']==dict(old_trace['entry'],deterministic=True,benchmark=False)
    assert {name:getattr(torch.backends.cudnn,name) for name in caller_backend}==caller_backend
    assert all(m.training==(strategy!='legacy') for n in (engine.g.net,engine.d) for m in n.modules())
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
    def check_training_backend(module,inputs):
        assert {name:getattr(torch.backends.cudnn,name) for name in caller_backend}==caller_backend
    training_hooks=[net.register_forward_pre_hook(check_training_backend) for net in (engine.g.net,control.g.net)]
    try:
        engine.step(real, label); control.step(real, label)
    finally:
        for handle in training_hooks: handle.remove()
    assert all(all(torch.equal(v, other[k]) for k, v in n.state_dict().items())
               for n, other in zip((g.net, engine.d), state(control)))
    # Mixed caller modes and dropout must survive private BN-only observation.
    old_transpose_mode=g.net[2].training
    g.net[2].training=not old_transpose_mode
    g.net.add_module('observation_dropout',torch.nn.Dropout(.5))
    modes=[(m,m.training) for net in (g.net,engine.d) for m in net.modules()]
    opt_state=[(v,v._version,v.clone()) for opt in getattr(engine,'optimizers',{}).values()
               for item in opt.state.values() for v in item.values() if isinstance(v,torch.Tensor)]
    bn_live={name:v for name,v in g.net.named_buffers()}
    weights={name:v.detach().clone().requires_grad_(True) for name,v in g.net.named_parameters()}
    saved=state(engine); saved_rng=torch.get_rng_state().clone(); captures=[]; fail_batch=False
    def inspect_bn_forward(module,inputs,output):
        batch=module[1].training
        assert not module.training and not module.observation_dropout.training
        assert all(not m.training for m in engine.d.modules())
        assert torch.is_grad_enabled() and torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
        assert all(torch.equal(v,weights[name]) and v.requires_grad for name,v in module.named_parameters())
        if batch:
            assert module[1].running_mean is not bn_live['1.running_mean']
            assert int(module[1].num_batches_tracked)==int(bn_live['1.num_batches_tracked'])+1
            torch.rand(3)  # Even a random observational hook must be isolated by fork_rng.
            if fail_batch: raise ValueError('batchstats observation failure')
        captures.append((batch,output.detach().clone()))
    handle=None
    try:
        current_eval,_=diag.probe(engine,real,label,(8,9),{'min':0.,'max':10.})
        handle=g.net.register_forward_hook(inspect_bn_forward)
        compared,_=diag.probe(engine,real,label,(8,9),{'min':0.,'max':10.},compare_bn_modes=True)
        assert [mode for mode,_ in captures]==[False,False,True]
        assert compared['statistics']==current_eval['statistics']
        assert compared['activations']==current_eval['activations']  # B never pollutes eval activations.
        assert 'bn_mode_comparison' not in current_eval
        report=compared['bn_mode_comparison']
        # Independently compute batch normalization over N/H/W (biased variance).
        condition=(label==1).float()
        h=torch.nn.functional.conv2d(condition,weights['0.weight'],weights['0.bias'])
        h=(h-h.mean((0,2,3),keepdim=True))*torch.rsqrt(h.var((0,2,3),unbiased=False,keepdim=True)+g.net[1].eps)
        h=h*weights['1.weight'][None,:,None,None]+weights['1.bias'][None,:,None,None]
        expected=torch.tanh(torch.nn.functional.conv_transpose2d(h,weights['2.weight'],weights['2.bias']))
        torch.testing.assert_close(captures[2][1],expected,atol=1e-7,rtol=1e-6)
        captured_stats,_=diag.plane_statistics(real,label,captures[2][1],(8,9),{'min':0.,'max':10.})
        assert report['batchstats_statistics']==captured_stats
        difference=(captures[0][1].double()-captures[2][1].double()).abs()[label>=0]
        assert report['eval_vs_batchstats']['max_abs_diff']==float(difference.max())
        assert report['eval_vs_batchstats']['mean_abs_diff']==float(difference.mean())
        assert report['policy']['eval_reference'].endswith('quality_gates_eval_only')
        assert diag.assess_quality({'before':observation,'after':compared},1)==diag.assess_quality({'before':observation,'after':current_eval},1)
        assert 'mode_previews' not in compared
        # Opt-in preview arrays are the discarded native planes of these same forwards.
        captured=len(captures); previews={}
        with_previews,_=diag.probe(engine,real,label,(8,9),{'min':0.,'max':10.},
            compare_bn_modes=True,save_mode_previews=True,mode_previews=previews)
        assert len(captures)==captured+3  # exactly one compared probe: no extra forward
        _,batchstats_native=diag.plane_statistics(real,label,captures[2][1],(8,9),{'min':0.,'max':10.})
        assert set(previews)=={'batchstats'} and np.array_equal(previews['batchstats'],batchstats_native)
        assert 'mode_previews' not in with_previews
        assert with_previews['statistics']==current_eval['statistics']
        handle.remove(); handle=None
        old_momentum=g.net[1].momentum
        g.net[1].momentum=None if strategy=='legacy' else .1
        calibration_calls=[]; fail_calibration=None
        def inspect_calibration(module,inputs,output):
            index=len(calibration_calls)
            assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
            assert not module.training and not module.observation_dropout.training
            assert all(not m.training for m in engine.d.modules())
            assert module[1].running_var is not bn_live['1.running_var']
            calibration_calls.append({'train':module[1].training,'grad':torch.is_grad_enabled(),
                'momentum':module[1].momentum,'var':module[1].running_var,
                'mean':module[1].running_mean.detach().clone(),'variance':module[1].running_var.detach().clone(),
                'count':int(module[1].num_batches_tracked),'output':output.detach().clone()})
            if index in (3,4):
                assert module[1].momentum==1.0
                torch.rand(3)  # Calibration and eval share an isolated RNG context.
                if fail_calibration==index: raise RuntimeError('calibration functional forward failure')
        calibration_handle=g.net.register_forward_hook(inspect_calibration)
        try:
            calibrated,_=diag.probe(engine,real,label,(8,9),{'min':0.,'max':10.},compare_bn_calibration=True)
            assert [c['train'] for c in calibration_calls]==[False,False,True,True,False]
            assert [c['grad'] for c in calibration_calls]==[True,True,True,False,True]
            assert calibration_calls[3]['var'] is calibration_calls[4]['var']
            assert calibration_calls[3]['count']==calibration_calls[4]['count']==int(bn_live['1.num_batches_tracked'])+1
            # One-layer fixture: stored running variance must be UNBIASED.
            h=torch.nn.functional.conv2d(condition,weights['0.weight'],weights['0.bias'])
            mean=h.mean((0,2,3)); variance=h.var((0,2,3),unbiased=True)
            torch.testing.assert_close(calibration_calls[3]['mean'],mean,atol=1e-7,rtol=1e-6)
            torch.testing.assert_close(calibration_calls[3]['variance'],variance,atol=1e-7,rtol=1e-6)
            h=(h-mean[None,:,None,None])*torch.rsqrt(variance[None,:,None,None]+g.net[1].eps)
            h=h*weights['1.weight'][None,:,None,None]+weights['1.bias'][None,:,None,None]
            expected=torch.tanh(torch.nn.functional.conv_transpose2d(h,weights['2.weight'],weights['2.bias']))
            torch.testing.assert_close(calibration_calls[4]['output'],expected,atol=1e-7,rtol=1e-6)
            expected_stats,_=diag.plane_statistics(real,label,calibration_calls[4]['output'],(8,9),{'min':0.,'max':10.})
            report=calibrated['bn_calibration']
            assert report['calibrated_eval_statistics']==expected_stats
            assert report['policy']['calibration_count']==1 and report['policy']['momentum']==1.0
            assert report['policy']['stage']=='after_only' and report['bn_layers']==1
            assert calibrated['statistics']==current_eval['statistics'] and calibrated['activations']==current_eval['activations']
            assert diag.assess_quality({'before':observation,'after':calibrated},1)==diag.assess_quality({'before':observation,'after':current_eval},1)
            # The calibrated panel is that same private calibrated eval, not a new forward.
            captured_calls=len(calibration_calls); calibration_previews={}
            with_calibrated,_=diag.probe(engine,real,label,(8,9),{'min':0.,'max':10.},
                compare_bn_calibration=True,save_mode_previews=True,mode_previews=calibration_previews)
            assert len(calibration_calls)==captured_calls+5  # unchanged calibration call count
            _,calibrated_native=diag.plane_statistics(real,label,calibration_calls[-1]['output'],(8,9),
                {'min':0.,'max':10.})
            assert set(calibration_previews)=={'batchstats','calibrated_eval'}
            assert np.array_equal(calibration_previews['calibrated_eval'],calibrated_native)
            assert 'mode_previews' not in with_calibrated
            assert with_calibrated['statistics']==current_eval['statistics']
            for fail_calibration in (3,4):
                calibration_calls.clear()
                with pytest.raises(RuntimeError,match='calibration functional forward failure') as error:
                    diag.probe(engine,real,label,(8,9),{'min':0.,'max':10.},compare_bn_calibration=True)
                assert error.value.probe_backend['restored']
                assert g.net[1].momentum==(None if strategy=='legacy' else .1)
                assert all(m.training==mode for m,mode in modes)
                assert torch.equal(saved_rng,torch.get_rng_state())
                assert all(v._version==version and torch.equal(v,value) for v,version,value in opt_state)
                assert all(all(torch.equal(v,before[k]) for k,v in net.state_dict().items()) for net,before in zip((g.net,engine.d),saved))
        finally:
            calibration_handle.remove()
            g.net[1].momentum=old_momentum
        fail_batch=True
        handle=g.net.register_forward_hook(inspect_bn_forward)
        with pytest.raises(ValueError,match='batchstats observation failure') as error:
            diag.probe(engine,real,label,(8,9),{'min':0.,'max':10.},compare_bn_modes=True)
        assert error.value.probe_backend['restored']
        assert torch.equal(saved_rng,torch.get_rng_state())
        assert all(m.training==mode for m,mode in modes)
        assert all(v._version==version and torch.equal(v,value) for v,version,value in opt_state)
        assert all(all(torch.equal(v,before[k]) for k,v in net.state_dict().items()) for net,before in zip((g.net,engine.d),saved))
    finally:
        if handle is not None: handle.remove()
        delattr(g.net,'observation_dropout')
        g.net[2].training=old_transpose_mode
    # One float32 ULP of repeat-output variation is numerical stability, not
    # evidence that parameters, buffers or image-quality thresholds may drift.
    repeated_outputs=[]
    def tiny_repeat_change(module,inputs,output):
        changed=torch.nextafter(output,torch.full_like(output,float('inf'))) if repeated_outputs else output
        repeated_outputs.append(changed.detach().clone())
        return changed
    handle=g.net.register_forward_hook(tiny_repeat_change)
    try:
        tolerant,_=diag.probe(engine,real,label,(8,9),{'min':0.,'max':10.})
    finally:
        handle.remove()
    measured=float((repeated_outputs[0].double()-repeated_outputs[1].double()).abs().max())
    assert 0<measured<=1e-7
    assert tolerant['eval_repeat_within_tolerance'] and not tolerant['eval_repeat_exact']
    repeat=tolerant['eval_repeat_metrics']
    assert repeat['max_abs_diff']==measured and repeat['atol']==1e-7 and repeat['rtol']==0.
    assert repeat['matched_context'] and repeat['first_context']==repeat['repeated_context']
    assert not repeat['exact'] and repeat['within_tolerance']
    assert diag.REPEAT_OUTPUT_POLICY['atol']==1e-7 and diag.REPEAT_OUTPUT_POLICY['rtol']==0.
    assert diag.REPEAT_OUTPUT_POLICY['reference_float32_epsilon']==torch.finfo(torch.float32).eps
    if strategy!='legacy':
        quality=diag.assess_quality({'before':observation,'after':tolerant},1)
        assert quality['scalar_checks']['state'] and quality['learning_quality_passed'] is False
        # l1_only probe: the optimized G gradient is exactly grad(100*valid_L1);
        # the retained D(fake-for-G) adversarial term is observation only.
        from experiments.genseg_joint.core import FunctionalModel, norm
        l1_engine=PretrainEngine(TinyGenerator(),TinyGenerator().discriminator(),strategy=strategy,g_objective='l1_only')
        condition=(label==1).float(); valid=(label>=0).float()
        modes=[(m,m.training) for m in l1_engine.g.net.modules()]
        for module,_ in modes: module.eval()
        try:
            gf=FunctionalModel(l1_engine.g.net); gp=gf.parameters()
            l1=l1_engine.l1_weight*((gf(gp,condition)-real).abs()*valid).sum()/valid.sum()
            expected={name:norm([grad]) for name,grad in zip(gp,torch.autograd.grad(l1,tuple(gp.values())))}
        finally:
            for module,mode in modes: module.training=mode
        saved=state(l1_engine)
        observed,_=diag.probe(l1_engine,real,label,(8,9),{'min':0.,'max':10.})
        assert observed['g_objective']=='l1_only' and observed['g_adversarial_coefficient']==0.
        assert observed['g_l1_coefficient']==100. and 'adversarial observed' in observed['g_optimized_objective']
        assert observed['g_loss']==float(l1.detach())==observed['g_l1_loss']
        assert observed['g_adversarial_loss']>0 and observed['parameter_gradient_norms']['G']==expected
        assert all(all(torch.equal(v,old[k]) for k,v in net.state_dict().items())
                   for net,old in zip((l1_engine.g.net,l1_engine.d),saved))
    # Even under matched contexts, changed numerical output must fail closed.
    before=state(engine); calls=[]
    def change_repeat(module,inputs,output):
        calls.append(torch.is_grad_enabled())
        return output+.125 if len(calls)==2 else None
    handle=g.net.register_forward_hook(change_repeat)
    try:
        with pytest.raises(ValueError,match='Repeated eval output changed') as error:
            diag.probe(engine,real,label,(8,9),{'min':0.,'max':10.})
    finally:
        handle.remove()
    assert calls==[True,True]
    assert all(field in str(error.value) for field in
        ('max_abs_diff=0.125','dtype=torch.float32/torch.float32','device=cpu/cpu','matched_context=True',
         "'grad_enabled': True","'all_G_modules_eval': True"))
    assert all(all(torch.equal(v,old[k]) for k,v in net.state_dict().items()) for net,old in zip((g.net,engine.d),before))
    assert all(m.training==(strategy!='legacy') for net in (g.net,engine.d) for m in net.modules())
    for damage in ('nan','inf','dtype','shape'):
        calls=[]
        def invalid_repeat(module,inputs,output):
            calls.append(True)
            if len(calls)!=2: return None
            if damage=='nan': return torch.full_like(output,float('nan'))
            if damage=='inf': return torch.full_like(output,float('inf'))
            if damage=='dtype': return output.double()
            return output[:,:,:1,:]  # broadcastable shape must still be refused
        handle=g.net.register_forward_hook(invalid_repeat)
        try:
            with pytest.raises(ValueError,match='Repeated eval output changed'):
                diag.probe(engine,real,label,(8,9),{'min':0.,'max':10.})
        finally:
            handle.remove()
        assert all(all(torch.equal(v,old[k]) for k,v in net.state_dict().items()) for net,old in zip((g.net,engine.d),before))
        assert all(m.training==(strategy!='legacy') for net in (g.net,engine.d) for m in net.modules())
    def fail(*a, **k):
        assert torch.backends.cudnn.deterministic is True and torch.backends.cudnn.benchmark is False
        raise ValueError('hook observation failure')
    monkeypatch.setattr(diag, 'tensor_statistics', fail)
    before = state(engine)
    with pytest.raises(ValueError, match='hook observation') as error:
        diag.probe(engine, real, label, (8, 9), {'min': 0., 'max': 10.})
    trace=error.value.probe_backend
    assert trace['restored'] and trace['entry']==trace['exit']==diag._probe_backend_snapshot()
    assert trace['active']==dict(trace['entry'],deterministic=True,benchmark=False)
    assert all(not m._forward_hooks for n in (g.net, engine.d) for m in n.modules())
    assert all(all(torch.equal(v, old[k]) for k, v in n.state_dict().items()) for n, old in zip((g.net, engine.d), before))

    assert {name:getattr(torch.backends.cudnn,name) for name in caller_backend}==caller_backend
    assert (torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled())==caller_global
    # An exception must also leave the next actual training update unchanged.
    training_hooks=[net.register_forward_pre_hook(check_training_backend) for net in (engine.g.net,control.g.net)]
    try:
        engine.step(real,label); control.step(real,label)
    finally:
        for handle in training_hooks: handle.remove()
    assert all(all(torch.equal(v,other[k]) for k,v in net.state_dict().items())
               for net,other in zip((engine.g.net,engine.d),state(control)))


def cpu_transport(monkeypatch,*,cold=False):
    guards = []; waits = []; order = []
    def guard(index, **kwargs):
        assert index == 0 and kwargs.get('expected_uuid', 'CPU-fixture') == 'CPU-fixture'
        guards.append(kwargs); order.append('guard')
        return {'gpu_uuid': 'CPU-fixture', 'allocator_fraction': .5}
    monkeypatch.setattr(diag, 'gpu_guard', guard); monkeypatch.setattr(runtime, 'gpu_guard', guard)
    monkeypatch.setattr(runtime.time, 'sleep', waits.append)
    monkeypatch.setattr(diag, 'Generator', TinyGenerator)
    for name, result in (('is_initialized', False), ('is_available', True), ('max_memory_allocated', 0), ('is_current_stream_capturing',False)):
        monkeypatch.setattr(torch.cuda, name, lambda v=result: v)
    for name in ('set_per_process_memory_fraction', 'synchronize', 'empty_cache', 'manual_seed_all'):
        monkeypatch.setattr(torch.cuda, name, lambda *a, event=name: order.append(event)
                            if event in ('synchronize', 'empty_cache') else None)
    original = torch.Tensor.to
    def to_cpu(self, *args, **kwargs):
        if args and args[0] == 'cuda:0': args = ('cpu',)+args[1:]
        return original(self, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'to', to_cpu)
    if not cold: monkeypatch.setattr(Cohort, 'fit_scale', lambda *a,**k: pytest.fail('refit'))
    monkeypatch.setattr(Cohort, 'sample', lambda *a: pytest.fail('sample'))
    return guards, waits, order


@pytest.mark.parametrize('strategy,init,g_objective',[('legacy','gd500','gan_l1'),
    ('adam_trainbn_epoch_v1','gd500','gan_l1'),('adam_trainbn_epoch_v1','cold','gan_l1'),
    ('adam_trainbn_epoch_v1','cold','l1_only')])
@pytest.mark.parametrize('tamper', [None, 'input', 'implementation', 'png', 'pretrain', 'receipt_write', 'ready'])
def test_run_fixed_updates_final_audits_and_no_checkpoint(tmp_path, monkeypatch, tamper,strategy,init,g_objective):
    target = Target(tmp_path); run, digest = metadata_run(tmp_path, target, monkeypatch, weights=True)
    guards, waits, order = cpu_transport(monkeypatch,cold=init=="cold")
    built = []
    def build(**kwargs):
        order.append('generator'); generator = CountingGenerator(**kwargs); built.append(generator); return generator
    monkeypatch.setattr(diag, 'Generator', build)
    original_audit, original_probe, original_step = diag.audit, diag.probe, PretrainEngine.step
    def audit(*args):
        result = original_audit(*args); order.append('audit'); return result
    def probe(*args,**kwargs):
        order.append('probe'); return original_probe(*args,**kwargs)
    def step(self, *args):
        order.append('step'); return original_step(self, *args)
    monkeypatch.setattr(diag, 'audit', audit)
    monkeypatch.setattr(diag, 'probe', probe)
    monkeypatch.setattr(PretrainEngine, 'step', step)
    output = tmp_path/'diagnostic'
    original_render = diag.render; original_identity = diag.implementation_identity
    changed = {'implementation': False}; renders = []
    monkeypatch.setattr(diag, 'implementation_identity', lambda: {'diagnostic.py': 'changed'} if changed['implementation'] else original_identity())
    def render(*args,**kwargs):
        result = original_render(*args,**kwargs); renders.append(args[0])
        if len(renders) == 2:
            if tamper == 'input': target.changed = True
            elif tamper == 'implementation': changed['implementation'] = True
            elif tamper == 'png': renders[0].write_bytes(renders[0].read_bytes()+b'changed')
            elif tamper == 'pretrain': (run/'complete.json').write_bytes((run/'complete.json').read_bytes()+b'\n')
            elif tamper == 'ready': (output/'ready.json').write_bytes((output/'ready.json').read_bytes()+b'changed')
        return result
    monkeypatch.setattr(diag, 'render', render)
    original_write = diag.write_json
    def write(path, data):
        original_write(path, data)
        if path.name == 'ready.json': order.append('ready')
        if tamper == 'receipt_write' and path.name == 'diagnostic.json': raise OSError('receipt write failed')
    monkeypatch.setattr(diag, 'write_json', write)
    options = dict(pretrained_run=None if init=='cold' else run,
        pretrained_sha256=None if init=='cold' else digest, gpu=0, steps=3,strategy=strategy,
        initialization_mode=init,g_objective=g_objective)
    compare_bn_modes=init=='cold'
    compare_bn_calibration=compare_bn_modes
    if compare_bn_calibration: options['compare_bn_calibration']=True
    save_mode_previews=compare_bn_calibration
    if save_mode_previews: options['save_mode_previews']=True
    # A cold run has no dependency on historical GD500 files.
    if init=='cold' and tamper=='pretrain': tamper=None
    if tamper:
        with pytest.raises((ValueError, OSError)): diag.run_single_slice(target, output, **options)
        failed = read_json(output/'failed.json')
        assert failed['complete'] is False
        assert not (output/'diagnostic.json').exists()
        # Ready evidence is written before models, kept on failure, and referenced
        # only when it exists; its recorded digest still names the untampered file.
        assert (output/'ready.json').exists() and failed['ready']['file'] == 'ready.json'
        recorded = failed['ready']['sha256']
        assert (recorded != sha256(output/'ready.json')) if tamper == 'ready' else (recorded == sha256(output/'ready.json'))
    else:
        diag.run_single_slice(target, output, **options)
        r = read_json(output/'diagnostic.json')
        assert r['complete'] and r['diagnostic_only'] and r['completed_steps'] == 3 and r['weights_saved'] is False
        assert set(r['observations']) == {'before', 'after'}
        # F1: immutable readiness evidence bound to the actual scale/selection/arrays.
        ready = read_json(output/'ready.json')
        scaling = r['protocol']['scaling']
        assert ready['protocol']['scaling'] == scaling and ready['protocol']['selection'] == r['protocol']['selection']
        assert ready['protocol']['seed'] == diag.SEED and ready['protocol']['steps'] == 3
        assert ready['protocol']['g_objective'] == g_objective and ready['protocol']['rates'] == {'g': 2e-5, 'd': 2e-5}
        assert ready['stage'] == 'ready_after_fixed_plane_before_models_or_updates'
        assert ready['policy'] == diag.READY_POLICY and ready['started_sha256'] == sha256(output/'started.json')
        assert ready['same_training_slice'] is True and ready['formal_data_identity'] is False
        assert ready['complete'] is False and ready['diagnostic_only'] is True
        # Ready precedes model construction and every update, then is bound to the receipt.
        assert order.index('ready') < order.index('generator') < order.index('step')
        assert order[order.index('ready')-1] == 'audit'
        assert r['output_logs_sha256']['ready.json'] == sha256(output/'ready.json')
        # Independent CPU recomputation of the actual model-input digests.
        image = np.arange(4*9*11, dtype=np.float32).reshape(1, 4, 9, 11)
        labels = np.zeros_like(image, dtype=np.int16); labels[:, 0] = -1
        labels[0, 2, 3:5, 4:6] = 1
        fit = labels >= 0
        window = ({'min': float(image[fit].min()), 'max': float(image[fit].max())}
                  if init == 'cold' else {'min': -5., 'max': 500.})
        from experiments.genseg30.models import pad_plane
        z = int(np.flatnonzero((labels[0] == 1).any(axis=(1, 2)))[0])
        plane, plane_label = image[0, z], labels[0, z]
        normalized = (2*(plane-window['min'])/(window['max']-window['min'])-1).astype(np.float32)
        normalized = np.where(plane_label >= 0, normalized, 0).astype(np.float32)
        padded_label = pad_plane(plane_label.astype(np.float32), 256, -1)[None, None]
        model = {'condition': (padded_label == 1).astype(np.float32),
                 'normalized_target': pad_plane(normalized, 256, 0)[None, None], 'label': padded_label}
        assert (scaling['min'], scaling['max']) == (window['min'], window['max'])
        assert scaling['fit_ids'] == inner_split(target.train, target.val, diag.SEED)['train']
        for name, values in model.items():
            assert ready['arrays'][name] == {'dtype': 'float32', 'shape': [1, 1, 256, 256],
                'c_contiguous': True,
                'sha256': hashlib.sha256(np.ascontiguousarray(values).tobytes(order='C')).hexdigest()}
        # F2: optional after-only mode panels, validated on disk like the original PNGs.
        previews = r['observations']['after'].get('mode_previews', {})
        assert 'mode_previews' not in r['observations']['before']
        assert (r['protocol'].get('save_mode_previews') is True) == save_mode_previews
        assert (previews != {}) == save_mode_previews
        assert len(renders) == 2 + len(previews)
        files = sorted(path.name for path in output.glob('*.png'))
        assert files == (['after.png', 'after_batchstats.png', 'after_calibrated_eval.png', 'before.png']
                         if save_mode_previews else ['after.png', 'before.png'])
        for mode, preview in previews.items():
            spec = diag.MODE_PREVIEWS[mode]
            assert preview['file'] == spec['file'] and preview['title'] == spec['title']
            assert preview['mode'] == mode and preview['source_forward'] == spec['source']
            assert sha256(output/preview['file']) == preview['sha256']
            assert preview['display']['window'] == [scaling['min'], scaling['max']]
            assert preview['display']['panel_shape'] == list(r['protocol']['selection']['native_shape'])
        assert set(previews) == ({'batchstats', 'calibrated_eval'} if save_mode_previews else set())
        # The switch reuses existing forwards: the counts never depend on it.
        assert built[0].forwards['g'] == 3 + 4 + 2*int(compare_bn_modes) + 2*int(compare_bn_calibration)
        assert built[0].forwards['d'] == 3*3 + 2*3
        assert ('compare_bn_modes' in r['protocol'])==compare_bn_modes
        assert all(('bn_mode_comparison' in o)==compare_bn_modes for o in r['observations'].values())
        assert 'bn_calibration' not in r['observations']['before']
        assert ('bn_calibration' in r['observations']['after'])==compare_bn_calibration
        assert ('compare_bn_calibration' in r['protocol'])==compare_bn_calibration
        if compare_bn_calibration:
            assert read_json(output/'started.json')['bn_calibration_policy']==r['protocol']['bn_calibration_policy']==diag.BN_CALIBRATION_POLICY
        if compare_bn_modes:
            assert read_json(output/'started.json')['bn_mode_comparison_policy']==r['protocol']['bn_mode_comparison_policy']==diag.BN_MODE_COMPARISON_POLICY
        assert r['protocol']['rates'] == {'g': 2e-5, 'd': 2e-5} and r['protocol']['l1_weight'] == 100.
        assert r['protocol']['g_objective'] == g_objective
        assert r['protocol']['g_adversarial_coefficient'] == (0. if g_objective == 'l1_only' else 1.)
        assert r['protocol']['g_l1_coefficient'] == 100.
        assert read_json(output/'started.json')['g_objective'] == g_objective
        assert read_json(output/'started.json')['g_optimized_objective'] == r['protocol']['g_optimized_objective']
        assert all(o['g_objective'] == g_objective and
                   o['g_adversarial_coefficient'] == r['protocol']['g_adversarial_coefficient'] and
                   o['g_l1_coefficient'] == 100. for o in r['observations'].values())
        repeat_policy=r['protocol']['repeat_output_policy']
        assert repeat_policy['atol']==1e-7 and repeat_policy['rtol']==0.
        assert repeat_policy['reference_float32_epsilon']==torch.finfo(torch.float32).eps
        assert 'not a loss/image-quality tolerance' in repeat_policy['interpretation']
        assert read_json(output/'started.json')['repeat_output_policy']==repeat_policy
        started=read_json(output/'started.json')
        assert started['probe_backend_policy']==r['protocol']['probe_backend_policy']==diag.PROBE_BACKEND_POLICY
        assert started['backend_at_start']==r['protocol']['backend_at_start']==diag._probe_backend_snapshot()
        assert all(o['probe_backend']['restored'] and o['probe_backend']['entry']==o['probe_backend']['exit']
                   for o in r['observations'].values())
        if strategy!='legacy':
            assert 'eval_repeat_within_tolerance_atol1e-07_rtol0' in r['protocol']['quality_gates']['state']
        entries = [json.loads(line) for line in (output/'steps.jsonl').read_text().splitlines()]
        assert [e['step'] for e in entries] == [1, 2, 3]
        assert all(e['g_objective'] == g_objective and
                   e['g_adversarial_coefficient'] == r['protocol']['g_adversarial_coefficient'] and
                   e['g_l1_coefficient'] == 100. for e in entries)
        if g_objective == 'l1_only':
            assert all(e['g_loss'] == e['g_l1_loss'] and e['g_adversarial_loss'] > 0 for e in entries)
        elif strategy != 'legacy':
            assert all(e['g_loss'] != e['g_l1_loss'] and e['g_adversarial_loss'] > 0 for e in entries)
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
    split=inner_split(target.train,target.val,diag.SEED)
    assert target.reads == (split['train'] if init=='cold' else [])+[sorted(split['train'])[0]]
    if not tamper and strategy!='legacy':
        assert r['protocol']['strategy']==strategy and r['protocol']['quality_gates']['fixed_budget_steps']==3
        quality=r['quality_assessment']
        assert quality['scalar_checks']['state'] and quality['learning_quality_passed'] is False
        assert quality['automatic_joint'] is False and quality['automatic_budget_expansion'] is False
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
    refused = read_json(output/'failed.json')
    assert refused['completed_steps'] == 0 and not target.reads
    # An early refusal never writes or references readiness evidence.
    assert not (output/'ready.json').exists() and 'ready' not in refused
    assert not (output/'diagnostic.json').exists()
    with pytest.raises(ValueError, match='Fresh'): diag.run_single_slice(target, output, **options)
    # The preview switch is refused before any data read or output creation.
    with pytest.raises(ValueError, match='Mode previews require'):
        diag.run_single_slice(target, tmp_path/'previews', pretrained_run=run, pretrained_sha256=digest,
            gpu=0, steps=3, save_mode_previews=True)
    assert not (tmp_path/'previews').exists()
    with pytest.raises(ValueError, match='l1_only G objective requires'):
        diag.run_single_slice(target, tmp_path/'ablation', pretrained_run=run, pretrained_sha256=digest,
            gpu=0, steps=3, strategy='adam_trainbn_epoch_v1', initialization_mode='gd500',
            g_objective='l1_only')
    assert not (tmp_path/'ablation').exists() and not target.reads
