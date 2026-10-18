"""Fixed inner61 training-plane diagnostic; no reusable weights or formal result."""
import json
from pathlib import Path
import random
import subprocess

import numpy as np
from experiments.genseg30.contract import (require, write_json, sha256, check_output,
    checked_file, assert_identities, validate_arrays)
from .data import Cohort, inner_split
from .models import Generator, source_identity
from .pretrain import policy, load_pretrained
from .preview import audit_gan, select_planes, render, TITLE_HEIGHT
from .runtime import audit, gpu_guard, between_step_guard, implementation_identity

SEED = 20261018
GD500_SHA256 = '33dc65d16c4d03fafa1d2d9884d61e11f298a9621751c126b2b72c1bc08957a6'
CLASSIFICATION = 'exploratory_fixed_training_slice_diagnostic_only_not_formal'


def initialization(target, run, digest):
    """Strict default-policy GD500 metadata audit before models/patient arrays."""
    require(digest == GD500_SHA256, 'This diagnostic requires the approved GD500 SHA256')
    receipt, identities = audit_gan(target, run, digest)
    expected = dict(policy(), seed=SEED, steps=500,
        split=inner_split(target.train, target.val, SEED),
        target_identity=target.identity, target_manifest_sha256=target.manifest_sha256,
        model_config={'G': {'width': 64, 'downs': 8, 'seed': SEED}, 'D_width': 64},
        scaling=receipt['protocol']['scaling'])
    for key, value in expected.items():
        require(receipt['protocol'].get(key) == value, 'Diagnostic GD500 policy mismatch: '+key)
    return receipt, identities, expected


def fixed_plane(cohort, scale):
    """No fallback patient, sampling, flips or scaling fit."""
    from experiments.genseg30.models import pad_plane
    patient = sorted(cohort.train)[0]
    require(patient in cohort.split['train'] and patient not in cohort.val+cohort.external,
            'Forbidden patient before fixed-plane array read')
    image, label, _ = cohort.load('train', patient)
    validate_arrays(image, label)
    planes, _ = select_planes(label[0])
    lesion = [z for z, category in planes if category == 'lesion']
    require(bool(lesion), 'First sorted inner61 patient has no valid lesion plane; no fallback')
    z = lesion[0]
    native_image = image[0, z].copy(); native_label = label[0, z].copy()
    normalized = (2*(native_image-scale['min'])/(scale['max']-scale['min'])-1).astype(np.float32)
    normalized = np.where(native_label >= 0, normalized, 0).astype(np.float32)
    x = pad_plane(normalized, 256, 0)[None, None]
    y = pad_plane(native_label.astype(np.float32), 256, -1)[None, None]
    record = {'role': 'train', 'patient': patient, 'z': z, 'category': 'lesion',
              'native_shape': list(native_label.shape), 'padded_shape': list(x.shape[2:]),
              'selection': 'sorted(inner61)[0]; first valid lesion plane; no fallback/flip/random sample'}
    return x, y, native_image, native_label, record


def tensor_statistics(values):
    import torch
    values = values.detach()
    require(values.numel() > 0 and bool(torch.isfinite(values).all()), 'Empty/nonfinite diagnostic values')
    return {'count': values.numel(), 'mean': float(values.mean()),
            'std': float(values.std(unbiased=False)), 'min': float(values.min()), 'max': float(values.max())}


def plane_statistics(real, label, fake, native_shape, scale):
    import torch
    require(fake.shape == real.shape == label.shape, 'Diagnostic output shape mismatch')
    h, w = native_shape
    require(0 < h <= real.shape[-2] and 0 < w <= real.shape[-1], 'Invalid native diagnostic shape')
    x, y, f = (v.detach()[0, 0, :h, :w] for v in (real, label, fake))
    require(bool(torch.isfinite(f).all()), 'Nonfinite generated plane')
    groups = {}
    for name, mask in (('valid', y >= 0), ('foreground', y == 1), ('background', y == 0)):
        if not bool(mask.any()):
            groups[name] = None
            continue
        a, b = x[mask], f[mask]
        to_native = lambda t: (t+1)*.5*(scale['max']-scale['min'])+scale['min']
        groups[name] = {'target_normalized': tensor_statistics(a), 'generated_normalized': tensor_statistics(b),
            'target_native': tensor_statistics(to_native(a)), 'generated_native': tensor_statistics(to_native(b)),
            'normalized_l1': float((a-b).abs().mean()), 'generated_spatial_std': float(b.std(unbiased=False))}
    require(groups['valid'] is not None, 'No valid diagnostic pixels')
    native = ((f+1)*.5*(scale['max']-scale['min'])+scale['min']).cpu().numpy().astype(np.float32)
    native[y.cpu().numpy() < 0] = 0
    require(np.isfinite(native).all(), 'Nonfinite inverse scaling')
    return groups, native


def probe(engine, real, label, native_shape, scale):
    """Private functional gradients only; hooks never return replacement outputs."""
    import torch
    from .core import FunctionalModel, gan_generator_loss, adversarial, clean, norm
    nets = {'G': engine.g.net, 'D': engine.d}
    require(all(not module.training for net in nets.values() for module in net.modules()),
            'Probe requires the existing frozen-BN/eval policy')
    refs = {prefix+'.'+name: p for prefix, net in nets.items() for name, p in net.named_parameters()}
    refs.update({'alpha.'+k: getattr(engine.g.networks, k) for k in ('conv_arch', 'upconv_arch')})
    require(all(not refs['alpha.'+k].requires_grad for k in ('conv_arch', 'upconv_arch')), 'Frozen alpha required')
    versions = {k: (p._version, p.requires_grad, p.grad, None if p.grad is None else p.grad._version)
                for k, p in refs.items()}
    buffers = {prefix+'.'+name: (v, v.detach().clone()) for prefix, net in nets.items() for name, v in net.named_buffers()}
    modules = [(m, m.training, tuple(m._forward_hooks)) for net in nets.values() for m in net.modules()]
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(real.device).clone() if real.is_cuda else None
    python_rng = random.getstate()
    def numpy_rng():
        s = np.random.get_state()
        return (s[0], s[1].tobytes(), *s[2:])
    numpy_state = numpy_rng()
    handles = []; activations = {}
    def observe(name):
        def hook(module, inputs, output):
            activations.setdefault(name, []).append(tensor_statistics(output))
            return None
        return hook
    try:
        for prefix, net in nets.items():
            for name, module in net.named_modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.BatchNorm2d)):
                    handles.append(module.register_forward_hook(observe(prefix+'.'+name)))
        real = clean(real, label)
        condition = (label == 1).to(real.dtype); valid = (label >= 0).to(real.dtype)
        gf, df = FunctionalModel(engine.g.net), FunctionalModel(engine.d)
        gp, dp = gf.parameters(), df.parameters()
        fake = gf(gp, condition)
        g_loss = gan_generator_loss(df, dp, condition, fake, real, valid, engine.l1_weight)
        d_loss = .5*(adversarial(df(dp, torch.cat((condition, real*valid), 1)), True)+
                     adversarial(df(dp, torch.cat((condition, fake.detach()*valid), 1)), False))
        require(bool(torch.isfinite(g_loss)) and bool(torch.isfinite(d_loss)), 'Nonfinite probe loss')
        gradients = {}
        for prefix, loss, params in (('G', g_loss, gp), ('D', d_loss, dp)):
            grads = torch.autograd.grad(loss, tuple(params.values()))
            require(all(bool(torch.isfinite(g).all()) for g in grads), 'Nonfinite probe gradient')
            gradients[prefix] = {name: norm([g]) for name, g in zip(params, grads)}
        statistics, generated = plane_statistics(real, label, fake, native_shape, scale)
        result = {'statistics': statistics, 'g_loss': float(g_loss.detach()), 'd_loss': float(d_loss.detach()),
                  'parameter_gradient_norms': gradients, 'activations': activations}
    finally:
        for handle in handles:
            handle.remove()
        current = {prefix+'.'+name: p for prefix, net in nets.items() for name, p in net.named_parameters()}
        current.update({'alpha.'+k: getattr(engine.g.networks, k) for k in ('conv_arch', 'upconv_arch')})
        require(current.keys() == refs.keys(), 'Probe changed live parameter keys')
        for name, p in refs.items():
            version, requires_grad, grad, grad_version = versions[name]
            require(current[name] is p and p._version == version and p.requires_grad == requires_grad and
                    p.grad is grad and (grad is None or grad._version == grad_version),
                    'Probe changed live parameter/gradient: '+name)
        current_buffers = {prefix+'.'+name: v for prefix, net in nets.items() for name, v in net.named_buffers()}
        require(current_buffers.keys() == buffers.keys() and all(current_buffers[k] is v and torch.equal(v, saved)
                for k, (v, saved) in buffers.items()), 'Probe changed live buffers')
        require(all(m.training == mode and tuple(m._forward_hooks) == hooks for m, mode, hooks in modules),
                'Probe changed module modes/hooks')
        require(torch.equal(torch.get_rng_state(), cpu_rng) and (cuda_rng is None or
                torch.equal(torch.cuda.get_rng_state(real.device), cuda_rng)) and random.getstate() == python_rng and
                numpy_rng() == numpy_state, 'Probe changed RNG')
    result['purity'] = {'live_parameters': 'same_identity_and_version', 'live_gradients': 'same_identity_and_version',
                        'buffers': 'same_identity_and_exact_values', 'RNG': 'exact_unchanged', 'hooks': 'removed_in_finally'}
    return result, generated


def run_single_slice(target, output, *, pretrained_run, pretrained_sha256, gpu, steps, seed=SEED):
    require(type(steps) is int and steps > 0, 'Explicit positive diagnostic steps required')
    require(type(seed) is int and seed == SEED, 'Fixed diagnostic seed 20261018 required')
    require(type(gpu) is int and gpu >= 0, 'Physical GPU index must be a nonnegative integer')
    run = Path(pretrained_run).resolve()
    output = check_output(output, list(target.protected)+[run])
    receipt, identities, expected = initialization(target, run, pretrained_sha256)
    target.require_no_unpack_writes()
    inventory = {p: target.inventory(p) for p in target.train}
    source = source_identity(); implementation = implementation_identity()
    protocol = {'format': 1, 'classification': CLASSIFICATION, 'diagnostic_only': True, 'complete': False,
        'seed': seed, 'steps': steps, 'pretrained_run': str(run), 'pretrained_sha256': pretrained_sha256,
        'saved_protocol': receipt['protocol'], 'source': source, 'implementation_files_sha256': implementation,
        'input_inventory': inventory, 'target_identity': target.identity, 'target_manifest_sha256': target.manifest_sha256,
        'rates': expected['rates'], 'l1_weight': expected['l1_weight'], 'BN': expected['BN'],
        'architecture': expected['architecture'], 'optimizer': expected['optimizer'], 'scaling': expected['scaling'],
        'probe_schedule': 'before probe inside first guarded update unit then engine.step with no intervening util resample; after probe in guarded unit after all completed updates',
        'guard': 'unchanged runtime gpu_guard; same UUID/80percent/headroom/allocator ceiling; synchronize then empty_cache; util-only max5x1s only after completed updates',
        'initialization': 'saved G/D and alpha only; fresh fixed seed; no RNG/optimizer restoration; no S',
        'interpretation': 'Fixed training sample observations only; no generalization/capacity/BN root-cause conclusion or automatic learned/not-learned threshold',
        'weights_saved': False,
        'repository_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True).stdout.strip()}
    output.mkdir(parents=True, exist_ok=False)
    engine = None
    def final_audit():
        assert_identities(identities)
        audit(target, inventory)
        require(source_identity() == source and implementation_identity() == implementation,
                'Diagnostic source/implementation changed')
        require(sha256(checked_file(run, 'gan.pt')) == pretrained_sha256, 'GD500 checkpoint changed')
    try:
        write_json(output/'started.json', protocol)
        with (output/'guard_events.jsonl').open('x', encoding='utf-8') as guard_log, \
                (output/'steps.jsonl').open('x', encoding='utf-8') as step_log:
            def emit(event):
                guard_log.write(json.dumps(event, allow_nan=False)+'\n'); guard_log.flush()
            initial = gpu_guard(gpu)  # immediate refusal before any model/patient arrays; no cooldown
            protocol['initial_resources'] = initial
            emit({'unit': 'initial', 'resources': initial})
            import torch
            from .core import PretrainEngine
            require(not torch.cuda.is_initialized(), 'GPU UUID binding must precede CUDA initialization')
            require(torch.cuda.is_available(), 'CUDA-enabled PyTorch required')
            torch.cuda.set_per_process_memory_fraction(initial['allocator_fraction'], 0)
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            final_audit()
            cohort = Cohort(target, seed)
            require(cohort.split == expected['split'], 'Diagnostic cohort changed')
            # Recheck immediately before the sole patient-array load. Never fit scale.
            snap = gpu_guard(gpu, expected_uuid=initial['gpu_uuid'])
            emit({'unit': 'before_patient_arrays', 'resources': snap})
            x, y, native_image, native_label, selection = fixed_plane(cohort, expected['scaling'])
            protocol['selection'] = selection
            final_audit()
            snap = gpu_guard(gpu, expected_uuid=initial['gpu_uuid'])  # immediately before models, no wait
            emit({'unit': 'before_models', 'resources': snap})
            ceiling = min(initial['allocator_fraction'], snap['allocator_fraction'])
            torch.cuda.set_per_process_memory_fraction(ceiling, 0)
            protocol['applied_allocator_fraction'] = ceiling
            generator = Generator(seed=seed, device='cuda:0'); discriminator = generator.discriminator()
            load_pretrained(run, pretrained_sha256, generator, discriminator, expected=expected)
            engine = PretrainEngine(generator, discriminator)  # SAME default G/D update, BN, alpha, L1
            records = {}
            arrays = None
            def guarded_unit(unit):
                final_audit()
                torch.cuda.synchronize(); torch.cuda.empty_cache()
                options = dict(expected_uuid=initial['gpu_uuid'])
                if engine.steps > 0:
                    state = between_step_guard(gpu, emit=emit, **options)
                else:
                    state = gpu_guard(gpu, **options)
                state['applied_allocator_fraction'] = ceiling
                emit({'unit': unit, 'completed_steps': engine.steps, 'resources': state})
                return state
            def observe(stage, state):
                observation, generated = probe(engine, *arrays, selection['native_shape'], expected['scaling'])
                observation['resources_before'] = state
                path = output/(stage+'.png')
                observation['display'] = render(path, native_image, native_label, generated, expected['scaling'])
                observation['png'] = {'file': path.name, 'sha256': sha256(path)}
                records[stage] = observation
            for index in range(steps):
                state = guarded_unit('update_'+str(index+1))
                if arrays is None:
                    arrays = tuple(torch.from_numpy(np.ascontiguousarray(a)).to('cuda:0') for a in (x, y))
                    observe('before', state)
                metrics = engine.step(*arrays)
                torch.cuda.synchronize()
                entry = {'step': engine.steps, 'selection': selection, 'resources_before': state,
                         'cuda_peak_allocated': torch.cuda.max_memory_allocated(), **metrics}
                step_log.write(json.dumps(entry, allow_nan=False)+'\n'); step_log.flush()
                print(json.dumps(entry, allow_nan=False), flush=True)
            observe('after', guarded_unit('after_probe'))
        require(engine.steps == steps, 'Incomplete diagnostic step budget')
        from PIL import Image
        for observation in records.values():
            path = checked_file(output, observation['png']['file'])
            require(sha256(path) == observation['png']['sha256'], 'Diagnostic PNG changed')
            with Image.open(path) as image:
                h, w = selection['native_shape']
                require(image.format == 'PNG' and image.size == (4*w, h+TITLE_HEIGHT), 'Invalid diagnostic PNG')
                image.verify()
        require(not any(output.glob('*.pt')) and not (output/'complete.json').exists(), 'Diagnostic must not save weights/training receipt')
        log_hashes = {name: sha256(checked_file(output, name)) for name in ('started.json', 'steps.jsonl', 'guard_events.jsonl')}
        final_audit()
        # This diagnostic-only receipt is LAST. Never emit a valid GAN/joint completion receipt.
        write_json(output/'diagnostic.json', {'format': 1, 'complete': True, 'diagnostic_only': True,
            'classification': CLASSIFICATION, 'completed_steps': engine.steps, 'weights_saved': False,
            'protocol': protocol, 'observations': records, 'output_logs_sha256': log_hashes})
    except BaseException as exc:
        # Remove only our own possibly partial success marker in this freshly created output.
        (output/'diagnostic.json').unlink(missing_ok=True)
        write_json(output/'failed.json', {'format': 1, 'complete': False, 'diagnostic_only': True,
            'classification': CLASSIFICATION, 'completed_steps': 0 if engine is None else engine.steps,
            'error_type': type(exc).__name__, 'error': str(exc), 'resource_guard': getattr(exc, 'diagnostic', None)})
        raise
    return output
