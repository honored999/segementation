"""Fixed inner61 training-plane diagnostic; no reusable weights or formal result."""
import hashlib
import json
import inspect
from pathlib import Path
import random
import subprocess

import numpy as np
from experiments.genseg30.contract import (require, write_json, sha256, check_output,
    checked_file, assert_identities, validate_arrays, capture_identities)
from .convergence import (CONVERGENCE_POLICY, SNAPSHOT_FILE, TRAJECTORY_FILE, convergence_plan,
    preview_filename, save_diagnostic_snapshot, stage_name, trajectory)
from .data import Cohort, inner_split
from .models import Generator, source_identity
from .pretrain import policy, load_pretrained, REPAIR, STRATEGIES
from .preview import audit_gan, select_planes, render, TITLE_HEIGHT
from .runtime import audit, gpu_guard, between_step_guard, implementation_identity

SEED = 20261018
GD500_SHA256 = '33dc65d16c4d03fafa1d2d9884d61e11f298a9621751c126b2b72c1bc08957a6'
CLASSIFICATION = 'exploratory_fixed_training_slice_diagnostic_only_not_formal'
REPEAT_ATOL = 1e-7
REPEAT_RTOL = 0.
REPEAT_OUTPUT_POLICY = {'atol':REPEAT_ATOL,'rtol':REPEAT_RTOL,
    'reference_float32_epsilon':1.1920928955078125e-7,
    'requirements':'same_shape_device_dtype; finite_both; matched_eval_grad_context; no_broadcast; no_equal_nan',
    'interpretation':'Fixed numerical repeatability near normalized unit-scale float32 epsilon; not a loss/image-quality tolerance; parameter/BN/optimizer/RNG/mode purity remains strict'}


READY_POLICY = {'stage':'after_fixed_plane_scale_and_identity_reaudit; before_any_model_construction_or_update',
    'immutability':'single_fresh_exclusive_write; never rewritten, overwritten or backfilled',
    'identity':'fixed inner61 training slice only; not a formal full-dataset identity and not a generalization result',
    'fields':'reuses the started protocol seed/steps/g_objective/rates/l1_weight/scaling/selection/target/source identities',
    'arrays':'actual float32 condition, normalized target and label; dtype/shape/C-contiguous order/SHA256 hashed from the existing CPU arrays; no GPU transfer, RNG consumption or input change',
    'conventions':{'condition':'label==1','normalized_target':'inner61 valid-pixel minmax to [-1,1] then ignore set to 0',
        'label':'padded float32 label in {-1,0,1}','metric_region':'label>=0'},
    'binding':'the final diagnostic.json output_logs_sha256 includes this ready.json and the final identity audit re-verifies it; started.json is never modified',
    'failure':'retained on failure; failed.json references it only when it was written; an early refusal has no ready'}


MODE_PREVIEWS = {'batchstats':{'file':'after_batchstats.png','title':'batchstats',
        'source':'existing private-buffer batchstats forward'},
    'calibrated_eval':{'file':'after_calibrated_eval.png','title':'private calibrated eval',
        'source':'existing private-buffer calibrated eval forward'}}

MODE_PREVIEW_POLICY = {'switch':'save_mode_previews',
    'scope':'diagnose-single-slice after observation only; requires compare_bn_modes or compare_bn_calibration',
    'files':{mode:spec['file'] for mode,spec in MODE_PREVIEWS.items()},
    'titles':{mode:spec['title'] for mode,spec in MODE_PREVIEWS.items()},
    'source':'the existing batchstats forward or the existing private calibrated eval forward; no extra G or D forward, no second mode switch and no second calibration',
    'display':'preview.render common inner61 window on the native grid, black background outside the valid region, label overlay; no per-image minmax stretch; the real panel stays the real target',
    'authority':'after.png, statistics, activations, losses and the quality gate stay on the original eval; these PNGs only localize the compared mode and are not post-hoc formal inference',
    'arrays':'generated native planes travel only to the renderer and never enter the statistics/receipt JSON'}


PROBE_BACKEND_POLICY = {'scope':'entire_read_only_probe_G_D_forwards_and_private_gradients; engine.step_outside',
    'deterministic':True,'benchmark':False,'enabled':'preserve_caller','allow_tf32':'preserve_caller',
    'benchmark_limit':'preserve_caller_when_available','fp32_precision':'preserve_caller_when_available',
    'global_deterministic_algorithms':'unchanged','RNG':'no_reset',
    'restore':'official_cudnn.flags_on_success_and_exception; verify_entry_exit_snapshot',
    'evidence':'CPU_scope_and_restore_only; CUDA_repeatability_and_unique_root_cause_unverified'}


BN_MODE_COMPARISON_POLICY = {'eval_reference':'original_eval_with_saved_running_stats; quality_gates_eval_only',
    'batchstats':'G_BatchNorm_modules_train_only; every_other_G_module_eval; D_not_called',
    'weights_input':'same_private_functional_parameters_and_condition_as_eval; same_grad_enabled_context',
    'buffers':'FunctionalModel_clones_buffers_per_call; no_live_commit_or_num_batches_tracked_change',
    'RNG':'extra_forward_fork_rng_CPU_and_selected_CUDA; restore_on_success_and_exception',
    'backend':'same_local_probe_cudnn_context','comparison_region':'native_valid_label_ge0; excludes_padding_and_ignore',
    'output':'statistics_only; original_before_after_PNG_remain_eval; no_automatic_quality_or_BN_causality_claim'}


BN_CALIBRATION_POLICY = {'stage':'after_only','method':'one_private_BN_only_train_pass_then_all_G_eval',
    'calibration_count':1,'momentum':1.0,'weights':'same_private_parameters_frozen; calibration_no_grad; calibrated_eval_matches_original_grad_context; no_Adam_update',
    'buffers':'one_persistent_cloned_buffer_dict_shared_by_calibration_and_eval; never_live_commit',
    'variance':'PyTorch_unbiased_running_variance; calibrated_eval_not_assumed_equal_to_batchstats',
    'calibration_region':'same_fixed_inner61_training_slice_condition_full_padded_forward; no_BN_pixel_mask',
    'metric_region':'native_valid_label_ge0; padding_ignore_excluded',
    'scope':'single_training_slice_observation_not_inner61_cohort_calibration_or_inference_policy',
    'restore':'BN_momentum_and_all_G_modes_try_finally; CPU_selected_CUDA_fork_rng; same_local_backend',
    'authority':'original_eval_statistics_PNG_activations_quality_unchanged; statistics_only_no_extra_D'}


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


def ready_arrays(x, y):
    """Hash the actual float32 model inputs on CPU; no GPU copy, RNG or mutation."""
    arrays = {'condition': (y == 1).astype(np.float32), 'normalized_target': x, 'label': y}
    report = {}
    for name, values in arrays.items():
        require(type(values) is np.ndarray and values.dtype == np.float32 and values.ndim == 4,
                'Ready evidence requires the actual float32 [1,1,H,W] array: '+name)
        contiguous = np.ascontiguousarray(values)
        require(contiguous.flags['C_CONTIGUOUS'] and contiguous.dtype == np.float32,
                'Ready evidence requires a C-contiguous float32 array: '+name)
        report[name] = {'dtype': 'float32', 'shape': list(contiguous.shape), 'c_contiguous': True,
                        'sha256': hashlib.sha256(contiguous.tobytes(order='C')).hexdigest()}
    require(report['condition']['shape'] == report['label']['shape'] == report['normalized_target']['shape'],
            'Ready arrays must share the one model input shape')
    return report


def write_ready(output, protocol, x, y):
    """Write the single immutable readiness record after the fixed plane, before models."""
    path = output/'ready.json'
    write_json(path, {'format': 1, 'classification': CLASSIFICATION, 'diagnostic_only': True,
        'complete': False, 'stage': 'ready_after_fixed_plane_before_models_or_updates',
        'policy': dict(READY_POLICY), 'protocol': dict(protocol),
        'started_sha256': sha256(checked_file(output, 'started.json')),
        'same_training_slice': True, 'formal_data_identity': False, 'arrays': ready_arrays(x, y)})
    return {'file': path.name, 'sha256': sha256(path)}


def tensor_statistics(values):
    import torch
    values = values.detach()
    require(values.numel() > 0 and bool(torch.isfinite(values).all()), 'Empty/nonfinite diagnostic values')
    return {'count': values.numel(), 'mean': float(values.mean()),
            'std': float(values.std(unbiased=False)), 'min': float(values.min()), 'max': float(values.max()),
            'range': float(values.max()-values.min())}


def plane_statistics(real, label, fake, native_shape, scale):
    import torch
    require(fake.shape == real.shape == label.shape, 'Diagnostic output shape mismatch')
    h, w = native_shape
    require(0 < h <= real.shape[-2] and 0 < w <= real.shape[-1], 'Invalid native diagnostic shape')
    x, y, f = (v.detach()[0, 0, :h, :w] for v in (real, label, fake))
    require(bool(torch.isfinite(f).all()), 'Nonfinite generated plane')
    groups = {}
    valid_median=x[y>=0].median()
    for name, mask in (('valid', y >= 0), ('foreground', y == 1), ('background', y == 0)):
        if not bool(mask.any()):
            groups[name] = None
            continue
        a, b = x[mask], f[mask]
        to_native = lambda t: (t+1)*.5*(scale['max']-scale['min'])+scale['min']
        groups[name] = {'target_normalized': tensor_statistics(a), 'generated_normalized': tensor_statistics(b),
            'target_native': tensor_statistics(to_native(a)), 'generated_native': tensor_statistics(to_native(b)),
            'normalized_l1': float((a-b).abs().mean()), 'generated_spatial_std': float(b.std(unbiased=False)),
            'optimal_constant_normalized':float(a.median()),'optimal_constant_l1':float((a-a.median()).abs().mean()),
            'valid_median_constant_normalized':float(valid_median),'valid_median_constant_l1':float((a-valid_median).abs().mean())}
        # Each subregion's own median is its optimal constant; the common valid
        # median is separately evaluated inside that same subregion.
    require(groups['valid'] is not None, 'No valid diagnostic pixels')
    native = ((f+1)*.5*(scale['max']-scale['min'])+scale['min']).cpu().numpy().astype(np.float32)
    native[y.cpu().numpy() < 0] = 0
    require(np.isfinite(native).all(), 'Nonfinite inverse scaling')
    return groups, native


def _probe_backend_snapshot():
    import torch
    snapshot={key:getattr(torch.backends.cudnn,key) for key in
              ('enabled','benchmark','deterministic','allow_tf32')}
    # Precision properties are absent before PyTorch 2.9; only snapshot those present.
    for key,owner,attribute in (
            ('benchmark_limit',torch.backends.cudnn,'benchmark_limit'),
            ('conv_fp32_precision',getattr(torch.backends.cudnn,'conv',None),'fp32_precision'),
            ('matmul_fp32_precision',torch.backends.cuda.matmul,'fp32_precision')):
        if hasattr(owner,attribute): snapshot[key]=getattr(owner,attribute)
    snapshot.update(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled())
    return snapshot


def probe(engine, real, label, native_shape, scale, *, compare_bn_modes=False,compare_bn_calibration=False,
          save_mode_previews=False,mode_previews=None):
    """Locally constrain cuDNN for observation; actual training keeps caller flags."""
    import torch
    require(type(compare_bn_modes) is bool,'compare_bn_modes must be boolean')
    require(type(compare_bn_calibration) is bool,'compare_bn_calibration must be boolean')
    require(type(save_mode_previews) is bool,'save_mode_previews must be boolean')
    compare_bn_modes=compare_bn_modes or compare_bn_calibration
    # Previews only ride on arrays the compared forwards already produce.
    require(not save_mode_previews or (compare_bn_modes and isinstance(mode_previews,dict)),
            'Mode previews require a BN comparison mode and a local output container')
    entry=_probe_backend_snapshot()
    backend={'entry':entry}; failure=None
    try:
        # Official set_flags mutates only non-None inputs (also in 2.7/2.8).
        # Inspect optional keywords rather than requiring newer flags signatures.
        options=dict(enabled=None,benchmark=False,deterministic=True,allow_tf32=None)
        supported=inspect.signature(torch.backends.cudnn.flags).parameters
        options.update({key:None for key in ('benchmark_limit','fp32_precision') if key in supported})
        with torch.backends.cudnn.flags(**options):
            backend['active']=_probe_backend_snapshot()
            require(backend['active']==dict(entry,deterministic=True,benchmark=False),
                    'Probe changed unrelated backend flags')
            comparison_options={'compare_bn_modes':True} if compare_bn_modes else {}
            if compare_bn_calibration: comparison_options['compare_bn_calibration']=True
            if save_mode_previews:
                comparison_options.update(save_mode_previews=True,mode_previews=mode_previews)
            result,generated=_probe(engine,real,label,native_shape,scale,**comparison_options)
    except BaseException as exc:
        failure=exc
        raise
    finally:
        # The official context has already exited, including when probe fails.
        backend['exit']=_probe_backend_snapshot()
        backend['restored']=backend['exit']==entry
        if failure is not None: failure.probe_backend=backend
        require(backend['restored'],'Probe failed to restore caller backend flags')
    result['probe_backend']=backend
    return result,generated


def _probe(engine, real, label, native_shape, scale, *, compare_bn_modes=False,compare_bn_calibration=False,
           save_mode_previews=False,mode_previews=None):
    """Private functional gradients only; hooks never return replacement outputs."""
    import torch
    from .core import FunctionalModel, adversarial, clean, norm, generator_objective
    nets = {'G': engine.g.net, 'D': engine.d}
    refs = {prefix+'.'+name: p for prefix, net in nets.items() for name, p in net.named_parameters()}
    refs.update({'alpha.'+k: getattr(engine.g.networks, k) for k in ('conv_arch', 'upconv_arch')})
    require(all(not refs['alpha.'+k].requires_grad for k in ('conv_arch', 'upconv_arch')), 'Frozen alpha required')
    versions = {k: (p._version, p.requires_grad, p.grad, None if p.grad is None else p.grad._version)
                for k, p in refs.items()}
    buffers = {prefix+'.'+name: (v, v.detach().clone()) for prefix, net in nets.items() for name, v in net.named_buffers()}
    modules = [(m, m.training, tuple(m._forward_hooks)) for net in nets.values() for m in net.modules()]
    bn_momenta=[(m,m.momentum) for net in nets.values() for m in net.modules()
                if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)]
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(real.device).clone() if real.is_cuda else None
    python_rng = random.getstate()
    def numpy_rng():
        s = np.random.get_state()
        return (s[0], s[1].tobytes(), *s[2:])
    numpy_state = numpy_rng()
    optimizer_tensors=[(v,v._version) for opt in getattr(engine,'optimizers',{}).values()
        for state in opt.state.values() for v in state.values() if isinstance(v,torch.Tensor)]
    handles = []; activations = {}
    def observe(name):
        def hook(module, inputs, output):
            activations.setdefault(name, []).append(tensor_statistics(output))
            return None
        return hook
    try:
        for net in nets.values(): net.eval()
        for prefix, net in nets.items():
            for name, module in net.named_modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.BatchNorm2d)):
                    handles.append(module.register_forward_hook(observe(prefix+'.'+name)))
        real = clean(real, label)
        condition = (label == 1).to(real.dtype); valid = (label >= 0).to(real.dtype)
        gf, df = FunctionalModel(engine.g.net), FunctionalModel(engine.d)
        gp, dp = gf.parameters(), df.parameters()
        first_context={'grad_enabled':torch.is_grad_enabled(),'inference_mode':torch.is_inference_mode_enabled(),
                       'all_G_modules_eval':all(not m.training for m in engine.g.net.modules())}
        fake = gf(gp, condition)
        # The D(fake-for-G) forward is always observed; l1_only never backpropagates it.
        d_fake_for_g = df(dp, torch.cat((condition, fake*valid), 1))
        g_loss, objective = generator_objective(d_fake_for_g, ((fake-real).abs()*valid).sum(), valid.sum(),
            engine.l1_weight, getattr(engine,'g_objective','gan_l1'))
        d_loss = .5*(adversarial(df(dp, torch.cat((condition, real*valid), 1)), True)+
                     adversarial(df(dp, torch.cat((condition, fake.detach()*valid), 1)), False))
        require(bool(torch.isfinite(g_loss)) and bool(torch.isfinite(d_loss)), 'Nonfinite probe loss')
        gradients = {}
        for prefix, loss, params in (('G', g_loss, gp), ('D', d_loss, dp)):
            grads = torch.autograd.grad(loss, tuple(params.values()))
            require(all(bool(torch.isfinite(g).all()) for g in grads), 'Nonfinite probe gradient')
            gradients[prefix] = {name: norm([g]) for name, g in zip(params, grads)}
        statistics, generated = plane_statistics(real, label, fake, native_shape, scale)
        # Match the first forward's eval/grad conditions; detach only AFTER the
        # repeated forward so its unused graph is released without changing dispatch.
        repeated_context={'grad_enabled':torch.is_grad_enabled(),'inference_mode':torch.is_inference_mode_enabled(),
                          'all_G_modules_eval':all(not m.training for m in engine.g.net.modules())}
        repeated=gf(gp,condition).detach()
        matched_context=first_context==repeated_context
        compatible=fake.shape==repeated.shape and fake.device==repeated.device and fake.dtype==repeated.dtype
        finite=compatible and bool(torch.isfinite(fake.detach()).all()) and bool(torch.isfinite(repeated).all())
        exact=finite and torch.equal(fake.detach(),repeated)
        max_abs_diff=float((fake.detach().double()-repeated.double()).abs().max()) if finite else None
        # Fixed absolute comparison (rtol=0), after compatibility/finite guards:
        # never broadcast, equate NaNs, or relax any live-state purity check.
        within_tolerance=matched_context and finite and max_abs_diff<=REPEAT_ATOL
        if not within_tolerance:
            require(False,'Repeated eval output changed: '+
                f'max_abs_diff={max_abs_diff!r}; atol={REPEAT_ATOL}; rtol={REPEAT_RTOL}; '+
                f'dtype={fake.dtype}/{repeated.dtype}; device={fake.device}/{repeated.device}; '+
                f'shape={tuple(fake.shape)}/{tuple(repeated.shape)}; finite_outputs={finite}; '+
                f'matched_context={matched_context}; first_context={first_context}; repeated_context={repeated_context}')
        repeat_metrics={'max_abs_diff':max_abs_diff,'atol':REPEAT_ATOL,'rtol':REPEAT_RTOL,
            'matched_context':matched_context,'first_context':first_context,'repeated_context':repeated_context,
            'exact':exact,'within_tolerance':within_tolerance}
        bn_counts={key:{name:int(m.num_batches_tracked) for name,m in net.named_modules()
            if isinstance(m,torch.nn.BatchNorm2d)} for key,net in nets.items()}
        adam_steps={key:[int(state['step']) for state in opt.state.values()]
                    for key,opt in getattr(engine,'optimizers',{}).items()}
        result = {'statistics': statistics, 'g_loss': float(g_loss.detach()), 'd_loss': float(d_loss.detach()),
                  **objective, 'parameter_gradient_norms': gradients, 'activations': activations,
                  'eval_repeat_exact':exact,'eval_repeat_within_tolerance':within_tolerance,
                  'eval_repeat_metrics':repeat_metrics,'bn_counts':bn_counts,'adam_steps':adam_steps}
        if compare_bn_modes:
            # Keep authoritative activations eval-only; B has its own scalar report.
            for handle in handles: handle.remove()
            handles.clear()
            g_modes=[(module,module.training) for module in engine.g.net.modules()]
            try:
                # All G modules are already eval; switch ONLY BN. FunctionalModel
                # uses the identical gp and fresh private buffers, with no commit.
                for module,_ in g_modes:
                    if isinstance(module,torch.nn.modules.batchnorm._BatchNorm): module.training=True
                devices=[real.device.index] if real.is_cuda else []
                with torch.random.fork_rng(devices=devices), torch.set_grad_enabled(first_context['grad_enabled']):
                    batchstats=gf(gp,condition).detach()
            finally:
                for module,mode in g_modes: module.training=mode
            batch_statistics,batchstats_native=plane_statistics(real,label,batchstats,native_shape,scale)
            # Reuse this forward's discarded native plane for the optional preview only.
            if save_mode_previews: mode_previews['batchstats']=batchstats_native
            difference=(fake.detach().double()-batchstats.double()).abs()[label>=0]
            result['bn_mode_comparison']={'policy':dict(BN_MODE_COMPARISON_POLICY),
                'batchstats_statistics':batch_statistics,'grad_enabled':first_context['grad_enabled'],
                'eval_vs_batchstats':{'region':BN_MODE_COMPARISON_POLICY['comparison_region'],
                    'max_abs_diff':float(difference.max()),'mean_abs_diff':float(difference.mean())}}

        if compare_bn_calibration:
            from torch.func import functional_call
            for handle in handles: handle.remove()
            handles.clear()
            g_modes=[(m,m.training) for m in engine.g.net.modules()]
            norms=[(m,m.momentum) for m,_ in g_modes if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)]
            require(bool(norms) and all(m.track_running_stats for m,_ in norms),'Calibration requires tracked G BN')
            private_buffers={key:value.clone() for key,value in gf.buffers.items()}
            try:
                for module,_ in norms: module.training=True; module.momentum=1.0
                devices=[real.device.index] if real.is_cuda else []
                with torch.random.fork_rng(devices=devices):
                    # Persist the updates from this call for the following eval.
                    with torch.no_grad():
                        functional_call(engine.g.net,(gp,private_buffers),(condition,),strict=True)
                    for module,_ in g_modes: module.training=False
                    with torch.set_grad_enabled(first_context['grad_enabled']):
                        calibrated=functional_call(engine.g.net,(gp,private_buffers),(condition,),strict=True).detach()
            finally:
                for module,momentum in norms: module.momentum=momentum
                for module,mode in g_modes: module.training=mode
            calibrated_statistics,calibrated_native=plane_statistics(real,label,calibrated,native_shape,scale)
            if save_mode_previews: mode_previews['calibrated_eval']=calibrated_native
            difference=(fake.detach().double()-calibrated.double()).abs()[label>=0]
            result['bn_calibration']={'policy':dict(BN_CALIBRATION_POLICY),'bn_layers':len(norms),
                'calibrated_eval_statistics':calibrated_statistics,
                'eval_vs_calibrated':{'region':BN_CALIBRATION_POLICY['metric_region'],
                    'max_abs_diff':float(difference.max()),'mean_abs_diff':float(difference.mean())}}

    finally:
        for handle in handles:
            handle.remove()
        for module,mode,_ in modules: module.training=mode
        require(all(v._version==version for v,version in optimizer_tensors),'Probe changed optimizer state')
        require(all(m.momentum==momentum for m,momentum in bn_momenta),'Probe changed BN momentum')
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


def run_single_slice(target, output, *, pretrained_run=None, pretrained_sha256=None, gpu, steps, seed=SEED,
                     strategy='legacy',initialization_mode='gd500',g_objective='gan_l1',
                     compare_bn_modes=False,compare_bn_calibration=False,save_mode_previews=False,
                     convergence_audit=False,_convergence_schedule=None):
    require(type(compare_bn_modes) is bool,'compare_bn_modes must be boolean')
    require(type(compare_bn_calibration) is bool,'compare_bn_calibration must be boolean')
    require(type(save_mode_previews) is bool,'save_mode_previews must be boolean')
    compare_bn_modes=compare_bn_modes or compare_bn_calibration
    # Explicitly refuse an unsupported combination before data or output exists.
    require(not save_mode_previews or compare_bn_modes,
            'Mode previews require compare_bn_modes or compare_bn_calibration')
    require(type(steps) is int and steps > 0, 'Explicit positive diagnostic steps required')
    require(type(seed) is int and seed == SEED, 'Fixed diagnostic seed 20261018 required')
    require(type(gpu) is int and gpu >= 0, 'Physical GPU index must be a nonnegative integer')
    require(strategy in STRATEGIES,'Unknown diagnostic strategy')
    require(initialization_mode in ('gd500','cold'),'Unknown diagnostic initialization')
    cold=initialization_mode=='cold'
    require(g_objective in ('gan_l1','l1_only'),'Unknown diagnostic G objective')
    require(g_objective=='gan_l1' or (cold and strategy==REPAIR),
            'l1_only G objective requires adam_trainbn_epoch_v1 cold initialization')
    # The bounded convergence plan is fixed before any data read or output creation.
    require(type(convergence_audit) is bool,'convergence_audit must be boolean')
    plan = None
    if convergence_audit:
        require(strategy==REPAIR and cold and g_objective=='gan_l1',
                'Bounded convergence audit requires adam_trainbn_epoch_v1 cold gan_l1')
        plan = convergence_plan(steps, _convergence_schedule)
        require(plan['steps']==steps,'Bounded convergence audit budget mismatch')
    else:
        require(_convergence_schedule is None,'Private convergence schedule requires the audit option')
    require((pretrained_run is None)==(pretrained_sha256 is None),'Paired diagnostic source required')
    require((cold and strategy==REPAIR and pretrained_run is None) or (not cold and pretrained_run is not None),
            'cold requires repair and no source; gd500 requires locked source')
    run = None if cold else Path(pretrained_run).resolve()
    output = check_output(output, list(target.protected)+([] if cold else [run]))
    if cold:
        receipt=None; identities={}
        expected=dict(policy(),seed=seed,split=inner_split(target.train,target.val,seed))
    else:
        receipt, identities, expected = initialization(target, run, pretrained_sha256)
    candidate_policy=policy(strategy=strategy)
    target.require_no_unpack_writes()
    inventory = {p: target.inventory(p) for p in target.train}
    source = source_identity(); implementation = implementation_identity()
    from .core import g_objective_fields
    objective_fields = g_objective_fields(g_objective, expected['l1_weight'])
    protocol = {'format': 1, 'classification': CLASSIFICATION, 'diagnostic_only': True, 'complete': False,
        'seed': seed, 'steps': steps, 'pretrained_run': None if cold else str(run), 'pretrained_sha256': pretrained_sha256,
        'saved_protocol': None if cold else receipt['protocol'], 'source': source, 'implementation_files_sha256': implementation,
        'input_inventory': inventory, 'target_identity': target.identity, 'target_manifest_sha256': target.manifest_sha256,
        'rates': expected['rates'], 'l1_weight': expected['l1_weight'], **objective_fields, 'BN': expected['BN'],
        'architecture': expected['architecture'], 'optimizer': expected['optimizer'], 'scaling': expected.get('scaling'),
        'repeat_output_policy':dict(REPEAT_OUTPUT_POLICY),
        'probe_backend_policy':dict(PROBE_BACKEND_POLICY),'backend_at_start':_probe_backend_snapshot(),
        'probe_schedule': 'before probe inside first guarded update unit then engine.step with no intervening util resample; after probe in guarded unit after all completed updates',
        'guard': 'unchanged runtime gpu_guard; same UUID/80percent/headroom/allocator ceiling; synchronize then empty_cache; util-only max5x1s only after completed updates',
        'initialization': 'saved G/D and alpha only; fresh fixed seed; no RNG/optimizer restoration; no S',
        'interpretation': 'Fixed training sample observations only; no generalization/capacity/BN root-cause conclusion or automatic learned/not-learned threshold',
        'weights_saved': False,
        'repository_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True).stdout.strip()}
    if strategy==REPAIR:
        protocol.update({k:candidate_policy[k] for k in ('strategy','protocol_version','optimizer','BN','actual_order','failure','source_fidelity')})
        protocol['interpretation']='Fixed training-slice scalar engineering gate only; structure/semantics manual pending; no root-cause, efficacy or paper-reproduction conclusion'
        protocol['initialization']=('identical_seed_cold_G_D_alpha_buffers' if cold else 'locked_legacy_GD500_G_D_alpha_buffers')+'; fresh_Adam_zero_moments; no_resume'
        protocol['quality_gates']={'fixed_budget_steps':steps,'suggested_budget_steps':100,
            'valid_l1':'after_strictly_less_than_valid_target_median_optimal_constant_l1',
            'lesion_l1':'after_less_or_equal_before; foreground_is_label1',
            'state':f'eval_repeat_within_tolerance_atol{REPEAT_ATOL:g}_rtol{REPEAT_RTOL:g}; same_shape_device_dtype_finite_matched_context; BN_delta_per_layer_G_steps_D_3steps; every_Adam_step_exact',
            'std_range':'report_only; cannot_establish_structure_or_semantics',
            'constant_baselines':'valid_median_global; each_subregion_own_median_optimum_and_valid_median_reference',
            'structure_semantics':'manual_pending_fixed_inner61_lesion_and_empty_previews',
            'failure':'keep_results_no_auto_retry_budget_expansion_or_joint',
            'automatic_budget_expansion':False,'automatic_joint':False}
    if compare_bn_modes:
        protocol.update(compare_bn_modes=True,bn_mode_comparison_policy=dict(BN_MODE_COMPARISON_POLICY))
    if compare_bn_calibration:
        protocol.update(compare_bn_calibration=True,bn_calibration_policy=dict(BN_CALIBRATION_POLICY))
    if save_mode_previews:
        protocol.update(save_mode_previews=True,mode_preview_policy=dict(MODE_PREVIEW_POLICY))
    if plan is not None:
        protocol.update(convergence_audit=True,convergence_plan=plan,convergence_policy=dict(CONVERGENCE_POLICY),
            weights_policy='one diagnostic_state.pt after the fixed budget and final audit; not pretraining, not joint handoff, no resume, no optimizer moments',
            compare_bn_modes=True,bn_mode_comparison_policy=dict(BN_MODE_COMPARISON_POLICY),
            compare_bn_calibration=True,bn_calibration_policy=dict(BN_CALIBRATION_POLICY),
            save_mode_previews=True,mode_preview_policy=dict(MODE_PREVIEW_POLICY),
            probe_schedule='before probe inside the first guarded update unit then engine.step; each planned point probe immediately after its completed update inside that same guarded unit; the final planned point is the after observation with no duplicate probe and no observation after it',
            observation_authority='quality assessment uses only before and the final original eval; intermediate points and compared modes are localization only')
    output.mkdir(parents=True, exist_ok=False)
    engine = None
    ready = {'info': None}
    snapshot = {'info': None}
    def final_audit():
        assert_identities(identities)
        audit(target, inventory)
        require(source_identity() == source and implementation_identity() == implementation,
                'Diagnostic source/implementation changed')
        if run is not None:
            require(sha256(checked_file(run, 'gan.pt')) == pretrained_sha256, 'GD500 checkpoint changed')
        if ready['info'] is not None:
            require(sha256(checked_file(output, ready['info']['file'])) == ready['info']['sha256'],
                    'Diagnostic ready.json changed')
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
            # Legacy/recovery reuse the saved scale; cold fits only inner61.
            snap = gpu_guard(gpu, expected_uuid=initial['gpu_uuid'])
            emit({'unit': 'before_patient_arrays', 'resources': snap})
            if cold:
                expected['scaling']=cohort.fit_scale(skip_all_ignore=True)
                protocol['scaling']=expected['scaling']
            x, y, native_image, native_label, selection = fixed_plane(cohort, expected['scaling'])
            protocol['selection'] = selection
            final_audit()
            # Immutable readiness evidence from the actual scale/selection/CPU arrays,
            # written once before any model or update; a failure here never fakes one.
            ready['info'] = write_ready(output, protocol, x, y)
            identities.update(capture_identities([output/'ready.json']))
            snap = gpu_guard(gpu, expected_uuid=initial['gpu_uuid'])  # immediately before models, no wait
            emit({'unit': 'before_models', 'resources': snap})
            ceiling = min(initial['allocator_fraction'], snap['allocator_fraction'])
            torch.cuda.set_per_process_memory_fraction(ceiling, 0)
            protocol['applied_allocator_fraction'] = ceiling
            generator = Generator(seed=seed, device='cuda:0'); discriminator = generator.discriminator()
            if run is not None:
                load_pretrained(run, pretrained_sha256, generator, discriminator, expected=expected)
            engine = PretrainEngine(generator, discriminator,strategy=strategy,g_objective=g_objective)  # SAME default G/D update, BN, alpha, L1
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
            def observe(stage, state, *, point=False):
                compare=compare_bn_modes or plan is not None
                comparison_options={'compare_bn_modes':True} if compare else {}
                calibrate=compare_bn_calibration or plan is not None
                if calibrate and (stage=='after' or point): comparison_options['compare_bn_calibration']=True
                previews = {}
                if (save_mode_previews and stage=='after') or (plan is not None and point):
                    comparison_options.update(save_mode_previews=True,mode_previews=previews)
                observation, generated = probe(engine, *arrays, selection['native_shape'], expected['scaling'],**comparison_options)
                observation['resources_before'] = state
                path = output/(stage+'.png' if plan is None else preview_filename(stage,'eval'))
                observation['display'] = render(path, native_image, native_label, generated, expected['scaling'])
                observation['png'] = {'file': path.name, 'sha256': sha256(path)}
                if plan is not None:
                    observation.update(scheduled_point=point, final_after=stage=='after')
                if previews:
                    observation['mode_previews'] = {}
                    for mode, native in previews.items():
                        spec = MODE_PREVIEWS[mode]
                        expected_name = spec['file'] if plan is None else preview_filename(stage,mode)
                        file = output/expected_name
                        require(file.name == expected_name, 'Unexpected mode preview filename')
                        # Same native grid/window/label overlay; only the mode title changes.
                        display = render(file, native_image, native_label, native, expected['scaling'], mode=spec['title'])
                        observation['mode_previews'][mode] = {'file': file.name, 'mode': mode,
                            'title': spec['title'], 'source_forward': spec['source'],
                            'sha256': sha256(file), 'display': display}
                records[stage] = observation
            points = set() if plan is None else set(plan['points'])
            for index in range(steps):
                state = guarded_unit('update_'+str(index+1))
                if arrays is None:
                    arrays = tuple(torch.from_numpy(np.ascontiguousarray(a)).to('cuda:0') for a in (x, y))
                    observe('before', state)
                metrics = engine.step(*arrays)
                torch.cuda.synchronize()
                scheduled = plan is not None and engine.steps in points
                entry = {'step': engine.steps, 'selection': selection, 'resources_before': state,
                         'cuda_peak_allocated': torch.cuda.max_memory_allocated(), **objective_fields, **metrics}
                if scheduled:
                    entry['convergence_observation'] = stage_name(engine.steps, steps)
                step_log.write(json.dumps(entry, allow_nan=False)+'\n'); step_log.flush()
                print(json.dumps(entry, allow_nan=False), flush=True)
                # A failed point probe stops the run: no further update is attempted.
                if scheduled:
                    observe(entry['convergence_observation'], state, point=True)
            if plan is None:
                observe('after', guarded_unit('after_probe'))
        require(engine.steps == steps, 'Incomplete diagnostic step budget')
        from PIL import Image
        h, w = selection['native_shape']
        for observation in records.values():
            for entry in (observation['png'], *observation.get('mode_previews', {}).values()):
                path = checked_file(output, entry['file'])
                require(sha256(path) == entry['sha256'], 'Diagnostic PNG changed')
                with Image.open(path) as image:
                    require(image.format == 'PNG' and image.size == (4*w, h+TITLE_HEIGHT), 'Invalid diagnostic PNG')
                    image.verify()
        if plan is None:
            require(not any(output.glob('*.pt')) and not (output/'complete.json').exists(),
                    'Diagnostic must not save weights/training receipt')
        else:
            require(not any(output.glob('*.pt')),
                    'The diagnostic state snapshot is written once, after the full fixed budget')
            require(not (output/'complete.json').exists(), 'Diagnostic must not save a training receipt')
        final_audit()
        convergence = None
        if plan is not None:
            require(ready['info'] is not None, 'Convergence snapshot requires readiness evidence')
            snapshot['info'] = save_diagnostic_snapshot(output, engine, protocol,
                ready=ready['info'], source=source, seed=seed)
            require(sorted(path.name for path in output.glob('*.pt')) == [SNAPSHOT_FILE],
                    'Exactly one diagnostic-only state snapshot is permitted')
            body = trajectory(plan, records)
            body['snapshot'] = dict(snapshot['info'])
            write_json(output/TRAJECTORY_FILE, body)
            convergence = {'plan': plan, 'policy': dict(CONVERGENCE_POLICY),
                'trajectory_file': TRAJECTORY_FILE, 'trajectory_sha256': sha256(checked_file(output, TRAJECTORY_FILE)),
                'snapshot': dict(snapshot['info']), 'scheduled_points_complete': True,
                'intermediate_points_are_final_result': False}
        log_hashes = {name: sha256(checked_file(output, name)) for name in ('started.json', 'ready.json', 'steps.jsonl', 'guard_events.jsonl')}
        if convergence is not None:
            log_hashes[TRAJECTORY_FILE] = convergence['trajectory_sha256']
            log_hashes[SNAPSHOT_FILE] = convergence['snapshot']['sha256']
        # This diagnostic-only receipt is LAST. Never emit a valid GAN/joint completion receipt.
        quality={'quality_assessment':assess_quality(records,steps)} if strategy==REPAIR else {}
        write_json(output/'diagnostic.json', {'format': 1, 'complete': True, 'diagnostic_only': True,
            'classification': CLASSIFICATION, 'completed_steps': engine.steps,
            'weights_saved': plan is not None, 'protocol': protocol, 'observations': records,
            'output_logs_sha256': log_hashes, **quality,
            **({} if convergence is None else {'convergence': convergence,
                'reusable_checkpoint_saved': False})})
    except BaseException as exc:
        # Remove only our own possibly partial success marker in this freshly created output.
        (output/'diagnostic.json').unlink(missing_ok=True)
        failure = {'format': 1, 'complete': False, 'diagnostic_only': True,
            'classification': CLASSIFICATION, 'completed_steps': 0 if engine is None else engine.steps,
            'error_type': type(exc).__name__, 'error': str(exc), 'resource_guard': getattr(exc, 'diagnostic', None),
            'probe_backend':getattr(exc,'probe_backend',None)}
        # Reference only evidence that really exists: early refusals have no ready.
        if ready['info'] is not None:
            failure['ready'] = {'file': ready['info']['file'], 'sha256': ready['info']['sha256']}
        # A snapshot is referenced once written; a partially validated one is marked partial.
        partial = snapshot['info'] or getattr(exc, 'snapshot_partial', None)
        if partial is not None:
            failure['diagnostic_state'] = dict(partial)
        write_json(output/'failed.json', failure)
        raise
    return output


def assess_quality(records,steps):
    """Fixed engineering scalar gate; manual image/semantic review always pending."""
    before,after=records['before'],records['after']
    a,b=before['statistics'],after['statistics']
    state=all(record['eval_repeat_within_tolerance'] for record in (before,after))
    for key,count in (('G',steps),('D',3*steps)):
        state=state and bool(after['bn_counts'][key]) and before['bn_counts'][key].keys()==after['bn_counts'][key].keys()
        state=state and all(value-before['bn_counts'][key][name]==count for name,value in after['bn_counts'][key].items())
    state=state and set(after['adam_steps'])=={'g','d'} and all(values and all(v==steps for v in values)
        for values in after['adam_steps'].values())
    checks={'state':bool(state),
        'valid_beats_optimal_constant':b['valid']['normalized_l1']<b['valid']['optimal_constant_l1'],
        'lesion_not_worse':a['foreground'] is not None and b['foreground'] is not None and
            b['foreground']['normalized_l1']<=a['foreground']['normalized_l1']}
    return {'scalar_checks':checks,'scalar_pass':all(checks.values()),
        'status':'MANUAL_STRUCTURE_SEMANTICS_PENDING' if all(checks.values()) else 'FAILED_ENGINEERING_GATE',
        'learning_quality_passed':False,'structure_semantics':'manual_pending',
        'automatic_budget_expansion':False,'automatic_joint':False}
