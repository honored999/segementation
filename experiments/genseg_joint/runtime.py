"""Fresh outputs, immutable inputs and guarded single-GPU pilot orchestration."""
import csv
import json
import math
import os
from pathlib import Path
import random
import subprocess
import numpy as np
from experiments.genseg30.contract import require, check_output, write_json, sha256, assert_identities

TARGET_SHA256='23d25e2b2a6d7c7ca2a270690a9d4d35116ff4f834458b48a9049ff889894a56'
CLASSIFICATION='engineering_genseg_joint_author_unet_coldstart_not_formal'


def gpu_guard(index, *, expected_uuid=None, min_free_vram_gib=12., min_free_ram_gib=8.):
    from experiments.genseg3mm.targets import bind_gpu
    require(type(index) is int and index>=0,'Physical GPU index must be a nonnegative integer')
    # Query only the requested physical GPU; other cards may run unrelated jobs.
    result=subprocess.run(['nvidia-smi','-i',str(index),
                           '--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total',
                           '--format=csv,noheader,nounits'],check=True,capture_output=True,text=True)
    rows=list(csv.reader(result.stdout.strip().splitlines()))
    require(len(rows)==1 and len(rows[0])==5,f'Physical GPU {index}: expected one five-field telemetry row')
    try:
        idx,uid,util,used,total=[v.strip() for v in rows[0]]
        measured_index=int(idx)
        util,used,total=float(util),float(used),float(total)
    except ValueError as exc:
        raise ValueError(f'Physical GPU {index}: invalid numeric telemetry {rows[0]!r}') from exc
    require(measured_index==index,f'Physical GPU {index}: telemetry index mismatch ({measured_index})')
    require(all(math.isfinite(v) and v>=0 for v in (util,used,total)) and total>0,
            f'Physical GPU {index}: invalid utilization={util}, memory={used}/{total} MiB; '
            'finite nonnegative metrics and positive total required')
    memory_pct=100*used/total
    context=(f'Physical GPU {index}: utilization={util:g}%, memory={used:g}/{total:g} MiB '
             f'({memory_pct:g}%); require utilization<80% and VRAM<80%')
    require(util<80 and memory_pct<80,'Resource guard refused: '+context)
    # Keep the real CPU/RAM/headroom/selected-GPU guard and UUID environment binding.
    try:
        snap=bind_gpu(index,min_vram_gib=min_free_vram_gib,min_ram_gib=min_free_ram_gib)
    except ValueError as exc:
        raise ValueError(f'{context}: {exc}') from exc
    require(bool(uid) and uid==snap['gpu_uuid'],f'Physical GPU {index}: selected telemetry/binding UUID mismatch')
    if expected_uuid is not None:
        require(uid==expected_uuid,f'Physical GPU {index}: UUID changed from {expected_uuid} to {uid}')
    selected={'index':index,'uuid':uid,'util_pct':util,'used_mib':used,'total_mib':total}
    # Compatibility key: a ONE-element list, explicitly scoped to the selected card.
    snap['all_gpus']=[selected]
    snap['gpu_scope']='selected_physical_gpu_only'
    # This is an allocation ceiling, not a prediction of the unknown peak.
    snap['allocator_fraction']=max(0.,.79-selected['used_mib']/selected['total_mib'])
    require(snap['allocator_fraction']>0,'No guarded GPU allocation budget')
    return snap


def audit(target, inventory):
    target.verify_identity(target.identity)
    target.verify_inventory(inventory)
    assert_identities(target._identities)


def save_checkpoint(output,engine,protocol,rng,*,final_audit=None):
    import torch
    from .models import source_identity
    output=Path(output)
    payload={'format':1,'classification':protocol.get('classification',CLASSIFICATION),'steps':engine.steps,
             'G':engine.g.net.state_dict(),'D':engine.d.state_dict(),'S':engine.s.state_dict(),
             'arch':{k:getattr(engine.g.networks,k).detach().cpu().clone() for k in ('upconv_arch','conv_arch')},
             'model_config':{'G':engine.g.config,'S':'pinned_author_UNet(1,1,bilinear=False)'},
             'optimizers':engine.optimizer_state(),'protocol':protocol,'rng':rng,'source':source_identity()}
    path=output/'joint.pt'
    with path.open('xb') as stream: torch.save(payload,stream)
    if final_audit is not None: final_audit()
    # Completion receipt LAST; a failed/partial serialization is never a complete run.
    write_json(output/'complete.json',{'format':1,'classification':protocol.get('classification',CLASSIFICATION),'complete':True,
        'steps':engine.steps,'checkpoint':'joint.pt','checkpoint_sha256':sha256(path),'protocol':protocol})


def implementation_identity():
    root=Path(__file__).resolve().parent
    return {p.name:sha256(p) for p in sorted(root.glob('*.py'))}


def run(target,output,*,gpu,steps,seed=20261018,g_lr=2e-5,d_lr=2e-5,s_lr=1e-5,arch_lr=1e-4,
        l1_weight=100.,min_free_vram_gib=12.,min_free_ram_gib=8.,pretrain_steps=0,
        pretrained_run=None,pretrained_sha256=None,phase='joint'):
    require(type(steps) is int and steps>0 and pretrain_steps==0,'Positive explicit steps; no combined pretraining or resume')
    require(phase in ('joint','gan_pretrain'),'Invalid phase')
    require(type(seed) is int and seed==20261018,'Fixed inner61/15 seed 20261018 required')
    require((pretrained_run is None)==(pretrained_sha256 is None),'Paired pretrained-run and pretrained-sha256 required')
    require(phase=='joint' or pretrained_run is None,'Pretrain phase cannot load prior weights; no resume')
    require(all(math.isfinite(v) and v>0 for v in (g_lr,d_lr,s_lr,arch_lr)), 'Positive finite rates required')
    require(math.isfinite(l1_weight) and l1_weight>=0,'Invalid L1 weight')
    from .pretrain import policy, audit_pretrained, load_pretrained, save_pretrain
    from .data import inner_split
    handoff_expected=dict(policy(g_lr=g_lr,d_lr=d_lr,l1_weight=l1_weight),seed=seed,
        split=inner_split(target.train,target.val,seed),target_identity=target.identity,
        target_manifest_sha256=target.manifest_sha256,
        model_config={'G':{'width':64,'downs':8,'seed':seed},'D_width':64})
    receipt=None
    from experiments.genseg30.contract import capture_identities
    protected=list(target.protected)
    pretrain_identities={}
    if pretrained_run is not None:
        protected.append(Path(pretrained_run).resolve())
        pretrain_identities=capture_identities([Path(pretrained_run)/'gan.pt',Path(pretrained_run)/'complete.json'])
        receipt=audit_pretrained(pretrained_run,pretrained_sha256,expected=handoff_expected)
        assert_identities(pretrain_identities)
    output=check_output(output,protected)
    guard_args=dict(min_free_vram_gib=min_free_vram_gib,min_free_ram_gib=min_free_ram_gib)
    initial=gpu_guard(gpu,**guard_args)  # before any model or patient image load
    import torch
    from .data import Cohort
    from .models import create_models,source_identity
    from .core import JointEngine
    require(not torch.cuda.is_initialized(),'GPU UUID binding must precede CUDA initialization')
    require(torch.cuda.is_available(),'CUDA-enabled PyTorch required on target PC')
    torch.cuda.set_per_process_memory_fraction(initial['allocator_fraction'],0)
    target.require_no_unpack_writes()
    cohort=Cohort(target,seed)
    inventory={p:target.inventory(p) for p in target.train}
    protocol={'classification':CLASSIFICATION,'complete':False,'seed':seed,'steps':steps,'pretrain_steps':0,
        'split':cohort.split,'target_identity':target.identity,'target_manifest_sha256':target.manifest_sha256,
        'initial_resources':initial,'source':source_identity(),'implementation_files_sha256':implementation_identity(),
        'models':{'G_width':64,'G_downs':8,'D_width':64,'S':'exact_author_UNet(1,1)_width64','dtype':'float32'},
        'rates':{'g':g_lr,'d':d_lr,'s':s_lr,'alpha':arch_lr},'l1_weight':l1_weight,
        'optimizer':{'G_D_S':'virtual_and_actual_stateless_SGD_no_momentum_decay',
                     'alpha':{'algorithm':'staged_Adam','lr':arch_lr,'betas':[.5,.999],'eps':1e-8,'weight_decay':0.,
                              'source_fidelity':'Eq8_SGD_to_Adam_deliberate_author_code_informed_adaptation'}},'unroll_steps':1,
        'segmentation_loss':'BCEWithLogits_plus_softDice_smooth1; real1_fake1',
        'all_real_input':'same_3mm_masked_target_for_G_D_S_internal_validation',
        'fake_condition':'train_label_random_independent_xy_flip_fixed_numpy_seed',
        'pad':'native_full_planes_bottom_right_to_multiple256; image0_label-1_no_resize',
        'normalization':'inner61_only_valid_pixel_minmax_minus1_plus1_no_clip',
        'inherited_preprocessing':'3mm_target_background_fill_from_old76_metadata_and_original_nnUNet_grid_not_refit',
        'old_generator':'provenance_only_never_loaded_as_model',
        'architecture':'upconv_only; unused_plain_conv_alpha_frozen_not_updated',
        'derivative':'full_autodiff_direct_alpha_in_Gprime_forward_plus_indirect_virtualG_route; Eq9_interpretation',
        'BN':'eval_frozen_buffers_all_virtual_and_actual_calls_deviation',
        'actual_order':'alpha_then_G_D_gradients_using_new_alpha_then_S_using_updated_G',
        'guard':'80percent_all_resources_and_free_headroom_before_models_and_every_step; synchronize_then_empty_cache_before_step_guard; fail_no_cooldown_no_interference',
        'selection':'last_complete_step_only; no_outer19_reads_or_metrics; no_resume',
        'runtime':{'torch':str(torch.__version__),'numpy':np.__version__},
        'repository_commit':subprocess.run(['git','rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()}
    protocol['phase']=phase
    if phase=='gan_pretrain':
        protocol.update(handoff_expected)
        protocol['classification']='engineering_genseg_gd_pretrain_not_formal'
        # Remove joint-only declarations: no segmenter, virtual chain or alpha Adam exists.
        for key in ('models','derivative','unroll_steps','segmentation_loss','actual_order','fake_condition'):
            protocol.pop(key,None)
        protocol['pretrain_steps']=steps
    elif receipt is not None:
        protocol['classification']='engineering_genseg_joint_pretrained_GD_fresh_UNet_not_formal'
        protocol['pretrain_steps']=receipt['steps']
        protocol['pretrained']={'run':str(Path(pretrained_run).resolve()),'sha256':pretrained_sha256,
            'steps':receipt['steps'],'rng':'fresh_joint_seed; pretrain_RNG_not_restored',
            'S':'fresh_author_UNet','alpha_Adam':'fresh_empty_moments','resume':False}
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'started.json',protocol)
    rng=np.random.default_rng(seed); random.seed(seed); torch.manual_seed(seed)
    engine=None
    def final_audit():
        assert_identities(pretrain_identities)
        audit(target,inventory)
        require(implementation_identity()==protocol['implementation_files_sha256'],'Implementation source changed during pilot')
        require(source_identity()==protocol['source'],'Pinned model source changed during pilot')
    try:
        scale=cohort.fit_scale(); protocol['scaling']=scale
        if receipt is not None:
            require(scale==receipt['protocol']['scaling'],'Pretrain scaling mismatch')
            handoff_expected['scaling']=scale
        final_audit()
        snap=gpu_guard(gpu,expected_uuid=initial['gpu_uuid'],**guard_args)
        # Recheck before models; never raise the initially permitted ceiling.
        allocator_ceiling=min(initial['allocator_fraction'],snap['allocator_fraction'])
        torch.cuda.set_per_process_memory_fraction(allocator_ceiling,0)
        if phase=='gan_pretrain':
            from .models import Generator
            from .core import PretrainEngine
            g=Generator(seed=seed,device='cuda:0'); d=g.discriminator()
            engine=PretrainEngine(g,d,g_lr=g_lr,d_lr=d_lr,l1_weight=l1_weight)
        else:
            g,d,s=create_models(seed=seed,device='cuda:0')
            if receipt is not None:
                load_pretrained(pretrained_run,pretrained_sha256,g,d,expected=handoff_expected)
            engine=JointEngine(g,d,s,g_lr=g_lr,d_lr=d_lr,s_lr=s_lr,arch_lr=arch_lr,l1_weight=l1_weight)
        with (output/'steps.jsonl').open('x',encoding='utf-8') as log:
            for index in range(steps):
                torch.cuda.synchronize()
                # Previous inputs were deleted; release only unused own allocator cache.
                # Live model/optimizer tensors remain counted by the unchanged guard.
                torch.cuda.empty_cache()
                snap=gpu_guard(gpu,expected_uuid=initial['gpu_uuid'],**guard_args)
                snap['applied_allocator_fraction']=allocator_ceiling
                snap['cache_policy']='synchronize_then_empty_cache; no_retry'
                final_audit()
                tx,ty,train_record=cohort.sample('train',rng,scale)
                if phase=='gan_pretrain':
                    arrays=[torch.from_numpy(np.ascontiguousarray(a)).to('cuda:0') for a in (tx,ty)]
                    metrics=engine.step(*arrays)
                    entry={'step':index+1,'phase':phase,'train':train_record}
                else:
                    vx,vy,val_record=cohort.sample('validation',rng,scale)
                    fy=ty.copy(); flip_y=bool(rng.integers(2)); flip_x=bool(rng.integers(2))
                    if flip_y: fy=np.flip(fy,-2)
                    if flip_x: fy=np.flip(fy,-1)
                    arrays=[torch.from_numpy(np.ascontiguousarray(a)).to('cuda:0') for a in (tx,ty,vx,vy,fy)]
                    metrics=engine.step(*arrays)
                    entry={'step':index+1,'phase':phase,'train':train_record,'internal_validation':val_record,
                           'fake_mask':{'source_role':'train','flip_y':flip_y,'flip_x':flip_x}}
                torch.cuda.synchronize()
                entry.update(resources_before=snap,cuda_peak_allocated=torch.cuda.max_memory_allocated(),**metrics)
                log.write(json.dumps(entry,allow_nan=False)+'\n'); log.flush()
                print(json.dumps(entry,allow_nan=False),flush=True)
                del arrays
        audit(target,inventory)
        write_json(output/'protocol.json',protocol)
        # Serialize RNG and weights without any old checkpoint model load.
        saver=save_pretrain if phase=='gan_pretrain' else save_checkpoint
        saver(output,engine,protocol,{'numpy':rng.bit_generator.state,'python':random.getstate(),
            'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state_all()},
            final_audit=final_audit)
    except Exception as exc:
        # A failed final audit never writes the complete receipt.
        write_json(output/'failed.json',{'complete':False,'error_type':type(exc).__name__,'error':str(exc),
            'completed_steps':0 if engine is None else engine.steps,
            'diagnostics':{} if engine is None else getattr(engine,'last_diagnostics',{})})
        raise
    return output
