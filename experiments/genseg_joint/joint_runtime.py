"""Explicit new joint CLI orchestration; never selected by legacy defaults."""
import json
import random
import time
from pathlib import Path
import subprocess
import numpy as np
from experiments.genseg30.contract import require,check_output,write_json,sha256,capture_identities,assert_identities
from .joint_repair import RepairJointEngine,STRATEGY,CLASSIFICATION
from .joint_checkpoint import POLICY,audit_transition,load_transition,save
from .joint_data import InputSnapshot,PositiveSampler,cache_plan
from .joint_resources import ResourceGuard
from .runtime import implementation_identity


def check(target,pretrained_run,pretrained_sha256,cache_budget_bytes,loader_workspace_bytes,model_reserve_bytes,*,nnunet=False):
    from .data import inner_split
    audit_transition(target,pretrained_run,pretrained_sha256)
    split=inner_split(target.train,target.val,20261018)
    require(type(model_reserve_bytes) is int and model_reserve_bytes>0,'Explicit positive model reserve required')
    plan=cache_plan(target,split['train']+split['validation'],loader_workspace_bytes)
    estimate=plan['final_cache_estimated_bytes']
    require(estimate<=cache_budget_bytes,'Snapshot estimate exceeds explicit RAM budget')
    extra={}
    if nnunet:
        from .nnunet_s import from_source,padded_shape,STRATEGY as nn_strategy
        descriptor=from_source(target.original)
        shapes={p:list(padded_shape(descriptor,v['shape'][-2:])) for p,v in plan['cases'].items()}
        extra=dict(s_descriptor=descriptor,padded_shapes=shapes)
    return dict(**extra,strategy=nn_strategy if nnunet else STRATEGY,cache_estimated_bytes=estimate,cache_budget_bytes=cache_budget_bytes,
        **{k:v for k,v in plan.items() if k!='cases'},model_reserve_bytes=model_reserve_bytes,
        startup_required_bytes=estimate+plan['max_case_load_transient_bytes']+model_reserve_bytes,
        model_peak='UNKNOWN; explicit reserve and live guard required; no peak guarantee',
        arrays_loaded=False,weights_loaded=False,training_started=False)


def run(target,output,*,pretrained_run,pretrained_sha256,steps=None,epochs=None,gpu,cache_budget_bytes,model_reserve_bytes,loader_workspace_bytes,nnunet=False,resume_from=None,resume_sha256=None):
    require((steps is None)!=(epochs is None),'Choose exactly one of steps or epochs')
    budget=steps if epochs is None else epochs
    require(type(budget) is int and budget>0,'Explicit positive unit/epoch budget required')
    require(type(model_reserve_bytes) is int and model_reserve_bytes>0,'Explicit conservative model/optimizer/higher-order RAM reserve required')
    from . import joint_resume as recovery
    enabled=nnunet and epochs is not None
    require((resume_from is None)==(resume_sha256 is None),'Paired recovery path/hash required')
    require(resume_from is None or enabled,'Recovery only supports nnU-Net epochs')
    require(not enabled or epochs<=80,'Epoch recovery total budget must be <=80')
    guard=ResourceGuard(gpu)
    if resume_from is not None: guard(4*Path(resume_from).stat().st_size+model_reserve_bytes)
    parent=None if resume_from is None else recovery.load(resume_from,resume_sha256)
    if parent is not None:
        require(parent['planned_epochs']==epochs and parent['completed_epochs']<epochs,'Recovery total budget mismatch/completed')
    admission=check(target,pretrained_run,pretrained_sha256,cache_budget_bytes,loader_workspace_bytes,model_reserve_bytes,nnunet=nnunet)
    descriptor=admission.get("s_descriptor")
    output=check_output(output,list(target.protected)+[Path(pretrained_run)]+([] if resume_from is None else [Path(resume_from).resolve().parent]))
    initial=guard(model_reserve_bytes)
    import torch
    require(not torch.cuda.is_initialized(),'Binding must precede CUDA initialization')
    if parent is not None:
        require(parent['payload']['protocol']['initial_resources']['gpu_uuid']==initial['gpu_uuid'],'Recovery GPU UUID mismatch')
    guard.bind()
    require(torch.cuda.is_available(),'CUDA required for future production run')
    torch.cuda.set_per_process_memory_fraction(initial['allocator_fraction'],0)
    target.require_no_unpack_writes()
    from .data import Cohort,EpochSampler
    from .models import create_models,source_identity
    identities=capture_identities([Path(pretrained_run)/'gan.pt',Path(pretrained_run)/'complete.json'])
    prehash={p:sha256(p) for p in identities}
    started=time.perf_counter()
    snapshot=InputSnapshot(Cohort(target,20261018),budget_bytes=cache_budget_bytes,
        resource_guard=lambda remaining:guard(int(remaining)+model_reserve_bytes),
        loader_workspace_bytes=loader_workspace_bytes)
    cohort=snapshot.cohort
    scale=cohort.fit_scale(skip_all_ignore=True) if parent is None else parent['payload']['protocol']['scaling']
    sampler=PositiveSampler(cohort)
    real_sampler=None if epochs is None else EpochSampler(cohort,np.random.default_rng(20261018))
    if real_sampler is not None:
        steps=epochs*len(real_sampler.planes)
    source=source_identity()
    code=implementation_identity()
    from .joint_checkpoint import nn_policy
    protocol=dict(nn_policy(recovery=enabled) if nnunet else POLICY,steps=steps,split=cohort.split,scaling=scale,source=source,
        target_identity=target.identity,target_manifest_sha256=target.manifest_sha256,
        model_config={'G':{'width':64,'downs':8,'seed':20261018},'S':descriptor if nnunet else 'pinned_author_UNet(1,1,bilinear=False)'},
        pretrained_sha256=pretrained_sha256,pretrained_run=str(Path(pretrained_run).resolve()),
        implementation_files_sha256=code,initial_resources=initial,
        repository_commit=subprocess.run(['git','rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip(),
        learning_quality_passed=False,model_reserve_bytes=model_reserve_bytes,
        cache_load_budget={k:v for k,v in snapshot.plan.items() if k!='cases'},
        inherited_preprocessing='old76_target_fill_and_original_grid_not_refit; outer19_development_not_independent_test',
        selection='last_complete_unit; no_external19_arrays_or_metrics; no_resume',
        small_lesions='positive_only_not_size_balanced; no_small_lesion_improvement_claim')
    if nnunet:
        protocol.update(run_purpose='engineering_unit_preflight' if epochs is None else 'controlled_epoch_trial_not_formal',
            s_initialization=descriptor['initialization'],s_normalizer=descriptor['normalizer'],
            s_parameter_count=descriptor['parameter_count'],padding='bottom_right_zero_image_ignore_label_lcm256_actual_strides',
            checkpoint_selection='final_only_no_best_no_early_stop_no_resume',model_peak='UNKNOWN')
        print(json.dumps(dict(S=descriptor['kind'],normalizer=descriptor['normalizer'],
            initialization=descriptor['initialization'],parameter_count=descriptor['parameter_count'],
            versions=descriptor['runtime']['versions'],epochs=epochs,units=steps,model_peak='UNKNOWN')),flush=True)
    if real_sampler is not None:
        protocol.update(epochs=epochs,total_valid_planes=len(real_sampler.planes),
            real_sampling='inner61_valid_planes_without_replacement_including_negatives',
            epoch_progress=real_sampler.progress()['epoch_progress'])
    lineage=[] if parent is None else parent['lineage']+[dict(parent_run=str(Path(resume_from).resolve().parent),checkpoint_sha256=resume_sha256,completed_epochs=parent['completed_epochs'],completed_units=parent['payload']['steps'])]
    if enabled:
        protocol.update(epoch_recovery=recovery.FORMAT,resume_lineage=lineage,
            selection='last_complete_unit; no_external19_arrays_or_metrics; epoch_resume_enabled',
            checkpoint_selection='final_only_no_best_no_early_stop; epoch_recovery_separate')
        print(json.dumps(dict(epoch_recovery=recovery.FORMAT,save_every_epochs=1,retain=2,total_epochs=epochs,resumed_units=0 if parent is None else parent['payload']['steps'])),flush=True)
    timing=dict(snapshot.timings,startup=time.perf_counter()-started,log=0.,checkpoint_measurement='final_wall_in_timing.json_not_known_before_serialization')
    def full_audit():
        snapshot.full_audit(); assert_identities(identities)
        require(prehash=={p:sha256(p) for p in identities},'Pretrained artifact changed')
        require(source==source_identity() and code==implementation_identity(),'Code/source changed')
    def light_audit():
        snapshot.lightweight_audit(); assert_identities(identities)
        require(source==source_identity(),'Pinned model source changed')
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'started.json',protocol)
    engine=None
    rng=np.random.default_rng(20261018); random.seed(20261018); torch.manual_seed(20261018)
    try:
        full_audit(); guard(model_reserve_bytes)
        g,d,s=create_models(seed=20261018,device='cuda:0',s_descriptor=descriptor) if nnunet else create_models(seed=20261018,device='cuda:0')
        if parent is None: load_transition(target,pretrained_run,pretrained_sha256,g,d,scale)
        engine=RepairJointEngine(g,d,s)
        if parent is not None:
            full_audit()
            require(parent['payload']['snapshot']['input_hashes']==snapshot.hashes,'Recovery input content mismatch')
            recovery.restore(engine,parent,real_sampler,sampler,rng,protocol)
            protocol['epoch_progress']=real_sampler.progress()['epoch_progress']
            del parent
        with (output/'steps.jsonl').open('x',encoding='utf-8') as log:
            for index in range(engine.steps,steps):
                unit=time.perf_counter()
                start=time.perf_counter(); light_audit(); audit_wall=time.perf_counter()-start
                start=time.perf_counter(); torch.cuda.synchronize(); resources=guard(phase='between_units'); resource_wall=time.perf_counter()-start
                start=time.perf_counter()
                tx,ty,tr=cohort.sample('train',rng,scale) if real_sampler is None else real_sampler.next(scale)
                vx,vy,vr=cohort.sample('validation',rng,scale)
                fy,fr=sampler.next(scale)
                arrays=[torch.from_numpy(np.ascontiguousarray(a)).to('cuda:0') for a in (tx,ty,vx,vy,fy)]
                sample_wall=time.perf_counter()-start
                if nnunet:
                    from .nnunet_s import pad_unit
                    arrays=pad_unit(arrays,descriptor)
                metrics=engine.step(*arrays)
                start=time.perf_counter(); torch.cuda.synchronize(); sync_wall=time.perf_counter()-start
                entry=dict(step=index+1,train=tr,internal_validation=vr,fake_mask=fr,resources_before=resources,
                    cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved(),
                    lightweight_audit_wall=audit_wall,resource_guard_wall=resource_wall,
                    sample_and_transfer_host_wall=sample_wall,unit_boundary_sync_wall=sync_wall,
                    unit_wall=time.perf_counter()-unit,**metrics)
                if real_sampler is not None:
                    progress=real_sampler.progress()
                    protocol['epoch_progress']=progress['epoch_progress']
                    entry.update(epochs=epochs,**progress)
                    ep=progress['epoch_progress']
                    prefix=f"Epoch {ep['epoch']}/{epochs} | {ep['planes']}/{ep['total']} | "
                else:
                    prefix=''
                print(f"{prefix}Unit {index+1}/{steps} | S loss {metrics['s_loss']:.5g} | Val loss {metrics['internal_validation_loss']:.5g}",flush=True)
                start=time.perf_counter(); log.write(json.dumps(entry,allow_nan=False)+'\n'); log.flush()
                timing['log']+=time.perf_counter()-start
                del arrays,metrics,entry
                if enabled and real_sampler.position==len(real_sampler.planes):
                    recovery.save_epoch(output,engine,protocol,recovery.capture_rng(rng,sampler,real_sampler),
                        snapshot=dict(input_hashes=snapshot.hashes,input_stats=snapshot.stats),timing=timing,
                        full_audit=full_audit,guard=lambda reserve:guard(reserve,phase='between_units'),lineage=lineage)
        full_audit()
        timing['startup_and_units']=time.perf_counter()-started
        start=time.perf_counter()
        save(output,engine,protocol,dict(numpy=rng.bit_generator.state,positive=sampler.state_dict(),
            python=random.getstate(),torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all()),
            snapshot=dict(input_hashes=snapshot.hashes,input_stats=snapshot.stats,nbytes=snapshot.nbytes,
                          estimated_bytes=snapshot.estimated_bytes,cache_budget_bytes=cache_budget_bytes),
            timing=timing,final_audit=full_audit)
        timing['checkpoint']=time.perf_counter()-start
        write_json(output/'timing.json',timing)
        write_json(output/'protocol.json',protocol)
    except BaseException as exc:
        # Completion is removed only if our own post-save output failed, never touched in old runs.
        try:
            complete=output/'complete.json'
            if complete.exists(): complete.unlink()
            write_json(output/'failed.json',dict(complete=False,error_type=type(exc).__name__,error=str(exc),
                resource_snapshot=getattr(exc,'resource_snapshot',None),
                completed_steps=0 if engine is None else engine.steps,
                failure_stage=None if engine is None else engine.failure_stage,
                poisoned=False if engine is None else engine.poisoned))
        except Exception as save_exc:
            try:
                note=f'Failure receipt save failed: {type(save_exc).__name__}: {save_exc}'
                add_note=getattr(exc,'add_note',None)
                if callable(add_note):
                    add_note(note)
                else:
                    exc.failure_receipt_save_error=note
            except BaseException:
                pass  # Best-effort diagnostics must never replace the original failure.

        raise
    return output
