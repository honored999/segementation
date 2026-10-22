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


def check(target,pretrained_run,pretrained_sha256,cache_budget_bytes,loader_workspace_bytes,model_reserve_bytes):
    from .data import inner_split
    audit_transition(target,pretrained_run,pretrained_sha256)
    split=inner_split(target.train,target.val,20261018)
    require(type(model_reserve_bytes) is int and model_reserve_bytes>0,'Explicit positive model reserve required')
    plan=cache_plan(target,split['train']+split['validation'],loader_workspace_bytes)
    estimate=plan['final_cache_estimated_bytes']
    require(estimate<=cache_budget_bytes,'Snapshot estimate exceeds explicit RAM budget')
    return dict(strategy=STRATEGY,cache_estimated_bytes=estimate,cache_budget_bytes=cache_budget_bytes,
        **{k:v for k,v in plan.items() if k!='cases'},model_reserve_bytes=model_reserve_bytes,
        startup_required_bytes=estimate+plan['max_case_load_transient_bytes']+model_reserve_bytes,
        model_peak='UNKNOWN; explicit reserve and live guard required; no peak guarantee',
        arrays_loaded=False,weights_loaded=False,training_started=False)


def run(target,output,*,pretrained_run,pretrained_sha256,steps=None,epochs=None,gpu,cache_budget_bytes,model_reserve_bytes,loader_workspace_bytes):
    require((steps is None)!=(epochs is None),'Choose exactly one of steps or epochs')
    budget=steps if epochs is None else epochs
    require(type(budget) is int and budget>0,'Explicit positive unit/epoch budget required')
    require(type(model_reserve_bytes) is int and model_reserve_bytes>0,'Explicit conservative model/optimizer/higher-order RAM reserve required')
    check(target,pretrained_run,pretrained_sha256,cache_budget_bytes,loader_workspace_bytes,model_reserve_bytes)
    output=check_output(output,list(target.protected)+[Path(pretrained_run)])
    guard=ResourceGuard(gpu)
    initial=guard(model_reserve_bytes)
    import torch
    require(not torch.cuda.is_initialized(),'Binding must precede CUDA initialization')
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
    scale=cohort.fit_scale(skip_all_ignore=True)
    sampler=PositiveSampler(cohort)
    real_sampler=None if epochs is None else EpochSampler(cohort,np.random.default_rng(20261018))
    if real_sampler is not None:
        steps=epochs*len(real_sampler.planes)
    source=source_identity()
    code=implementation_identity()
    protocol=dict(POLICY,steps=steps,split=cohort.split,scaling=scale,source=source,
        target_identity=target.identity,target_manifest_sha256=target.manifest_sha256,
        model_config={'G':{'width':64,'downs':8,'seed':20261018},'S':'pinned_author_UNet(1,1,bilinear=False)'},
        pretrained_sha256=pretrained_sha256,pretrained_run=str(Path(pretrained_run).resolve()),
        implementation_files_sha256=code,initial_resources=initial,
        repository_commit=subprocess.run(['git','rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip(),
        learning_quality_passed=False,model_reserve_bytes=model_reserve_bytes,
        cache_load_budget={k:v for k,v in snapshot.plan.items() if k!='cases'},
        inherited_preprocessing='old76_target_fill_and_original_grid_not_refit; outer19_development_not_independent_test',
        selection='last_complete_unit; no_external19_arrays_or_metrics; no_resume',
        small_lesions='positive_only_not_size_balanced; no_small_lesion_improvement_claim')
    if real_sampler is not None:
        protocol.update(epochs=epochs,total_valid_planes=len(real_sampler.planes),
            real_sampling='inner61_valid_planes_without_replacement_including_negatives',
            epoch_progress=real_sampler.progress()['epoch_progress'])
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
        g,d,s=create_models(seed=20261018,device='cuda:0')
        load_transition(target,pretrained_run,pretrained_sha256,g,d,scale)
        engine=RepairJointEngine(g,d,s)
        with (output/'steps.jsonl').open('x',encoding='utf-8') as log:
            for index in range(steps):
                unit=time.perf_counter()
                start=time.perf_counter(); light_audit(); audit_wall=time.perf_counter()-start
                start=time.perf_counter(); torch.cuda.synchronize(); resources=guard(phase='between_units'); resource_wall=time.perf_counter()-start
                start=time.perf_counter()
                tx,ty,tr=cohort.sample('train',rng,scale) if real_sampler is None else real_sampler.next(scale)
                vx,vy,vr=cohort.sample('validation',rng,scale)
                fy,fr=sampler.next(scale)
                arrays=[torch.from_numpy(np.ascontiguousarray(a)).to('cuda:0') for a in (tx,ty,vx,vy,fy)]
                sample_wall=time.perf_counter()-start
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
