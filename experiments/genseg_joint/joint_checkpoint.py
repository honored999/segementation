"""Strict weights-only phase transition and preview-only joint schema v2."""
from pathlib import Path
import math
import re
import torch
from experiments.genseg30.contract import require, read_json, write_json, sha256
from .pretrain import audit_pretrained, load_pretrained, policy, REPAIR
from .models import source_identity
from .joint_repair import STRATEGY, CLASSIFICATION

PRETRAIN_SHA = '1547b42e65a7f21e8452d3509ce862f059884dd7e2bb82047eea7b33b7a5b75b'


def expected_pretrain(target):
    from .data import inner_split
    return dict(policy(strategy=REPAIR), seed=20261018,
        split=inner_split(target.train,target.val,20261018),target_identity=target.identity,
        target_manifest_sha256=target.manifest_sha256,
        model_config={'G':{'width':64,'downs':8,'seed':20261018},'D_width':64})


def audit_transition(target,run,digest):
    require(digest == PRETRAIN_SHA, 'New joint requires the pinned completed epoch20 checkpoint')
    receipt = audit_pretrained(run,digest,expected=expected_pretrain(target))
    p = receipt['protocol']
    require(p['epochs'] == 20 and receipt['steps'] == 19400 and
        p['epoch_sampling']['total_valid_planes'] == 970, 'Expected completed 20 epoch/19400 update transition')
    require(p['scaling']['min'] == -1.1009682416915894 and p['scaling']['max'] == 8.59938907623291,
            'Epoch20 scaling mismatch')
    return receipt


def load_transition(target,run,digest,g,d,scale):
    receipt = audit_transition(target,run,digest)
    require(scale == receipt['protocol']['scaling'], 'Snapshot scale differs from pretrained inner61')
    expected = dict(expected_pretrain(target),scaling=scale)
    # Full pretrain schema/moments/BN/source validation remains active. Only G/D and alpha copied.
    return load_pretrained(run,digest,g,d,expected=expected,for_joint=False)


POLICY = dict(strategy=STRATEGY,phase='joint',classification=CLASSIFICATION,
    seed=20261018, actual_optimizer='fresh_Adam_G2e-5_D2e-5_S1e-5_betas.5_.999_eps1e-8_decay0',
    virtual_optimizer='one_step_SGD_G2e-5_S1e-5_eval_BN_private_entry_snapshot_approximation',
    alpha_optimizer='fresh_staged_Adam_1e-4_upconv_only_conv_frozen',
    sampling='real_unchanged_patient_plane_including_negative; independent_uniform_inner61_positive_xy_flips; inner15_architecture_only',
    transition='weights_only_G_D_BN_both_alpha; fresh_author_UNet_S; fresh_all_moments_RNG_sampler; no_resume',
    audit_policy='startup_postcache_final_full_SHA; unit_stat_and_code_SHA; private_immutable_snapshot; stat_not_content_equivalent',
    fidelity='controlled_adaptation_not_exact_Adam_derivative_not_paper_reproduction; quality_gate_not_passed',
    timing_policy='stage_host_enqueue_wall; CUDA_execution_and_sync_charged_to_unit_boundary')


def validate_protocol(p):
    require(isinstance(p,dict),'Joint protocol required')
    for k,v in POLICY.items(): require(p.get(k)==v,'Repair joint protocol mismatch: '+k)
    require(type(p.get('steps')) is int and p['steps']>0,'Positive completed joint count required')
    require(p.get('source')==source_identity(),'Pinned source mismatch')
    split=p.get('split',{})
    require(set(split)=={'train','validation','external'} and
        [len(split[k]) for k in ('train','validation','external')]==[61,15,19] and
        len(set(sum(split.values(),[])))==95,'Disjoint fixed roles required')
    scale=p.get('scaling',{})
    require(scale.get('method')=='inner61_valid_pixel_global_minmax_to_minus1_plus1_no_clip' and
        scale.get('fit_ids')==split['train'] and all(type(scale.get(k)) in (int,float) and
        math.isfinite(scale[k]) for k in ('min','max')) and scale['max']>scale['min'],'Invalid scale')
    require(p.get('pretrained_sha256')==PRETRAIN_SHA,'Pinned transition hash required')
    require(p.get('model_config')=={'G':{'width':64,'downs':8,'seed':20261018},
        'S':'pinned_author_UNet(1,1,bilinear=False)'},'Joint model configuration mismatch')
    require(isinstance(p.get('target_identity'),dict) and re.fullmatch('[0-9a-f]{64}',p.get('target_manifest_sha256','')),
        'Target identity required')
    require(isinstance(p.get('implementation_files_sha256'),dict) and p['implementation_files_sha256'] and
        all(re.fullmatch('[0-9a-f]{64}',v) for v in p['implementation_files_sha256'].values()),'Code identity required')


def validate_payload(payload,receipt):
    require(isinstance(payload,dict) and set(payload)=={'format','classification','steps','G','D','S','arch',
        'model_config','optimizers','protocol','rng','source','initial_bn','timing','snapshot'},'Joint v2 schema mismatch')
    require(payload['format']==2,'Joint v2 format required')
    for k in ('format','classification','steps','protocol'):
        require(payload[k]==receipt[k],'Joint receipt mismatch: '+k)
    p=payload['protocol']; validate_protocol(p)
    require(payload['source']==p['source'] and payload['model_config']==p['model_config'],'Joint identity mismatch')
    require(payload['classification']==CLASSIFICATION,'Joint classification mismatch')
    steps=payload['steps']; opts=payload['optimizers']
    require(isinstance(opts,dict) and set(opts)=={'g','d','s','alpha'},'Joint optimizers missing')
    for key,rate,count in (('g',2e-5,1),('d',2e-5,3),('s',1e-5,2)):
        values=payload[key.upper()]
        require(isinstance(values,dict) and values and all(isinstance(v,torch.Tensor) and
            v.layout==torch.strided and bool(torch.isfinite(v).all()) for v in values.values()),'Invalid model tensors')
        bn={k:int(v) for k,v in values.items() if k.endswith('num_batches_tracked')}
        initial=payload['initial_bn'].get(key.upper(),{})
        require(bn and bn.keys()==initial.keys() and all(bn[k]==initial[k]+steps*count for k in bn),
            'Joint BN counter mismatch')
        expected_initial={'g':19400,'d':58200,'s':0}[key]
        require(all(v==expected_initial for v in initial.values()),'Pretrain/fresh S BN origin mismatch')
        for name in bn:
            counter=values[name]; prefix=name.removesuffix('num_batches_tracked')
            mean,var=values.get(prefix+'running_mean'),values.get(prefix+'running_var')
            require(counter.dtype==torch.int64 and counter.ndim==0 and isinstance(mean,torch.Tensor) and
                isinstance(var,torch.Tensor) and mean.shape==var.shape and mean.ndim==1 and
                mean.dtype==var.dtype==torch.float32 and bool((var>=0).all()),'Invalid joint BN buffers')
        names=[k for k in values if not k.endswith(('running_mean','running_var','num_batches_tracked'))]
        opt=opts[key]
        require(set(opt)=={'algorithm','lr','betas','eps','weight_decay','steps','parameter_names','state_dict'} and
            opt['algorithm']=='Adam' and opt['steps']==steps and opt['lr']==rate and tuple(opt['betas'])==(.5,.999)
            and opt['eps']==1e-8 and opt['weight_decay']==0 and opt['parameter_names']==names,'Joint Adam policy mismatch')
        validate_adam(opt['state_dict'],[values[k] for k in names],steps,rate)
    arch=payload['arch']
    require(isinstance(arch,dict) and set(arch)=={'conv_arch','upconv_arch'} and all(isinstance(v,torch.Tensor)
        and v.layout==torch.strided and bool(torch.isfinite(v).all()) for v in arch.values()),'Invalid joint alpha')
    require(set(opts['alpha'])=={'algorithm','steps','state_dict'} and opts['alpha']['algorithm']=='staged_Adam'
        and opts['alpha']['steps']==steps,'Alpha optimizer mismatch')
    validate_adam(opts['alpha']['state_dict'],[arch['upconv_arch']],steps,1e-4)
    require(isinstance(payload['rng'],dict) and set(payload['rng'])=={'numpy','positive','python','torch_cpu','torch_cuda'},
        'Joint RNG/sampler progress required')
    positive=payload['rng']['positive']
    require(isinstance(positive,dict) and positive.get('steps')==steps and type(positive.get('positive_planes')) is int
        and positive['positive_planes']>0,'Positive sampler count mismatch')
    require(isinstance(payload['timing'],dict) and isinstance(payload['snapshot'],dict),'Timing/snapshot identity required')


def validate_adam(state,params,steps,rate):
    require(isinstance(state,dict) and set(state)=={'state','param_groups'} and len(state['param_groups'])==1,
        'Invalid Adam state')
    group=state['param_groups'][0]; ids=list(range(len(params)))
    require(group.get('params')==ids and group.get('lr')==rate and tuple(group.get('betas',()))==(.5,.999)
        and group.get('eps')==1e-8 and group.get('weight_decay')==0. and
        all(group.get(k) is False for k in ('amsgrad','maximize','capturable','differentiable')) and
        all(group.get(k) in (None,False) for k in ('foreach','fused','decoupled_weight_decay')),
        'Invalid Adam group/ownership')
    require(set(state['state'])==set(ids),'Invalid Adam moment ownership')
    for i,param in enumerate(params):
        item=state['state'][i]
        require(set(item)=={'step','exp_avg','exp_avg_sq'},'Invalid Adam moment schema')
        counter=item['step']
        require(isinstance(counter,torch.Tensor) and counter.ndim==0 and bool(torch.isfinite(counter))
            and float(counter)==steps,'Adam step mismatch')
        for k in ('exp_avg','exp_avg_sq'):
            v=item[k]
            require(isinstance(v,torch.Tensor) and v.shape==param.shape and v.dtype==param.dtype and
                bool(torch.isfinite(v).all()) and (k!='exp_avg_sq' or bool((v>=0).all())),'Invalid Adam moments')


def save(output,engine,protocol,rng,*,snapshot,timing,final_audit):
    require(not engine.poisoned and engine.steps==protocol['steps'],'Incomplete/poisoned engine')
    payload=dict(format=2,classification=CLASSIFICATION,steps=engine.steps,
        G=engine.g.net.state_dict(),D=engine.d.state_dict(),S=engine.s.state_dict(),
        arch={k:getattr(engine.g.networks,k).detach().cpu().clone() for k in ('conv_arch','upconv_arch')},
        model_config=protocol['model_config'],optimizers=engine.optimizer_state(),protocol=protocol,
        rng=rng,source=protocol['source'],initial_bn=engine.initial_bn,timing=timing,snapshot=snapshot)
    receipt=dict(format=2,classification=CLASSIFICATION,steps=engine.steps,protocol=protocol,
        complete=True,checkpoint='joint.pt')
    validate_payload(payload,receipt)
    final_audit()
    path=Path(output)/'joint.pt'
    with path.open('xb') as stream: torch.save(payload,stream)
    final_audit()
    receipt['checkpoint_sha256']=sha256(path)
    write_json(Path(output)/'complete.json',receipt)


def audit_joint(target,run,digest):
    from .data import inner_split
    from experiments.genseg30.contract import checked_file,capture_identities,assert_identities
    receipt_path=checked_file(run,'complete.json'); checkpoint=checked_file(run,'joint.pt')
    identities=capture_identities([receipt_path,checkpoint])
    r=read_json(receipt_path)
    require(r.get('format')==2 and r.get('complete') is True and r.get('checkpoint')=='joint.pt' and
        r.get('classification')==CLASSIFICATION,'Completed repair joint receipt required')
    validate_protocol(r['protocol'])
    p=r['protocol']
    require(r.get('steps')==p['steps'] and p['split']==inner_split(target.train,target.val,20261018), 'Joint split/count mismatch')
    require(p['target_identity']==target.identity and p['target_manifest_sha256']==target.manifest_sha256,'Joint target mismatch')
    require(re.fullmatch('[0-9a-f]{64}',digest) and r.get('checkpoint_sha256')==digest==sha256(checkpoint),'Joint hash mismatch')
    target.verify_identity(target.identity); assert_identities(identities)
    return r,identities
