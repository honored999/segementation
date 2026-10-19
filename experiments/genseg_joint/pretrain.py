"""Completed GAN-phase handoff. No interrupted-run resume or segmenter weights."""
from pathlib import Path
import math
import re
from experiments.genseg30.contract import require, read_json, write_json, sha256

CLASSIFICATION='engineering_genseg_gd_pretrain_not_formal'
REPAIR='adam_trainbn_epoch_v1'
STRATEGIES=('legacy',REPAIR)


def policy(*,g_lr=2e-5,d_lr=2e-5,l1_weight=100.,strategy="legacy"):
    from .models import source_identity
    require(strategy in STRATEGIES,'Unknown pretraining strategy')
    result={'phase':'gan_pretrain','classification':CLASSIFICATION,'source':source_identity(),
        'rates':{'g':g_lr,'d':d_lr},'l1_weight':l1_weight,
        'optimizer':'stateless_SGD_no_momentum_decay','BN':'eval_frozen_buffers',
        'architecture':'both_alpha_frozen','normalization':'inner61_only_valid_pixel_minmax_minus1_plus1_no_clip',
        'pad':'native_full_planes_bottom_right_to_multiple256; image0_label-1_no_resize',
        'selection':'fixed_budget_last_complete; no_inner15_external19_arrays; no_resume'}
    if strategy==REPAIR:
        require(g_lr==d_lr==2e-5 and l1_weight==100.,'Repair uses fixed Adam rates and L1')
        result.update(strategy=REPAIR,protocol_version=2,optimizer='live_Adam_betas0.5_0.999_eps1e-8_no_decay',
            BN='train_live_buffers_G1_D3_per_layer_per_step',
            actual_order='G_once_D_fake_for_G_then_D_real_D_detached_fake; both_gradients_preupdate; commit_G_then_D',
            sampling='inner61_seeded_epoch_without_replacement_valid_planes_only; all_ignore_excluded; batch1',
            failure='live_mutation_nonatomic; exception_poisons_engine_no_retry_no_complete',
            source_fidelity='controlled_engineering_combination_not_proven_gray_image_cause_or_paper_reproduction',
            joint_handoff='rejected_until_stage_B_learning_gate_and_stage_C_implementation_review')
    return result


def validate_protocol(protocol):
    require(isinstance(protocol,dict),'Invalid pretrain protocol')
    require(isinstance(protocol.get('rates'),dict) and set(protocol['rates'])=={'g','d'},'Invalid rates')
    strategy=protocol.get('strategy','legacy')
    require(strategy in STRATEGIES,'Unknown pretraining strategy')
    if strategy=='legacy': require('protocol_version' not in protocol,'Legacy cannot claim repair version')
    for key,value in policy(strategy=strategy,**{k+'_lr':protocol.get('rates',{}).get(k,0) for k in ('g','d')},
                            l1_weight=protocol.get('l1_weight',-1)).items():
        require(protocol.get(key)==value,'Pretrain policy mismatch: '+key)
    require(type(protocol.get('steps')) is int and protocol['steps']>0,'Invalid pretrain step budget')
    if strategy==REPAIR:
        require(type(protocol.get('epochs')) is int and protocol['epochs']>0 and
                type(protocol.get('max_steps')) is int and protocol['max_steps']>0,'Explicit repair epochs/max_steps required')
        progress=protocol.get('epoch_sampling')
        validate_progress(progress,protocol['steps'])
        require(protocol['steps']==min(protocol['max_steps'],protocol['epochs']*progress['total_valid_planes']),
                'Repair budget/count mismatch')
    require(type(protocol.get('seed')) is int and 0<=protocol['seed']<2**32,'Seed must be uint32')
    require(all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in protocol['rates'].values()),'Invalid rates')
    require(isinstance(protocol['l1_weight'],(int,float)) and math.isfinite(protocol['l1_weight']) and protocol['l1_weight']>=0,'Invalid L1')
    require(isinstance(protocol.get('model_config'),dict),'Missing model config')
    require(isinstance(protocol.get('split'),dict) and set(protocol['split'])=={'train','validation','external'},'Missing split')
    scale=protocol.get('scaling',{})
    require(isinstance(scale,dict) and scale.get('method')=='inner61_valid_pixel_global_minmax_to_minus1_plus1_no_clip' and
        scale.get('fit_ids')==protocol['split']['train'] and
        all(isinstance(scale.get(k),(int,float)) and math.isfinite(scale[k]) for k in ('min','max')) and
        scale['max']>scale['min'],'Invalid inner61 scaling')
    require(isinstance(protocol.get('target_identity'),dict) and
        isinstance(protocol.get('target_manifest_sha256'),str) and
        re.fullmatch('[0-9a-f]{64}',protocol['target_manifest_sha256']) is not None,'Invalid target identity')


def audit_pretrained(run,expected_sha256,*,expected,for_joint=False):
    """Metadata and streamed hash only: no torch.load, model or patient arrays."""
    require(isinstance(expected_sha256,str) and re.fullmatch('[0-9a-f]{64}',expected_sha256) is not None,'Explicit lowercase checkpoint SHA256 required')
    run=Path(run).resolve(); receipt=read_json(run/'complete.json')
    require(receipt.get('format')==1 and receipt.get('phase')=='gan_pretrain' and
        receipt.get('classification')==CLASSIFICATION and receipt.get('complete') is True and
        receipt.get('checkpoint')=='gan.pt','Completed GAN pretraining receipt required; no resume/joint checkpoint')
    validate_protocol(receipt.get('protocol'))
    if for_joint:
        require(receipt['protocol'].get('strategy','legacy')=='legacy',
            'Repair pretraining cannot enter legacy joint/check: stage B learning gate and stage C adaptation are pending')
    require(receipt.get('steps')==receipt['protocol']['steps'],'Pretrain count mismatch')
    for k,v in expected.items(): require(receipt['protocol'].get(k)==v,'Pretrain protocol mismatch: '+k)
    require(receipt.get('checkpoint_sha256')==expected_sha256==sha256(run/'gan.pt'),'Pretrain checkpoint hash mismatch')
    return receipt


def save_pretrain(output,engine,protocol,rng,*,sampler=None,final_audit=None):
    import torch
    validate_protocol(protocol)
    require(not getattr(engine,'poisoned',False),'Cannot save poisoned engine')
    require(getattr(engine,'strategy','legacy')==protocol.get('strategy','legacy'),'Engine strategy mismatch')
    require(engine.steps==protocol['steps'],'Only completed pretraining may be handed off')
    require(engine.rates==protocol['rates'] and engine.l1_weight==protocol['l1_weight'],'Engine policy mismatch')
    payload={'format':1,'phase':'gan_pretrain','classification':CLASSIFICATION,'steps':engine.steps,
        'G':engine.g.net.state_dict(),'D':engine.d.state_dict(),
        'arch':{k:getattr(engine.g.networks,k).detach().cpu().clone() for k in ('conv_arch','upconv_arch')},
        'optimizers':engine.optimizer_state(),'protocol':protocol,'rng':rng}
    if protocol.get('strategy')==REPAIR:
        require(sampler is not None,'Repair checkpoint requires epoch sampler state')
        payload['sampler']=sampler
        validate_training_state(payload)
    path=Path(output)/'gan.pt'
    with path.open('xb') as stream: torch.save(payload,stream)
    if final_audit is not None: final_audit()
    write_json(Path(output)/'complete.json',{'format':1,'phase':'gan_pretrain','classification':CLASSIFICATION,
        'complete':True,'steps':engine.steps,'checkpoint':'gan.pt','checkpoint_sha256':sha256(path),'protocol':protocol})


def load_pretrained(run,expected_sha256,generator,discriminator,*,expected,for_joint=False):
    """Validate all tensors privately, then copy; never restore RNG/optimizer/S."""
    import torch
    receipt=audit_pretrained(run,expected_sha256,expected=expected,for_joint=for_joint)
    # Hash and deserialize from the same open file; recheck bytes after loading.
    path=Path(run)/'gan.pt'
    def stream_hash(stream):
        import hashlib
        digest=hashlib.sha256()
        for chunk in iter(lambda:stream.read(1024*1024),b''): digest.update(chunk)
        return digest.hexdigest()
    with path.open('rb') as stream:
        require(stream_hash(stream)==expected_sha256,'Pretrain checkpoint changed')
        stream.seek(0); payload=torch.load(stream,map_location='cpu',weights_only=True)
        stream.seek(0)
        require(stream_hash(stream)==expected_sha256,'Pretrain checkpoint changed')
    validate_payload_schema(payload,receipt)
    require(payload['protocol']['model_config']['G']==generator.config,'Generator configuration mismatch')
    validate_training_state(payload,generator=generator,discriminator=discriminator)
    def validate(values,reference,name):
        require(isinstance(values,dict) and values.keys()==reference.keys(),'State keys mismatch: '+name)
        for k,v in values.items():
            r=reference[k]
            require(isinstance(v,torch.Tensor) and v.layout==torch.strided and v.shape==r.shape and
                v.dtype==r.dtype and bool(torch.isfinite(v).all()),'Invalid/nonfinite tensor: '+name+'.'+k)
    validate(payload['G'],generator.net.state_dict(),'G')
    validate(payload['D'],discriminator.state_dict(),'D')
    validate(payload['arch'],{k:getattr(generator.networks,k) for k in ('conv_arch','upconv_arch')},'arch')
    # Stage every device transfer before live mutation, including architecture aliases.
    staged=[]
    for net,key in ((generator.net,'G'),(discriminator,'D')):
        staged.append((net,{k:v.to(net.state_dict()[k].device).clone() for k,v in payload[key].items()}))
    arch={k:v.to(generator.device).clone().requires_grad_(False) for k,v in payload['arch'].items()}
    with torch.no_grad():
        for net,values in staged:
            for k,v in net.state_dict().items(): v.copy_(values[k])
    n=generator.networks
    n.conv_arch=arch['conv_arch']; n.upconv_arch=arch['upconv_arch']
    n._conv_arch_parameters=[n.conv_arch]; n._upconv_arch_parameters=[n.upconv_arch]
    n._arch_parameters=[n.upconv_arch,n.conv_arch]
    generator.pretrain_strategy=payload['protocol'].get('strategy','legacy')
    return receipt


def validate_progress(progress,steps):
    require(isinstance(progress,dict),'Missing epoch progress')
    total=progress.get('total_valid_planes')
    require(type(total) is int and total>0,'Invalid valid-plane denominator')
    position=(steps-1)%total+1 if steps else 0
    require(progress=={'actual_steps':steps,'total_valid_planes':total,'completed_epochs':steps//total,
        'epoch_progress':{'epoch':max(1,(steps-1)//total+1),'planes':position,'total':total},
        'unique_planes_seen':min(steps,total),'unique_coverage':min(steps,total)/total},'Epoch progress mismatch')


def validate_payload_schema(payload,receipt):
    keys={'format','phase','classification','steps','G','D','arch','optimizers','protocol','rng'}
    if receipt['protocol'].get('strategy')==REPAIR: keys.add('sampler')
    require(isinstance(payload,dict) and set(payload)==keys,'Invalid GAN checkpoint schema')
    require(all(payload[k]==receipt[k] for k in ('format','phase','classification','steps','protocol')),'Checkpoint/receipt mismatch')


def validate_training_state(payload,*,generator=None,discriminator=None):
    """Audit real Adam ownership/moments/steps and live BN before any restore.

    Preview uses saved G/D tensor maps only; no optimizer or D is constructed.
    Legacy artifacts keep their actual stateless SGD identity.
    """
    import torch
    protocol=payload['protocol']; steps=payload['steps']; opts=payload['optimizers']
    if protocol.get('strategy','legacy')=='legacy':
        expected={k:{'algorithm':'stateless_SGD','lr':v,'momentum':0.,'weight_decay':0.,'steps':steps,'state':{}}
                  for k,v in protocol['rates'].items()}
        require(opts==expected,'Pretrain optimizer policy/count mismatch')
        require(isinstance(payload['rng'],dict),'Missing pretrain RNG record')
        return
    require(protocol.get('strategy')==REPAIR and isinstance(opts,dict) and set(opts)=={'g','d'},'Invalid repair optimizer schema')
    for key,net,forwards in (('g',None if generator is None else generator.net,1),('d',discriminator,3)):
        values=payload[key.upper()]
        require(isinstance(values,dict) and bool(values),'Missing repair model state')
        require(all(isinstance(v,torch.Tensor) and v.layout==torch.strided and bool(torch.isfinite(v).all())
                    for v in values.values()),'Invalid/nonfinite repair model tensor')
        counts=[k for k in values if k.endswith('.num_batches_tracked')]
        require(bool(counts),'Repair requires tracked BN buffers')
        for name in counts:
            prefix=name.removesuffix('num_batches_tracked'); counter=values[name]
            mean=values.get(prefix+'running_mean'); var=values.get(prefix+'running_var')
            require(counter.dtype==torch.int64 and counter.shape==torch.Size([]) and int(counter)==steps*forwards,
                    'Repair per-layer BN count mismatch')
            require(isinstance(mean,torch.Tensor) and isinstance(var,torch.Tensor) and mean.shape==var.shape and
                    mean.ndim==1 and mean.dtype==var.dtype==torch.float32 and bool((var>=0).all()),'Invalid BN statistics')
        names=[k for k in values if not k.endswith(('.running_mean','.running_var','.num_batches_tracked'))]
        if net is not None:
            require(names==list(dict(net.named_parameters())),'Adam model parameter ownership mismatch')
        opt=opts[key]
        require(isinstance(opt,dict) and set(opt)=={'algorithm','lr','betas','eps','weight_decay','steps','parameter_names','state_dict'} and
            opt['algorithm']=='Adam' and opt['lr']==2e-5 and tuple(opt['betas'])==(.5,.999) and
            opt['eps']==1e-8 and opt['weight_decay']==0. and opt['steps']==steps and opt['parameter_names']==names,
            'Invalid Adam policy/parameter ownership')
        state=opt['state_dict']
        require(isinstance(state,dict) and set(state)=={'state','param_groups'} and isinstance(state['state'],dict) and
                isinstance(state['param_groups'],list) and len(state['param_groups'])==1,'Invalid Adam state schema')
        group=state['param_groups'][0]; ids=list(range(len(names)))
        require(isinstance(group,dict) and group.get('params')==ids and group.get('lr')==2e-5 and
            tuple(group.get('betas',()))==(.5,.999) and group.get('eps')==1e-8 and group.get('weight_decay')==0.,
            'Invalid Adam group/ownership')
        require(all(group.get(k) is False for k in ('amsgrad','maximize','capturable','differentiable')) and
                all(group.get(k) in (None,False) for k in ('foreach','fused','decoupled_weight_decay')),
                'Unexpected Adam optimizer variant')
        require(set(state['state'])==set(ids),'Missing/extra Adam parameter moments')
        for index,name in enumerate(names):
            item=state['state'][index]; param=values[name]
            require(isinstance(item,dict) and set(item)=={'step','exp_avg','exp_avg_sq'},'Invalid Adam moments schema')
            counter=item['step']
            require(isinstance(counter,torch.Tensor) and counter.shape==torch.Size([]) and
                counter.dtype in (torch.float32,torch.float64) and bool(torch.isfinite(counter)) and float(counter)==steps,
                'Adam step mismatch')
            for field in ('exp_avg','exp_avg_sq'):
                v=item[field]
                require(isinstance(v,torch.Tensor) and v.layout==torch.strided and v.shape==param.shape and
                    v.dtype==param.dtype and bool(torch.isfinite(v).all()) and
                    (field!='exp_avg_sq' or bool((v>=0).all())),'Invalid Adam moment tensor: '+name)
    sampler=payload.get('sampler'); rng=payload['rng']
    require(isinstance(sampler,dict) and set(sampler)=={'format','planes','order','position','steps','rng','progress'} and
        sampler['format']==1 and sampler['steps']==steps and sampler['progress']==protocol['epoch_sampling'],
        'Invalid saved epoch sampler')
    planes=sampler['planes']; total=protocol['epoch_sampling']['total_valid_planes']
    require(isinstance(planes,list) and len(planes)==total and
        all(isinstance(v,list) and len(v)==2 and v[0] in protocol['split']['train'] and type(v[1]) is int and v[1]>=0 for v in planes),
        'Invalid sampler planes/roles')
    require(planes==sorted(planes) and len({tuple(v) for v in planes})==total and
        isinstance(sampler['order'],list) and all(type(v) is int for v in sampler['order']) and
        sorted(sampler['order'])==list(range(total)) and sampler['position']==protocol['epoch_sampling']['epoch_progress']['planes'],
        'Invalid sampler permutation/cursor')
    require(isinstance(rng,dict) and set(rng)=={'numpy','python','torch_cpu','torch_cuda'} and sampler['rng']==rng['numpy'],
            'Missing/mismatched repair RNG state')
    import numpy as np
    import random
    try:
        np.random.default_rng().bit_generator.state=rng['numpy']
        random.Random().setstate(rng['python'])
    except (TypeError,ValueError,KeyError) as exc:
        raise ValueError('Invalid repair RNG state') from exc
    require(isinstance(rng['torch_cuda'],list),'Invalid CUDA RNG record')
    require(all(isinstance(v,torch.Tensor) and v.dtype==torch.uint8 and v.ndim==1 and v.numel()>0
                for v in [rng['torch_cpu'],*rng['torch_cuda']]),'Invalid torch RNG record')
