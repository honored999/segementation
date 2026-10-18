"""Completed GAN-phase handoff. No interrupted-run resume or segmenter weights."""
from pathlib import Path
import math
import re
from experiments.genseg30.contract import require, read_json, write_json, sha256

CLASSIFICATION='engineering_genseg_gd_pretrain_not_formal'


def policy(*,g_lr=2e-5,d_lr=2e-5,l1_weight=100.):
    from .models import source_identity
    return {'phase':'gan_pretrain','classification':CLASSIFICATION,'source':source_identity(),
        'rates':{'g':g_lr,'d':d_lr},'l1_weight':l1_weight,
        'optimizer':'stateless_SGD_no_momentum_decay','BN':'eval_frozen_buffers',
        'architecture':'both_alpha_frozen','normalization':'inner61_only_valid_pixel_minmax_minus1_plus1_no_clip',
        'pad':'native_full_planes_bottom_right_to_multiple256; image0_label-1_no_resize',
        'selection':'fixed_budget_last_complete; no_inner15_external19_arrays; no_resume'}


def validate_protocol(protocol):
    require(isinstance(protocol,dict),'Invalid pretrain protocol')
    require(isinstance(protocol.get('rates'),dict) and set(protocol['rates'])=={'g','d'},'Invalid rates')
    for key,value in policy(**{k+'_lr':protocol.get('rates',{}).get(k,0) for k in ('g','d')},
                            l1_weight=protocol.get('l1_weight',-1)).items():
        require(protocol.get(key)==value,'Pretrain policy mismatch: '+key)
    require(type(protocol.get('steps')) is int and protocol['steps']>0,'Invalid pretrain step budget')
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


def audit_pretrained(run,expected_sha256,*,expected):
    """Metadata and streamed hash only: no torch.load, model or patient arrays."""
    require(isinstance(expected_sha256,str) and re.fullmatch('[0-9a-f]{64}',expected_sha256) is not None,'Explicit lowercase checkpoint SHA256 required')
    run=Path(run).resolve(); receipt=read_json(run/'complete.json')
    require(receipt.get('format')==1 and receipt.get('phase')=='gan_pretrain' and
        receipt.get('classification')==CLASSIFICATION and receipt.get('complete') is True and
        receipt.get('checkpoint')=='gan.pt','Completed GAN pretraining receipt required; no resume/joint checkpoint')
    validate_protocol(receipt.get('protocol'))
    require(receipt.get('steps')==receipt['protocol']['steps'],'Pretrain count mismatch')
    for k,v in expected.items(): require(receipt['protocol'].get(k)==v,'Pretrain protocol mismatch: '+k)
    require(receipt.get('checkpoint_sha256')==expected_sha256==sha256(run/'gan.pt'),'Pretrain checkpoint hash mismatch')
    return receipt


def save_pretrain(output,engine,protocol,rng,*,final_audit=None):
    import torch
    validate_protocol(protocol)
    require(engine.steps==protocol['steps'],'Only completed pretraining may be handed off')
    require(engine.rates==protocol['rates'] and engine.l1_weight==protocol['l1_weight'],'Engine policy mismatch')
    payload={'format':1,'phase':'gan_pretrain','classification':CLASSIFICATION,'steps':engine.steps,
        'G':engine.g.net.state_dict(),'D':engine.d.state_dict(),
        'arch':{k:getattr(engine.g.networks,k).detach().cpu().clone() for k in ('conv_arch','upconv_arch')},
        'optimizers':engine.optimizer_state(),'protocol':protocol,'rng':rng}
    path=Path(output)/'gan.pt'
    with path.open('xb') as stream: torch.save(payload,stream)
    if final_audit is not None: final_audit()
    write_json(Path(output)/'complete.json',{'format':1,'phase':'gan_pretrain','classification':CLASSIFICATION,
        'complete':True,'steps':engine.steps,'checkpoint':'gan.pt','checkpoint_sha256':sha256(path),'protocol':protocol})


def load_pretrained(run,expected_sha256,generator,discriminator,*,expected):
    """Validate all tensors privately, then copy; never restore RNG/optimizer/S."""
    import torch
    receipt=audit_pretrained(run,expected_sha256,expected=expected)
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
    require(isinstance(payload,dict) and set(payload)=={'format','phase','classification','steps','G','D','arch','optimizers','protocol','rng'},'Invalid GAN checkpoint schema')
    require(all(payload[k]==receipt[k] for k in ('format','phase','classification','steps','protocol')),'Checkpoint/receipt mismatch')
    require(payload['protocol']['model_config']['G']==generator.config,'Generator configuration mismatch')
    expected_opt={k:{'algorithm':'stateless_SGD','lr':v,'momentum':0.,'weight_decay':0.,'steps':payload['steps'],'state':{}}
                  for k,v in payload['protocol']['rates'].items()}
    require(payload['optimizers']==expected_opt,'Pretrain optimizer policy/count mismatch')
    require(isinstance(payload['rng'],dict),'Missing pretrain RNG record')
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
    return receipt
