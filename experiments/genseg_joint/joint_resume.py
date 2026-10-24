"""Epoch-only recovery container, deliberately separate from final receipts."""
from copy import deepcopy
from pathlib import Path
import os
import random
import re
import numpy as np
import torch
from experiments.genseg30.contract import require, sha256, read_json, write_json
from .joint_checkpoint import validate_payload, nn_policy

FORMAT = 'genseg_nnunet_epoch_resume_v1'

def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().to('cpu', copy=True)
    if isinstance(value, dict): return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    return deepcopy(value)

def optimizers(engine):
    result={}
    for key,net,opt,rate in [('g',engine.g.net,engine.gan.optimizers['g'],2e-5),
                            ('d',engine.d,engine.gan.optimizers['d'],2e-5),
                            ('s',engine.s,engine.s_optimizer,1e-5)]:
        params=list(net.parameters())
        require(len(opt.param_groups)==1 and len(opt.param_groups[0]['params'])==len(params) and
                all(a is b for a,b in zip(params,opt.param_groups[0]['params'])), 'Adam ownership changed')
        result[key]=dict(algorithm='Adam',lr=rate,betas=(.5,.999),eps=1e-8,weight_decay=0.,
            steps=engine.steps,parameter_names=list(dict(net.named_parameters())),state_dict=cpu_tree(opt.state_dict()))
    result['alpha']=dict(algorithm='staged_Adam',steps=engine.steps,state_dict=cpu_tree(engine._alpha_state))
    return result

def reserve_bytes(engine):
    # CPU payload plus validation/load workspace, including tied state aliases.
    tensors=[v for _,net in engine.networks() for v in net.state_dict().values()]
    for opt in [*engine.gan.optimizers.values(),engine.s_optimizer]:
        tensors += [v for state in opt.state.values() for v in state.values() if isinstance(v,torch.Tensor)]
    tensors += [v for state in engine._alpha_state['state'].values() for v in state.values() if isinstance(v,torch.Tensor)]
    return 3*sum(v.numel()*v.element_size() for v in tensors)+16*1024**2

def capture_rng(rng,positive,real):
    return dict(numpy=deepcopy(rng.bit_generator.state),positive=positive.state_dict(),real=real.state_dict(),
        numpy_global=(lambda st:(st[0],torch.from_numpy(st[1].astype(np.int64)),st[2],st[3],st[4]))(np.random.get_state()),
        python=random.getstate(),torch_cpu=torch.get_rng_state(),
        torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])

def validate(container):
    require(isinstance(container,dict) and set(container)=={'format','payload','planned_epochs','completed_epochs','modes','lineage'}, 'Recovery schema mismatch')
    require(container['format']==FORMAT, 'Recovery format mismatch')
    p=container['payload']; protocol=p['protocol']; steps=p['steps']
    epochs=container['planned_epochs']; completed=container['completed_epochs']
    require(type(epochs) is int and 0<epochs<=80 and type(completed) is int and 0<completed<=epochs, 'Recovery epoch budget mismatch')
    require(protocol.get('epoch_recovery')==FORMAT and protocol.get('resume_lineage')==container['lineage'],'Recovery protocol/lineage mismatch')
    total=protocol['total_valid_planes']
    require(type(total) is int and total>0 and protocol['epochs']==epochs and protocol['steps']==epochs*total and steps==completed*total, 'Recovery boundary mismatch')
    # Preserve strict historical final validator; this is a separate container.
    shadow=dict(p,rng={k:v for k,v in p['rng'].items() if k not in ('real','numpy_global')})
    validate_payload(shadow,{k:shadow[k] for k in ('format','classification','steps','protocol')},nnunet=True)
    r=p['rng']; require(set(r)=={'numpy','numpy_global','positive','real','python','torch_cpu','torch_cuda'}, 'Recovery RNG missing')
    real=r['real']; planes=real['planes']; order=real['order']
    require(real['format']==1 and len(planes)==total and planes==sorted(planes) and len({tuple(v) for v in planes})==total and
        all(isinstance(v,list) and len(v)==2 and v[0] in protocol['split']['train'] and type(v[1]) is int and v[1]>=0 for v in planes), 'Recovery planes invalid')
    require(all(type(v) is int for v in order) and sorted(order)==list(range(total)) and real['position']==total and
        real['steps']==steps and real['progress']['completed_epochs']==completed and
        real['progress']['epoch_progress']==protocol['epoch_progress'], 'Recovery incomplete epoch/permutation')
    positive=r['positive']
    require(positive['steps']==steps and positive['planes']==sorted(positive['planes']) and
        len(positive['planes'])==positive['positive_planes'] and len({tuple(v) for v in positive['planes']})==len(positive['planes']) and
        all(v in planes for v in positive['planes']), 'Recovery positive inventory invalid')
    try:
        for state in (r['numpy'],real['rng'],positive['rng']): np.random.default_rng().bit_generator.state=state
        random.Random().setstate(r['python'])
        state=r['numpy_global']; require(len(state)==5 and isinstance(state[1],torch.Tensor) and state[1].dtype==torch.int64,'NumPy global RNG invalid')
        np.random.RandomState().set_state((state[0],state[1].numpy().astype(np.uint32),*state[2:]))
        torch.Generator().set_state(r['torch_cpu'])
    except (TypeError,ValueError,RuntimeError,KeyError) as exc: raise ValueError('Recovery RNG invalid') from exc
    require(isinstance(r['torch_cuda'],list) and all(isinstance(v,torch.Tensor) and v.dtype==torch.uint8 and v.ndim==1 and v.numel()>0 for v in r['torch_cuda']), 'CUDA RNG invalid')
    require(set(container['modes'])=={'G','D','S'} and all(isinstance(v,dict) and all(type(m) is bool for m in v.values()) for v in container['modes'].values()), 'Recovery modes invalid')
    require(isinstance(container['lineage'],list) and all(isinstance(v,dict) and
        set(v)=={'parent_run','checkpoint_sha256','completed_epochs','completed_units'} and
        isinstance(v['parent_run'],str) and re.fullmatch('[0-9a-f]{64}',v['checkpoint_sha256']) and
        type(v['completed_epochs']) is int and 0<v['completed_epochs']<=completed and
        v['completed_units']==v['completed_epochs']*total for v in container['lineage']), 'Recovery lineage invalid')
    return container

def _hash_stream(stream):
    import hashlib
    digest=hashlib.sha256()
    for chunk in iter(lambda:stream.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def load(path,digest):
    path=Path(path)
    require(not path.is_symlink() and re.fullmatch('[0-9a-f]{64}',digest or ''), 'Recovery path/hash invalid')
    with path.open('rb') as stream:
        before=_hash_stream(stream)
        require(before==digest,'Recovery hash mismatch')
        stream.seek(0); result=torch.load(stream,map_location='cpu',weights_only=True)
        stream.seek(0); require(_hash_stream(stream)==before==sha256(path),'Recovery changed while reading')
    return validate(result)

def save_epoch(output,engine,protocol,rng,*,snapshot,timing,full_audit,guard,lineage):
    require(engine.s_descriptor is not None and not engine.poisoned and not engine.gan.poisoned and
        engine.failure_stage is None and engine.steps==engine.gan.steps==rng['real']['steps'], 'Recovery poisoned/incomplete engine')
    guard(reserve_bytes(engine))
    full_audit()
    payload=dict(format=3,classification=protocol['classification'],steps=engine.steps,
        **{key:cpu_tree(net.state_dict()) for key,net in engine.networks()},
        arch={k:cpu_tree(getattr(engine.g.networks,k)) for k in ('conv_arch','upconv_arch')},
        model_config=protocol['model_config'],optimizers=optimizers(engine),protocol=deepcopy(protocol),rng=cpu_tree(rng),
        source=protocol['source'],initial_bn=deepcopy(engine.initial_bn),timing=deepcopy(timing),snapshot=deepcopy(snapshot))
    container=dict(format=FORMAT,payload=payload,planned_epochs=protocol['epochs'],
        completed_epochs=engine.steps//protocol['total_valid_planes'],lineage=deepcopy(lineage),
        modes={key:{name:m.training for name,m in net.named_modules()} for key,net in engine.networks()})
    validate(container)
    root=Path(output).resolve(strict=True)
    require(not Path(output).is_symlink(), 'Recovery output symlink refused')
    name=f'resume_epoch_{container["completed_epochs"]:04d}.pt'
    path=root/name; temp=root/(name+'.tmp'); pointer=root/'resume_latest.json'; pointer_temp=root/'resume_latest.json.tmp'
    require(not any(p.exists() or p.is_symlink() for p in (path,temp,pointer_temp)), 'Recovery destination exists')
    prior=read_json(pointer) if pointer.exists() else {'history':[]}
    require(not pointer.is_symlink(), 'Recovery pointer symlink refused')
    with temp.open('xb') as stream:
        torch.save(container,stream); stream.flush(); os.fsync(stream.fileno())
    digest=sha256(temp); load(temp,digest); full_audit()
    os.replace(temp,path)
    history=prior['history']+[dict(checkpoint=name,sha256=digest)]
    write_json(pointer_temp,dict(format=FORMAT,history=history[-2:]))
    with pointer_temp.open('r+b') as stream: os.fsync(stream.fileno())
    os.replace(pointer_temp,pointer)
    # Only our exact checkpoint basenames, never recursive cleanup.
    for item in history[:-2]:
        old=item['checkpoint']
        require(re.fullmatch(r'resume_epoch_\d{4}\.pt',old) is not None,'Unsafe recovery cleanup')
        candidate=root/old
        require(not candidate.is_symlink() and candidate.resolve().parent==root, 'Recovery cleanup escaped root')
        require(sha256(candidate)==item['sha256'], 'Recovery cleanup identity changed')
        candidate.unlink()
    return path,digest

def restore(engine,container,real,positive,rng,protocol):
    validate(container); p=container['payload']; saved=p['protocol']
    for key in ('epochs','steps','split','scaling','source','target_identity','target_manifest_sha256','model_config',
                'pretrained_sha256','implementation_files_sha256','total_valid_planes'):
        require(saved[key]==protocol[key], 'Recovery binding mismatch: '+key)
    require(container['completed_epochs']<container['planned_epochs'], 'Recovery budget already completed')
    require(p['rng']['real']['planes']==[[a,b] for a,b in real.planes] and
            p['rng']['positive']['planes']==[[a,b] for a,b in positive.planes], 'Recovery inventory mismatch')
    for key,net in engine.networks():
        current=net.state_dict(); values=p[key]
        require(current.keys()==values.keys() and all(current[k].shape==v.shape and current[k].dtype==v.dtype for k,v in values.items()), 'Recovery model dtype/shape mismatch')
        require(set(container['modes'][key])==set(dict(net.named_modules())), 'Recovery module mode mismatch')
    with torch.no_grad():
        for key,net in engine.networks(): net.load_state_dict(p[key],strict=True)
        for key in ('conv_arch','upconv_arch'):
            live=getattr(engine.g.networks,key); value=p['arch'][key]
            require(live.shape==value.shape and live.dtype==value.dtype,'Recovery alpha dtype/shape mismatch')
            live.copy_(value)
    n=engine.g.networks
    require(n._arch_parameters[0] is n.upconv_arch and n._arch_parameters[1] is n.conv_arch and
            n._upconv_arch_parameters[0] is n.upconv_arch and n._conv_arch_parameters[0] is n.conv_arch, 'Recovery alpha alias mismatch')
    for key,opt in [('g',engine.gan.optimizers['g']),('d',engine.gan.optimizers['d']),('s',engine.s_optimizer)]: opt.load_state_dict(p['optimizers'][key]['state_dict'])
    engine._alpha_state=cpu_tree(p['optimizers']['alpha']['state_dict'])
    device=n.upconv_arch.device
    for state in engine._alpha_state['state'].values():
        for k,v in state.items():
            if isinstance(v,torch.Tensor) and k!='step': state[k]=v.to(device)
    engine.steps=engine.gan.steps=p['steps']; engine.initial_bn=deepcopy(p['initial_bn'])
    for key,net in engine.networks():
        for name,m in net.named_modules(): m.training=container['modes'][key][name]
    from .nnunet_s import audit_norm
    audit_norm(engine.s,engine.s_descriptor)
    real.load_state_dict(p['rng']['real']); positive.load_state_dict(p['rng']['positive'])
    # Last operation: construction/cache/audits must precede this RNG restoration.
    state=p['rng']['numpy_global']; np.random.set_state((state[0],state[1].numpy().astype(np.uint32),*state[2:]))
    rng.bit_generator.state=p['rng']['numpy']; random.setstate(p['rng']['python']); torch.set_rng_state(p['rng']['torch_cpu'])
    require(len(p['rng']['torch_cuda'])==(torch.cuda.device_count() if torch.cuda.is_initialized() else 0),'Recovery CUDA RNG device count mismatch')
    if p['rng']['torch_cuda']: torch.cuda.set_rng_state_all(p['rng']['torch_cuda'])
