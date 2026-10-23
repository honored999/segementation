"""Explicit plans-bound 1-logit PlainConvUNet adaptation, never a Trainer."""
from copy import deepcopy
import importlib.metadata
import inspect
import math
from pathlib import Path
import torch
from experiments.genseg30.contract import require, sha256

KIND='nnunet2d_plainconv_1logit_v1'
STRATEGY='joint_adam_positive_nnunet2d_sgdproxy_v1'
CLASSIFICATION='engineering_genseg_joint_nnunet2d_not_formal'
NETWORK='dynamic_network_architectures.architectures.unet.PlainConvUNet'


def runtime_source_files():
    import dynamic_network_architectures as dna
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    paths={Path(inspect.getfile(get_network_from_plans)),Path(inspect.getfile(PlansManager)),
        Path(inspect.getfile(torch.nn.InstanceNorm2d)),
        Path(inspect.getfile(torch.nn.functional))}
    root=Path(dna.__file__).parent
    files={'dna/'+str(p.relative_to(root)).replace('\\','/'):p
        for p in sorted(root.rglob('*.py'))}
    files.update({p.name:p for p in paths})
    return files


def runtime_identity():
    return dict(versions={k:importlib.metadata.version(k) for k in
        ('nnunetv2','dynamic-network-architectures','torch')},files={k:sha256(p) for k,p in runtime_source_files().items()})


def architecture(plans,dataset):
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    require(isinstance(plans,dict) and isinstance(dataset,dict),'Plans/dataset dictionaries required')
    require(set(dataset.get('channel_names',{}))=={'0'} and
        str(dataset['channel_names']['0']).lower()=='dwi','Single DWI input required')
    labels=dataset.get('labels',{})
    require(len(labels)==2 and all(type(v) is int for v in labels.values()) and
        set(labels.values())=={0,1},'Single-channel binary dataset required')
    cm=PlansManager(plans).get_configuration('2d')
    require(len(cm.patch_size)==2,'2D plans required')
    a=dict(network_class_name=cm.network_arch_class_name,arch_kwargs=deepcopy(cm.network_arch_init_kwargs),
        _kw_requires_import=list(cm.network_arch_init_kwargs_req_import))
    k=a['arch_kwargs']; norm=k.get('norm_op_kwargs') or {}
    require(a['network_class_name']==NETWORK and k.get('conv_op')=='torch.nn.modules.conv.Conv2d',
        'Only actual 2D PlainConvUNet plans supported')
    require(k.get('norm_op')=='torch.nn.modules.instancenorm.InstanceNorm2d' and
        norm.get('track_running_stats',False) is False,'InstanceNorm without running buffers required')
    require(k.get('dropout_op') is None,'Dropout plans unsupported by deterministic proxy contract')
    n=k.get('n_stages'); strides=k.get('strides')
    require(type(n) is int and n>=2 and isinstance(strides,list) and len(strides)==n and
        all(isinstance(s,(list,tuple)) and len(s)==2 and all(type(v) is int and v>0 for v in s) for s in strides)
        and strides[0]==[1,1], 'Explicit 2D stage strides required')
    require(all(key in a['_kw_requires_import'] for key in ('conv_op','norm_op','nonlin')),
        'Missing plans module import items')
    return a


def _factory(a):
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    return get_network_from_plans(a['network_class_name'],deepcopy(a['arch_kwargs']),
        a['_kw_requires_import'],1,1,allow_init=True,deep_supervision=False)


def _layout(a):
    # Metadata-only construction: no production-size tensors or forward on CPU.
    with torch.device('meta'): model=_factory(a)
    parameters=dict(model.named_parameters()); owners={id(v):k for k,v in parameters.items()}
    aliases={k:owners[id(v)] for k,v in model.named_parameters(remove_duplicate=False) if k!=owners[id(v)]}
    inactive=[k for k in parameters if any(k.startswith('decoder.seg_layers.'+str(i)+'.')
        for i in range(len(model.decoder.seg_layers)-1))]
    require(not list(model.named_buffers()),'Unexpected S buffers')
    normalizers={k:dict(eps=m.eps,momentum=m.momentum,affine=m.affine,track_running_stats=m.track_running_stats)
        for k,m in model.named_modules() if isinstance(m,torch.nn.InstanceNorm2d)}
    return dict(normalizers=normalizers,parameter_names=list(parameters),aliases=aliases,inactive_parameters=inactive,
        tensors={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in model.state_dict().items()},
        parameter_count=sum(v.numel() for v in parameters.values()))


def describe(plans,dataset,source_identity):
    a=architecture(plans,dataset)
    require(all(isinstance(source_identity.get(k),str) and len(source_identity[k])==64
        for k in ('plans_sha256','dataset_sha256','splits_sha256')),'Bound SOURCE hashes required')
    return dict(kind=KIND,plans=deepcopy(plans),dataset=deepcopy(dataset),configuration='2d',
        architecture=a,source_identity=deepcopy(source_identity),runtime=runtime_identity(),
        input_channels=1,output_channels=1,deep_supervision=False,initialization='official_factory_random_He_no_S_weights',
        normalizer='plans_InstanceNorm2d_track_running_stats_false_no_S_BN_or_buffers',
        **_layout(a))


def validate_descriptor(d):
    require(isinstance(d,dict) and d.get('kind')==KIND,'nnU-Net S kind required')
    expected=describe(d['plans'],d['dataset'],d['source_identity'])
    require(d==expected,'nnU-Net plans/norm/runtime/layout descriptor mismatch')
    return d


def from_source(source):
    return describe(source.plans,source.dataset_json,source.identity)


def audit_norm(model,d):
    require(not list(model.named_buffers()),'nnU-Net S forbids BN/running/unexpected buffers')
    norms=[m for m in model.modules() if isinstance(m,torch.nn.modules.instancenorm._InstanceNorm)]
    require(norms and all(type(m) is torch.nn.InstanceNorm2d and not m.track_running_stats for m in norms),
        'Unexpected S InstanceNorm contract')
    require(not any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()),'Unexpected S BN')
    observed={k:dict(eps=m.eps,momentum=m.momentum,affine=m.affine,track_running_stats=m.track_running_stats)
        for k,m in model.named_modules() if isinstance(m,torch.nn.InstanceNorm2d)}
    require(observed==d['normalizers'],'S InstanceNorm configuration changed')
    require(list(dict(model.named_parameters()))==d['parameter_names'],'S parameter ownership changed')
    owners={id(v):k for k,v in model.named_parameters()}
    aliases={k:owners[id(v)] for k,v in model.named_parameters(remove_duplicate=False) if k!=owners[id(v)]}
    require(aliases==d['aliases'],'S shared parameter aliases changed')


def build(d,device='cpu'):
    validate_descriptor(d)
    model=_factory(d['architecture']).to(device)
    model.genseg_s_descriptor=deepcopy(d)
    model.genseg_inactive_parameters=tuple(d['inactive_parameters'])
    audit_norm(model,d)
    return model


def validate_state(values,d):
    validate_descriptor(d)
    require(isinstance(values,dict) and values.keys()==d['tensors'].keys(),'S state keys mismatch')
    for k,spec in d['tensors'].items():
        v=values[k]
        require(isinstance(v,torch.Tensor) and v.layout==torch.strided and v.device.type!='meta' and
            list(v.shape)==spec['shape'] and str(v.dtype)==spec['dtype'] and bool(torch.isfinite(v).all()),
            'S tensor shape/dtype/finite mismatch: '+k)
    require(all(torch.equal(values[k],values[owner]) for k,owner in d['aliases'].items()),'S shared alias mismatch')


def restore(values,d):
    validate_state(values,d)
    model=build(d); model.load_state_dict(values,strict=True)
    return model.eval().requires_grad_(False)


def padded_shape(d,shape,*,multiple=256):
    strides=d['architecture']['arch_kwargs']['strides']
    factors=[math.prod(s[i] for s in strides) for i in range(2)]
    multiples=[math.lcm(multiple,f) for f in factors]
    padded=tuple(((int(v)+m-1)//m)*m for v,m in zip(shape,multiples))
    require(len(shape)==2 and all(int(v)>0 for v in shape),'Positive 2D shape required')
    require(math.prod(v//f for v,f in zip(padded,factors))>1,'Deepest InstanceNorm spatial size must exceed one')
    return padded


def pad_unit(arrays,d):
    import torch.nn.functional as F
    result=[]
    for index,t in enumerate(arrays):
        h,w=padded_shape(d,t.shape[-2:])
        result.append(F.pad(t,(0,w-t.shape[-1],0,h-t.shape[-2]),value=-1 if index in (1,3,4) else 0))
    return result
