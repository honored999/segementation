"""Tiny real nnU-Net contracts; no production-capacity model or real data."""
from copy import deepcopy
import pytest
import torch
from experiments.genseg_joint.nnunet_s import describe, build, validate_state, audit_norm, padded_shape
from experiments.genseg_joint.tests.test_joint_repair import TinyG, proxy_path_object
from experiments.genseg_joint.joint_repair import RepairJointEngine


def descriptor():
    from experiments.genseg30.contract import read_json, REPO
    plans=read_json(REPO/'standalone_nnunet2d/reference/nnUNetPlans.json')
    k=plans['configurations']['2d']['architecture']['arch_kwargs']
    k.update(n_stages=3,features_per_stage=[2,3,4],kernel_sizes=[[3,3]]*3,
        strides=[[1,1],[2,2],[2,2]],n_conv_per_stage=[1]*3,n_conv_per_stage_decoder=[1]*2)
    return describe(plans,{'channel_names':{'0':'DWI'},'labels':{'background':0,'lesion':1}},
        {'plans_sha256':'a'*64,'dataset_sha256':'b'*64,'splits_sha256':'c'*64})


def engine():
    torch.set_num_threads(1); torch.manual_seed(42)
    g=TinyG()
    original=g.net.forward
    def forward(x):
        # Fixed spatial context avoids the affine scale invariance of IN when
        # the old 1x1 fixture G receives a binary mask.
        context=torch.sin(torch.arange(x.shape[-2]*x.shape[-1],device=x.device,dtype=x.dtype)).reshape(1,1,*x.shape[-2:])
        return original(x)+.2*context
    g.net.forward=forward
    return RepairJointEngine(g,torch.nn.Sequential(torch.nn.BatchNorm2d(2),torch.nn.Conv2d(2,1,1)),build(descriptor()))


def batch():
    x=torch.linspace(-.7,.8,64).reshape(1,1,8,8)
    y=torch.zeros_like(x); y[:,:,2:6,2:6]=1; y[:,:,:1]=-1
    return x,y,x*.7,y.clone(),y.flip(-1)


def test_factory_norm_alias_and_strides():
    from dynamic_network_architectures.architectures.unet import PlainConvUNet
    d=descriptor(); s=build(d)
    assert type(s) is PlainConvUNet and s.decoder.encoder is s.encoder
    assert s(torch.ones(1,1,8,12)).shape==(1,1,8,12)
    assert not list(s.named_buffers()) and d['aliases']
    assert padded_shape(d,(7,11))==(256,256)
    from experiments.genseg_joint.nnunet_s import pad_unit
    args=batch(); padded=pad_unit(args,d)
    assert all(v.shape==(1,1,256,256) for v in padded)
    for i,v in enumerate(padded):
        torch.testing.assert_close(v[:,:,:8,:8],args[i],rtol=0,atol=0)
        assert (v[:,:,8:,:]==(-1 if i in (1,3,4) else 0)).all()
    norm=next(m for m in s.modules() if isinstance(m,torch.nn.InstanceNorm2d))
    norm.eps*=2
    with pytest.raises(ValueError,match='configuration'): audit_norm(s,d)
    norm.eps/=2
    with pytest.raises(ValueError,match='InstanceNorm'): padded_shape(d,(4,4),multiple=4)
    broken=deepcopy(d); broken['architecture']['arch_kwargs']['norm_op']='torch.nn.BatchNorm2d'
    with pytest.raises(ValueError): build(broken)
    s.register_buffer('unexpected',torch.ones(1))
    with pytest.raises(ValueError,match='buffers'): audit_norm(s,d)


def test_actual_adam_private_proxy_and_unused_heads():
    e=engine(); args=batch(); before=deepcopy(e.s.state_dict())
    e.hyperloss(*args,e.g.networks.upconv_arch)
    assert all(torch.equal(v,before[k]) for k,v in e.s.state_dict().items()), [(k,float((v-before[k]).abs().max())) for k,v in e.s.state_dict().items() if not torch.equal(v,before[k])]
    for _ in range(3):
        result=e.step(*args)
        assert result['bn_updates']==dict(G=1,D=3,S=0)
    assert e.steps==3 and not e.initial_bn['S']
    opt=e.optimizer_state()['s']
    assert len(opt['parameter_names'])==len(list(e.s.parameters()))
    assert len(opt['state_dict']['state'])<len(opt['parameter_names'])


@pytest.mark.parametrize('mode',['direct','indirect'])
def test_real_nnunet_proxy_frozen_anchor_fd(monkeypatch,mode):
    e=engine(); args=tuple(v.double() for v in batch())
    for _,n in e.networks(): n.double()
    f,_=proxy_path_object(e,args,monkeypatch,mode)
    a=torch.tensor(.3,dtype=torch.double,requires_grad=True); eps=1e-3
    grad,=torch.autograd.grad(f(a),a)
    fd=(f((a.detach()+eps).requires_grad_())-f((a.detach()-eps).requires_grad_()))/(2*eps)
    assert abs(grad.item())>1e-12
    assert abs((grad-fd).item())<.01*abs(grad.item())
    torch.testing.assert_close(grad,fd,rtol=.002,atol=1e-11)
    print(mode,grad.item(),fd.item())


def test_indirect_sever_negative(monkeypatch):
    e=engine(); args=tuple(v.double() for v in batch())
    for _,n in e.networks(): n.double()
    f,_=proxy_path_object(e,args,monkeypatch,'indirect',sever=True)
    a=torch.tensor(.3,dtype=torch.double,requires_grad=True)
    grad,=torch.autograd.grad(f(a),a,allow_unused=True)
    assert grad is None


def test_state_alias_roundtrip_and_support(tmp_path):
    from experiments.genseg_joint.nnunet_s import restore
    from experiments import genseg_segment as seg
    import numpy as np
    d=descriptor(); s=build(d).eval(); path=tmp_path/'s.pt'
    torch.save(s.state_dict(),path)
    values=torch.load(path,weights_only=True); r=restore(values,d)
    x=torch.randn(1,1,8,12)
    with torch.no_grad(): torch.testing.assert_close(s(x),r(x),rtol=0,atol=0)
    bad=deepcopy(values); alias=next(iter(d['aliases'])); bad[alias]=bad[alias]+1
    with pytest.raises(ValueError,match='alias'): validate_state(bad,d)
    vol=np.ones((2,7,11),np.float32); support=np.ones(vol.shape,bool); support[:,:,0]=False
    pred=seg.infer_prepared_volume(r,vol,valid_support=support,pad_multiple=4)
    assert pred.shape==vol.shape and not pred[~support].any()


def test_schema3_complete_save_load_and_rejections(tmp_path):
    from types import SimpleNamespace
    from experiments import genseg_segment as seg
    from experiments.genseg_joint import joint_checkpoint as cp
    from experiments.genseg_joint.tests.test_segment import receipt_fixture
    from experiments.genseg30.contract import read_json, sha256
    from experiments.genseg_joint.preview import load_checkpoint
    e=engine(); d=e.s_descriptor
    for key,n in e.networks():
        initial={'G':19400,'D':58200,'S':0}[key]
        for k,v in n.state_dict().items():
            if k.endswith('num_batches_tracked'): v.fill_(initial); e.initial_bn[key][k]=initial
    for _ in range(3): e.step(*batch())
    fixture=tmp_path/'meta'; fixture.mkdir(); _,old=receipt_fixture(fixture)
    p=dict(old['protocol'],**cp.nn_policy()); p['steps']=3
    p['target_identity'].update(d['source_identity']); p['model_config']['S']=d
    rng=dict(numpy={},positive=dict(steps=3,positive_planes=1),python=(),torch_cpu=torch.get_rng_state(),torch_cuda=[])
    out=tmp_path/'new'; out.mkdir()
    cp.save(out,e,p,rng,snapshot={},timing={},final_audit=lambda:None)
    r=read_json(out/'complete.json'); payload=load_checkpoint(out,'joint.pt',r['checkpoint_sha256'])
    cp.validate_payload(payload,r,nnunet=True)
    assert r['format']==3
    with pytest.raises(ValueError): cp.validate_payload(payload,r)
    with pytest.raises(ValueError): seg.audit_receipt(out,nnunet=True) # three units cannot pass final80 gate
    # Synthetic completion metadata only: no claim to have trained 80 epochs.
    for obj in (payload,r): obj['steps']=77600
    p['steps']=77600; payload['protocol']=p; r['protocol']=p
    payload['rng']['positive']['steps']=77600
    for key,origin,count in (('G',19400,1),('D',58200,3)):
        for name in payload['initial_bn'][key]: payload[key][name].fill_(origin+77600*count)
    for opt in payload['optimizers'].values():
        opt['steps']=77600
        for state in opt['state_dict']['state'].values(): state['step'].fill_(77600)
    torch.save(payload,out/'joint.pt'); r['checkpoint_sha256']=sha256(out/'joint.pt')
    import json
    (out/'complete.json').write_text(json.dumps(r))
    target=SimpleNamespace(train=p['split']['train']+p['split']['validation'],val=p['split']['external'],
        identity=p['target_identity'],manifest_sha256=p['target_manifest_sha256'],verify_identity=lambda i:None)
    restored,_=seg.load_audited_segmenter(out,r['checkpoint_sha256'],target=target,nnunet=True)
    e.s.eval()
    with torch.no_grad(): torch.testing.assert_close(e.s(batch()[0]),restored(batch()[0]),rtol=0,atol=0)
    with pytest.raises(ValueError): seg.load_audited_segmenter(out,r['checkpoint_sha256'],target=target)
    for fault in ('alias','norm','kind','plan','moment','ownership','step','alpha','source','dtype','shape','nan'):
        bad=deepcopy(payload); receipt=deepcopy(r)
        if fault=='alias':
            alias=next(iter(d['aliases'])); bad['S'][alias]=bad['S'][alias]+1
        elif fault=='norm': bad['S']['running_mean']=torch.zeros(1)
        elif fault in ('dtype','shape','nan'):
            key=d['parameter_names'][0]
            if fault=='dtype': bad['S'][key]=bad['S'][key].double()
            elif fault=='shape': bad['S'][key]=bad['S'][key].reshape(-1)
            else: bad['S'][key].fill_(float('nan'))
        elif fault in ('kind','plan'):
            desc=bad['protocol']['model_config']['S']
            if fault=='kind': desc['kind']='author_unet'
            else: desc['plans']['configurations']['2d']['architecture']['arch_kwargs']['features_per_stage'][0]+=1
        elif fault=='moment': bad['optimizers']['s']['state_dict']['state'][0]['exp_avg_sq'].fill_(-1)
        elif fault=='ownership': bad['optimizers']['s']['parameter_names'].reverse()
        elif fault=='step': bad['optimizers']['s']['state_dict']['state'][0]['step'].fill_(1)
        elif fault=='alpha': bad['arch']['upconv_arch'].fill_(float('nan'))
        elif fault=='source': bad['protocol']['target_identity']['plans_sha256']='0'*64
        receipt['protocol']=bad['protocol']
        with pytest.raises(ValueError): cp.validate_payload(bad,receipt,nnunet=True)


def test_nnunet_s_exact19_prediction_support_geometry_and_scoring(tmp_path,monkeypatch):
    from experiments import genseg_segment as seg
    from experiments.genseg_joint.tests.test_segment import prediction_fixture
    binding,images,_=prediction_fixture(tmp_path,monkeypatch)
    model=build(descriptor()).eval().requires_grad_(False)
    binding.receipt['format']=3
    monkeypatch.setattr(model,'to',lambda *a,**kw:model) # CPU hardware substitution only
    calls=[]
    def load(*a,**kw):
        calls.append(kw); assert kw['nnunet'] is True
        return model,{}
    monkeypatch.setattr(seg,'load_audited_segmenter',load)
    out=tmp_path/'pred'; cache=tmp_path/'cache'
    r=seg.predict(binding,tmp_path/'run',out,cache,model_reserve_bytes=1,case_reserve_bytes=1)
    assert len(seg.verify_predictions(out,r,images))==19 and calls
    for case in r['cases'].values():
        image,support=seg.load_prepared(case)
        assert support.dtype.name=='bool'
    assert seg.state_digest(model)==r['state_before']==r['state_after']
    gt=tmp_path/'gt'; gt.mkdir()
    for f in out.glob('*.nii.gz'): (gt/f.name).write_bytes(f.read_bytes())
    assert seg.accept_for_scoring(out,images,gt,tmp_path/'metrics')==tmp_path/'metrics'
    broken=deepcopy(r); broken['external_ids'].pop()
    with pytest.raises(ValueError): seg.verify_predictions(out,broken,images)


def test_explicit_new_runtime_check_three_units_save_preview_cpu(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    import blosc2
    from experiments.genseg_joint.tests.test_joint_repair import snapshot_target
    from experiments.genseg_joint import joint_runtime as rt,models,preview,nnunet_s
    from experiments.genseg_joint.joint_checkpoint import PRETRAIN_SHA
    from experiments.genseg30.contract import read_json,sha256
    target=snapshot_target(tmp_path); d=descriptor()
    target.original.plans=d['plans']; target.original.dataset_json=d['dataset']; target.original.identity=d['source_identity']
    target.identity=d['source_identity']; target.manifest_sha256='a'*64
    target.require_no_unpack_writes=lambda:None
    original_load=target.load_train
    def load(*a):
        x,y,props=original_load(*a)
        return np.tile(x,(1,1,2,2)),np.tile(y,(1,1,2,2)),props
    target.load_train=load
    mask=tmp_path/'mask.npz'; np.savez_compressed(mask,keep=np.ones((2,8,8),bool))
    for patient in target.train:
        target.manifest['cases'][patient]['source_shape']=[1,2,8,8]
        for suffix,dtype in (('.b2nd',np.float32),('_seg.b2nd',np.int16)):
            blosc2.asarray(np.zeros((1,2,8,8),dtype),urlpath=str(tmp_path/(patient+suffix)),mode='w')
    pre=tmp_path/'pre'; pre.mkdir(); (pre/'gan.pt').write_bytes(b'fixture'); (pre/'complete.json').write_text('{}')
    monkeypatch.setattr(rt,'audit_transition',lambda *a:None)
    class Guard:
        def __init__(self,*a): pass
        def __call__(self,*a,**kw): return dict(allocator_fraction=.1)
        def bind(self): pass
    monkeypatch.setattr(rt,'ResourceGuard',Guard)
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:False)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:False)
    for name in ('set_per_process_memory_fraction','synchronize'):
        monkeypatch.setattr(torch.cuda,name,lambda *a:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda:0)
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[])
    original_to=torch.Tensor.to
    monkeypatch.setattr(torch.Tensor,'to',lambda self,*a,**kw:original_to(self,*(('cpu',)+a[1:] if a and a[0]=='cuda:0' else a),**kw))
    def create(**kw):
        assert kw['s_descriptor']==d
        e=engine(); return e.g,e.d,e.s
    monkeypatch.setattr(models,'create_models',create)
    def transition(target,run,digest,g,disc,scale):
        for net,initial in ((g.net,19400),(disc,58200)):
            for name,v in net.state_dict().items():
                if name.endswith('num_batches_tracked'): v.fill_(initial)
    monkeypatch.setattr(rt,'load_transition',transition)
    import experiments.genseg_joint.data as data
    monkeypatch.setattr(data,'pad_plane',lambda a,*args:a.copy())
    # Padding arithmetic is tested separately; keep this real higher-order CPU
    # run at 8x8 rather than the production 256+ full-plane transport boundary.
    monkeypatch.setattr(nnunet_s,'pad_unit',lambda arrays,desc:arrays)
    check=rt.check(target,pre,PRETRAIN_SHA,100000,1024,1024,nnunet=True)
    assert check['s_descriptor']==d and not check['arrays_loaded']
    out=tmp_path/'joint'
    rt.run(target,out,pretrained_run=pre,pretrained_sha256=PRETRAIN_SHA,steps=3,gpu=1,
        cache_budget_bytes=100000,loader_workspace_bytes=1024,model_reserve_bytes=1024,nnunet=True)
    receipt=read_json(out/'complete.json')
    assert receipt['format']==3 and receipt['steps']==3
    assert receipt['protocol']['run_purpose']=='engineering_unit_preflight'
    monkeypatch.setattr(preview,'gpu_guard',lambda *a,**kw:dict(allocator_fraction=.1))
    monkeypatch.setattr(preview,'Generator',lambda **kw:object())
    monkeypatch.setattr(preview,'restore_generator',lambda *a:None)
    monkeypatch.setattr(preview,'generate',lambda g,y,s:(np.zeros_like(y,dtype=np.float32),list(y.shape)))
    previews=tmp_path/'preview'
    preview.run_preview(target,out,sha256(out/'joint.pt'),previews,gpu=1,patient_limit=2,phase='joint_nnunet')
    assert read_json(previews/'preview.json')['phase']=='joint_nnunet'
