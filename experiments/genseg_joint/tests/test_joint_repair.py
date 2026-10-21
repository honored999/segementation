from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.genseg_joint.joint_repair import RepairJointEngine
from experiments.genseg_joint.joint_data import PositiveSampler
from experiments.genseg_joint.core import JointEngine
from experiments.genseg_joint.data import Cohort


class TinyG:
    def __init__(self):
        self.networks = SimpleNamespace(upconv_arch=torch.tensor(.3),conv_arch=torch.tensor(.1))
        n = self.networks
        n._conv_arch_parameters=[n.conv_arch]; n._upconv_arch_parameters=[n.upconv_arch]
        n._arch_parameters=[n.upconv_arch,n.conv_arch]
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn=torch.nn.BatchNorm2d(1)
                self.conv=torch.nn.Conv2d(1,1,1)
            def forward(inner,x):
                return inner.conv(inner.bn(x))*(1+n.upconv_arch)
        self.net=Net()
        self.pretrain_strategy='adam_trainbn_epoch_v1'
    def restore_arch(self,arch):
        for k,v in arch.items(): setattr(self.networks,k,v.detach().clone())


def engine():
    torch.set_num_threads(1); torch.manual_seed(42)
    g=TinyG()
    d=torch.nn.Sequential(torch.nn.BatchNorm2d(2),torch.nn.Conv2d(2,1,1))
    s=torch.nn.Sequential(torch.nn.BatchNorm2d(1),torch.nn.Conv2d(1,1,1))
    return RepairJointEngine(g,d,s)


def batch():
    x=torch.linspace(-.7,.8,16).reshape(1,1,4,4)
    y=torch.zeros_like(x); y[:,:,1:3,1:3]=1; y[:,:,:1]=-1
    return x,y,x*.7,y.clone(),y.flip(-1)


def test_fresh_adam_multistep_bn_and_private_validation():
    e=engine(); args=batch()
    assert all(not o.state for o in [*e.gan.optimizers.values(),e.s_optimizer])
    with pytest.raises(ValueError,match='legacy'):
        JointEngine(e.g,e.d,e.s)
    for step in (1,2,3):
        before=[deepcopy(n.state_dict()) for _,n in e.networks()]
        modes=[m.training for _,n in e.networks() for m in n.modules()]
        rng=torch.get_rng_state().clone()
        e.hyperloss(*args,e.g.networks.upconv_arch)
        assert all(torch.equal(v,b[k]) for (_,n),b in zip(e.networks(),before) for k,v in n.state_dict().items())
        assert torch.equal(rng,torch.get_rng_state())
        assert modes==[m.training for _,n in e.networks() for m in n.modules()]
        result=e.step(*args)
        assert result['bn_updates']==dict(G=1,D=3,S=2)
        for key,n in e.networks():
            assert all(int(v)==step*dict(G=1,D=3,S=2)[key] for k,v in n.state_dict().items() if k.endswith('num_batches_tracked'))
        assert all(float(state['step'])==step for o in [*e.gan.optimizers.values(),e.s_optimizer] for state in o.state.values())
        assert e.g.networks._arch_parameters[0] is e.g.networks.upconv_arch
    assert e.steps==3


def test_proxy_hypergradient_finite_difference():
    e=engine(); args=tuple(v.double() for v in batch())
    for _,n in e.networks(): n.double()
    a=torch.tensor(.3,dtype=torch.double,requires_grad=True)
    grad,=torch.autograd.grad(e.hyperloss(*args,a),a)
    eps=1e-4
    fd=(e.hyperloss(*args,a.detach().clone().add(eps).requires_grad_())-
        e.hyperloss(*args,a.detach().clone().sub(eps).requires_grad_()))/(2*eps)
    assert abs(float(grad))>1e-12
    torch.testing.assert_close(grad,fd,atol=1e-8,rtol=1e-5)


def test_failure_poison_no_retry(monkeypatch):
    e=engine()
    monkeypatch.setattr(e.s_optimizer,'step',lambda: (_ for _ in ()).throw(RuntimeError('injected')))
    with pytest.raises(RuntimeError,match='injected'): e.step(*batch())
    assert e.poisoned and e.steps==0 and e.gan.steps==1 and e.failure_stage=='actual_S'
    with pytest.raises(ValueError,match='poisoned'): e.step(*batch())
    with pytest.raises(ValueError,match='poisoned'): e.optimizer_state()


def test_positive_rng_isolation_and_real_negatives():
    class Target:
        train=[f't{i}' for i in range(76)]; val=[f'e{i}' for i in range(19)]
        def training_dataset(self): return None
        def load_train(self,d,p):
            x=np.ones((1,2,4,4),np.float32); y=np.zeros_like(x,dtype=np.int16); y[:,1,1,1]=1
            return x,y,{}
    c=Cohort(Target(),20261018); scale=dict(min=0.,max=2.)
    rng=np.random.default_rng(71); saved=deepcopy(rng.bit_generator.state)
    sampler=PositiveSampler(c)
    assert any(not (c.sample('train',rng,scale)[1]==1).any() for _ in range(20))
    rng.bit_generator.state=saved
    for _ in range(10):
        label,record=sampler.next(scale)
        assert (label==1).sum()==1 and record['patient'] in c.train and record['category']=='positive'
    assert rng.bit_generator.state==saved


def snapshot_target(tmp_path):
    class Target:
        train=[f't{i:02}' for i in range(76)]; val=[f'e{i:02}' for i in range(19)]
        identity={}; protected=[]
        def __init__(self):
            self.path=tmp_path/'input.bin'; self.path.write_bytes(b'bound-input')
            from experiments.genseg30.contract import capture_identities
            self._identities=capture_identities([self.path])
            self.manifest={'cases':{p:{'source_shape':[1,2,4,4]} for p in self.train}}
            import blosc2
            from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
            self.original=SimpleNamespace(folder=tmp_path,dataset_class=nnUNetDatasetBlosc2)
            self._case_paths={}
            mask=tmp_path/'mask.npz'; np.savez_compressed(mask,keep=np.ones((2,4,4),bool))
            for patient in self.train:
                blosc2.asarray(np.zeros((1,2,4,4),np.float32),urlpath=str(tmp_path/(patient+'.b2nd')))
                blosc2.asarray(np.zeros((1,2,4,4),np.int16),urlpath=str(tmp_path/(patient+'_seg.b2nd')))
                self._case_paths[patient]={'mask_file':mask}
            self.reads=0; self.audits=0
        def training_dataset(self): return None
        def inventory(self,p): return []
        def verify_identity(self,i): pass
        def verify_inventory(self,i): self.audits+=1
        def _paths_for(self,p): return [str(self.path)]
        def load_train(self,d,p):
            assert p in self.train
            self.reads+=1
            x=np.arange(32,dtype=np.float32).reshape(1,2,4,4)/32
            y=np.zeros_like(x,dtype=np.int16); y[:,1,1,1]=1
            return x,y,{}
    return Target()


def test_snapshot_content_sampling_and_budget(tmp_path):
    from experiments.genseg_joint.joint_data import InputSnapshot
    target=snapshot_target(tmp_path); raw=Cohort(target,20261018)
    with pytest.raises(ValueError,match='budget'):
        InputSnapshot(raw,budget_bytes=1,resource_guard=lambda n:None,loader_workspace_bytes=1024)
    snap=InputSnapshot(raw,budget_bytes=100000,resource_guard=lambda n:None,loader_workspace_bytes=1024)
    assert target.reads==76 and snap.nbytes<=snap.estimated_bytes
    x,y,_=snap.cohort.load('train',snap.cohort.train[0])
    assert not x.flags.writeable and not y.flags.writeable
    with pytest.raises(ValueError): x.setflags(write=True)
    scale=raw.fit_scale(); reads=target.reads
    a=np.random.default_rng(42); b=np.random.default_rng(42)
    for _ in range(8):
        slow=raw.sample('train',a,scale); fast=snap.cohort.sample('train',b,scale)
        assert slow[2]==fast[2]
        for av,bv in zip(slow[:2],fast[:2]): np.testing.assert_array_equal(av,bv)
    prior=target.reads; audits=target.audits
    for _ in range(8): snap.cohort.sample('validation',b,scale); snap.lightweight_audit()
    assert target.reads==prior and target.audits==audits and prior>reads
    target.path.write_bytes(b'changed-inp')
    with pytest.raises(ValueError,match='stat'): snap.lightweight_audit()


def test_snapshot_load_change_and_end_hash_change(tmp_path):
    from experiments.genseg_joint.joint_data import InputSnapshot
    import os
    target=snapshot_target(tmp_path); raw=Cohort(target,20261018)
    original=target.load_train
    def changed(*args):
        result=original(*args); target.path.write_bytes(b'new-content'); return result
    target.load_train=changed
    with pytest.raises(ValueError,match='changed'): InputSnapshot(raw,budget_bytes=100000,resource_guard=lambda n:None,loader_workspace_bytes=1024)
    target.load_train=original
    from experiments.genseg30.contract import capture_identities
    target._identities=capture_identities([target.path])
    snap=InputSnapshot(raw,budget_bytes=100000,resource_guard=lambda n:None,loader_workspace_bytes=1024)
    old=target.path.stat(); target.path.write_bytes(b'bad-content'); os.utime(target.path,ns=(old.st_atime_ns,old.st_mtime_ns))
    # Lightweight stat cannot prove equal content; final full digest must reject it.
    snap.lightweight_audit()
    with pytest.raises(ValueError): snap.full_audit()


def test_same_new_engine_snapshot_vs_raw_update_parity(tmp_path,monkeypatch):
    from experiments.genseg_joint.joint_data import InputSnapshot
    target=snapshot_target(tmp_path); raw=Cohort(target,20261018)
    # Native 4x4 planes avoid unnecessary 256 padding for this tiny numerical comparison.
    import experiments.genseg_joint.data as data
    monkeypatch.setattr(data,'pad_plane',lambda a,*args:a.copy())
    snap=InputSnapshot(raw,budget_bytes=100000,resource_guard=lambda n:None,loader_workspace_bytes=1024)
    scale=raw.fit_scale(); engines=[engine(),engine()]
    samplers=[PositiveSampler(c) for c in (raw,snap.cohort)]
    rngs=[np.random.default_rng(54),np.random.default_rng(54)]
    for _ in range(2):
        records=[]
        for c,e,rng,sampler in zip((raw,snap.cohort),engines,rngs,samplers):
            x,y,tr=c.sample('train',rng,scale); v,vy,vr=c.sample('validation',rng,scale); fy,fr=sampler.next(scale)
            records.append((tr,vr,fr)); e.step(*(torch.from_numpy(a) for a in (x,y,v,vy,fy)))
        assert records[0]==records[1]
        for (_,a),(_,b) in zip(engines[0].networks(),engines[1].networks()):
            for key,value in a.state_dict().items(): torch.testing.assert_close(value,b.state_dict()[key],rtol=1e-6,atol=1e-8)
        for a,b in zip([*engines[0].gan.optimizers.values(),engines[0].s_optimizer],
                       [*engines[1].gan.optimizers.values(),engines[1].s_optimizer]):
            for i,item in a.state_dict()['state'].items():
                for k,v in item.items(): torch.testing.assert_close(v,b.state_dict()['state'][i][k],rtol=1e-6,atol=1e-8)
        torch.testing.assert_close(engines[0].g.networks.upconv_arch,engines[1].g.networks.upconv_arch,rtol=1e-6,atol=1e-8)


def production_protocol():
    from experiments.genseg_joint.joint_checkpoint import POLICY,PRETRAIN_SHA
    from experiments.genseg_joint.models import source_identity
    from experiments.genseg_joint.data import inner_split
    split=inner_split([f't{i:02}' for i in range(76)],[f'e{i:02}' for i in range(19)],20261018)
    return dict(POLICY,steps=1,split=split,source=source_identity(),pretrained_sha256=PRETRAIN_SHA,
        target_identity={},target_manifest_sha256='a'*64,implementation_files_sha256={'tiny.py':'b'*64},
        scaling=dict(method='inner61_valid_pixel_global_minmax_to_minus1_plus1_no_clip',min=0.,max=1.,fit_ids=split['train']),
        model_config={'G':{'width':64,'downs':8,'seed':20261018},'S':'pinned_author_UNet(1,1,bilinear=False)'})


def test_joint_checkpoint_roundtrip_and_malformed_state(tmp_path):
    from experiments.genseg_joint.joint_checkpoint import save,validate_payload
    e=engine()
    for key,n in e.networks():
        initial={'G':19400,'D':58200,'S':0}[key]
        for k,v in n.state_dict().items():
            if k.endswith('num_batches_tracked'): v.fill_(initial); e.initial_bn[key][k]=initial
    e.step(*batch()); p=production_protocol()
    rng=dict(numpy={},positive=dict(steps=1,positive_planes=1,rng={}),python=(),torch_cpu=torch.get_rng_state(),torch_cuda=[])
    save(tmp_path,e,p,rng,snapshot={},timing={},final_audit=lambda:None)
    import json
    receipt=json.loads((tmp_path/'complete.json').read_text())
    payload=torch.load(tmp_path/'joint.pt',map_location='cpu',weights_only=True)
    validate_payload(payload,receipt)
    # Restored G produces identical repeatable eval outputs, without restoring optimizers.
    restored=engine(); restored.g.net.load_state_dict(payload['G'])
    restored.g.networks.upconv_arch=payload['arch']['upconv_arch']
    e.g.net.eval(); restored.g.net.eval(); x=batch()[0]
    with torch.no_grad():
        expected=e.g.net(x); actual=restored.g.net(x)
        torch.testing.assert_close(expected,actual,rtol=0,atol=0)
        torch.testing.assert_close(actual,restored.g.net(x),rtol=0,atol=0)
    bad=deepcopy(payload); bad['optimizers']['s']['state_dict']['state'][0]['step'].fill_(2)
    with pytest.raises(ValueError,match='step'): validate_payload(bad,receipt)
    bad=deepcopy(payload); bad['arch']['upconv_arch'].fill_(float('nan'))
    with pytest.raises(ValueError,match='alpha'): validate_payload(bad,receipt)
    bad=deepcopy(payload); bad['protocol']['source']={}
    with pytest.raises(ValueError): validate_payload(bad,receipt)
    bad=deepcopy(payload); bad['protocol']['split']['validation'][0]=bad['protocol']['split']['train'][0]
    with pytest.raises(ValueError): validate_payload(bad,receipt)


def test_resource_guard_one_query_nonblocking_and_fail_closed(monkeypatch):
    from experiments.genseg_joint import joint_resources as r
    import sys
    psutil=SimpleNamespace()
    monkeypatch.setitem(sys.modules,'psutil',psutil)
    calls=[]; queries=[]
    monkeypatch.setattr(psutil,'cpu_percent',lambda interval: calls.append(interval) or 2.,raising=False)
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(percent=20.,total=100*1024**3,available=80*1024**3),raising=False)
    monkeypatch.setattr(r.subprocess,'run',lambda cmd,**kw:queries.append(cmd) or SimpleNamespace(stdout='1, GPU-one, 2, 100, 32768'))
    g=r.ResourceGuard(1); g(1024); g(); g()
    assert calls==[.5,None,None] and len(queries)==3 and all('-i' in q and q[q.index('-i')+1]=='1' for q in queries)
    monkeypatch.setattr(psutil,'cpu_percent',lambda interval:float('nan'))
    with pytest.raises(ValueError,match='telemetry'): g()


def test_new_cli_explicit_budget_and_legacy_rejection():
    from experiments.genseg_joint.joint_cli import parser
    with pytest.raises(SystemExit): parser().parse_args(['train'])
    e=engine()
    with pytest.raises(ValueError,match='legacy'): JointEngine(e.g,e.d,e.s)


def test_pinned_generator_higher_order_and_bn_cpu():
    from experiments.genseg_joint.models import Generator
    torch.set_num_threads(1); torch.manual_seed(73)
    g=Generator(width=1,downs=5,seed=20261018)
    g.pretrain_strategy='adam_trainbn_epoch_v1'
    d=g.discriminator(1)
    s=torch.nn.Sequential(torch.nn.BatchNorm2d(1),torch.nn.Conv2d(1,1,1))
    e=RepairJointEngine(g,d,s)
    x=torch.linspace(-.7,.8,4096).reshape(1,1,64,64)
    y=torch.zeros_like(x); y[:,:,20:40,20:40]=1; y[:,:,:2]=-1
    before={k:v.clone() for k,v in g.net.state_dict().items()}
    value=e.hyperloss(x,y,x*.9,y,y.flip(-1),g.networks.upconv_arch)
    grad,=torch.autograd.grad(value,g.networks.upconv_arch)
    assert torch.isfinite(grad).all() and grad.double().abs().sum()>0
    assert all(torch.equal(v,before[k]) for k,v in g.net.state_dict().items())
    result=e.step(x,y,x*.9,y,y.flip(-1))
    assert result['alpha_gradient_norm']>0 and e.steps==1


def test_snapshot_no_positive_stops(tmp_path):
    target=snapshot_target(tmp_path); original=target.load_train
    def negative(*a):
        x,y,m=original(*a); return x,np.zeros_like(y),m
    target.load_train=negative
    with pytest.raises(ValueError,match='No positive'): PositiveSampler(Cohort(target,20261018))


def test_transition_wrong_hash_no_weight_load(monkeypatch):
    from experiments.genseg_joint.joint_checkpoint import audit_transition
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('unsafe load'))
    with pytest.raises(ValueError,match='pinned'): audit_transition(None,'missing','a'*64)


def test_new_runtime_check_load_save_preview_cpu(tmp_path,monkeypatch):
    from experiments.genseg_joint import joint_runtime as rt,joint_checkpoint as cp,models,preview
    from experiments.genseg30.contract import sha256,read_json
    from experiments.genseg_joint.joint_checkpoint import PRETRAIN_SHA
    target=snapshot_target(tmp_path); target.protected=[]; target.manifest_sha256='a'*64
    target.require_no_unpack_writes=lambda:None
    pre=tmp_path/'pretrain'; pre.mkdir(); (pre/'gan.pt').write_bytes(b'fixture'); (pre/'complete.json').write_text('{}')
    # Hardware, large checkpoint transition and model size replaced; production snapshot/engine/save/audits execute.
    monkeypatch.setattr(rt,'audit_transition',lambda *a:None)
    class Guard:
        def __init__(self,*a): pass
        def __call__(self,*a): return dict(allocator_fraction=.1,gpu_uuid='CPU-fixture')
        def bind(self): pass
    monkeypatch.setattr(rt,'ResourceGuard',Guard)
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:False)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:False)
    monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',lambda *a:None)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda:0)
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[])
    original_to=torch.Tensor.to
    monkeypatch.setattr(torch.Tensor,'to',lambda self,*a,**kw:original_to(self,*(('cpu',)+a[1:] if a and a[0]=='cuda:0' else a),**kw))
    def create(**kw):
        e=engine(); return e.g,e.d,e.s
    monkeypatch.setattr(models,'create_models',create)
    def transition(target,run,digest,g,d,scale):
        for net,initial in ((g.net,19400),(d,58200)):
            for k,v in net.state_dict().items():
                if k.endswith('num_batches_tracked'): v.fill_(initial)
    monkeypatch.setattr(rt,'load_transition',transition)
    import experiments.genseg_joint.data as data
    monkeypatch.setattr(data,'pad_plane',lambda a,*args:a.copy())
    assert rt.check(target,pre,PRETRAIN_SHA,100000,1024,1024)['arrays_loaded'] is False
    output=tmp_path/'joint'
    rt.run(target,output,pretrained_run=pre,pretrained_sha256=PRETRAIN_SHA,steps=2,gpu=1,
        cache_budget_bytes=100000,model_reserve_bytes=1024,loader_workspace_bytes=1024)
    receipt=read_json(output/'complete.json'); payload=torch.load(output/'joint.pt',weights_only=True)
    cp.validate_payload(payload,receipt)
    assert receipt['steps']==2 and target.reads==76
    records=[__import__('json').loads(s) for s in (output/'steps.jsonl').read_text().splitlines()]
    assert len(records)==2 and all(r['fake_mask']['foreground_pixels']>0 for r in records)
    assert all('unit_boundary_sync_wall' in r and 'stage_host_wall_seconds' in r for r in records)
    cp.audit_joint(target,output,sha256(output/'joint.pt'))
    # Real repair audit/schema and renderer; substitute GPU-sized G restore only.
    monkeypatch.setattr(preview,'gpu_guard',lambda *a,**kw:dict(allocator_fraction=.1))
    monkeypatch.setattr(preview,'Generator',lambda **kw:object())
    monkeypatch.setattr(preview,'restore_generator',lambda *a:None)
    monkeypatch.setattr(preview,'generate',lambda g,y,s:(np.zeros_like(y,dtype=np.float32),list(y.shape)))
    out=tmp_path/'preview'
    preview.run_preview(target,output,sha256(output/'joint.pt'),out,gpu=1,patient_limit=2,phase='joint_repair')
    result=read_json(out/'preview.json')
    assert len(result['previews'])==2 and all(r['category']=='lesion' for r in result['previews'])
    assert result['phase']=='joint_repair'


def test_snapshot_transient_refused_before_loader(tmp_path):
    from experiments.genseg_joint.joint_data import InputSnapshot,cache_plan
    import blosc2
    target=snapshot_target(tmp_path); raw=Cohort(target,20261018)
    patient=raw.train[0]
    for suffix,dtype in (('.b2nd',np.float64),('_seg.b2nd',np.int64)):
        blosc2.asarray(np.zeros((1,2,4,4),dtype),urlpath=str(tmp_path/(patient+suffix)),mode='w')
    plan=cache_plan(target,raw.train+raw.val,1024)
    final=plan['final_cache_estimated_bytes']; transient=plan['cases'][patient]['load_transient_bytes']
    assert plan['cases'][patient]['final_bytes']==32*12
    assert transient==32*(16+8+2+64)+1024
    assert final<final+transient
    calls=[]
    def guard(n):
        calls.append(n)
        if n>final: raise ValueError('projected transient headroom')
    with pytest.raises(ValueError,match='transient'):
        InputSnapshot(raw,budget_bytes=final,resource_guard=guard,loader_workspace_bytes=1024)
    assert target.reads==0 and calls==[final+transient]


@pytest.mark.parametrize('bad',['backend','dtype','metadata'])
def test_snapshot_unaudited_metadata_refused_before_load(tmp_path,bad):
    from experiments.genseg_joint.joint_data import InputSnapshot
    import blosc2
    target=snapshot_target(tmp_path); raw=Cohort(target,20261018); patient=raw.train[0]
    if bad=='backend': target.original.dataset_class=object
    elif bad=='dtype':
        blosc2.asarray(np.zeros((1,2,4,4),np.uint64),urlpath=str(tmp_path/(patient+'_seg.b2nd')),mode='w')
    else: target.manifest['cases'][patient].pop('source_shape')
    with pytest.raises((ValueError,KeyError)):
        InputSnapshot(raw,budget_bytes=100000,resource_guard=lambda n:None,loader_workspace_bytes=1024)
    assert target.reads==0


@pytest.mark.parametrize('ratio,accepted',[(.789,True),(.79,False),(.795,False),(.8,False)])
def test_allocator_fraction_boundary(monkeypatch,ratio,accepted):
    from experiments.genseg_joint import joint_resources as r
    import sys
    ps=SimpleNamespace(cpu_percent=lambda interval:2.,virtual_memory=lambda:SimpleNamespace(
        percent=20.,total=100*1024**3,available=80*1024**3))
    monkeypatch.setitem(sys.modules,'psutil',ps)
    monkeypatch.setattr(r.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout=f'1, GPU-one, 2, {ratio*100000}, 100000'))
    g=r.ResourceGuard(1)
    if accepted: assert g()['allocator_fraction']>0
    else:
        with pytest.raises(ValueError,match='allocator|80 percent'): g()


def proxy_path_object(e,args,monkeypatch,mode,*,sever=False,zero_s=False):
    """Freeze numerical anchors at a0 ONCE, including across FD perturbations."""
    from experiments.genseg_joint import joint_repair as repair
    from experiments.genseg_joint.core import virtual_chain,virtual_sgd,bind_alpha
    a0=torch.tensor(.3,dtype=torch.double,requires_grad=True)
    captured={}
    def capture(gp,sp,gl,sl,vl,glr,slr):
        result=virtual_chain(gp,sp,gl,sl,vl,glr,slr)
        captured.update(vg={k:v.detach().clone() for k,v in result[1].items()})
        return result
    with monkeypatch.context() as m:
        m.setattr(repair,'virtual_chain',capture)
        e.hyperloss(*args,a0)
    counts={'gl':0,'sl':0,'vl':0}
    phase={'synthetic':False}
    def binding(g,a):
        return bind_alpha(g,a0.detach() if mode=='indirect' and phase['synthetic'] else a)
    monkeypatch.setattr(repair,'bind_alpha',binding)
    def isolated(gp,sp,gl,sl,vl,glr,slr):
        def generator(p): counts['gl']+=1; return gl(p)
        def segmenter(p,vg):
            counts['sl']+=1
            if mode=='direct': vg=captured['vg']
            elif sever: vg={k:v.detach() for k,v in vg.items()}
            if mode=='indirect':
                phase['synthetic']=True
                try: return sl(p,vg)
                finally: phase['synthetic']=False
            return sl(p,vg)
        def validation(p): counts['vl']+=1; return vl(p)
        return virtual_chain(gp,sp,generator,segmenter,validation,glr,0. if zero_s else slr)
    monkeypatch.setattr(repair,'virtual_chain',isolated)
    return lambda a:e.hyperloss(*args,a),counts


@pytest.mark.parametrize('mode',['direct','indirect'])
def test_production_proxy_isolated_paths(monkeypatch,mode):
    e=engine(); args=tuple(v.double() for v in batch())
    for _,n in e.networks(): n.double()
    f,counts=proxy_path_object(e,args,monkeypatch,mode)
    a=torch.tensor(.3,dtype=torch.double,requires_grad=True); eps=1e-4
    grad,=torch.autograd.grad(f(a),a)
    fd=(f((a.detach()+eps).requires_grad_())-f((a.detach()-eps).requires_grad_()))/(2*eps)
    assert torch.isfinite(grad) and abs(float(grad))>1e-12
    assert abs(float((grad-fd).detach())) < .01*abs(float(grad))
    torch.testing.assert_close(grad,fd,rtol=1e-5,atol=1e-8)
    assert counts=={'gl':3,'sl':3,'vl':3}
    print(f'{mode}: grad={grad.item():.16g}, fd={fd.item():.16g}')


@pytest.mark.parametrize('mode',['direct','indirect'])
@pytest.mark.parametrize('control',['sever','zero_s'])
def test_production_proxy_negative_controls(monkeypatch,mode,control):
    e=engine(); args=tuple(v.double() for v in batch())
    for _,n in e.networks(): n.double()
    f,_=proxy_path_object(e,args,monkeypatch,mode,sever=control=='sever',zero_s=control=='zero_s')
    a=torch.tensor(.3,dtype=torch.double,requires_grad=True); eps=1e-4
    value=f(a); grad,=torch.autograd.grad(value,a,allow_unused=True)
    fd=(f((a.detach()+eps).requires_grad_())-f((a.detach()-eps).requires_grad_()))/(2*eps)
    if control=='sever' and mode=='indirect':
        # Re-detaching at each alpha severs autograd, NOT numerical dependence.
        assert grad is None and abs(fd.item())>1e-12
    elif control=='zero_s':
        # Graph remains connected via multiplication by zero S rate; derivative is exactly zero.
        assert grad is not None and grad.item()==0
        torch.testing.assert_close(fd,torch.zeros_like(fd),rtol=0,atol=0)
    else:
        assert grad is not None and abs(grad.item())>1e-12
        torch.testing.assert_close(grad,fd,rtol=1e-5,atol=1e-8)


def test_production_indirect_oracle_detects_injected_sever(monkeypatch):
    # RED sensitivity: run the actual isolated-path oracle with only the virtual
    # G output graph detached, and require its missing-gradient assertion to fail.
    from experiments.genseg_joint import core
    original=core.virtual_sgd
    calls=[0]
    def broken(parameters,loss,lr):
        result=original(parameters,loss,lr); calls[0]+=1
        # First is anchor capture G, then capture S, then evaluated G.
        return {k:v.detach() for k,v in result.items()} if calls[0]==3 else result
    monkeypatch.setattr(core,'virtual_sgd',broken)
    with pytest.raises(RuntimeError,match='not have been used'):
        test_production_proxy_isolated_paths(monkeypatch,'indirect')
    assert calls[0]>=4
