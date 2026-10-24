"""Tiny CPU production-engine recovery evidence, never patient evidence."""
from copy import deepcopy
from types import SimpleNamespace
import random
import numpy as np
import pytest
import torch
from experiments.genseg_joint import joint_resume as recovery
from experiments.genseg_joint.data import EpochSampler
from experiments.genseg_joint.joint_data import PositiveSampler
from experiments.genseg_joint.tests.test_nnunet_s import engine, batch
from experiments.genseg_joint.tests.test_segment import receipt_fixture
from experiments.genseg_joint.joint_checkpoint import nn_policy
from experiments.genseg30.contract import read_json


def setup(tmp_path):
    e=engine()
    # TinyG.restore_arch omits this production alias; mirror Generator.restore_arch.
    e.g.networks._conv_arch_parameters=[e.g.networks.conv_arch]
    for key,n in e.networks():
        for k,v in n.state_dict().items():
            if k.endswith('num_batches_tracked'):
                origin={'G':19400,'D':58200}[key]; v.fill_(origin); e.initial_bn[key][k]=origin
    tmp_path.mkdir(exist_ok=True)
    _,r=receipt_fixture(tmp_path)
    p=dict(r['protocol'],**nn_policy(recovery=True)); p.update(epochs=2,steps=4,total_valid_planes=2,epoch_recovery=recovery.FORMAT,resume_lineage=[])
    p.update(selection='last_complete_unit; no_external19_arrays_or_metrics; epoch_resume_enabled',
        checkpoint_selection='final_only_no_best_no_early_stop; epoch_recovery_separate')
    p['model_config']['S']=e.s_descriptor; p['target_identity'].update(e.s_descriptor['source_identity'])
    patient=p['split']['train'][0]
    class Cohort:
        train=[patient]
        def load(self,role,patient):
            x=np.ones((1,2,8,8),np.float32); y=np.zeros_like(x); y[:,:,2:6,2:6]=1
            return x,y,{}
        def plane(self,role,patient,z,scale):
            args=batch(); return args[0].numpy(),args[1].numpy(),dict(patient=patient,z=z)
    c=Cohort(); real=EpochSampler(c,np.random.default_rng(6)); pos=PositiveSampler(c)
    rng=np.random.default_rng(7); random.seed(8)
    return e,p,real,pos,rng


def epoch(e,p,real,pos,rng):
    records=[]
    for _ in range(2):
        x,y,record=real.next(p['scaling']); fy,fr=pos.next(p['scaling'])
        val=float(rng.uniform(.5,.9)); random.random(); np.random.random(); torch.rand(1)
        e.step(torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(x)*val,torch.from_numpy(y),torch.from_numpy(fy))
        records.append((record,fr,val))
    p['epoch_progress']=real.progress()['epoch_progress']
    return records


def save(out,e,p,real,pos,rng):
    out.mkdir(exist_ok=True)
    return recovery.save_epoch(out,e,p,recovery.capture_rng(rng,pos,real),snapshot={},timing={},
        full_audit=lambda:None,guard=lambda n:None,lineage=[])


def equal(a,b):
    if isinstance(a,torch.Tensor): assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): equal(x,y)
    else: assert a==b


def test_continuous_vs_fresh_resume_all_state(tmp_path):
    e,p,real,pos,rng=setup(tmp_path/'a'); epoch(e,p,real,pos,rng)
    before=recovery.capture_rng(rng,pos,real)
    grads=[None if v.grad is None else v.grad.clone() for _,n in e.networks() for v in n.parameters()]
    path,digest=save(tmp_path/'out',e,p,real,pos,rng)
    equal(before,recovery.capture_rng(rng,pos,real))
    equal(grads,[v.grad for _,n in e.networks() for v in n.parameters()])
    records=epoch(e,p,real,pos,rng)
    expected={k:deepcopy(n.state_dict()) for k,n in e.networks()}
    opts=e.optimizer_state(); arch={k:getattr(e.g.networks,k).clone() for k in ('conv_arch','upconv_arch')}
    states=recovery.capture_rng(rng,pos,real)
    b,q,rr,pp,rg=setup(tmp_path/'b')
    recovery.restore(b,recovery.load(path,digest),rr,pp,rg,q)
    assert epoch(b,q,rr,pp,rg)==records
    equal(expected,{k:n.state_dict() for k,n in b.networks()}); equal(opts,b.optimizer_state())
    equal(arch,{k:getattr(b.g.networks,k) for k in arch}); equal(states,recovery.capture_rng(rg,pp,rr))
    assert b.steps==b.gan.steps==4 and b.initial_bn==e.initial_bn
    assert b.s.decoder.encoder is b.s.encoder
    assert not list(b.s.named_buffers())
    assert not (tmp_path/'out'/'complete.json').exists()
    from experiments import genseg_segment as seg
    with pytest.raises((ValueError,FileNotFoundError)): seg.audit_receipt(tmp_path/'out',digest,nnunet=True)


@pytest.mark.parametrize('fault',['hash','rng','sampler','boundary','budget','binding','poison','partial'])
def test_reject(tmp_path,fault):
    e,p,real,pos,rng=setup(tmp_path/'a'); epoch(e,p,real,pos,rng)
    if fault in ('poison','partial'):
        if fault=='poison': e.poisoned=True
        else: e.gan.steps+=1
        with pytest.raises(ValueError): save(tmp_path/'out',e,p,real,pos,rng)
        return
    path,digest=save(tmp_path/'out',e,p,real,pos,rng)
    if fault=='hash':
        with pytest.raises(ValueError,match='hash'): recovery.load(path,'0'*64)
        return
    saved=recovery.load(path,digest)
    if fault=='rng': del saved['payload']['rng']['torch_cpu']
    if fault=='sampler': saved['payload']['rng']['real']['order']=[0,0]
    if fault=='boundary': saved['payload']['rng']['real']['position']=1
    if fault=='budget': saved['planned_epochs']=3
    if fault=='binding': saved['payload']['protocol']['target_manifest_sha256']='0'*64
    b,q,rr,pp,rg=setup(tmp_path/'b')
    with pytest.raises((ValueError,KeyError)): recovery.restore(b,saved,rr,pp,rg,q)


@pytest.mark.parametrize('stage',['file','pointer'])
def test_atomic_failure_preserves_previous(tmp_path,monkeypatch,stage):
    e,p,real,pos,rng=setup(tmp_path/'a'); epoch(e,p,real,pos,rng)
    out=tmp_path/'out'; path,digest=save(out,e,p,real,pos,rng)
    previous=read_json(out/'resume_latest.json')
    epoch(e,p,real,pos,rng)
    original=recovery.os.replace
    def fail(src,dst):
        if (stage=='file' and str(dst).endswith('.pt')) or (stage=='pointer' and str(dst).endswith('.json')): raise OSError('injected')
        return original(src,dst)
    monkeypatch.setattr(recovery.os,'replace',fail)
    with pytest.raises(OSError,match='injected'): save(out,e,p,real,pos,rng)
    assert read_json(out/'resume_latest.json')==previous
    recovery.load(path,digest)
    assert not (out/'complete.json').exists()


def test_retention_and_completed_budget_refusal(tmp_path):
    e,p,real,pos,rng=setup(tmp_path/'a'); p.update(epochs=3,steps=6)
    out=tmp_path/'out'
    for _ in range(3):
        epoch(e,p,real,pos,rng); path,digest=save(out,e,p,real,pos,rng)
    assert sorted(v.name for v in out.glob('*.pt'))==['resume_epoch_0002.pt','resume_epoch_0003.pt']
    pointer=read_json(out/'resume_latest.json'); assert len(pointer['history'])==2
    for item in pointer['history']: recovery.load(out/item['checkpoint'],item['sha256'])
    b,q,rr,pp,rg=setup(tmp_path/'b'); q.update(epochs=3,steps=6)
    with pytest.raises(ValueError,match='completed'): recovery.restore(b,recovery.load(path,digest),rr,pp,rg,q)


def test_cli_resume_requires_new_epoch_strategy_before_io():
    from experiments.genseg_joint.joint_cli import main
    base=['train','--strategy','joint_adam_trainbn_positive_sgdproxy_v1']
    for key in ('source','raw-root','targets','baseline-run','output','pretrained-run'): base+=['--'+key,'fixture']
    base+=['--pretrained-sha256','a'*64,'--cache-budget-bytes','100000',
           '--loader-workspace-bytes','1024','--model-reserve-bytes','1024','--steps','2']
    with pytest.raises(ValueError,match='Paired'): main(base+['--resume-from','missing.pt'])
    with pytest.raises(ValueError,match='epoch'): main(base+['--resume-from','missing.pt','--resume-sha256','a'*64])

def test_runtime_interrupt_then_new_attempt_resume(tmp_path,monkeypatch):
    from experiments.genseg_joint import joint_runtime as rt,models,data
    from experiments.genseg_joint.joint_checkpoint import PRETRAIN_SHA
    from experiments.genseg30.contract import sha256
    e,p,real,pos,rng=setup(tmp_path/'fixture'); c=real.cohort
    c.split=p['split']; c.val=p['split']['validation']
    c.fit_scale=lambda **kw:deepcopy(p['scaling'])
    c.sample=lambda role,rng,scale:c.plane(role,p['split'][role][0],int(rng.integers(2)),scale)
    target=SimpleNamespace(protected=[],identity=p['target_identity'],manifest_sha256=p['target_manifest_sha256'],require_no_unpack_writes=lambda:None)
    pre=tmp_path/'pre'; pre.mkdir(); (pre/'gan.pt').write_bytes(b'fake GD20'); (pre/'complete.json').write_text('{}')
    monkeypatch.setattr(data,'Cohort',lambda *a,**kw:c)
    class Snapshot:
        def __init__(self,*a,**kw):
            self.cohort=c; self.timings={}; self.plan={}; self.hashes={}; self.stats={}
            self.nbytes=0; self.estimated_bytes=0
        def full_audit(self): pass
        def lightweight_audit(self): pass
    monkeypatch.setattr(rt,'InputSnapshot',Snapshot)
    monkeypatch.setattr(rt,'check',lambda *a,**kw:dict(s_descriptor=e.s_descriptor))
    class Guard:
        def __init__(self,*a): pass
        def __call__(self,*a,**kw): return dict(allocator_fraction=.1,gpu_uuid='CPU-fixture')
        def bind(self): pass
    monkeypatch.setattr(rt,'ResourceGuard',Guard)
    for name,value in [('is_initialized',False),('is_available',True),('is_current_stream_capturing',False),('max_memory_allocated',0),('max_memory_reserved',0)]:
        monkeypatch.setattr(torch.cuda,name,lambda value=value:value)
    monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',lambda *a:None)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[])
    original_to=torch.Tensor.to
    monkeypatch.setattr(torch.Tensor,'to',lambda self,*a,**kw:original_to(self,*(('cpu',)+a[1:] if a and a[0]=='cuda:0' else a),**kw))
    def create(**kw):
        fresh=engine(); g=fresh.g; original=g.restore_arch
        def restore_alias(values):
            original(values); n=g.networks
            n._conv_arch_parameters=[n.conv_arch]; n._upconv_arch_parameters=[n.upconv_arch]; n._arch_parameters=[n.upconv_arch,n.conv_arch]
        g.restore_arch=restore_alias
        return g,fresh.d,fresh.s
    monkeypatch.setattr(models,'create_models',create)
    transitions=[]
    def transition(target,run,digest,g,d,scale):
        transitions.append(True)
        for net,origin in ((g.net,19400),(d,58200)):
            for name,v in net.state_dict().items():
                if name.endswith('num_batches_tracked'): v.fill_(origin)
    monkeypatch.setattr(rt,'load_transition',transition)
    from experiments.genseg_joint import nnunet_s
    monkeypatch.setattr(nnunet_s,'pad_unit',lambda arrays,descriptor:arrays)
    original_save=recovery.save_epoch
    def interrupt(*a,**kw):
        original_save(*a,**kw); raise RuntimeError('interrupt after committed epoch')
    monkeypatch.setattr(recovery,'save_epoch',interrupt)
    kwargs=dict(pretrained_run=pre,pretrained_sha256=PRETRAIN_SHA,epochs=2,gpu=1,
        cache_budget_bytes=100000,model_reserve_bytes=1024,loader_workspace_bytes=1024,nnunet=True)
    old=tmp_path/'attempt1'
    with pytest.raises(RuntimeError,match='interrupt'): rt.run(target,old,**kwargs)
    path=old/'resume_epoch_0001.pt'; digest=sha256(path); before={v.name:v.read_bytes() for v in old.iterdir()}
    assert not (old/'complete.json').exists()
    monkeypatch.setattr(recovery,'save_epoch',original_save)
    c.fit_scale=lambda **kw:pytest.fail('Resume refitted scale')
    new=tmp_path/'attempt2'; rt.run(target,new,resume_from=path,resume_sha256=digest,**kwargs)
    import json
    logs=[json.loads(v) for v in (new/'steps.jsonl').read_text().splitlines()]
    assert [v['step'] for v in logs]==[3,4] and len(transitions)==1
    receipt=read_json(new/'complete.json'); assert receipt['steps']==4
    assert receipt['protocol']['resume_lineage'][0]['checkpoint_sha256']==digest
    assert before=={v.name:v.read_bytes() for v in old.iterdir()}


def test_save_load_restore_without_file_digest(tmp_path,monkeypatch):
    import hashlib
    monkeypatch.delattr(hashlib,'file_digest',raising=False)
    e,p,real,pos,rng=setup(tmp_path/'a'); epoch(e,p,real,pos,rng)
    expected={k:deepcopy(n.state_dict()) for k,n in e.networks()}
    expected_rng=recovery.capture_rng(rng,pos,real)
    path,digest=save(tmp_path/'out',e,p,real,pos,rng)
    b,q,rr,pp,rg=setup(tmp_path/'b')
    recovery.restore(b,recovery.load(path,digest),rr,pp,rg,q)
    assert b.steps==b.gan.steps==2
    equal(expected,{k:n.state_dict() for k,n in b.networks()})
    equal(expected_rng,recovery.capture_rng(rg,pp,rr))
