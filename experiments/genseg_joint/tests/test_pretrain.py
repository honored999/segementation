from copy import deepcopy
import torch
import pytest
from experiments.genseg_joint import core
from experiments.genseg_joint.models import Generator


def tiny():
    torch.set_num_threads(1)
    torch.manual_seed(48)
    g=Generator(width=1,downs=5,seed=48)
    return g,g.discriminator(1)


def batch():
    x=torch.randn(1,1,64,64)*.2
    y=torch.zeros_like(x); y[:,:,20:40,20:40]=1; y[:,:,:3]=-1
    return x,y


def snapshot(g,d):
    return [{k:v.clone() for k,v in n.state_dict().items()} for n in (g.net,d)]


def unchanged(g,d,before):
    assert all(all(torch.equal(v,b[k]) for k,v in n.state_dict().items()) for n,b in zip((g.net,d),before))


@pytest.mark.parametrize("strategy", ["legacy", "adam_trainbn_epoch_v1"])
def test_gan_step_frozen_arch_and_transaction(monkeypatch,strategy):
    assert hasattr(core,'PretrainEngine'), 'missing frozen-architecture pretraining engine'
    g,d=tiny(); engine=core.PretrainEngine(g,d,strategy=strategy)
    before=snapshot(g,d); alpha={k:getattr(g.networks,k).clone() for k in ('conv_arch','upconv_arch')}
    x,y=batch(); metrics=engine.step(x,y)
    assert engine.steps==1 and all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert all(any(not torch.equal(v,b[k]) for k,v in n.named_parameters()) for n,b in zip((g.net,d),before))
    assert all(torch.equal(getattr(g.networks,k),v) and not getattr(g.networks,k).requires_grad for k,v in alpha.items())
    if strategy != 'legacy':
        for net,old,count in zip((g.net,d),before,(1,3)):
            tracked=[k for k in old if k.endswith('num_batches_tracked')]
            assert tracked and all(int(net.state_dict()[k]-old[k])==count for k in tracked)
        for opt in engine.optimizer_state().values():
            assert opt['algorithm']=='Adam'
            assert all(v['step'].item()==1 and torch.isfinite(v['exp_avg']).all()
                       for v in opt['state_dict']['state'].values())
    before=snapshot(g,d)
    with pytest.raises(ValueError,match='Nonfinite'): engine.step(x*float('nan'),y)
    unchanged(g,d,before); assert engine.steps==1
    def bad(*args,**kwargs): return tuple(torch.full_like(v,float('inf')) for v in args[1])
    monkeypatch.setattr(torch.autograd,'grad',bad)
    with pytest.raises(ValueError,match='Nonfinite'): engine.step(x,y)
    assert engine.steps==1
    if strategy=='legacy': unchanged(g,d,before)
    else:
        assert engine.poisoned
        with pytest.raises(ValueError,match='poison'): engine.step(x,y)


def protocol(g):
    from experiments.genseg_joint.pretrain import policy
    return dict(policy(),seed=48,steps=1,split={'train':['a'],'validation':['b'],'external':['c']},
        target_identity={'id':'target'},target_manifest_sha256='a'*64,
        scaling={'method':'inner61_valid_pixel_global_minmax_to_minus1_plus1_no_clip','min':-2.,'max':3.,'fit_ids':['a']},
        model_config={'G':g.config,'D_width':1})


def test_completed_pretrain_real_joint_handoff(tmp_path):
    assert hasattr(core,'PretrainEngine'), 'missing pretrain handoff'
    from experiments.genseg_joint.pretrain import save_pretrain, load_pretrained, audit_pretrained
    from experiments.genseg30.contract import sha256
    g,d=tiny(); e=core.PretrainEngine(g,d); x,y=batch(); e.step(x,y)
    p=protocol(g); save_pretrain(tmp_path,e,p,{'torch_cpu':torch.get_rng_state()})
    digest=sha256(tmp_path/'gan.pt'); before=snapshot(g,d)
    receipt=audit_pretrained(tmp_path,digest,expected=p)
    assert receipt['phase']=='gan_pretrain' and receipt['complete']
    fresh,fd=tiny(); load_pretrained(tmp_path,digest,fresh,fd,expected=p)
    unchanged(fresh,fd,before)
    from experiments.genseg_joint.models import author_unet
    s=author_unet(); first=next(s.parameters()); sb=first.detach().clone()
    joint=core.JointEngine(fresh,fd,s)
    assert joint.steps==0 and not joint._alpha_state['state']
    assert torch.equal(first,sb)
    metrics=joint.step(x,y,x*.7,torch.flip(y,[-1]),torch.flip(y,[-2]))
    assert joint.steps==1 and metrics['alpha_change_norm']>0
    assert e.steps==1


@pytest.mark.parametrize('field',['split','target_identity','target_manifest_sha256','scaling','source','rates','model_config','BN','optimizer','phase'])
def test_metadata_mismatch_no_tensor_load(tmp_path,monkeypatch,field):
    assert hasattr(core,'PretrainEngine'), 'missing pretrain handoff'
    from experiments.genseg_joint.pretrain import save_pretrain,audit_pretrained
    from experiments.genseg30.contract import read_json,write_json,sha256
    from types import SimpleNamespace
    p=protocol(SimpleNamespace(config={'width':1,'downs':5,'seed':48}))
    (tmp_path/'gan.pt').write_bytes(b'metadata-only-no-tensors')
    write_json(tmp_path/'complete.json',{'format':1,'phase':'gan_pretrain',
        'classification':p['classification'],'complete':True,'steps':1,'checkpoint':'gan.pt',
        'checkpoint_sha256':sha256(tmp_path/'gan.pt'),'protocol':p})
    r=read_json(tmp_path/'complete.json'); r['protocol'][field]='wrong'
    (tmp_path/'complete.json').unlink(); write_json(tmp_path/'complete.json',r)
    monkeypatch.setattr(torch,'load',lambda *a,**k: pytest.fail('metadata audit loaded tensors'))
    with pytest.raises(ValueError): audit_pretrained(tmp_path,sha256(tmp_path/'gan.pt'),expected=p)


@pytest.mark.parametrize('damage',['G','D','arch','receipt','hash','joint','optimizer','shape','dtype','missing','schema'])
def test_bad_handoff_never_mutates(tmp_path,damage):
    assert hasattr(core,'PretrainEngine'), 'missing pretrain handoff'
    from experiments.genseg_joint.pretrain import save_pretrain,load_pretrained
    from experiments.genseg30.contract import read_json,write_json,sha256
    g,d=tiny(); e=core.PretrainEngine(g,d); e.steps=1; p=protocol(g); save_pretrain(tmp_path,e,p,{})
    path=tmp_path/'gan.pt'; r=read_json(tmp_path/'complete.json')
    if damage in ('G','D','arch','optimizer','shape','dtype','missing','schema'):
        payload=torch.load(path,weights_only=True)
        if damage=='optimizer': payload['optimizers']['g']['lr']=7.
        elif damage=='shape': payload['G'][next(iter(payload['G']))]=torch.ones(1)
        elif damage=='dtype':
            key=next(iter(payload['D'])); payload['D'][key]=payload['D'][key].double()
        elif damage=='missing': payload['arch'].pop('conv_arch')
        elif damage=='schema': payload['S']={}
        else:
            key=next(iter(payload[damage])); payload[damage][key]=torch.tensor(float('nan'))
        torch.save(payload,path); r['checkpoint_sha256']=sha256(path)
    if damage=='receipt': r['complete']=False
    if damage=='joint': r['phase']='joint'; r['checkpoint']='joint.pt'
    (tmp_path/'complete.json').unlink(); write_json(tmp_path/'complete.json',r)
    before=snapshot(g,d); alpha=g.networks.upconv_arch.clone()
    with pytest.raises(ValueError): load_pretrained(tmp_path,'0'*64 if damage=='hash' else sha256(path),g,d,expected=p)
    unchanged(g,d,before); assert torch.equal(g.networks.upconv_arch,alpha)


def test_pretrain_cli_and_paired_handoff():
    from experiments.genseg_joint.cli import parser
    base=['--source','s','--raw-root','r','--targets','t','--baseline-run','b','--output','o']
    p=parser(); a=p.parse_args(['pretrain']+base+['--gpu','1','--steps','1'])
    assert a.steps==1
    a=p.parse_args(['train']+base+['--gpu','1','--steps','1','--pretrained-run','p','--pretrained-sha256','a'*64])
    assert a.pretrained_sha256=='a'*64
    a=p.parse_args(['pretrain']+base+['--gpu','1','--steps','100','--epochs','1','--strategy','adam_trainbn_epoch_v1'])
    assert a.epochs==1 and a.steps==100
    a=p.parse_args(['diagnose-single-slice']+base+['--gpu','1','--steps','100','--strategy','adam_trainbn_epoch_v1','--initialization','cold'])
    assert a.pretrained_run is None and a.initialization=='cold'
    with pytest.raises(SystemExit): p.parse_args(['pretrain']+base+['--gpu','1','--steps','100','--strategy','adam_trainbn_epoch_v1'])


@pytest.mark.parametrize('extra',[
    ['--pretrained-run','p'],['--pretrained-sha256','a'*64],
    ['--pretrained-run','p','--pretrained-sha256','bad'],['--seed','123']])
def test_cli_invalid_handoff_before_source_access(monkeypatch,extra):
    from experiments.genseg_joint.cli import main
    import experiments.genseg30.contract as contract
    monkeypatch.setattr(contract,'Source',lambda *a: pytest.fail('invalid mode reached source'))
    base=['check','--source','s','--raw-root','r','--targets','t','--baseline-run','b','--output','o']
    with pytest.raises(ValueError): main(base+extra)


def test_pretrain_final_audit_failure_no_receipt(tmp_path):
    from experiments.genseg_joint.pretrain import save_pretrain
    g,d=tiny(); e=core.PretrainEngine(g,d); e.steps=1
    def reject(): raise ValueError('changed input')
    with pytest.raises(ValueError,match='changed input'):
        save_pretrain(tmp_path,e,protocol(g),{},final_audit=reject)
    assert (tmp_path/'gan.pt').exists() and not (tmp_path/'complete.json').exists()


def test_check_handoff_metadata_only(tmp_path,monkeypatch,capsys):
    from experiments.genseg_joint.cli import main
    from experiments.genseg_joint.pretrain import policy
    from experiments.genseg_joint.data import inner_split
    from experiments.genseg30.contract import write_json,sha256
    import experiments.genseg30.contract as contract
    import experiments.genseg3mm.targets as targets
    import experiments.genseg_joint.models as models
    class Source:
        def __init__(self,*a): pass
        def require_no_unpack_writes(self): pass
    class Target:
        train=[f't{i}' for i in range(76)]; val=[f'e{i}' for i in range(19)]; protected=[]
        identity={'test':True}; manifest_sha256='a'*64
        def __init__(self,*a,**kw): pass
        def load_train(self,*a): pytest.fail('check read arrays')
    p=dict(policy(),seed=20261018,steps=1,split=inner_split(Target.train,Target.val,20261018),
        target_identity=Target.identity,target_manifest_sha256=Target.manifest_sha256,
        model_config={'G':{'width':64,'downs':8,'seed':20261018},'D_width':64})
    p['scaling']={'method':'inner61_valid_pixel_global_minmax_to_minus1_plus1_no_clip','min':0.,'max':1.,'fit_ids':p['split']['train']}
    pre=tmp_path/'pre'; pre.mkdir(); (pre/'gan.pt').write_bytes(b'metadata only')
    digest=sha256(pre/'gan.pt')
    write_json(pre/'complete.json',dict(format=1,phase='gan_pretrain',classification=p['classification'],
        complete=True,steps=1,checkpoint='gan.pt',checkpoint_sha256=digest,protocol=p))
    monkeypatch.setattr(contract,'Source',Source); monkeypatch.setattr(targets,'TargetDataset',Target)
    def forbidden(*a,**kw): pytest.fail('metadata check built model/loaded tensors/CUDA')
    monkeypatch.setattr(models,'create_models',forbidden); monkeypatch.setattr(torch,'load',forbidden)
    monkeypatch.setattr(torch.cuda,'is_available',forbidden)
    base=['check','--source','s','--raw-root','r','--targets','t','--baseline-run','b',
          '--pretrained-run',str(pre),'--pretrained-sha256',digest,'--output']
    assert main(base+[str(tmp_path/'joint')])==0
    assert not (tmp_path/'joint').exists()
    for output in (pre,pre/'nested',tmp_path):
        with pytest.raises(ValueError,match='outside'): main(base+[str(output)])


@pytest.mark.parametrize('phase',['gan_pretrain','joint','joint_changed_receipt'])
def test_phase_orchestration_cpu_arrays_and_immutable_handoff(tmp_path,monkeypatch,phase):
    # CPU harness substitutes CUDA transport/telemetry, not production numerical steps.
    import numpy as np
    import experiments.genseg_joint.runtime as runtime
    import experiments.genseg_joint.models as models
    from experiments.genseg_joint.data import inner_split
    from experiments.genseg30.contract import read_json,sha256
    class Target:
        train=[f't{i:02}' for i in range(76)]; val=[f'e{i:02}' for i in range(19)]
        protected=[]; identity={'test':True}; manifest_sha256='a'*64; _identities={}
        def __init__(self): self.reads=[]
        def require_no_unpack_writes(self): pass
        def training_dataset(self): return object()
        def inventory(self,p): return []
        def verify_inventory(self,inv): pass
        def verify_identity(self,i): assert i==self.identity
        def load_train(self,d,p):
            self.reads.append(p)
            x=np.arange(64*64,dtype=np.float32).reshape(1,1,64,64)/4096
            y=np.zeros_like(x); y[:,:,20:40,20:40]=1
            return x,y,{}
    target=Target(); split=inner_split(target.train,target.val,20261018)
    monkeypatch.setattr(runtime,'gpu_guard',lambda *a,**k:{'gpu_uuid':'CPU-harness','allocator_fraction':.5})
    for name,value in [('is_initialized',False),('is_available',True),('max_memory_allocated',0),('get_rng_state_all',[]),('is_current_stream_capturing',False)]:
        monkeypatch.setattr(torch.cuda,name,lambda v=value:v)
    monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',lambda *a:None)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    original_to=torch.Tensor.to
    def cpu_to(self,*a,**kw):
        if a and a[0]=='cuda:0': a=('cpu',)+a[1:]
        return original_to(self,*a,**kw)
    monkeypatch.setattr(torch.Tensor,'to',cpu_to)
    original=models.Generator
    def generator(*a,**kw):
        return original(width=1,downs=5,seed=20261018,device='cpu')
    monkeypatch.setattr(models,'Generator',generator)
    # Patch only model sizes in this CPU orchestration fixture; source data path remains real Cohort.
    def small_models(**kw):
        g=generator(); return g,g.discriminator(1),torch.nn.Conv2d(1,1,1)
    monkeypatch.setattr(models,'create_models',small_models)
    pre=tmp_path/'pre'
    if phase=='gan_pretrain':
        monkeypatch.setattr(models,'create_models',lambda **kw:pytest.fail('pretraining constructed S'))
        runtime.run(target,pre,gpu=1,steps=1,phase='gan_pretrain')
        r=read_json(pre/'complete.json')
        assert r['phase']=='gan_pretrain' and r['steps']==1
        assert set(target.reads)==set(split['train'])
        assert r['protocol']['optimizer']=='stateless_SGD_no_momentum_decay'
    else:
        # Produce a same-runtime metadata receipt, but tiny actual G/D tensors for CPU transport.
        from experiments.genseg_joint.pretrain import save_pretrain,policy
        from experiments.genseg_joint.data import Cohort
        from experiments.genseg_joint.core import PretrainEngine
        g,d=small_models()[:2]; e=PretrainEngine(g,d); e.steps=1
        scale=Cohort(target,20261018).fit_scale()
        p=dict(policy(),seed=20261018,steps=1,split=split,target_identity=target.identity,
            target_manifest_sha256=target.manifest_sha256,scaling=scale,
            model_config={'G':{'width':64,'downs':8,'seed':20261018},'D_width':64})
        pre.mkdir(); save_pretrain(pre,e,p,{})
        digest=sha256(pre/'gan.pt')
        # Tiny shape config is a fixture substitution only; retain strict production loader checks.
        from experiments.genseg_joint import pretrain
        original_load=pretrain.load_pretrained
        def load(*a,**kw):
            a[2].config=p['model_config']['G']
            receipt=original_load(*a,**kw)
            if phase=='joint_changed_receipt':
                path=pre/'complete.json'; path.write_bytes(path.read_bytes()+b'\n')
            return receipt
        monkeypatch.setattr(pretrain,'load_pretrained',load)
        output=tmp_path/'joint'
        if phase=='joint_changed_receipt':
            with pytest.raises(ValueError,match='immutable training input'):
                runtime.run(target,output,gpu=1,steps=1,pretrained_run=pre,pretrained_sha256=digest)
            assert not (output/'complete.json').exists()
            assert read_json(output/'failed.json')['completed_steps']==0
            return
        runtime.run(target,output,gpu=1,steps=1,pretrained_run=pre,pretrained_sha256=digest)
        r=read_json(output/'complete.json')
        assert r['steps']==1 and r['protocol']['pretrain_steps']==1
        assert r['protocol']['pretrained']['resume'] is False
        assert not set(target.reads)&set(split['external'])
        assert set(target.reads)&set(split['validation'])
        assert sha256(pre/'gan.pt')==digest


@pytest.mark.parametrize('flag',['--s-lr','--arch-lr','--pretrain-steps'])
def test_pretrain_rejects_joint_only_options(flag):
    from experiments.genseg_joint.cli import parser
    base=['pretrain','--source','s','--raw-root','r','--targets','t','--baseline-run','b',
          '--output','o','--gpu','1','--steps','1']
    with pytest.raises(SystemExit): parser().parse_args(base+[flag,'1'])



def test_checkpoint_load_without_python311_file_digest(tmp_path,monkeypatch):
    import hashlib
    from experiments.genseg_joint.pretrain import save_pretrain,load_pretrained
    from experiments.genseg30.contract import sha256
    g,d=tiny(); e=core.PretrainEngine(g,d); e.steps=1; p=protocol(g)
    save_pretrain(tmp_path,e,p,{})
    monkeypatch.delattr(hashlib,'file_digest',raising=False)
    load_pretrained(tmp_path,sha256(tmp_path/'gan.pt'),g,d,expected=p)
