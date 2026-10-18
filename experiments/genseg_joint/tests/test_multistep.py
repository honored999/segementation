"""Multi-step production engines with CPU transport and simulated CUDA cache telemetry."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.genseg_joint import runtime, models, core
from experiments.genseg_joint.data import inner_split
from experiments.genseg30.contract import read_json


def harness(monkeypatch, *, phase, fault=None):
    import experiments.genseg3mm.targets as targets
    torch.set_num_threads(1)
    state={'cache':0, 'live':0, 'steps':0, 'events':[], 'ceilings':[], 'queries':0, 'waits':[]}
    class Target:
        train=[f't{i:02}' for i in range(76)]; val=[f'e{i:02}' for i in range(19)]
        protected=[]; identity={'synthetic':True}; manifest_sha256='a'*64; _identities={}
        def __init__(self): self.reads=[]
        def require_no_unpack_writes(self): pass
        def training_dataset(self): return object()
        def inventory(self,p): return []
        def verify_inventory(self,inv): pass
        def verify_identity(self,i): assert i==self.identity
        def load_train(self,d,p):
            self.reads.append(p)
            x=np.arange(4096,dtype=np.float32).reshape(1,1,64,64)/4096
            y=np.zeros_like(x); y[:,:,20:40,20:40]=1
            return x,y,{}
    def used(): return 1000+state['live']+state['cache']
    original_run=runtime.subprocess.run
    def query(command,**kw):
        if command[0]!='nvidia-smi': return original_run(command,**kw)
        state['events'].append('telemetry')
        state['queries']+=1
        if fault=='initial_busy': return SimpleNamespace(stdout='1, GPU-one, 80, 1000, 32768')
        if state['steps'] and fault=='malformed': return SimpleNamespace(stdout='bad')
        uid='GPU-other' if state['steps'] and fault=='uuid' else 'GPU-one'
        util=80 if state['steps'] and fault=='busy' else 2
        memory=27000 if state['steps'] and fault=='vram' else used()
        if fault=='budget_increase' and state['queries']==2: memory=0
        return SimpleNamespace(stdout=f'1, {uid}, {util}, {memory}, 32768')
    def snapshot(index):
        snap=dict(cpu_pct=2.,ram_used_pct=20.,ram_available=32*1024**3,
                  gpu_uuid='GPU-one',gpu_util_pct=2.,vram_available=(32768-used())*1024**2)
        if state['steps']:
            if fault=='cpu': snap['cpu_pct']=80.
            if fault=='ram': snap['ram_used_pct']=80.
            if fault=='headroom': snap['vram_available']=11*1024**3
        return snap
    monkeypatch.setattr(runtime.time,'sleep',state['waits'].append)
    monkeypatch.setattr(runtime.subprocess,'run',query)
    monkeypatch.setattr(targets,'_resource_snapshot',snapshot)
    for name,value in [('is_initialized',False),('is_available',True),('max_memory_allocated',0),
                       ('get_rng_state_all',[]),('is_current_stream_capturing',False)]:
        monkeypatch.setattr(torch.cuda,name,lambda v=value:v)
    monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',lambda f,*a:state['ceilings'].append(f))
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:state['events'].append('sync'))
    def release():
        assert state['events'][-1]=='sync'
        state['events'].append('release'); state['cache']=0
    monkeypatch.setattr(torch.cuda,'empty_cache',release)
    original_to=torch.Tensor.to
    def cpu_to(self,*a,**kw):
        if a and a[0]=='cuda:0': a=('cpu',)+a[1:]
        return original_to(self,*a,**kw)
    monkeypatch.setattr(torch.Tensor,'to',cpu_to)
    original_generator=models.Generator
    def generator(**kw):
        state['live']=4000
        g=original_generator(width=1,downs=5,seed=20261018,device='cpu')
        original_disc=g.discriminator
        g.discriminator=lambda *a:original_disc(1)
        return g
    monkeypatch.setattr(models,'Generator',generator)
    def create(**kw):
        g=generator(); return g,g.discriminator(),models.author_unet()
    monkeypatch.setattr(models,'create_models',create)
    cls=core.PretrainEngine if phase=='gan_pretrain' else core.JointEngine
    original_step=cls.step
    def step(self,*a):
        result=original_step(self,*a)
        state['steps']+=1; state['cache']=24000
        return result
    monkeypatch.setattr(cls,'step',step)
    return Target(),state


@pytest.mark.parametrize('phase',['gan_pretrain','joint'])
def test_three_steps_reclaims_only_unused_cache_and_preserves_roles(tmp_path,monkeypatch,phase):
    target,state=harness(monkeypatch,phase=phase)
    output=tmp_path/phase
    runtime.run(target,output,gpu=1,steps=3,phase=phase)
    receipt=read_json(output/'complete.json')
    entries=[json.loads(line) for line in (output/'steps.jsonl').read_text().splitlines()]
    assert receipt['complete'] and receipt['steps']==state['steps']==3
    assert [e['step'] for e in entries]==[1,2,3]
    split=inner_split(target.train,target.val,20261018)
    assert all(e['train']['patient'] in split['train'] for e in entries)
    if phase=='joint':
        assert all(e['internal_validation']['patient'] in split['validation'] for e in entries)
        assert all(e['alpha_gradient_norm']>0 and e['alpha_change_norm']>0 for e in entries)
    else: assert not any('internal_validation' in e for e in entries)
    assert not set(target.reads)&set(split['external'])
    assert state['events'].count('release')==3
    assert all(state['events'][i-1]=='sync' and state['events'][i+1]=='telemetry'
               for i,event in enumerate(state['events']) if event=='release')
    assert len(state['ceilings'])==2 and state['ceilings'][1]<=state['ceilings'][0]
    assert all(e['resources_before']['all_gpus'][0]['used_mib']==5000 for e in entries)
    assert 'empty_cache' in receipt['protocol']['guard']
    assert all(e['resources_before']['applied_allocator_fraction']==state['ceilings'][-1] for e in entries)
    payload=torch.load(output/receipt['checkpoint'],weights_only=True)
    assert payload['steps']==3
    assert all(torch.isfinite(v).all() for key in ('G','D') for v in payload[key].values())


@pytest.mark.parametrize('fault',['busy','vram','cpu','ram','headroom','uuid','malformed'])
def test_repeated_guard_still_fails_closed_after_cache_release(tmp_path,monkeypatch,fault):
    target,state=harness(monkeypatch,phase='gan_pretrain',fault=fault)
    output=tmp_path/'failed'
    with pytest.raises(ValueError): runtime.run(target,output,gpu=1,steps=3,phase='gan_pretrain')
    assert state['steps']==1
    assert not (output/'complete.json').exists()
    assert read_json(output/'failed.json')['completed_steps']==1
    # Only persistent utilization gets the exact bounded waits.
    assert state['waits']==([1]*5 if fault=='busy' else [])
    if fault=='busy':
        assert read_json(output/'failed.json')['resource_guard']['exhausted']
    assert state['events'].count('release')==2


def test_initial_guard_unchanged_no_cache_cleanup_or_output(tmp_path,monkeypatch):
    target,state=harness(monkeypatch,phase='gan_pretrain',fault='initial_busy')
    output=tmp_path/'blocked'
    with pytest.raises(ValueError,match='Resource guard refused'):
        runtime.run(target,output,gpu=1,steps=3,phase='gan_pretrain')
    assert not output.exists() and state['steps']==0
    assert state['events']==['telemetry'] and not state['ceilings']
    assert not state['waits']


def test_premodel_recheck_never_increases_initial_allocator_budget(tmp_path,monkeypatch):
    target,state=harness(monkeypatch,phase='gan_pretrain',fault='budget_increase')
    runtime.run(target,tmp_path/'budget',gpu=1,steps=3,phase='gan_pretrain')
    assert len(state['ceilings'])==2 and state['ceilings'][1]==state['ceilings'][0]
