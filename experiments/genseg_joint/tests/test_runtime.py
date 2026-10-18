import sys
import torch
import pytest
from experiments.genseg_joint.cli import parser
from experiments.genseg_joint.runtime import save_checkpoint
from experiments.genseg30.contract import read_json, sha256


def test_cli_explicit_budget_and_fixed_defaults():
    base=['--source','s','--raw-root','r','--targets','t','--baseline-run','b','--output','o']
    a=parser().parse_args(['check']+base)
    assert a.command=='check'
    with pytest.raises(SystemExit): parser().parse_args(['train']+base+['--gpu','0'])
    with pytest.raises(SystemExit): parser().parse_args(['train']+base+['--gpu','0','--steps','0'])
    a=parser().parse_args(['train']+base+['--gpu','0','--steps','1'])
    assert a.pretrain_steps==0 and a.arch_lr==1e-4 and a.s_lr==1e-5 and a.l1_weight==100.
    assert a.targets_sha256=='23d25e2b2a6d7c7ca2a270690a9d4d35116ff4f834458b48a9049ff889894a56'


def test_checkpoint_all_states_roundtrip_and_fresh(tmp_path):
    from types import SimpleNamespace
    net=torch.nn.Linear(1,1)
    engine=SimpleNamespace(g=SimpleNamespace(net=net,config={'width':1},
        networks=SimpleNamespace(upconv_arch=torch.ones(2,3),conv_arch=torch.zeros(2,3))),
        d=net,s=net,steps=1,optimizer_state=lambda:{'g':{'algorithm':'stateless_SGD','steps':1}})
    protocol={'classification':'engineering_only','complete':False,'seed':123,'split':{'train':['a']}}
    save_checkpoint(tmp_path,engine,protocol,{'numpy':{'state':123}})
    record=read_json(tmp_path/'complete.json')
    assert record['complete'] and record['checkpoint_sha256']==sha256(tmp_path/'joint.pt')
    payload=torch.load(tmp_path/'joint.pt',weights_only=True)
    assert set(payload)>={'G','D','S','arch','optimizers','protocol','rng','source'}
    assert payload['protocol']==protocol and payload['optimizers']['g']['steps']==1
    with pytest.raises(FileExistsError): save_checkpoint(tmp_path,engine,protocol,{})


def test_failed_postserialization_audit_never_marks_complete(tmp_path):
    from types import SimpleNamespace
    net=torch.nn.Linear(1,1)
    engine=SimpleNamespace(g=SimpleNamespace(net=net,config={},networks=SimpleNamespace(
        upconv_arch=torch.ones(2,3),conv_arch=torch.zeros(2,3))),d=net,s=net,steps=1,optimizer_state=lambda:{})
    def reject(): raise ValueError('inventory changed')
    with pytest.raises(ValueError,match='inventory changed'):
        save_checkpoint(tmp_path,engine,{}, {},final_audit=reject)
    assert (tmp_path/'joint.pt').is_file() and not (tmp_path/'complete.json').exists()


def selected_gpu_fixture(monkeypatch, *, rows=None, resources=None):
    from types import SimpleNamespace
    import experiments.genseg3mm.targets as targets
    import experiments.genseg_joint.runtime as runtime
    inventory={0:'0, GPU-zero, 95, 29491.2, 32768',1:'1, GPU-one, 3, 3276.8, 32768'}
    queries=[]
    def query(command,**kwargs):
        queries.append(command)
        if rows is not None: output=rows
        elif '-i' in command: output=inventory[int(command[command.index('-i')+1])]
        else: output='\n'.join(inventory.values())
        return SimpleNamespace(stdout=output)
    def snapshot(index):
        return dict(cpu_pct=2.,ram_used_pct=20.,ram_available=32*1024**3,
                    gpu_uuid='GPU-one' if index==1 else 'GPU-zero',gpu_util_pct=3.,
                    vram_available=32*1024**3) | (resources or {})
    # Substitute measured OS/GPU telemetry only; gpu_guard validates both snapshots locally.
    monkeypatch.setattr(targets,'_resource_snapshot',snapshot)
    monkeypatch.setattr(runtime.subprocess,'run',query)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','test-original')
    return queries


def test_selected_idle_gpu_ignores_busy_other_card(monkeypatch):
    from experiments.genseg_joint.runtime import gpu_guard
    queries=selected_gpu_fixture(monkeypatch)
    snap=gpu_guard(1,expected_uuid='GPU-one')
    assert snap['all_gpus']==[{'index':1,'uuid':'GPU-one','util_pct':3.,'used_mib':3276.8,'total_mib':32768.}]
    assert snap['allocator_fraction']==pytest.approx(.69)
    assert all(command[:3]==['nvidia-smi','-i','1'] for command in queries)
    assert all('--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total' in command for command in queries)
    with pytest.raises(ValueError,match=r'GPU 0.*95.*80'):
        gpu_guard(0)


@pytest.mark.parametrize('row', ['1, GPU-one, 80, 3276.8, 32768','1, GPU-one, 3, 26214.4, 32768'])
def test_selected_gpu_busy_util_or_vram_rejected(monkeypatch,row):
    from experiments.genseg_joint.runtime import gpu_guard
    selected_gpu_fixture(monkeypatch,rows=row)
    with pytest.raises(ValueError,match=r'GPU 1.*80'): gpu_guard(1)


@pytest.mark.parametrize('row', [
    '0, GPU-one, 3, 3276.8, 32768','1, GPU-wrong, 3, 3276.8, 32768',
    '1, GPU-one, nan, 10, 100','1, GPU-one, 3, inf, 100',
    '1, GPU-one, 3, 10, nan','1, GPU-one, -1, 10, 100',
    '1, GPU-one, 3, -1, 100','1, GPU-one, 3, 10, 0',
    '1, GPU-one, 3, 10, -100','1, GPU-one, invalid, 10, 100',
    '1, GPU-one, 3, 10', '',
    '1, GPU-one, 3, 3276.8, 32768\n0, GPU-zero, 95, 29491.2, 32768'])
def test_selected_gpu_malformed_metrics_or_identity_rejected(monkeypatch,row):
    from experiments.genseg_joint.runtime import gpu_guard
    selected_gpu_fixture(monkeypatch,rows=row)
    with pytest.raises(ValueError,match='GPU 1'): gpu_guard(1,expected_uuid='GPU-one')


def test_selected_gpu_expected_uuid_change_rejected(monkeypatch):
    from experiments.genseg_joint.runtime import gpu_guard
    selected_gpu_fixture(monkeypatch)
    with pytest.raises(ValueError,match=r'GPU 1.*UUID'): gpu_guard(1,expected_uuid='GPU-old')


@pytest.mark.parametrize('resources',[
    {'cpu_pct':80.},{'ram_used_pct':80.},{'gpu_util_pct':80.},
    {'ram_available':7*1024**3},{'vram_available':11*1024**3}])
def test_selected_idle_gpu_still_enforces_real_bind_resource_guards(monkeypatch,resources):
    from experiments.genseg_joint.runtime import gpu_guard
    selected_gpu_fixture(monkeypatch,resources=resources)
    with pytest.raises(ValueError,match='Resource guard refused'): gpu_guard(1)


def test_check_mode_never_constructs_model_or_reads_patient_arrays(tmp_path,monkeypatch,capsys):
    from experiments.genseg_joint.cli import main
    import experiments.genseg30.contract as contract
    import experiments.genseg3mm.targets as targets
    import experiments.genseg_joint.models as models
    class Source:
        def __init__(self,*a): pass
        def require_no_unpack_writes(self): pass
    class Target:
        train=[f't{i}' for i in range(76)]; val=[f'e{i}' for i in range(19)]; protected=[]
        def __init__(self,*a,**kw): pass
        def load_train(self,*a): raise AssertionError('check read a patient')
    monkeypatch.setattr(contract,'Source',Source)
    monkeypatch.setattr(targets,'TargetDataset',Target)
    def forbidden(*a,**kw): raise AssertionError('check constructed model/loaded weights')
    monkeypatch.setattr(models,'create_models',forbidden)
    monkeypatch.setattr(torch,'load',forbidden)
    output=tmp_path/'fresh'
    assert main(['check','--source','s','--raw-root','r','--targets','t','--baseline-run','b','--output',str(output)])==0
    assert not output.exists() and '(61, 15, 19)' in capsys.readouterr().out


def test_pinned_source_content_tamper_rejected(tmp_path,monkeypatch):
    import experiments.genseg_joint.models as models
    import shutil
    original=models.REPO
    for name in models.FILES|models.UNET_FILES:
        dst=tmp_path/'reference/GenSeg'/name; dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(original/'reference/GenSeg'/name,dst)
    monkeypatch.setattr(models,'REPO',tmp_path)
    assert models.source_identity()['commit']==models.UPSTREAM_COMMIT
    path=tmp_path/'reference/GenSeg/unet/unet_model.py'
    path.write_bytes(path.read_bytes()+b'\n# tampered')
    with pytest.raises(ValueError,match='Pinned'): models.source_identity()


@pytest.mark.parametrize('secondary',[False,True])
def test_between_steps_busy_then_idle(monkeypatch,secondary):
    from experiments.genseg_joint import runtime
    import experiments.genseg3mm.targets as targets
    selected_gpu_fixture(monkeypatch)
    calls=[]; waits=[]; events=[]
    if secondary:
        original=targets._resource_snapshot
        def sample(index):
            snap=original(index); calls.append(index)
            snap['gpu_util_pct']=81. if len(calls)==1 else 2.
            return snap
        monkeypatch.setattr(targets,'_resource_snapshot',sample)
    else:
        original=runtime.subprocess.run
        def sample(command,**kw):
            result=original(command,**kw); calls.append(command)
            if len(calls)==1: result.stdout='1, GPU-one, 81, 3276.8, 32768'
            return result
        monkeypatch.setattr(runtime.subprocess,'run',sample)
    monkeypatch.setattr(runtime.time,'sleep',waits.append)
    snap=runtime.between_step_guard(1,emit=events.append,expected_uuid='GPU-one')
    assert waits==[1] and snap['guard_attempts']==2 and snap['guard_waited_seconds']==1
    assert events[0]['attempt']==1 and len(calls)==2


def test_persistent_busy_exact_bound(monkeypatch):
    from experiments.genseg_joint import runtime
    selected_gpu_fixture(monkeypatch,rows='1, GPU-one, 81, 3276.8, 32768')
    waits=[]; events=[]
    monkeypatch.setattr(runtime.time,'sleep',waits.append)
    with pytest.raises(runtime.SelectedGPUUtilizationBusy) as caught:
        runtime.between_step_guard(1,emit=events.append,expected_uuid='GPU-one')
    assert waits==[1]*5 and len(events)==6
    assert caught.value.diagnostic['exhausted'] and events[-1]['attempt']==6


@pytest.mark.parametrize('resources',[
    {'cpu_pct':80.},{'ram_used_pct':80.},{'cpu_pct':float('nan')},
    {'gpu_util_pct':float('inf')},{'gpu_util_pct':101.},
    {'ram_available':7*1024**3},{'vram_available':11*1024**3},
    {'gpu_uuid':'wrong'},{'vram_available':float('nan')}])
def test_busy_with_permanent_secondary_fault_never_waits(monkeypatch,resources):
    from experiments.genseg_joint import runtime
    selected_gpu_fixture(monkeypatch,rows='1, GPU-one, 81, 3276.8, 32768',resources=resources)
    waits=[]; events=[]
    monkeypatch.setattr(runtime.time,'sleep',waits.append)
    with pytest.raises((ValueError,KeyError)) as caught:
        runtime.between_step_guard(1,emit=events.append,expected_uuid='GPU-one')
    assert not isinstance(caught.value,runtime.SelectedGPUUtilizationBusy)
    assert waits==events==[]


@pytest.mark.parametrize('row',[
    '0, GPU-one, 81, 3276.8, 32768','1, GPU-other, 81, 3276.8, 32768',
    '1, GPU-one, 81, 27000, 32768','1, GPU-one, 81, 22000, 32768',
    '1, GPU-one, nan, 3276.8, 32768','1, GPU-one, 101, 3276.8, 32768'])
def test_busy_with_primary_fault_never_waits(monkeypatch,row):
    from experiments.genseg_joint import runtime
    selected_gpu_fixture(monkeypatch,rows=row)
    waits=[]
    monkeypatch.setattr(runtime.time,'sleep',waits.append)
    with pytest.raises(ValueError) as caught:
        runtime.between_step_guard(1,emit=lambda e:None,expected_uuid='GPU-one')
    assert not isinstance(caught.value,runtime.SelectedGPUUtilizationBusy) and not waits


def test_secondary_busy_initial_guard_is_immediate(monkeypatch):
    from experiments.genseg_joint import runtime
    selected_gpu_fixture(monkeypatch,resources={'gpu_util_pct':81.})
    waits=[]
    monkeypatch.setattr(runtime.time,'sleep',waits.append)
    with pytest.raises(runtime.SelectedGPUUtilizationBusy): runtime.gpu_guard(1)
    assert not waits


def test_retry_does_not_catch_interrupt(monkeypatch):
    from experiments.genseg_joint import runtime
    selected_gpu_fixture(monkeypatch,rows='1, GPU-one, 81, 3276.8, 32768')
    def interrupt(seconds): raise KeyboardInterrupt()
    monkeypatch.setattr(runtime.time,'sleep',interrupt)
    with pytest.raises(KeyboardInterrupt): runtime.between_step_guard(1,emit=lambda e:None)
