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


def test_all_gpu_guard_rejects_foreign_busy_gpu_and_vram(monkeypatch):
    from types import SimpleNamespace
    from experiments.genseg_joint.runtime import gpu_guard
    import experiments.genseg3mm.targets as targets
    import experiments.genseg_joint.runtime as runtime
    monkeypatch.setattr(targets,'bind_gpu',lambda *a,**kw:{'gpu_uuid':'GPU-a'})
    for rows in ['0, GPU-a, 1, 1, 100\n1, GPU-b, 90, 1, 100',
                 '0, GPU-a, 1, 80, 100']:
        monkeypatch.setattr(runtime.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=rows))
        with pytest.raises(ValueError,match='80%'): gpu_guard(0)


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
