import json
from pathlib import Path
import pytest
from standalone_nnunet2d.data.dataset import load_fold_cases

def test_external_single_fold(tmp_path):
    path = tmp_path / 'split.json'
    path.write_text(json.dumps([{'train':['external-a'],'val':['external-b']}]), encoding='utf-8')
    assert load_fold_cases(0, 'train', splits_file=path) == ('external-a',)
    assert load_fold_cases(0, 'val', splits_file=path) == ('external-b',)
    with pytest.raises(ValueError):
        load_fold_cases(1, 'train', splits_file=path)
    assert all(load_fold_cases(fold, 'train') for fold in range(5))

from copy import deepcopy
from dataclasses import replace
import random
import numpy as np
import torch
from torch import nn
from standalone_nnunet2d import formal_train as train
from standalone_nnunet2d.data.dataset import read_splits
from standalone_nnunet2d.data.data_source import RawNiftiCaseSource, PreprocessedB2ndCaseSource
from standalone_nnunet2d.engine import formal_validation as selection
from standalone_nnunet2d.training import formal_checkpoint as ck
from standalone_nnunet2d.training.official_config import OfficialTrainerSchedule, PolyLRScheduler
from standalone_nnunet2d.models.factory import get_model_contract
from standalone_nnunet2d.data.nifti_io import NiftiVolume, write_nifti


def split_file(tmp_path, folds=None):
    path=tmp_path/'splits.json'
    path.write_text(json.dumps(folds or [{'train':['external-a'],'val':['external-b']}]),encoding='utf-8')
    return path


@pytest.mark.parametrize('folds',[
    [], [{'train':[],'val':['b']}], [{'train':['a','a'],'val':['b']}],
    [{'train':['a'],'val':['a']}], [{'train':['../a'],'val':['b']}],
    [{'train':['a/b'],'val':['b']}], [{'train':['C:a'],'val':['b']}],
    [{'train':['a\\b'],'val':['b']}], [{'train':[3],'val':['b']}],
    [{'train':['a'],'val':['b'],'extra':[]}],
])
def test_bad_splits_rejected(tmp_path,folds):
    path=tmp_path/'bad.json'; path.write_text(json.dumps(folds),encoding='utf-8')
    with pytest.raises(ValueError): read_splits(path)


def test_patient_crossing_and_missing_mapping(tmp_path):
    path=split_file(tmp_path)
    for mapping in ({'external-a':'patient','external-b':'patient'},{'external-a':'patient'}):
        with pytest.raises(ValueError): read_splits(path,patient_map=mapping)
    assert read_splits(path,patient_map={'external-a':'p1','external-b':'p2'})


def write_source(tmp_path, source):
    root=tmp_path/source; root.mkdir()
    label=np.zeros((2,4,4),dtype=np.int16); label[0,1,1]=1
    image=np.arange(32,dtype=np.float32).reshape(2,4,4)
    for case in ('external-a','external-b'):
        if source=='raw_nifti_online':
            write_nifti(root/'imagesTr'/f'{case}_0000.nii.gz',NiftiVolume(image,(.4892368018627167,.4892368018627167,3.),(0,0,0)))
            write_nifti(root/'labelsTr'/f'{case}.nii.gz',NiftiVolume(label,(.4892368018627167,.4892368018627167,3.),(0,0,0)))
        else:
            blosc=pytest.importorskip('blosc2')
            for suffix,arr in (('',image),('_seg',label)):
                blosc.save(blosc.asarray(arr[None],chunks=(1,1,4,4)),str(root/f'{case}{suffix}.b2nd'))
    (root/'dataset.json').write_text(json.dumps({'channel_names':{'0':'DWI'},'labels':{'background':0,'lesion':1}}),encoding='utf-8')
    return root


@pytest.mark.parametrize('source',['raw_nifti_online','nnunet_preprocessed_b2nd'])
def test_real_sources_patch_and_selection_share_external_split(tmp_path,monkeypatch,source):
    path=split_file(tmp_path); root=write_source(tmp_path,source)
    ds_train,ds_val=train.build_formal_datasets(root,fold=0,patch_size=(4,4),use_mask_for_norm=(False,),data_source=source,splits_file=path)
    assert ds_train.case_ids==('external-a',) and ds_val.case_ids==('external-b',)
    ds_train.augment=False
    assert ds_train[0][0].shape==(1,4,4)
    assert ds_val[0][1].shape==(4,4)
    monkeypatch.setattr(selection,'predict_volume',lambda model,image,device,**kw: np.zeros(image.array.shape,dtype=np.int16))
    for case_source in (None,ds_val._case_source):
        result=selection.select_fold(object(),root,data_source=source,fold=0,device=torch.device('cpu'),case_source=case_source,splits_file=path)
        assert result['case_ids']==('external-b',)
    with pytest.raises(ValueError):
        selection.select_fold(object(),root,data_source=source,fold=0,device=torch.device('cpu'),case_source=ds_train._case_source,splits_file=path)
    source_type=RawNiftiCaseSource if source=='raw_nifti_online' else PreprocessedB2ndCaseSource
    with pytest.raises(ValueError): source_type(root,fold=0,split='train',case_ids=('external-b',),splits_file=path)


def identity_fixture(tmp_path):
    root=tmp_path/'source';root.mkdir()
    (root/'dataset.json').write_text(json.dumps({'channel_names':{'0':'DWI'},'labels':{'background':0,'lesion':1}}),encoding='utf-8')
    plans=tmp_path/'plans.json'; plans.write_text(json.dumps({'configurations':{'2d':{'patch_size':[512,512],'use_mask_for_norm':[False]}}}),encoding='utf-8')
    split=split_file(tmp_path)
    kw=dict(stage='pretrain',dataset_id='Dataset508_ISLES2022DWI',splits_file=split,plans=plans,fold=0,data_source='nnunet_preprocessed_b2nd',data_root=root,patch_size=(512,512),model_name='h2former',source_version='synthetic-v1')
    return kw


def test_identity_and_finetune_fixed_split(tmp_path):
    kw=identity_fixture(tmp_path); identity=train.build_source_identity(**kw)
    assert identity['stage']=='pretrain'
    ck.validate_source_identity(identity,{**identity,'root':'/relocated'})
    for key in ('stage','dataset_id','split_sha256','plans_sha256','cases_sha256','source_version','type'):
        with pytest.raises(ValueError): ck.validate_source_identity(identity,{**identity,key:'changed'})
    for changes in ({'dataset_id':'Dataset501_DWI'},{'splits_file':None},{'patch_size':(4,4)},{'model_name':'h2former_lite_upernet'}):
        with pytest.raises(ValueError): train.build_source_identity(**{**kw,**changes})
    with pytest.raises(ValueError): train.build_source_identity(**{**kw,'stage':'finetune','dataset_id':'Dataset501'})
    target=train.build_source_identity(**{**kw,'stage':'finetune','dataset_id':'Dataset501','splits_file':None})
    assert target['dataset_id']=='Dataset501'
    changed=json.loads(kw['plans'].read_text());changed['note']='different'
    kw['plans'].write_text(json.dumps(changed),encoding='utf-8')
    with pytest.raises(ValueError): ck.validate_source_identity(identity,train.build_source_identity(**kw))


class Tiny(nn.Module):
    def __init__(self):
        super().__init__(); self.weight=nn.Parameter(torch.zeros(2,1));self.register_buffer('running',torch.ones(2));self.register_buffer('count',torch.tensor(0))


def checkpoint_fixture(tmp_path):
    kw=identity_fixture(tmp_path); identity=train.build_source_identity(**kw)
    model=Tiny(); model.weight.data.fill_(7);model.running.fill_(8);model.count.fill_(9)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0001);scheduler=PolyLRScheduler(optimizer,1000)
    config=train.build_formal_config(fold=0,epochs=1000,schedule=OfficialTrainerSchedule(),model_name='h2former',optimizer_name='adamw')
    config.update(source_identity=identity,initialization_provenance=None)
    path=tmp_path/'pretrain.pth'
    ck.save_formal_checkpoint(model,optimizer,scheduler,path,ck.FormalTrainerState(200,50000,.9,0,.8,190,.8,10),config,checkpoint_root=tmp_path)
    return path,config,model


def test_strict_weights_only_transfers_all_buffers_no_training_rng(tmp_path):
    path,config,source=checkpoint_fixture(tmp_path)
    target=Tiny();opt=torch.optim.AdamW(target.parameters(),lr=.0001);sched=PolyLRScheduler(opt,1000)
    random.seed(51);np.random.seed(52);torch.manual_seed(53)
    before=(random.getstate(),deepcopy(np.random.get_state()),torch.get_rng_state().clone())
    opt_before=deepcopy(opt.state_dict());sched_before=deepcopy(sched.__dict__)
    provenance=ck.initialize_pretrained_model(target,path)
    assert all(torch.equal(v,source.state_dict()[k]) for k,v in target.state_dict().items())
    assert opt.state_dict()==opt_before and sched.ctr==sched_before['ctr']
    assert random.getstate()==before[0] and np.array_equal(np.random.get_state()[1],before[1][1]) and torch.equal(torch.get_rng_state(),before[2])
    assert provenance['source_identity']==config['source_identity'] and len(provenance['sha256'])==64


@pytest.mark.parametrize('bad',['model','mode','channels','labels','stage','missing','extra','shape','dtype','nan','buffer','schema'])
def test_rejected_transfer_is_atomic(tmp_path,bad):
    path,config,_=checkpoint_fixture(tmp_path);payload,_=ck.read_formal_payload(path)
    if bad=='model':payload['metadata']['model_name']='h2former_lite_upernet'
    elif bad=='mode':payload['metadata']['supervision_mode']='deep_supervision'
    elif bad=='channels':payload['metadata']['resolved_config']['model']['in_channels']=2
    elif bad=='labels':payload['metadata']['resolved_config']['source_identity']['labels']={'background':0,'lesion':2}
    elif bad=='stage':payload['metadata']['resolved_config']['source_identity']['stage']='finetune'
    elif bad=='missing':del payload['model_state_dict']['count']
    elif bad=='extra':payload['model_state_dict']['extra']=torch.zeros(1)
    elif bad=='shape':payload['model_state_dict']['weight']=torch.zeros(3,1)
    elif bad=='dtype':payload['model_state_dict']['weight']=payload['model_state_dict']['weight'].double()
    elif bad=='nan':payload['model_state_dict']['running'][0]=float('nan')
    elif bad=='buffer':payload['model_state_dict']['count']=torch.tensor(1.)
    else:payload['format_version']=42
    torch.save(payload,path);target=Tiny();before=deepcopy(target.state_dict())
    with pytest.raises(ValueError):ck.initialize_pretrained_model(target,path)
    assert all(torch.equal(v,before[k]) for k,v in target.state_dict().items())


@pytest.mark.parametrize('field',['stage','dataset_id','split_sha256','plans_sha256','source_version'])
def test_resume_identity_rejected_before_model_optimizer_rng(tmp_path,field):
    path,config,_=checkpoint_fixture(tmp_path);target=Tiny();opt=torch.optim.AdamW(target.parameters());sched=PolyLRScheduler(opt,1000)
    before=deepcopy(target.state_dict());rng=torch.get_rng_state().clone();opt_before=deepcopy(opt.state_dict())
    with pytest.raises(ValueError):ck.load_formal_checkpoint(target,opt,sched,path,fold=0,checkpoint_root=tmp_path,source_identity={**config['source_identity'],field:'changed'})
    assert opt.state_dict()==opt_before and torch.equal(torch.get_rng_state(),rng)
    assert all(torch.equal(v,before[k]) for k,v in target.state_dict().items())


def test_stage_resume_restores_full_state_after_root_relocation(tmp_path):
    path,config,source=checkpoint_fixture(tmp_path);target=Tiny();opt=torch.optim.AdamW(target.parameters(),lr=.3);sched=PolyLRScheduler(opt,1000)
    result=ck.load_formal_checkpoint(target,opt,sched,path,fold=0,checkpoint_root=tmp_path,source_identity={**config['source_identity'],'root':'relocated'})
    assert result.epoch==200 and result.global_step==50000 and result.state.checks_without_improvement==10
    assert all(torch.equal(v,source.state_dict()[k]) for k,v in target.state_dict().items())
    assert opt.param_groups[0]['lr']==.0001


def test_cli_mutually_exclusive():
    with pytest.raises(SystemExit): train.build_parser().parse_args(['--raw-root','raw','--plans','plan','--output-root','out','--resume','a','--pretrained-checkpoint','b'])


def test_output_both_overlap_directions_and_existing(tmp_path):
    data=tmp_path/'data';data.mkdir();source=tmp_path/'source';source.mkdir();weights=source/'weights.pth'
    for out in (data,data/'child',tmp_path,source/'child'):
        with pytest.raises(ValueError):train.validate_stage_output(out,data_root=data,pretrained=weights)
    existing=tmp_path/'existing';existing.mkdir()
    with pytest.raises(ValueError):train.validate_stage_output(existing,data_root=data)
    train.validate_stage_output(tmp_path/'new',data_root=data)
    protected=Path(train.__file__).resolve().parents[1]
    with pytest.raises(ValueError):train.validate_stage_output(protected/'new',data_root=data)
    run=tmp_path/'run';run.mkdir();(run/'resolved_config.json').write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError):train.validate_stage_output(run/'nested',data_root=data)
    with pytest.raises(ValueError):train.validate_stage_output(tmp_path/'wrong',data_root=data,resume=run/'checkpoint_latest.pth')


def test_dry_run_no_model_loading_training_or_output(tmp_path,monkeypatch):
    kw=identity_fixture(tmp_path);out=tmp_path/'new-run'
    def forbidden(*a,**k): raise AssertionError('dry-run touched compute')
    for name in ('build_model','build_formal_datasets','run_formal_epochs','initialize_pretrained_model','write_resolved_config'):
        monkeypatch.setattr(train,name,forbidden)
    args=['--stage','pretrain','--dataset-id',kw['dataset_id'],'--splits-file',str(kw['splits_file']),'--preprocessed-root',str(kw['data_root']),'--plans',str(kw['plans']),'--output-root',str(out),'--model','h2former','--supervision-mode','single_output','--optimizer','adamw','--device','cpu']
    assert train.main(args)==0 and not out.exists()


def test_early_stop_100_boundary_check_count_strict_best():
    state=ck.FormalTrainerState(99,1,.5,0,.5,90,.5,9)
    state,best,stop=train.update_selection_state(state,.5005,completed_epoch=99)
    assert best and not stop and state.checks_without_improvement==0
    state,best,stop=train.update_selection_state(state,.5006,completed_epoch=100)
    assert best and not stop and state.checks_without_improvement==1 and state.early_stop_reference_dice==.5
    for epoch in range(110,200,10): state,best,stop=train.update_selection_state(state,.5006,completed_epoch=epoch)
    assert state.checks_without_improvement==10 and stop
    state,best,stop=train.update_selection_state(state,.502,completed_epoch=200)
    assert state.checks_without_improvement==0 and not stop


def test_legacy_entry_cannot_resume_marked_stage(tmp_path):
    path,config,_=checkpoint_fixture(tmp_path)
    model=Tiny();before=deepcopy(model.state_dict());opt=torch.optim.AdamW(model.parameters())
    with pytest.raises(ValueError,match='stage'):
        ck.load_formal_checkpoint(model,opt,path,fold=0,checkpoint_root=tmp_path)
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())


def test_reused_source_entire_split_conflict(tmp_path,monkeypatch):
    path=split_file(tmp_path); root=write_source(tmp_path,'raw_nifti_online')
    source=RawNiftiCaseSource(root,fold=0,split='val',splits_file=path)
    other=tmp_path/'other.json';other.write_text(json.dumps([{'train':['changed'],'val':['external-b']}]),encoding='utf-8')
    monkeypatch.setattr(selection,'predict_volume',lambda model,image,device,**kw:np.zeros(image.array.shape,dtype=np.int16))
    with pytest.raises(ValueError,match='split'):
        selection.select_fold(object(),root,data_source='raw_nifti_online',fold=0,device=torch.device('cpu'),case_source=source,splits_file=other)


def test_main_finetune_resume_without_source_file_preserves_terminal_state(tmp_path,monkeypatch):
    source_dir=tmp_path/'external-run';source_dir.mkdir()
    path,source_config,_=checkpoint_fixture(source_dir)
    root=tmp_path/'target-data';root.mkdir()
    (root/'dataset.json').write_text(json.dumps({'channel_names':{'0':'DWI'},'labels':{'background':0,'lesion':1}}),encoding='utf-8')
    plans=source_dir/'plans.json'
    out=tmp_path/'finetune-run'
    models=[]
    def build(*args,**kwargs):
        model=Tiny();models.append(model);return model
    monkeypatch.setattr(train,'build_model',build)
    monkeypatch.setattr(train,'build_formal_datasets',lambda *args,**kwargs: ('train','val'))
    monkeypatch.setattr(train,'build_formal_loaders',lambda *args,**kwargs:([],[]))
    monkeypatch.setattr(train,'run_formal_epochs',lambda **kwargs:iter(()))
    args=['--stage','finetune','--preprocessed-root',str(root),'--plans',str(plans),'--output-root',str(out),'--model','h2former','--supervision-mode','single_output','--optimizer','adamw','--device','cpu','--confirm-run']
    assert train.main(args+['--pretrained-checkpoint',str(path)])==0
    config=json.loads((out/'resolved_config.json').read_text())
    provenance=config['initialization_provenance']
    assert provenance['sha256'] and models[0].weight[0,0]==7
    models[0].weight.data.fill_(13)
    opt=torch.optim.AdamW(models[0].parameters(),lr=.0001);sched=PolyLRScheduler(opt,1000)
    target=out/'checkpoint_latest.pth'
    ck.save_formal_checkpoint(models[0],opt,sched,target,ck.FormalTrainerState(200,50000,.8,0,.8,190,.8,10),config,checkpoint_root=out)
    path.unlink() # Only this test's tiny synthetic source checkpoint.
    def forbidden(*args,**kwargs):raise AssertionError('terminal resume reinitialized or entered training')
    monkeypatch.setattr(train,'initialize_pretrained_model',forbidden)
    monkeypatch.setattr(train,'run_formal_epochs',forbidden)
    monkeypatch.setattr(train,'select_fold',forbidden)
    assert train.main(args+['--resume',str(target)])==0
    assert models[-1].weight[0,0]==13
    assert json.loads((out/'resolved_config.json').read_text())['initialization_provenance']==provenance
    # A moved source protocol must be rejected before constructing a target model.
    before=len(models)
    with pytest.raises(SystemExit):train.main(args+['--resume',str(target),'--source-version','different'])
    assert len(models)==before


def test_legacy_checkpoint_cannot_be_external_resume(tmp_path):
    path,config,_=checkpoint_fixture(tmp_path);payload,_=ck.read_formal_payload(path)
    del payload['metadata']['resolved_config']['source_identity']
    torch.save(payload,path)
    target=Tiny();opt=torch.optim.AdamW(target.parameters())
    with pytest.raises(ValueError,match='historical'):
        ck.load_formal_checkpoint(target,opt,path,fold=0,checkpoint_root=tmp_path,source_identity=config['source_identity'])


def test_pretrain_manifest_contract_and_raw_spacing_rejection(tmp_path):
    kw=identity_fixture(tmp_path)
    with pytest.raises(ValueError,match='fixed in-plane'):
        train.build_source_identity(**{**kw,'data_source':'raw_nifti_online'})
    manifest=kw['data_root']/'dataset.json'
    for data in ({'channel_names':{'0':'DWI','1':'ADC'},'labels':{'background':0,'lesion':1}}, {'channel_names':{'0':'DWI'},'labels':{'background':0,'lesion':2}}):
        manifest.write_text(json.dumps(data),encoding='utf-8')
        with pytest.raises(ValueError,match='single-channel'):train.build_source_identity(**kw)


def test_resume_provenance_mismatch_is_preload_atomic(tmp_path):
    path,config,_=checkpoint_fixture(tmp_path);payload,_=ck.read_formal_payload(path)
    payload['metadata']['resolved_config']['initialization_provenance']={'sha256':'source-a'}
    torch.save(payload,path)
    target=Tiny();before=deepcopy(target.state_dict());opt=torch.optim.AdamW(target.parameters())
    with pytest.raises(ValueError,match='provenance'):
        ck.load_formal_checkpoint(target,opt,path,fold=0,checkpoint_root=tmp_path,source_identity=config['source_identity'],initialization_provenance={'sha256':'source-b'})
    assert all(torch.equal(v,before[k]) for k,v in target.state_dict().items())


def test_checkpoint_config_alias_source_conflict_rejected(tmp_path):
    path,_,_=checkpoint_fixture(tmp_path);payload,_=ck.read_formal_payload(path)
    payload['metadata']['config']=deepcopy(payload['metadata']['config'])
    payload['metadata']['config']['source_identity']['dataset_id']='different'
    torch.save(payload,path)
    target=Tiny();before=deepcopy(target.state_dict())
    with pytest.raises(ValueError,match='aliases'):ck.initialize_pretrained_model(target,path)
    assert all(torch.equal(v,before[k]) for k,v in target.state_dict().items())


def test_normalized_output_boundary(tmp_path):
    data=tmp_path/'data';data.mkdir()
    with pytest.raises(ValueError):train.validate_stage_output(data/'unused'/'..'/'new',data_root=data)
    source=tmp_path/'other-preprocessed';source.mkdir()
    with pytest.raises(ValueError):train.validate_stage_output(source/'new',data_root=data,protected_roots=(source,))


def test_patient_mapping_in_manifest_and_fractional_plan_rejected(tmp_path):
    kw=identity_fixture(tmp_path)
    manifest=kw['data_root']/'dataset.json';data=json.loads(manifest.read_text())
    data['patient_map']={'external-a':'same-patient','external-b':'same-patient'}
    manifest.write_text(json.dumps(data),encoding='utf-8')
    with pytest.raises(ValueError,match='patient'):train.build_source_identity(**kw)
    del data['patient_map'];manifest.write_text(json.dumps(data),encoding='utf-8')
    plan=json.loads(kw['plans'].read_text());plan['configurations']['2d']['patch_size']=[512.5,512]
    kw['plans'].write_text(json.dumps(plan),encoding='utf-8')
    with pytest.raises(ValueError,match='exactly'):train.build_source_identity(**kw)


def assert_deep_equal(a, b):
    if isinstance(b, torch.Tensor): assert torch.equal(a, b)
    elif isinstance(b, np.ndarray): assert np.array_equal(a, b)
    elif isinstance(b, dict):
        assert a.keys() == b.keys()
        for k in b: assert_deep_equal(a[k], b[k])
    elif isinstance(b, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b): assert_deep_equal(x, y)
    else: assert a == b


@pytest.mark.parametrize('field,value', [('root', None), ('root', 7), ('dataset_id', 42), ('source_version', []), ('plans_sha256', 'invalid'), ('dataset_metadata_sha256', None), ('patient_map_sha256', False), ('type', 'unknown'), ('channels', True)])
def test_source_schema_rejection_before_initialization(tmp_path, field, value):
    path, _, _ = checkpoint_fixture(tmp_path); payload, _ = ck.read_formal_payload(path)
    source = payload['metadata']['resolved_config']['source_identity']
    if value is None: source.pop(field)
    else: source[field] = value
    torch.save(payload, path); target = Tiny(); before = deepcopy(target.state_dict())
    with pytest.raises(ValueError, match='source'): ck.initialize_pretrained_model(target, path)
    assert_deep_equal(target.state_dict(), before)


@pytest.mark.parametrize('bad', ['model', 'stage', 'root'])
@pytest.mark.parametrize('confirmed', [False, True])
def test_invalid_source_cli_rejected_before_compute(tmp_path, monkeypatch, bad, confirmed):
    source_dir = tmp_path / 'external-run'; source_dir.mkdir()
    path, _, _ = checkpoint_fixture(source_dir); payload, _ = ck.read_formal_payload(path)
    if bad == 'model':
        payload['metadata']['model_name'] = 'plain_conv_unet'
        payload['metadata']['resolved_config']['model'] = get_model_contract('plain_conv_unet', supervision_mode='single_output').as_dict()
    elif bad == 'stage': payload['metadata']['resolved_config']['source_identity']['stage'] = 'finetune'
    else: del payload['metadata']['resolved_config']['source_identity']['root']
    torch.save(payload, path)
    root = tmp_path / 'target'; root.mkdir()
    (root / 'dataset.json').write_text(json.dumps({'channel_names': {'0': 'DWI'}, 'labels': {'background': 0, 'lesion': 1}}), encoding='utf-8')
    out = tmp_path / 'new-run'
    def forbidden(*args, **kwargs): raise AssertionError('invalid source reached compute')
    for name in ('build_model', 'build_formal_datasets', 'build_formal_loaders'): monkeypatch.setattr(train, name, forbidden)
    args = ['--stage', 'finetune', '--preprocessed-root', str(root), '--plans', str(source_dir / 'plans.json'), '--output-root', str(out), '--model', 'h2former', '--supervision-mode', 'single_output', '--optimizer', 'adamw', '--device', 'cpu', '--pretrained-checkpoint', str(path)]
    with pytest.raises(SystemExit): train.main(args + (['--confirm-run'] if confirmed else []))
    assert not out.exists()


@pytest.mark.parametrize('bad', ['optimizer_groups', 'scheduler_step', 'epoch', 'rng_python', 'rng_numpy', 'rng_torch'])
@pytest.mark.parametrize('marked', [True, False])
def test_resume_corruption_preserves_all_state(tmp_path, bad, marked):
    path, config, _ = checkpoint_fixture(tmp_path); payload, _ = ck.read_formal_payload(path)
    if not marked: del payload['metadata']['resolved_config']['source_identity']
    if bad == 'optimizer_groups': payload['optimizer_state_dict']['param_groups'] = []
    elif bad == 'scheduler_step': payload['metadata']['scheduler_state']['step'] = 'invalid'
    elif bad == 'epoch': del payload['metadata']['epoch']
    elif bad == 'rng_python': payload['metadata']['rng_state']['python'] = ()
    elif bad == 'rng_numpy': payload['metadata']['rng_state']['numpy'] = ('invalid',)
    else: payload['metadata']['rng_state']['torch_cpu'] = torch.zeros(2)
    torch.save(payload, path)
    model = Tiny(); opt = torch.optim.AdamW(model.parameters(), lr=.03)
    model.weight.sum().backward(); opt.step(); opt.zero_grad()
    sched = PolyLRScheduler(opt, 1000); sched.step(5)
    before = deepcopy((model.state_dict(), opt.state_dict(), {k:v for k,v in sched.__dict__.items() if k != 'optimizer'}, ck.capture_rng_state()))
    with pytest.raises((ValueError, KeyError, TypeError, RuntimeError)):
        ck.load_formal_checkpoint(model, opt, sched, path, fold=0, checkpoint_root=tmp_path, **({'source_identity':config['source_identity']} if marked else {}))
    assert_deep_equal((model.state_dict(), opt.state_dict(), {k:v for k,v in sched.__dict__.items() if k != 'optimizer'}, ck.capture_rng_state()), before)


@pytest.mark.parametrize('bad', [{}, {'sha256': 'x'}, {'path': 3}, {'source_identity': {}}])
def test_malformed_resume_provenance_is_contract_error(tmp_path, bad):
    path, config, _ = checkpoint_fixture(tmp_path); payload, _ = ck.read_formal_payload(path)
    payload['metadata']['resolved_config']['initialization_provenance'] = bad
    torch.save(payload, path); model = Tiny(); before = deepcopy(model.state_dict()); opt = torch.optim.AdamW(model.parameters())
    with pytest.raises(ValueError, match='provenance'):
        ck.load_formal_checkpoint(model, opt, path, fold=0, checkpoint_root=tmp_path, source_identity=config['source_identity'], initialization_provenance=bad)
    assert_deep_equal(model.state_dict(), before)


@pytest.mark.parametrize('failure', ['model', 'optimizer', 'scheduler', 'rng'])
def test_resume_application_failure_rolls_back_and_preserves_error(tmp_path, monkeypatch, failure):
    path, config, _ = checkpoint_fixture(tmp_path)
    model = Tiny(); opt = torch.optim.AdamW(model.parameters(), lr=.03)
    model.weight.sum().backward(); opt.step(); opt.zero_grad()
    sched = PolyLRScheduler(opt, 1000); sched.step(5)
    before = deepcopy((model.state_dict(), opt.state_dict(), {k:v for k,v in sched.__dict__.items() if k != 'optimizer'}, ck.capture_rng_state()))
    error = RuntimeError('injected application failure')
    def fail_model(*a, **k):
        model.weight.data.fill_(42); model.running.fill_(43); raise error
    def fail_optimizer(optimizer, *a):
        if optimizer is opt:
            opt.param_groups[0]['lr'] = 4; opt.state[model.weight]['exp_avg'] = torch.full_like(model.weight, 4)
            raise error
    def fail_scheduler(*a, **k):
        sched.ctr = 42; opt.param_groups[0]['lr'] = 43; random.random(); raise error
    def fail_rng(*a, **k):
        random.seed(42); np.random.seed(43); torch.manual_seed(44); raise error
    if failure == 'model': monkeypatch.setattr(model, 'load_state_dict', fail_model)
    elif failure == 'optimizer': opt.register_load_state_dict_post_hook(fail_optimizer)
    elif failure == 'scheduler': monkeypatch.setattr(sched, 'step', fail_scheduler)
    else: monkeypatch.setattr(ck, '_restore_rng_state', fail_rng)
    # Capture any instance method added by the fault injection as well.
    before = deepcopy((model.state_dict(), opt.state_dict(), {k:v for k,v in sched.__dict__.items() if k != 'optimizer'}, ck.capture_rng_state()))
    with pytest.raises(RuntimeError) as caught:
        ck.load_formal_checkpoint(model, opt, sched, path, fold=0, checkpoint_root=tmp_path, source_identity=config['source_identity'])
    assert caught.value is error
    assert sched.optimizer is opt
    assert_deep_equal((model.state_dict(), opt.state_dict(), {k:v for k,v in sched.__dict__.items() if k != 'optimizer'}, ck.capture_rng_state()), before)


def test_resume_rollback_error_keeps_original_exception(tmp_path, monkeypatch):
    path, config, _ = checkpoint_fixture(tmp_path)
    model = Tiny(); opt = torch.optim.AdamW(model.parameters()); sched = PolyLRScheduler(opt, 1000)
    original = RuntimeError('original apply error'); rollback = RuntimeError('rollback RNG error')
    def fail(*a, **k): raise original
    def fail_rollback(*a, **k): raise rollback
    monkeypatch.setattr(model, 'load_state_dict', fail)
    monkeypatch.setattr(ck.random, 'setstate', fail_rollback)
    with pytest.raises(RuntimeError) as caught:
        ck.load_formal_checkpoint(model, opt, sched, path, fold=0, checkpoint_root=tmp_path, source_identity=config['source_identity'])
    assert caught.value is original and caught.value.__cause__ is rollback
    assert any('rollback failed' in note for note in caught.value.__notes__)


def test_pretrained_loader_revalidates_changed_source(tmp_path):
    path, _, _ = checkpoint_fixture(tmp_path); payload, _ = ck.read_formal_payload(path)
    ck.validate_pretrained_payload(payload)
    del payload['metadata']['resolved_config']['source_identity']['root']
    torch.save(payload, path)
    model = Tiny(); before = deepcopy(model.state_dict())
    with pytest.raises(ValueError, match='source'): ck.initialize_pretrained_model(model, path)
    assert_deep_equal(model.state_dict(), before)


@pytest.mark.parametrize('confirmed', [False, True])
def test_cli_resume_missing_provenance_root_is_early_contract_error(tmp_path, monkeypatch, confirmed):
    source_dir = tmp_path / 'external'; source_dir.mkdir()
    path, _, _ = checkpoint_fixture(source_dir)
    provenance = ck.initialize_pretrained_model(Tiny(), path)
    del provenance['source_identity']['root']
    root = tmp_path / 'target'; root.mkdir()
    (root / 'dataset.json').write_text(json.dumps({'channel_names': {'0':'DWI'}, 'labels': {'background':0, 'lesion':1}}), encoding='utf-8')
    plans = source_dir / 'plans.json'
    identity = train.build_source_identity(stage='finetune', dataset_id='Dataset501', splits_file=None, plans=plans, fold=0, data_source='nnunet_preprocessed_b2nd', data_root=root, patch_size=(512,512), model_name='h2former')
    config = train.build_formal_config(fold=0, epochs=1000, schedule=OfficialTrainerSchedule(), model_name='h2former', optimizer_name='adamw')
    config.update(source_identity=identity, initialization_provenance=provenance)
    out = tmp_path / 'run'; out.mkdir()
    (out / 'resolved_config.json').write_text(json.dumps(config), encoding='utf-8')
    target = out / 'checkpoint_latest.pth'; model = Tiny(); opt = torch.optim.AdamW(model.parameters()); sched = PolyLRScheduler(opt,1000)
    ck.save_formal_checkpoint(model,opt,sched,target,ck.FormalTrainerState(200,50000,.8,0),config,checkpoint_root=out)
    def forbidden(*a, **k): raise AssertionError('malformed provenance reached compute')
    monkeypatch.setattr(train,'build_model',forbidden); monkeypatch.setattr(train,'build_formal_datasets',forbidden)
    args = ['--stage','finetune','--preprocessed-root',str(root),'--plans',str(plans),'--output-root',str(out),'--model','h2former','--supervision-mode','single_output','--optimizer','adamw','--device','cpu','--resume',str(target)]
    before = {p.name:p.read_bytes() for p in out.iterdir()}
    with pytest.raises(SystemExit): train.main(args + (['--confirm-run'] if confirmed else []))
    assert before == {p.name:p.read_bytes() for p in out.iterdir()}
    with pytest.raises(ValueError, match='provenance'):
        ck.load_formal_checkpoint(model,opt,sched,target,fold=0,checkpoint_root=out,source_identity=identity,initialization_provenance=provenance)


def test_full_resume_restores_nonempty_optimizer_state(tmp_path):
    path, config, source = checkpoint_fixture(tmp_path)
    opt = torch.optim.AdamW(source.parameters(),lr=.0001); source.weight.sum().backward(); opt.step(); opt.zero_grad()
    sched = PolyLRScheduler(opt,1000); sched.step(17)
    state = ck.FormalTrainerState(18,4500,.8,0)
    ck.save_formal_checkpoint(source,opt,sched,path,state,config,checkpoint_root=tmp_path)
    target = Tiny(); target_opt = torch.optim.AdamW(target.parameters(),lr=.03)
    target.weight.sum().backward(); target_opt.step(); target_opt.zero_grad()
    target_sched = PolyLRScheduler(target_opt,1000)
    restored = ck.load_formal_checkpoint(target,target_opt,target_sched,path,fold=0,checkpoint_root=tmp_path,source_identity=config['source_identity'])
    assert restored.state == state and restored.scheduler_step == 17
    assert_deep_equal(target.state_dict(), source.state_dict())
    assert_deep_equal(target_opt.state_dict(), opt.state_dict())
