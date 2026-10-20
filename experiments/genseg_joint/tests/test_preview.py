import numpy as np
import pytest
from experiments.genseg_joint import preview


def test_selection_and_ignore():
    y=np.zeros((5,3,4),dtype=np.int16); y[0]=-1; y[2,1,1]=1
    selected,missing=preview.select_planes(y)
    assert selected==[(2,'lesion'),(1,'empty'),(3,'empty')]
    assert missing==[]
    selected,missing=preview.select_planes(np.full((2,3,4),-1))
    assert selected==[] and set(missing)=={'lesion','empty_1','empty_2','all_planes_ignore'}


def test_renderer_native_common_window(tmp_path):
    from PIL import Image
    x=np.array([[0,5,10],[-10,30,3]],dtype=np.float32)
    y=np.array([[0,1,0],[-1,0,0]],dtype=np.int16)
    path=tmp_path/'panels.png'
    report=preview.render(path,x,y,x,{'min':0.,'max':10.})
    im=np.array(Image.open(path)); h=preview.TITLE_HEIGHT
    assert im.shape==(2+h,12,3)
    assert np.array_equal(im[h:,:3],im[h:,6:9])
    assert (im[h+1,0]==0).all() and (im[h+1,6]==0).all()
    assert (im[h,10]==[204,51,51]).all()
    assert report['target_clipped_high']==1 and report['target_clipped_low']==0


def test_cli_preview_contract():
    from experiments.genseg_joint.cli import parser
    base=['preview','--source','a','--raw-root','b','--targets','c','--baseline-run','d','--output','e',
          '--run','r','--checkpoint-sha256','a'*64,'--gpu','1']
    args=parser().parse_args(base)
    assert args.patient_limit==4 and args.phase=='joint'
    assert parser().parse_args(base+['--phase','gan_pretrain']).phase=='gan_pretrain'
    for value in ('0','62'):
        with pytest.raises(SystemExit): parser().parse_args(base+['--patient-limit',value])
    with pytest.raises(SystemExit): parser().parse_args(base[:-3]+['bad','--gpu','1'])

from copy import deepcopy
from types import SimpleNamespace
import json
import torch
from experiments.genseg30.contract import write_json,sha256,REPO
from experiments.genseg_joint.data import inner_split
from experiments.genseg_joint.models import Generator,source_identity


class Target:
    def __init__(self,root):
        self.train=[f'case{i:03}' for i in range(76)]; self.val=[f'outer{i:03}' for i in range(19)]
        self.identity={'source':'synthetic'}; self.manifest_sha256='a'*64
        self.protected=[REPO,root/'protected']; self._identities={}; self.reads=[]; self.changed=False
    def verify_identity(self,value):
        if self.changed or value!=self.identity: raise ValueError('target changed')
    def inventory(self,p): return {'synthetic':p}
    def verify_inventory(self,value):
        assert set(value)==set(self.train)
        if self.changed: raise ValueError('target changed')
    def require_no_unpack_writes(self): pass
    def training_dataset(self): return SimpleNamespace(identifiers=self.train)
    def load_train(self,dataset,p):
        self.reads.append(p)
        x=np.arange(4*9*11,dtype=np.float32).reshape(1,4,9,11)
        y=np.zeros_like(x,dtype=np.int16); y[:,0]=-1; y[0,2,3:5,4:6]=1
        return x,y,{}


def protocol(target):
    split=inner_split(target.train,target.val,20261018)
    return {'classification':preview.CLASSIFICATION,'phase':'joint','steps':1,'seed':20261018,
        'split':split,'target_identity':target.identity,'target_manifest_sha256':target.manifest_sha256,
        'source':source_identity(),'models':{'G_width':64,'G_downs':8,'D_width':64,
            'S':'exact_author_UNet(1,1)_width64','dtype':'float32'},
        'normalization':'inner61_only_valid_pixel_minmax_minus1_plus1_no_clip',
        'pad':'native_full_planes_bottom_right_to_multiple256; image0_label-1_no_resize',
        'architecture':'upconv_only; unused_plain_conv_alpha_frozen_not_updated',
        'BN':'eval_frozen_buffers_all_virtual_and_actual_calls_deviation',
        'all_real_input':'same_3mm_masked_target_for_G_D_S_internal_validation',
        'scaling':{'method':'inner61_valid_pixel_global_minmax_to_minus1_plus1_no_clip',
            'min':-5.,'max':500.,'fit_ids':split['train']},'implementation_files_sha256':{'old.py':'b'*64}}


def run_files(tmp_path,target):
    folder=tmp_path/'run';folder.mkdir()
    (folder/'joint.pt').write_bytes(b'safe metadata fixture only')
    receipt={'format':1,'classification':preview.CLASSIFICATION,'complete':True,'steps':1,
        'checkpoint':'joint.pt','checkpoint_sha256':sha256(folder/'joint.pt'),'protocol':protocol(target)}
    write_json(folder/'complete.json',receipt)
    return folder,receipt


@pytest.mark.parametrize('field',['phase','split','source','target_identity','target_manifest_sha256','models','scaling','BN','normalization','pad'])
def test_bad_receipt_before_tensor_or_patient_load(tmp_path,monkeypatch,field):
    target=Target(tmp_path);folder,receipt=run_files(tmp_path,target)
    receipt['protocol'][field]='wrong'
    (folder/'complete.json').write_text(json.dumps(receipt))
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('loaded tensors'))
    monkeypatch.setattr(preview,'Generator',lambda *a,**k:pytest.fail('constructed G'))
    with pytest.raises(ValueError): preview.run_preview(target,folder,receipt['checkpoint_sha256'],tmp_path/'output',gpu=0)
    assert not target.reads and not (tmp_path/'output').exists()


@pytest.mark.parametrize('damage',['hash','complete','classification'])
def test_bad_identity(tmp_path,damage):
    t=Target(tmp_path);folder,r=run_files(tmp_path,t)
    if damage=='hash': (folder/'joint.pt').write_bytes(b'tampered')
    else:
        r[damage]=False if damage=='complete' else 'gan_pretrain'
        (folder/'complete.json').write_text(json.dumps(r))
    with pytest.raises(ValueError):preview.audit_joint(t,folder,r['checkpoint_sha256'])


def tiny_payload(g,receipt):
    return {'format':1,'classification':receipt['classification'],'steps':1,'protocol':receipt['protocol'],
        'G':{k:v.clone() for k,v in g.net.state_dict().items()},'arch':{k:getattr(g.networks,k).clone() for k in ('conv_arch','upconv_arch')},
        'model_config':{'G':g.config,'S':'pinned_author_UNet(1,1,bilinear=False)'},'D':{},'S':{},'optimizers':{},'rng':{},'source':receipt['protocol']['source']}


def test_real_pinned_generator_restore_generate(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    g=Generator(width=1,downs=5,seed=20261018)
    t=Target(tmp_path);_,r=run_files(tmp_path,t)
    payload=tiny_payload(g,r)
    payload['arch']['upconv_arch'].add_(.3)
    serialized=tmp_path/'tiny.pt'
    torch.save(payload,serialized)
    payload=torch.load(serialized,map_location='cpu',weights_only=True)
    fresh=Generator(width=1,downs=5,seed=20261018)
    preview.restore_generator(fresh,payload)
    n=fresh.networks
    assert n._upconv_arch_parameters[0] is n.upconv_arch and n._arch_parameters[1] is n.conv_arch
    assert torch.equal(n.upconv_arch,payload['arch']['upconv_arch'])
    assert not fresh.net.training and not any(p.requires_grad for p in fresh.net.parameters())
    before={k:v.clone() for k,v in fresh.net.state_dict().items()}
    label=np.zeros((33,45),dtype=np.int16);label[5:10,6:11]=1;label[:2]=-1
    a,pad=preview.generate(fresh,label,{'min':10.,'max':30.})
    b,_=preview.generate(fresh,label,{'min':10.,'max':30.})
    assert a.shape==label.shape and pad==[256,256] and np.array_equal(a,b) and np.isfinite(a).all()
    assert (a[:2]==0).all() and (a[2:]>=10).all() and (a[2:]<=30).all()
    assert all(torch.equal(v,before[k]) for k,v in fresh.net.state_dict().items())
    example=__import__('os').environ.get('GENSEG_PREVIEW_SYNTHETIC_OUTPUT')
    if example:
        from pathlib import Path
        out=Path(example);out.mkdir(parents=True,exist_ok=False)
        yy,xx=np.indices((96,96));real=(xx+yy).astype(np.float32)/190*20+10
        mask=np.zeros((96,96),dtype=np.int16);mask[((xx-55)**2+(yy-45)**2)<100]=1;mask[:4]=-1
        fake,_=preview.generate(fresh,mask,{'min':10.,'max':30.})
        preview.render(out/'synthetic.png',real,mask,fake,{'min':10.,'max':30.})


@pytest.mark.parametrize('damage',['keys','shape','dtype','finite','arch'])
def test_invalid_tensor_transaction(tmp_path,damage):
    g=Generator(width=1,downs=5,seed=20261018);t=Target(tmp_path);_,r=run_files(tmp_path,t)
    p=tiny_payload(g,r);key=next(k for k,v in p['G'].items() if v.dtype==torch.float32)
    before={k:v.clone() for k,v in g.net.state_dict().items()}
    if damage=='keys':p['G'].pop(key)
    if damage=='shape':p['G'][key]=torch.ones(2)
    if damage=='dtype':p['G'][key]=p['G'][key].double()
    if damage=='finite':p['G'][key].fill_(float('nan'))
    if damage=='arch':p['arch']['upconv_arch'].fill_(float('nan'))
    with pytest.raises(ValueError):preview.restore_generator(g,p)
    assert all(torch.equal(v,before[k]) for k,v in g.net.state_dict().items())


def test_safe_cpu_load_and_receipt_binding(tmp_path,monkeypatch):
    t=Target(tmp_path);folder,r=run_files(tmp_path,t)
    g=Generator(width=1,downs=5,seed=20261018);p=tiny_payload(g,r)
    # Metadata load validates production width before G mutation; no large weights needed.
    p['model_config']['G']={'width':64,'downs':8,'seed':20261018}
    torch.save(p,folder/'joint.pt');digest=sha256(folder/'joint.pt')
    original=torch.load;calls=[]
    def load(*a,**kw):calls.append(kw);return original(*a,**kw)
    monkeypatch.setattr(torch,'load',load)
    result=preview.load_joint(folder,digest,r)
    assert calls==[{'map_location':'cpu','weights_only':True}]
    assert result['source']==r['protocol']['source']
    result['protocol']=deepcopy(result['protocol']);result['protocol']['seed']=7
    torch.save(result,folder/'joint.pt')
    with pytest.raises(ValueError,match='receipt'):preview.load_joint(folder,sha256(folder/'joint.pt'),r)


@pytest.mark.parametrize('phase',['joint','gan_pretrain'])
@pytest.mark.parametrize('tamper',[None,'target','receipt','png'])
def test_orchestration_inner61_and_completion(tmp_path,monkeypatch,tamper,phase):
    t=Target(tmp_path)
    folder,r=gan_run_files(tmp_path,t) if phase=='gan_pretrain' else run_files(tmp_path,t)
    out=tmp_path/'output';options={} if phase=='joint' else {'phase':phase}
    monkeypatch.setattr(preview,'gpu_guard',lambda *a,**k:{'allocator_fraction':.1})
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:False)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',lambda *a:None)
    # Substitute hardware/model size only; real metadata, cohort, role gate, audits, renderer execute.
    monkeypatch.setattr(preview,'load_joint',lambda *a:{})
    monkeypatch.setattr(preview,'load_gan',lambda *a:{})
    monkeypatch.setattr(preview,'Generator',lambda **k:object())
    monkeypatch.setattr(preview,'restore_generator',lambda *a:None)
    monkeypatch.setattr(preview,'generate',lambda g,y,s:(np.full(y.shape,123.,np.float32),[256,256]))
    monkeypatch.setattr(preview.Cohort,'fit_scale',lambda *a:pytest.fail('refit forbidden'))
    original=preview.render;count=[0]
    def render(*a):
        result=original(*a);count[0]+=1
        if tamper=='target':t.changed=True
        if tamper=='receipt':(folder/'complete.json').write_text('changed')
        if tamper=='png' and count[0]==3:
            next(out.glob('*.png')).write_bytes(b'tampered')
        return result
    monkeypatch.setattr(preview,'render',render)
    if tamper:
        with pytest.raises(ValueError):preview.run_preview(t,folder,r['checkpoint_sha256'],out,gpu=0,patient_limit=1,**options)
        assert not (out/'preview.json').exists()
    else:
        preview.run_preview(t,folder,r['checkpoint_sha256'],out,gpu=0,**options)
        manifest=json.loads((out/'preview.json').read_text())
        assert manifest['complete'] and len(manifest['previews'])==12
        assert manifest['phase']==phase and manifest['checkpoint']==r['checkpoint']
        assert 'weights_only=True, map_location=cpu' in manifest['cpu_payload']
        assert [(v['z'],v['category']) for v in manifest['previews']]==[(2,'lesion'),(1,'empty'),(3,'empty')]*4
        assert all(v['display']['window']==[-5.,500.] for v in manifest['previews'])
        assert t.reads==sorted(r['protocol']['split']['train'])[:4]
        assert set(t.reads).isdisjoint(r['protocol']['split']['validation']+r['protocol']['split']['external'])
        assert manifest['scaling']==r['protocol']['scaling']
        assert all(v['native_shape']==[9,11] for v in manifest['previews'])
        with pytest.raises(ValueError,match='Fresh'):preview.run_preview(t,folder,r['checkpoint_sha256'],out,gpu=0,**options)


def test_protected_paths_before_arrays(tmp_path):
    t=Target(tmp_path);folder,r=run_files(tmp_path,t)
    for out in (folder/'preview',folder.parent,t.protected[0]/'preview'):
        with pytest.raises(ValueError):preview.run_preview(t,folder,r['checkpoint_sha256'],out,gpu=0)
    assert not t.reads


def test_blended_overlay_preserves_contrast(tmp_path):
    from PIL import Image
    x=np.array([[0.,10.]])
    path=tmp_path/'blend.png';r=preview.render(path,x,np.ones_like(x),x,{'min':0.,'max':10.})
    pixels=np.array(Image.open(path))[preview.TITLE_HEIGHT]
    assert pixels[6].tolist()==[153,0,0] and pixels[7].tolist()==[255,102,102]
    assert r['overlay']['alpha']==.6


def test_exact_inverse_scale_mask_padding_and_no_grad():
    class Net(torch.nn.Module):
        def forward(self,x):
            assert not torch.is_grad_enabled()
            assert x.shape==(1,1,256,256)
            assert set(x.unique().tolist())=={0.,1.}
            assert x[0,0,2,3]==1 and x[0,0,0,0]==0
            assert x[0,0,40:,40:].count_nonzero()==0
            return torch.full_like(x,.25)
    y=np.zeros((33,35),dtype=np.int16);y[0,0]=-1;y[2,3]=1
    result,pad=preview.generate(SimpleNamespace(net=Net(),device='cpu'),y,{'min':-10.,'max':30.})
    assert result.shape==(33,35) and pad==[256,256]
    assert result[0,0]==0 and (result[y>=0]==15.).all()


def test_empty_and_missing_categories():
    chosen,missing=preview.select_planes(np.zeros((1,3,4),dtype=np.int16))
    assert chosen==[(0,'empty')] and missing==['lesion','empty_2']
    chosen,missing=preview.select_planes(np.ones((1,3,4),dtype=np.int16))
    assert chosen==[(0,'lesion')] and missing==['empty_1','empty_2']


def test_all_selected_ignore_never_complete(tmp_path,monkeypatch,capsys):
    t=Target(tmp_path);folder,r=run_files(tmp_path,t);out=tmp_path/'output'
    monkeypatch.setattr(preview,'gpu_guard',lambda *a,**k:{'allocator_fraction':.1})
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:False)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',lambda *a:None)
    monkeypatch.setattr(preview,'load_joint',lambda *a:{})
    monkeypatch.setattr(preview,'Generator',lambda **k:object())
    monkeypatch.setattr(preview,'restore_generator',lambda *a:None)
    def ignored(dataset,p):
        t.reads.append(p)
        return np.zeros((1,2,3,4),np.float32),np.full((1,2,3,4),-1,np.int16),{}
    monkeypatch.setattr(t,'load_train',ignored)
    with pytest.raises(ValueError,match='No renderable planes'):
        preview.run_preview(t,folder,r['checkpoint_sha256'],out,gpu=0,patient_limit=1)
    assert not (out/'preview.json').exists()
    assert 'PREVIEW COMPLETE' not in capsys.readouterr().out
    failure=json.loads((out/'failed.json').read_text())
    assert failure['complete'] is False and failure['patients'][0]['missing_categories']==[
        'lesion','empty_1','empty_2','all_planes_ignore']
    assert failure['patients'][0]['all_ignore_planes']==[0,1]


def gan_run_files(tmp_path,target):
    from experiments.genseg_joint.pretrain import policy
    folder,r=run_files(tmp_path,target);p=r['protocol']
    (folder/'gan.pt').write_bytes(b'safe metadata fixture only')
    r.update(phase='gan_pretrain',classification=policy()['classification'],checkpoint='gan.pt',
        checkpoint_sha256=sha256(folder/'gan.pt'),protocol=dict(policy(),
            **{k:p[k] for k in ('steps','seed','split','target_identity','target_manifest_sha256','scaling','implementation_files_sha256')},
            model_config={'G':{'width':64,'downs':8,'seed':20261018},'D_width':64}))
    (folder/'complete.json').write_text(json.dumps(r))
    return folder,r


@pytest.mark.parametrize('damage',['complete','phase','hash','scaling','model_config','rates'])
def test_gan_receipt_rejected_before_model_or_arrays(tmp_path,monkeypatch,damage):
    t=Target(tmp_path);folder,r=gan_run_files(tmp_path,t);out=tmp_path/'output'
    if damage=='hash': (folder/'gan.pt').write_bytes(b'changed')
    elif damage in ('complete','phase'): r[damage]=False if damage=='complete' else 'joint'
    else: r['protocol'][damage]='wrong'
    (folder/'complete.json').write_text(json.dumps(r))
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('loaded tensors'))
    monkeypatch.setattr(preview,'Generator',lambda *a,**k:pytest.fail('constructed G'))
    monkeypatch.setattr(preview,'gpu_guard',lambda *a,**k:pytest.fail('reached GPU guard'))
    with pytest.raises(ValueError):
        preview.run_preview(t,folder,r['checkpoint_sha256'],out,gpu=0,phase='gan_pretrain')
    assert not t.reads and not out.exists()

def test_gan_safe_cpu_load_same_handle_and_receipt(tmp_path,monkeypatch):
    t=Target(tmp_path);folder,r=gan_run_files(tmp_path,t);path=folder/'gan.pt'
    payload={k:r[k] for k in ('format','phase','classification','steps','protocol')}
    payload.update(G={'weight':torch.ones(1)},D={},arch={},rng={},optimizers={
        k:{'algorithm':'stateless_SGD','lr':v,'momentum':0.,'weight_decay':0.,'steps':r['steps'],'state':{}}
        for k,v in r['protocol']['rates'].items()})
    torch.save(payload,path);r['checkpoint_sha256']=sha256(path)
    (folder/'complete.json').write_text(json.dumps(r))
    preview.audit_gan(t,folder,r['checkpoint_sha256'])
    original=torch.load;original_hash=preview.hash_stream;loads=[];hashes=[]
    def load(stream,**kw):loads.append((stream,kw));return original(stream,**kw)
    def stream_hash(stream):hashes.append(stream);return original_hash(stream)
    monkeypatch.setattr(torch,'load',load);monkeypatch.setattr(preview,'hash_stream',stream_hash)
    result=preview.load_gan(folder,r['checkpoint_sha256'],r)
    assert len(loads)==1 and loads[0][1]=={'map_location':'cpu','weights_only':True}
    assert hashes==[loads[0][0],loads[0][0]] and result['G']['weight'].device.type=='cpu'
    result['steps']+=1;torch.save(result,path)
    with pytest.raises(ValueError,match='receipt'):preview.load_gan(folder,sha256(path),r)


@pytest.mark.parametrize('when',['before','during'])
def test_gan_checkpoint_hash_before_and_after_load(tmp_path,monkeypatch,when):
    t=Target(tmp_path);folder,r=gan_run_files(tmp_path,t);path=folder/'gan.pt'
    def load(stream,**kw):
        assert when=='during'
        with path.open('ab') as writer:writer.write(b'changed during load')
        return {}
    monkeypatch.setattr(torch,'load',load)
    digest='0'*64 if when=='before' else r['checkpoint_sha256']
    with pytest.raises(ValueError,match='checkpoint changed'):preview.load_gan(folder,digest,r)

def test_gan_generator_only_restore_preserves_all_alpha_aliases(tmp_path,monkeypatch):
    from experiments.genseg_joint import models
    monkeypatch.setattr(Generator,'discriminator',lambda *a,**k:pytest.fail('constructed D'))
    monkeypatch.setattr(models,'author_unet',lambda *a,**k:pytest.fail('constructed S'))
    g=Generator(width=1,downs=5,seed=20261018)
    t=Target(tmp_path);_,r=gan_run_files(tmp_path,t);p=tiny_payload(g,r)
    for key in ('model_config','S','source'):p.pop(key)
    p['phase']='gan_pretrain';p['protocol']=deepcopy(r['protocol'])
    p['protocol']['model_config']['G']=g.config
    p['arch']['upconv_arch'].add_(.3)
    preview.restore_generator(g,p);n=g.networks
    assert all(torch.equal(v,p['G'][k]) for k,v in g.net.state_dict().items())
    assert torch.equal(n.upconv_arch,p['arch']['upconv_arch'])
    assert n._conv_arch_parameters[0] is n.conv_arch and n._upconv_arch_parameters[0] is n.upconv_arch
    assert n._arch_parameters[0] is n.upconv_arch and n._arch_parameters[1] is n.conv_arch
    assert not n.conv_arch.requires_grad and not n.upconv_arch.requires_grad
    assert not g.net.training and not any(v.requires_grad for v in g.net.parameters())


@pytest.mark.parametrize('phase',['joint','gan_pretrain'])
def test_wrong_preview_phase_before_load(tmp_path,monkeypatch,phase):
    t=Target(tmp_path)
    folder,r=gan_run_files(tmp_path,t) if phase=='joint' else run_files(tmp_path,t)
    (folder/('joint.pt' if phase=='joint' else 'gan.pt')).write_bytes(b'safe metadata fixture only')
    monkeypatch.setattr(preview,'gpu_guard',lambda *a,**k:pytest.fail('reached GPU guard'))
    with pytest.raises(ValueError):
        preview.run_preview(t,folder,r['checkpoint_sha256'],tmp_path/'output',gpu=0,phase=phase)
    assert not t.reads

# Diagnostic snapshot fixtures stay tiny; only the externally approved file identity
# and production dimensions are substituted, never the audit/restore implementation.
def diagnostic_files(tmp_path, monkeypatch):
    from experiments.genseg_joint import diagnostic_preview as dp, diagnostic as d, convergence as c
    t=Target(tmp_path); folder=tmp_path/'diagnostic'; folder.mkdir()
    real_receipt=read_json_fixture()
    p=deepcopy(real_receipt['protocol'])
    p.update(target_identity=t.identity,target_manifest_sha256=t.manifest_sha256,
        input_inventory={q:t.inventory(q) for q in t.train})
    p['scaling']['fit_ids']=inner_split(t.train,t.val,20261018)['train']
    cohort=preview.Cohort(t,20261018)
    x,y,_,_,selection=d.fixed_plane(cohort,p['scaling']); p['selection']=selection
    g=Generator(width=1,downs=5,seed=20261018)
    with torch.no_grad():
        for key,v in g.net.state_dict().items():
            if 'running_mean' in key:v.fill_(.2)
            if 'num_batches_tracked' in key:v.fill_(7)
    g.networks.upconv_arch.add_(.2)
    ready={'format':1,'classification':d.CLASSIFICATION,'diagnostic_only':True,'complete':False,
        'stage':'ready_after_fixed_plane_before_models_or_updates','protocol':p,
        'policy':d.READY_POLICY,'arrays':d.ready_arrays(x,y)}
    write_json(folder/'ready.json',ready)
    payload={'format':1,'classification':c.CLASSIFICATION,'diagnostic_only':True,
        'not_pretraining':True,'not_joint_handoff':True,'no_resume':True,'training_receipt':False,
        'checkpoint_name':'not_gan_not_joint','steps':500,'seed':20261018,
        'models':{'G':{k:v.clone() for k,v in g.net.state_dict().items()},'D':{},
            'alpha':{k:getattr(g.networks,k).clone() for k in ('conv_arch','upconv_arch')}},
        'model_config':{'G':g.config,'D_width':64},'source':p['source'],'protocol':p,
        'input':{'ready_file':'ready.json','ready_sha256':sha256(folder/'ready.json'),
            'selection':selection,'scaling':p['scaling'],'condition':'label==1','metric_region':'label>=0'}}
    # ready is immutable; producer later adds only this resource ceiling.
    p['applied_allocator_fraction']=.79
    torch.save(payload,folder/c.SNAPSHOT_FILE)
    digest=sha256(folder/c.SNAPSHOT_FILE); size=(folder/c.SNAPSHOT_FILE).stat().st_size
    r=deepcopy(real_receipt);r['protocol']=p
    r['convergence']['snapshot'].update(sha256=digest,bytes=size)
    r['output_logs_sha256'].update({'ready.json':sha256(folder/'ready.json'),c.SNAPSHOT_FILE:digest})
    for name in ('started.json','steps.jsonl','guard_events.jsonl'):
        (folder/name).write_text('{}')
        r['output_logs_sha256'][name]=sha256(folder/name)
    write_json(folder/'trajectory.json',{'classification':c.CLASSIFICATION,'diagnostic_only':True,
        'plan':c.convergence_plan(500),'scheduled_points_complete':True})
    r['output_logs_sha256']['trajectory.json']=sha256(folder/'trajectory.json')
    r['convergence'].update(trajectory_file='trajectory.json',trajectory_sha256=sha256(folder/'trajectory.json'))
    write_json(folder/'diagnostic.json',r)
    monkeypatch.setattr(dp,'EXPECTED_SHA256',digest);monkeypatch.setattr(dp,'EXPECTED_BYTES',size)
    monkeypatch.setattr(dp,'MODEL_CONFIG',payload['model_config'])
    ready=json.loads((folder/'ready.json').read_text(encoding='utf-8-sig'))
    return dp,t,folder,r,ready,payload,g


def read_json_fixture():
    # Stable synthetic protocol based on the producer's actual public constants.
    from experiments.genseg_joint import diagnostic as d, convergence as c
    from experiments.genseg_joint.runtime import implementation_identity
    p={'format':1,'classification':d.CLASSIFICATION,'diagnostic_only':True,'seed':20261018,
       'steps':500,'strategy':'adam_trainbn_epoch_v1','convergence_audit':True,
       'convergence_plan':c.convergence_plan(500),'convergence_policy':c.CONVERGENCE_POLICY,
       'initialization':'identical_seed_cold_G_D_alpha_buffers; fresh_Adam_zero_moments; no_resume',
       'g_objective':'gan_l1','architecture':'both_alpha_frozen','repeat_output_policy':d.REPEAT_OUTPUT_POLICY,
       'probe_backend_policy':d.PROBE_BACKEND_POLICY,'source':source_identity(),
       'repository_commit':'830642cc9ad07c84211188369d5315ad59b6dfcc',
       'implementation_files_sha256':implementation_identity(),
       'scaling':{'method':'inner61_valid_pixel_global_minmax_to_minus1_plus1_no_clip','min':-5.,'max':500.}}
    return {'format':1,'classification':d.CLASSIFICATION,'diagnostic_only':True,'complete':True,
       'completed_steps':500,'weights_saved':True,'reusable_checkpoint_saved':False,'protocol':p,
       'convergence':{'plan':c.convergence_plan(500),'policy':c.CONVERGENCE_POLICY,
           'scheduled_points_complete':True,'snapshot':{'format':1,'file':c.SNAPSHOT_FILE,'steps':500,
           'not_pretraining':True,'not_joint_handoff':True,'no_resume':True,'optimizer_moments':False,
           'rng_state':False,'weights_only_load':True,'readback_exact':True}},'output_logs_sha256':{}}


@pytest.mark.parametrize('damage',[None,'hash','schema','seed','scale','source','ready','finite','shape','receipt_seed','receipt_scale','receipt_source','receipt_ready','failed','code'])
def test_diagnostic_audit_restore(tmp_path,monkeypatch,damage):
    torch.set_num_threads(1)
    dp,t,folder,r,ready,payload,g=diagnostic_files(tmp_path,monkeypatch)
    dp.audit_diagnostic(t,folder,dp.EXPECTED_SHA256)
    path=folder/'diagnostic_state.pt'
    if damage in ('receipt_seed','receipt_scale','receipt_source','receipt_ready','failed','code'):
        if damage=='receipt_seed':r['protocol']['seed']=7
        if damage=='receipt_scale':r['protocol']['scaling']['fit_ids']=[]
        if damage=='receipt_source':r['protocol']['source']={}
        if damage=='receipt_ready':r['output_logs_sha256']['ready.json']='0'*64
        if damage=='code':r['protocol']['implementation_files_sha256']['models.py']='0'*64
        if damage=='failed':(folder/'failed.json').write_text('{}')
        (folder/'diagnostic.json').write_text(json.dumps(r))
        with pytest.raises(ValueError):dp.audit_diagnostic(t,folder,dp.EXPECTED_SHA256)
        return
    if damage=='hash':
        path.write_bytes(b'damaged')
        with pytest.raises(ValueError):dp.audit_diagnostic(t,folder,dp.EXPECTED_SHA256)
        return
    if damage=='schema':payload['unexpected']=True
    if damage=='seed':payload['seed']=7
    if damage=='scale':payload['input']['scaling']=dict(min=0,max=1)
    if damage=='source':payload['source']={}
    if damage=='ready':payload['input']['ready_sha256']='0'*64
    key=next(k for k,v in payload['models']['G'].items() if v.dtype==torch.float32)
    if damage=='finite':payload['models']['G'][key].fill_(float('nan'))
    if damage=='shape':payload['models']['G'][key]=torch.ones(3)
    torch.save(payload,path)
    if damage in ('schema','seed','scale','source','ready'):
        with pytest.raises(ValueError):dp.load_diagnostic(folder,sha256(path),r,ready)
        return
    restored=dp.load_diagnostic(folder,sha256(path),r,ready)
    fresh=Generator(width=1,downs=5,seed=20261018)
    if damage:
        with pytest.raises(ValueError):preview.restore_generator(fresh,restored)
        return
    preview.restore_generator(fresh,restored)
    preview.restore_generator(g,restored)
    manifest,arrays=dp.coverage(preview.Cohort(t,20261018),r['protocol']['scaling'],r['protocol']['selection'],ready)
    assert len(manifest['planes'])==12 and sum(v['anchor'] for v in manifest['planes'])==1
    assert [v['patient'] for v in manifest['patients']]==sorted(inner_split(t.train,t.val,20261018)['train'])[:4]
    x,y,*_=arrays[0];before=dp.frozen_state(fresh)
    a,stats,_,_=dp.evaluate(fresh,x,y,manifest['planes'][0]['native_shape'],r['protocol']['scaling'])
    b,_,_,_=dp.evaluate(g,x,y,manifest['planes'][0]['native_shape'],r['protocol']['scaling'])
    assert np.array_equal(a,b) and dp.frozen_state(fresh)==before
    assert all(torch.equal(v,restored['G'][k]) for k,v in fresh.net.state_dict().items())
    assert torch.equal(fresh.networks.upconv_arch,restored['arch']['upconv_arch'])
    n=fresh.networks
    assert n._arch_parameters[0] is n.upconv_arch and n._arch_parameters[1] is n.conv_arch
    dp.compare_anchor(stats,stats)
    with pytest.raises(ValueError):dp.compare_anchor(stats,dict(stats,valid=dict(stats['valid'],normalized_l1=10)))
    # Different ignore extents, identical zero conditions: compare raw before masking.
    x,y,*_=arrays[1]; yy=y.copy();yy[:,:,2:4]=-1
    a,st,_,_=dp.evaluate(fresh,x,y,manifest['planes'][1]['native_shape'],r['protocol']['scaling'])
    b,_,_,_=dp.evaluate(fresh,x,yy,manifest['planes'][1]['native_shape'],r['protocol']['scaling'])
    assert np.array_equal(a,b) and st['foreground'] is None
    records=[manifest['planes'][1],manifest['planes'][1]]
    group=dp.condition_groups(records,[(x,y),(x+1,yy)],[a,b])
    assert group['applicable'] and group['groups'][0]['comparisons'][0]['target_common_valid_l1']==1
    assert not dp.condition_groups(records[:1],[(x,y)],[a])['applicable']
    for out in (folder,folder/'child',folder.parent,t.protected[0]/'out'):
        with pytest.raises(ValueError):dp.run_preview_diagnostic(t,folder,sha256(path),out,gpu=0)
    args=__import__('experiments.genseg_joint.cli',fromlist=['parser']).parser().parse_args([
        'preview-diagnostic','--source','a','--raw-root','b','--targets','c','--baseline-run','d',
        '--output','e','--run','r','--checkpoint-sha256','a'*64,'--gpu','1'])
    assert not hasattr(args,'patient_limit') and not hasattr(args,'steps')


@pytest.mark.parametrize('fail_anchor',[False,True])
def test_diagnostic_orchestration(tmp_path,monkeypatch,fail_anchor):
    from experiments.genseg_joint import diagnostic as d
    torch.set_num_threads(1)
    dp,t,folder,r,ready,payload,g=diagnostic_files(tmp_path,monkeypatch)
    restored=dp.load_diagnostic(folder,dp.EXPECTED_SHA256,r,ready)
    preview.restore_generator(g,restored)
    cohort=preview.Cohort(t,20261018)
    manifest,arrays=dp.coverage(cohort,r['protocol']['scaling'],r['protocol']['selection'],ready)
    x,y,*_=arrays[0]
    _,stats,_,active=dp.evaluate(g,x,y,manifest['planes'][0]['native_shape'],r['protocol']['scaling'])
    if fail_anchor:stats['valid']['normalized_l1']+=1
    r['observations']={'after':{'statistics':stats,'probe_backend':{'active':active}}}
    (folder/'diagnostic.json').write_text(json.dumps(r))
    monkeypatch.setattr(dp,'gpu_guard',lambda *a,**k:{'allocator_fraction':.1,'gpu_uuid':'cpu-fixture'})
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:False)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',lambda *a:None)
    monkeypatch.setattr(dp,'Generator',lambda **kw:Generator(**{k:v for k,v in kw.items() if k!='device'},device='cpu'))
    monkeypatch.setattr(Generator,'discriminator',lambda *a,**k:pytest.fail('D constructed'))
    output=tmp_path/'preview'
    if fail_anchor:
        with pytest.raises(ValueError,match='Anchor scalar mismatch'):
            dp.run_preview_diagnostic(t,folder,dp.EXPECTED_SHA256,output,gpu=0)
        assert (output/'failed.json').exists() and not (output/'preview_diagnostic.json').exists()
    else:
        dp.run_preview_diagnostic(t,folder,dp.EXPECTED_SHA256,output,gpu=0)
        receipt=json.loads((output/'preview_diagnostic.json').read_text())
        assert receipt['complete'] and len(receipt['previews'])==12
        assert receipt['condition_groups']['applicable']
        assert not receipt['historical_pixelwise_reproduction']
        assert not (output/'complete.json').exists()
        hashes=json.loads((output/'files_sha256.json').read_text())
        assert all(sha256(output/k)==v for k,v in hashes.items())
    # Coverage with missing category is not supplemented from later patients.
    original=t.load_train
    def empty(dataset,patient):
        x,y,meta=original(dataset,patient);y[y==1]=0
        return x,y,meta
    t.load_train=empty
    selection=dict(r['protocol']['selection'],z=1,category='lesion')
    xx,yy,_=cohort.plane('train',selection['patient'],1,r['protocol']['scaling'])
    missing,_=dp.coverage(cohort,r['protocol']['scaling'],selection,{'arrays':d.ready_arrays(xx,yy)})
    assert len(missing['planes'])==8 and all('lesion' in row['missing_categories'] for row in missing['patients'])
    # An original anchor outside the four selected patients is appended once only.
    selection=dict(selection,patient=cohort.train[4],z=2)
    xx,yy,_=cohort.plane('train',selection['patient'],2,r['protocol']['scaling'])
    appended,_=dp.coverage(cohort,r['protocol']['scaling'],selection,{'arrays':d.ready_arrays(xx,yy)})
    assert len(appended['planes'])==9 and appended['planes'][-1]['anchor']
