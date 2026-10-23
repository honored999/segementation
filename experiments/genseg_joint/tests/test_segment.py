"""Tiny CPU contracts; no patient data and no CUDA."""
from copy import deepcopy
from pathlib import Path
import json
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments import genseg_segment as seg
from experiments.genseg30.contract import REFERENCE, REFERENCE_SHA256, read_json, sha256


def receipt_fixture(tmp_path):
    from experiments.genseg_joint.joint_checkpoint import POLICY, PRETRAIN_SHA
    from experiments.genseg_joint.models import source_identity
    from experiments.genseg_joint.runtime import implementation_identity, TARGET_SHA256
    from experiments.genseg_joint.data import inner_split
    fixed = read_json(REFERENCE)[0]
    split = inner_split(sorted(fixed['train']), sorted(fixed['val']), 20261018)
    protocol = dict(POLICY, steps=77600, epochs=80, total_valid_planes=970,
        real_sampling='inner61_valid_planes_without_replacement_including_negatives',
        epoch_progress=dict(epoch=80, planes=970, total=970), split=split,
        source=source_identity(), pretrained_sha256=PRETRAIN_SHA,
        target_identity=dict(reference_sha256=REFERENCE_SHA256),
        target_manifest_sha256=TARGET_SHA256,
        implementation_files_sha256=implementation_identity(),
        scaling=dict(method='inner61_valid_pixel_global_minmax_to_minus1_plus1_no_clip',
                     min=0., max=1., fit_ids=split['train']),
        model_config={'G': {'width':64, 'downs':8, 'seed':20261018},
                      'S':'pinned_author_UNet(1,1,bilinear=False)'})
    run = tmp_path/'run'; run.mkdir()
    (run/'joint.pt').write_bytes(b'not-deserialized-by-metadata-audit')
    r = dict(format=2, complete=True, classification=POLICY['classification'],
        checkpoint='joint.pt', checkpoint_sha256=sha256(run/'joint.pt'),
        steps=77600, protocol=protocol)
    (run/'complete.json').write_text(json.dumps(r), encoding='utf-8')
    return run, r


def test_audit_rejects_partial_wrong_hash_split_source_and_failure(tmp_path, monkeypatch):
    run, r = receipt_fixture(tmp_path)
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('metadata audit loaded tensors'))
    seg.audit_receipt(run, r['checkpoint_sha256'])
    with pytest.raises(ValueError, match='hash'):
        seg.audit_receipt(run, '0'*64)
    for key, value in [('complete', False), ('steps', 1)]:
        bad = deepcopy(r); bad[key] = value
        (run/'complete.json').write_text(json.dumps(bad))
        with pytest.raises(ValueError): seg.audit_receipt(run, r['checkpoint_sha256'])
    for key, value in [('epochs',3), ('implementation_files_sha256',{'fake.py':'0'*64}),
                       ('split',dict(train=[],validation=[],external=[]))]:
        bad = deepcopy(r); bad['protocol'][key] = value
        (run/'complete.json').write_text(json.dumps(bad))
        with pytest.raises(ValueError): seg.audit_receipt(run, r['checkpoint_sha256'])
    (run/'complete.json').write_text(json.dumps(r))
    (run/'failed.json').write_text('{}')
    with pytest.raises(ValueError, match='failed'): seg.audit_receipt(run, r['checkpoint_sha256'])


def test_prediction_path_refused_before_metadata_or_reads(tmp_path, monkeypatch):
    def forbidden(*a, **k): pytest.fail('Unsafe output reached metadata/model/patient read')
    monkeypatch.setattr(seg,'bind_metadata',forbidden)
    raw=tmp_path/'raw'; raw.mkdir()
    common=['--run',str(tmp_path/'run'),'--source',str(tmp_path/'source'/'2d'),
        '--raw-root',str(raw),'--targets',str(tmp_path/'targets'),'--baseline',str(tmp_path/'baseline'),
        '--mask-cache',str(tmp_path/'cache'),'--model-reserve-gib','4','--case-reserve-gib','1']
    with pytest.raises(ValueError,match='outside'):
        seg.main(['predict',*common,'--output',str(raw/'forbidden')])
    old=tmp_path/'old'; old.mkdir()
    with pytest.raises(ValueError,match='Fresh'):
        seg.main(['predict',*common,'--output',str(old)])
    assert list(raw.iterdir())==[]


def test_ignore_region_is_not_generally_image_only():
    from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
    x = np.ones((1,3,5,7), np.float32); x[0,0,0,0] = 0
    label = np.zeros_like(x, dtype=np.int8); label[0,0,0,0] = 1
    _, image_only, _ = crop_to_nonzero(x, None)
    _, supervised, _ = crop_to_nonzero(x, label)
    assert image_only[0,0,0,0] == -1 and supervised[0,0,0,0] == 1
    assert not np.array_equal(image_only >= 0, supervised >= 0)


def test_production_s_strict_restore_output_and_eval_immutability():
    from experiments.genseg_joint.models import author_unet
    torch.set_num_threads(1)
    original = author_unet().eval()
    restored = seg.restore_segmenter(original.state_dict())
    x = torch.linspace(-1,1,32*32).reshape(1,1,32,32)
    with torch.inference_mode(): torch.testing.assert_close(original(x), restored(x), rtol=0, atol=0)
    before = seg.state_digest(restored)
    # Small padding multiple here is engineering-only; production default remains 256.
    volume = x.numpy()[0].repeat(2,axis=0)
    actual = seg.infer_prepared_volume(restored, volume, pad_multiple=32)
    assert actual.shape == volume.shape and set(np.unique(actual)) <= {0,1}
    assert seg.state_digest(restored) == before
    assert all(not m.training for m in restored.modules())
    broken = dict(original.state_dict()); broken.pop(next(iter(broken)))
    with pytest.raises(ValueError, match='keys'): seg.restore_segmenter(broken)


def test_fixed_threshold_full_planes_and_no_gt_argument():
    model = torch.nn.Conv2d(1,1,1).eval()
    with torch.no_grad(): model.weight.fill_(1); model.bias.zero_()
    x = np.array([[[-1,0,1],[2,-2,0]], [[0,0,0],[0,0,0]]], np.float32)
    result = seg.infer_prepared_volume(model, x)
    np.testing.assert_array_equal(result, x >= 0)
    model.train()
    with pytest.raises(ValueError, match='eval'): seg.infer_prepared_volume(model, x)


def test_crop_pad_transpose_and_real_resampler():
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    plans = read_json(seg.REPO/'standalone_nnunet2d/reference/nnUNetPlans.json')
    plans['transpose_forward'] = [2,0,1]; plans['transpose_backward'] = [1,2,0]
    pm = PlansManager(plans); cm = pm.get_configuration('2d')
    spacing = cm.spacing
    props = dict(spacing=[spacing[0]*2,spacing[1]*2,3.],
        shape_before_cropping=[4,7,9], bbox_used_for_cropping=[[1,3],[1,6],[2,8]],
        shape_after_cropping_and_before_resampling=[2,5,6])
    pred = np.ones((2,10,12), np.uint8)
    restored = seg.restore_binary_geometry(pred, pm, cm, props)
    expected = np.zeros((4,7,9), np.uint8); expected[1:3,1:6,2:8] = 1
    np.testing.assert_array_equal(restored, expected.transpose(1,2,0))
    assert restored.shape == (7,9,4)
    props['bbox_used_for_cropping'][0] = [1,5]
    with pytest.raises(ValueError, match='bbox'): seg.restore_binary_geometry(pred, pm, cm, props)


def test_exact19_rejects_missing_extra_duplicate():
    ids = sorted(read_json(REFERENCE)[0]['val'])
    seg.require_outer19(ids)
    for actual in (ids[:-1], ids+['foreign'], ids[:-1]+[ids[0]]):
        with pytest.raises(ValueError, match='outer19'): seg.require_outer19(actual)


def test_original_metrics_fixture(tmp_path, monkeypatch):
    import SimpleITK as sitk
    import evaluate_segmentation_metrics as metrics
    pred = tmp_path/'pred'; gt = tmp_path/'gt'; out = tmp_path/'metrics'
    pred.mkdir(); gt.mkdir()
    a = np.zeros((3,4,5),np.uint8); a[1,1:3,1:3] = 1
    img = sitk.GetImageFromArray(a); img.SetSpacing((.7,.8,3.)); img.SetOrigin((2.,3.,4.))
    for folder in (pred,gt): sitk.WriteImage(img,str(folder/'synthetic.nii.gz'))
    monkeypatch.setattr(sys,'argv',['evaluate_segmentation_metrics.py','--pred-dir',str(pred),
        '--gt-dir',str(gt),'--output-dir',str(out),'--f2-mode','paper'])
    metrics.main()
    r = read_json(out/'summary_metrics.json')
    assert r['n_cases'] == 1 and r['f2_mode'] == 'paper'
    assert r['metrics']['dice']['mean'] == 1 and r['metrics']['hd95_mm']['mean'] == 0


def test_full_joint_schema_safe_load_and_malformed_moments(tmp_path, monkeypatch):
    run, r = receipt_fixture(tmp_path)
    from experiments.genseg_joint.joint_checkpoint import validate_payload
    def tiny(): return torch.nn.Sequential(torch.nn.BatchNorm2d(1),torch.nn.Conv2d(1,1,1))
    def adam(params, rate):
        return dict(state={i:dict(step=torch.tensor(77600.),exp_avg=torch.zeros_like(v),
            exp_avg_sq=torch.zeros_like(v)) for i,v in enumerate(params)},param_groups=[dict(
                params=list(range(len(params))),lr=rate,betas=(.5,.999),eps=1e-8,weight_decay=0.,
                amsgrad=False,maximize=False,capturable=False,differentiable=False)])
    initial = {}; values = {}; opts = {}
    for key, origin, count, rate in [('G',19400,1,2e-5),('D',58200,3,2e-5),('S',0,2,1e-5)]:
        state = tiny().state_dict()
        initial[key] = {k:origin for k in state if k.endswith('num_batches_tracked')}
        for name in initial[key]: state[name].fill_(origin+77600*count)
        names = [k for k in state if not k.endswith(('running_mean','running_var','num_batches_tracked'))]
        opts[key.lower()] = dict(algorithm='Adam',steps=77600,lr=rate,betas=(.5,.999),eps=1e-8,
            weight_decay=0.,parameter_names=names,state_dict=adam([state[k] for k in names],rate))
        values[key] = state
    arch = dict(conv_arch=torch.zeros(1),upconv_arch=torch.zeros(1))
    opts['alpha'] = dict(algorithm='staged_Adam',steps=77600,state_dict=adam([arch['upconv_arch']],1e-4))
    payload = dict(format=2, classification=r['classification'],steps=77600,**values,
        protocol=r['protocol'],model_config=r['protocol']['model_config'],source=r['protocol']['source'],
        arch=arch,optimizers=opts,initial_bn=initial,timing={},snapshot={},
        rng=dict(numpy={},positive=dict(steps=77600,positive_planes=1),python=[],torch_cpu=torch.zeros(1),torch_cuda=[]))
    validate_payload(payload,r)
    def save(p):
        torch.save(p,run/'joint.pt')
        r['checkpoint_sha256'] = sha256(run/'joint.pt')
        (run/'complete.json').write_text(json.dumps(r))
    save(payload)
    monkeypatch.setattr(seg,'author_unet',tiny)
    fixed = read_json(REFERENCE)[0]
    target = SimpleNamespace(train=sorted(fixed['train']),val=sorted(fixed['val']),
        identity=r['protocol']['target_identity'],manifest_sha256=r['protocol']['target_manifest_sha256'])
    target.verify_identity = lambda identity: identity == target.identity or pytest.fail('target mismatch')
    load = torch.load; calls = []
    def safe_load(*a,**kw):
        calls.append(kw)
        assert kw['weights_only'] is True and kw['map_location'] == 'cpu'
        return load(*a,**kw)
    monkeypatch.setattr(torch,'load',safe_load)
    restored, _ = seg.load_audited_segmenter(run,r['checkpoint_sha256'],target=target)
    assert calls and all(not m.training for m in restored.modules())
    torch.testing.assert_close(restored.state_dict()['1.weight'],payload['S']['1.weight'])
    bad = deepcopy(payload); bad['optimizers']['s']['state_dict']['state'][0]['exp_avg_sq'].fill_(-1)
    save(bad)
    with pytest.raises(ValueError,match='moments'): seg.load_audited_segmenter(run,r['checkpoint_sha256'],target=target)
    save(payload)
    def changed(*a,**kw):
        result = safe_load(*a,**kw)
        with (run/'joint.pt').open('ab') as stream: stream.write(b'changed-after-load')
        return result
    monkeypatch.setattr(torch,'load',changed)
    with pytest.raises(ValueError,match='changed'): seg.load_audited_segmenter(run,r['checkpoint_sha256'],target=target)

@pytest.mark.parametrize('masked', [False, True])
def test_image_only_actual_preprocessor_and_saved_scale(tmp_path, monkeypatch, masked):
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    from nnunetv2.preprocessing.normalization.default_normalization_schemes import ZScoreNormalization
    plans = read_json(seg.REPO/'standalone_nnunet2d/reference/nnUNetPlans.json')
    plans['configurations']['2d']['use_mask_for_norm'] = [masked]
    pm = PlansManager(plans); cm = pm.get_configuration('2d')
    raw = np.zeros((1,3,8,10), np.float32)
    raw[0,:,1:7,1:9] = np.arange(3*6*8).reshape(3,6,8)+1
    raw[0,0,1,1] = 0
    calls = []
    original = ZScoreNormalization.run
    def normalize(self, image, seg=None):
        calls.append((self.use_mask_for_norm, seg.copy()))
        return original(self, image, seg)
    monkeypatch.setattr(ZScoreNormalization, 'run', normalize)
    def forbidden(*a, **kw): pytest.fail('GT accessed')
    monkeypatch.setattr(pm, 'get_label_manager', forbidden)
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    monkeypatch.setattr(SimpleITKIO, 'read_seg', forbidden)
    props = dict(spacing=[3.,*cm.spacing])
    image, valid, properties = seg.preprocess_image(raw, props, pm, cm, {'labels': {'background':0,'lesion':1}})
    assert calls[0][0] is masked and set(np.unique(calls[0][1])) == {-1,0}
    assert 'class_locations' not in properties and set(np.unique(valid)) <= {-1,0}
    keep = np.ones(image.shape[1:],bool); keep[:,2,2] = False
    result = seg.prepare_s_input(image,valid,keep,fill=-2.,scale=dict(min=-1.,max=2.))
    expected = image[0].copy(); expected[~keep] = -2.
    expected = (2*(expected+1.)/3.-1.).astype(np.float32)
    expected[valid[0]<0] = 0
    np.testing.assert_array_equal(result,expected)
    assert (result[keep & (valid[0]>=0)] != 0).any()
    assert result[:,2,2].min() < -1  # saved scale, no clipping


def test_hd_bet_retention_contract():
    from experiments.genseg3mm.targets import expand_keep_mask_3mm
    brain = np.zeros((2,11,13),bool); brain[0,5,6] = True
    keep = expand_keep_mask_3mm(brain,spacing_xyz=(2.,1.,4.))
    assert keep[1].all() and keep[0,8,6] and not keep[0,9,6]
    assert keep[0,5,7] and not keep[0,5,8]
    with pytest.raises(ValueError,match='empty'): expand_keep_mask_3mm(brain*False,spacing_xyz=(1.,1.,1.))


def test_asymmetric_real_preprocess_restore_writer(tmp_path):
    import SimpleITK as sitk
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    plans = read_json(seg.REPO/'standalone_nnunet2d/reference/nnUNetPlans.json')
    plans['transpose_forward'] = [2,0,1]; plans['transpose_backward'] = [1,2,0]
    pm = PlansManager(plans); cm = pm.get_configuration('2d')
    raw = np.zeros((7,9,4),np.float32); raw[1:6,2:8,1:3] = 5
    image = sitk.GetImageFromArray(raw)
    image.SetSpacing((3.,cm.spacing[1]*2,cm.spacing[0]*2))
    image.SetOrigin((12.,-3.,9.)); image.SetDirection((0.,-1.,0.,1.,0.,0.,0.,0.,1.))
    path = tmp_path/'raw.nii.gz'; sitk.WriteImage(image,str(path))
    rw = pm.image_reader_writer_class(); data, props = rw.read_images([str(path)])
    x, valid, props = seg.preprocess_image(data,props,pm,cm,{'labels':{'background':0,'lesion':1}})
    assert x.shape == (1,2,10,12)  # real unequal-spacing resampling
    pred = np.zeros(x.shape[1:],np.uint8); pred[0,2:6,2:8] = 1; pred[1,6:8,8:10] = 1
    restored = seg.restore_binary_geometry(pred,pm,cm,props)
    expected = np.zeros((4,7,9),np.uint8); expected[1,2:4,3:6] = 1; expected[2,4,6] = 1
    np.testing.assert_array_equal(restored,expected.transpose(1,2,0))
    out = tmp_path/'pred.nii.gz'
    geometry = seg.export_prediction(pred,out,path,pm,cm,props)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(sitk.ReadImage(str(out))),restored)
    assert geometry['size_xyz'] == list(image.GetSize())
    with pytest.raises(ValueError,match='exists'): seg.export_prediction(pred,out,path,pm,cm,props)

def prediction_fixture(tmp_path, monkeypatch):
    import SimpleITK as sitk
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from experiments.genseg_joint import joint_resources
    ids=sorted(read_json(REFERENCE)[0]['val'])
    raw_root=tmp_path/'raw'; base=tmp_path/'source'/'Dataset501_StrokeLesion'; base.mkdir(parents=True)
    images=raw_root/base.name/'imagesTr'; images.mkdir(parents=True)
    plans=read_json(seg.REPO/'standalone_nnunet2d/reference/nnUNetPlans.json')
    pm=PlansManager(plans); cm=pm.get_configuration('2d')
    data=np.zeros((2,6,7),np.float32); data[:,1:5,1:6]=np.arange(40).reshape(2,4,5)+1
    img=sitk.GetImageFromArray(data); img.SetSpacing((*cm.spacing[::-1],3.)); img.SetOrigin((2.,3.,4.))
    masks=tmp_path/'masks'; masks.mkdir()
    hd={}
    for cid in ids:
        raw=images/(cid+'_0000.nii.gz'); sitk.WriteImage(img,str(raw))
        mask=masks/(cid+'.nii.gz'); bm=sitk.GetImageFromArray((data>0).astype(np.uint8)); bm.CopyInformation(img)
        sitk.WriteImage(bm,str(mask))
        hd[cid]=dict(mask_file=str(mask),mask_sha256=sha256(mask),raw_sha256=sha256(raw))
    phases=[]
    class Guard:
        def __init__(self,*a): pass
        def __call__(self,*a,phase='startup',**kw):
            phases.append(phase)
            return dict(allocator_fraction=.1,phase=phase)
        def bind(self): pass
    monkeypatch.setattr(joint_resources,'ResourceGuard',Guard)
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:False)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',lambda *a:None)
    model=torch.nn.Conv2d(1,1,1).eval()
    model.to=lambda *a,**kw:model  # transport only; actual inference/geometry/gates execute on CPU
    monkeypatch.setattr(seg,'load_audited_segmenter',lambda *a,**kw:(model,{}))
    def native(b,cid,raw,cache,guard):
        case=cache/cid; case.mkdir()
        mask=case/'hd_bet_bet.nii.gz'; mask.write_bytes(Path(hd[cid]['mask_file']).read_bytes())
        return np.ones(data.shape,bool),dict(hd[cid],mask_file=str(mask.resolve()))
    monkeypatch.setattr(seg,'native_keep',native)
    metadata_root=tmp_path/'bound_metadata'; metadata_root.mkdir()
    metadata=metadata_root/'metadata.json'; metadata.write_text('{}')
    hashes={str(metadata):sha256(metadata)}
    audit=dict(metadata_files_sha256=hashes,inference_sources_sha256=hashes,training_sources_sha256=seg.implementation_identity(),protected_roots=[str(raw_root),str(base)])
    source=SimpleNamespace(val=ids,raw_root=raw_root,base=base,protected=[seg.REPO,raw_root,base],
                           dataset_json={'labels':{'background':0,'lesion':1}})
    binding=SimpleNamespace(source=source,pm=pm,cm=cm,verify=lambda **kw:None,
        manifest={'baseline':{'run':str(tmp_path/'baseline')},'background_fill':-2.},
        targets=tmp_path/'targets',weights=metadata,audit=audit,target=object(),
        receipt={'checkpoint_sha256':'a'*64,'protocol':{'scaling':{'min':-2.,'max':3.}}})
    return binding,images,phases


def test_actual_prediction_completion_and_scoring_gate(tmp_path,monkeypatch):
    import SimpleITK as sitk
    binding,images,phases=prediction_fixture(tmp_path,monkeypatch)
    output=tmp_path/'pred'; cache=tmp_path/'cache'
    receipt=seg.predict(binding,tmp_path/'run',output,cache,model_reserve_bytes=1,case_reserve_bytes=1)
    assert len(receipt['cases'])==19 and (output/'complete.json').exists() and 'between_units' in phases
    assert receipt['cache_root']==str(cache.resolve())
    assert read_json(output/'started.json')['cache_root']==str(cache.resolve())
    ids=seg.verify_predictions(output,receipt,images)
    for bad_root in (None,str(tmp_path),str(output),str(cache/ids[0])):
        broken=deepcopy(receipt)
        if bad_root is None: broken.pop('cache_root')
        else: broken['cache_root']=bad_root
        with pytest.raises(ValueError): seg.verify_predictions(output,broken,images)
    for field in ('prepared_file','properties_file','mask_file'):
        broken=deepcopy(receipt); rec=broken['cases'][ids[0]]
        owner=rec['hd_bet'] if field=='mask_file' else rec
        external=tmp_path/('escaped_'+Path(owner[field]).name)
        external.write_bytes(Path(owner[field]).read_bytes()); owner[field]=str(external)
        (output/(ids[0]+'.json')).write_text(json.dumps(rec),encoding='utf-8')
        with pytest.raises(ValueError,match='cache'): seg.verify_predictions(output,broken,images)
        (output/'complete.json').write_text(json.dumps(broken),encoding='utf-8')
        with pytest.raises(ValueError): seg.accept_for_scoring(output,images,tmp_path/'no_GT',tmp_path/'metrics_escape')
        (output/(ids[0]+'.json')).write_text(json.dumps(receipt['cases'][ids[0]]),encoding='utf-8')
    (output/'complete.json').write_text(json.dumps(receipt),encoding='utf-8')
    gt=tmp_path/'gt'; gt.mkdir()
    for cid in ids:
        img=sitk.ReadImage(str(output/(cid+'.nii.gz'))); sitk.WriteImage(img,str(gt/(cid+'.nii.gz')))
    metrics=tmp_path/'metrics'
    assert seg.accept_for_scoring(output,images,gt,metrics)==metrics
    assert not metrics.exists()  # admission must not score or write into GT
    gt_hashes={p.name:sha256(p) for p in gt.iterdir()}
    for protected in (cache/'metrics_new',cache,cache.parent):
        with pytest.raises(ValueError): seg.accept_for_scoring(output,images,gt,protected)
    assert gt_hashes=={p.name:sha256(p) for p in gt.iterdir()}
    assert not (cache/'metrics_new').exists()
    for bad in ('missing','extra','tamper','partial','duplicate','mask','geometry'):
        broken=deepcopy(receipt)
        path=output/(ids[0]+'.nii.gz'); original=path.read_bytes()
        if bad=='missing': path.unlink()
        if bad=='extra': (output/'foreign.nii.gz').write_bytes(original)
        if bad=='tamper': path.write_bytes(original+b'changed')
        if bad=='partial': (output/'partial.json').write_text('{}')
        if bad=='duplicate': broken['external_ids'][-1]=ids[0]
        if bad=='mask': broken['cases'][ids[0]]['hd_bet']['mask_sha256']='b'*64
        if bad=='geometry': broken['cases'][ids[0]]['geometry']['origin_xyz'][0]+=2
        with pytest.raises(ValueError): seg.verify_predictions(output,broken,images)
        path.write_bytes(original)
        (output/'foreign.nii.gz').unlink(missing_ok=True); (output/'partial.json').unlink(missing_ok=True)
    # A prediction-side failure must precede even a GT path inspection/read.
    (output/'failed.json').write_text('{}')
    with pytest.raises(ValueError,match='Failed'):
        seg.accept_for_scoring(output,images,tmp_path/'nonexistent_GT',metrics)
    (output/'failed.json').unlink()
    for protected in (output/'new', images/'new',binding.targets/'new',tmp_path/'run'/'new'):
        # Prediction refuses raw, existing outputs, target, run roots.
        with pytest.raises(ValueError):
            seg.predict(binding,tmp_path/'run',protected,tmp_path/'othercache',model_reserve_bytes=1,case_reserve_bytes=1)
    with pytest.raises(ValueError,match='Fresh'):
        seg.predict(binding,tmp_path/'run',output,cache,model_reserve_bytes=1,case_reserve_bytes=1)


@pytest.mark.parametrize('fault',['interrupt','post_complete','failure_receipt'])
def test_prediction_failure_removes_complete_preserves_original(tmp_path,monkeypatch,fault):
    binding,images,_=prediction_fixture(tmp_path,monkeypatch)
    original_verify=seg.verify_predictions; original_write=seg.write_json
    output=tmp_path/'pred'
    def verify(*a,**kw):
        if (output/'complete.json').exists(): raise RuntimeError('post_complete fault')
        return original_verify(*a,**kw)
    def write(path,value):
        if Path(path).name in ('complete.json','failed.json'): raise OSError('receipt disk failure')
        return original_write(path,value)
    if fault=='post_complete': monkeypatch.setattr(seg,'verify_predictions',verify)
    if fault=='failure_receipt': monkeypatch.setattr(seg,'write_json',write)
    if fault=='interrupt':
        monkeypatch.setattr(seg,'infer_prepared_volume',lambda *a,**kw:(_ for _ in ()).throw(KeyboardInterrupt('interrupted')))
    error=KeyboardInterrupt if fault=='interrupt' else RuntimeError if fault=='post_complete' else OSError
    with pytest.raises(error):
        seg.predict(binding,tmp_path/'run',output,tmp_path/'cache',model_reserve_bytes=1,case_reserve_bytes=1)
    assert not (output/'complete.json').exists()
    if fault!='failure_receipt':
        failed=read_json(output/'failed.json'); assert failed['complete'] is False and failed['partial_cases']

@pytest.mark.parametrize('masked',[False,True])
def test_real_metadata_binding_before_any_patient_array(tmp_path,monkeypatch,masked):
    from experiments.genseg30 import contract,pipeline,models
    from experiments.genseg3mm import targets
    run,r=receipt_fixture(tmp_path)
    base=tmp_path/'preprocessed'/'Dataset501_StrokeLesion'; folder=base/'nnUNetPlans_2d'; folder.mkdir(parents=True)
    plans=read_json(seg.REPO/'standalone_nnunet2d/reference/nnUNetPlans.json')
    plans['configurations']['2d']['use_mask_for_norm']=[masked]
    (base/'nnUNetPlans.json').write_text(json.dumps(plans))
    (base/'dataset.json').write_text(json.dumps(dict(channel_names={'0':'DWI'},labels={'background':0,'lesion':1},file_ending='.nii.gz')))
    (base/'splits_final.json').write_bytes(REFERENCE.read_bytes())
    fixed=read_json(REFERENCE)[0]
    for cid in fixed['train']+fixed['val']:
        for suffix in ('.b2nd','_seg.b2nd','.pkl'): (folder/(cid+suffix)).write_bytes(b'never_open_patient_array')
    raw=tmp_path/'raw'
    monkeypatch.delenv('nnUNet_raw',raising=False)
    source=contract.Source(folder,raw)
    def forbidden(*a,**kw): pytest.fail('Patient/GT/model read during metadata audit')
    monkeypatch.setattr(contract.Source,'load_train',forbidden)
    monkeypatch.setattr(contract.Source,'training_dataset',forbidden)
    monkeypatch.setattr(torch,'load',forbidden)
    baseline=tmp_path/'baseline'; baseline.mkdir(); (baseline/'generator.pt').write_bytes(b'not-loaded')
    meta=dict(complete=True,classification=contract.CLASSIFICATION,format=1,source=source.identity,
        upstream_commit=models.UPSTREAM_COMMIT,upstream_files=models.FILES,
        cases={cid:{'shape':[1,2,6,7]} for cid in source.train},
        settings=dict(targets.EXPECTED_SETTINGS,norm='batch',dropout=False,gan_loss='vanilla',scheduler='constant_lr'),
        actual_steps=5000,scaling={'method':'global_train_minmax','min':-2.,'max':9.},inventory={'saved':'metadata-only'})
    record=dict(metadata=meta,checkpoint_sha256=sha256(baseline/'generator.pt'))
    (baseline/'generator.json').write_text(json.dumps(record))
    weights=tmp_path/'weights'; weights.write_bytes(b'tiny_weight_hash_only')
    monkeypatch.setattr(targets,'_hd_bet_runtime',lambda:('2.0.1','hd-bet',weights,sha256(weights)))
    hd_package=tmp_path/'hd_package'; hd_package.mkdir(); (hd_package/'__init__.py').write_text('# synthetic metadata source')
    monkeypatch.setitem(sys.modules,'HD_BET',SimpleNamespace(__file__=str(hd_package/'__init__.py')))
    actual_version=seg.importlib.metadata.version
    monkeypatch.setattr(seg.importlib.metadata,'version',lambda name:'2.0.1' if name=='hd-bet' else actual_version(name))
    root=tmp_path/'targets'; root.mkdir()
    manifest=dict(format=1,complete=True,classification=targets.CLASSIFICATION,recipe=targets.RECIPE,
        source=source.identity,cases={cid:{} for cid in source.train},
        baseline=dict(metadata=meta,checkpoint_sha256=record['checkpoint_sha256'],run=str(baseline)),
        source_inventory=meta['inventory'],background_fill=-2.,
        hd_bet=dict(version='2.0.1',default_tta=True,weights_sha256=sha256(weights)))
    (root/'targets.json').write_text(json.dumps(manifest)); digest=sha256(root/'targets.json')
    monkeypatch.setattr(seg,'TARGET_SHA256',digest)
    identity=dict(source.identity,genseg3mm=dict(classification=targets.CLASSIFICATION,target_manifest_sha256=digest,
        baseline_checkpoint_sha256=record['checkpoint_sha256'],recipe=targets.RECIPE))
    r['protocol']['target_identity']=identity; r['protocol']['target_manifest_sha256']=digest
    (run/'complete.json').write_text(json.dumps(r))
    b=seg.bind_metadata(run,folder,raw,root,baseline)
    assert b.audit['use_mask_for_norm']==[masked] and b.audit['fill']==-2.
    assert b.audit['actual_plans_status']=='VERIFIED_ON_EXECUTING_HOST'
    assert b.target.train==sorted(fixed['train']) and b.target.val==sorted(fixed['val'])
    assert not hasattr(b.target,'load_train') and not hasattr(b.target,'training_dataset')
    assert str(run/'joint.pt') in b.audit['metadata_files_sha256']
    b.verify()
    plans['configurations']['2d']['use_mask_for_norm']=[not masked]
    (base/'nnUNetPlans.json').write_text(json.dumps(plans))
    with pytest.raises(ValueError): seg.bind_metadata(run,folder,raw,root,baseline)

@pytest.mark.parametrize('fault',['healthy','resource','terminate','wait','kill','reap','predict_receipt'])
def test_native_hd_command_receipt_and_resource_abort(tmp_path,monkeypatch,fault):
    import SimpleITK as sitk
    from experiments.genseg3mm.targets import RECIPE
    cache=tmp_path/'cache'
    if fault!='predict_receipt': cache.mkdir()
    raw=tmp_path/'raw.nii.gz'; image=sitk.GetImageFromArray(np.ones((2,5,7),np.float32))
    image.SetSpacing((1.,1.5,3.)); sitk.WriteImage(image,str(raw))
    weights=tmp_path/'weights'; weights.write_bytes(b'fixed')
    binding=SimpleNamespace(executable='hd-bet',weights=weights,weight_hash=sha256(weights),manifest={'recipe':RECIPE})
    original=RuntimeError('resource fault'); original.resource_snapshot={'gpu_util_pct':99}
    origin_tracebacks=[]
    processes=[]
    class Process:
        def __init__(self,command,**kwargs):
            self.command=command; self.waits=0; self.terminated=False; processes.append(self)
            assert '--save_bet_mask' in command and '--no_bet_image' in command and '--disable_tta' not in command
            assert command[command.index('-device')+1]=='cuda'
            inp=Path(command[command.index('-i')+1]); out=Path(command[command.index('-o')+1])
            las=sitk.ReadImage(str(inp)); mask=sitk.GetImageFromArray(np.ones(tuple(reversed(las.GetSize())),np.uint8))
            mask.CopyInformation(las); sitk.WriteImage(mask,str(out.with_name(out.name[:-7]+'_bet.nii.gz')))
        def wait(self,timeout=None):
            self.waits+=1
            if self.waits==1: raise seg.subprocess.TimeoutExpired('hd-bet',timeout)
            if self.waits==2:
                if fault=='wait': raise OSError('cleanup wait fault')
                if fault in ('kill','reap','predict_receipt'): raise seg.subprocess.TimeoutExpired('hd-bet',timeout)
            if self.waits==3 and fault=='reap': raise OSError('cleanup reap fault')
            return 0
        def terminate(self):
            self.terminated=True
            if fault=='terminate': raise OSError('cleanup terminate fault')
        def kill(self):
            self.killed=True
            if fault in ('kill','predict_receipt'): raise OSError('cleanup kill fault')
    monkeypatch.setattr(seg.subprocess,'Popen',Process)
    def guard(*a,phase='startup',**kw):
        if fault!='healthy' and phase=='between_units':
            try: raise original
            except RuntimeError:
                origin_tracebacks.append(original.__traceback__)
                raise
        return {'phase':phase,'gpu_util_pct':99 if phase=='between_units' else 0}
    if fault!='healthy':
        if fault=='predict_receipt':
            native=seg.native_keep
            binding,images,_=prediction_fixture(tmp_path,monkeypatch)
            binding.executable='hd-bet'; binding.weights=weights; binding.weight_hash=sha256(weights)
            binding.manifest['recipe']=RECIPE
            monkeypatch.setattr(seg,'native_keep',lambda b,c,r,k,g:native(b,c,r,k,guard))
            invoke=lambda:seg.predict(binding,tmp_path/'run',tmp_path/'pred',cache,model_reserve_bytes=1,case_reserve_bytes=1)
        else:
            invoke=lambda:seg.native_keep(binding,'synthetic',raw,cache,guard)
        with pytest.raises(RuntimeError,match='resource fault') as caught: invoke()
        assert caught.value is original and original.args==('resource fault',)
        assert original.resource_snapshot=={'gpu_util_pct':99}
        trace=original.__traceback__; chain=[]
        while trace is not None: chain.append(trace); trace=trace.tb_next
        assert origin_tracebacks[0] in chain
        assert processes[0].terminated and not list(cache.glob('*/complete.json'))
        if fault in ('terminate','wait','kill','reap','predict_receipt'):
            assert original.cleanup_errors and any(e['error_type']=='OSError' for e in original.cleanup_errors)
        if fault=='predict_receipt':
            failed=read_json(tmp_path/'pred'/'failed.json')
            assert failed['reason']=='resource fault' and failed['error_type']=='RuntimeError'
            assert failed['resource_snapshot']==original.resource_snapshot
            assert failed['cleanup_errors']==original.cleanup_errors
            assert not (tmp_path/'pred'/'complete.json').exists()
    else:
        keep,rec=seg.native_keep(binding,'synthetic',raw,cache,guard)
        assert keep.all() and keep.shape==(2,5,7)
        assert rec['raw_sha256']==sha256(raw) and rec['weights_sha256']==sha256(weights)
        assert read_json(cache/'synthetic'/'complete.json')==rec
        assert 'between_units' in (cache/'synthetic'/'resources.jsonl').read_text()
