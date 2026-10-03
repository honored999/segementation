import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import pytest
import report_comparison_selection as selection


def make_cohort(tmp_path):
    import SimpleITK as sitk
    root=tmp_path/'results'/selection.DATASET/selection.BASELINE
    pred=root/'fold_0'/'validation'; metrics=root/'fold_0'/'multi_metric_evaluation'
    images=tmp_path/'images'; labels=tmp_path/'labels'
    for p in (pred,metrics,images,labels): p.mkdir(parents=True)
    (root/'plans.json').write_text(json.dumps({'image_reader_writer':'SimpleITKIO','configurations':{'2d':{'architecture':{'network_class_name':'dynamic_network_architectures.architectures.unet.PlainConvUNet'}}}}),encoding='utf-8')
    (root/'dataset.json').write_text(json.dumps({'name':selection.DATASET,'channel_names':{'0':'DWI'},'labels':{'background':0,'lesion':1}}),encoding='utf-8')
    (root/'fold_0'/'checkpoint_best.pth').write_bytes(b'loader sentinel only')
    ids=[f'case{i}' for i in range(7)]
    rows=[]
    for i,cid in enumerate(ids):
        mask=np.zeros((5,8,8),np.uint8)
        if i<6:
            mask[1].flat[:i+1]=1; mask[3].flat[:2*(i+1)]=1; mask[4].flat[:i+1]=1
        for folder,name,a in ((images,cid+'_0000.nii.gz',np.arange(320,dtype=np.float32).reshape(5,8,8)),(labels,cid+'.nii.gz',mask),(pred,cid+'.nii.gz',mask)):
            image=sitk.GetImageFromArray(a); image.SetSpacing((2,3,4));sitk.WriteImage(image,str(folder/name))
        rows.append(cid+','+str(.9 if i%2==0 else .1))
    (metrics/'case_metrics.csv').write_text('case_id,dice\n'+'\n'.join(rows)+'\n',encoding='utf-8')
    splits=tmp_path/'splits_final.json'; splits.write_text(json.dumps([{'train':['train1'],'val':ids} for _ in range(5)]),encoding='utf-8')
    args=SimpleNamespace(baseline_model_dir=root,fold=0,splits_file=splits,images_dir=images,labels_dir=labels,prediction_dir=pred,baseline_metrics_file=metrics/'case_metrics.csv',output_json=tmp_path/'derived'/'selection.json',thresholds_ml=None)
    return args


@pytest.fixture
def cohort(tmp_path):
    args=make_cohort(tmp_path)
    doc=selection.generate(args)
    from generate_nnunet_result_report import collect
    info=dict(rows={cid:{'dice':.5} for cid in doc['population']},predictions=collect(args.prediction_dir),images=collect(args.images_dir,input_channel=True),labels=collect(args.labels_dir))
    selection.load_selection(args.output_json,info)
    return args,doc,info


def test_volume_groups_slices_empty_and_exclusive(cohort):
    args,doc,info=cohort
    assert len(doc['population'])==7 and len(doc['entries'])==6
    assert [(e['size_group'],e['baseline_role']) for e in doc['entries']]==selection.ORDER
    assert doc['excluded'][0]['case_id']=='case6'
    assert doc['entries'][0]['gt_volume_ml']==pytest.approx(5*4*24/1000)
    assert all(e['display_slices']==[1,3,4] and e['representative_slice']==3 for e in doc['entries'])
    assert doc['grouping']['thresholds_ml']==pytest.approx([.256,.416])
    with pytest.raises(ValueError,match='already exists'):selection.generate(args)
    selection.verify_selection(info)
    assert info['selection_verified']


@pytest.mark.parametrize('volumes,error', [([1]*6,'repeated'),([1,1,1,2,3,4],'two distinct'),([0]*6,'no positive'),([float('nan')]*6,'invalid')])
def test_group_degeneracy(volumes,error):
    with pytest.raises(ValueError,match=error): selection.choose([dict(case_id=str(i),gt_volume_ml=v,baseline_dice=.5) for i,v in enumerate(volumes)])


def test_thresholds_ties_and_short_slices():
    cases=[dict(case_id=str(i),gt_volume_ml=v,baseline_dice=.5) for i,v in enumerate([1,1,2,2,3,3])]
    entries,q=selection.choose(cases,[1,2])
    assert [e['case_id'] for e in entries]==['4','5','2','3','0','1']
    mask=np.zeros((4,2,2));mask[2]=1
    assert selection.slice_selection(mask)==([2],2)
    assert selection.slice_selection(np.zeros_like(mask))==([],None)
    with pytest.raises(ValueError):selection.choose(cases,[2,1])
    cases[0]['baseline_dice']=float('nan')
    with pytest.raises(ValueError):selection.choose(cases,[1,2])


@pytest.mark.parametrize('mutation', ['dice','duplicate','index','rep','type','geometry','locator'])
def test_reject_rehashed_invalid_schema(cohort,mutation):
    _,doc,_=cohort;doc=copy.deepcopy(doc);e=doc['entries'][0]
    if mutation=='dice':e['baseline_dice']=2
    if mutation=='duplicate':doc['entries'][1]['case_id']=e['case_id']
    if mutation=='index':e['display_slices']=[1,1,9]
    if mutation=='rep':e['representative_slice']=0
    if mutation=='type':e['display_slices']=[True,3,4]
    if mutation=='geometry':e['gt']['geometry']['Spacing'][0]=-1
    if mutation=='locator':e['gt']['locator']='../escape.nii.gz'
    doc['selection_id']=selection.selection_id(doc)
    with pytest.raises(ValueError):selection.validate(doc)


def test_hash_tamper_nonfinite_duplicate_keys(cohort,tmp_path):
    _,doc,_=cohort;other=copy.deepcopy(doc);other['locations']['images_dir']='moved'
    assert selection.selection_id(other)==doc['selection_id']
    other['entries'][0]['representative_slice']=1
    with pytest.raises(ValueError,match='selection_id'):selection.validate(other)
    p=tmp_path/'invalid.json'
    for text in ('{"a":1,"a":2}','{"a":NaN}'):
        p.write_text(text,encoding='utf-8')
        with pytest.raises(ValueError):selection.read_json(p)


def test_relocation_then_changed_content_and_geometry(cohort,tmp_path):
    args,doc,info=cohort
    moved=tmp_path/'relocated';shutil.copytree(args.images_dir,moved)
    info['images']={cid:moved/p.name for cid,p in info['images'].items()}
    selection.load_selection(args.output_json,info);selection.verify_selection(info)
    e=doc['entries'][0];p=info['images'][e['case_id']]
    p.write_bytes(p.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='fingerprint'):selection.verify_selection(info)
    # A re-signed altered geometry must still be rejected independently of content hash.
    info['images'][e['case_id']]=args.images_dir/p.name
    doc['entries'][0]['image']['geometry']['Origin'][0]=3
    doc['entries'][0]['gt']['geometry']['Origin'][0]=3
    doc['selection_id']=selection.selection_id(doc)
    info['selection']=doc
    with pytest.raises(ValueError,match='geometry'):selection.verify_selection(info)


def test_model_score_reversal_never_reselects_missing_population(cohort):
    args,doc,info=cohort
    def forbidden(*a):raise AssertionError('no reranking in frozen mode')
    first=selection.report_groups(info,forbidden)
    for row in info['rows'].values():row['dice']=1-row['dice']
    assert selection.report_groups(info,forbidden)==first
    for e in doc['entries']:
        assert selection.selected_slices(info,e['case_id'],None,forbidden)==([1,3,4],3)
    info['predictions'].pop('case6')
    with pytest.raises(ValueError,match='population'):selection.load_selection(args.output_json,info)


def test_copy_and_id_publication_contract(cohort,tmp_path):
    _,doc,info=cohort;out=tmp_path/'report';out.mkdir()
    with pytest.raises(ValueError,match='PENDING'):selection.save_selection(info,out)
    selection.verify_selection(info)
    text=selection.save_selection(info,out);(out/'report.txt').write_text(text,encoding='utf-8')
    selection.check_selection_output(info,out)
    (out/'report.txt').write_text('incomplete',encoding='utf-8')
    with pytest.raises(ValueError,match='incomplete'):selection.check_selection_output(info,out)
    (out/'comparison_selection.json').unlink()
    with pytest.raises(FileNotFoundError):selection.check_selection_output(info,out)


def test_official_metadata_check_no_voxels_or_models(cohort,monkeypatch,capsys):
    import generate_nnunet_result_report as report
    import SimpleITK as sitk
    args,doc,info=cohort
    fields=report.METRICS
    rows={cid:{k:(.5 if k in ('dice','iou','f2','recall') else 0) for k in fields} for cid in doc['population']}
    summary={'n_cases':7,'metrics':{k:{'valid_cases':7} for k in fields}}
    args.baseline_metrics_file.write_text('case_id,'+','.join(fields)+'\n'+'\n'.join(cid+','+','.join(str(r[k]) for k in fields) for cid,r in rows.items())+'\n',encoding='utf-8')
    (args.baseline_metrics_file.parent/'summary_metrics.json').write_text(json.dumps(summary),encoding='utf-8')
    # Full raw roots may include cases outside validation.
    shutil.copyfile(next(args.images_dir.glob('*.nii.gz')),args.images_dir/'extra_0000.nii.gz')
    shutil.copyfile(next(args.labels_dir.glob('*.nii.gz')),args.labels_dir/'extra.nii.gz')
    def forbidden(*a,**k):raise AssertionError('metadata only must not load')
    monkeypatch.setattr(sitk,'ReadImage',forbidden);monkeypatch.setattr(report,'_predictor',forbidden)
    argv=['--model-dir',str(args.baseline_model_dir),'--fold','0','--images-dir',str(args.images_dir),'--labels-dir',str(args.labels_dir),'--checkpoint',str(args.baseline_model_dir/'fold_0'/'checkpoint_best.pth'),'--selection-json',str(args.output_json),'--output-dir',str(args.output_json.parent/'report'),'--check']
    assert report.main(argv)==0
    text=capsys.readouterr().out
    assert doc['selection_id'] in text and 'PENDING' in text and not (args.output_json.parent/'report').exists()


def test_standalone_representative_and_same_forward():
    import torch
    import standalone_h2former_report as report
    from test_standalone_h2former_report import _tiny_model
    model=_tiny_model(True)
    maps,native,middle=report._slice_features(model,np.ones((5,32,32),np.float32),[1,3,4],torch.device('cpu'),representative_slice=3)
    assert sorted(maps)==[1,3,4] and native['original_slice']==middle['original_slice']==3
    assert native['encoder_stages']['original_slice']==3
    assert not model.decoder._forward_pre_hooks


def test_save_figure_only_main_and_corrupt_output(tmp_path):
    import matplotlib.pyplot as plt
    from report_visuals import save_figure,validate_visual_outputs
    fig,ax=plt.subplots(figsize=(2,2));ax.plot([0,1],[0,1])
    save_figure(fig,tmp_path/'summary.png',family='summary');plt.close(fig)
    assert [p.name for p in tmp_path.iterdir()]==['summary.png']
    validate_visual_outputs(tmp_path,['summary'])
    with pytest.raises(ValueError,match='incomplete'):validate_visual_outputs(tmp_path,['architecture_overview'])
    (tmp_path/'ppt').mkdir()
    with pytest.raises(ValueError,match='PPT'):validate_visual_outputs(tmp_path,['summary'])
@pytest.mark.parametrize('trainer',['nnUNetTrainer','nnUNetTrainerTopK10','nnUNetTrainerUPerNetTopK10EarlyStopping','h2former','h2former_lite_upernet','h2former_lite_upernet_w128_ppm1236'])
def test_all_entries_same_selection_full_metrics_and_no_ppt(cohort,tmp_path,monkeypatch,trainer):
    import generate_nnunet_result_report as official
    import standalone_h2former_report as standalone
    import standalone_nnunet2d.predict as predict
    from test_report_metrics_table import wire_no_forward
    from test_standalone_h2former_report import _metadata
    args,doc,info=cohort
    fields=official.METRICS
    rows={cid:{k:(1 if k in ('dice','iou','f2','recall') else 0) for k in fields} for cid in doc['population']}
    args.baseline_metrics_file.write_text('case_id,'+','.join(fields)+'\n'+'\n'.join(cid+','+','.join(str(r[k]) for k in fields) for cid,r in rows.items())+'\n',encoding='utf-8')
    (args.baseline_metrics_file.parent/'summary_metrics.json').write_text(json.dumps({'n_cases':7,'aggregation':'macro average over cases','metrics':{k:{'valid_cases':7} for k in fields}}),encoding='utf-8')
    model=tmp_path/'models'/selection.DATASET/(trainer+'__nnUNetPlans__2d');model.mkdir(parents=True)
    for name in ('plans.json','dataset.json'):shutil.copyfile(args.baseline_model_dir/name,model/name)
    cp=model/'fold_0'/'checkpoint_best.pth';cp.parent.mkdir();cp.write_bytes(b'never loaded')
    argv=['--model-dir',str(model),'--fold','0','--images-dir',str(args.images_dir),'--labels-dir',str(args.labels_dir),'--prediction-dir',str(args.prediction_dir),'--metrics-dir',str(args.baseline_metrics_file.parent),'--checkpoint',str(cp),'--selection-json',str(args.output_json),'--output-dir',str(tmp_path/'report'),'--device','cpu']
    metadata=None
    if trainer.startswith('h2former'):
        metadata=_metadata(trainer);metadata.update(run_type='official_alignment_pending',run_state='official_alignment_pending')
        config=model/'config.json';config.write_text(json.dumps(metadata['resolved_config']),encoding='utf-8')
        manifest=model/'manifest.json';manifest.write_text(json.dumps(dict(schema_version=1,checkpoint=metadata,policy=dict(output_space='source',run_state='official_alignment_pending',alignment_status='official_alignment_pending'),cases=[dict(case_id=cid,source_path=str(info['images'][cid]),prediction_path=str(info['predictions'][cid])) for cid in doc['population']])),encoding='utf-8')
        argv+=['--source','standalone-h2former','--config',str(config),'--manifest',str(manifest),'--allow-pending']
    elif trainer==official.TRAINER:argv+=['--prediction-checkpoint-declaration','synthetic source declaration; never real provenance']
    wire_no_forward(monkeypatch,metadata)
    # Sentinel verifies manifest identity before any model loader in every entry.
    if metadata is None:
        def loader(data,*a):
            assert data['selection_verified'] and data['selection']['selection_id']==doc['selection_id']
            return object()
        monkeypatch.setattr(official,'_predictor',loader)
    else:
        monkeypatch.setattr(standalone,'_slice_features',lambda m,a,s,d,representative_slice=None:({z:np.zeros((8,8)) for z in s},None,None))
    assert official.main(argv+['--check'])==0
    assert not (tmp_path/'report').exists()
    assert official.main(argv)==0
    output=tmp_path/'report'
    assert not (output/'ppt').exists()
    copy=selection.read_json(output/'comparison_selection.json')
    assert copy==doc
    assert doc['selection_id'] in (output/'report.txt').read_text(encoding='utf-8')
    import csv
    records=list(csv.DictReader((output/'metrics_table.csv').open(encoding='utf-8-sig')))
    assert {r['case_id'] for r in records if r['row_type']=='case'}==set(doc['population'])
    assert 'case6' in {r['case_id'] for r in records}


@pytest.mark.parametrize('bad', ['wrong_class','subclass'])
def test_standard_official_trainer_not_namesake(bad):
    from generate_nnunet_result_report import resolve_external_trainer,STANDARD_TRAINER
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    if bad=='subclass':
        class Impostor(nnUNetTrainer):pass
    else:Impostor=type('nnUNetTrainer',(),{})
    with pytest.raises(ValueError,match='identity'):resolve_external_trainer(lambda name:Impostor,STANDARD_TRAINER)
    assert resolve_external_trainer(lambda name:nnUNetTrainer,STANDARD_TRAINER) is nnUNetTrainer


def test_nifti_unknown_units_rejected(tmp_path):
    import SimpleITK as sitk
    import struct
    p=tmp_path/'unknown.nii';sitk.WriteImage(sitk.GetImageFromArray(np.zeros((2,2,2),np.uint8)),str(p))
    content=bytearray(p.read_bytes());assert struct.unpack('<i',content[:4])[0]==348
    content[123]=0;p.write_bytes(content)
    with pytest.raises(ValueError,match='units'):selection._volume(p)


@pytest.mark.parametrize('fault',['nan','conflicting_duplicate','missing_case','nonbinary','thresholds','split'])
def test_generator_rejects_invalid_sources(tmp_path,fault):
    import SimpleITK as sitk
    args=make_cohort(tmp_path)
    if fault=='nan':args.baseline_metrics_file.write_text(args.baseline_metrics_file.read_text().replace('case0,0.9','case0,NaN'),encoding='utf-8')
    if fault=='conflicting_duplicate':
        with args.baseline_metrics_file.open('a',encoding='utf-8') as f:f.write('case0,0.8\n')
    if fault=='missing_case':(args.prediction_dir/'case0.nii.gz').unlink()
    if fault=='nonbinary':
        p=args.labels_dir/'case0.nii.gz';image=sitk.GetImageFromArray(np.full((5,8,8),2,np.uint8));image.SetSpacing((2,3,4));sitk.WriteImage(image,str(p))
    if fault=='thresholds':args.thresholds_ml=[.001,.002]
    if fault=='split':
        split=json.loads(args.splits_file.read_text());split[0]['train']=['case0'];args.splits_file.write_text(json.dumps(split),encoding='utf-8')
    with pytest.raises(ValueError):selection.generate(args)
    assert not args.output_json.exists()
def test_official_report_consumes_frozen_order_and_representative(cohort,tmp_path,monkeypatch):
    import generate_nnunet_result_report as report
    import report_visuals
    args,doc,info=cohort
    selection.verify_selection(info)
    info.update(model=args.baseline_model_dir,checkpoint=args.baseline_model_dir/'fold_0'/'checkpoint_best.pth',plans=selection.read_json(args.baseline_model_dir/'plans.json'),dataset=selection.read_json(args.baseline_model_dir/'dataset.json'),metrics=args.baseline_metrics_file.parent,summary={},geometry_count=7,output=tmp_path/'render')
    info['output'].mkdir()
    seen=[]
    def no_selection(*a):raise AssertionError('current model must never reselect')
    monkeypatch.setattr(report,'select_cases',no_selection);monkeypatch.setattr(report,'select_slices',no_selection)
    monkeypatch.setattr(report,'_network_kind',lambda *a:'plain_unet')
    monkeypatch.setattr(report,'_original_feature_stages',lambda *a:(3,2))
    def diagnostic(path,predictor,stage,representative,middle):
        seen.append((path.name,representative));return np.zeros((5,8,8)),{},None,None
    monkeypatch.setattr(report,'_diagnostic_feature',diagnostic)
    for name in ('_plain_unet_overview_figure','_plain_unet_detail_figure','_native_channel_figure'):
        monkeypatch.setattr(report,name,lambda *a,**k:None)
    monkeypatch.setattr(report_visuals,'stage_figure',lambda *a:None)
    captured=[]
    monkeypatch.setattr(report,'_summary_figure',lambda items,*a:captured.extend(items))
    predictor=SimpleNamespace(network=SimpleNamespace(),configuration_manager=SimpleNamespace(patch_size=(128,128)),device='cpu')
    report.create_report(info,predictor,None)
    assert [i['cid'] for i in captured]==[e['case_id'] for e in doc['entries']]
    assert all(i['slices']==[1,3,4] for i in captured)
    assert [n for _,n in seen]==[3]*6
    assert all('baseline-' in i['group'] and 'baseline Dice' in i['group'] for i in captured)


@pytest.mark.parametrize('bad',['ResEncUNet','missing'])
def test_standard_plans_reject_before_checkpoint_read(cohort,bad):
    from generate_nnunet_result_report import _check_identity
    args,_,_=cohort
    plans=selection.read_json(args.baseline_model_dir/'plans.json')
    arch=plans['configurations']['2d']['architecture']
    if bad=='missing':arch.clear()
    else:arch['network_class_name']=bad
    (args.baseline_model_dir/'plans.json').write_text(json.dumps(plans),encoding='utf-8')
    with pytest.raises(ValueError,match='PlainConvUNet'):_check_identity(args.baseline_model_dir,args.baseline_model_dir/'fold_0'/'checkpoint_best.pth',0)
@pytest.mark.parametrize('kind',['plain_overview','plain_detail','upernet'])
def test_official_architecture_enlarged_text_containment(tmp_path,monkeypatch,kind):
    import report_visuals
    import generate_nnunet_result_report as report
    from matplotlib.patches import FancyBboxPatch
    channels=(32,64,128,256,512,512,512,512)
    encoder=SimpleNamespace(stages=list(range(8)),output_channels=channels,strides=((1,1),)+((2,2),)*7)
    decoder=SimpleNamespace(stages=list(range(7)),transpconvs=[SimpleNamespace(kernel_size=(2,2),stride=(2,2)) for _ in range(7)],seg_layers=[SimpleNamespace(out_channels=2) for _ in range(7)],pool_scales=(1,2,3,6),fpn_channels=128,classifier=SimpleNamespace(out_channels=2))
    network=SimpleNamespace(encoder=encoder,decoder=decoder,selected_feature_indices=(2,3,4,7))
    seen=[]
    def check(fig,output,**kw):
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        failures=[]
        from itertools import combinations
        for first,second in combinations(fig.texts,2):
            a,b=first.get_window_extent(renderer),second.get_window_extent(renderer)
            assert min(a.x1,b.x1)-max(a.x0,b.x0)<=0 or min(a.y1,b.y1)-max(a.y0,b.y0)<=0,(first.get_text(),second.get_text(),a.bounds,b.bounds)
        for ax in fig.axes:
            for patch in (p for p in ax.patches if isinstance(p,FancyBboxPatch)):
                center=(patch.get_x()+patch.get_width()/2,patch.get_y()+patch.get_height()/2)
                labels=[t for t in ax.texts if np.allclose(t.get_position(),center,rtol=0,atol=1e-8)]
                if patch.get_width()>19:
                    assert not labels  # decorative panel background, not a labeled node
                    continue
                assert len(labels)==1
                label=labels[0];b=patch.get_window_extent(renderer);t=label.get_window_extent(renderer)
                if t.x0<b.x0 or t.x1>b.x1 or t.y0<b.y0 or t.y1>b.y1:failures.append((label.get_text(),b.bounds,t.bounds))
            for text in ax.texts:
                b=text.get_window_extent(renderer)
                assert b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.x1+1 and b.y1<=fig.bbox.y1+1,(kind,text.get_text(),b.bounds)
        assert not failures,failures
        seen.append(True)
    monkeypatch.setattr(report_visuals,'save_figure',check)
    if kind=='plain_overview':report._plain_unet_overview_figure(network,(512,512),7,3,tmp_path/'overview.png')
    elif kind=='plain_detail':report._plain_unet_detail_figure(network,(512,512),7,3,tmp_path/'detail.png')
    else:report._upernet_overview_figure(network,7,3,tmp_path/'upernet.png')
    assert seen


def test_native_short_labels_actual_hw_and_no_overlap(tmp_path,monkeypatch):
    import report_visuals
    from generate_nnunet_result_report import _native_channel_figure
    from itertools import combinations
    native=dict(channels=np.arange(8*64*64,dtype=np.float32).reshape(8,64,64),channel_ids=[0,73,146,219,292,365,438,511],feature_shape=[1,512,64,64],stage=3,module='encoder.stages.3',original_slice=3,window_index=0)
    def capture(fig,path,**kwargs):
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        titles=[ax.title for ax in fig.axes if ax.images]
        assert len(titles)==8
        for title in titles:
            assert title.get_text().splitlines()==[f'Stage 3 | Ch {native["channel_ids"][titles.index(title)]}','64 x 64']
            box=title.get_window_extent(renderer)
            assert box.x0>=0 and box.x1<=fig.bbox.x1 and box.y1<=fig.bbox.y1
        for a,b in combinations(titles,2):
            aa,bb=a.get_window_extent(renderer),b.get_window_extent(renderer)
            assert min(aa.x1,bb.x1)-max(aa.x0,bb.x0)<=0 or min(aa.y1,bb.y1)-max(aa.y0,bb.y0)<=0
        text='\n'.join(t.get_text() for ax in fig.axes for t in ax.texts)
        assert 'encoder.stages.3' in text and '512 x 64 x 64' in text
    monkeypatch.setattr(report_visuals,'save_figure',capture)
    _native_channel_figure([dict(cid='synthetic',group='large | baseline-good | baseline Dice 0.900',native=native)],tmp_path/'native.png')

def test_frozen_missing_current_dice_remains_missing(cohort,tmp_path,monkeypatch):
    import report_visuals
    from generate_nnunet_result_report import _summary_figure
    _,doc,info=cohort
    entries=doc['entries'];raw=np.zeros((8,8));mask=raw.astype(bool)
    items=[dict(cid=e['case_id'],group=f"{e['size_group']} | baseline-{e['baseline_role']} | baseline Dice {e['baseline_dice']:.3f}",slices=[1,3,4],panels={z:(raw,mask,mask,raw) for z in (1,3,4)}) for e in entries]
    rows={i['cid']:{'dice':'NA'} for i in items}
    seen=[]
    def capture(fig,path,**kw):
        text='\n'.join(t.get_text() for ax in fig.axes for t in ax.texts)
        assert text.count('current Dice N/A')==6
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        headers=[ax for ax in fig.axes if ax.texts and 'baseline-' in ax.texts[0].get_text()]
        assert len(headers)==6
        # Three rows; each same-size good/bad pair lies at equal height.
        for a,b in zip(headers[::2],headers[1::2]):
            assert a.get_position().y0==pytest.approx(b.get_position().y0)
            assert a.get_position().x0<b.get_position().x0
        for ax in headers:
            text_box=ax.texts[0].get_window_extent(renderer);owner=ax.get_window_extent(renderer)
            assert text_box.x0>=owner.x0 and text_box.x1<=owner.x1 and text_box.y0>=owner.y0 and text_box.y1<=owner.y1
        seen.append(True)
    monkeypatch.setattr(report_visuals,'save_figure',capture)
    _summary_figure(items,rows,None,3,tmp_path/'summary.png')
    assert seen


@pytest.mark.parametrize('mutation',['checkpoint','sourcehash','populationhash','schema_type'])
def test_strict_source_identity_schema(cohort,mutation):
    _,doc,_=cohort;doc=copy.deepcopy(doc)
    if mutation=='checkpoint':doc['baseline']['checkpoint_status']='VERIFIED'
    if mutation=='sourcehash':doc['baseline']['plans_sha256']='bad'
    if mutation=='populationhash':doc['baseline']['prediction_fingerprints'].pop('case6')
    if mutation=='schema_type':doc['entries'][0]=['invalid entry']
    doc['selection_id']=selection.selection_id(doc)
    with pytest.raises(ValueError):selection.validate(doc)

def test_generator_missing_dice_fallback_explicit_thresholds(tmp_path):
    args=make_cohort(tmp_path)
    args.thresholds_ml=[.2,.4]
    args.baseline_metrics_file.write_text('case_id\n'+'\n'.join(f'case{i}' for i in range(7))+'\n',encoding='utf-8')
    doc=selection.generate(args)
    assert doc['grouping']['thresholds_ml']==[.2,.4]
    assert [e['case_id'] for e in doc['entries']]==['case4','case5','case2','case3','case0','case1']
    assert all(e['baseline_dice']==1 and e['dice_source'].startswith('recomputed saved prediction/GT') for e in doc['entries'])


@pytest.mark.parametrize('module,required',[('generate_report_comparison_selection',['--baseline-model-dir','--fold','--splits-file','--images-dir','--labels-dir','--prediction-dir','--baseline-metrics-file','--output-json','--thresholds-ml']),('generate_nnunet_result_report',['--source','--selection-json','--manifest','--config','--checkpoint','--check','--allow-pending','--confirm-prediction-checkpoint','--prediction-checkpoint-declaration'])])
def test_delivered_server_cli_help(module,required,capsys):
    import importlib
    with pytest.raises(SystemExit) as result:importlib.import_module(module).main(['--help'])
    assert result.value.code==0
    text=capsys.readouterr().out
    assert all(option in text for option in required)


# Exercise the real generator guard; only the voxel IO boundary is substituted.
METADATA_FAULTS = [
    ('dataset', 'adc', {'channel_names': {'0': 'ADC'}}),
    ('dataset', 'multichannel', {'channel_names': {'0': 'DWI', '1': 'ADC'}}),
    ('dataset', 'wrong_channel', {'channel_names': {'1': 'DWI'}}),
    ('dataset', 'channels_list', {'channel_names': ['DWI']}),
    ('dataset', 'channels_value', {'channel_names': {'0': 0}}),
    ('dataset', 'missing_channels', {'channel_names': None}),
    ('dataset', 'reversed', {'labels': {'background': 1, 'lesion': 0}}),
    ('dataset', 'foreground2', {'labels': {'background': 0, 'lesion': 2}}),
    ('dataset', 'bool_label', {'labels': {'background': False, 'lesion': 1}}),
    ('dataset', 'float_label', {'labels': {'background': 0, 'lesion': 1.0}}),
    ('dataset', 'string_label', {'labels': {'background': 0, 'lesion': '1'}}),
    ('dataset', 'region_label', {'labels': {'background': 0, 'lesion': [1]}}),
    ('dataset', 'duplicate_label', {'labels': {'background': 0, 'lesion': 0}}),
    ('dataset', 'extra_label', {'labels': {'background': 0, 'lesion': 1, 'other': 2}}),
    ('dataset', 'missing_labels', {'labels': None}),
    ('dataset', 'labels_list', {'labels': [0, 1]}),
    ('dataset', 'ignore_label', {'ignore_label': 2}),
    ('dataset', 'ignore_index', {'ignore_index': 255}),
    ('dataset', 'ignore_class', {'labels': {'background': 0, 'ignore': 1}}),
    ('dataset', 'regions', {'regions_class_order': [1]}),
    ('dataset', 'name_conflict', {'name': 'Dataset508_StrokeLesion'}),
    ('dataset', 'dataset_name_conflict', {'dataset_name': 'Dataset503_StrokeLesion'}),
    ('dataset', 'dataset_id_conflict', {'dataset_id': 508}),
    ('dataset', 'name_type', {'name': 501}),
    ('dataset', 'dataset_id_bool', {'dataset_id': True}),
    ('plans', 'wrong_reader', {'image_reader_writer': 'NibabelIO'}),
    ('plans', 'reader_type', {'image_reader_writer': ['SimpleITKIO']}),
    ('plans', 'missing_reader', {'image_reader_writer': None}),
    ('plans', 'missing_2d', {'configurations': {}}),
    ('plans', 'configurations_type', {'configurations': []}),
    ('plans', '2d_type', {'configurations': {'2d': []}}),
    ('plans', 'missing_architecture', {'configurations': {'2d': {}}}),
    ('plans', 'architecture_type', {'configurations': {'2d': {'architecture': []}}}),
    ('plans', 'wrong_architecture', {'configurations': {'2d': {'architecture': {'network_class_name': 'ResEncUNet'}}}}),
    ('plans', 'architecture_class_type', {'configurations': {'2d': {'architecture': {'network_class_name': 1}}}}),
    ('plans', 'dataset_name_conflict', {'dataset_name': 'Dataset508_StrokeLesion'}),
    ('plans', 'ignore', {'ignore_label': 2}),
]


@pytest.mark.parametrize('target,fault,updates', METADATA_FAULTS,
                         ids=[f'{t}-{f}' for t, f, _ in METADATA_FAULTS])
@pytest.mark.parametrize('existing_output', [False, True])
def test_generator_metadata_rejection_before_voxels(tmp_path, monkeypatch, target, fault, updates, existing_output):
    args = make_cohort(tmp_path)
    path = args.baseline_model_dir / (target + '.json')
    metadata = selection.read_json(path)
    for key, value in updates.items():
        if value is None:
            metadata.pop(key, None)
        else:
            metadata[key] = value
    path.write_text(json.dumps(metadata), encoding='utf-8')
    _assert_metadata_rejected(args, monkeypatch, existing_output)


def _assert_metadata_rejected(args, monkeypatch, existing_output=False):
    if existing_output:
        args.output_json.parent.mkdir()
        args.output_json.write_bytes(b'existing output must survive')
    before = {p: p.read_bytes() for p in args.splits_file.parent.rglob('*') if p.is_file()}
    reads = []
    def forbidden(*a, **kw):
        reads.append(a)
        raise AssertionError('invalid metadata reached voxel IO')
    monkeypatch.setattr(selection, '_volume', forbidden)
    with pytest.raises((ValueError, OSError)) as error:
        selection.generate(args)
    assert any(word in str(error.value) for word in ('dataset', 'plans'))
    assert not reads
    assert {p: p.read_bytes() for p in args.splits_file.parent.rglob('*') if p.is_file()} == before
    if not existing_output:
        assert not args.output_json.exists() and not args.output_json.parent.exists()


@pytest.mark.parametrize('target', ['dataset', 'plans'])
@pytest.mark.parametrize('content', ['NOT JSON', '[]', 'null', '1', '"text"',
                                    '{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'])
def test_generator_invalid_metadata_json(tmp_path, monkeypatch, target, content):
    args = make_cohort(tmp_path)
    (args.baseline_model_dir / (target + '.json')).write_text(content, encoding='utf-8')
    _assert_metadata_rejected(args, monkeypatch)


@pytest.mark.parametrize('target', ['dataset', 'plans'])
def test_generator_missing_metadata(tmp_path, monkeypatch, target):
    args = make_cohort(tmp_path)
    (args.baseline_model_dir / (target + '.json')).unlink()
    _assert_metadata_rejected(args, monkeypatch)


@pytest.mark.parametrize('fault', ['missing2d_wrongreader', 'both_not_json'])
def test_generator_review_combined_metadata_cases(tmp_path, monkeypatch, fault):
    args = make_cohort(tmp_path)
    if fault == 'both_not_json':
        for name in ('dataset.json', 'plans.json'):
            (args.baseline_model_dir / name).write_text('NOT JSON', encoding='utf-8')
    else:
        (args.baseline_model_dir / 'plans.json').write_text(json.dumps(
            {'image_reader_writer': 'NibabelIO', 'configurations': {}}), encoding='utf-8')
    _assert_metadata_rejected(args, monkeypatch)


@pytest.mark.parametrize('checkpoint', ['missing', 'unreadable'])
def test_generator_valid_metadata_without_checkpoint_access(tmp_path, monkeypatch, checkpoint):
    args = make_cohort(tmp_path)
    dataset_path = args.baseline_model_dir / 'dataset.json'
    dataset = selection.read_json(dataset_path)
    dataset.pop('name')  # Nonstandard identity fields are optional.
    dataset['labels'] = {'background': 0, 'stroke': 1}
    dataset_path.write_text(json.dumps(dataset), encoding='utf-8')
    cp = args.baseline_model_dir / 'fold_0' / 'checkpoint_best.pth'
    if checkpoint == 'missing':
        cp.unlink()
    original_open = Path.open
    def guarded_open(path, *a, **kw):
        if path.suffix == '.pth':
            raise AssertionError('baseline checkpoint must never be opened')
        return original_open(path, *a, **kw)
    monkeypatch.setattr(Path, 'open', guarded_open)
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file() and p.suffix != '.pth'}
    doc = selection.generate(args)
    from generate_nnunet_result_report import collect
    info = dict(rows={cid: {'dice': .5} for cid in doc['population']},
                images=collect(args.images_dir, input_channel=True), labels=collect(args.labels_dir),
                predictions=collect(args.prediction_dir))
    selection.load_selection(args.output_json, info)
    selection.verify_selection(info)
    assert info['selection_verified']
    assert [e['case_id'] for e in doc['entries']] == ['case4', 'case5', 'case2', 'case3', 'case0', 'case1']
    assert [(e['size_group'], e['baseline_role']) for e in doc['entries']] == selection.ORDER
    assert all(e['display_slices'] == [1, 3, 4] and e['representative_slice'] == 3 for e in doc['entries'])
    assert doc['grouping']['thresholds_ml'] == pytest.approx([.256, .416])
    assert doc['population'] == [f'case{i}' for i in range(7)]
    assert doc['baseline']['checkpoint_status'] == 'UNKNOWN; saved prediction checkpoint never loaded or independently verified'
    assert doc['generator_code_sha256']['report_comparison_selection.py'] == selection.file_hash(Path(selection.__file__))
    assert all(p.read_bytes() == content for p, content in before.items())


@pytest.mark.parametrize('target', ['dataset', 'plans'])
def test_generator_unreadable_metadata(tmp_path, monkeypatch, target):
    args = make_cohort(tmp_path)
    path = args.baseline_model_dir / (target + '.json')
    read_text = Path.read_text
    def unreadable(candidate, *a, **kw):
        if candidate == path:
            raise PermissionError('synthetic metadata access denied')
        return read_text(candidate, *a, **kw)
    monkeypatch.setattr(Path, 'read_text', unreadable)
    _assert_metadata_rejected(args, monkeypatch)


@pytest.mark.parametrize('dataset_id', [501, '501', 'Dataset501', selection.DATASET])
def test_generator_consistent_optional_identity(tmp_path, dataset_id):
    args = make_cohort(tmp_path)
    for name in ('dataset', 'plans'):
        path = args.baseline_model_dir / (name + '.json')
        metadata = selection.read_json(path)
        metadata.update(dataset_id=dataset_id, dataset_name=selection.DATASET, name='StrokeLesion')
        path.write_text(json.dumps(metadata), encoding='utf-8')
    assert len(selection.generate(args)['entries']) == 6
