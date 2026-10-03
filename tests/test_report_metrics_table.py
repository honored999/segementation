"""CPU-only synthetic report-table contracts; no model forward is permitted."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import report_metrics_table as table
from generate_nnunet_result_report import check_geometry, read_metrics, select_cases


def fixture_rows(n=7):
    rows = {}
    for i in reversed(range(n)):
        cid = f"case{i}"
        rows[cid] = dict(case_id=cid, dice="0.5", iou=str(1/3), f2="0.5", avd_percent="0", lcd="0", recall="0.5", hd95_mm="2.123456789012345", tp="1", fp="1", fn="1", text="preserved", pred_voxels="2", gt_voxels="2")
    return rows


def summary(rows):
    return dict(n_cases=len(rows), aggregation="macro average over cases", f2_mode="paper",
                metrics={k:dict(valid_cases=sum(table.metric_value(r[k]) is not None for r in rows.values()),
                                mean=np.mean([table.metric_value(r[k]) for r in rows.values() if table.metric_value(r[k]) is not None])
                                if any(table.metric_value(r[k]) is not None for r in rows.values()) else float('nan')) for k in table.METRICS})


def build(rows, **kw):
    return table.build_table(rows, summary(rows), {c:Path(c) for c in rows}, {c:Path(c) for c in rows}, **kw)


@pytest.mark.parametrize('p,g,expected', [([1,1,0,0],[1,0,1,0],(1,1,1)),([0,0],[0,0],(0,0,0)),([0,0],[1,1],(0,0,2)),([1,1],[0,0],(0,2,0)),([1,0],[1,0],(1,0,0))])
def test_manual_voxels(p,g,expected):
    assert tuple(table.voxel_counts(p,g).values()) == expected


def forbidden(*a,**kw):
    raise AssertionError('unexpected heavy/voxel dependency')


def test_complete_counts_no_reads_exact_integer(monkeypatch):
    monkeypatch.setattr(table, 'saved_mask_counts', forbidden)
    rows=fixture_rows(1); r=rows['case0']; n=2**53+17
    r.update(tp=str(n),fp='0',fn='0',pred_voxels=str(n),gt_voxels=str(n),dice='1',iou='1',recall='1',f2='1')
    result=build(rows,label_contract=forbidden)
    assert result['records'][0]['tp']==n and result['records'][-1]['tp']==n
    assert result['records'][0]['tp_source']=='source_csv'


def test_missing_counts_one_load_per_case(monkeypatch):
    import evaluate_segmentation_metrics as evaluation
    monkeypatch.setattr(evaluation,'evaluate_case',forbidden)
    monkeypatch.setattr(evaluation,'hd95_mm',forbidden)
    rows=fixture_rows(); rows['case1'].pop('fp'); rows['case2'].update(tp='',fp='',fn='')
    seen=[]
    def loader(p,g,c,f,geometry):
        seen.append(c)
        return dict(tp=1,fp=1,fn=1)
    result=build(rows,label_contract=lambda:1,check_geometry=check_geometry,loader=loader)
    assert seen==['case1','case2']
    r=result['records'][1]
    assert r['tp_source']=='source_csv' and r['fp_source']=='recomputed_from_saved_masks'
    rows['case1']['tp']='2'
    with pytest.raises(ValueError,match='conflicts with saved masks'):
        build(rows,label_contract=lambda:1,check_geometry=check_geometry,loader=loader)


@pytest.mark.parametrize('bad',[-1,1.0,True,'True','-1','1.5','1.0','NaN','Inf','1e3'])
def test_illegal_counts(bad):
    rows=fixture_rows(1); rows['case0']['tp']=bad
    with pytest.raises(ValueError,match='integer'): build(rows)


def test_alias_conflicts_and_missing():
    rows=fixture_rows(1); rows['case0']['TP']='2'
    with pytest.raises(ValueError,match='conflicting'): build(rows)
    rows['case0']['TP']='1'
    assert build(rows)['records'][0]['TP']==1
    rows['case0']['fp']=''
    with pytest.raises(ValueError,match='missing counts'): build(rows)


@pytest.mark.parametrize('key', ['Size','Spacing','Origin','Direction'])
def test_saved_masks_geometry(tmp_path,key):
    import SimpleITK as sitk
    a=sitk.GetImageFromArray(np.zeros((2,2,2),np.uint8)); b=sitk.Image(a)
    if key=='Size': b=sitk.GetImageFromArray(np.zeros((3,2,2),np.uint8))
    elif key=='Spacing': b.SetSpacing((2,1,1))
    elif key=='Origin': b.SetOrigin((1,0,0))
    else: b.SetDirection((0,1,0,1,0,0,0,0,-1))
    p,g=tmp_path/'p.nii.gz',tmp_path/'g.nii.gz'
    sitk.WriteImage(a,str(p)); sitk.WriteImage(b,str(g))
    with pytest.raises(ValueError,match=key): table.saved_mask_counts(p,g,'synthetic',1,check_geometry)


@pytest.mark.parametrize('bad',[2,-1,.5,float('nan'),float('inf')])
def test_invalid_mask_labels(bad):
    with pytest.raises(ValueError,match='invalid binary'): table.voxel_counts([bad],[0])


@pytest.mark.parametrize('doc',[{}, {'labels':{'background':0,'a':1,'b':2}}, {'labels':{'background':0,'a':[1]}}, {'labels':{'background':0,'ignore':1}}, {'labels':{'background':0,'a':1},'ignore_label':-1}, {'labels':{'background':0,'a':1},'regions_class_order':[1]}])
def test_unsupported_label_contract(doc):
    with pytest.raises(ValueError): table.binary_labels(doc)


def test_binary_contract_evidence():
    assert table.binary_labels({'labels':{'background':0,'lesion':1}})==1
    assert table.binary_labels({'model':{'num_classes':2}},standalone=True)==1
    with pytest.raises(ValueError): table.binary_labels({'model':{'num_classes':2},'labels':{'background':0,'lesion':2}},standalone=True)


def test_summary_missing_and_full_table_precision(tmp_path):
    rows=fixture_rows(8); rows['case7']['dice']='NaN'; rows['case7']['hd95_mm']='Inf'
    result=build(rows)
    assert len(result['records'])==10
    assert result['records'][7]['case_id']=='case7'
    assert 'case7' not in sum(select_cases(rows),[])
    txt=table.export_table(result,tmp_path)
    exported=list(csv.DictReader((tmp_path/'metrics_table.csv').open(encoding='utf-8-sig')))
    assert exported[0]['hd95_mm']=='2.123456789012345'
    assert exported[7]['dice']=='NA' and 'N/A' in txt
    assert exported[-2]['tp']=='' and exported[-1]['dice']==''
    assert exported[-1]['tp']=='8'
    assert all(r['text']=='preserved' for r in exported[:-2])
    assert 'valid N=7' in txt
    assert (tmp_path/'metrics_table.png').stat().st_size>1000
    rows['case6']['dice']='NaN'
    rows['case5']['dice']='NaN'
    with pytest.raises(ValueError,match='finite Dice'): select_cases(rows)


@pytest.mark.parametrize('fault',['n_cases','valid_cases','mean','case_ids','unknown','inf_valid'])
def test_summary_conflicts(fault):
    rows=fixture_rows(); s=summary(rows)
    if fault=='n_cases': s['n_cases']=8
    elif fault=='case_ids': s['case_ids']=['wrong']*7
    elif fault=='unknown': s.pop('aggregation')
    elif fault=='mean': s['metrics']['dice']['mean']=.2
    elif fault=='inf_valid': rows['case0']['hd95_mm']='Inf'
    else: s['metrics']['dice']['valid_cases']=6
    with pytest.raises(ValueError): table.build_table(rows,s,dict.fromkeys(rows),dict.fromkeys(rows))


def test_summary_fallback_unknown_f2_and_extra_columns():
    rows=fixture_rows(); s=summary(rows); s.pop('f2_mode'); s['metrics']['dice'].pop('mean')
    for r in rows.values(): r['f2']='.2718'
    s['metrics']['f2']['mean']=.2718
    result=table.build_table(rows,s,dict.fromkeys(rows),dict.fromkeys(rows))
    assert result['records'][-2]['dice']==.5
    assert 'text' not in result['records'][-2]
    assert 'unknown (not assumed)' in '\n'.join(result['notes'])


def test_source_consistency_and_coverage():
    rows=fixture_rows(1); rows['case0']['pred_voxels']='3'
    with pytest.raises(ValueError,match='pred_voxels'): build(rows)
    rows['case0']['pred_voxels']='2'; rows['case0']['dice']='.7'
    with pytest.raises(ValueError,match='dice conflicts'):
        table.build_table(rows,declared_summary(rows),dict.fromkeys(rows),dict.fromkeys(rows))
    with pytest.raises(ValueError,match='coverage'): table.build_table(rows,summary(rows),{},dict.fromkeys(rows))


def write_inputs(root, identity):
    import SimpleITK as sitk
    import generate_nnunet_result_report as official
    standalone=identity not in official.SUPPORTED_TRAINERS
    model=root/'Dataset501_StrokeLesion'/(identity if standalone else identity+'__nnUNetPlans__2d')
    dirs={k:root/k for k in ('images','labels','predictions','metrics')}
    model.mkdir(parents=True)
    for d in dirs.values(): d.mkdir()
    checkpoint=model/('checkpoint_best.pth' if standalone else 'fold_0/checkpoint_best.pth')
    checkpoint.parent.mkdir(exist_ok=True); checkpoint.write_bytes(b'NEVER LOADED SYNTHETIC')
    rows=fixture_rows()
    for cid,r in rows.items():
        for k,values in [('images',[1,2,3,4]),('labels',[1,0,1,0]),('predictions',[1,1,0,0])]:
            path=dirs[k]/(cid+('_0000' if k=='images' else '')+'.nii.gz')
            sitk.WriteImage(sitk.GetImageFromArray(np.array(values,np.uint8).reshape(1,2,2)),str(path))
        r.pop('tp'); r.pop('fp'); r.pop('fn')
    with (dirs['metrics']/'case_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(next(iter(rows.values())))); writer.writeheader(); writer.writerows(rows.values())
    (dirs['metrics']/'summary_metrics.json').write_text(json.dumps(summary(rows)))
    args=['--model-dir',str(model),'--fold','0','--images-dir',str(dirs['images']),'--labels-dir',str(dirs['labels']), '--prediction-dir',str(dirs['predictions']),'--metrics-dir',str(dirs['metrics']), '--checkpoint',str(checkpoint),'--output-dir',str(root/'output'),'--device','cpu']
    metadata=None
    if standalone:
        from test_standalone_h2former_report import _metadata
        metadata=_metadata(identity); metadata.update(run_type='official_alignment_pending',run_state='official_alignment_pending')
        config=model/'resolved_config.json'; config.write_text(json.dumps(metadata['resolved_config']))
        manifest=dirs['predictions']/'prediction_manifest.json'
        manifest.write_text(json.dumps(dict(schema_version=1,checkpoint=metadata,policy=dict(output_space='source',run_state='official_alignment_pending',alignment_status='official_alignment_pending'),cases=[dict(case_id=c,source_path=str(dirs['images']/(c+'_0000.nii.gz')),prediction_path=str(dirs['predictions']/(c+'.nii.gz'))) for c in rows])))
        args+=['--source','standalone-h2former','--config',str(config),'--manifest',str(manifest),'--allow-pending']
    else:
        (model/'dataset.json').write_text(json.dumps(dict(channel_names={'0':'DWI'},labels={'background':0,'lesion':1})))
        (model/'plans.json').write_text(json.dumps(dict(image_reader_writer='SimpleITKIO',configurations={'2d':{}})))
        if identity==official.TRAINER: args+=['--prediction-checkpoint-declaration','synthetic user statement']
    return args,metadata


ENTRIES=['nnUNetTrainerTopK10','nnUNetTrainerUPerNetTopK10EarlyStopping','h2former','h2former_lite_upernet','h2former_lite_upernet_w128_ppm1236']


def wire_no_forward(monkeypatch,metadata):
    """Entry/provenance tests substitute rendering only, keeping publication guards real."""
    import generate_nnunet_result_report as official
    import report_visuals
    from PIL import Image
    def artifact(path,family):
        path=Path(path); path.parent.mkdir(exist_ok=True,parents=True)
        Image.new('RGB',(64,36),'white').save(path)
    monkeypatch.setattr(table,'render_table',lambda h,c,n,p: artifact(p,'metrics_table'))
    monkeypatch.setattr(report_visuals,'stage_figure',lambda i,r,p: artifact(p,'encoder_stages'))
    if metadata is None:
        monkeypatch.setattr(official,'_predictor',lambda *a: object())
        def diagnostic(info,*a):
            for family in ('summary','encoder_stages','feature_channels','feature_channels_64x64','architecture_overview','architecture_detail'):
                artifact(info['output']/(('encoder_stages_heatmap' if family=='encoder_stages' else family)+'.png'),family)
            (info['output']/'report.txt').write_text('synthetic diagnostic replacement',encoding='utf-8')
        monkeypatch.setattr(official,'create_report',diagnostic)
    else:
        import standalone_h2former_report as report
        import standalone_nnunet2d.predict as predict
        monkeypatch.setattr(predict,'_read_checkpoint',lambda *a: ({},metadata))
        monkeypatch.setattr(predict,'_load_model',lambda *a: (SimpleNamespace(),metadata))
        monkeypatch.setattr(report,'_slice_features',lambda *a,**kw: ({0:np.zeros((2,2))},None,None))
        def plot(*a,**kw):
            target=next(x for x in reversed(a) if isinstance(x,Path))
            artifact(target,target.stem)
        monkeypatch.setattr(report,'_summary_figure',plot)
        monkeypatch.setattr(report,'_native_channel_figure',plot)
        monkeypatch.setattr(report,'_architecture',plot)


@pytest.mark.parametrize('identity',ENTRIES)
def test_real_entry_shared_table_no_forward(tmp_path,monkeypatch,identity):
    from generate_nnunet_result_report import main
    args,metadata=write_inputs(tmp_path,identity)
    wire_no_forward(monkeypatch,metadata)
    assert main(args)==0
    assert not (tmp_path/'output'/'ppt').exists()
    output=tmp_path/'output'
    records=list(csv.DictReader((output/'metrics_table.csv').open(encoding='utf-8-sig')))
    assert len(records)==9 and records[-1]['tp']=='7'
    assert records[0]['tp_source']=='recomputed_from_saved_masks'
    assert 'Full evaluation metrics table' in (output/'report.txt').read_text(encoding='utf-8')


@pytest.mark.parametrize('identity',[ENTRIES[1],ENTRIES[2]])
@pytest.mark.parametrize('failure',['csv','png','render','count'])
def test_real_entry_atomic_failures(tmp_path,monkeypatch,identity,failure):
    from generate_nnunet_result_report import main
    args,metadata=write_inputs(tmp_path,identity); wire_no_forward(monkeypatch,metadata)
    if failure=='count': monkeypatch.setattr(table,'saved_mask_counts',lambda *a: (_ for _ in ()).throw(ValueError('count failure')))
    elif failure=='render': monkeypatch.setattr(table,'render_table',lambda *a: (_ for _ in ()).throw(RuntimeError('render failure')))
    else:
        real=table.export_table
        def missing(t,out):
            result=real(t,out); (out/('metrics_table.'+failure)).unlink(); return result
        import generate_nnunet_result_report as o
        import standalone_h2former_report as s
        monkeypatch.setattr(o,'export_table',missing); monkeypatch.setattr(s,'export_table',missing)
    with pytest.raises((SystemExit,RuntimeError)): main(args)
    assert not (tmp_path/'output').exists() and not list(tmp_path.glob('.output-*'))


@pytest.mark.parametrize('identity',ENTRIES)
def test_check_never_reads_voxels_or_model(tmp_path,monkeypatch,identity):
    import SimpleITK as sitk
    import generate_nnunet_result_report as o
    args,metadata=write_inputs(tmp_path,identity)
    monkeypatch.setattr(sitk,'ReadImage',forbidden)
    monkeypatch.setattr(sitk.ImageFileReader,'ReadImageInformation',forbidden)
    monkeypatch.setattr(o,'_predictor',forbidden)
    import standalone_nnunet2d.predict as predict
    monkeypatch.setattr(predict,'_read_checkpoint',forbidden); monkeypatch.setattr(predict,'_load_model',forbidden)
    assert o.main(args+['--check'])==0
    assert not (tmp_path/'output').exists()


def test_long_table_visual_artifacts(tmp_path):
    rows=fixture_rows(12)
    for cid in list(rows):
        r=rows.pop(cid); new=cid+'_long_synthetic_identifier_'*4; r['case_id']=new
        for i in range(3): r[f'extra_text_{i}']='synthetic long preserved text '*4
        rows[new]=r
    table.export_table(build(rows),tmp_path)
    assert (tmp_path/'metrics_table.png').exists()


@pytest.mark.parametrize("mode,expected", [("paper",5/16),("standard",5/19)])
def test_f2_definitions_not_interchanged(mode,expected):
    rows=fixture_rows(1)
    rows["case0"].update(tp="1",fp="2",fn="3",pred_voxels="3",gt_voxels="4",
                        dice=str(2/7),iou=str(1/6),recall="0.25",f2=str(expected))
    source=declared_summary(rows, mode)
    result=table.build_table(rows,source,dict.fromkeys(rows),dict.fromkeys(rows))
    assert result["records"][0]["f2"]==str(expected)
    source["f2_mode"]="standard" if mode=="paper" else "paper"
    with pytest.raises(ValueError,match="f2 conflicts"):
        table.build_table(rows,source,dict.fromkeys(rows),dict.fromkeys(rows))


@pytest.mark.parametrize("fault",["nonnumeric","malformed","duplicate_columns","missing_id","extra_case"])
def test_read_metrics_rejects_invalid_source(tmp_path,fault):
    rows=fixture_rows(1); r=rows["case0"]
    if fault=="nonnumeric": r["hd95_mm"]="not a number"
    source=summary(fixture_rows(1))
    fields=list(r)
    content=",".join(fields)+"\n"+",".join(str(r[k]) for k in fields)+"\n"
    if fault=="malformed": content=content.rstrip()+",extra\n"
    elif fault=="duplicate_columns": content=content.replace("case_id,","case_id,case_id,",1)
    elif fault=="missing_id": content=content.replace("case0,",",",1)
    (tmp_path/"case_metrics.csv").write_text(content,encoding="utf-8")
    (tmp_path/"summary_metrics.json").write_text(json.dumps(source),encoding="utf-8")
    with pytest.raises(ValueError): read_metrics(tmp_path,{"other"} if fault=="extra_case" else {"case0"})


FORMULAS = dict(dice="2tp/(2tp+fp+fn)", iou="tp/(tp+fp+fn)",
                recall="tp/(tp+fn)", f2="5tp/(5tp+4fp+fn)")


def declared_summary(rows, mode="paper"):
    source = summary(rows)
    source["f2_mode"] = mode
    source["metric_definitions"] = {
        k: dict(formula=v, zero_denominator="missing") for k, v in FORMULAS.items()}
    if mode == "standard":
        source["metric_definitions"]["f2"]["formula"] = "5tp/(5tp+fp+4fn)"
    source["count_foreground"] = "positive"
    return source


@pytest.mark.parametrize("missing", ["", "   ", None, "NA", "N/A", "NaN", "Inf"])
def test_missing_export_cells(tmp_path, monkeypatch, missing):
    rows = fixture_rows(1)
    rows["case0"]["dice"] = missing
    captured = []
    monkeypatch.setattr(table, "render_table", lambda h, c, n, p: captured.extend(c))
    txt = table.export_table(build(rows), tmp_path)
    exported = list(csv.DictReader((tmp_path/"metrics_table.csv").open(encoding="utf-8-sig")))
    assert exported[0]["dice"] == exported[-2]["dice"] == "NA"
    assert exported[-1]["dice"] == exported[-2]["tp"] == ""
    assert captured[0][2] == captured[-2][2] == "N/A"
    assert captured[-1][2] == captured[-2][9] == ""
    assert "\tN/A\t" in txt


def test_unknown_formula_not_inferred(monkeypatch):
    monkeypatch.setattr(table, "saved_mask_counts", forbidden)
    rows = fixture_rows(1)
    rows["case0"].update(dice=".7", f2=".8")
    result = build(rows, label_contract=forbidden)
    assert result["records"][0]["dice"] == ".7"
    assert "unknown" in result["records"][0]["count_label_definition"]
    assert "caller-declared" not in result["records"][0]["count_label_definition"]


@pytest.mark.parametrize("mode", ["paper", "standard"])
@pytest.mark.parametrize("key", ["dice", "iou", "recall", "f2"])
def test_declared_zero_denominator_conflict(mode, key):
    rows = fixture_rows(1)
    rows["case0"].update(tp="0", fp="0", fn="0", pred_voxels="0", gt_voxels="0",
                        dice="NA", iou="NA", recall="NA", f2="NA")
    source = declared_summary(rows, mode)
    table.build_table(rows, source, dict.fromkeys(rows), dict.fromkeys(rows))
    rows["case0"][key] = "1"
    with pytest.raises(ValueError, match=key+" conflicts"):
        table.build_table(rows, declared_summary(rows, mode), dict.fromkeys(rows), dict.fromkeys(rows))


@pytest.mark.parametrize("counts,values", [
    ((0, 2, 0), ("0", "0", "NA", "0")),
    ((0, 0, 2), ("0", "0", "0", "0")),
])
def test_declared_empty_boundaries(counts, values):
    rows = fixture_rows(1)
    tp, fp, fn = counts
    rows["case0"].update(zip(("dice", "iou", "recall", "f2"), values))
    rows["case0"].update(tp=str(tp), fp=str(fp), fn=str(fn), pred_voxels=str(tp+fp), gt_voxels=str(tp+fn))
    for mode in ("paper", "standard"):
        result = table.build_table(rows, declared_summary(rows, mode), dict.fromkeys(rows), dict.fromkeys(rows))
        assert result["records"][0]["dice"] == "0"
    if fp:
        rows["case0"]["recall"] = "0"
        with pytest.raises(ValueError, match="recall conflicts"):
            table.build_table(rows, declared_summary(rows), dict.fromkeys(rows), dict.fromkeys(rows))


def test_unknown_empty_definition_and_individual_contracts():
    rows = fixture_rows(1)
    rows["case0"].update(tp="0", fp="0", fn="0", pred_voxels="0", gt_voxels="0",
                        dice="1", iou="1", recall="1", f2="1")
    build(rows)
    source = summary(rows)
    source["metric_definitions"] = {"dice": dict(formula=FORMULAS["dice"], zero_denominator="unknown")}
    table.build_table(rows, source, dict.fromkeys(rows), dict.fromkeys(rows))
    rows["case0"].update(tp="1", fp="1", fn="1", pred_voxels="2", gt_voxels="2", dice=".5", iou=".7")
    source = summary(rows)
    source["metric_definitions"] = {"dice": dict(formula=FORMULAS["dice"], zero_denominator="missing")}
    table.build_table(rows, source, dict.fromkeys(rows), dict.fromkeys(rows))
    rows["case0"]["dice"] = ".7"
    source["metrics"] = summary(rows)["metrics"]
    with pytest.raises(ValueError, match="dice conflicts"):
        table.build_table(rows, source, dict.fromkeys(rows), dict.fromkeys(rows))


@pytest.mark.parametrize("identity", [ENTRIES[1], ENTRIES[2]])
@pytest.mark.parametrize("known", [False, True])
def test_real_entry_definition_contract(tmp_path, monkeypatch, identity, known):
    from generate_nnunet_result_report import main
    args, metadata = write_inputs(tmp_path, identity)
    wire_no_forward(monkeypatch, metadata)
    path = tmp_path/"metrics"/"case_metrics.csv"
    with path.open(newline="") as f:
        reader = csv.DictReader(f); fields = reader.fieldnames; rows = {r["case_id"]: r for r in reader}
    for r in rows.values():
        r.update(tp="1", fp="1", fn="1", dice=".7")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields+["tp", "fp", "fn"])
        writer.writeheader(); writer.writerows(rows.values())
    source = declared_summary(rows) if known else summary(rows)
    (tmp_path/"metrics"/"summary_metrics.json").write_text(json.dumps(source))
    if known:
        with pytest.raises(SystemExit): main(args)
        assert not (tmp_path/"output").exists()
        assert not list(tmp_path.glob(".output-*"))
    else:
        assert main(args) == 0
        text = (tmp_path/"output"/"report.txt").read_text(encoding="utf-8")
        assert "unknown" in text


@pytest.mark.parametrize("mode", ["paper", "standard"])
def test_declared_f2_nonzero_numeric_conflict(mode):
    rows = fixture_rows(1)
    rows["case0"].update(tp="1", fp="2", fn="3", pred_voxels="3", gt_voxels="4",
                        dice=str(2/7), iou=str(1/6), recall=".25", f2=".9")
    with pytest.raises(ValueError, match="f2 conflicts"):
        table.build_table(rows, declared_summary(rows, mode), dict.fromkeys(rows), dict.fromkeys(rows))


@pytest.mark.parametrize("definition", [None, [], {"dice": "wrong"},
    {"dice": {"formula": "guess"}}, {"dice": {"formula": []}},
    {"dice": {"formula": FORMULAS["dice"], "zero_denominator": "one"}},
    {"dice": {"zero_denominator": "missing"}}, {"unknown_metric": {}},
    {"dice": {"evaluator": "guessed"}}])
def test_invalid_definition_declarations(definition):
    rows = fixture_rows(1); source = summary(rows)
    source["metric_definitions"] = definition
    with pytest.raises(ValueError, match="definition|denominator"):
        table.build_table(rows, source, dict.fromkeys(rows), dict.fromkeys(rows))


def test_fallback_labels_do_not_establish_source_definitions():
    rows = fixture_rows(1)
    rows["case0"].update(tp="", dice=".7", f2=".8")
    result = build(rows, label_contract=lambda: 1, check_geometry=check_geometry,
                   loader=lambda *a: dict(tp=1, fp=1, fn=1))
    assert result["records"][0]["dice"] == ".7"
    assert "source foreground unknown" in result["records"][0]["count_label_definition"]


def test_small_missing_real_png(tmp_path, monkeypatch):
    rows = fixture_rows(1); rows["case0"]["dice"] = ""
    real = table.render_table
    captured = []
    def render(headers, cells, notes, path):
        captured.extend(cells)
        real(headers, cells, notes, path)
    monkeypatch.setattr(table, "render_table", render)
    txt = table.export_table(build(rows), tmp_path)
    assert captured[0][2] == captured[-2][2] == "N/A"
    assert captured[-1][2] == captured[-2][9] == ""
    assert "N/A" in txt
    assert (tmp_path/"metrics_table.png").stat().st_size > 1000


@pytest.mark.parametrize("identity", ENTRIES[2:])
@pytest.mark.parametrize("scenario", ["unknown", "mode_only", "paper", "standard", "conflict"])
def test_standalone_full_report_definition_provenance(tmp_path, monkeypatch, capsys, identity, scenario):
    from generate_nnunet_result_report import main
    args, metadata = write_inputs(tmp_path, identity)
    wire_no_forward(monkeypatch, metadata)
    path = tmp_path/"metrics"/"case_metrics.csv"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = {r["case_id"]: r for r in reader}
    known = scenario in ("paper", "standard", "conflict")
    mode = "standard" if scenario == "standard" else "paper"
    # Asymmetric FP/FN distinguishes paper F2 from standard F2.
    for row in rows.values():
        if known:
            row.update(tp="1", fp="2", fn="3", pred_voxels="3", gt_voxels="4",
                       dice=str(2/7), iou=str(1/6),
                       recall=".25", f2=str(5/(19 if mode == "standard" else 16)))
        else:
            row.update(tp="1", fp="1", fn="1", dice=".7")
    if scenario == "conflict":
        rows["case0"]["dice"] = ".7"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields+["tp", "fp", "fn"])
        writer.writeheader()
        writer.writerows(rows.values())
    source = declared_summary(rows, mode) if known else summary(rows)
    if scenario == "unknown":
        source.pop("f2_mode")
    (tmp_path/"metrics"/"summary_metrics.json").write_text(json.dumps(source))
    if scenario == "conflict":
        with pytest.raises(SystemExit):
            main(args)
        assert "dice conflicts" in capsys.readouterr().err
        assert not (tmp_path/"output").exists()
        assert not list(tmp_path.glob(".output-*"))
        return
    assert main(args) == 0
    output = tmp_path/"output"
    text = (output/"report.txt").read_text(encoding="utf-8")
    body, appendix = text.split("Full evaluation metrics table", 1)
    # Inspect the complete report, including body: an unknown appendix cannot
    # excuse a conflicting unconditional definition in the diagnostic prose.
    for assertion in ("Metric definitions from evaluate_segmentation_metrics.py:",
                      "Dice=2TP/(2TP+FP+FN)", "F2 definition:",
                      "F2 paper mode=", "AVD=abs(",
                      "LCD=absolute lesion-count difference", "HD95=95th percentile"):
        assert assertion not in text
    assert "optional source summary metric_definitions" in body
    assert "caller-declared, not independently verified" in body
    assert "Missing declarations remain unknown" in body
    assert "per-metric definitions and provenance in the appended full evaluation metrics table" in body
    assert "Existing metrics are read only; only missing TP/FP/FN are recomputed from saved full-volume masks." in body
    assert "AVD percent, HD95 mm" in body
    assert "Source report f2_mode: " + source.get("f2_mode", "unknown") in body
    assert "mode alone does not establish a formula" in body
    assert "Case metric definitions (caller-declared source summary; not independently verified):" in appendix
    with (output/"metrics_table.csv").open(encoding="utf-8-sig", newline="") as handle:
        exported = list(csv.DictReader(handle))
    cases = [r for r in exported if r["case_id"] in rows]
    assert len(cases) == len(rows)
    for record in cases:
        original = rows[record["case_id"]]
        for key in ("dice", "iou", "recall", "f2", "tp", "fp", "fn"):
            assert float(record[key]) == float(original[key])
            assert f'"{key}": "{original[key]}"' in body
        assert record["tp_source"] == "source_csv"
    if known:
        formula = source["metric_definitions"]["f2"]["formula"]
        other = "5tp/(5tp+4fp+fn)" if mode == "standard" else "5tp/(5tp+fp+4fn)"
        assert "f2: formula=" + formula in appendix
        assert other not in text
        assert "dice: formula=2tp/(2tp+fp+fn)" in appendix
        assert all("caller-declared >0" in r["count_label_definition"] for r in cases)
    else:
        for key in FORMULAS:
            assert key + ": formula=unknown, zero_denominator=unknown" in appendix
        assert "Source count foreground: unknown" in appendix
        assert all("source foreground unknown" in r["count_label_definition"] for r in cases)
        assert all("caller-declared >0" not in r["count_label_definition"] for r in cases)
        assert all(float(r["dice"]) == .7 for r in cases)
