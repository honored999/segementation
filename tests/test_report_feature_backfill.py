import copy
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import batch_generate_result_reports as batch
import generate_nnunet_result_report as report
import report_comparison_selection as selection


def document():
    ids = [f"case{i:02}" for i in range(19)]
    h = "a" * 64
    geometry = dict(Size=[8,8,5], Spacing=[1,1,1], Origin=[0,0,0], Direction=[1,0,0,0,1,0,0,0,1])
    entries = []
    for i, (group, role) in enumerate(selection.ORDER):
        entries.append(dict(case_id=ids[i], size_group=group, baseline_role=role,
                            baseline_dice=.5, dice_source="saved", gt_volume_ml={"large":3,"medium":2,"small":1}[group],
                            display_slices=[1], representative_slice=1,
                            image=dict(locator=ids[i]+"_0000.nii.gz",sha256=h,geometry=geometry),
                            gt=dict(locator=ids[i]+".nii.gz",sha256=h,geometry=geometry)))
    doc = dict(schema_version=1,dataset=selection.DATASET,fold=0,population=ids,entries=entries,
               split_sha256=h,baseline=dict(trainer_plans_configuration=selection.BASELINE,
               plans_sha256=h,dataset_sha256=h,metrics_sha256=h,prediction_fingerprints=dict.fromkeys(ids,h),
               checkpoint_status="UNKNOWN; saved prediction checkpoint never loaded or independently verified"),
               generator_code_sha256=dict.fromkeys(("report_comparison_selection.py","generate_report_comparison_selection.py"),h),
               axis=dict(array_order="z/y/x (SimpleITK); not NIfTI x/y/z or anatomical plane",axis=0,index_base=0),
               grouping=dict(thresholds_ml=[1,2]))
    doc["selection_id"] = selection.selection_id(doc)
    return selection.validate(doc)


def setup_batch(tmp_path):
    old = tmp_path / "batch_reports_dataset501_v2"
    saved = old / "job"
    saved.mkdir(parents=True)
    doc = document()
    sel = old / "comparison_selection.json"
    sel.write_text(json.dumps(doc))
    (saved / "comparison_selection.json").write_text(json.dumps(doc))
    paths = {}
    for key in ("model","prediction","metrics","images","labels"):
        paths[key] = tmp_path / key
        paths[key].mkdir()
    row = dict(status="saved",source="official-nnunet",model=str(paths["model"]),prediction=str(paths["prediction"]),metrics=str(paths["metrics"]),report=str(saved),features="not requested")
    summary = old / "batch_summary.csv"
    with summary.open("w",newline="",encoding="utf-8-sig") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(row));writer.writeheader();writer.writerow(row)
        writer.writerow(dict(row,status="skipped",report="",reason="") if "reason" in row else dict(row,status="skipped",report=""))
    args=SimpleNamespace(existing_summary=summary,output_dir=tmp_path/"batch_reports_dataset501_v2_features_v1",selection_json=sel,
                         images_dir=paths["images"],labels_dir=paths["labels"],trainer_extensions={},dry_run=False,features=True,features_only=True,
                         allow_pending=False,device="cpu")
    return args, row


@pytest.mark.parametrize("config", ["2d_stage5","2d_stage6"])
def test_exact_depth_checkpoint_metadata(tmp_path, config):
    model=tmp_path/f"nnUNetTrainerPlainConvDepthTopK10EarlyStopping__nnUNetPlansPlainConvDepth__{config}"
    model.mkdir()
    plans=dict(plans_name="nnUNetPlansPlainConvDepth",configurations={config:dict(encoder_last_stage=int(config[-1]))})
    (model/"plans.json").write_text(json.dumps(plans))
    metadata=dict(trainer_name=model.name.split("__")[0],init_args=dict(configuration=config,fold=0,plans=plans))
    report.validate_checkpoint_identity(metadata,model,strict=True)
    for key,value in (("configuration","2d"),("fold",1),("fold",None)):
        bad=copy.deepcopy(metadata);bad["init_args"][key]=value
        with pytest.raises(ValueError): report.validate_checkpoint_identity(bad,model,strict=True)
    bad=copy.deepcopy(metadata);bad["init_args"]["plans"]["configurations"][config]["encoder_last_stage"]=99
    with pytest.raises(ValueError,match="plans differ"): report.validate_checkpoint_identity(bad,model,strict=True)


@pytest.mark.parametrize("name", [
    "nnUNetTrainerPlainConvDepthTopK10EarlyStopping__nnUNetPlans__2d_stage6",
    "nnUNetTrainerPlainConvDepthTopK10EarlyStopping__nnUNetPlansPlainConvDepth__2d_stage7",
    "nnUNetTrainerUPerNetSelectedStagesTopK10EarlyStopping__nnUNetPlansUPerNetStages_s3456__2d",
    "nnUNetTrainerForeground50__nnUNetPlans__2d_unknown"])
def test_arbitrary_suffix_rejected(name):
    with pytest.raises(ValueError): report.model_identity(Path(name))


def test_selected_stages_exact_name():
    assert report.model_identity(Path("nnUNetTrainerUPerNetSelectedStagesTopK10EarlyStopping__nnUNetPlansUPerNetStages_s2345__2d"))[2] == "2d"


@pytest.mark.parametrize("feature_result,status", [("skipped: correct source missing","feature_skipped"),("generated: diagnostics","feature_saved"),("failed: generator exit 1","feature_failed")])
def test_backfill_no_saved_report_or_scan(tmp_path, monkeypatch, feature_result,status):
    args,row=setup_batch(tmp_path)
    original=args.existing_summary.read_bytes()
    calls=[]
    def forbidden(*a,**kw): raise AssertionError("full report path must not run")
    for name in ("scan","discover","prepare","saved_report","verify_volumes"):
        monkeypatch.setattr(batch,name,forbidden)
    def features(job,args,target,log):
        calls.append(job)
        return feature_result,status=="feature_failed"
    monkeypatch.setattr(batch,"features",features)
    assert batch.backfill(args)==int(status=="feature_failed")
    records=json.loads((args.output_dir/"batch_summary.json").read_text())["records"]
    assert len(calls)==1 and records[-1]["status"]==status
    assert records[0]["features"].startswith("not run:")
    with (args.output_dir/"batch_summary.csv").open(encoding="utf-8-sig") as handle:
        csvrows=list(csv.DictReader(handle))
    assert csvrows[-1]["existing_report"]==row["report"]
    assert args.existing_summary.read_bytes()==original
    assert not (args.output_dir/"metrics_table.csv").exists()


@pytest.mark.parametrize("unsafe", ["existing","nested","outside_sibling"])
def test_backfill_output_boundary(tmp_path,monkeypatch,unsafe):
    args,_=setup_batch(tmp_path)
    args.output_dir={"existing":args.existing_summary.parent,"nested":args.existing_summary.parent/"new","outside_sibling":tmp_path/"elsewhere"/"new"}[unsafe]
    with pytest.raises(ValueError): batch.backfill(args)


def test_extension_exact_file_identity(tmp_path,monkeypatch):
    name="nnUNetTrainerForeground50"
    path=tmp_path/(name+".py");path.write_text("class "+name+": pass")
    from types import ModuleType
    module=ModuleType(name);module.__file__=str(path)
    exec(compile(path.read_text(),str(path),"exec"),module.__dict__)
    monkeypatch.setitem(sys.modules,name,module)
    monkeypatch.setenv("nnUNet_extTrainer",str(tmp_path))
    assert report.resolve_external_trainer(lambda _:getattr(module,name),name,tmp_path).__name__==name
    module.__file__=str(tmp_path/"wrong.py")
    with pytest.raises(ValueError,match="class file identity"):report.resolve_external_trainer(lambda _:getattr(module,name),name,tmp_path)


def test_missing_source_truthful_skip(tmp_path):
    model=tmp_path/"nnUNetTrainerGroupedTopK10__nnUNetPlans__2d"
    args=SimpleNamespace(features_only=True,trainer_extensions={"nnUNetTrainerGroupedTopK10":tmp_path})
    command,note=batch.feature_command(dict(source="official-nnunet",model=model),args,tmp_path/"out")
    assert command is None and "class file missing" in note


def test_h2_hash_guard_unchanged(tmp_path):
    cp=tmp_path/"checkpoint_best.pt";cp.write_bytes(b"current mutable weights")
    job=dict(source="standalone-h2former",manifest_data=dict(checkpoint=dict(path=str(cp),sha256="a"*64)))
    command,note=batch.feature_command(job,SimpleNamespace(),tmp_path/"out")
    assert command is None and "hash mismatch" in note


@pytest.mark.parametrize("trainer,count,config", [
    ("nnUNetTrainerPlainConvDepthTopK10EarlyStopping",6,dict(encoder_last_stage=5)),
    ("nnUNetTrainerUPerNetNoStage7TopK10EarlyStopping",7,{}),
    ("nnUNetTrainerUPerNetSelectedStagesTopK10EarlyStopping",8,dict(architecture=dict(arch_kwargs=dict(n_stages=8)),upernet_feature_indices=[2,3,4,5]))])
def test_actual_runtime_encoder_metadata(trainer,count,config):
    encoder=SimpleNamespace(stages=list(range(count)),output_channels=[32]*count,strides=[(2,2)]*count)
    network=SimpleNamespace(encoder=encoder,selected_feature_indices=[2,3,4,5])
    report.validate_runtime_encoder(network,trainer,config)
    encoder.stages.pop()
    with pytest.raises(ValueError,match="stage count"): report.validate_runtime_encoder(network,trainer,config)


def test_features_only_main_never_builds_metric_table(tmp_path,monkeypatch):
    output=tmp_path/"fresh"
    info=dict(model=tmp_path/"nnUNetTrainerTopK10__nnUNetPlans__2d",output=output,
              rows=dict.fromkeys(document()["population"],{}),selection=document(),
              plans=dict(image_reader_writer="SimpleITKIO"),images={},labels={},predictions={},
              metrics=tmp_path/"metrics",checkpoint=tmp_path/"cp")
    monkeypatch.setattr(report,"inspect_sources",lambda _:info)
    counts=[]
    monkeypatch.setattr(report,"verify_full_geometry",lambda i:counts.append(len(i["rows"])) or len(i["rows"]))
    monkeypatch.setattr(report,"verify_selection",lambda _:None)
    monkeypatch.setattr(report,"_predictor",lambda *_:object())
    names=("encoder_stages_heatmap.png","summary.png","feature_channels.png","feature_channels_64x64.png","report.txt")
    def create(info,*_):
        for name in names:(info["output"]/name).write_text("synthetic output sentinel")
    monkeypatch.setattr(report,"create_report",create)
    monkeypatch.setattr(report,"save_selection",lambda *_:"")
    monkeypatch.setattr(report,"check_selection_output",lambda *_:None)
    def forbidden(*a,**kw):raise AssertionError("metrics must not run")
    monkeypatch.setattr(report,"build_table",forbidden)
    import report_visuals
    monkeypatch.setattr(report_visuals,"validate_visual_outputs",lambda *_:None)
    args=["--model-dir",str(info["model"]),"--fold","0","--images-dir",str(tmp_path),"--labels-dir",str(tmp_path),
          "--checkpoint",str(info["checkpoint"]),"--selection-json",str(tmp_path/"selection.json"),"--output-dir",str(output),"--features-only"]
    assert report.main(args)==0
    assert counts==[6]
    assert set(p.name for p in output.iterdir())==set(names)


@pytest.mark.parametrize("mutation", ["parent_spacing","transpose","reader"])
def test_checkpoint_inherited_global_plans_mismatch(tmp_path,mutation):
    name="nnUNetTrainerPlainConvDepthTopK10EarlyStopping"
    model=tmp_path/(name+"__nnUNetPlansPlainConvDepth__2d_stage6");model.mkdir()
    plans=dict(plans_name="nnUNetPlansPlainConvDepth",transpose_forward=[0,1,2],image_reader_writer="SimpleITKIO",
               configurations={"base":dict(spacing=[1,1]),"2d_stage6":dict(inherits_from="base",encoder_last_stage=6)})
    (model/"plans.json").write_text(json.dumps(plans))
    saved=copy.deepcopy(plans)
    if mutation=="parent_spacing":saved["configurations"]["base"]["spacing"]=[2,2]
    elif mutation=="transpose":saved["transpose_forward"]=[2,1,0]
    else:saved["image_reader_writer"]="NibabelIO"
    metadata=dict(trainer_name=name,init_args=dict(configuration="2d_stage6",fold=0,plans=saved))
    with pytest.raises(ValueError,match="full plans differ"):report.validate_checkpoint_identity(metadata,model,strict=True)


def test_upernet_extension_not_overwritten(tmp_path):
    model=tmp_path/(report.TRAINER+"__nnUNetPlans__2d")
    (model/"fold_0").mkdir(parents=True)
    (model/"fold_0"/"checkpoint_best.pth").write_bytes(b"sentinel")
    (tmp_path/(report.TRAINER+".py")).write_text("# source path sentinel")
    args=SimpleNamespace(features_only=True,trainer_extensions={report.TRAINER:tmp_path},images_dir=tmp_path,labels_dir=tmp_path,selection_json=tmp_path/"sel",device="cpu")
    command,_=batch.feature_command(dict(source="official-nnunet",model=model,prediction=tmp_path,metrics=tmp_path),args,tmp_path/"output")
    assert command[command.index("--trainer-extension-dir")+1]==str(tmp_path)
    assert "--prediction-checkpoint-declaration" in command and "--features-only" in command


@pytest.mark.parametrize("mutation", ["path","hash","missing_hash","valid"])
def test_direct_h2_snapshot_guard(tmp_path,mutation):
    from standalone_h2former_report import require_snapshot_checkpoint
    from report_comparison_selection import file_hash
    cp=tmp_path/"best400.pt";cp.write_bytes(b"historical snapshot")
    current=tmp_path/"best.pt";current.write_bytes(b"mutable current")
    manifest=dict(checkpoint=dict(path=str(cp),sha256=file_hash(cp)))
    passed=cp
    if mutation=="path":passed=current
    elif mutation=="hash":manifest["checkpoint"]["sha256"]="a"*64
    elif mutation=="missing_hash":del manifest["checkpoint"]["sha256"]
    if mutation=="valid":require_snapshot_checkpoint(manifest,passed)
    else:
        with pytest.raises(ValueError):require_snapshot_checkpoint(manifest,passed)


@pytest.mark.parametrize("run,export,slot", [
    ("H2Former_fold0_bs4_preprocessed","full_volume_predictions_best1000","best"),
    ("H2Former_fold0_bs4_preprocessed","full_volume_predictions_latest1000","latest"),
    ("H2Former_fold0_bs4_adamw","full_volume_predictions_best","best"),
    ("H2Former_LiteUPerNet_AdamW_EarlyStop_bs4/fold_0","latest_prediction","latest"),
    ("H2Former_UPerNet_W128_PPM1236/fold_0","full_volume_predictions_best","best")])
def test_exact_final_h2_slots(tmp_path,run,export,slot):
    from standalone_h2former_report import final_h2_slot
    model=tmp_path/run
    assert final_h2_slot(model,model/export/"predictions")==slot


@pytest.mark.parametrize("run,export", [
    ("H2Former_fold0_bs4_preprocessed","full_volume_predictions_best400"),
    ("H2Former_fold0_bs4_preprocessed","full_volume_predictions_best2000"),
    ("H2Former_fold0_bs4_preprocessed","full_volume_predictions_latest800"),
    ("H2Former_fold0_bs4_screening","full_volume_predictions_best"),
    ("H2Former_fold0_bs4_adamw","full_volume_predictions_best1000"),
    ("unknown_run","full_volume_predictions_best")])
def test_no_numeric_largest_or_screening_selection(tmp_path,run,export):
    from standalone_h2former_report import final_h2_slot
    model=tmp_path/run
    with pytest.raises(ValueError):final_h2_slot(model,model/export/"predictions")


def final_rows(tmp_path):
    rows=[dict(status="saved",source="official-nnunet",model=str(tmp_path/f"official{i}"),prediction=str(tmp_path/f"pred{i}"),report=str(tmp_path/f"report{i}")) for i in range(12)]
    for run,exports in (
        ("H2Former_fold0_bs4_preprocessed",["full_volume_predictions_best1000","full_volume_predictions_latest1000","full_volume_predictions_best400","full_volume_predictions_best600","full_volume_predictions_best800","full_volume_predictions_latest600","full_volume_predictions_latest800"]),
        ("H2Former_fold0_bs4_adamw",["full_volume_predictions_best","full_volume_predictions_latest"]),
        ("H2Former_LiteUPerNet_AdamW_EarlyStop_bs4/fold_0",["best_prediction","latest_prediction"]),
        ("H2Former_UPerNet_W128_PPM1236/fold_0",["best_prediction","full_volume_predictions_best"]),
        ("H2Former_fold0_bs4_screening",["full_volume_predictions"])):
        model=tmp_path/run
        rows.extend(dict(status="saved",source="standalone-h2former",model=str(model),prediction=str(model/export/"predictions"),report=str(tmp_path/(run.replace('/','_')+export))) for export in exports)
    rows.append(dict(rows[0],status="skipped"))
    return rows


def test_final_allowlist_19_and_exclusion_reasons_7(tmp_path):
    rows=final_rows(tmp_path)
    kept,excluded=batch.select_final_records(rows)
    assert len(kept)==19 and len(excluded)==7
    assert sum(r["source"]=="official-nnunet" for r in kept)==12
    assert all(r["exclusion_reason"] for r in excluded)
    w128=[r for r in kept if "W128" in r["model"]]
    assert len(w128)==1 and "full_volume_predictions_best" in w128[0]["prediction"]
    summary=tmp_path/"summary.csv";summary.write_text("metadata sentinel")
    doc=batch.write_final_allowlist(tmp_path,rows,summary)
    assert doc["retained_count"]==19 and len(doc["unrun"])==1
    with (tmp_path/"final_report_allowlist.csv").open(encoding="utf-8-sig") as handle:
        assert len(list(csv.DictReader(handle)))==19


@pytest.mark.parametrize("features,final,current,valid", [(True,True,True,True),(True,True,False,True),(True,False,True,False),(False,True,True,False),(False,True,False,False)])
def test_current_batch_mode_requires_explicit_final_features(features,final,current,valid):
    args=SimpleNamespace(features_only=features,final_h2_only=final,use_current_h2_checkpoints=current)
    if valid:batch.validate_backfill_flags(args)
    else:
        with pytest.raises(ValueError):batch.validate_backfill_flags(args)


@pytest.mark.parametrize("layout", ["root","checkpoints","missing","conflict","manifest_disambiguated","outside"])
def test_current_checkpoint_exact_unique_slots(tmp_path,layout):
    from standalone_h2former_report import current_h2_checkpoint
    model=tmp_path/"H2Former_fold0_bs4_adamw";model.mkdir()
    (model/"checkpoints").mkdir()
    root=model/"checkpoint_best.pth";nested=model/"checkpoints"/"checkpoint_best.pth"
    manifest=dict(checkpoint={})
    if layout in ("root","conflict","manifest_disambiguated"):root.write_bytes(b"synthetic current root")
    if layout in ("checkpoints","conflict","manifest_disambiguated"):nested.write_bytes(b"synthetic current nested")
    if layout=="manifest_disambiguated":manifest["checkpoint"]["path"]=str(nested)
    if layout=="outside":
        outside=tmp_path/"checkpoint_best.pth";outside.write_bytes(b"outside")
        manifest["checkpoint"]["path"]=str(outside)
    pred=model/"full_volume_predictions_best"/"predictions"
    if layout in ("missing","conflict","outside"):
        with pytest.raises(ValueError):current_h2_checkpoint(model,pred,manifest)
    else:
        expected=nested if layout in ("checkpoints","manifest_disambiguated") else root
        assert current_h2_checkpoint(model,pred,manifest)==expected


@pytest.mark.parametrize("mutation", ["normal","no_final","no_features","confirmed","historical_export"])
def test_direct_current_diagnostic_guard(tmp_path,mutation):
    from standalone_h2former_report import validate_diagnostic_current_mode
    model=tmp_path/"H2Former_fold0_bs4_preprocessed"
    args=SimpleNamespace(diagnostic_current_checkpoint=True,features_only=True,final_h2_only=True,source="standalone-h2former",
                         confirm_prediction_checkpoint=False,prediction_checkpoint_declaration=None,model_dir=model,
                         prediction_dir=model/"full_volume_predictions_best1000"/"predictions")
    if mutation=="no_final":args.final_h2_only=False
    if mutation=="no_features":args.features_only=False
    if mutation=="confirmed":args.confirm_prediction_checkpoint=True
    if mutation=="historical_export":args.prediction_dir=model/"full_volume_predictions_best400"/"predictions"
    if mutation=="normal":validate_diagnostic_current_mode(args)
    else:
        with pytest.raises(ValueError):validate_diagnostic_current_mode(args)


def test_current_feature_command_bypasses_only_snapshot_hash(tmp_path):
    model=tmp_path/"H2Former_fold0_bs4_preprocessed";model.mkdir()
    cp=model/"checkpoint_best.pth";cp.write_bytes(b"synthetic current checkpoint")
    job=dict(source="standalone-h2former",model=model,prediction=model/"full_volume_predictions_best1000"/"predictions",
             metrics=tmp_path/"metrics",manifest=tmp_path/"manifest.json",config=tmp_path/"config.json",existing_report=tmp_path/"oldreport",
             manifest_data=dict(checkpoint=dict(path=str(model/"gone.pth"),sha256="a"*64)))
    args=SimpleNamespace(features_only=True,final_h2_only=True,use_current_h2_checkpoints=True,allow_pending=True,
                         images_dir=tmp_path,labels_dir=tmp_path,selection_json=tmp_path/"sel",device="cpu")
    command,_=batch.feature_command(job,args,tmp_path/"out")
    assert command[command.index("--checkpoint")+1]==str(cp)
    assert "--diagnostic-current-checkpoint" in command and "--final-h2-only" in command
    assert "--existing-report" in command and "--confirm-prediction-checkpoint" not in command
    args.use_current_h2_checkpoints=False
    command,note=batch.feature_command(job,args,tmp_path/"out")
    assert command is None and "SHA256 proof" in note


@pytest.mark.parametrize("mutate", [False,True])
def test_current_checkpoint_stability_after_capture(tmp_path,mutate):
    from standalone_h2former_report import require_stable_current_checkpoint
    from report_comparison_selection import file_hash
    cp=tmp_path/"checkpoint_best.pth";cp.write_bytes(b"synthetic originally loaded weights")
    digest=file_hash(cp)
    require_stable_current_checkpoint(cp,digest)
    if mutate:
        cp.write_bytes(b"synthetic changed during capture")
        with pytest.raises(ValueError,match="abort publication"):
            require_stable_current_checkpoint(cp,digest)
    else:
        require_stable_current_checkpoint(cp,digest)


def test_historical_mode_does_not_add_current_digest_reads(tmp_path):
    from standalone_h2former_report import require_stable_current_checkpoint
    require_stable_current_checkpoint(tmp_path/"not_read.pth",None)
