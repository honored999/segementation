"""Small synthetic contract tests; no real H2Former or checkpoint is loaded."""
import copy
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

pytest.importorskip("matplotlib").use("Agg")

from standalone_h2former_report import _architecture, _capture, _identity, _manifest_cases, _slice_features, _source_statement, inspect
from generate_nnunet_result_report import protected_output
from standalone_nnunet2d.models.factory import (
    H2FORMER, H2FORMER_LITE_UPERNET_W128_PPM1236 as B,
    PPM1236_ARCHITECTURE, get_model_contract,
)


def _metadata(name):
    model = get_model_contract(name).as_dict()
    result = {"model_name": name, "supervision_mode": "single_output",
              "resolved_config": {"model": model}}
    if name == B:
        result["architecture"] = dict(PPM1236_ARCHITECTURE)
    return result


@pytest.mark.parametrize("name", [H2FORMER, B])
def test_identity_accepts_exact_checkpoint_config_manifest(name):
    metadata = _metadata(name)
    config = copy.deepcopy(metadata["resolved_config"])
    manifest = {"checkpoint": copy.deepcopy(metadata)}
    if name == B:
        config["model"]["ppm_scales"] = list(config["model"]["ppm_scales"])
        manifest["checkpoint"]["architecture"]["ppm_scales"] = [1, 2, 3, 6]
        manifest["checkpoint"]["resolved_config"]["model"]["ppm_scales"] = [1, 2, 3, 6]
    assert _identity(metadata, config, manifest) == name


def test_identity_rejects_variant_conflict_and_wrong_architecture():
    metadata = _metadata(B)
    wrong = copy.deepcopy(metadata["resolved_config"])
    wrong["model"]["name"] = H2FORMER
    with pytest.raises(ValueError):
        _identity(metadata, wrong, {})
    wrong = copy.deepcopy(metadata)
    wrong["architecture"]["ppm_scales"] = (1, 2, 4)
    with pytest.raises(ValueError):
        _identity(wrong, metadata["resolved_config"], {})
    manifest = {"checkpoint": _metadata(H2FORMER)}
    serialized_config = copy.deepcopy(metadata["resolved_config"])
    serialized_config["model"]["ppm_scales"] = [1, 2, 3, 6]
    with pytest.raises(ValueError, match="manifest"):
        _identity(metadata, serialized_config, manifest)


def test_unknown_and_user_confirmed_source_labels():
    assert _source_statement(False, None).startswith("UNKNOWN")
    assert _source_statement(True, "  chosen best  ") == "USER CONFIRMED (user statement only): chosen best"
    with pytest.raises(ValueError, match="confirmation"):
        _source_statement(True, "  ")
    with pytest.raises(ValueError, match="confirmation"):
        _source_statement(False, "unsupported claim")


def test_manifest_case_paths_and_pending(tmp_path):
    source = tmp_path / "case_0000.nii.gz"
    pred = tmp_path / "case.nii.gz"
    manifest = {"schema_version": 1,
                "policy": {"output_space": "source", "run_state": "official_alignment_pending",
                           "alignment_status": "official_alignment_pending"},
                "cases": [{"case_id": "case", "source_path": str(source),
                           "prediction_path": str(pred)}]}
    with pytest.raises(ValueError, match="pending"):
        _manifest_cases(manifest, {"case": source}, {"case": pred}, allow_pending=False)
    assert _manifest_cases(manifest, {"case": source}, {"case": pred}, allow_pending=True)
    manifest["cases"][0]["prediction_path"] = str(source)
    with pytest.raises(ValueError, match="path mismatch"):
        _manifest_cases(manifest, {"case": source}, {"case": pred}, allow_pending=True)


def test_output_overlap_rejected(tmp_path):
    with pytest.raises(ValueError, match="overlap"):
        protected_output(tmp_path / "source" / "child", [tmp_path / "source"])


def test_inspect_accepts_prediction_subset_of_raw_cases(tmp_path):
    roots = {name: tmp_path / name for name in ("model", "images", "labels", "predictions", "metrics")}
    for root in roots.values():
        root.mkdir()
    ids = [f"case{i}" for i in range(6)]
    for cid in [*ids, "extra_training_case"]:
        (roots["images"] / f"{cid}_0000.nii.gz").touch()
        (roots["labels"] / f"{cid}.nii.gz").touch()
    for cid in ids:
        (roots["predictions"] / f"{cid}.nii.gz").touch()
    fields = ("dice", "iou", "f2", "avd_percent", "lcd", "recall", "hd95_mm")
    (roots["metrics"] / "case_metrics.csv").write_text(
        "case_id,"+",".join(fields)+"\n"+"".join(f"{cid},"+",".join(["0.5"]*len(fields))+"\n" for cid in ids))
    (roots["metrics"] / "summary_metrics.json").write_text(json.dumps(
        {"n_cases": 6, "metrics": {key: {"valid_cases": 6} for key in fields}}))
    checkpoint = roots["model"] / "checkpoint_best.pth"
    checkpoint.touch()
    config = roots["model"] / "resolved_config.json"
    config.write_text(json.dumps({"model": get_model_contract(H2FORMER).as_dict()}))
    manifest = roots["predictions"] / "prediction_manifest.json"
    manifest.write_text(json.dumps({"schema_version":1,
        "policy":{"output_space":"source","run_state":"official_alignment_pending","alignment_status":"official_alignment_pending"},
        "cases":[{"case_id":cid,"source_path":str(roots["images"] / f"{cid}_0000.nii.gz"),
                  "prediction_path":str(roots["predictions"] / f"{cid}.nii.gz")} for cid in ids]}))
    args = SimpleNamespace(model_dir=roots["model"],images_dir=roots["images"],labels_dir=roots["labels"],
                           prediction_dir=roots["predictions"],metrics_dir=roots["metrics"],
                           checkpoint=checkpoint,config=config,manifest=manifest,output_dir=tmp_path / "new_report",
                           fold=0,allow_pending=True)
    info = inspect(args)
    assert set(info["images"]) == set(ids)
    assert set(info["predictions"]) == set(ids)


def test_capture_fused_decoder_inputs_and_cleanup():
    torch = pytest.importorskip("torch")

    class Decode(torch.nn.Module):
        def forward(self, deep, middle):
            return deep.mean() + middle.mean()

    class Model(torch.nn.Module):
        image_size = 512

        def __init__(self, bad=False):
            super().__init__()
            self.decode4 = Decode()
            self.bad = bad

        def forward(self, tile):
            deep = torch.arange(32*32, dtype=torch.float32).reshape(1, 1, 32, 32).expand(1, 512, 32, 32)
            middle = torch.ones(1, 256, 64 if not self.bad else 32, 64)
            return self.decode4(deep, middle)

    model = Model()
    got = _capture(model, torch.zeros(1, 1, 512, 512))
    assert got["deep"]["channels"].shape == (8, 32, 32)
    assert got["middle"]["channels"].shape == (8, 64, 64)
    assert got["deep"]["channels"][0, 0, 1] == 1
    assert len(model.decode4._forward_pre_hooks) == 0
    bad = Model(bad=True)
    with pytest.raises(ValueError, match="BCHW shape"):
        _capture(bad, torch.zeros(1, 1, 512, 512))
    assert len(bad.decode4._forward_pre_hooks) == 0


def test_lite_capture_same_window_and_source_mapping(tmp_path):
    import numpy as np
    torch = pytest.importorskip("torch")

    class Decoder(torch.nn.Module):
        pool_scales = (1, 2, 3, 6)

        def __init__(self):
            super().__init__()
            self.classifier = torch.nn.Conv2d(128, 2, 1)

        def forward(self, features, *, output_size):
            return torch.zeros(1, 2, *output_size)

    class Lite(torch.nn.Module):
        image_size = 64

        def __init__(self):
            super().__init__()
            self.decoder = Decoder()
            self.calls = 0

        def forward(self, tile):
            self.calls += 1
            return self.decoder((torch.ones(1, 64, 32, 32),
                                 torch.ones(1, 128, 16, 16),
                                 torch.ones(1, 256, 8, 8),
                                 torch.ones(1, 512, 4, 4)), output_size=tile.shape[-2:])

    model = Lite()
    maps, deep, middle = _slice_features(model, np.ones((1, 40, 40), np.float32), [0], torch.device("cpu"))
    assert model.calls == 1
    assert maps[0].shape == (40, 40)
    assert np.allclose(maps[0], 1)
    assert deep["window_index"] == middle["window_index"] == 0
    assert deep["window_count"] == middle["window_count"] == 1
    assert deep["channels"].shape == (8, 4, 4)
    assert middle["channels"].shape == (8, 8, 8)
    assert not model.decoder._forward_pre_hooks
    overview = tmp_path / "overview.png"
    detail = tmp_path / "detail.png"
    _architecture(model, "h2former_lite_upernet_w128_ppm1236", overview)
    _architecture(model, "h2former_lite_upernet_w128_ppm1236", detail, detail=True)
    assert overview.stat().st_size > 1000 and detail.stat().st_size > 1000


class _ReachedLoader(RuntimeError):
    pass


def _run_probe(monkeypatch, tmp_path, name=H2FORMER, state="official_alignment_pending"):
    import standalone_h2former_report as report
    import standalone_nnunet2d.predict as predict
    import SimpleITK as sitk
    metadata = _metadata(name)
    metadata.update(run_type=state, run_state=state)
    config = json.loads(json.dumps(metadata["resolved_config"]))
    manifest = json.loads(json.dumps({"checkpoint": metadata, "schema_version": 1,
        "policy": {"output_space": "source", "run_state": state, "alignment_status": state}, "cases": []}))
    cp = tmp_path / "synthetic.pth"
    cp.write_bytes(b"synthetic placeholder; never loaded")
    info = dict(paths={"checkpoint": cp, "config": tmp_path / "config.json",
                      "manifest": tmp_path / "manifest.json", "metrics_dir": tmp_path / "metrics"},
                output=tmp_path / "report", images={}, labels={}, predictions={}, rows={},
                summary={}, config=config, manifest=manifest)
    def inspected(args):
        info["policy"] = report._manifest_cases(manifest, info["images"], info["predictions"],
                                                allow_pending=args.allow_pending)
        return info
    monkeypatch.setattr(report, "inspect", inspected)
    monkeypatch.setattr(predict, "_read_checkpoint", lambda path: ({}, metadata))
    calls = []
    def loader(*args):
        calls.append(1)
        raise _ReachedLoader()
    monkeypatch.setattr(predict, "_load_model", loader)
    args = SimpleNamespace(check=False, allow_pending=True, device="cpu",
                           confirm_prediction_checkpoint=False, prediction_checkpoint_declaration=None)
    return report, predict, args, info, metadata, manifest, calls


@pytest.mark.parametrize("bad", ["missing", "list", "empty", "partial", "conflict", "incomplete_ppm", "bad_nested_type"])
def test_manifest_identity_rejected_before_loader(monkeypatch, tmp_path, bad):
    report, _, args, info, _, manifest, calls = _run_probe(monkeypatch, tmp_path, B)
    if bad == "missing":
        manifest.pop("checkpoint")
    elif bad == "list":
        manifest["checkpoint"] = []
    elif bad == "empty":
        manifest["checkpoint"] = {}
    elif bad == "partial":
        manifest["checkpoint"] = {"model_name": B}
    elif bad == "conflict":
        manifest["checkpoint"]["resolved_config"]["model"]["name"] = H2FORMER
    elif bad == "bad_nested_type":
        manifest["checkpoint"]["resolved_config"] = []
    else:
        manifest["checkpoint"]["architecture"].pop("ppm_scales")
    with pytest.raises(ValueError):
        report.run(args)
    assert len(calls) == 0


def _synthetic_evidence(tmp_path):
    # Existing production-validator fixture; this is engineering evidence only.
    from standalone_nnunet2d.tests.test_alignment_evidence import _valid_evidence
    return _valid_evidence(tmp_path)


@pytest.mark.parametrize("bad", ["aligned_missing", "aligned_forged", "aligned_conflict", "pending_forged", "pending_unapproved", "state_conflict"])
def test_manifest_alignment_rejected_before_loader(monkeypatch, tmp_path, bad):
    report, _, args, _, metadata, manifest, calls = _run_probe(monkeypatch, tmp_path)
    if bad.startswith("aligned"):
        metadata.update(run_type="official_aligned", run_state="official_aligned", alignment_evidence=_synthetic_evidence(tmp_path))
        policy = manifest["policy"]
        policy.update(run_state="official_aligned", alignment_status="official_aligned")
        if bad == "aligned_forged":
            policy["alignment_evidence"] = {"bogus": True}
        elif bad == "aligned_conflict":
            policy["alignment_evidence"] = _synthetic_evidence(tmp_path)
            policy["alignment_evidence"]["inference"]["oracle_repeat_count"] = 4
    elif bad == "pending_forged":
        manifest["policy"]["alignment_evidence"] = {"bogus": True}
    elif bad == "pending_unapproved":
        args.allow_pending = False
    else:
        manifest["policy"].update(run_state="official_aligned", alignment_status="official_aligned",
                                   alignment_evidence=_synthetic_evidence(tmp_path))
    with pytest.raises(ValueError):
        report.run(args)
    assert not calls


@pytest.mark.parametrize("name", [H2FORMER, B])
@pytest.mark.parametrize("aligned", [False, True])
def test_valid_identity_alignment_reaches_loader(monkeypatch, tmp_path, name, aligned):
    report, _, args, _, metadata, manifest, calls = _run_probe(monkeypatch, tmp_path, name)
    if aligned:
        evidence = _synthetic_evidence(tmp_path)
        metadata.update(run_type="official_aligned", run_state="official_aligned", alignment_evidence=evidence)
        manifest["policy"].update(run_state="official_aligned", alignment_status="official_aligned", alignment_evidence=evidence)
    with pytest.raises(_ReachedLoader):
        report.run(args)
    assert len(calls) == 1


def _tiny_model(lite=True):
    import torch
    class Decoder(torch.nn.Module):
        pool_scales = (1, 2, 3, 6)
        in_channels = (64, 128, 256, 512)
        def __init__(self):
            super().__init__()
            self.classifier = torch.nn.Conv2d(128, 2, 1)
        def forward(self, *features, **kwargs):
            return torch.zeros(1, 2, 64, 64)
    class Tiny(torch.nn.Module):
        image_size = 64
        def __init__(self):
            super().__init__()
            setattr(self, "decoder" if lite else "decode4", Decoder())
        def forward(self, tile):
            fs = tuple(torch.ones(1, c, 64//(2**(i+1)), 64//(2**(i+1))) for i,c in enumerate((64,128,256,512)))
            return self.decoder(fs, output_size=(64,64)) if lite else self.decode4(fs[3],fs[2])
    return Tiny()


@pytest.mark.parametrize("lite", [False, True])
def test_architecture_semantic_routes_and_dynamic_sizes(monkeypatch, tmp_path, lite):
    import matplotlib.figure
    saved = []
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", lambda fig,*a,**k: saved.append(fig))
    model = _tiny_model(lite)
    for detail in (False, True):
        _architecture(model, B if lite else H2FORMER, tmp_path / "preview.png", detail=detail)
        fig = saved[-1]
        labels = "\n".join(t.get_text() for ax in fig.axes for t in ax.texts)
        assert "64 x 64" in labels and "512 x 512" not in labels
        assert "window logits" in labels and "diagnostic magnitude" in labels
        edges = {p.get_gid() for ax in fig.axes for p in ax.patches}
        if lite:
            assert {"S1->L1", "S2->L2", "S3->L3", "S4->PPM", "P4->P3", "P3->P2", "P2->P1"} <= edges
            if detail:
                assert {"S4->pool1", "S4->pool2", "S4->pool3", "S4->pool6", "concat->bottleneck", "S4->concat", "P4->fusion"} <= edges
        else:
            assert {"S4->decode4", "S3->decode4", "S2->decode3", "S1->decode2", "decode4->decode3", "decode3->decode2", "decode2->decode0"} <= edges


def test_native_mapping_non_square_multi_window():
    import numpy as np
    import torch
    from standalone_h2former_report import _native_mapping
    maps, deep, middle = _slice_features(_tiny_model(), np.ones((2,45,105),np.float32), [1],torch.device("cpu"))
    d, m = _native_mapping(deep), _native_mapping(middle)
    assert maps[1].shape == (45,105)
    for value in (d,m):
        assert value["original_slice"] == 1
        assert value["window_index"] == 0 and value["window_count"] == 3
        assert value["window_yx_padded"] == (0,0)
        assert value["padding_offsets_yx"] == ((9,10),(0,0))
        assert value["source_shape_yx"] == (45,105) and value["padded_shape_yx"] == (64,105)
        assert value["window_shape_yx"] == (64,64)
        assert "channels" not in value
    assert d["feature_shape"] == (1,512,4,4) and m["feature_shape"] == (1,256,8,8)
    assert d["channel_ids"] == deep["channel_ids"]
    assert d["module"] == m["module"] == "decoder"
    assert (d["input_index"], m["input_index"]) == (3,2)


@pytest.mark.parametrize("lite", [False, True])
def test_six_outputs_and_failure_cleanup(monkeypatch, tmp_path, lite):
    import numpy as np
    import SimpleITK as sitk
    import matplotlib.figure
    report, predict, args, info, metadata, manifest, calls = _run_probe(monkeypatch,tmp_path,B if lite else H2FORMER)
    config = info["config"]
    if lite:
        config["data_source"] = {"type": "nnunet_preprocessed_b2nd", "root": "synthetic training reference only"}
    fields = ("dice","iou","f2","avd_percent","lcd","recall","hd95_mm")
    for i in range(6):
        cid=f"synthetic{i}"
        for kind in ("images","labels","predictions"):
            root=tmp_path/kind; root.mkdir(exist_ok=True)
            path=root/(cid+".nii.gz")
            array=np.arange(2*45*105,dtype=np.float32).reshape(2,45,105) if kind=="images" else np.zeros((2,45,105),np.uint8)
            if kind!="images": array[1,15:20,35:40]=1
            sitk.WriteImage(sitk.GetImageFromArray(array),str(path))
            info[kind][cid]=path.resolve()
        manifest["cases"].append({"case_id":cid,"source_path":str(info["images"][cid]),"prediction_path":str(info["predictions"][cid])})
        info["rows"][cid]={k:(i+.1)/6 for k in fields}
    model=_tiny_model(lite)
    monkeypatch.setattr(predict,"_load_model",lambda *a:(model,metadata))
    # Controlled rendering resolution only; all production plotting executes.
    save=matplotlib.figure.Figure.savefig
    def small(fig,*a,**kw):
        kw["dpi"]=60
        return save(fig,*a,**kw)
    monkeypatch.setattr(matplotlib.figure.Figure,"savefig",small)
    assert report.run(args)==0
    expected={"summary.png","architecture_overview.png","architecture_detail.png","feature_channels.png","feature_channels_64x64.png","report.txt"}
    assert {p.name for p in info["output"].iterdir()}==expected
    assert all((info["output"]/p).stat().st_size>0 for p in expected)
    txt=(info["output"]/"report.txt").read_text(encoding="utf-8")
    assert "Training data backend: " + ("nnunet_preprocessed_b2nd" if lite else "unknown") in txt
    assert "Diagnostic prediction entry: source NIfTI" in txt
    deep=[json.loads(line.split(": ",1)[1]) for line in txt.splitlines() if line.startswith("Native deep mapping:")]
    mid=[json.loads(line.split(": ",1)[1]) for line in txt.splitlines() if line.startswith("Native middle mapping:")]
    assert len(deep)==len(mid)==6
    for d,m in zip(deep,mid):
        for k in ("original_slice","window_index","window_yx_padded","window_count","padding_offsets_yx","source_shape_yx","padded_shape_yx","window_shape_yx"):
            assert d[k]==m[k]
        assert d["original_slice"]==1 and d["window_count"]==3
        assert d["padding_offsets_yx"]==[[9,10],[0,0]]
        assert d["source_shape_yx"]==[45,105] and d["padded_shape_yx"]==[64,105]
        assert d["feature_shape"]==[1,512,4,4] and m["feature_shape"]==[1,256,8,8]
        assert "channels" not in d and "channels" not in m
    args_output=tmp_path/"failure_report"
    info["output"]=args_output
    def fail(*a,**k): raise RuntimeError("synthetic plotting failure")
    monkeypatch.setattr(report,"_architecture",fail)
    with pytest.raises(RuntimeError,match="synthetic plotting failure"): report.run(args)
    assert not args_output.exists() and not list(tmp_path.glob(".failure_report-*"))


@pytest.mark.parametrize("evidence", [None, {"bogus": True}])
def test_manifest_aligned_evidence_required(evidence):
    manifest={"schema_version":1, "policy":{"output_space":"source", "run_state":"official_aligned",
        "alignment_status":"official_aligned", "alignment_evidence": evidence}, "cases":[]}
    with pytest.raises(ValueError, match="evidence"):
        _manifest_cases(manifest, {}, {}, allow_pending=True)
