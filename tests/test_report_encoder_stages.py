"""Tiny CPU evidence for diagnostic stages, windows, display and publication."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch

torch.set_num_threads(1)


def official_fixture():
    class Pre:
        def __init__(self, verbose=False): pass
        def run_case(self, images, seg, plans, config, dataset):
            assert seg is None
            a = -np.arange(1, 13, dtype=np.float32).reshape(1, 1, 2, 6)
            return a, None, dict(shape_before_cropping=(1,2,6),
                shape_after_cropping_and_before_resampling=(1,2,6),
                bbox_used_for_cropping=((0,1),(0,2),(0,6)), spacing=(1,1,1))
    class Config:
        preprocessor_class=Pre
        patch_size=(2,3)
        spacing=(1,1)
        def resampling_fn_probabilities(self, a, shape, *args): return a
    class Stage(torch.nn.Module):
        def __init__(self, c, hw): super().__init__(); self.c=c; self.hw=hw
        def forward(self, x):
            return torch.nn.functional.interpolate(x, self.hw, mode='nearest').repeat(1,self.c,1,1)
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.encoder=torch.nn.Module()
            self.encoder.stages=torch.nn.ModuleList([Stage(2,(2,3)),Stage(3,(64,64)),Stage(4,(1,2))])
            self.encoder.output_channels=(2,3,4); self.selected_feature_indices=(1,2)
            self.calls=0; self.mode='ok'
        def forward(self, x):
            self.calls+=1
            for i,s in enumerate(self.encoder.stages):
                if self.mode=='missing' and i==0: continue
                s(x)
                if self.mode=='duplicate' and i==0: s(x)
            if self.mode=='error': raise RuntimeError('tiny forward error')
            return x
    net=Net()
    return SimpleNamespace(network=net, configuration_manager=Config(),
        plans_manager=SimpleNamespace(transpose_forward=(0,1,2),transpose_backward=(0,1,2)),
        dataset_json={},device=torch.device('cpu'),
        _internal_get_sliding_window_slicers=lambda shape:[
            (slice(None),0,slice(0,2),slice(0,3)),(slice(None),0,slice(0,2),slice(3,6))])


def test_all_official_stages_same_center_window():
    from generate_nnunet_result_report import _diagnostic_feature
    p=official_fixture()
    full, provenance, native, middle=_diagnostic_feature(Path('synthetic'),p,2,0,1)
    packet=native['encoder_stages']
    assert p.network.calls==2
    assert native['window_index']==middle['window_index']==packet['window_index']==1
    assert np.array_equal(packet['input'], -np.array([[4,5,6],[10,11,12]],np.float32))
    assert [s['stage'] for s in packet['stages']]==[0,1,2]
    assert packet['stages'][0]['feature_shape']==[1,2,2,3]
    assert np.array_equal(packet['stages'][0]['magnitude'], abs(packet['input']))
    assert [s['module'] for s in provenance['encoder_stages']['stages']]==[
        'encoder.stages.0','encoder.stages.1','encoder.stages.2']
    json.dumps(provenance)
    assert all(s['magnitude'].ndim==2 and isinstance(s['magnitude'],np.ndarray) for s in packet['stages'])
    assert all(not s._forward_hooks for s in p.network.encoder.stages)


@pytest.mark.parametrize('mode', ['missing','duplicate','error'])
def test_official_stage_cleanup(mode):
    from generate_nnunet_result_report import _diagnostic_feature
    p=official_fixture(); p.network.mode=mode
    with pytest.raises((ValueError,RuntimeError)):
        _diagnostic_feature(Path('synthetic'),p,2,0,1)
    assert all(not s._forward_hooks for s in p.network.encoder.stages)


def hybrid_fixture(lite=False):
    class Decode(torch.nn.Module):
        def forward(self, deep, skip): return skip+10000
    class LiteDecode(torch.nn.Module):
        def forward(self, features, **kwargs): return features[0]+10000
    class Hybrid(torch.nn.Module):
        image_size=32
        def __init__(self):
            super().__init__(); self.calls=0; self.mode='ok'; self.expected=None
            if lite: self.decoder=LiteDecode()
            else: self.decode4=Decode(); self.decode3=Decode(); self.decode2=Decode()
        def forward(self, tile):
            self.calls+=1; fs=[]
            for i,c in enumerate((64,128,256,512)):
                side=self.image_size//2**(i+1)
                yy,xx=torch.meshgrid(torch.arange(side),torch.arange(side),indexing='ij')
                # Distinct channel, x/y and window markers; already fused BCHW.
                f=-(100*i + 10*yy + xx + tile[0,0,0,0]).float()[None,None].repeat(1,c,1,1)
                f=f-torch.arange(c)[None,:,None,None]
                fs.append(f)
            self.expected=fs
            if lite: out=self.decoder(tuple(fs),output_size=tile.shape[-2:])
            else:
                out=self.decode4(fs[3],fs[2])
                if self.mode=='duplicate': self.decode4(fs[3],fs[2])
                if self.mode!='missing': out=self.decode3(out,fs[1]); out=self.decode2(out,fs[0])
            if self.mode=='error': raise RuntimeError('tiny hybrid error')
            return out
    return Hybrid()


@pytest.mark.parametrize('lite',[False,True])
def test_hybrid_all_fused_stages_and_first_tile(lite):
    from standalone_h2former_report import _slice_features, _capture, _native_mapping
    m=hybrid_fixture(lite)
    a=np.arange(32*48,dtype=np.float32).reshape(1,32,48)
    maps,deep,middle=_slice_features(m,a,[0],torch.device('cpu'))
    assert m.calls==2 and maps[0].shape==(32,48)
    packet=deep['encoder_stages']
    assert packet['window_index']==deep['window_index']==middle['window_index']==0
    assert np.array_equal(packet['input'],a[0,:32,:32])
    expected=hybrid_fixture(lite); tile=torch.from_numpy(packet['input'])[None,None]
    expected(tile)
    for i,s in enumerate(packet['stages']):
        assert s['stage']==i
        assert s['input_index']==(0 if lite or i==3 else 1)
        if lite: assert s['feature_index']==i
        assert np.array_equal(s['magnitude'],expected.expected[i].abs().mean(1)[0].numpy())
        assert not torch.is_tensor(s['magnitude'])
    assert [s['module'] for s in packet['stages']]==(['decoder']*4 if lite else ['decode2','decode3','decode4','decode4'])
    json.dumps(_native_mapping(deep))
    # Nonselected tiles retain neither stage maps nor channel samples.
    got=_capture(m,tile,retain_native=False)
    assert 'stages' not in got and 'channels' not in got['deep']
    assert 'channels' not in got['middle']
    assert all(not mod._forward_pre_hooks for mod in m.modules())


@pytest.mark.parametrize('mode',['missing','duplicate','error'])
def test_hybrid_cleanup(mode):
    from standalone_h2former_report import _capture
    m=hybrid_fixture(); m.mode=mode
    with pytest.raises((ValueError,RuntimeError)): _capture(m,torch.zeros(1,1,32,32))
    assert all(not mod._forward_pre_hooks for mod in m.modules())


@pytest.mark.parametrize('values,status,count',[
    ([[1,1],[1,1]],'constant',0),
    ([[np.nan,np.inf],[-np.inf,np.nan]],'no_finite_values',4),
    ([[0,np.nan],[np.inf,2]],'finite_with_invalid',2),
    ([[-1e308,0],[1e308,1]],'finite',0)])
def test_display_nonfinite_extreme(values,status,count):
    from report_visuals import normalize_intensity
    display,meta=normalize_intensity(values)
    assert meta['status']==status and meta['nonfinite_count']==count
    assert np.isfinite(display).all() and display.min()>=0 and display.max()<=1
    if status in ('constant','no_finite_values'): assert not display.any()


def test_stage_render_and_preview_metadata(tmp_path):
    from report_visuals import stage_figure, validate_visual_outputs
    p=official_fixture()
    from generate_nnunet_result_report import _diagnostic_feature
    _,_,native,middle=_diagnostic_feature(Path('synthetic'),p,2,0,1)
    items=[dict(cid='合成长病例_0123456789_abcdefghijklmnopqrstuvwxyz', group='High Dice',
        slices=[0],native=native,intermediate_native=middle)]
    stage_figure(items,{items[0]['cid']:dict(dice='.5')},tmp_path/'encoder_stages_heatmap.png')
    assert not (tmp_path/'ppt').exists()
    validate_visual_outputs(tmp_path,required_families=['encoder_stages'])
    (tmp_path/'encoder_stages_heatmap.png').unlink()
    with pytest.raises(ValueError,match='incomplete'): validate_visual_outputs(tmp_path,required_families=['encoder_stages'])


@pytest.mark.parametrize('lite',[False,True])
def test_actual_production_forward_features_spatial_order_and_output(lite):
    from standalone_nnunet2d.models.h2former import H2Former
    from standalone_nnunet2d.models.h2former_lite_upernet import H2FormerLiteUPerNet
    from standalone_h2former_report import _capture
    # No production constructor or full network: run real forward_features and
    # forward with tiny deterministic branch modules on a 32px grid.
    class CNN(torch.nn.Module):
        def __init__(self,c,side,offset): super().__init__(); self.c=c; self.side=side; self.offset=offset
        def forward(self,x):
            x=torch.nn.functional.interpolate(x[:,:1],(self.side,self.side),mode='nearest')
            return x.repeat(1,self.c,1,1)+torch.arange(self.c)[None,:,None,None]+self.offset
    class Patch(torch.nn.Module):
        def forward(self,x): return torch.full((1,16*16,64),100.)
    class MS(torch.nn.Module):
        def __init__(self,c,side,offset): super().__init__(); self.c=c; self.side=side; self.offset=offset
        def forward(self,t): return torch.full((1,self.c,self.side,self.side),self.offset)
    class Swin(torch.nn.Module):
        def forward(self,t): return t+7
    class Decode(torch.nn.Module):
        def forward(self,deep,skip): return skip+10000
    class Lite(torch.nn.Module):
        def forward(self,fs,**kw): return fs[0]+10000
    cls=H2FormerLiteUPerNet if lite else H2Former
    m=cls.__new__(cls); torch.nn.Module.__init__(m)
    m.image_size=32; m.in_channels=1
    m.patch_embed=Patch(); m.conv1=CNN(64,16,10)
    m.bn1=m.relu=m.maxpool=m.layer1=torch.nn.Identity()
    m.layer2=CNN(128,8,20); m.layer3=CNN(256,4,30); m.layer4=CNN(512,2,40)
    m.MS2=MS(128,8,200); m.MS3=MS(256,4,300); m.MS4=MS(512,2,400)
    m.swin_layers=torch.nn.ModuleList([Swin() for _ in range(4)])
    if lite: m.decoder=Lite()
    else: m.decode4=Decode(); m.decode3=Decode(); m.decode2=Decode(); m.decode0=torch.nn.Identity()
    m.eval()
    tile=torch.arange(32*32,dtype=torch.float32).reshape(1,1,32,32)
    with torch.inference_mode():
        expected=m.forward_features(tile); before=m(tile)
    got=_capture(m,tile)
    with torch.inference_mode(): after=m(tile)
    assert type(before)==type(after) and torch.equal(before,after)
    for i,sample in enumerate(got['stages']):
        assert np.array_equal(sample['magnitude'],expected[i].abs().mean(1)[0].numpy())
        assert sample['feature_shape']==list(expected[i].shape)
        assert not torch.is_tensor(sample['magnitude'])
    assert np.array_equal(got['deep']['channels'][0],expected[3][0,0].numpy())
    assert not torch.equal(expected[0],m.layer1(m.conv1(tile)))
    assert all(not mod._forward_pre_hooks for mod in m.modules())


def test_metrics_page_coverage_all_columns_rows_summaries(tmp_path):
    from test_report_metrics_table import fixture_rows, build
    import report_metrics_table as t
    rows=fixture_rows(9)
    rows['case0']['dice']='NaN'
    rows['case1']['extra_long_provenance']='Long source statement '*8
    rows['case0']['long_module_reference']='encoder.stages.very_long_synthetic_module_name_repeat_'*3
    rows['case0']['source_note']='Synthetic only; UNKNOWN / pending stays UNKNOWN / pending'
    source=build(rows)
    t.export_table(source,tmp_path)
    assert not (tmp_path/'ppt').exists()
    import csv
    exported=list(csv.DictReader((tmp_path/'metrics_table.csv').open(encoding='utf-8-sig')))
    assert len(exported)==len(source['records'])
    assert set(exported[0])==set(source['columns'])


@pytest.mark.parametrize('identity',['nnUNetTrainerUPerNetTopK10EarlyStopping','h2former'])
@pytest.mark.parametrize('missing',['stage','main_png','unexpected_ppt'])
def test_new_output_atomic_publication(tmp_path,monkeypatch,identity,missing):
    from test_report_metrics_table import write_inputs,wire_no_forward
    import generate_nnunet_result_report as o
    import report_visuals as v
    args,metadata=write_inputs(tmp_path,identity); wire_no_forward(monkeypatch,metadata)
    # Remove a produced artifact immediately before the real completeness guard.
    real=v.validate_visual_outputs
    def check(output,*a):
        if missing=='stage': (output/'encoder_stages_heatmap.png').unlink()
        elif missing=='main_png': (output/'summary.png').unlink()
        else: (output/'ppt').mkdir()
        # Main-image completeness is also checked after this point by this probe.
        if not (output/'encoder_stages_heatmap.png').is_file(): raise ValueError('report output incomplete: stage')
        return real(output,*a)
    monkeypatch.setattr(v,'validate_visual_outputs',check)
    with pytest.raises(SystemExit): o.main(args)
    assert not (tmp_path/'output').exists() and not list(tmp_path.glob('.output-*'))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_stage_sample_finite_extreme_reduction(dtype):
    from report_visuals import stage_sample, normalize_intensity, metadata_only
    values = [1e38, 2e38, 2.5e38, 3e38] if dtype == torch.float32 else [1e308, 1.2e308, 1.5e308, 1.7e308]
    x = torch.tensor(values, dtype=dtype).reshape(1, 1, 2, 2).repeat(1, 2, 1, 1).requires_grad_()
    sample = stage_sample(x, 0, "synthetic")
    assert sample["feature_nonfinite_count"] == 0
    assert np.isfinite(sample["magnitude"]).all()
    np.testing.assert_allclose(sample["magnitude"], x.detach()[0, 0].numpy(), rtol=1e-14)
    display, meta = normalize_intensity(sample["magnitude"])
    assert meta["status"] == "finite"
    assert np.all(np.diff(display.ravel()) > 0)
    assert sample["magnitude"].ndim == 2 and not torch.is_tensor(sample["magnitude"])
    assert "magnitude" not in metadata_only(sample)
    json.dumps(metadata_only(sample))


@pytest.mark.parametrize("channels", [1, 3])
def test_stage_sample_normal_constant_invalid(channels):
    from report_visuals import stage_sample, normalize_intensity
    x = torch.arange(-6, 6, dtype=torch.float64).reshape(1, 1, 3, 4).repeat(1, channels, 1, 1)
    sample = stage_sample(x, 1, "synthetic")
    np.testing.assert_allclose(sample["magnitude"], x.abs().mean(1)[0].numpy(), rtol=1e-14, atol=1e-14)
    for constant in (0., -3.):
        s = stage_sample(torch.full_like(x, constant), 1, "synthetic")
        display, meta = normalize_intensity(s["magnitude"])
        assert meta["status"] == "constant" and not display.any()
    x[0, 0, 0, 0] = float("nan"); x[0, 0, 0, 1] = float("inf")
    s = stage_sample(x, 1, "synthetic")
    assert s["feature_nonfinite_count"] == 2
    assert s["display"]["nonfinite_count"] == 2


@pytest.mark.parametrize("name,width,scales", [
    ("h2former", None, None), ("h2former_lite_upernet", 64, (1,2,4)),
    ("h2former_lite_upernet_w128_ppm1236",128,(1,2,3,6))])
def test_hybrid_encoder_page_node_geometry(tmp_path, monkeypatch, name, width, scales):
    import report_visuals as v
    from standalone_h2former_report import _architecture
    from matplotlib.patches import FancyBboxPatch
    captures = []
    def capture(fig, output, **kwargs):
        if Path(output).name != "architecture_detail.png": return
        fig.set_dpi(v.DPI); fig.canvas.draw(); renderer = fig.canvas.get_renderer()
        ax = next(a for a in fig.axes if a.get_visible())
        nodes = {p.get_gid(): p for p in ax.patches if isinstance(p, FancyBboxPatch)}
        assert len(nodes) == 24
        for key, patch in nodes.items():
            text = next((t for t in ax.texts if t.get_gid() == key), None)
            # Old implementation has no text gid; center identifies its owner.
            if text is None:
                center = (patch.get_x()+patch.get_width()/2, patch.get_y()+patch.get_height()/2)
                text = next(t for t in ax.texts if t.get_position() == center)
            b = patch.get_window_extent(renderer); t = text.get_window_extent(renderer)
            assert b.x0+2 <= t.x0 and t.x1 <= b.x1-2 and b.y0+2 <= t.y0 and t.y1 <= b.y1-2, (key, b.bounds, t.bounds)
        for stage in range(1,5):
            a=nodes[f"CNN{stage}"].get_window_extent(renderer)
            b=nodes[f"MS{stage}"].get_window_extent(renderer)
            assert a.x1+2 <= b.x0
        edges={p.get_gid() for p in ax.patches if not isinstance(p,FancyBboxPatch)}
        assert len(edges)==27
        # Every real curved/straight arrow is checked in display coordinates.
        # Shrink text bbox 1px for glyph/antialias tolerance; endpoints stay
        # outside text, including diagonals and routed inter-stage connectors.
        for arrow in (p for p in ax.patches if not isinstance(p,FancyBboxPatch)):
            path=arrow.get_transform().transform_path(arrow.get_path())
            for text in (t for t in ax.texts if t.get_gid() in nodes):
                b=text.get_window_extent(renderer)
                from matplotlib.transforms import Bbox
                interior=Bbox.from_extents(b.x0+1,b.y0+1,b.x1-1,b.y1-1)
                from matplotlib.path import Path as MplPath
                # to_polygons resolves CURVE/CLOSEPOLY; the dummy CLOSEPOLY
                # vertex at (0,0) must not become a fictitious diagonal wire.
                assert all(not MplPath(poly).intersects_bbox(interior,filled=False)
                           for poly in path.to_polygons(closed_only=False)), (arrow.get_gid(),text.get_gid())
        assert all(f"CNN{i}->add{i}" in edges and f"MS{i}->add{i}" in edges for i in range(1,5))
        captures.append(True)
    monkeypatch.setattr(v,"save_figure",capture)
    model=SimpleNamespace(image_size=512)
    if width: model.decoder=SimpleNamespace(in_channels=(64,128,256,512),pool_scales=scales,
        classifier=SimpleNamespace(in_channels=width,out_channels=2))
    _architecture(model,name,tmp_path/"architecture_detail.png",detail=True)
    assert captures


@pytest.mark.parametrize("layer", ["native", "intermediate_native"])
def test_native_main_case_row_separation(tmp_path, monkeypatch, layer):
    import report_visuals as v
    from generate_nnunet_result_report import _native_channel_figure
    ids=["\u4e2d\u6587\u5408\u6210\u957f\u75c5\u4f8b_"+"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"*4,"next_case","empty_case"]
    native=dict(channels=np.arange(24,dtype=np.float32).reshape(3,2,4),channel_ids=[0,2,4],
        feature_shape=[1,6,2,4],stage=3,module="encoder.stages.3",original_slice=12,window_index=4)
    items=[dict(cid=cid,group="High Dice",**{layer:native if i<2 else None}) for i,cid in enumerate(ids)]
    seen=[]
    def capture(fig, output, **kwargs):
        fig.set_dpi(v.DPI); fig.canvas.draw(); renderer=fig.canvas.get_renderer()
        assert kwargs["coverage"]["cases"] == (ids if Path(output).parent.name != "ppt" else [next(i["cid"] for i in items if i["cid"] in kwargs["coverage"]["cases"])])
        if Path(output).parent.name == "ppt": return
        headers=[t for ax in fig.axes for t in ax.texts if (t.get_gid() or "").startswith("case-header:")]
        assert len(headers)==3
        boxes=[t.get_window_extent(renderer) for t in headers]
        for a,b in zip(boxes,boxes[1:]): assert b.y1+2 <= a.y0
        for header in headers:
            b=header.get_window_extent(renderer)
            owner=header.axes.get_window_extent(renderer)
            assert owner.x0 <= b.x0 and b.x1 <= owner.x1 and owner.y0 <= b.y0 and b.y1 <= owner.y1
            for ax in fig.axes:
                if ax is header.axes: continue
                for text in [ax.title,ax.yaxis.label,*ax.texts]:
                    if text.get_text():
                        t=text.get_window_extent(renderer)
                        assert min(b.x1,t.x1)-max(b.x0,t.x0)<=1 or min(b.y1,t.y1)-max(b.y0,t.y0)<=1
                if ax.images:
                    t=ax.get_window_extent(renderer)
                    assert min(b.x1,t.x1)-max(b.x0,t.x0)<=1 or min(b.y1,t.y1)-max(b.y0,t.y0)<=1
        assert all(not ax.yaxis.label.get_text() for ax in fig.axes)
        seen.append(True)
    monkeypatch.setattr(v,"save_figure",capture)
    _native_channel_figure(items,tmp_path/"channels.png",layer=layer)
    assert seen


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_stage_sample_dtype_limits_and_mixed_channels(dtype):
    from report_visuals import stage_sample
    maximum=torch.finfo(dtype).max
    x=torch.tensor([maximum,maximum/2,-maximum/4,0],dtype=dtype).reshape(1,1,2,2).repeat(1,3,1,1)
    sample=stage_sample(x,0,"synthetic")
    assert np.isfinite(sample["magnitude"]).all()
    np.testing.assert_allclose(sample["magnitude"],x[0,0].double().abs().numpy(),rtol=1e-14)
    x=torch.tensor([[-1.,2.,-3.,4.],[4.,-3.,2.,-1.],[2.,-4.,1.,-3.]],dtype=dtype).reshape(1,3,2,2)
    sample=stage_sample(x,0,"synthetic")
    np.testing.assert_allclose(sample["magnitude"],x.double().abs().mean(1)[0].numpy(),rtol=1e-14,atol=1e-14)


@pytest.mark.parametrize("name,width,scales", [
    ("h2former", None, None), ("h2former_lite_upernet", 64, (1,2,4)),
    ("h2former_lite_upernet_w128_ppm1236",128,(1,2,3,6))])
@pytest.mark.parametrize("detail", [False, True])
def test_architecture_all_panels_containment(tmp_path, monkeypatch, name, width, scales, detail):
    """Measure every owner node on the final main and production page canvases."""
    import report_visuals as v
    from standalone_h2former_report import _architecture
    from matplotlib.patches import FancyBboxPatch
    from itertools import combinations
    original = v.save_figure
    captures = []
    failures = []
    expected = []
    def capture(fig, output, **kwargs):
        fig.set_dpi(v.DPI)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        panels = []
        for ax in fig.axes:
            if not ax.get_visible():
                continue
            nodes = {p.get_gid(): p for p in ax.patches if isinstance(p, FancyBboxPatch)}
            labels = {t.get_gid(): t for t in ax.texts if t.get_gid() is not None}
            edges = {p.get_gid() for p in ax.patches if not isinstance(p, FancyBboxPatch)}
            assert nodes and edges and set(labels) == set(nodes)
            assert all(a in nodes and b in nodes for a,b in (e.split("->") for e in edges))
            panels.append((set(nodes), edges, {k:t.get_text() for k,t in labels.items()}))
            for key, patch in nodes.items():
                b = patch.get_window_extent(renderer)
                t = labels[key].get_window_extent(renderer)
                margins = (t.x0-b.x0, b.x1-t.x1, t.y0-b.y0, b.y1-t.y1)
                if min(margins) < 2:
                    failures.append((Path(output).name, key, margins))
                assert labels[key].get_fontsize() >= (14 if Path(output).parent.name == 'ppt' else 12)
            from matplotlib.transforms import Bbox
            from matplotlib.path import Path as MplPath
            for arrow in (p for p in ax.patches if not isinstance(p, FancyBboxPatch)):
                path = arrow.get_transform().transform_path(arrow.get_path())
                for key,text in labels.items():
                    b = text.get_window_extent(renderer)
                    interior = Bbox.from_extents(b.x0+1,b.y0+1,b.x1-1,b.y1-1)
                    assert all(not MplPath(poly).intersects_bbox(interior,filled=False)
                               for poly in path.to_polygons(closed_only=False)), (Path(output).name,arrow.get_gid(),key)
            for a,b in combinations(nodes.values(),2):
                aa,bb = a.get_window_extent(renderer),b.get_window_extent(renderer)
                assert min(aa.x1,bb.x1)-max(aa.x0,bb.x0) <= 0 or min(aa.y1,bb.y1)-max(aa.y0,bb.y0) <= 0
            for artist in [*nodes.values(), *ax.texts, ax._left_title]:
                b = artist.get_window_extent(renderer)
                assert b.x0 >= -1 and b.y0 >= -1 and b.x1 <= fig.bbox.x1+1 and b.y1 <= fig.bbox.y1+1
        if Path(output).parent.name != 'ppt':
            assert len(panels) == (4 if detail else 3)
            expected.extend(panels)
        else:
            assert len(panels) == 1
            assert panels[0] == expected[len(captures)-1]
            assert next(a for a in fig.axes if a.get_visible())._left_title.get_fontsize() >= 22
        captures.append(panels)
        original(fig, output, **kwargs)
    monkeypatch.setattr(v, "save_figure", capture)
    model = SimpleNamespace(image_size=512)
    if width:
        model.decoder = SimpleNamespace(in_channels=(64,128,256,512),pool_scales=scales,
            classifier=SimpleNamespace(in_channels=width,out_channels=2))
    family = "architecture_detail" if detail else "architecture_overview"
    _architecture(model, name, tmp_path/(family+".png"), detail=detail)
    assert len(captures) == 1
    counts = [(24,27)]
    if width and detail:
        counts.append((3+2*len(scales), 3*len(scales)+2))
    counts.append((14,16) if width else (8,7))
    if not width and detail:
        counts.append((5,4))
    counts.append((6,4))
    assert [(len(nodes),len(edges)) for nodes,edges,_ in expected] == counts
    assert not failures, failures
