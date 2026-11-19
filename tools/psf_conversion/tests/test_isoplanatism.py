import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isoplanatism_core as c


def test_boundary_union():
    assert c.boundary_stats(torch.ones(8,8),1).edge_energy_fraction == pytest.approx(28/64)
    a=torch.zeros(8,8); a[0,0]=1
    assert c.boundary_stats(a,1).edge_energy_fraction == 1


def test_roi_intersection():
    w=c.roi_window(16,5,0,20)
    assert w['coverage_fraction']==0
    assert w['requested_x0']==26
    assert c.roi_sum(torch.ones(16,16),w)==0


@pytest.mark.parametrize('dy,dx',[(3,0),(-3,0),(0,3),(0,-3),(.6,-1.3)])
def test_shift(dy,dx):
    y,x=torch.meshgrid(torch.arange(48),torch.arange(48),indexing='ij')
    ref=torch.exp(-((y-21.)**2+(x-19.)**2)/12)+.3*torch.exp(-((y-29.)**2+(x-30.)**2)/6)
    mov,_=c.sample_shifted(ref,-dy,-dx)
    s=c.estimate_shift(ref,mov)
    assert s.valid
    assert s.dy_px==pytest.approx(dy,abs=.12)
    assert s.dx_px==pytest.approx(dx,abs=.12)
    aligned,mask=c.sample_shifted(mov,s.dy_px,s.dx_px)
    assert c.compare_shapes(ref,aligned,mask).shape_error < .06


def test_undefined():
    z=torch.zeros(16,16)
    assert not c.estimate_shift(z,z).valid
    r=c.compare_shapes(torch.ones_like(z),torch.ones_like(z),torch.ones_like(z,dtype=torch.bool))
    assert r.pearson is None


def test_physical_change_is_numerically_valid():
    s=c.ShiftEstimate(0,0,1,1,True,None,False)
    r=c.ShapeComparison(True,None,100,100,1,1,1,-.5,None)
    assert c.classify_validity(r,r,s,c.Thresholds())[0]=='ok'


import json
import validate_isoplanatism as cli


def config_file(tmp_path, **updates):
    data=dict(n=64,dx_nm=400,distance_um=80,wavelength_nm=450,camera_pixel_um=3.45,
              patch_camera=1,patch_meta=8,grayscale=True)
    data.update(updates)
    p=tmp_path/'config.json'; p.write_text(json.dumps(data))
    return p


def test_source_schema(tmp_path):
    cfg=c.load_source_config(config_file(tmp_path))
    assert cfg.padding==64
    with pytest.raises(c.ValidationError):
        c.load_source_config(config_file(tmp_path,patch_meta=9))
    with pytest.raises(c.ValidationError):
        c.load_source_config(config_file(tmp_path,args={'n':32}))


def test_artifact_validation(tmp_path):
    p=tmp_path/'kernel_pos.npz'
    np.savez(p,width_nm=np.ones((64,64)),wavelength_nm=450)
    with pytest.raises(c.ValidationError): c.load_artifact(p)
    np.savez(p,width_nm=np.empty((0,0)),psf_raw=np.zeros((1,64,64)),wavelength_nm=450,active_gray=False)
    a=c.load_artifact(p); assert a.skipped
    c.cross_check_config_artifact(c.load_source_config(config_file(tmp_path)),a)


def test_low_overlap():
    ref=torch.arange(256,dtype=torch.float64).reshape(16,16)
    aligned,mask=c.sample_shifted(ref,14,0)
    assert not c.compare_shapes(ref,aligned,mask).valid


def test_output_guard(tmp_path):
    with pytest.raises(c.ValidationError): cli.prepare_output(Path(__file__).parent/'bad')
    with pytest.raises(c.ValidationError): cli.prepare_output(tmp_path)
    source=tmp_path/'source'; source.mkdir()
    with pytest.raises(c.ValidationError): cli.prepare_output(source/'child',[source])


def test_real_asm_selfcheck(tmp_path):
    # Persistent simulations require a user destination. This is disposable pytest evidence.
    out=tmp_path/'ideal'
    manifest=cli.main(['--self-check','--out',str(out),'--angles','-0.5','0','0.5'])
    assert manifest['status']=='COMPLETED'
    rows=json.loads((out/'metrics.json').read_text())
    assert len(rows)==12
    assert all(r['padding_stability']['status']=='consistent' for r in rows)
    assert all(r['isoplanatism_acceptance']=='NOT_SPECIFIED' for r in rows)
    assert len(list(out.glob('*.npz')))==12
    assert sorted(p.name for p in out.glob('*.png'))==sorted(
        name for r in rows for name in (r['png_file'],r['full_plane_png_file']))
    with np.load(next(out.glob('*.npz'))) as data:
        assert data['valid_mask'].dtype==bool
        assert data['extended_field'].ndim==2


def test_truncation_stress():
    g=c.Geometry(64,400e-9,450e-9,80e-6)
    field=cli.ideal_field(g)
    engine=c.PropagationEngine(g)
    tilted=c.apply_carrier(field,g,12,0)
    full,crop,_=engine.extended(tilted,128)
    ratio=c.energy(c.intensity(crop),g.dx_m)/c.energy(c.intensity(full),g.dx_m)
    status,_=c.classify_truncation(c.boundary_stats(c.intensity(crop)),ratio,'NOT_CHECKED',c.Thresholds())
    print('ASM truncation stress:', status, 'output/extended=',ratio)
    assert status=='truncated'


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_postprocessing():
    g=c.Geometry(16,400e-9,450e-9,80e-6)
    field=cli.ideal_field(g).cuda()
    field=c.apply_carrier(field,g,.5,0)
    _,crop,layer=c.PropagationEngine(g,'cuda').extended(field,16)
    image=c.intensity(crop)
    assert c.estimate_shift(image,image).valid
    aligned,mask=c.sample_shifted(image,0,0)
    assert c.compare_shapes(image,aligned,mask).valid
    assert c.boundary_stats(image).edge_energy_fraction<=1
    assert c.sampling_diagnostics(field,layer,g,.5,0).valid
    print('CUDA propagation/carrier/CPU diagnostics: passed')


def test_missing_inputs_and_output_preserved(tmp_path):
    out=tmp_path/'absent'
    with pytest.raises(SystemExit): cli.main(['--out',str(out)])
    assert not out.exists()
    with pytest.raises(c.ValidationError):
        c.load_source_config(config_file(tmp_path,padding=0))


def test_zero_branch_manifest(tmp_path):
    config=config_file(tmp_path)
    artifact=tmp_path/'zero.npz'
    np.savez(artifact,width_nm=np.empty((0,0)),wavelength_nm=450,psf_raw=np.zeros((1,64,64)),active_gray=False)
    out=tmp_path/'zero-output'
    manifest=cli.main(['--artifact',str(artifact),'--config',str(config),'--out',str(out)])
    assert manifest['skipped_artifacts'][0]['reason'].startswith('empty width')
    assert json.loads((out/'metrics.json').read_text())==[]
    assert not list(out.glob('*.npz'))



def test_padding_equal_energy_different_shapes():
    a=np.zeros((8,8)); a[2:5,2:5]=1
    b=np.zeros((8,8)); b[2:5,4:7]=1
    def record(padding,psf):
        return dict(padding=padding,psf=psf,npz_file=f'p{padding}.npz',
                    shape_error_measured=0.,energy_output_au=float(psf.sum()),
                    shift_measured_dy_px=0.,shift_measured_dx_px=0.)
    result=c.padding_stability(record(8,a),record(16,b),c.Thresholds())
    assert result['status']=='inconsistent'
    assert result['checks']['direct_psf_shape_error'] > .01


def controlled_scan(monkeypatch,tmp_path,paddings,angles=(0.,),label='fixture'):
    g=c.Geometry(16,400e-9,450e-9,8e-6)
    original=c.PropagationEngine
    class ControlledEngine(original):
        def extended(self,field,padding):
            y,x=torch.meshgrid(torch.arange(16),torch.arange(16),indexing='ij')
            psf=torch.exp(-((y-7.5)**2+(x-7.5)**2)/(7. if padding==32 else 3.))
            psf=psf/psf.sum()
            crop=torch.sqrt(psf).to(torch.complex64)
            full=torch.nn.functional.pad(crop,(padding,)*4)
            self.transfer_function(16+2*padding)
            return full,crop,self._layer(0,16+2*padding)
        def direct(self,field,padding):
            return self.extended(field,padding)[1]
    monkeypatch.setattr(c,'PropagationEngine',ControlledEngine)
    tmp_path.mkdir(exist_ok=True)
    return cli.scan(cli.ideal_field(g),g,paddings,['x'],angles,7,'cpu',tmp_path,label)


def test_padding_all_candidates_and_order(monkeypatch,tmp_path):
    results=[]
    for index,paddings in enumerate(([16,32,48],[48,16,32])):
        out=tmp_path/str(index)
        rows=controlled_scan(monkeypatch,out,paddings)
        assert all(row['padding_stability']['status']=='inconsistent' for row in rows)
        summary=rows[0]['padding_stability']
        assert summary['baseline_padding']==16
        pairs=summary['comparisons']
        assert {(p['baseline_padding'],p['candidate_padding']) for p in pairs}=={(16,32),(16,48)}
        assert {p['candidate_padding']:p['status'] for p in pairs}=={32:'inconsistent',48:'consistent'}
        for pair in pairs:
            with np.load(out/pair['baseline_npz_file']) as base, np.load(out/pair['candidate_npz_file']) as candidate:
                a=base['psf'].astype(float).ravel(); b=candidate['psf'].astype(float).ravel()
                expected=np.linalg.norm(a/np.linalg.norm(a)-b/np.linalg.norm(b))
                assert pair['checks']['direct_psf_shape_error']==pytest.approx(expected)
        results.append({p['candidate_padding']:p['status'] for p in pairs})
    assert results[0]==results[1]


def test_angle_artifact_mapping(monkeypatch,tmp_path):
    rows=controlled_scan(monkeypatch,tmp_path,[16],angles=(.1,.10000001))
    cli.save_rows(tmp_path,rows)
    rows=json.loads((tmp_path/'metrics.json').read_text())
    assert len(rows)==3
    assert len(list(tmp_path.glob('*.npz')))==len(rows)
    assert sorted(p.name for p in tmp_path.glob('*.png'))==sorted(
        name for r in rows for name in (r['png_file'],r['full_plane_png_file']))
    assert len({r['sample_id'] for r in rows})==len(rows)
    assert {r['angle_deg'] for r in rows}=={0.,.1,.10000001}
    for row in rows:
        assert row['padding_stability']['status']=='NOT_CHECKED'
        assert (tmp_path/row['png_file']).is_file()
        with np.load(tmp_path/row['npz_file']) as data:
            assert data['angle_deg'].item()==row['angle_deg']
            assert data['sample_id'].item()==row['sample_id']
            assert data['psf'].shape==(16,16)
            assert float(data['psf'].sum())==pytest.approx(row['energy_output_au']/(400e-9)**2)



@pytest.mark.parametrize('invalid',[np.zeros((8,8)),np.full((8,8),np.nan)])
def test_padding_undefined_not_consistent(invalid):
    base=dict(padding=8,psf=np.ones((8,8)),npz_file='base.npz',energy_output_au=64.,
              shift_measured_dy_px=0.,shift_measured_dx_px=0.)
    candidate=dict(base,padding=16,psf=invalid,npz_file='candidate.npz')
    assert c.padding_stability(base,candidate,c.Thresholds())['status']=='indeterminate'


def test_unexpected_artifact_collision_preserves_existing(monkeypatch,tmp_path):
    rows=controlled_scan(monkeypatch,tmp_path,[16])
    path=tmp_path/rows[0]['npz_file']
    before=path.read_bytes()
    with pytest.raises(c.ValidationError,match='collision'):
        controlled_scan(monkeypatch,tmp_path,[16])
    assert path.read_bytes()==before


def test_selfcheck_requires_fixture_baseline(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(['--self-check','--paddings','128','192','--out',str(tmp_path/'unused')])
    assert not (tmp_path/'unused').exists()
