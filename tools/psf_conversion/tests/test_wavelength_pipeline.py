"""Wavelength contract tests; tiny synthetic data, no medical images."""
import json,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
import torch
TOOL=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(TOOL))
import isoplanatism_core as c
def run(*args):
 return subprocess.run([sys.executable,*map(str,args)],capture_output=True,text=True,encoding='utf-8',errors='replace')
@pytest.fixture(scope='module')
def fits(tmp_path_factory):
 root=tmp_path_factory.mktemp('wavelength_pipeline')
 out={}
 for wave in (450,532):
  target=root/str(wave)
  args=[TOOL/'fit_psf_gray.py','--demo','--limit','1','--n','32','--patch-camera','1',
        '--distance-um','80','--steps','2','--device','cpu','--out',target]
  if wave==532:args+=['--wavelength-nm','532','--fft-size','128']
  result=run(*args)
  assert result.returncode==0,result.stdout+result.stderr
  out[wave]=target
 return out
@pytest.mark.parametrize('wave',[450,532])
def test_saved_field_reproduces_at_actual_wavelength(fits,wave):
 cfg=c.load_source_config(fits[wave]/'config.json')
 assert cfg.wavelength_nm==wave
 assert cfg.padding==(48 if wave==532 else 32)
 assert cfg.raw['fft_size']==(128 if wave==532 else 96)
 art=c.load_artifact(fits[wave]/'kernel_000_pos.npz')
 c.cross_check_config_artifact(cfg,art)
 field=c.build_transmission(art.width_nm,art.wavelength_nm)
 pred=c.intensity(c.PropagationEngine(c.Geometry.from_config(cfg),'cpu').direct(field,cfg.padding)).numpy()
 with np.load(art.path) as z:
  np.testing.assert_allclose(pred,z['psf_raw'][0],rtol=3e-5,atol=3e-5)
 assert art.wavelength_nm==wave
def test_wrong_wavelength_produces_different_response(fits):
 cfg=c.load_source_config(fits[532]/'config.json')
 art=c.load_artifact(fits[532]/'kernel_000_pos.npz')
 right=c.build_transmission(art.width_nm,532)
 wrong=c.build_transmission(art.width_nm,450)
 assert float(torch.linalg.norm(right-wrong))>1
 with pytest.raises(c.ValidationError,match='wavelength mismatch'):
  c.cross_check_config_artifact(c.load_source_config(fits[450]/'config.json'),art)
def test_bad_fft_config_rejected(fits,tmp_path):
 cfg=json.loads((fits[532]/'config.json').read_text())
 cfg['fft_size']+=2
 p=tmp_path/'bad.json';p.write_text(json.dumps(cfg))
 with pytest.raises(c.ValidationError,match='fft_size conflicts'):
  c.load_source_config(p)
def test_unsupported_wave_rejected(tmp_path):
 np.savez(tmp_path/'bad.npz',width_nm=np.full((32,32),200),wavelength_nm=520)
 with pytest.raises(c.ValidationError,match='unsupported wavelength'):
  c.load_artifact(tmp_path/'bad.npz')
def test_532_preview_and_validation_cli(fits,tmp_path):
 preview=tmp_path/'preview'
 r=run(TOOL/'render_psf_preview.py','--results',fits[532],'--out',preview)
 assert r.returncode==0,r.stdout+r.stderr
 index=json.loads((preview/'index.json').read_text())
 assert index and all(s['wavelength_nm']==532 for row in index for s in row.get('sources',[]))
 out=tmp_path/'field'
 r=run(TOOL/'validate_isoplanatism.py','--artifact',fits[532]/'kernel_000_pos.npz',
       '--artifact',fits[532]/'kernel_000_neg.npz','--config',fits[532]/'config.json',
       '--axes','x','--angles','0','.01','--paddings','48','--device','cpu','--out',out)
 assert r.returncode==0,r.stdout+r.stderr
 manifest=json.loads((out/'manifest.json').read_text())
 assert manifest['status']=='COMPLETED'
 assert manifest['geometry']['wavelength_m']==pytest.approx(532e-9)
def test_source_lock_active():
 assert c.verify_locked_sources(TOOL)

def test_532_window_scan_uses_artifact_wavelength(fits,tmp_path):
 config=tmp_path/'scan_config.json'
 config.write_text(json.dumps(dict(fft_sizes=[96,128,160],corner_theta_deg=[.01,-.01],
   corner_source='Synthetic contract check only',roi_size_px=8,camera_pixel_um=3.45)))
 out=tmp_path/'windows'
 r=run(TOOL/'window_convergence.py','--artifact',fits[532]/'kernel_000_pos.npz',
   '--artifact',fits[532]/'kernel_000_neg.npz','--config',fits[532]/'config.json',
   '--run-config',config,'--device','cpu','--out',out)
 assert r.returncode==0,r.stdout+r.stderr
 doc=json.loads((out/'window_convergence.json').read_text())
 assert doc['status']=='COMPLETED' and doc['geometry']['wavelength_m']==pytest.approx(532e-9)
 row=next(x for x in doc['rows'] if x['fft_size']==128 and x['point']=='center' and x['sign']=='pos')
 with np.load(out/row['npz_file']) as z, np.load(fits[532]/'kernel_000_pos.npz') as fit:
  np.testing.assert_allclose(z['roi_intensity'],fit['psf_raw'][0,12:20,12:20],rtol=3e-5,atol=3e-5)
