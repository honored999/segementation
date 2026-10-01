import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import SimpleITK as sitk

SCRIPT = Path(__file__).resolve().parents[1] / 'convert_isles2022_to_nnunet.py'
spec = importlib.util.spec_from_file_location('isles_conversion', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def image(path, array=None, **geometry):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = sitk.GetImageFromArray(np.zeros((2, 3, 4), dtype=np.float32)
                                if array is None else array)
    for field, value in geometry.items():
        getattr(img, 'Set' + field)(value)
    sitk.WriteImage(img, str(path))


@pytest.fixture
def dataset(tmp_path):
    source = tmp_path / 'source'
    name = 'sub-strokecase0001_ses-0001'
    dwi = source / 'sub-strokecase0001/ses-0001/dwi' / (name + '_dwi.nii.gz')
    mask = source / 'derivatives/sub-strokecase0001/ses-0001' / (name + '_msk.nii.gz')
    image(dwi)
    image(mask, np.ones((2, 3, 4), dtype=np.uint8))
    return source, tmp_path / 'raw/Dataset508_ISLES2022DWI', dwi, mask


def test_copy_and_sources_unchanged(dataset):
    source, output, dwi, mask = dataset
    original = [p.read_bytes() for p in (dwi, mask)]
    assert m.convert(source, output, 1)['status'] == 'complete'
    assert (output / 'imagesTr/ISLES_0001_0000.nii.gz').read_bytes() == original[0]
    assert (output / 'labelsTr/ISLES_0001.nii.gz').read_bytes() == original[1]
    assert [p.read_bytes() for p in (dwi, mask)] == original
    metadata = json.loads((output / 'dataset.json').read_text())
    assert metadata == {'channel_names': {'0': 'DWI'},
                        'labels': {'background': 0, 'lesion': 1},
                        'numTraining': 1, 'file_ending': '.nii.gz',
                        'overwrite_image_reader_writer': 'SimpleITKIO'}
    manifest = json.loads((output / 'conversion_manifest.json').read_text())
    for kind, p in (('dwi', dwi), ('label', mask)):
        row = manifest['cases'][0]
        assert row[f'source_{kind}_sha256'] == row[f'output_{kind}_sha256'] == m.sha256(p)
    assert not list(output.rglob('*splits*'))


def test_check_cli_no_writes(dataset):
    source, output, _, _ = dataset
    before = sorted(str(p.relative_to(source.parent)) for p in source.parent.rglob('*'))
    result = subprocess.run([sys.executable, str(SCRIPT), '--source-root', str(source),
                             '--output-dir', str(output), '--expected-cases', '1', '--check'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['status'] == 'checked'
    assert before == sorted(str(p.relative_to(source.parent)) for p in source.parent.rglob('*'))


@pytest.mark.parametrize('problem', ['count', 'missing_mask', 'orphan_mask', 'ambiguous'])
def test_pairing_errors(dataset, problem):
    source, output, dwi, mask = dataset
    if problem == 'missing_mask':
        mask.unlink()
    elif problem == 'orphan_mask':
        dwi.unlink()
    elif problem == 'ambiguous':
        image(source / 'unexpected' / dwi.name)
    with pytest.raises(m.ConversionError):
        m.convert(source, output, 2 if problem == 'count' else 1)
    assert not output.parent.exists()


@pytest.mark.parametrize('field,value', [
    ('Spacing', (2, 1, 1)), ('Origin', (1, 0, 0)),
    ('Direction', (-1, 0, 0, 0, 1, 0, 0, 0, 1)), ('size', None)])
def test_geometry(dataset, field, value):
    source, output, _, mask = dataset
    if field == 'size':
        image(mask, np.zeros((3, 3, 4), dtype=np.uint8))
    else:
        image(mask, **{field: value})
    with pytest.raises(m.ConversionError, match='Geometry mismatch'):
        m.convert(source, output, 1)
    assert not output.exists()


@pytest.mark.parametrize('which,value', [('mask', 2), ('mask', .5), ('mask', np.nan),
                                        ('dwi', np.inf), ('dwi', np.nan)])
def test_invalid_pixels(dataset, which, value):
    source, output, dwi, mask = dataset
    image(mask if which == 'mask' else dwi, np.full((2, 3, 4), value, dtype=np.float32))
    with pytest.raises(m.ConversionError):
        m.convert(source, output, 1)
    assert not output.exists()


@pytest.mark.parametrize('array', [np.zeros((3, 4), dtype=np.float32),
                                  np.zeros((2, 2, 3, 4), dtype=np.float32)])
def test_not_3d_scalar(dataset, array):
    source, output, dwi, _ = dataset
    image(dwi, array)
    with pytest.raises(m.ConversionError, match='3D scalar'):
        m.convert(source, output, 1)


@pytest.mark.parametrize('conflict', ['existing', 'sibling', 'inside', 'ancestor'])
def test_output_boundaries(dataset, conflict):
    source, output, _, _ = dataset
    if conflict == 'existing':
        output.mkdir(parents=True)
    elif conflict == 'sibling':
        (output.parent / 'Dataset508_Other').mkdir(parents=True)
    elif conflict == 'inside':
        output = source / 'output'
    else:
        output = source.parent
    with pytest.raises(m.ConversionError):
        m.convert(source, output, 1, check=True)


def test_escape_link(dataset, tmp_path):
    source, output, dwi, _ = dataset
    outside = tmp_path / 'outside.nii.gz'
    outside.write_bytes(dwi.read_bytes())
    dwi.unlink()
    try:
        dwi.symlink_to(outside)
    except OSError:
        dwi.write_bytes(outside.read_bytes())
        outside_dir = tmp_path / 'outside_dir'
        outside_dir.mkdir()
        junction = source / 'escape'
        result = subprocess.run(['cmd', '/c', 'mklink', '/J', str(junction),
                                 str(outside_dir)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    with pytest.raises(m.ConversionError, match='escapes root'):
        m.convert(source, output, 1)
    assert not output.exists()


def test_copy_failure_retained(dataset, monkeypatch, capsys):
    source, output, _, _ = dataset
    def fail(*args):
        raise OSError('injected copy failure')
    monkeypatch.setattr(m, 'copy_verified', fail)
    with pytest.raises(OSError, match='injected'):
        m.convert(source, output, 1)
    assert output.exists()
    assert not (output / 'dataset.json').exists()
    assert 'INCOMPLETE' in capsys.readouterr().err
    with pytest.raises(m.ConversionError, match='already exists'):
        m.convert(source, output, 1)


def test_corruption_detected(dataset, monkeypatch):
    _, output, dwi, _ = dataset
    output.mkdir(parents=True)
    actual_hash = m.sha256(dwi)
    original = m.sha256
    monkeypatch.setattr(m, 'sha256', lambda p: 'corrupt' if p.parent == output else original(p))
    with pytest.raises(m.ConversionError, match='integrity'):
        m.copy_verified(dwi, output / 'copy.nii.gz', actual_hash)


def test_late_bad_case_no_output(dataset):
    source, output, _, _ = dataset
    name = 'sub-strokecase0002_ses-0001'
    image(source / 'sub-strokecase0002/ses-0001/dwi' / (name + '_dwi.nii.gz'))
    image(source / 'derivatives/sub-strokecase0002/ses-0001' / (name + '_msk.nii.gz'),
          np.full((2, 3, 4), 2, dtype=np.uint8))
    with pytest.raises(m.ConversionError):
        m.convert(source, output, 2)
    assert not output.parent.exists()


def test_resolved_output_overlap(dataset):
    source, _, _, _ = dataset
    alias = source.parent / 'alias'
    result = subprocess.run(['cmd', '/c', 'mklink', '/J', str(alias), str(source)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    with pytest.raises(m.ConversionError, match='overlap'):
        m.convert(source, alias / 'output', 1, check=True)


def test_cli_failure_nonzero(dataset):
    source, output, _, mask = dataset
    image(mask, np.full((2, 3, 4), 2, dtype=np.uint8))
    result = subprocess.run([sys.executable, str(SCRIPT), '--source-root', str(source),
                             '--output-dir', str(output), '--expected-cases', '1'],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert 'FAILED' in result.stderr
    assert not output.exists()


@pytest.mark.parametrize('check', [True, False])
@pytest.mark.parametrize('name', [
    'Dataset501_WRONG', 'Dataset508', 'Dataset508_', 'Dataset508_Other',
    'dataset508_ISLES2022DWI', 'Dataset508_ISLES2022DWI_extra'])
def test_reject_wrong_output_name(dataset, check, name):
    source, output, dwi, mask = dataset
    output = output.parent / name
    before = [p.read_bytes() for p in (dwi, mask)]
    assert m.main(['--source-root', str(source), '--output-dir', str(output),
                   '--expected-cases', '1'] + (['--check'] if check else [])) == 1
    assert not output.exists()
    assert not output.parent.exists()
    assert [p.read_bytes() for p in (dwi, mask)] == before


@pytest.mark.parametrize('check', [True, False])
@pytest.mark.parametrize('alias', ['same_case_hardlink', 'cross_dwi_hardlink',
                                  'cross_mask_hardlink', 'cross_role_hardlink',
                                  'same_case_symlink'])
def test_reject_source_file_alias(dataset, check, alias):
    source, output, dwi, mask = dataset
    expected = 1
    if alias.startswith('cross'):
        expected = 2
        name = 'sub-strokecase0002_ses-0001'
        dwi2 = source / 'sub-strokecase0002/ses-0001/dwi' / (name + '_dwi.nii.gz')
        mask2 = source / 'derivatives/sub-strokecase0002/ses-0001' / (name + '_msk.nii.gz')
        image(dwi2)
        image(mask2, np.ones((2, 3, 4), dtype=np.uint8))
        original, linked = {'cross_dwi_hardlink': (dwi, dwi2),
                            'cross_mask_hardlink': (mask, mask2),
                            'cross_role_hardlink': (dwi, mask2)}[alias]
    else:
        original, linked = dwi, mask
    linked.unlink()
    if alias == 'same_case_symlink':
        try:
            linked.symlink_to(original)
        except OSError as exc:
            if getattr(exc, 'winerror', None) != 1314:
                raise
            pytest.skip('Windows file symlink privilege unavailable (WinError 1314); '
                        'hard-link identity coverage remains mandatory')
    else:
        os.link(original, linked)
    assert os.path.samefile(original, linked)
    files = sorted(source.rglob('*.nii.gz'))
    before = [p.read_bytes() for p in files]
    assert m.main(['--source-root', str(source), '--output-dir', str(output),
                   '--expected-cases', str(expected)] + (['--check'] if check else [])) == 1
    assert not output.exists()
    assert not output.parent.exists()
    assert [p.read_bytes() for p in files] == before


@pytest.mark.parametrize('check', [True, False])
def test_identical_bytes_independent_files_accepted(dataset, check):
    source, output, dwi, mask = dataset
    mask.write_bytes(dwi.read_bytes())
    assert not os.path.samefile(dwi, mask)
    before = [p.read_bytes() for p in (dwi, mask)]
    assert m.convert(source, output, 1, check=check)['status'] == (
        'checked' if check else 'complete')
    assert [p.read_bytes() for p in (dwi, mask)] == before
    if check:
        assert not output.parent.exists()
    else:
        assert (output / 'imagesTr/ISLES_0001_0000.nii.gz').read_bytes() == before[0]
        assert (output / 'labelsTr/ISLES_0001.nii.gz').read_bytes() == before[1]
