"""Server-deferred small synthetic geometry tests; never run locally."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import nibabel as nib
import pytest
import SimpleITK as sitk

from experiments.hdbet_nnunet import cli, native
from experiments.genseg3mm.targets import expand_keep_mask_3mm


def test_mask_values_preserves_signed_dtype_and_native_values():
    data = np.arange(-12, 12, dtype=np.int16).reshape(2, 3, 4)
    keep = np.zeros(data.shape, dtype=bool)
    keep[:, 1, :] = True
    result = native.mask_values(data, keep, np)
    assert result.dtype == data.dtype and np.array_equal(result[keep], data[keep])
    assert np.all(result[~keep] == 0) and data[0, 0, 0] == -12
    with pytest.raises(ValueError):
        native.mask_values(data.astype(float) * np.nan, keep, np)


def test_cached_native_roundtrip_does_not_read_gt_array(tmp_path, monkeypatch):
    image = tmp_path / "image.nii.gz"
    label = tmp_path / "label.nii.gz"
    data = np.arange(9 * 11 * 3, dtype=np.int16).reshape(9, 11, 3)
    affine = np.diag([-2., 2., 8., 1.])
    affine[:3, 3] = [18, -12, 5]
    x = nib.Nifti1Image(data, affine)
    x.header.set_xyzt_units("mm")
    nib.save(x, image)
    nib.save(nib.Nifti1Image(np.ones_like(data, dtype=np.uint8), affine), label)
    raw_sitk = sitk.ReadImage(str(image))
    brain = np.zeros((3, 11, 9), dtype=np.uint8)
    brain[1, 5, 4] = 1
    mask_image = sitk.GetImageFromArray(brain)
    mask_image.CopyInformation(raw_sitk)
    mask = tmp_path / "cached_mask.nii.gz"
    sitk.WriteImage(sitk.DICOMOrient(mask_image, "LAS"), str(mask), True)
    output = tmp_path / "HD-BET_outputs/trial"
    for p in (output / "nnUNet_raw" / cli.NAME / "imagesTr", output / "nnUNet_raw" / cli.NAME / "labelsTr"):
        p.mkdir(parents=True)
    info = {"output": str(output), "train": ["synthetic"], "cases": {"synthetic": {
        "image": str(image), "label": str(label), "cached_mask": str(mask),
        "raw_sha256": cli.sha(image), "label_sha256": cli.sha(label)}}, "files": {str(mask): cli.sha(mask)}}
    original_load = nib.load
    label_header = original_load(label)
    class HeaderOnlyLabel:
        shape = label_header.shape
        affine = label_header.affine
        @property
        def dataobj(self):
            raise AssertionError("GT array accessed without opt-in")
    def guarded_load(path, *a, **kw):
        return HeaderOnlyLabel() if Path(path) == label else original_load(path, *a, **kw)
    monkeypatch.setattr(nib, "load", guarded_load)
    seen = []
    monkeypatch.setattr(cli, "guard", lambda gpu, uid=None: seen.append((gpu, uid)))
    result, artifacts = native.process_case(SimpleNamespace(gpu=1, audit_label_coverage=False), info, "synthetic", "GPU-fixed")
    actual = np.asanyarray(original_load(output / "nnUNet_raw" / cli.NAME / "imagesTr" / image.name).dataobj)
    keep = expand_keep_mask_3mm(brain, spacing_xyz=raw_sitk.GetSpacing()).transpose(2, 1, 0)
    assert np.array_equal(actual[keep], data[keep]) and np.all(actual[~keep] == 0)
    assert np.array_equal(actual[:, :, 0], data[:, :, 0]) and np.array_equal(actual[:, :, 2], data[:, :, 2])
    assert result["empty_native_planes"] == [0, 2] and result["readback_verified"] is True
    assert result["coverage_audit"] == "not_requested" and result["qa_planes"] == 3
    assert len(list((output / "cases/synthetic/qa").glob("*.png"))) == 3
    assert cli.sha(output / "nnUNet_raw" / cli.NAME / "labelsTr" / label.name) == cli.sha(label)
    assert seen == [(1, "GPU-fixed")] and artifacts

@pytest.mark.parametrize("qcode,scode", [(0, 0), (0, 2), (1, 0), (1, 2)])
def test_save_native_preserves_unequal_transform_encodings(tmp_path, qcode, scode):
    data = np.arange(3 * 4 * 5, dtype=np.int16).reshape(3, 4, 5)
    angle = .17
    direction = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    qform = np.eye(4)
    qform[:3, :3] = direction @ np.diag([2, 2, 8])
    qform[:3, 3] = [11, -7, 5]
    sform = qform.copy()
    sform[:3, 3] += [1, 2, 3]
    image = nib.Nifti1Image(data, qform)
    image.header.set_xyzt_units("mm")
    image.set_qform(qform, qcode)
    image.set_sform(sform, scode)
    source_path, output = tmp_path / "source.nii.gz", tmp_path / "masked.nii.gz"
    nib.save(image, source_path)
    source = nib.load(source_path)
    masked = np.asanyarray(source.dataobj).copy()
    masked[0] = 0
    native.save_native_dwi(source, masked, output, nib, np)
    saved = nib.load(output)
    assert saved.header["qform_code"] == qcode and saved.header["sform_code"] == scode
    for field in ("quatern_b", "quatern_c", "quatern_d", "pixdim", "srow_x", "srow_y", "srow_z"):
        assert np.array_equal(saved.header[field], source.header[field])
