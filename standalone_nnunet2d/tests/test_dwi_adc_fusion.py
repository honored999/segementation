from __future__ import annotations

from pathlib import Path

import numpy as np

import standalone_nnunet2d.data.dataset as dataset_module
import standalone_nnunet2d.data.inference_preprocessing as inference_module
from standalone_nnunet2d.data.dataset import StrokeSliceDataset
from standalone_nnunet2d.data.input_mode import InputMode
from standalone_nnunet2d.data.nifti_io import NiftiVolume


IDENTITY_DIRECTION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def _volume(array: np.ndarray) -> NiftiVolume:
    return NiftiVolume(
        array.astype(np.float32),
        (1.0, 1.0, 4.0),
        (0.0, 0.0, 0.0),
        IDENTITY_DIRECTION,
    )


def test_fusion_dataset_and_inference_share_c3_preparation_without_alignment(monkeypatch) -> None:
    dwi = np.zeros((2, 5, 5), dtype=np.float32)
    adc = np.zeros((2, 5, 5), dtype=np.float32)
    dwi[:, 1:4, 1:4] = np.arange(1.0, 19.0, dtype=np.float32).reshape(2, 3, 3)
    adc[:, 1:4, 1:4] = np.arange(19.0, 1.0, -1.0, dtype=np.float32).reshape(2, 3, 3)
    label = np.zeros((2, 5, 5), dtype=np.float32)
    volumes = (_volume(dwi), _volume(adc))

    def forbidden(*args, **kwargs):
        raise AssertionError("fusion mode must not call symmetry or bilateral difference")

    monkeypatch.setattr(dataset_module, "estimate_quasi_symmetric_alignment", forbidden)
    monkeypatch.setattr(dataset_module, "bilateral_difference", forbidden)
    monkeypatch.setattr(dataset_module, "read_case_images", lambda root, case_id: volumes)
    monkeypatch.setattr(dataset_module, "read_nifti", lambda path: _volume(label))

    dataset = object.__new__(StrokeSliceDataset)
    dataset.raw_root = Path("unused")
    dataset.case_ids = ("case001",)
    dataset.target_spacing_xy = (1.0, 1.0)
    dataset.channel_specs = ((0, "DWI"), (1, "ADC"))
    dataset.input_mode = InputMode.DWI_ADC_FUSION
    training_input, training_label = dataset.load_case("case001")

    monkeypatch.setattr(
        inference_module, "resolve_channel_specs", lambda root: ((0, "DWI"), (1, "ADC"))
    )
    monkeypatch.setattr(inference_module, "read_case_images", lambda root, case_id: volumes)
    prepared = inference_module.prepare_dwi_adc_fusion_case(
        Path("unused"), "case001", target_spacing_xy=(1.0, 1.0)
    )

    assert training_input.shape == (3, 2, 5, 5)
    assert training_label.shape == (2, 5, 5)
    np.testing.assert_allclose(prepared.model_input, training_input)
    assert np.all(training_input[2, :, 0, :] == 0.0)
    assert np.all(training_input[2, :, :, 0] == 0.0)
