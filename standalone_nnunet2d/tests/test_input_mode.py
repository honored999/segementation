from __future__ import annotations

import numpy as np
import pytest

from standalone_nnunet2d.data.input_mode import INPUT_SPECS, InputMode, input_spec
from standalone_nnunet2d.data.nifti_io import NiftiVolume
from standalone_nnunet2d.data.symmetry_alignment import bilateral_difference


IDENTITY_DIRECTION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


@pytest.mark.parametrize(
    ("mode", "physical_modalities", "effective_channels"),
    [
        (InputMode.DWI, ("DWI",), 1),
        (InputMode.DWI_ADC, ("DWI", "ADC"), 2),
        (InputMode.DWI_BILATERAL, ("DWI",), 2),
        (InputMode.DWI_ADC_BILATERAL, ("DWI", "ADC"), 4),
    ],
)
def test_input_modes_have_one_exact_input_spec(
    mode: InputMode,
    physical_modalities: tuple[str, ...],
    effective_channels: int,
) -> None:
    spec = input_spec(mode)

    assert INPUT_SPECS[mode] is spec
    assert spec.physical_modalities == physical_modalities
    assert spec.effective_input_channels == effective_channels
    assert spec.slice_context == 1


def test_input_spec_accepts_stable_string_values_and_rejects_unknown_values() -> None:
    assert input_spec("dwi_adc_bilateral") is INPUT_SPECS[InputMode.DWI_ADC_BILATERAL]
    with pytest.raises(ValueError, match="unsupported input mode"):
        input_spec("dwi_adc_signed")


def test_dwi_adc_bilateral_spec_declares_ordered_signed_recipes_and_dwi_alignment_reference() -> None:
    spec = input_spec(InputMode.DWI_ADC_BILATERAL)

    assert spec.physical_modalities == ("DWI", "ADC")
    assert spec.channel_recipes == (
        "DWI",
        "ADC",
        "DWI_LR_SIGNED_DIFF",
        "ADC_LR_SIGNED_DIFF",
    )
    assert spec.requires_alignment is True
    assert spec.alignment_reference_modality == "DWI"


def test_signed_difference_uses_anatomical_lr_axis_from_direction() -> None:
    # Physical X is the array's Y index for this supported axis-aligned direction.
    direction = (0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    image = np.zeros((1, 9, 9), dtype=np.float32)
    image[0, 2, 4] = 3.0
    volume = NiftiVolume(image, (1.0, 1.0, 4.0), (0.0, 0.0, 0.0), direction)

    difference = bilateral_difference(volume, mode="signed")

    assert difference[0, 2, 4] > 0.0
    assert difference[0, 6, 4] < 0.0
    np.testing.assert_allclose(difference[0, 2, 4], -difference[0, 6, 4])


def test_legacy_bilateral_builder_remains_absolute_and_nonnegative() -> None:
    from standalone_nnunet2d.data.symmetry_alignment import build_bilateral_asymmetry_channels

    image = np.zeros((1, 9, 9), dtype=np.float32)
    image[0, 2, 3] = 3.0
    volume = NiftiVolume(image, (1.0, 1.0, 4.0), (0.0, 0.0, 0.0), IDENTITY_DIRECTION)

    channels = build_bilateral_asymmetry_channels(volume)

    assert channels.shape == (2, 1, 9, 9)
    assert np.all(channels[1] >= 0.0)
    assert channels[1, 0, 2, 3] == channels[1, 0, 2, 5] == 3.0
