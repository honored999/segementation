"""Pure DWI/ADC fusion channel construction shared by online and offline paths."""

from __future__ import annotations

import numpy as np


def _masked_minmax(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    normalized = np.zeros(array.shape, dtype=np.float32)
    values = np.asarray(array, dtype=np.float32)[mask]
    if values.size == 0:
        return normalized
    minimum = float(values.min())
    span = float(values.max()) - minimum
    if span > 0.0:
        normalized[mask] = (values - minimum) / span
    return normalized


def build_dwi_adc_fusion_channel(
    dwi_array: np.ndarray,
    adc_array: np.ndarray,
) -> np.ndarray:
    """Return masked ``DWI_01 + (1 - ADC_01)`` as float32."""
    dwi = np.asarray(dwi_array)
    adc = np.asarray(adc_array)
    if dwi.shape != adc.shape:
        raise ValueError(f"DWI and ADC shapes must match: {dwi.shape} != {adc.shape}")
    valid_mask = (
        np.isfinite(dwi)
        & np.isfinite(adc)
        & (dwi != 0.0)
        & (adc != 0.0)
    )
    dwi_01 = _masked_minmax(dwi, valid_mask)
    adc_01 = _masked_minmax(adc, valid_mask)
    fusion = np.zeros(dwi.shape, dtype=np.float32)
    fusion[valid_mask] = dwi_01[valid_mask] + (1.0 - adc_01[valid_mask])
    return fusion
