"""Stable input-mode contracts for the standalone 2D pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping


class InputMode(str, Enum):
    """Named physical-to-model input contracts."""

    DWI = "dwi"
    DWI_ADC = "dwi_adc"
    DWI_BILATERAL = "dwi_bilateral"
    DWI_ADC_BILATERAL = "dwi_adc_bilateral"


@dataclass(frozen=True)
class InputSpec:
    """The immutable channel contract associated with one input mode."""

    physical_modalities: tuple[str, ...]
    channel_recipes: tuple[str, ...]
    requires_alignment: bool
    alignment_reference_modality: str | None
    effective_input_channels: int
    slice_context: int = 1

    @property
    def physical_input_channels(self) -> int:
        return len(self.physical_modalities)


INPUT_SPECS: Mapping[InputMode, InputSpec] = MappingProxyType(
    {
        InputMode.DWI: InputSpec(
            physical_modalities=("DWI",),
            channel_recipes=("DWI",),
            requires_alignment=False,
            alignment_reference_modality=None,
            effective_input_channels=1,
        ),
        InputMode.DWI_ADC: InputSpec(
            physical_modalities=("DWI", "ADC"),
            channel_recipes=("DWI", "ADC"),
            requires_alignment=False,
            alignment_reference_modality=None,
            effective_input_channels=2,
        ),
        InputMode.DWI_BILATERAL: InputSpec(
            physical_modalities=("DWI",),
            channel_recipes=("DWI", "DWI_LR_ABS_DIFF"),
            requires_alignment=True,
            alignment_reference_modality="DWI",
            effective_input_channels=2,
        ),
        InputMode.DWI_ADC_BILATERAL: InputSpec(
            physical_modalities=("DWI", "ADC"),
            channel_recipes=(
                "DWI",
                "ADC",
                "DWI_LR_SIGNED_DIFF",
                "ADC_LR_SIGNED_DIFF",
            ),
            requires_alignment=True,
            alignment_reference_modality="DWI",
            effective_input_channels=4,
        ),
    }
)


def input_spec(mode: InputMode | str) -> InputSpec:
    """Resolve one supported mode from its enum or stable string value."""

    try:
        resolved_mode = mode if isinstance(mode, InputMode) else InputMode(mode)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unsupported input mode: {mode!r}") from error
    return INPUT_SPECS[resolved_mode]
