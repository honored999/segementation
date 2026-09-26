"""Explicit model contracts and construction for standalone 2D training/inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Mapping
from typing import Any

from torch import nn

from standalone_nnunet2d.config import load_model_config
from standalone_nnunet2d.models.h2former import H2Former
from standalone_nnunet2d.models.h2former_lite_upernet import H2FormerLiteUPerNet
from standalone_nnunet2d.models.plain_conv_unet import PlainConvUNet2D
from standalone_nnunet2d.models.plain_conv_unet_lite_upernet import PlainConvUNetLiteUPerNet


PLAIN_CONV_UNET = "plain_conv_unet"
H2FORMER = "h2former"
H2FORMER_LITE_UPERNET = "h2former_lite_upernet"
H2FORMER_LITE_UPERNET_W128 = "h2former_lite_upernet_w128"
H2FORMER_LITE_UPERNET_W128_PPM1236 = "h2former_lite_upernet_w128_ppm1236"
PPM1236_ARCHITECTURE = {"fpn_channels": 128, "ppm_scales": (1, 2, 3, 6), "ppm_out_channels": 128}
PLAIN_CONV_UNET_LITE_UPERNET = "plain_conv_unet_lite_upernet"
MODEL_NAMES = (
    PLAIN_CONV_UNET,
    H2FORMER,
    H2FORMER_LITE_UPERNET,
    H2FORMER_LITE_UPERNET_W128,
    H2FORMER_LITE_UPERNET_W128_PPM1236,
    PLAIN_CONV_UNET_LITE_UPERNET,
)
DEEP_SUPERVISION = "deep_supervision"
SINGLE_OUTPUT = "single_output"

_SINGLE_OUTPUT_ONLY_MODELS = frozenset(
    {H2FORMER, H2FORMER_LITE_UPERNET, H2FORMER_LITE_UPERNET_W128, H2FORMER_LITE_UPERNET_W128_PPM1236, PLAIN_CONV_UNET_LITE_UPERNET}
)
_PLAIN_CONV_UNET_MODELS = frozenset({PLAIN_CONV_UNET})


@dataclass(frozen=True)
class ModelContract:
    name: str
    in_channels: int
    num_classes: int
    image_size: int | None
    supervision_mode: str
    deep_supervision: bool
    loss_name: str

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.name == H2FORMER_LITE_UPERNET_W128_PPM1236:
            result.update(PPM1236_ARCHITECTURE)
        return result


_CONTRACTS = {
    PLAIN_CONV_UNET: ModelContract(
        name=PLAIN_CONV_UNET,
        in_channels=1,
        num_classes=2,
        image_size=None,
        supervision_mode=DEEP_SUPERVISION,
        deep_supervision=True,
        loss_name="DeepSupervisionLoss",
    ),
    H2FORMER: ModelContract(
        name=H2FORMER,
        in_channels=1,
        num_classes=2,
        image_size=512,
        supervision_mode=SINGLE_OUTPUT,
        deep_supervision=False,
        loss_name="DiceCrossEntropyLoss",
    ),
    H2FORMER_LITE_UPERNET: ModelContract(
        name=H2FORMER_LITE_UPERNET,
        in_channels=1,
        num_classes=2,
        image_size=512,
        supervision_mode=SINGLE_OUTPUT,
        deep_supervision=False,
        loss_name="DiceCrossEntropyLoss",
    ),
    H2FORMER_LITE_UPERNET_W128: ModelContract(
        name=H2FORMER_LITE_UPERNET_W128,
        in_channels=1,
        num_classes=2,
        image_size=512,
        supervision_mode=SINGLE_OUTPUT,
        deep_supervision=False,
        loss_name="DiceCrossEntropyLoss",
    ),
    H2FORMER_LITE_UPERNET_W128_PPM1236: ModelContract(
        name=H2FORMER_LITE_UPERNET_W128_PPM1236,
        in_channels=1, num_classes=2, image_size=512,
        supervision_mode=SINGLE_OUTPUT, deep_supervision=False,
        loss_name="DiceCrossEntropyLoss",
    ),
    PLAIN_CONV_UNET_LITE_UPERNET: ModelContract(
        name=PLAIN_CONV_UNET_LITE_UPERNET,
        in_channels=1,
        num_classes=2,
        image_size=None,
        supervision_mode=SINGLE_OUTPUT,
        deep_supervision=False,
        loss_name="DiceCrossEntropyLoss",
    ),
}


def get_model_contract(
    model_name: str = PLAIN_CONV_UNET,
    *,
    supervision_mode: str | None = None,
) -> ModelContract:
    """Return the explicit contract for one supported model name."""
    try:
        contract = _CONTRACTS[model_name]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"unsupported model_name {model_name!r}; choices are {MODEL_NAMES}"
        ) from error
    resolved_supervision_mode = (
        contract.supervision_mode if supervision_mode is None else supervision_mode
    )
    if model_name in _SINGLE_OUTPUT_ONLY_MODELS and resolved_supervision_mode != SINGLE_OUTPUT:
        raise ValueError(
            f"model {model_name!r} requires supervision_mode={SINGLE_OUTPUT!r}, "
            f"got {resolved_supervision_mode!r}"
        )
    if model_name in _PLAIN_CONV_UNET_MODELS and resolved_supervision_mode not in {
        DEEP_SUPERVISION,
        SINGLE_OUTPUT,
    }:
        raise ValueError(
            f"model {model_name!r} does not support supervision_mode={resolved_supervision_mode!r}"
        )
    return ModelContract(
        name=contract.name,
        in_channels=contract.in_channels,
        num_classes=contract.num_classes,
        image_size=contract.image_size,
        supervision_mode=resolved_supervision_mode,
        deep_supervision=resolved_supervision_mode == DEEP_SUPERVISION,
        loss_name=(
            "DeepSupervisionLoss"
            if resolved_supervision_mode == DEEP_SUPERVISION
            else "DiceCrossEntropyLoss"
        ),
    )


def build_model(
    model_name: str = PLAIN_CONV_UNET,
    *,
    supervision_mode: str | None = None,
    inference: bool = False,
) -> nn.Module:
    """Build a model from its stable explicit name.

    ``inference=True`` only disables PlainConvUNet's auxiliary outputs for the
    existing inference contract.  It does not change the training contract.
    H2Former always remains a single-output model.
    """
    contract = get_model_contract(model_name, supervision_mode=supervision_mode)
    if contract.name == PLAIN_CONV_UNET:
        return PlainConvUNet2D(
            load_model_config(),
            deep_supervision=False if inference else contract.deep_supervision,
        )
    if contract.name == PLAIN_CONV_UNET_LITE_UPERNET:
        return PlainConvUNetLiteUPerNet(load_model_config())
    if contract.name == H2FORMER:
        return H2Former(
            in_channels=contract.in_channels,
            num_classes=contract.num_classes,
            image_size=contract.image_size or 512,
        )
    if contract.name in {H2FORMER_LITE_UPERNET, H2FORMER_LITE_UPERNET_W128, H2FORMER_LITE_UPERNET_W128_PPM1236}:
        return H2FormerLiteUPerNet(
            in_channels=contract.in_channels,
            num_classes=contract.num_classes,
            image_size=contract.image_size or 512,
            fpn_channels=128 if contract.name in {H2FORMER_LITE_UPERNET_W128, H2FORMER_LITE_UPERNET_W128_PPM1236} else 64,
            pool_scales=(1, 2, 3, 6) if contract.name == H2FORMER_LITE_UPERNET_W128_PPM1236 else (1, 2, 4),
        )
    raise AssertionError(f"unhandled model contract: {contract.name}")


def _identity_from_mapping(value: Mapping[str, Any], *, source: str) -> tuple[str, str]:
    has_name = "name" in value
    has_mode = "supervision_mode" in value
    if has_name != has_mode:
        raise ValueError(f"{source} must contain both name and supervision_mode")
    if not has_name:
        raise ValueError(f"{source} must contain name and supervision_mode")
    contract = get_model_contract(str(value["name"]), supervision_mode=str(value["supervision_mode"]))
    for key, expected in contract.as_dict().items():
        if contract.name == H2FORMER_LITE_UPERNET_W128_PPM1236 and key in PPM1236_ARCHITECTURE:
            if key not in value:
                raise ValueError(f"{source} is missing required architecture field {key!r}")
            actual = value[key]
            if type(actual) is not type(expected) or (
                isinstance(expected, tuple) and any(type(item) is not int for item in actual)
            ):
                raise ValueError(f"{source} field {key!r} has invalid architecture type")
        if key in value and value[key] != expected:
            raise ValueError(f"{source} field {key!r} conflicts with model contract")
    return contract.name, contract.supervision_mode


def resolve_checkpoint_model_identity(metadata: Mapping[str, Any]) -> tuple[str, str]:
    """Resolve explicit checkpoint identity, retaining legacy PlainConv default."""
    top_keys = {"model_name", "supervision_mode"}
    present_top = top_keys.intersection(metadata)
    if present_top and present_top != top_keys:
        raise ValueError("checkpoint model identity must contain model_name and supervision_mode")
    top_identity = None
    if present_top == top_keys:
        architecture = {}
        if metadata["model_name"] == H2FORMER_LITE_UPERNET_W128_PPM1236:
            architecture = metadata.get("architecture", {})
            if not isinstance(architecture, Mapping) or set(architecture) != set(PPM1236_ARCHITECTURE):
                raise ValueError("checkpoint architecture must contain exactly the B architecture fields")
        top_identity = _identity_from_mapping(
            {**architecture, "name": metadata["model_name"], "supervision_mode": metadata["supervision_mode"]},
            source="checkpoint metadata",
        )

    nested_identities: list[tuple[str, str]] = []
    for config_key in ("config", "resolved_config"):
        config = metadata.get(config_key)
        if config is None:
            continue
        if not isinstance(config, Mapping):
            raise ValueError(f"checkpoint {config_key} must be a mapping")
        if "model" not in config:
            continue
        nested_model = config["model"]
        if not isinstance(nested_model, Mapping):
            raise ValueError(f"checkpoint {config_key}.model must be a mapping")
        nested_identities.append(
            _identity_from_mapping(nested_model, source=f"checkpoint {config_key}.model")
        )

    identities = [identity for identity in (top_identity, *nested_identities) if identity is not None]
    if identities and any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("checkpoint model identity conflicts between metadata and resolved_config")
    if identities:
        return identities[0]
    return PLAIN_CONV_UNET, DEEP_SUPERVISION


__all__ = [
    "DEEP_SUPERVISION",
    "H2FORMER",
    "H2FORMER_LITE_UPERNET",
    "H2FORMER_LITE_UPERNET_W128",
    "H2FORMER_LITE_UPERNET_W128_PPM1236",
    "PPM1236_ARCHITECTURE",
    "MODEL_NAMES",
    "ModelContract",
    "PLAIN_CONV_UNET",
    "PLAIN_CONV_UNET_LITE_UPERNET",
    "SINGLE_OUTPUT",
    "build_model",
    "get_model_contract",
    "resolve_checkpoint_model_identity",
]
