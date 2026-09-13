from __future__ import annotations

import gc

import pytest
import torch

from standalone_nnunet2d import formal_train
from standalone_nnunet2d.losses.compound import DiceCrossEntropyLoss
from standalone_nnunet2d.losses.deep_supervision import DeepSupervisionLoss
from standalone_nnunet2d.models.factory import (
    H2FORMER,
    PLAIN_CONV_UNET,
    DEEP_SUPERVISION,
    SINGLE_OUTPUT,
    build_model,
    get_model_contract,
)
from standalone_nnunet2d.models.h2former import H2Former
from standalone_nnunet2d.models.plain_conv_unet import PlainConvUNet2D
from standalone_nnunet2d.training.official_config import OfficialTrainerSchedule


def _parser_arguments() -> list[str]:
    return [
        "--raw-root",
        "synthetic-raw",
        "--output-root",
        "synthetic-output",
        "--plans",
        "synthetic-plans.json",
    ]


def test_parser_defaults_to_plain_conv_unet() -> None:
    arguments = formal_train.build_parser().parse_args(_parser_arguments())

    assert arguments.model == PLAIN_CONV_UNET


def test_parser_accepts_explicit_h2former() -> None:
    arguments = formal_train.build_parser().parse_args(_parser_arguments() + ["--model", H2FORMER])

    assert arguments.model == H2FORMER


def test_stage3_parser_and_contract_resolve_model_specific_supervision_defaults() -> None:
    plain_arguments = formal_train.build_parser().parse_args(_parser_arguments())
    h2_arguments = formal_train.build_parser().parse_args(
        _parser_arguments() + ["--model", H2FORMER]
    )

    assert plain_arguments.supervision_mode is None
    assert h2_arguments.supervision_mode is None
    assert get_model_contract(PLAIN_CONV_UNET, supervision_mode=plain_arguments.supervision_mode).supervision_mode == DEEP_SUPERVISION
    assert get_model_contract(H2FORMER, supervision_mode=h2_arguments.supervision_mode).supervision_mode == SINGLE_OUTPUT


def test_stage3_plain_matched_single_output_builds_tensor_and_dice_ce() -> None:
    contract = get_model_contract(PLAIN_CONV_UNET, supervision_mode=SINGLE_OUTPUT)
    model = build_model(PLAIN_CONV_UNET, supervision_mode=SINGLE_OUTPUT)
    loss, validation_loss = formal_train.build_training_losses(
        PLAIN_CONV_UNET, supervision_mode=SINGLE_OUTPUT
    )
    image = torch.zeros((1, 1, 256, 256))
    target = torch.zeros((1, 256, 256), dtype=torch.long)

    logits = model(image)
    value = loss(logits, target)
    value.backward()

    assert contract.deep_supervision is False
    assert contract.loss_name == "DiceCrossEntropyLoss"
    assert isinstance(loss, DiceCrossEntropyLoss)
    assert isinstance(validation_loss, DiceCrossEntropyLoss)
    assert logits.shape == (1, 2, 256, 256)
    assert torch.isfinite(value)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)


def test_stage3_h2_and_plain_matched_use_the_same_base_loss_type() -> None:
    plain_loss, _ = formal_train.build_training_losses(
        PLAIN_CONV_UNET, supervision_mode=SINGLE_OUTPUT
    )
    h2_loss, _ = formal_train.build_training_losses(
        H2FORMER, supervision_mode=SINGLE_OUTPUT
    )

    assert type(plain_loss) is type(h2_loss) is DiceCrossEntropyLoss


def test_stage3_h2_deep_supervision_is_rejected() -> None:
    with pytest.raises(ValueError, match="supervision_mode"):
        get_model_contract(H2FORMER, supervision_mode=DEEP_SUPERVISION)


def test_plain_factory_training_and_inference_supervision_modes() -> None:
    training_model = build_model(PLAIN_CONV_UNET)
    inference_model = build_model(PLAIN_CONV_UNET, inference=True)

    assert training_model.deep_supervision is True
    assert inference_model.deep_supervision is False


def test_training_losses_follow_explicit_model_contracts() -> None:
    plain_training_loss, plain_validation_loss = formal_train.build_training_losses(PLAIN_CONV_UNET)
    h2_training_loss, h2_validation_loss = formal_train.build_training_losses(H2FORMER)

    assert isinstance(plain_training_loss, DeepSupervisionLoss)
    assert isinstance(plain_training_loss.base_loss, DiceCrossEntropyLoss)
    assert isinstance(plain_validation_loss, DiceCrossEntropyLoss)
    assert len(plain_training_loss.weights) == 7
    assert plain_training_loss.weights == pytest.approx(
        (32 / 63, 16 / 63, 8 / 63, 4 / 63, 2 / 63, 1 / 63, 0)
    )
    assert isinstance(h2_training_loss, DiceCrossEntropyLoss)
    assert h2_training_loss is h2_validation_loss


def test_factory_contracts_are_explicit_and_reject_unsupported_combinations() -> None:
    plain = get_model_contract(PLAIN_CONV_UNET)
    h2 = get_model_contract(H2FORMER)

    assert plain.name == PLAIN_CONV_UNET
    assert plain.in_channels == 1
    assert plain.num_classes == 2
    assert plain.image_size is None
    assert plain.supervision_mode == DEEP_SUPERVISION
    assert h2.name == H2FORMER
    assert h2.in_channels == 1
    assert h2.num_classes == 2
    assert h2.image_size == 512
    assert h2.supervision_mode == SINGLE_OUTPUT

    with pytest.raises(ValueError, match="supervision_mode"):
        get_model_contract(H2FORMER, supervision_mode=DEEP_SUPERVISION)
    matched = get_model_contract(PLAIN_CONV_UNET, supervision_mode=SINGLE_OUTPUT)
    assert matched.supervision_mode == SINGLE_OUTPUT
    assert matched.deep_supervision is False
    assert matched.loss_name == "DiceCrossEntropyLoss"


def test_factory_builds_both_explicit_model_names() -> None:
    plain = build_model(PLAIN_CONV_UNET)
    assert isinstance(plain, PlainConvUNet2D)
    del plain
    gc.collect()

    h2 = build_model(H2FORMER)
    assert isinstance(h2, H2Former)
    del h2
    gc.collect()


def test_factory_h2former_can_be_built_for_explicit_inference() -> None:
    model = build_model(H2FORMER, inference=True)
    assert isinstance(model, H2Former)
    del model
    gc.collect()


def test_h2former_single_output_uses_dice_cross_entropy_with_finite_gradients() -> None:
    model = build_model(H2FORMER)
    image = torch.zeros((1, 1, 512, 512))
    target = torch.zeros((1, 512, 512), dtype=torch.long)

    logits = model(image)
    loss = DiceCrossEntropyLoss()(logits, target)
    loss.backward()

    assert logits.shape == (1, 2, 512, 512)
    assert torch.isfinite(loss)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)
    del loss, logits, target, image, model
    gc.collect()


def test_resolved_config_records_distinct_model_identity_and_plan_hash() -> None:
    schedule = OfficialTrainerSchedule(num_iterations_per_epoch=1, num_val_iterations_per_epoch=1)

    plain = formal_train.build_formal_config(
        fold=0,
        epochs=1,
        schedule=schedule,
        model_name=PLAIN_CONV_UNET,
    )
    h2 = formal_train.build_formal_config(
        fold=0,
        epochs=1,
        schedule=schedule,
        model_name=H2FORMER,
    )
    matched = formal_train.build_formal_config(
        fold=0,
        epochs=1,
        schedule=schedule,
        model_name=PLAIN_CONV_UNET,
        supervision_mode=SINGLE_OUTPUT,
    )

    assert plain["model"] == {
        "name": PLAIN_CONV_UNET,
        "in_channels": 1,
        "num_classes": 2,
        "image_size": None,
        "supervision_mode": DEEP_SUPERVISION,
        "deep_supervision": True,
        "loss_name": "DeepSupervisionLoss",
    }
    assert h2["model"] == {
        "name": H2FORMER,
        "in_channels": 1,
        "num_classes": 2,
        "image_size": 512,
        "supervision_mode": SINGLE_OUTPUT,
        "deep_supervision": False,
        "loss_name": "DiceCrossEntropyLoss",
    }
    assert matched["model"] == {
        "name": PLAIN_CONV_UNET,
        "in_channels": 1,
        "num_classes": 2,
        "image_size": None,
        "supervision_mode": SINGLE_OUTPUT,
        "deep_supervision": False,
        "loss_name": "DiceCrossEntropyLoss",
    }
    assert len({plain["plan_hash"], matched["plan_hash"], h2["plan_hash"]}) == 3
    invariant_keys = set(plain) - {"model", "plan_hash"}
    assert {key: plain[key] for key in invariant_keys} == {
        key: matched[key] for key in invariant_keys
    }
    assert {key: matched[key] for key in invariant_keys} == {
        key: h2[key] for key in invariant_keys
    }
