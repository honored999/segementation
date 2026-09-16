"""Synthetic contracts for the Foreground50 full-resolution detail refiner."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from dynamic_network_architectures.architectures.unet import PlainConvUNet


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
if str(EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTENSION_ROOT))


def _implementation():
    try:
        detail = importlib.import_module("detail_refinement")
        trainer_module = importlib.import_module(
            "nnUNetTrainerForeground50DetailRefine"
        )
    except ModuleNotFoundError as exc:
        pytest.fail(f"DetailRefine implementation is missing: {exc}")
    return detail, trainer_module.nnUNetTrainerForeground50DetailRefine


def _network(deep_supervision: bool) -> PlainConvUNet:
    model = PlainConvUNet(
        input_channels=1,
        n_stages=3,
        features_per_stage=(8, 16, 24),
        conv_op=torch.nn.Conv2d,
        kernel_sizes=((3, 3), (3, 3), (3, 3)),
        strides=((1, 1), (2, 2), (2, 2)),
        n_conv_per_stage=(1, 1, 1),
        num_classes=2,
        n_conv_per_stage_decoder=(1, 1),
        conv_bias=True,
        norm_op=torch.nn.InstanceNorm2d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=deep_supervision,
    )
    model.apply(model.initialize)
    return model


def _configuration(deep_supervision: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        network_arch_class_name=(
            "dynamic_network_architectures.architectures.unet.PlainConvUNet"
        ),
        network_arch_init_kwargs={
            "n_stages": 3,
            "features_per_stage": (8, 16, 24),
            "conv_op": "torch.nn.Conv2d",
            "kernel_sizes": ((3, 3), (3, 3), (3, 3)),
            "strides": ((1, 1), (2, 2), (2, 2)),
            "n_conv_per_stage": (1, 1, 1),
            "n_conv_per_stage_decoder": (1, 1),
            "conv_bias": True,
            "norm_op": "torch.nn.InstanceNorm2d",
            "norm_op_kwargs": {"eps": 1e-5, "affine": True},
            "dropout_op": None,
            "dropout_op_kwargs": None,
            "nonlin": "torch.nn.LeakyReLU",
            "nonlin_kwargs": {"inplace": True},
        },
        network_arch_init_kwargs_req_import=(
            "conv_op",
            "norm_op",
            "dropout_op",
            "nonlin",
        ),
    )


def test_refinement_changes_only_highest_resolution_output_and_starts_as_identity() -> None:
    detail, _ = _implementation()
    torch.manual_seed(17)
    baseline = _network(deep_supervision=True).eval()
    refined = _network(deep_supervision=True).eval()
    refined.load_state_dict(baseline.state_dict())
    original_lower_head = refined.decoder.seg_layers[0]
    original_high_head = refined.decoder.seg_layers[-1]

    detail.attach_detail_refinement(refined)

    assert refined.decoder.seg_layers[0] is original_lower_head
    assert refined.decoder.seg_layers[-1].segmentation_head is original_high_head
    assert refined.decoder.seg_layers[-1].detail_refiner.input_channels == 8
    projection = refined.decoder.seg_layers[-1].detail_refiner.output_projection
    assert torch.count_nonzero(projection.weight) == 0
    assert projection.bias is not None
    assert torch.count_nonzero(projection.bias) == 0

    x = torch.randn(2, 1, 32, 40)
    with torch.no_grad():
        baseline_outputs = baseline(x)
        refined_outputs = refined(x)
    assert [tuple(t.shape) for t in refined_outputs] == [
        (2, 2, 32, 40),
        (2, 2, 16, 20),
    ]
    assert len(refined_outputs) == len(baseline_outputs)
    for actual, expected in zip(refined_outputs, baseline_outputs):
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)


def test_trainer_builder_preserves_paired_backbone_initialization_and_single_output() -> None:
    detail, trainer = _implementation()
    del detail
    configuration = _configuration()

    torch.manual_seed(1234)
    baseline = _network(deep_supervision=False)
    torch.manual_seed(1234)
    refined = trainer.build_network_architecture(
        SimpleNamespace(), configuration, 1, 2, False
    )

    refined_state = refined.state_dict()
    for key, value in baseline.state_dict().items():
        if key == "decoder.seg_layers.1.weight":
            refined_key = "decoder.seg_layers.1.segmentation_head.weight"
        elif key == "decoder.seg_layers.1.bias":
            refined_key = "decoder.seg_layers.1.segmentation_head.bias"
        else:
            refined_key = key
        torch.testing.assert_close(refined_state[refined_key], value)

    x = torch.randn(1, 1, 32, 32)
    assert refined(x).shape == (1, 2, 32, 32)
    assert refined.decoder.deep_supervision is False
    projection = refined.decoder.seg_layers[-1].detail_refiner.output_projection
    assert torch.count_nonzero(projection.weight) == 0


def test_gradient_reaches_early_refiner_only_after_projection_update() -> None:
    detail, _ = _implementation()
    model = _network(deep_supervision=False)
    detail.attach_detail_refinement(model)
    refiner = model.decoder.seg_layers[-1].detail_refiner
    x = torch.randn(2, 1, 32, 32)

    model(x).square().mean().backward()
    assert torch.isfinite(refiner.output_projection.weight.grad).all()
    assert torch.count_nonzero(refiner.output_projection.weight.grad) > 0
    assert refiner.input_projection.weight.grad is not None
    assert torch.count_nonzero(refiner.input_projection.weight.grad) == 0

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        refiner.output_projection.weight.add_(0.01)
    model(x).square().mean().backward()
    assert torch.isfinite(refiner.input_projection.weight.grad).all()
    assert torch.count_nonzero(refiner.input_projection.weight.grad) > 0


def test_state_dict_round_trip_is_strict_and_baseline_weights_are_rejected() -> None:
    detail, _ = _implementation()
    source = _network(deep_supervision=True)
    target = _network(deep_supervision=True)
    detail.attach_detail_refinement(source)
    detail.attach_detail_refinement(target)
    target.load_state_dict(source.state_dict(), strict=True)

    baseline = _network(deep_supervision=True)
    with pytest.raises(RuntimeError, match="Missing key|Unexpected key"):
        target.load_state_dict(baseline.state_dict(), strict=True)


def test_deep_supervision_toggle_keeps_refined_high_resolution_head() -> None:
    detail, _ = _implementation()
    model = _network(deep_supervision=True)
    detail.attach_detail_refinement(model)
    x = torch.randn(1, 1, 32, 32)
    assert isinstance(model(x), list)

    model.decoder.deep_supervision = False
    output = model(x)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 2, 32, 32)
    assert model.decoder.seg_layers[-1].detail_refiner.output_projection is not None


def test_attachment_rejects_unsupported_network_instead_of_guessing() -> None:
    detail, _ = _implementation()
    with pytest.raises(TypeError, match="decoder.seg_layers"):
        detail.attach_detail_refinement(torch.nn.Identity())


def test_complete_trainer_initialization_keeps_zero_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, trainer_type = _implementation()
    plans_path = (
        Path(__file__).resolve().parents[2]
        / "standalone_nnunet2d"
        / "reference"
        / "nnUNetPlans.json"
    )
    dataset_path = plans_path.with_name("dataset.json")
    plans = json.loads(plans_path.read_text(encoding="utf-8"))
    plans["continue_training"] = False
    dataset_json = json.loads(dataset_path.read_text(encoding="utf-8"))
    preprocessed_root = tmp_path / "preprocessed"
    preprocessed_data = (
        preprocessed_root / "Dataset501_StrokeLesion" / "nnUNetPlans_2d"
    )
    preprocessed_data.mkdir(parents=True)
    (preprocessed_data / "synthetic_case.npz").touch()
    monkeypatch.setenv("nnUNet_preprocessed", str(preprocessed_root))
    monkeypatch.setenv("nnUNet_results", str(tmp_path / "results"))
    monkeypatch.setenv("nnUNet_compile", "false")

    trainer = trainer_type(plans, "2d", 0, dataset_json, device=torch.device("cpu"))
    trainer.initialize()

    assert trainer.was_initialized
    assert trainer.oversample_foreground_percent == 0.50
    projection = trainer.network.decoder.seg_layers[-1].detail_refiner.output_projection
    assert torch.count_nonzero(projection.weight) == 0
    if projection.bias is not None:
        assert torch.count_nonzero(projection.bias) == 0

    checkpoint_path = tmp_path / "checkpoint_detail_refine.pth"
    trainer.current_epoch = 3
    trainer.save_checkpoint(str(checkpoint_path))
    with torch.no_grad():
        projection.weight.fill_(1)
    trainer.load_checkpoint(str(checkpoint_path))
    assert trainer.current_epoch == 4
    assert torch.count_nonzero(projection.weight) == 0

    baseline_checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    baseline_weights = {}
    refined_prefix = "decoder.seg_layers.6."
    for key, value in baseline_checkpoint["network_weights"].items():
        if key.startswith(refined_prefix + "detail_refiner."):
            continue
        if key.startswith(refined_prefix + "segmentation_head."):
            key = refined_prefix + key.removeprefix(
                refined_prefix + "segmentation_head."
            )
        baseline_weights[key] = value
    baseline_checkpoint["network_weights"] = baseline_weights
    baseline_checkpoint["trainer_name"] = "nnUNetTrainerForeground50"
    baseline_checkpoint_path = tmp_path / "checkpoint_baseline_architecture.pth"
    torch.save(baseline_checkpoint, baseline_checkpoint_path)
    with pytest.raises(RuntimeError, match="Missing key|Unexpected key"):
        trainer.load_checkpoint(str(baseline_checkpoint_path))


def test_official_external_resolver_discovers_detail_refine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _implementation()
    monkeypatch.setenv("nnUNet_extTrainer", str(EXTENSION_ROOT))
    from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name

    resolved = recursive_find_trainer_class_by_name(
        "nnUNetTrainerForeground50DetailRefine"
    )
    assert resolved.__name__ == "nnUNetTrainerForeground50DetailRefine"
    assert resolved.__module__ == "nnUNetTrainerForeground50DetailRefine"


def test_predictor_sliding_window_and_mirrored_tta_use_refined_network() -> None:
    detail, _ = _implementation()
    model = _network(deep_supervision=False).eval()
    detail.attach_detail_refinement(model)

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=False,
        use_mirroring=True,
        perform_everything_on_device=False,
        device=torch.device("cpu"),
        allow_tqdm=False,
    )
    predictor.network = model
    predictor.configuration_manager = SimpleNamespace(patch_size=(16, 16))
    predictor.label_manager = SimpleNamespace(num_segmentation_heads=2)
    predictor.allowed_mirroring_axes = (0, 1)

    image = torch.randn(1, 1, 20, 24)
    logits = predictor.predict_sliding_window_return_logits(image)
    assert logits.shape == (2, 1, 20, 24)
    assert torch.isfinite(logits).all()
