"""Real tiny PlainConvUNet aliasing and checkpoint contracts; CPU only."""

from copy import deepcopy
import json
import random
from types import SimpleNamespace

import numpy as np
from pathlib import Path
import sys

import pytest
import torch
import torch.nn as nn
from dynamic_network_architectures.architectures.unet import PlainConvUNet

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ckpd import (  # noqa: E402
    apply_ckpd_adapters,
    capture_rng_state,
    decompose_conv2d,
    restore_rng_state,
    hash_file,
    hash_identifier,
    merged_state_dict,
    module_fingerprint,
)


def _assert_rng_states_equal(first, second):
    assert first["python"] == second["python"]
    assert first["numpy"][0] == second["numpy"][0]
    assert np.array_equal(first["numpy"][1], second["numpy"][1])
    assert first["numpy"][2:] == second["numpy"][2:]
    assert torch.equal(first["torch_cpu"], second["torch_cpu"])
    if first["torch_cuda"] is None or second["torch_cuda"] is None:
        assert first["torch_cuda"] is second["torch_cuda"] is None
    else:
        assert len(first["torch_cuda"]) == len(second["torch_cuda"])
        assert all(torch.equal(a, b) for a, b in zip(first["torch_cuda"], second["torch_cuda"]))


def tiny_plain_conv_unet():
    return PlainConvUNet(
        input_channels=1,
        n_stages=2,
        features_per_stage=(4, 8),
        conv_op=nn.Conv2d,
        kernel_sizes=(3, 3),
        strides=(1, 2),
        n_conv_per_stage=(1, 1),
        num_classes=2,
        n_conv_per_stage_decoder=(1,),
        conv_bias=True,
        norm_op=nn.InstanceNorm2d,
        norm_op_kwargs={"affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=True,
    )


def test_alias_safe_encoder_freeze_decoder_gradients_and_merged_inference():
    torch.manual_seed(11)
    network = tiny_plain_conv_unet().eval()
    baseline = deepcopy(network).eval()
    assert network.decoder.encoder is network.encoder
    conv_aliases = dict(network.named_modules(remove_duplicate=False))
    candidates = [name for name, module in network.encoder.named_modules()
                  if isinstance(module, nn.Conv2d)]
    assert candidates
    selected = "encoder." + candidates[0]
    conv = dict(network.named_modules())[selected]
    assert any(module is conv and name.startswith("decoder.encoder.")
               for name, module in conv_aliases.items())
    d = conv.weight[0].numel()
    decomposition = decompose_conv2d(
        conv.weight.detach(), torch.eye(d), rank=1, damping_relative=0.01)
    apply_ckpd_adapters(network, {selected: decomposition})

    assert network.decoder.encoder is network.encoder
    assert dict(network.named_modules())[selected] is conv
    assert not conv.parametrizations.weight.original.requires_grad
    assert conv.parametrizations.weight[0].A.requires_grad
    assert conv.parametrizations.weight[0].B.requires_grad
    assert all(not p.requires_grad for p in network.encoder.parameters()
               if p is not conv.parametrizations.weight[0].A
               and p is not conv.parametrizations.weight[0].B)
    decoder_only = [p for n, p in network.named_parameters()
                    if n.startswith("decoder.") and "encoder" not in n]
    assert decoder_only and all(p.requires_grad for p in decoder_only)

    x = torch.randn(2, 1, 16, 16)
    expected = baseline(x)
    actual = network(x)
    assert isinstance(actual, (tuple, list))
    assert len(actual) == len(expected)
    for actual_head, expected_head in zip(actual, expected):
        assert torch.allclose(actual_head, expected_head, atol=2e-5, rtol=2e-5)

    network.train()
    network(x)[0].square().mean().backward()
    assert conv.parametrizations.weight[0].A.grad is not None
    assert conv.parametrizations.weight[0].B.grad is not None
    assert conv.parametrizations.weight.original.grad is None
    assert all(p.grad is None for p in network.encoder.parameters()
               if p is not conv.parametrizations.weight[0].A
               and p is not conv.parametrizations.weight[0].B)
    assert any(p.grad is not None for p in decoder_only)

    exported = merged_state_dict(network)
    plain_inference = tiny_plain_conv_unet().eval()
    plain_inference.load_state_dict(exported, strict=True)
    network.eval()
    with torch.no_grad():
        adapted = network(x)
        merged = plain_inference(x)
    for adapted_head, merged_head in zip(adapted, merged):
        assert torch.allclose(adapted_head, merged_head, atol=2e-5, rtol=2e-5)


def test_fresh_trainer_requires_calibration_and_source_binding_before_train_start():
    from nnUNetTrainerCKPDTopK10EarlyStopping import nnUNetTrainerCKPDTopK10EarlyStopping

    trainer_type = nnUNetTrainerCKPDTopK10EarlyStopping
    assert trainer_type.__mro__[1].__name__ == "nnUNetTrainerTopK10EarlyStopping"
    assert trainer_type.CKPD_DEFAULT_RANK == 4
    assert trainer_type.CKPD_DEFAULT_LAYER_COUNT == 4


def _artifact_fixture(tmp_path, network, monkeypatch, *, selected_case="source-train-0", plans_exists=True):
    from nnUNetTrainerCKPDTopK10EarlyStopping import nnUNetTrainerCKPDTopK10EarlyStopping as Trainer
    from ckpd import encoder_conv_layers
    import ckpd

    tmp_path.mkdir(parents=True, exist_ok=True)
    source_preprocessed = tmp_path / "Dataset508_ISLES2022DWI"
    target_preprocessed = tmp_path / "Dataset501_StrokeLesion"
    source_preprocessed.mkdir()
    target_preprocessed.mkdir()
    source_split = source_preprocessed / "splits_final.json"
    target_split = target_preprocessed / "splits_final.json"
    source_split.write_text(json.dumps([{"train": ["source-train-0", "source-train-1"],
                                         "val": ["source-val-0"]}]), encoding="utf-8")
    target_split.write_text(json.dumps([{"train": ["target-train-0"], "val": ["target-val-0"]}]), encoding="utf-8")
    plans = tmp_path / "nnUNetPlans.json"
    dataset_json = tmp_path / "dataset.json"
    checkpoint = tmp_path / "source-checkpoint.pth"
    if plans_exists:
        plans.write_text('{"plans_name":"nnUNetPlansFrom501","dataset_name":"Dataset508_ISLES2022DWI"}', encoding="utf-8")
    dataset_json.write_text('{"name":"Dataset508_ISLES2022DWI","channel_names":{"0":"MRI"}}', encoding="utf-8")
    checkpoint.write_bytes(b"tiny synthetic checkpoint identity")

    convs = encoder_conv_layers(network)
    assert len(convs) >= Trainer.CKPD_DEFAULT_LAYER_COUNT
    selected = list(convs)[:Trainer.CKPD_DEFAULT_LAYER_COUNT]
    factors = {}
    for name in selected:
        conv = convs[name]
        dimension = conv.weight.numel() // conv.out_channels
        factors[name] = decompose_conv2d(conv.weight.detach().cpu(), torch.eye(dimension), rank=4).__dict__
    artifact = {
        "schema_version": 1,
        "source_dataset": "Dataset508_ISLES2022DWI",
        "source_fold": 0,
        "source_trainer_name": "nnUNetTrainerTopK10EarlyStopping",
        "source_plans_name": "nnUNetPlansFrom501",
        "source_data_identifier": "nnUNetPlansFrom501_2d",
        "source_dataset_name": "Dataset508_ISLES2022DWI",
        "source_split_path": str(source_split),
        "source_split_sha256": hash_file(source_split),
        "source_train_case_hashes": sorted(hash_identifier(x) for x in ["source-train-0", "source-train-1"]),
        "source_val_case_hashes": [hash_identifier("source-val-0")],
        "source_encoder_semantic_signature": (
            ckpd.encoder_semantic_signature(network)
            if hasattr(ckpd, "encoder_semantic_signature") else "not-yet-implemented"
        ),
        "patch_size": [16, 16],
        "selected_case_hashes": [hash_identifier(selected_case)],
        "source_checkpoint_sha256": hash_file(checkpoint),
        "source_encoder_fingerprint": module_fingerprint(network.encoder),
        "configuration": "2d",
        "source_plans_path": str(plans),
        "source_plans_sha256": hash_file(plans) if plans.exists() else "0" * 64,
        "source_dataset_json_path": str(dataset_json),
        "source_dataset_json_sha256": hash_file(dataset_json),
        "encoder_conv_metadata": Trainer._encoder_conv_metadata(convs),
        "selected_layers": selected,
        "sample_count": 1,
        "seed": 20261012,
        "covariance": "per-case mean of sampled uncentered Conv2d input-patch second moments; case means averaged equally",
        "max_positions_per_case_per_layer": 256,
        "damping_relative": 0.01,
        "max_covariance_mib": 512,
        "asr_definition": "descending SVD(W C_reg): sigma[-rank] / sigma[-1]; lowest K scores selected (paper v2)",
        "rank": 4,
        "layer_count": 4,
        "factors": factors,
    }
    artifact_path = tmp_path / "calibration.pth"
    torch.save(artifact, artifact_path)
    monkeypatch.setenv("NNUNET_CKPD_CALIBRATION", str(artifact_path))
    monkeypatch.setenv("NNUNET_CKPD_SOURCE_CHECKPOINT", str(checkpoint))
    trainer = object.__new__(Trainer)
    trainer.network = network
    trainer.fold = 0
    trainer.configuration_name = "2d"
    trainer.preprocessed_dataset_folder_base = str(target_preprocessed)
    trainer.configuration_manager = SimpleNamespace(patch_size=(16, 16))
    return trainer, artifact_path


def test_calibration_manifest_accepts_only_hashed_dataset508_train_cases(tmp_path, monkeypatch):
    from nnUNetTrainerCKPDTopK10EarlyStopping import nnUNetTrainerCKPDTopK10EarlyStopping as Trainer

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    network = PlainConvUNet(
        input_channels=1, n_stages=3, features_per_stage=(8, 16, 32), conv_op=nn.Conv2d,
        kernel_sizes=(3, 3, 3), strides=(1, 2, 2), n_conv_per_stage=(1, 1, 2),
        num_classes=2, n_conv_per_stage_decoder=(1, 1), conv_bias=True,
        norm_op=nn.InstanceNorm2d, norm_op_kwargs={"affine": True},
        dropout_op=None, dropout_op_kwargs=None, nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True}, deep_supervision=True,
    )
    trainer, artifact_path = _artifact_fixture(tmp_path, network, monkeypatch)
    artifact, _, source_hash = trainer._load_and_validate_calibration()
    assert artifact["selected_case_hashes"] == [hash_identifier("source-train-0")]
    assert artifact["source_val_case_hashes"] == [hash_identifier("source-val-0")]
    assert len(source_hash) == 64
    assert Trainer.CKPD_DEFAULT_LAYER_COUNT == len(artifact["selected_layers"])

    artifact["factors"][artifact["selected_layers"][0]]["base_weight"] += 0.5
    torch.save(artifact, artifact_path)
    with pytest.raises(RuntimeError, match="do not reconstruct"):
        trainer._load_and_validate_calibration()


def test_calibration_rejects_validation_case_and_missing_plan_path(tmp_path, monkeypatch):
    from nnUNetTrainerCKPDTopK10EarlyStopping import nnUNetTrainerCKPDTopK10EarlyStopping as Trainer

    network = PlainConvUNet(
        input_channels=1, n_stages=3, features_per_stage=(8, 16, 32), conv_op=nn.Conv2d,
        kernel_sizes=(3, 3, 3), strides=(1, 2, 2), n_conv_per_stage=(1, 1, 2),
        num_classes=2, n_conv_per_stage_decoder=(1, 1), conv_bias=True,
        norm_op=nn.InstanceNorm2d, norm_op_kwargs={"affine": True},
        dropout_op=None, dropout_op_kwargs=None, nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True}, deep_supervision=True,
    )
    trainer, _ = _artifact_fixture(tmp_path, network, monkeypatch, selected_case="source-val-0")
    with pytest.raises(RuntimeError, match="not a nonempty subset"):
        trainer._load_and_validate_calibration()

    other, _ = _artifact_fixture(tmp_path / "missing-plan", network, monkeypatch, plans_exists=False)
    with pytest.raises(RuntimeError, match="plans or dataset.json differs"):
        other._load_and_validate_calibration()



def test_calibration_rejects_source_validation_overlap_with_target_split(tmp_path, monkeypatch):
    network = PlainConvUNet(
        input_channels=1, n_stages=3, features_per_stage=(8, 16, 32), conv_op=nn.Conv2d,
        kernel_sizes=(3, 3, 3), strides=(1, 2, 2), n_conv_per_stage=(1, 1, 2),
        num_classes=2, n_conv_per_stage_decoder=(1, 1), conv_bias=True,
        norm_op=nn.InstanceNorm2d, norm_op_kwargs={"affine": True},
        dropout_op=None, dropout_op_kwargs=None, nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True}, deep_supervision=True,
    )
    trainer, _ = _artifact_fixture(tmp_path, network, monkeypatch)
    target_split = Path(trainer.preprocessed_dataset_folder_base) / "splits_final.json"
    target_split.write_text(
        json.dumps([{"train": ["source-val-0"], "val": ["target-val-0"]}]), encoding="utf-8")
    with pytest.raises(RuntimeError, match="split identifiers overlap"):
        trainer._load_and_validate_calibration()


@pytest.mark.parametrize("mismatch", ["norm_eps", "activation", "patch_size"])
def test_calibration_rejects_encoder_semantics_and_patch_size_mismatch(tmp_path, monkeypatch, mismatch):
    network = PlainConvUNet(
        input_channels=1, n_stages=3, features_per_stage=(8, 16, 32), conv_op=nn.Conv2d,
        kernel_sizes=(3, 3, 3), strides=(1, 2, 2), n_conv_per_stage=(1, 1, 2),
        num_classes=2, n_conv_per_stage_decoder=(1, 1), conv_bias=True,
        norm_op=nn.InstanceNorm2d, norm_op_kwargs={"affine": True},
        dropout_op=None, dropout_op_kwargs=None, nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True}, deep_supervision=True,
    )
    trainer, _ = _artifact_fixture(tmp_path, network, monkeypatch)
    original_fingerprint = module_fingerprint(network.encoder)
    if mismatch == "norm_eps":
        next(module for module in network.encoder.modules()
             if isinstance(module, nn.InstanceNorm2d)).eps *= 10
        expected_error = "semantic signature"
    elif mismatch == "activation":
        next(module for module in network.encoder.modules()
             if isinstance(module, nn.LeakyReLU)).negative_slope += 0.2
        expected_error = "semantic signature"
    else:
        trainer.configuration_manager.patch_size = (32, 16)
        expected_error = "patch_size"
    assert module_fingerprint(network.encoder) == original_fingerprint
    with pytest.raises(RuntimeError, match=expected_error):
        trainer._load_and_validate_calibration()



def _standard_resume_metadata_fixture(best_ema=0.5, early_best=0.5):
    from nnUNetTrainerCKPDTopK10EarlyStopping import nnUNetTrainerCKPDTopK10EarlyStopping as Trainer

    trainer = object.__new__(Trainer)
    trainer.num_epochs = 10
    trainer.configuration_name = "2d"
    trainer.fold = 0
    trainer.grad_scaler = None
    checkpoint = {
        "trainer_name": Trainer.__name__,
        "current_epoch": 0,
        "logging": {},
        "_best_ema": best_ema,
        "init_args": {"configuration": "2d", "fold": 0},
        "inference_allowed_mirroring_axes": None,
        "early_stopping_state": {
            "best_monitored_ema_dice": early_best,
            "epochs_without_improvement": 0,
            "triggered": False,
        },
        "grad_scaler_state": None,
    }
    return trainer, checkpoint


@pytest.mark.parametrize("field", ["_best_ema", "best_monitored_ema_dice"])
@pytest.mark.parametrize("value", [None, 0, 0.5, np.float32(0.5), np.float64(0.5)])
def test_resume_metadata_accepts_finite_real_scalars_and_none(field, value):
    from nnUNetTrainerCKPDTopK10EarlyStopping import nnUNetTrainerCKPDTopK10EarlyStopping as Trainer

    trainer, checkpoint = _standard_resume_metadata_fixture()
    if field == "_best_ema":
        checkpoint["_best_ema"] = value
    else:
        checkpoint["early_stopping_state"]["best_monitored_ema_dice"] = value
    Trainer._validate_standard_resume_metadata(trainer, checkpoint)


@pytest.mark.parametrize("field", ["_best_ema", "best_monitored_ema_dice"])
@pytest.mark.parametrize("value", [
    True, np.bool_(True), np.nan, np.inf, -np.inf, "0.5", 0.5 + 0j,
    np.array(0.5), np.array([0.5]), torch.tensor(0.5),
])
def test_resume_metadata_rejects_nonfinite_or_nonreal_ema_values(field, value):
    from nnUNetTrainerCKPDTopK10EarlyStopping import nnUNetTrainerCKPDTopK10EarlyStopping as Trainer

    trainer, checkpoint = _standard_resume_metadata_fixture()
    if field == "_best_ema":
        checkpoint["_best_ema"] = value
    else:
        checkpoint["early_stopping_state"]["best_monitored_ema_dice"] = value
    with pytest.raises(RuntimeError, match="invalid"):
        Trainer._validate_standard_resume_metadata(trainer, checkpoint)


def test_checkpoint_resume_restores_next_optimizer_update_and_exports_plain_weights(tmp_path, monkeypatch):
    import json
    from nnunetv2.training.logging.nnunet_logger import MetaLogger
    from nnUNetTrainerCKPDTopK10EarlyStopping import nnUNetTrainerCKPDTopK10EarlyStopping as Trainer
    from ckpd import encoder_conv_layers

    def make_trainer(network, root):
        trainer = object.__new__(Trainer)
        trainer.network = network
        trainer.fold = 0
        trainer.configuration_name = "2d"
        trainer.preprocessed_dataset_folder_base = str(root / "Dataset501_StrokeLesion")
        trainer.plans_manager = type("Plans", (), {"plans": {"plans_name": "nnUNetPlans", "configuration": "2d"}})()
        trainer.was_initialized = True
        trainer.device = torch.device("cpu")
        trainer.is_ddp = False
        trainer._do_i_compile = lambda: False
        trainer.initial_lr = 0.01
        trainer.weight_decay = 3e-5
        trainer.num_epochs = 3
        trainer.current_epoch = 1
        trainer.local_rank = 0
        trainer.disable_checkpointing = False
        trainer.grad_scaler = None
        trainer.logger = MetaLogger(str(root), False)
        trainer._best_ema = 0.6
        trainer._early_stopping_best_ema = 0.59
        trainer._epochs_without_improvement = 3
        trainer._early_stopping_triggered = False
        trainer.my_init_kwargs = {"configuration": "2d", "fold": 0}
        trainer.inference_allowed_mirroring_axes = (0, 1)
        trainer.print_to_log_file = lambda *args, **kwargs: None
        trainer._ckpd_adapters_installed = False
        trainer._ckpd_resumed = False
        trainer._ckpd_selected_layers = []
        trainer._ckpd_artifact_sha256 = None
        trainer._ckpd_source_sha256 = None
        trainer._ckpd_encoder_fingerprint = None
        trainer._ckpd_target_split_sha256 = None
        trainer._ckpd_target_plans_sha256 = None
        trainer.optimizer, trainer.lr_scheduler = trainer.configure_optimizers()
        return trainer

    root = tmp_path / "target"
    target_dir = root / "Dataset501_StrokeLesion"
    target_dir.mkdir(parents=True)
    (target_dir / "splits_final.json").write_text(
        json.dumps([{"train": ["train-0"], "val": ["val-0"]}]), encoding="utf-8")
    torch.manual_seed(101)
    plain = PlainConvUNet(
        input_channels=1, n_stages=4, features_per_stage=(8, 16, 32, 32), conv_op=nn.Conv2d,
        kernel_sizes=(3, 3, 3, 3), strides=(1, 2, 2, 2), n_conv_per_stage=(1, 1, 2, 1),
        num_classes=2, n_conv_per_stage_decoder=(1, 1, 1), conv_bias=True,
        norm_op=nn.InstanceNorm2d, norm_op_kwargs={"affine": True},
        dropout_op=None, dropout_op_kwargs=None, nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True}, deep_supervision=True,
    )
    baseline = deepcopy(plain)
    baseline_keys = set(plain.state_dict())
    source = make_trainer(plain, root)
    source._ckpd_encoder_fingerprint = module_fingerprint(source.network.encoder)
    source_convs = encoder_conv_layers(source.network)
    selected = list(source_convs)[:4]
    factors = {}
    singulars, asrs, dampings = {}, {}, {}
    for name in selected:
        conv = source_convs[name]
        d = conv.weight.numel() // conv.out_channels
        factor = decompose_conv2d(conv.weight.detach(), torch.eye(d), rank=4)
        factors[name] = factor
        singulars[name], asrs[name], dampings[name] = factor.singular_values, factor.asr, factor.damping
    apply_ckpd_adapters(source.network, factors)
    source.optimizer, source.lr_scheduler = source.configure_optimizers()
    source._ckpd_adapters_installed = True
    source._ckpd_selected_layers = selected
    source._ckpd_artifact_sha256 = "b" * 64
    source._ckpd_source_sha256 = "a" * 64
    source._ckpd_singular_values = singulars
    source._ckpd_asr = asrs
    source._ckpd_damping = dampings
    source._assert_target_split_unchanged()
    source.network.train()
    x = torch.randn(2, 1, 16, 16)

    def update(trainer):
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer.network(x)[0].square().mean().backward()
        trainer.optimizer.step()

    update(source)
    source.lr_scheduler.step()
    source.logger.log("mean_fg_dice", np.float32(0.6), step=0)
    source._best_ema = source.logger.get_value("ema_fg_dice", step=0)
    source._early_stopping_best_ema = np.float64(0.59)
    assert isinstance(source._best_ema, np.float32)
    checkpoint_path = tmp_path / "checkpoint_latest.pth"
    source.save_checkpoint(str(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert all(".parametrizations.weight." not in key for key in checkpoint["network_weights"])
    assert set(checkpoint["network_weights"]) == baseline_keys
    assert isinstance(checkpoint["_best_ema"], np.float32)
    assert isinstance(checkpoint["early_stopping_state"]["best_monitored_ema_dice"], np.float64)

    # --val_best reloads checkpoint_best into the same adapted live Trainer.
    source.load_checkpoint(checkpoint_path)
    source.load_checkpoint(checkpoint_path)
    assert source._ckpd_adapters_installed and source._ckpd_resumed

    invalid_rng_checkpoint = deepcopy(checkpoint)
    invalid_rng_checkpoint["ckpd_training_state"]["rng_state"]["numpy"] = (
        "MT19937", np.zeros(8, dtype=np.uint32), 624, 0, 0.0
    )
    invalid_rng_trainer = make_trainer(deepcopy(baseline), root)
    invalid_rng_network = invalid_rng_trainer.network
    invalid_rng_optimizer = invalid_rng_trainer.optimizer
    invalid_rng_scheduler = invalid_rng_trainer.lr_scheduler
    invalid_rng_logger = invalid_rng_trainer.logger
    invalid_rng_weights = {
        name: value.detach().clone()
        for name, value in invalid_rng_trainer.network.state_dict().items()
    }
    rng_before_invalid = capture_rng_state()
    with pytest.raises(RuntimeError, match="RNG"):
        invalid_rng_trainer.load_checkpoint(invalid_rng_checkpoint)
    assert invalid_rng_trainer.network is invalid_rng_network
    assert invalid_rng_trainer.optimizer is invalid_rng_optimizer
    assert invalid_rng_trainer.lr_scheduler is invalid_rng_scheduler
    assert invalid_rng_trainer.logger is invalid_rng_logger
    assert all(torch.equal(invalid_rng_weights[name], value)
               for name, value in invalid_rng_trainer.network.state_dict().items())
    _assert_rng_states_equal(capture_rng_state(), rng_before_invalid)

    import nnUNetTrainerCKPDTopK10EarlyStopping as trainer_module
    late_failure_trainer = make_trainer(deepcopy(baseline), root)
    late_failure_network = late_failure_trainer.network
    late_failure_optimizer = late_failure_trainer.optimizer
    late_failure_scheduler = late_failure_trainer.lr_scheduler
    late_failure_logger = late_failure_trainer.logger
    late_failure_weights = {
        name: value.detach().clone()
        for name, value in late_failure_trainer.network.state_dict().items()
    }
    late_failure_logger_state = deepcopy(late_failure_logger.get_checkpoint())
    late_failure_rng = capture_rng_state()
    original_restore = trainer_module.restore_rng_state

    def restore_then_fail(state):
        original_restore(state)
        raise RuntimeError("injected late RNG restore failure")

    with monkeypatch.context() as patch:
        patch.setattr(trainer_module, "restore_rng_state", restore_then_fail)
        with pytest.raises(RuntimeError, match="injected late RNG restore failure"):
            late_failure_trainer.load_checkpoint(checkpoint)
    assert late_failure_trainer.network is late_failure_network
    assert late_failure_trainer.optimizer is late_failure_optimizer
    assert late_failure_trainer.lr_scheduler is late_failure_scheduler
    assert late_failure_trainer.logger is late_failure_logger
    assert late_failure_trainer.current_epoch == 1
    assert not late_failure_trainer._ckpd_adapters_installed
    assert not late_failure_trainer._ckpd_resumed
    assert late_failure_trainer._ckpd_selected_layers == []
    assert late_failure_trainer._early_stopping_checkpoint_state() == {
        "best_monitored_ema_dice": 0.59,
        "epochs_without_improvement": 3,
        "triggered": False,
    }
    assert late_failure_logger.get_checkpoint() == late_failure_logger_state
    assert all(torch.equal(late_failure_weights[name], value)
               for name, value in late_failure_trainer.network.state_dict().items())
    _assert_rng_states_equal(capture_rng_state(), late_failure_rng)

    invalid_metadata = deepcopy(checkpoint)
    invalid_metadata["current_epoch"] = "not-an-epoch"
    invalid_trainer = make_trainer(deepcopy(baseline), root)
    before_invalid_load = {name: value.detach().clone()
                           for name, value in invalid_trainer.network.state_dict().items()}
    with pytest.raises(RuntimeError, match="current_epoch"):
        invalid_trainer.load_checkpoint(invalid_metadata)
    assert invalid_trainer.current_epoch == 1
    assert all(torch.equal(before_invalid_load[name], value)
               for name, value in invalid_trainer.network.state_dict().items())

    resume_initialization = deepcopy(baseline)
    with torch.no_grad():
        for parameter in resume_initialization.parameters():
            parameter.add_(0.25)
    restored = make_trainer(resume_initialization, root)
    restored.load_checkpoint(checkpoint)
    assert source.current_epoch == checkpoint["current_epoch"]
    assert restored.current_epoch == checkpoint["current_epoch"]
    assert restored._early_stopping_checkpoint_state() == source._early_stopping_checkpoint_state()
    assert restored._ckpd_resumed and restored._ckpd_adapters_installed
    update(source)
    update(restored)
    source_merged = merged_state_dict(source.network)
    restored_merged = merged_state_dict(restored.network)
    assert source_merged.keys() == restored_merged.keys()
    for name in source_merged:
        assert torch.allclose(source_merged[name], restored_merged[name], atol=1e-7, rtol=1e-6), name
    assert source.lr_scheduler.state_dict() == restored.lr_scheduler.state_dict()

    modules = dict(source.network.named_modules(remove_duplicate=False))
    nonselected_aliases = None
    for name, conv in encoder_conv_layers(source.network).items():
        if name not in selected:
            aliases = [alias for alias, module in modules.items() if module is conv]
            if len(aliases) > 1:
                nonselected_aliases = aliases
                break
    assert nonselected_aliases
    malformed = deepcopy(checkpoint)
    malformed_key = nonselected_aliases[-1] + ".weight"
    malformed["network_weights"][malformed_key] = malformed["network_weights"][malformed_key] + 1
    rejected = make_trainer(deepcopy(baseline), root)
    pre_load = {name: value.detach().clone() for name, value in rejected.network.state_dict().items()}
    with pytest.raises(RuntimeError, match="merged inference aliases disagree"):
        rejected.load_checkpoint(malformed)
    assert all(torch.equal(pre_load[name], value) for name, value in rejected.network.state_dict().items())


def test_results_root_overlap_and_existing_fresh_run_are_rejected_without_blocking_validation(tmp_path, monkeypatch):
    from nnunet_ext_trainers.nnUNetTrainerCKPDTopK10EarlyStopping import (
        _assert_results_root_isolated, _has_prior_run_artifacts,
        nnUNetTrainerCKPDTopK10EarlyStopping as Trainer,
    )

    raw_root = tmp_path / "raw"
    preprocessed_root = tmp_path / "preprocessed"
    raw_root.mkdir()
    preprocessed_root.mkdir()
    monkeypatch.setenv("nnUNet_raw", str(raw_root))
    monkeypatch.setenv("nnUNet_preprocessed", str(preprocessed_root))
    monkeypatch.setenv("nnUNet_results", str(raw_root / "results"))
    with pytest.raises(RuntimeError, match="overlaps protected nnUNet_raw"):
        _assert_results_root_isolated()
    isolated = tmp_path / "results"
    monkeypatch.setenv("nnUNet_results", str(isolated))
    assert _assert_results_root_isolated() == isolated.resolve()

    fresh_folder = tmp_path / "fresh"
    fresh_folder.mkdir()
    assert not _has_prior_run_artifacts(fresh_folder)
    (fresh_folder / "checkpoint_best.pth").write_bytes(b"synthetic")
    assert _has_prior_run_artifacts(fresh_folder)
    trainer = object.__new__(Trainer)
    trainer._ckpd_adapters_installed = False
    trainer._ckpd_resumed = False
    trainer._ckpd_prior_run_artifacts = True
    with pytest.raises(RuntimeError, match="existing run artifacts"):
        trainer.run_training()
