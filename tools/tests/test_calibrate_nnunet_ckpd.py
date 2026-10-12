from __future__ import annotations

import json
import pickle
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from dynamic_network_architectures.architectures.unet import PlainConvUNet

ROOT = Path(__file__).resolve().parents[2]
EXTENSION_ROOT = ROOT / "nnunet_ext_trainers"
TOOLS_ROOT = ROOT / "tools"
for entry in (EXTENSION_ROOT, TOOLS_ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import calibrate_nnunet_ckpd as calibrator


def _tiny_plain_unet():
    return PlainConvUNet(
        input_channels=1, n_stages=3, features_per_stage=(8, 16, 32), conv_op=nn.Conv2d,
        kernel_sizes=(3, 3, 3), strides=(1, 2, 2), n_conv_per_stage=(1, 1, 2),
        num_classes=2, n_conv_per_stage_decoder=(1, 1), conv_bias=True,
        norm_op=nn.InstanceNorm2d, norm_op_kwargs={"affine": True},
        dropout_op=None, dropout_op_kwargs=None, nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True}, deep_supervision=True,
    )


class _FakeConfiguration:
    data_identifier = "nnUNetPlansFrom501_2d"
    patch_size = (16, 16)
    network_arch_class_name = "dynamic_network_architectures.architectures.unet.PlainConvUNet"


class _FakePlansManager:
    def __init__(self, plans):
        self.plans = plans
        self.dataset_name = plans["dataset_name"]
        self.plans_name = plans["plans_name"]

    def get_configuration(self, _name):
        return _FakeConfiguration()

    def get_label_manager(self, _dataset_json):
        return SimpleNamespace(num_segmentation_heads=2)


def _fixture(tmp_path):
    dataset_root = tmp_path / "Dataset508_ISLES2022DWI"
    dataset_root.mkdir()
    data_dir = dataset_root / "nnUNetPlansFrom501_2d"
    data_dir.mkdir()
    (dataset_root / "splits_final.json").write_text(
        json.dumps([{"train": ["case-train-0"], "val": ["case-val-0"]}]), encoding="utf-8")
    data = np.linspace(-1, 1, 2 * 16 * 16, dtype=np.float32).reshape(1, 2, 16, 16)
    seg = np.zeros((1, 2, 16, 16), dtype=np.int16)
    np.savez_compressed(data_dir / "case-train-0.npz", data=data, seg=seg)
    with (data_dir / "case-train-0.pkl").open("wb") as stream:
        pickle.dump({"synthetic": True}, stream)

    plans = {
        "dataset_name": "Dataset508_ISLES2022DWI",
        "plans_name": "nnUNetPlansFrom501",
        "configurations": {"2d": {"patch_size": [16, 16]}},
    }
    dataset_json = {"name": "Dataset508_ISLES2022DWI", "channel_names": {"0": "MRI"}}
    plans_path = tmp_path / "metadata" / "nnUNetPlansFrom501.json"
    plans_path.parent.mkdir()
    plans_path.write_text(json.dumps(plans), encoding="utf-8")
    dataset_path = dataset_root / "dataset.json"
    dataset_path.write_text(json.dumps(dataset_json), encoding="utf-8")

    checkpoint = (tmp_path / "nnUNet_results_ISLES508_CKPD_ES" / "Dataset508_ISLES2022DWI"
                  / "nnUNetTrainerTopK10EarlyStopping__nnUNetPlansFrom501__2d" / "fold_0" / "checkpoint_best.pth")
    checkpoint.parent.mkdir(parents=True)
    torch.manual_seed(91)
    source_network = _tiny_plain_unet()
    torch.save({
        "network_weights": source_network.state_dict(),
        "trainer_name": "nnUNetTrainerTopK10EarlyStopping",
        "init_args": {"plans": plans, "configuration": "2d", "fold": 0, "dataset_json": dataset_json,
                      "device": torch.device("cpu")},
    }, checkpoint)
    args = SimpleNamespace(
        preprocessed_dir=dataset_root,
        plans_json=plans_path,
        dataset_json=dataset_path,
        source_checkpoint=checkpoint,
        output=tmp_path / "calibration.pth",
        configuration="2d", fold=0, cases=1, rank=4, layers=4,
        max_positions=8, max_covariance_mib=16, damping_relative=0.01,
        seed=20261012, cpu_threads=1, device=torch.device("cpu"), patch_size=(16, 16),
    )
    return args, plans, dataset_json


def _install_fake_runtime(monkeypatch, event_order):
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.utilities.plans_handling import plans_handler

    monkeypatch.setattr(plans_handler, "PlansManager", _FakePlansManager)

    def build(_plans_manager, _configuration_manager, _in_channels, _heads, enable_deep_supervision):
        assert enable_deep_supervision
        event_order.append("build_network")
        return _tiny_plain_unet()

    monkeypatch.setattr(nnUNetTrainer, "build_network_architecture", staticmethod(build))
    monkeypatch.setattr(calibrator, "_early_resource_preflight", lambda _args: event_order.append("early_preflight"))
    monkeypatch.setattr(calibrator, "_resource_preflight", lambda _args, _network, _dimension: event_order.append("detailed_preflight"))


def test_calibrate_uses_resolved_prepared_config_and_source_train_case_only(tmp_path, monkeypatch):
    args, _plans, _dataset_json = _fixture(tmp_path)
    order = []
    _install_fake_runtime(monkeypatch, order)
    artifact = calibrator.calibrate(args)

    assert args.configuration_data_dir == args.preprocessed_dir / "nnUNetPlansFrom501_2d"
    assert artifact["selected_case_hashes"] == [calibrator.hash_identifier("case-train-0")]
    assert artifact["source_train_case_hashes"] == [calibrator.hash_identifier("case-train-0")]
    assert artifact["source_val_case_hashes"] == [calibrator.hash_identifier("case-val-0")]
    assert artifact["source_encoder_semantic_signature"]
    assert artifact["patch_size"] == [16, 16]
    assert artifact["sample_count"] == 1
    assert artifact["source_trainer_name"] == "nnUNetTrainerTopK10EarlyStopping"
    assert artifact["source_plans_name"] == "nnUNetPlansFrom501"
    assert artifact["source_data_identifier"] == "nnUNetPlansFrom501_2d"
    assert order.index("early_preflight") < order.index("build_network") < order.index("detailed_preflight")


def test_source_checkpoint_fold_metadata_mismatch_fails_before_network_or_weight_load(tmp_path, monkeypatch):
    args, _plans, _dataset_json = _fixture(tmp_path)
    checkpoint = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    checkpoint["init_args"]["fold"] = 1
    torch.save(checkpoint, args.source_checkpoint)
    del checkpoint

    order = []
    _install_fake_runtime(monkeypatch, order)
    from nnunetv2.run import load_pretrained_weights as loader_module
    monkeypatch.setattr(loader_module, "load_pretrained_weights",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("weights loaded before metadata validation")))

    with pytest.raises(RuntimeError, match="configuration/fold"):
        calibrator.calibrate(args)
    assert "build_network" not in order
    assert order == ["early_preflight"]


def test_calibration_output_rejects_raw_preprocessed_and_source_experiment_roots(tmp_path, monkeypatch):
    args, _plans, _dataset_json = _fixture(tmp_path)
    raw_root = tmp_path / "raw"
    preprocessed_root = tmp_path / "preprocessed-global"
    raw_root.mkdir()
    preprocessed_root.mkdir()
    monkeypatch.setenv("nnUNet_raw", str(raw_root))
    monkeypatch.setenv("nnUNet_preprocessed", str(preprocessed_root))
    roots = calibrator._configured_output_protected_roots(args)

    with pytest.raises(RuntimeError, match="protected source/output root"):
        calibrator._assert_output_is_safe(raw_root / "artifact.pth", roots)
    with pytest.raises(RuntimeError, match="protected source/output root"):
        calibrator._assert_output_is_safe(preprocessed_root / "artifact.pth", roots)
    source_dataset_root = args.source_checkpoint.parents[2]
    with pytest.raises(RuntimeError, match="protected source/output root"):
        calibrator._assert_output_is_safe(source_dataset_root / "artifact.pth", roots)


def test_cuda_origin_checkpoint_is_rejected_for_cpu_before_network_or_loader(tmp_path, monkeypatch):
    args, _plans, _dataset_json = _fixture(tmp_path)
    checkpoint = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    checkpoint["init_args"]["device"] = torch.device("cuda")
    torch.save(checkpoint, args.source_checkpoint)
    del checkpoint

    order = []
    _install_fake_runtime(monkeypatch, order)
    with pytest.raises(RuntimeError, match="CPU calibration cannot load a CUDA-origin checkpoint"):
        calibrator.calibrate(args)
    assert order == ["early_preflight"]


def test_source_cuda_index_must_be_visible_and_match_requested_device(monkeypatch):
    args = SimpleNamespace(device=torch.device("cuda:0"))
    monkeypatch.setattr(calibrator.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(calibrator.torch.cuda, "device_count", lambda: 1)
    with pytest.raises(RuntimeError, match="hidden or unavailable under CUDA_VISIBLE_DEVICES"):
        calibrator._validate_source_checkpoint_device(args, {"device": torch.device("cuda:1")})

    monkeypatch.setattr(calibrator.torch.cuda, "device_count", lambda: 2)
    with pytest.raises(RuntimeError, match="differs from requested visible device"):
        calibrator._validate_source_checkpoint_device(args, {"device": torch.device("cuda:1")})


def test_gpu_utilization_check_uses_all_physical_rows_even_with_device_remap(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(calibrator.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(stdout="0, 4\n1, 85\n"))
    rows = calibrator._physical_gpu_utilizations()
    assert [row["physical_index"] for row in rows] == [0, 1]
    with pytest.raises(RuntimeError, match="physical GPU utilization >=80%"):
        calibrator._require_gpu_utilization_headroom(rows)


def test_configuration_identifier_cannot_escape_dataset_root(tmp_path):
    root = tmp_path / "Dataset508_ISLES2022DWI"
    (root / "nnUNetPlansFrom501_2d").mkdir(parents=True)
    assert calibrator._resolve_configuration_data_dir(root, "nnUNetPlansFrom501_2d") == root.resolve() / "nnUNetPlansFrom501_2d"
    with pytest.raises(RuntimeError, match="escapes Dataset508 root"):
        calibrator._resolve_configuration_data_dir(root, "..")
