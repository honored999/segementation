"""Tiny real single-device Trainer checks; CUDA runs are manual server evidence.

CPU controls are locally runnable. CUDA checks must fail, never silently skip,
when a requested GPU/runtime is unavailable. No real data, optimizer steps,
DDP/AMP/compile or full-size performance experiments.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "nnunet_ext_trainers/tests"))
from test_upernet_selected_stages import DATASET, Trainer, plans, tensor_state


@pytest.mark.parametrize("device_name", ["cpu", "cuda:0"], ids=["cpu", "cuda0"])
@pytest.mark.parametrize("indices", [[0, 2], [0, 3]], ids=["unused-trailing", "last-stage"])
def test_actual_single_device_trainer(tmp_path, monkeypatch, device_name, indices):
    is_cuda = device_name == "cuda:0"
    if is_cuda:
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "Select physical GPU 0 explicitly"
        assert torch.cuda.is_available(), "Requested CUDA check cannot run without CUDA"
        assert torch.cuda.device_count() == 1, "Expose only GPU 0 to this test process"
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
    assert not torch.distributed.is_initialized(), "This probe is single device, not DDP"
    for key, value in {"nnUNet_results": str(tmp_path / "r"),
                       "nnUNet_preprocessed": str(tmp_path / "p"),
                       "nnUNet_raw": str(tmp_path / "raw"),
                       "nnUNet_compile": "false", "nnUNet_wandb_enabled": "0",
                       "nnUNet_mlflow_enabled": "0"}.items():
        monkeypatch.setenv(key, value)
    folder = tmp_path / "p/Dataset999_Synthetic/OriginalPlans_2d"
    folder.mkdir(parents=True)
    np.savez_compressed(folder / "synthetic.npz", data=np.zeros((1, 1, 16, 16), dtype=np.float32))
    data = plans(indices, n=4, strides=[[1, 1]] + [[2, 2]] * 3)
    data["plans_name"] = "S" + "".join(map(str, indices))
    data["continue_training"] = False
    device = torch.device(device_name)
    original = Trainer(deepcopy(data), "2d", 0, DATASET, device)
    original.initialize()  # real official initialization; no parent/device shim
    assert not original.is_ddp and original.enable_deep_supervision is False
    model = original.network
    assert len(model.encoder.stages) == 4 and model.selected_feature_indices == tuple(indices)
    assert all(parameter.device == device for parameter in model.parameters())
    assert original.optimizer.param_groups[0]["lr"] == .01
    assert original.grad_scaler is not None if is_cuda else original.grad_scaler is None
    initial = tensor_state(model)
    called = []
    handle = model.encoder.stages[-1].register_forward_hook(lambda *args: called.append(True))
    try:
        for _ in range(2):
            original.optimizer.zero_grad(set_to_none=True)
            image = torch.randn(1, 1, 17, 17, device=device)
            target = torch.randint(0, 3, (1, 1, 17, 17), device=device)
            logits = model(image)
            loss = original.loss(logits, target)  # inherited single-output TopK10 loss, FP32
            assert logits.shape == (1, 3, 17, 17) and torch.isfinite(logits).all()
            assert torch.isfinite(loss)
            loss.backward()
            for index, stage in enumerate(model.encoder.stages):
                for parameter in stage.parameters():
                    if index > indices[-1]:
                        assert parameter.grad is None
                    else:
                        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert called == [True, True]
    finally:
        handle.remove()
    # Forward/backward evidence only: no optimizer.step or real training loop.
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in initial.items())
    original.current_epoch = 320
    original._early_stopping_best_ema = .7
    original._epochs_without_improvement = 11
    checkpoint = tmp_path / "synthetic.pth"
    original.save_checkpoint(str(checkpoint))
    receiver = Trainer(deepcopy(data), "2d", 0, DATASET, device)
    receiver.load_checkpoint(str(checkpoint))  # real reconstruction and inherited strict recovery
    assert receiver.was_initialized and receiver.current_epoch == 321
    assert receiver._epochs_without_improvement == 11
    assert all(torch.equal(value, receiver.network.state_dict()[key]) for key, value in initial.items())
    assert 'effective selected-stage identity' in Path(receiver.log_file).read_text(encoding="utf-8")
    invalid = torch.load(checkpoint, map_location=device, weights_only=False)
    invalid["network_weights"]["_extra_state"]["indices"] = [0, 3] if indices == [0, 2] else [0, 2]
    invalid["optimizer_state"] = {"must_not_load": True}
    invalid["current_epoch"] = 999
    with pytest.raises(ValueError, match="identity"):
        receiver.load_checkpoint(invalid)
    assert receiver.current_epoch == 321
    assert receiver.optimizer.param_groups[0]["lr"] == .01
    assert all(torch.equal(value, receiver.network.state_dict()[key]) for key, value in initial.items())
    if is_cuda:
        torch.cuda.synchronize(0)
        print("CUDA_SINGLE_DEVICE_SYNTHETIC_OK " + json.dumps({
            "device": device_name, "name": torch.cuda.get_device_name(0),
            "capability": torch.cuda.get_device_capability(0), "indices": indices,
            "torch_peak_allocated_MiB": torch.cuda.max_memory_allocated(0) / 2**20,
            "input": [1, 1, 17, 17], "output": [1, 3, 17, 17],
            "boundaries": "FP32 only; no optimizer step/DDP/AMP/compile/real data"}))
