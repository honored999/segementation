"""Zero-branch return, gain and NPZ contract; no model or propagation."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fit_psf_gray450 as fit


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_zero_branch_cpu_export(device, tmp_path, monkeypatch):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable; CPU coverage is not CUDA reproduction')

    def forbidden(*args, **kwargs):
        pytest.fail('Zero branch must not construct a model or optimizer')

    monkeypatch.setattr(fit, 'Metasurface', forbidden)
    monkeypatch.setattr(torch.optim, 'Adam', forbidden)
    target = torch.zeros((1, 8, 8), dtype=torch.float32, device=device)
    width, psf, history, active = fit.fit_one(target, SimpleNamespace())
    target_cpu = target.cpu()
    # This is the unchanged production gain/export path that failed on CUDA.
    gains = ((psf * target_cpu).flatten(1).sum(1) /
             psf.square().flatten(1).sum(1).clamp_min(1e-12))
    assert width is None and history == []
    assert psf.device.type == active.device.type == 'cpu'
    assert psf.shape == target.shape and psf.dtype == target.dtype
    assert active.dtype == torch.bool and active.shape == (1,) and not active.any()
    assert torch.equal(psf, target_cpu) and torch.equal(gains, torch.zeros(1))
    artifact = tmp_path / 'kernel_000_pos.npz'
    np.savez_compressed(artifact, width_nm=np.empty((0, 0)), psf_raw=psf.numpy(),
                        target=target_cpu.numpy(), active_gray=active.numpy(),
                        gain_gray=gains.numpy(), history=np.array(history), wavelength_nm=450)
    summary = dict(index=0, sign='pos', skipped=not bool(active.any()),
                   steps=len(history), best_loss=min(history) if history else None)
    (tmp_path / 'summary.json').write_text(json.dumps([summary]), encoding='utf-8')
    with np.load(artifact, allow_pickle=False) as saved:
        assert saved['width_nm'].shape == (0, 0)
        assert not saved['psf_raw'].any() and not saved['active_gray'].any()
        assert not saved['gain_gray'].any() and saved['history'].size == 0
    assert summary == dict(index=0, sign='pos', skipped=True, steps=0, best_loss=None)
