"""One large CPU P99 regression plus tiny exact/legacy-linear comparisons."""
import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import isoplanatism_core as core


def production_p99(values):
    # Before the fix this deliberately exercises the actual legacy expression.
    function = getattr(core, '_linear_p99_cpu', None)
    return float(torch.quantile(values, .99)) if function is None else function(values)


def test_above_torch_quantile_limit():
    n = (1 << 24) + 1
    rank = (n - 1) * .99
    split = math.floor(rank) + 1
    values = torch.zeros(n, dtype=torch.float32)
    values[split:] = 1
    before = hashlib.sha256(memoryview(values.numpy())).hexdigest()
    result = production_p99(values)
    expected = rank - math.floor(rank)
    assert result == pytest.approx(expected, abs=2e-6)
    assert hashlib.sha256(memoryview(values.numpy())).hexdigest() == before
    print(f'large: numel={n}, dtype={values.dtype}, rank={rank}, result={result}, expected={expected}')


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('data', [[2.5], [0., 0., 1., 1.], [-2., .25, .5, 1.5, 3., 7.]])
def test_small_matches_legacy_linear(dtype, data):
    values = torch.tensor(data, dtype=dtype)
    original = values.clone()
    expected = float(torch.quantile(values, .99))
    result = production_p99(values)
    error = abs(result - expected)
    tolerance = 2e-6 if dtype == torch.float32 else 1e-12
    assert result == pytest.approx(expected, abs=tolerance)
    assert torch.equal(values, original) and values.dtype == dtype
    print(f'small: n={len(data)}, dtype={dtype}, abs_error={error}, tolerance={tolerance}')

@pytest.mark.parametrize('selected_util', [0, 80])
def test_guard_uses_selected_physical_gpu_and_keeps_all_records(monkeypatch, selected_util):
    from types import SimpleNamespace
    import psutil
    import validate_isoplanatism as cli
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    monkeypatch.setattr(psutil, 'cpu_percent', lambda interval: 10)
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: SimpleNamespace(percent=20, available=100*2**30))
    monkeypatch.setattr(cli.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=f'0, 95, 10080, 32607\n1, {selected_util}, 2023, 32607\n'))
    record = cli.resource_preflight()
    assert [g['physical_index'] for g in record['all_gpus']] == [0, 1]
    assert [g['physical_index'] for g in record['gpus']] == [1]
    assert record['all_gpus'][0]['util_percent'] == 95
    if selected_util == 80:
        with pytest.raises(core.ValidationError, match='GPU occupied'):
            cli.guard(record, 800, [800,1600], 'cuda:0')
    else:
        cli.guard(record, 800, [800,1600], 'cuda:0')
        assert record['decision'] == 'NORMAL_SINGLE_THREAD_SERIAL'
