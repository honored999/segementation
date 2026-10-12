"""Small CPU contracts for the CKPD Conv2d math and bounded patch sampler."""

from pathlib import Path
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ckpd import (  # noqa: E402
    CKPDWeightParametrization,
    decompose_conv2d,
    sample_conv2d_patches,
    select_smallest_asr,
)


def test_sampled_patches_match_unfold_second_moment_for_all_positions():
    torch.manual_seed(4)
    x = torch.randn(2, 2, 7, 8)
    conv = nn.Conv2d(2, 3, kernel_size=(3, 2), stride=(2, 1), padding=(1, 0), bias=False)
    sampled = sample_conv2d_patches(x, conv, max_positions=1000, seed=12)
    expected = F.unfold(x, conv.kernel_size, conv.dilation, conv.padding, conv.stride)
    expected = expected.transpose(1, 2).reshape(-1, expected.shape[1])
    assert sampled.shape == expected.shape
    assert torch.allclose(sampled.T @ sampled / (sampled.shape[0] / x.shape[0]),
                          expected.T @ expected / (expected.shape[0] / x.shape[0]), atol=1e-5)


def test_sampled_patches_are_bounded_and_repeatable():
    x = torch.arange(2 * 2 * 32 * 32, dtype=torch.float32).reshape(2, 2, 32, 32)
    conv = nn.Conv2d(2, 3, kernel_size=5, padding=2, bias=False)
    first = sample_conv2d_patches(x, conv, max_positions=7, seed=51)
    second = sample_conv2d_patches(x, conv, max_positions=7, seed=51)
    assert first.shape == (14, 50)
    assert torch.equal(first, second)


def test_rank_one_decomposition_reconstructs_weight_exactly():
    torch.manual_seed(7)
    weight = torch.randn(4, 2, 3, 3)
    raw = torch.randn(18, 18)
    covariance = raw @ raw.T / raw.shape[0] + torch.eye(18) * 0.25
    result = decompose_conv2d(weight, covariance, rank=1, damping_relative=0.01)
    reconstructed = result.base_weight.flatten(1) + result.B @ result.A
    assert result.A.shape == (1, 18)
    assert result.B.shape == (4, 1)
    assert torch.allclose(reconstructed, weight.flatten(1), atol=2e-5, rtol=2e-5)
    conv = nn.Conv2d(2, 4, 3, padding=1, bias=False)
    conv.weight.data.copy_(weight)
    x = torch.randn(2, 2, 12, 13)
    merged = nn.functional.conv2d(
        x,
        reconstructed.reshape_as(weight),
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
    )
    assert torch.allclose(conv(x), merged, atol=2e-5, rtol=2e-5)


def test_parametrization_product_stays_fp32_inside_cpu_autocast():
    torch.manual_seed(74)
    original = torch.randn(3, 1, 3, 3)
    A = torch.randn(4, 9) * 1.7
    B = torch.randn(3, 4) * 1.3
    adapter = CKPDWeightParametrization(A, B)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = adapter(original)
        actual.square().mean().backward()

    expected = original + (B.detach() @ A.detach()).reshape_as(original)
    assert actual.dtype == torch.float32
    assert torch.allclose(actual.detach(), expected, atol=1e-6, rtol=1e-6)
    assert adapter.A.grad is not None and torch.isfinite(adapter.A.grad).all()
    assert adapter.B.grad is not None and torch.isfinite(adapter.B.grad).all()
    assert adapter.A.grad.dtype == torch.float32
    assert adapter.B.grad.dtype == torch.float32


@pytest.mark.parametrize("rank", [0, -1, 5])
def test_decomposition_rejects_invalid_rank(rank):
    with pytest.raises(ValueError, match="rank"):
        decompose_conv2d(torch.randn(4, 2, 3, 3), torch.eye(18), rank=rank)


def test_decomposition_rejects_nonfinite_or_wrong_covariance():
    weight = torch.randn(4, 2, 3, 3)
    bad = torch.eye(18)
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        decompose_conv2d(weight, bad, rank=1)
    with pytest.raises(ValueError, match="shape"):
        decompose_conv2d(weight, torch.eye(17), rank=1)


def test_layer_selection_uses_paper_v2_asr_and_stable_name_ties():
    # sigma[-r] / sigma[-1], with singular values supplied descending.
    expected = ["alpha", "charlie"]
    actual = select_smallest_asr(
        {"bravo": torch.tensor([8.0, 4.0, 2.0, 1.0]),
         "alpha": torch.tensor([8.0, 4.0, 1.0, 1.0]),
         "charlie": torch.tensor([8.0, 3.0, 1.0, 1.0])},
        rank=2,
        count=2,
    )
    assert actual == expected


def test_layer_selection_rejects_impossible_count_or_rank():
    scores = {"one": torch.tensor([2.0, 1.0])}
    with pytest.raises(ValueError, match="count"):
        select_smallest_asr(scores, rank=1, count=2)
    with pytest.raises(ValueError, match="rank"):
        select_smallest_asr(scores, rank=3, count=1)
