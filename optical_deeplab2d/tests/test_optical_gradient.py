import torch

from optical_deeplab2d.models.optical_conv import OpticalConv2d


def test_optical_kernel_receives_finite_nonzero_gradient() -> None:
    layer = OpticalConv2d()
    layer(torch.randn(2, 1, 16, 16)).square().mean().backward()
    gradient = layer.conv.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and torch.count_nonzero(gradient)

