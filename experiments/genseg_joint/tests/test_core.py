import torch
import pytest
from experiments.genseg_joint.core import virtual_chain, segmentation_loss, FunctionalModel


def analytic(alpha, *, indirect=False, sever=False, cut_virtual=False, slr=.17, val=.8):
    gp = {'w': torch.tensor(.6, dtype=torch.double, requires_grad=True)}
    sp = {'w': torch.tensor(.3, dtype=torch.double, requires_grad=True)}
    # Fixed detached direct alpha is essential: FD varies virtual-update alpha ONLY.
    direct = torch.tensor(.4, dtype=torch.double) if indirect else alpha
    def gl(p):
        return ((p['w'] * (1 + alpha) - 1.2)**2 + .2 * p['w']**2)
    def train(p, vg):
        weight = vg['w'].detach() if cut_virtual else vg['w']
        fake = weight * (1 + direct)
        if sever:
            fake = fake.detach()
        return (p['w'] * .7 - .2)**2 + (p['w'] * fake - .9)**2
    return virtual_chain(gp, sp, gl, train, lambda p: (p['w'] * .9 - val)**2, .13, slr)[0]


@pytest.mark.parametrize('indirect', [False, True])
def test_hypergradient_finite_difference_and_indirect_sentinel(indirect):
    a = torch.tensor(.4, dtype=torch.double, requires_grad=True)
    grad, = torch.autograd.grad(analytic(a, indirect=indirect), a)
    eps = 1e-5
    fd = (analytic(torch.tensor(.4+eps, dtype=torch.double, requires_grad=True), indirect=indirect) -
          analytic(torch.tensor(.4-eps, dtype=torch.double, requires_grad=True), indirect=indirect))/(2*eps)
    assert abs(float(grad)) > 1e-6
    torch.testing.assert_close(grad, fd, atol=1e-8, rtol=1e-6)


def test_severed_chain_zero_s_lr_and_validation_target():
    a = torch.tensor(.4, dtype=torch.double, requires_grad=True)
    assert torch.autograd.grad(analytic(a, sever=True), a, allow_unused=True)[0] is None
    assert torch.autograd.grad(analytic(a, slr=0), a, allow_unused=True)[0].item() == 0
    x = torch.autograd.grad(analytic(a, val=.8), a)[0]
    y = torch.autograd.grad(analytic(a, val=.1), a)[0]
    assert x != y


def test_ignore_pixel_loss_invariance():
    y = torch.tensor([[[[1., -1.], [0., 1.]]]])
    x = torch.zeros_like(y, requires_grad=True)
    z = x.detach().clone(); z[0,0,0,1] = 900
    torch.testing.assert_close(segmentation_loss(x,y), segmentation_loss(z,y))
    grad, = torch.autograd.grad(segmentation_loss(x,y), x)
    assert grad[0,0,0,1] == 0


def test_functional_live_parameter_and_buffer_isolation():
    net = torch.nn.Sequential(torch.nn.BatchNorm2d(1), torch.nn.Conv2d(1,1,1))
    state = {k:v.clone() for k,v in net.state_dict().items()}
    fn = FunctionalModel(net)
    p = fn.parameters()
    out = fn(p, torch.randn(2,1,4,4))
    assert out.requires_grad
    assert all(torch.equal(v, state[k]) for k,v in net.state_dict().items())
    assert all(p[k].data_ptr() != v.data_ptr() for k,v in net.named_parameters())


def test_cut_virtual_g_kills_indirect_but_direct_route_remains():
    a=torch.tensor(.4,dtype=torch.double,requires_grad=True)
    indirect=torch.autograd.grad(analytic(a,indirect=True,cut_virtual=True),a,allow_unused=True)[0]
    assert indirect is None
    direct=torch.autograd.grad(analytic(a,cut_virtual=True),a)[0]
    assert abs(float(direct))>1e-6
