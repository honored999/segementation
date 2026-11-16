"""Small numerical checks only; no medical data or full random network."""
import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tools.psf_conversion import psf_first_conv as production
from tools.psf_conversion.psf_first_conv import (
    effective_kernel, FirstConvHook, KEY, LEGACY_OPERATOR, load_psfs, operator_record,
    PIXEL_OPERATOR, psf_canvas, superposition_kernel)

S, C = 17, 8


def replicated_target(n, s, component):
    """Saved-target analogue: every 3x3 element replicated s times around the origin."""
    o, c = n // 2, s // 2
    target = np.zeros((n, n))
    start = o - c - s
    for a in range(3):
        for b in range(3):
            target[start+a*s:start+(a+1)*s, start+b*s:start+(b+1)*s] = component[a, b]
    return target


def explicit_superposition(x, psf, origin, s):
    """Independent D-model reference: no production kernel, no coarse operator.

    D(X)[s*i+c, s*j+c] = X[i,j]; Z is the sum of one complete shifted PSF per
    logical pixel; Y is the s x s block mean.
    """
    h, w = x.shape
    c, (oy, ox), (n0, n1) = s // 2, origin, psf.shape
    z = np.zeros((h*s, w*s))
    for i in range(h):
        for j in range(w):
            top, left = s*i + c - oy, s*j + c - ox
            ys, ye = max(top, 0), min(top + n0, h*s)
            xs, xe = max(left, 0), min(left + n1, w*s)
            if ys < ye and xs < xe:
                z[ys:ye, xs:xe] += x[i, j] * psf[ys-top:ye-top, xs-left:xe-left]
    return z.reshape(h, s, w, s).mean(axis=(1, 3))


def block_centre_reference(x, psf, origin, s):
    """U17-style chain: replicate the input, physical C, sample block centres, /s**2."""
    h, w = x.shape
    c, (oy, ox), (n0, n1) = s // 2, origin, psf.shape
    fine = torch.from_numpy(np.repeat(np.repeat(x, s, 0), s, 1))[None, None]
    padded = F.pad(fine, (n1-1-ox, ox, n0-1-oy, oy))
    full = F.conv2d(padded, torch.from_numpy(psf[::-1, ::-1].copy())[None, None])
    rows, cols = s*np.arange(h) + c, s*np.arange(w) + c
    return full[0, 0][rows[:, None], cols[None, :]].numpy() / float(s*s)


@pytest.mark.parametrize('s, n', [(3, 17), (17, 137)])
def test_pixel_pulse_recovers_delta(s, n):
    """A centre-only target must stay a delta: the block mean must not smear it."""
    o = n // 2
    component = np.zeros((3, 3))
    component[1, 1] = 1.0
    psf = replicated_target(n, s, component)
    x = np.zeros((5, 5))
    x[2, 2] = 1.0
    patch = explicit_superposition(x, psf, (o, o), s)
    delta = np.zeros((5, 5))
    delta[2, 2] = 1.0
    np.testing.assert_array_equal(patch, delta)
    kernel = superposition_kernel(psf, (o, o), step=s)
    actual = F.conv2d(torch.from_numpy(x)[None, None], torch.from_numpy(kernel)[None, None],
                      padding=kernel.shape[-1]//2)[0, 0].numpy()
    np.testing.assert_allclose(actual, delta, rtol=1e-12, atol=1e-12)
    legacy = F.conv2d(torch.from_numpy(x)[None, None],
                      torch.from_numpy(effective_kernel(psf, (o, o)))[None, None],
                      padding=effective_kernel(psf, (o, o)).shape[-1]//2)[0, 0].numpy()
    print(f's={s}: pixel centre={actual[2,2]:.12f}, legacy centre={legacy[2,2]:.12f}, '
          f'49/81={49/81:.12f}')
    # The brief's legacy artefact value must not survive into the pixel protocol.
    assert actual[2, 2] == 1.0 and actual[2, 2] != 49/81
    assert abs(legacy[2, 2] - 1.0) > 0.05


@pytest.mark.parametrize('n', [51, 64])
def test_explicit_reference(n):
    # Noncentral peripheral response, negative input, asymmetric branches, zero branch.
    x = torch.tensor([[[[1., -2., 0.], [0., 3., 1.]]]], dtype=torch.float64)
    pos = np.zeros((n,n)); neg = pos.copy()
    pos[n//2+2, n//2-5] = 2
    pos[0, -1] = 0.7
    neg[-1, 1] = 1.1
    bias, scale, gp, gn = 0.4, 2.3, 0.8, 1.7
    p = gp*pos-gn*neg
    k = torch.from_numpy(effective_kernel(p, (n//2,n//2)))[None,None]
    actual = F.conv2d(x, k, padding=k.shape[-1]//2)*scale+bias
    fine = x.repeat_interleave(17,-2).repeat_interleave(17,-1)
    # Physical C: fine[t + origin - p]; Conv2d cross-correlates flipped P.
    o = n//2
    padded = F.pad(fine, (n-1-o,o,n-1-o,o))
    reference = F.avg_pool2d(F.conv2d(padded, torch.from_numpy(p[::-1,::-1].copy())[None,None]), 17,17)*scale/289+bias
    torch.testing.assert_close(actual, reference, rtol=1e-10, atol=1e-12)
    assert not effective_kernel(np.zeros_like(pos), (o,o)).any()
    ideal = np.zeros_like(pos)
    ideal[o-25:o+26,o-25:o+26] = np.array([[1,2,0],[0,3,1],[2,0,4]]).repeat(17,0).repeat(17,1)
    ideal_k = torch.from_numpy(effective_kernel(ideal,(o,o)))[None,None]
    ideal_y = F.conv2d(x,ideal_k,padding=ideal_k.shape[-1]//2)
    electrical = F.conv2d(x,torch.tensor([[[[4.,0.,2.],[1.,3.,0.],[0.,2.,1.]]]],dtype=x.dtype),padding=1)
    print('local max ideal-electrical:', (ideal_y-electrical).abs().max().item(),
          'fitted-ideal:', (actual-ideal_y).abs().max().item())


def psf_directory(tmp_path, weight, n=51):
    """Minimal on-disk PSF campaign laid out exactly as load_psfs expects."""
    fft = n + 102
    (tmp_path/'config.json').write_text(json.dumps(dict(
        n=n, patch_meta=51, target_rotated_180=True, grayscale=True,
        wavelength_nm=450, fft_size=fft, padding=(fft-n)//2)))
    np.save(tmp_path/'electrical_kernels.npy', weight)
    scales = np.maximum(np.abs(weight).max(axis=(1, 2, 3)), 1e-12)
    np.save(tmp_path/'shared_scales.npy', scales.astype(np.float32))
    for channel in range(32):
        for sign, polarity in (('pos', 1), ('neg', -1)):
            component = np.maximum(polarity*weight[channel, 0, ::-1, ::-1]/scales[channel], 0)
            active = bool(component.any())
            target = np.zeros((1, n, n), np.float32)
            start = n//2 - 25
            target[0, start:start+51, start:start+51] = component.repeat(17, 0).repeat(17, 1)
            np.savez(tmp_path/f'kernel_{channel:03d}_{sign}.npz', psf_raw=target, target=target,
                     gain_gray=np.array([1.0 if active else 0.0], np.float32),
                     active_gray=np.array([active]), wavelength_nm=np.array(450))
    return tmp_path


def test_ideal_pixel_kernel_matches_electrical_convolution(tmp_path):
    """Ideal mode: signed 3x3 electrical kernel, unit gain, restored shared scales."""
    weight = np.zeros((32, 1, 3, 3), np.float32)
    weight[3, 0] = np.array([[4, 0, -2], [1, 3.5, 0], [0, -2, 1]], np.float32)
    weight[7, 0] = np.array([[-1.25, 2, 0.5], [0, -0.75, 1], [2, 0, -3]], np.float32)
    directory = psf_directory(tmp_path, weight)
    kernel, hashes = load_psfs(directory, torch.from_numpy(weight),
                               operator=PIXEL_OPERATOR, ideal=True)
    assert kernel.shape == (32, 1, 3, 3) and len(hashes) == 67
    np.testing.assert_array_equal(kernel[3, 0].numpy(), weight[3, 0])
    x = torch.linspace(-2, 2, 2*1*9*11, dtype=torch.float64).reshape(2, 1, 9, 11)
    actual = F.conv2d(x, kernel.double(), padding=1)
    electrical = F.conv2d(x, torch.from_numpy(weight).double(), padding=1)
    for channel in (3, 7):
        torch.testing.assert_close(actual[:, channel], electrical[:, channel],
                                   rtol=1e-12, atol=1e-12)
    assert torch.count_nonzero(actual[:, 0]) == 0
    legacy, _ = load_psfs(directory, torch.from_numpy(weight),
                          operator=LEGACY_OPERATOR, ideal=True)
    legacy_y = F.conv2d(x, legacy.double(), padding=legacy.shape[-1]//2)
    assert (legacy_y[:, 3]-electrical[:, 3]).abs().max() > 1e-3
    print('ideal pixel vs electrical:', (actual[:, 3]-electrical[:, 3]).abs().max().item(),
          ' legacy vs electrical:', (legacy_y[:, 3]-electrical[:, 3]).abs().max().item())


@pytest.mark.parametrize('n', [101, 100])
def test_superposition_kernel_matches_explicit_shifts(n, monkeypatch):
    """Explicit per-pixel shift-sum vs the equivalent kernel; 100 is an even canvas."""
    o = n // 2
    pos, neg = np.zeros((n, n)), np.zeros((n, n))
    pos[o+3, o-7] = 2.0                      # inside the central 51x51
    pos[o+30, o+41] = 0.9                    # outside it, still must contribute
    neg[2, n-3] = 1.1
    neg[o-44, o+5] = 0.35                    # outside it, negative branch
    gp, gn, scale, bias = 0.8, 1.7, 2.3, 0.4
    psf = gp*pos - gn*neg
    x = np.random.default_rng(7).normal(size=(7, 7))

    boom = lambda *a, **k: pytest.fail('explicit reference consulted the production kernel')
    monkeypatch.setattr(production, 'superposition_kernel', boom)
    reference = explicit_superposition(x, psf, (o, o), S)
    monkeypatch.undo()

    kernel = superposition_kernel(psf, (o, o))
    actual = F.conv2d(torch.from_numpy(x)[None, None], torch.from_numpy(kernel)[None, None],
                      padding=kernel.shape[-1]//2)[0, 0].numpy()
    np.testing.assert_allclose(actual, reference, rtol=1e-10, atol=1e-12)

    trimmed = psf.copy()
    trimmed[o+30, o+41] = 0.0
    trimmed_kernel = superposition_kernel(trimmed, (o, o))
    trimmed_y = F.conv2d(torch.from_numpy(x)[None, None],
                         torch.from_numpy(trimmed_kernel)[None, None],
                         padding=trimmed_kernel.shape[-1]//2)[0, 0].numpy()
    change = np.abs(actual - trimmed_y).max()
    print(f'n={n}: outer-51x51 response changes the output by {change:.6f}')
    assert change > 1e-3

    centre_sampled = block_centre_reference(x, psf, (o, o), S)
    np.testing.assert_allclose(actual, centre_sampled, rtol=1e-10, atol=1e-12)

    legacy_kernel = effective_kernel(psf, (o, o))
    legacy = F.conv2d(torch.from_numpy(x)[None, None],
                      torch.from_numpy(legacy_kernel)[None, None],
                      padding=legacy_kernel.shape[-1]//2)[0, 0].numpy()
    assert not np.allclose(actual, legacy, rtol=1e-6, atol=1e-9)
    assert np.abs(actual - legacy).max() > 1e-4 * np.abs(actual).max() + 1e-9

    scaled = scale*actual + bias
    positive = explicit_superposition(np.maximum(x, 0), psf, (o, o), S)
    negative = explicit_superposition(np.maximum(-x, 0), psf, (o, o), S)
    np.testing.assert_allclose(scale*(positive - negative) + bias, scaled,
                               rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        F.conv2d(torch.from_numpy(np.maximum(x, 0))[None, None],
                 torch.from_numpy(kernel)[None, None], padding=kernel.shape[-1]//2)[0, 0].numpy()
        - F.conv2d(torch.from_numpy(np.maximum(-x, 0))[None, None],
                   torch.from_numpy(kernel)[None, None], padding=kernel.shape[-1]//2)[0, 0].numpy(),
        actual, rtol=1e-10, atol=1e-12)


def test_operator_selection_is_explicit():
    assert operator_record(PIXEL_OPERATOR, 800)['psf_origin'] == [400, 400]
    assert operator_record(PIXEL_OPERATOR, 800)['pixel_step'] == 17
    assert 'operator_id' in operator_record(LEGACY_OPERATOR, 800)
    assert operator_record(LEGACY_OPERATOR, 800) != operator_record(PIXEL_OPERATOR, 800)
    with pytest.raises(ValueError, match='Unknown first-layer operator'):
        operator_record('pixel_psf_superposition', 800)
    with pytest.raises(ValueError, match='Unknown first-layer operator'):
        production._kernel_fn('u17_block_mean')


@pytest.mark.parametrize('n', [85, 64])
def test_non_800_artifact_records_its_own_origin(tmp_path, n):
    """Provenance must follow the loaded artifact canvas, not an assumed 800."""
    weight = np.zeros((32, 1, 3, 3), np.float32)
    weight[5, 0] = np.array([[1, 0, -2], [0, 3, 0], [2, 0, 1]], np.float32)
    directory = psf_directory(tmp_path, weight, n=n)
    assert psf_canvas(directory) == (n, (n//2, n//2))
    kernel, hashes = load_psfs(directory, torch.from_numpy(weight),
                               operator=PIXEL_OPERATOR, ideal=True)
    record = operator_record(PIXEL_OPERATOR, n)
    assert record['psf_origin'] == [n//2, n//2] and record['pixel_step'] == 17
    assert record['psf_origin'] != [400, 400]
    # The kernel footprint scales with the artifact canvas; its parity and layout
    # are still the protocol's, so hook acceptance is the meaningful check.
    assert kernel.shape[:2] == (32, 1) and kernel.shape[2] == kernel.shape[3]
    assert kernel.shape[2] % 2 == 1 and len(hashes) == 67
    assert float(kernel[5, 0].abs().sum()) > 0


def test_hook_alias_reload_remove():
    from dynamic_network_architectures.building_blocks.simple_conv_blocks import ConvDropoutNormReLU
    network = nn.Module()
    block = ConvDropoutNormReLU(nn.Conv2d,1,32,3,1,True,norm_op=nn.InstanceNorm2d,nonlin=nn.LeakyReLU)
    network.encoder = nn.Module()
    network.encoder.stages = nn.Sequential(nn.Sequential(nn.Module()))
    network.encoder.stages[0][0].convs = nn.Sequential(block)
    assert block.conv is block.all_modules[0]
    state = {k:v.clone() for k,v in network.state_dict().items()}
    x = torch.arange(30,dtype=torch.float32).reshape(1,1,5,6)/10-1
    before = block(x).detach()
    kernel = torch.zeros(32,1,3,3); kernel[:,:,1,1] = 2
    hook = FirstConvHook(network,kernel)
    network.load_state_dict(state)
    result = block(x)
    assert result.shape == (1,32,5,6) and torch.isfinite(result).all() and hook.calls == 1
    assert not torch.allclose(before,result)
    assert all(torch.equal(v,state[k]) for k,v in network.state_dict().items())
    assert all(not p.requires_grad for p in network.parameters())
    hook.remove()
    torch.testing.assert_close(block(x),before)


def test_source_mismatch(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps(dict(n=51, patch_meta=51, target_rotated_180=True, grayscale=True, wavelength_nm=450,fft_size=153,padding=51)))
    np.save(tmp_path/'electrical_kernels.npy',np.ones((32,1,3,3),np.float32))
    with pytest.raises(ValueError,match='Checkpoint'):
        load_psfs(tmp_path,torch.zeros(32,1,3,3))


def test_artifact_loading_and_rejections(tmp_path):
    config = dict(n=51, patch_meta=51, target_rotated_180=True, grayscale=True,
                  wavelength_nm=450, fft_size=153, padding=51)
    config_path = tmp_path/'config.json'
    config_path.write_text(json.dumps(config))
    w = np.zeros((32,1,3,3),np.float32)
    w[:,0,1,0] = 2
    np.save(tmp_path/'electrical_kernels.npy',w)
    np.save(tmp_path/'shared_scales.npy',np.full(32,2,np.float32))
    for c in range(32):
        for sign in ('pos','neg'):
            target = np.zeros((1,51,51),np.float32)
            if sign == 'pos':
                target[0,17:34,34:51] = 1
            np.savez(tmp_path/f'kernel_{c:03d}_{sign}.npz',psf_raw=target,target=target,
                     gain_gray=np.array([1 if sign=='pos' else 0],np.float32),
                     active_gray=np.array([sign=='pos']),wavelength_nm=np.array(450))
    kernel, hashes = load_psfs(tmp_path,torch.from_numpy(w))
    expected = effective_kernel(target[0],(25,25)) # last branch is zero
    assert not expected.any()
    assert kernel.shape[0] == 32 and len(hashes) == 67 and kernel.abs().sum() > 0
    path = tmp_path/'kernel_000_neg.npz'
    with np.load(path) as data:
        values = dict(data)
    values['psf_raw'] = np.ones((1,51,51),np.float32)
    np.savez(path,**values)
    with pytest.raises(ValueError,match='Invalid PSF branch'):
        load_psfs(tmp_path,torch.from_numpy(w))
    path.unlink()
    with pytest.raises(FileNotFoundError):
        load_psfs(tmp_path,torch.from_numpy(w))
    config['target_rotated_180'] = False
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='geometry/wavelength/rotation'):
        load_psfs(tmp_path,torch.from_numpy(w))
