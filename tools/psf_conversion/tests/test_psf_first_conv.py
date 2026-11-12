"""Small numerical checks only; no medical data or full random network."""
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from tools.psf_conversion.psf_first_conv import effective_kernel, FirstConvHook, KEY, load_psfs


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
    import json
    (tmp_path/'config.json').write_text(json.dumps(dict(n=51, patch_meta=51, target_rotated_180=True, grayscale=True, wavelength_nm=450,fft_size=153,padding=51)))
    np.save(tmp_path/'electrical_kernels.npy',np.ones((32,1,3,3),np.float32))
    with pytest.raises(ValueError,match='Checkpoint'):
        load_psfs(tmp_path,torch.zeros(32,1,3,3))


def test_artifact_loading_and_rejections(tmp_path):
    import json
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
