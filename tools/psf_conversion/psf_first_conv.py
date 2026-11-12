"""Fixed U17 -> physical linear PSF convolution -> mean R17 -> scale/289.

The coarse correlation coefficient at offset d is
s/17**4 * sum_{a,b=0..16} P[o+a-b-17*d].  The triangular
sum includes both replication and detector averaging; all PSF samples survive.
Saved targets are already rotated. Physical convolution is converted to
correlation only here, through the minus sign on d (never by peak alignment).
"""
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

KEY = 'encoder.stages.0.0.convs.0.conv'
FACTOR = 17


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def effective_kernel(psf, origin):
    """Return odd square correlation kernel, including BOTH area factors."""
    p = np.asarray(psf, dtype=np.float64)
    if p.ndim != 2 or not np.isfinite(p).all():
        raise ValueError('PSF must be a finite plane')
    if any(o < 0 or o >= n for o, n in zip(origin, p.shape)):
        raise ValueError('Invalid origin')
    radius = math.ceil((max(*origin, *(n-1-o for n, o in zip(p.shape, origin))) + 16) / 17)
    # Two full box convolutions per axis give the separable triangular weights.
    for axis in (0, 1):
        for _ in range(2):
            pad = [(0, 0), (0, 0)]
            pad[axis] = (17, 16)
            cs = np.cumsum(np.pad(p, pad), axis=axis)
            hi = [slice(None), slice(None)]
            lo = hi.copy()
            hi[axis], lo[axis] = slice(17, None), slice(None, -17)
            p = cs[tuple(hi)] - cs[tuple(lo)]
    offsets = np.arange(-radius, radius+1)
    yy, xx = np.meshgrid(origin[0]+16-17*offsets, origin[1]+16-17*offsets, indexing='ij')
    valid = (yy >= 0) & (yy < p.shape[0]) & (xx >= 0) & (xx < p.shape[1])
    result = np.zeros(yy.shape, dtype=np.float64)
    result[valid] = p[yy[valid], xx[valid]] / 17**4
    return result


def load_psfs(directory, weight):
    root = Path(directory)
    config = json.loads((root/'config.json').read_text(encoding='utf-8'))
    n = config['n']
    if (config.get('patch_meta') != 51 or config.get('target_rotated_180') is not True
            or config.get('grayscale') is not True or config.get('wavelength_nm') not in (450, 532)
            or not isinstance(n, int) or n < 51
            or config.get('fft_size', 0) < n
            or (config['fft_size']-n) % 2
            or config.get('padding') != (config['fft_size']-n)//2):
        raise ValueError('Incompatible PSF geometry/wavelength/rotation config')
    kernels = np.load(root/'electrical_kernels.npy', allow_pickle=False)
    expected = weight.detach().cpu().float().numpy()
    if kernels.shape != (32, 1, 3, 3) or not np.isfinite(kernels).all() or not np.array_equal(kernels, expected):
        raise ValueError('Checkpoint first layer differs from electrical_kernels')
    scales = np.load(root/'shared_scales.npy', allow_pickle=False)
    expected_scales = np.maximum(np.abs(kernels).max(axis=(1,2,3)), 1e-12)
    if scales.shape != (32,) or not np.isfinite(scales).all() or not np.array_equal(scales, expected_scales):
        raise ValueError('Invalid shared scales')
    files = [root/'config.json', root/'electrical_kernels.npy', root/'shared_scales.npy']
    output = []
    for channel in range(32):
        signed = None
        for sign, polarity in [('pos', 1), ('neg', -1)]:
            path = root/f'kernel_{channel:03d}_{sign}.npz'
            component = np.maximum(polarity*kernels[channel, 0, ::-1, ::-1]/scales[channel], 0)
            target = np.zeros((n,n), dtype=np.float32)
            start = n//2-25
            target[start:start+51, start:start+51] = component.repeat(17,0).repeat(17,1)
            with np.load(path, allow_pickle=False) as data:
                raw, gain = data['psf_raw'], data['gain_gray']
                active = bool(component.any())
                if (raw.shape != (1,n,n) or gain.shape != (1,) or
                        data['target'].shape != (1,n,n) or
                        not np.array_equal(data['target'][0], target) or
                        data['active_gray'].shape != (1,) or bool(data['active_gray'][0]) != active or
                        data['wavelength_nm'].item() != config['wavelength_nm'] or
                        not np.isfinite(raw).all() or not np.isfinite(gain).all() or
                        (raw < 0).any() or (gain < 0).any() or
                        (not active and (raw.any() or gain.any()))):
                    raise ValueError(f'Invalid PSF branch: {path.name}')
                branch = polarity*gain.item()*effective_kernel(raw[0], (n//2,n//2))
            signed = branch if signed is None else signed+branch
            files.append(path)
        output.append(signed*scales[channel])
    return torch.from_numpy(np.stack(output)[:,None]).float(), {str(p.name): sha256(p) for p in files}


class FirstConvHook:
    """No registered parameters/buffers: Predictor state_dict reload stays intact."""
    def __init__(self, network, kernel):
        conv = network.get_submodule(KEY)
        if (not isinstance(conv, nn.Conv2d) or tuple(conv.weight.shape) != (32,1,3,3)
                or conv.stride != (1,1) or conv.padding != (1,1) or conv.dilation != (1,1)
                or conv.groups != 1 or conv.padding_mode != 'zeros'):
            raise ValueError('Unsupported first convolution')
        if kernel.ndim != 4 or kernel.shape[:2] != (32,1) or kernel.shape[2] != kernel.shape[3] or kernel.shape[2]%2 != 1 or not torch.isfinite(kernel).all():
            raise ValueError('Invalid effective kernel')
        self.kernel = kernel.detach()
        self.calls = 0
        network.requires_grad_(False)
        def replace(module, args, original):
            self.calls += 1
            x = args[0]
            # Disable autocast to keep the convolution and bias addition in input dtype.
            with torch.autocast(device_type=x.device.type, enabled=False):
                y = F.conv2d(x, self.kernel.to(device=x.device, dtype=x.dtype), padding=self.kernel.shape[-1]//2)
                if module.bias is not None:
                    y = y + module.bias.to(dtype=x.dtype)[None,:,None,None]
            return y.to(original.dtype)
        self.handle = conv.register_forward_hook(replace)

    def remove(self):
        self.handle.remove()
