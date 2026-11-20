"""Two first-layer protocols, selected explicitly.

legacy_block_input (default; existing fitted commands and records keep it)
    Every logical input pixel is replicated into a 17x17 block U17, the block is
    convolved with the full PSF, the 17x17 block mean R17 is read out and an
    extra 1/289 is applied:
        Y_c = b_c + s_c/289 * R17[ g+ C(P+) - g- C(P-) ] evaluated on U17(X).
    The coarse coefficient mixes input replication and detector averaging, so the
    input window is smoothed twice.

pixel_psf_superposition_v1 (new; opt-in only)
    s = 17, c = 8, PSF origin o = (n//2, n//2):
        D(X)[s*i+c, s*j+c] = X[i, j], 0 on every other fine-grid point
        Z_c[t] = sum_ij X[i, j] * ( g+ P+[o + t - (s*i+c, s*j+c)]
                                  - g- P-[o + t - (s*i+c, s*j+c)] )
        Y_c = scale_c * AvgPool_s,stride_s(Z_c) + bias_c
    Each logical pixel excites one complete PSF and is shifted by its own
    isoplanatic position; nothing is replicated.  There is no extra 1/289: the
    single mean read-out divides by s**2 exactly once.

    Equivalent correlation kernel at kernel offset v (Conv2d index k = v + p):
        A[u] = 1/s**2 * sum_{a,b=0..s-1} Q[o_y + s*u_y + a - c, o_x + s*u_x + b - c]
    with Q = g+ P+ - g- P-, A[u] stored at v = -u (Conv2d correlates with the
    flipped kernel) and everything outside the saved PSF treated as zero.  The
    summation windows tile the plane as u varies, so every saved PSF sample --
    including the far periphery -- contributes exactly once; nothing is dropped,
    cropped, resized or peak-aligned.  D17 + mean R17 with this A is algebraically
    equal to U17 input + block-centre sampling + /289 at any fixed PSF (see
    SUMMARY.md); it is NOT equal to the legacy chain, which adds a second box sum.

Saved targets are already rotated 180 degrees.  Physical convolution becomes a
correlation only through the sign of u, never by peak alignment.

The model is linear intensity superposition: incoherent input, or per-pixel
time-multiplexed acquisition.  It does not assert coherence between pixels.
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
CENTER = FACTOR // 2
KERNEL_SIZES = (3, 5, 7)
LEGACY_OPERATOR = 'legacy_block_input'
PIXEL_OPERATOR = 'pixel_psf_superposition_v1'
OPERATORS = {
    LEGACY_OPERATOR:
        'U17 -> physical C(origin=n//2) -> mean R17 stride17 -> s/289 + bias',
    PIXEL_OPERATOR:
        'D17(centre=(8,8), step=17) -> one complete PSF per logical pixel, '
        'shifted and summed -> mean R17 stride17 -> * s + bias; the sole 1/17^2 '
        'lives inside the correlation kernel, with no extra /289 read-out factor',
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_operator(operator):
    if operator not in OPERATORS:
        raise ValueError(f'Unknown first-layer operator: {operator!r}')
    return OPERATORS[operator]


def operator_record(operator, n):
    """Provenance for one inference run; identical for ideal and fitted PSFs."""
    record = {'operator': check_operator(operator), 'operator_id': operator}
    if operator == PIXEL_OPERATOR:
        record.update({
            'pixel_step': FACTOR,
            'pixel_centre': [CENTER, CENTER],
            'psf_origin': [n // 2, n // 2],
            'fine_grid': 'D(X)[17i+8, 17j+8] = X[i,j], all other points 0',
            'readout': 'block mean R17 stride17 starting at (0,0)',
            'normalisation': 'scale/17**2 inside the correlation kernel; '
                             'no extra 1/289 read-out factor',
            'superposition': 'linear intensity; incoherent or per-pixel '
                             'time-multiplexed input, not coherent field addition',
        })
    return record


def effective_kernel(psf, origin):
    """Legacy: odd square correlation kernel with BOTH area factors."""
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


def superposition_kernel(psf, origin, *, step=FACTOR):
    """Pixel-protocol correlation kernel: one complete PSF per logical pixel.

    A[u] = 1/s**2 * sum_{a,b=0..s-1} P[o + s*u + (a,b) - c], see the module
    docstring.  Only the single mean read-out contributes the 1/s**2.

    `step` is the protocol factor s; 17 is the only production value and the
    keyword exists so a focused check can replay the same formula at s = 3.
    """
    p = np.asarray(psf, dtype=np.float64)
    if p.ndim != 2 or not np.isfinite(p).all():
        raise ValueError('PSF must be a finite plane')
    if any(o < 0 or o >= n for o, n in zip(origin, p.shape)):
        raise ValueError('Invalid origin')
    if not isinstance(step, int) or step < 1 or step % 2 == 0:
        raise ValueError('Protocol factor must be a positive odd integer')
    s, c = step, step // 2
    # Window start o-c-s*u must still overlap the array.  Windows that fall wholly
    # outside are clipped to an empty sum (they tile the plane and are all zero).
    reach = max((origin[0] - c + s - 1) // s, (origin[1] - c + s - 1) // s,
                (p.shape[0] - 1 - origin[0] + c) // s,
                (p.shape[1] - 1 - origin[1] + c) // s)
    offsets = np.arange(-reach, reach + 1)
    rows = origin[0] - c - s * offsets
    cols = origin[1] - c - s * offsets
    # Zero-padded width-s box sums, evaluated at every window start at once.
    csum_rows = np.concatenate([np.zeros((1, p.shape[1])), np.cumsum(p, axis=0)], axis=0)
    lo, hi = np.clip(rows, 0, p.shape[0]), np.clip(rows + s, 0, p.shape[0])
    band = csum_rows[hi] - csum_rows[lo]
    csum_cols = np.concatenate([np.zeros((band.shape[0], 1)), np.cumsum(band, axis=1)], axis=1)
    lo, hi = np.clip(cols, 0, p.shape[1]), np.clip(cols + s, 0, p.shape[1])
    return (csum_cols[:, hi] - csum_cols[:, lo]) / float(s * s)


def _kernel_fn(operator):
    check_operator(operator)
    return superposition_kernel if operator == PIXEL_OPERATOR else effective_kernel


def psf_canvas(directory):
    """Validated (n, origin) of one PSF artifact; origin is always (n//2, n//2)."""
    n = _validated_config(Path(directory))['n']
    return n, (n//2, n//2)


def _validated_config(root):
    config = json.loads((root/'config.json').read_text(encoding='utf-8'))
    n = config['n']
    k = config.get('first_kernel_size', 3)
    if (k not in KERNEL_SIZES or config.get('sample_cell', FACTOR) != FACTOR
            or config.get('patch_meta') != FACTOR*k or config.get('target_rotated_180') is not True
            or config.get('grayscale') is not True or config.get('wavelength_nm') not in (450, 532)
            or not isinstance(n, int) or n < FACTOR*k
            or config.get('center', [n//2]*2) != [n//2]*2
            or config.get('fft_size', 0) < n
            or (config['fft_size']-n) % 2
            or config.get('padding') != (config['fft_size']-n)//2):
        raise ValueError('Incompatible PSF geometry/wavelength/rotation config')
    return config


def load_psfs(directory, weight, *, operator=LEGACY_OPERATOR, ideal=False):
    """Load the signed first-layer kernel under an explicitly chosen protocol."""
    kernel_fn = _kernel_fn(operator)
    root = Path(directory)
    config = _validated_config(root)
    n = config['n']
    kernels = np.load(root/'electrical_kernels.npy', allow_pickle=False)
    expected = weight.detach().cpu().float().numpy()
    k = config.get('first_kernel_size', 3)
    patch = FACTOR*k
    if k != 3 and operator != PIXEL_OPERATOR:
        raise ValueError('Enlarged first layers require pixel PSF superposition')
    if kernels.shape != (32, 1, k, k) or not np.isfinite(kernels).all() or not np.array_equal(kernels, expected):
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
            start = n//2-patch//2
            target[start:start+patch, start:start+patch] = component.repeat(FACTOR,0).repeat(FACTOR,1)
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
                # Ideal control uses the exact replicated electrical target and unit gain.
                branch = polarity*(1.0 if ideal else gain.item())*kernel_fn(target if ideal else raw[0], (n//2,n//2))
            signed = branch if signed is None else signed+branch
            files.append(path)
        output.append(signed*scales[channel])
    return torch.from_numpy(np.stack(output)[:,None]).float(), {str(p.name): sha256(p) for p in files}


class FirstConvHook:
    """No registered parameters/buffers: Predictor state_dict reload stays intact."""
    def __init__(self, network, kernel):
        conv = network.get_submodule(KEY)
        if (not isinstance(conv, nn.Conv2d) or conv.in_channels != 1 or conv.out_channels != 32
                or conv.kernel_size[0] not in KERNEL_SIZES
                or conv.kernel_size[0] != conv.kernel_size[1]
                or conv.stride != (1,1) or conv.padding != (conv.kernel_size[0]//2,)*2 or conv.dilation != (1,1)
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
