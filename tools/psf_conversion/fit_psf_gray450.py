"""Standalone grayscale kernel-to-PSF fitting at 450 nm.

Input kernels: exactly [32, 1, 3, 3], validated before --limit.
Adapted from the preserved fit_psf.py; see README.md for provenance and limits.
The polynomial phase library is specific to the original P400/H600 geometry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from optics.fitted_phase_parameters_p400 import compute_phase_fitted
from optics.propagation import create_propagation_layer


def load_kernels(path, key):
    path = Path(path)
    if path.suffix == '.npy':
        kernels = torch.from_numpy(np.load(path, allow_pickle=False))
    else:
        state = torch.load(path, map_location='cpu', weights_only=True)
        if isinstance(state, dict):
            for wrapper in ('model_state_dict', 'state_dict'):
                if wrapper in state:
                    state = state[wrapper]
                    break
            if key not in state:
                raise KeyError(f'{key!r} absent; available keys: {list(state)[:30]}')
            kernels = state[key]
        else:
            kernels = state
    kernels = torch.as_tensor(kernels).detach().float().cpu()
    return validate_kernels(kernels)


def validate_kernels(kernels):
    if tuple(kernels.shape) != (32, 1, 3, 3):
        raise ValueError('Expected exactly [32, 1, 3, 3] grayscale kernels before --limit')
    if not torch.isfinite(kernels).all():
        raise ValueError('Kernels contain NaN/Inf')
    return kernels


def build_target(kernel, n, patch):
    """Match the original nearest-neighbor target construction on the meta grid."""
    if not 1 <= patch <= n:
        raise ValueError(f'Target patch {patch} must fit simulation grid {n}')
    up = F.interpolate(kernel[None], size=(patch, patch), mode='nearest')[0]
    target = kernel.new_zeros(1, n, n)
    start = n // 2 - patch // 2
    target[:, start:start + patch, start:start + patch] = up
    return target


class Metasurface(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.raw_width = nn.Parameter(torch.empty(args.n, args.n, device=args.device).uniform_(-1, 1))
        self.propagate = create_propagation_layer(
            distance=args.distance_um * 1e-6,
            wavelength=450e-9,
            pixel_size=args.dx_nm * 1e-9, device=args.device, padding=args.n,
        )

    def width(self):
        return 80 + 240 * self.raw_width.sigmoid()

    def forward(self):
        # Evaluate the high-order fitted polynomial in float64 to reduce cancellation.
        width = self.width().double()
        phase = compute_phase_fitted(width, 450)[None]
        field = torch.exp(1j * phase).to(torch.complex64)[None]
        return self.propagate(field)[0].abs().square()


def shape_loss(psf, target, active, mode):
    p = psf[active].flatten(1)
    t = target[active].flatten(1)
    if mode != 'l2_shape':
        raise ValueError('Only l2_shape is supported for grayscale fitting')
    p = p / p.norm(dim=1, keepdim=True).clamp_min(1e-12)
    t = t / t.norm(dim=1, keepdim=True).clamp_min(1e-12)
    s = (p - t).square().sum(1)
    return s.square().mean().clamp_min(1e-24).sqrt()


def fit_one(target, args):
    active = target.flatten(1).sum(1) > 0
    if not active.any():
        return None, torch.zeros_like(target), [], active
    model = Metasurface(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best, best_width, stale = float('inf'), None, 0
    history = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        psf = model()
        loss = shape_loss(psf, target, active, args.loss)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite loss')
        value = loss.item()
        history.append(value)
        # Snapshot before optimizer.step: loss and saved parameters refer to same state.
        if value < best:
            best, stale = value, 0
            best_width = model.raw_width.detach().clone()
        else:
            stale += 1
        loss.backward()
        if model.raw_width.grad is None or not torch.isfinite(model.raw_width.grad).all():
            raise RuntimeError('Missing or non-finite width gradients')
        optimizer.step()
        if step == 0 or (step + 1) % 50 == 0:
            print(f'  step={step + 1} loss={value:.6g}', flush=True)
        if stale >= args.patience:
            break
    with torch.no_grad():
        model.raw_width.copy_(best_width)
        return model.width().cpu(), model().cpu(), history, active.cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernels', help='.npy, tensor .pt, or state_dict .pth')
    parser.add_argument('--key', default='optical_kernels')
    parser.add_argument('--demo', action='store_true', help='Synthetic [32,1,3,3] kernels; NOT trained task weights')
    parser.add_argument('--out', default='results')
    parser.add_argument('--limit', type=int, help='Fit first N output channels')
    parser.add_argument('--n', type=int, default=800)
    parser.add_argument('--dx-nm', type=float, default=400)
    parser.add_argument('--camera-pixel-um', type=float, default=3.45)
    parser.add_argument('--patch-camera', type=float, default=6)
    parser.add_argument('--distance-um', type=float, default=2400)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--loss', choices=['l2_shape'], default='l2_shape')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if bool(args.kernels) == args.demo:
        parser.error('Choose exactly one of --kernels FILE or --demo')
    if min(args.n, args.steps, args.patience, args.camera_pixel_um,
           args.patch_camera, args.distance_um, args.lr) <= 0:
        parser.error('Grid, steps, patience, physical dimensions and learning rate must be positive')
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be positive')
    if args.dx_nm != 400:
        parser.error('Bundled P400 phase library requires --dx-nm 400')
    torch.manual_seed(args.seed)
    kernels = (validate_kernels(torch.randn(32, 1, 3, 3)) if args.demo else load_kernels(args.kernels, args.key))
    if args.limit:
        kernels = kernels[:args.limit]
    # PyTorch conv2d is correlation; an incoherent optical PSF performs convolution.
    # Rotate the target by 180 degrees so physical PSF * image matches conv2d(image,K).
    physical_kernels = kernels.flip((-2, -1))
    # Preserve original shared pos/neg scale per output channel.
    scales = physical_kernels.abs().flatten(1).amax(1).clamp_min(1e-12)
    normalized = physical_kernels / scales[:, None, None, None]
    patch = int(args.patch_camera * args.camera_pixel_um * 1000 / args.dx_nm)
    if not 1 <= patch <= args.n:
        parser.error(f'Target patch {patch} exceeds grid {args.n}; increase --n or reduce --patch-camera')
    out = Path(args.out).resolve()
    source_dir = Path(__file__).resolve().parent
    if out == source_dir or source_dir in out.parents:
        parser.error('Choose an output directory outside the PSF tool source directory')
    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error('Output already exists; choose a fresh --out directory')
    np.save(out / 'electrical_kernels.npy', kernels.numpy())
    np.save(out / 'shared_scales.npy', scales.numpy())
    config = dict(vars(args), patch_meta=patch, grayscale=True, wavelength_nm=450,
                  target_rotated_180=True, zero_channels='ignored; must be blocked or subtracted experimentally',
                  phase_library='P400 H600 W80-320; phase-only fitted approximation')
    (out / 'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    summary = []
    for index, kernel in enumerate(normalized):
        for sign, component in [('pos', kernel.clamp_min(0)), ('neg', (-kernel).clamp_min(0))]:
            print(f'kernel={index} sign={sign}', flush=True)
            target = build_target(component, args.n, patch).to(args.device)
            width, psf, history, active = fit_one(target, args)
            target_cpu = target.cpu()
            # Least-squares gain for the grayscale channel: target ~= gain * raw PSF.
            # This scale is for numerical comparison, not guaranteed detector calibration.
            gains = ((psf * target_cpu).flatten(1).sum(1) /
                     psf.square().flatten(1).sum(1).clamp_min(1e-12))
            np.savez_compressed(out / f'kernel_{index:03d}_{sign}.npz',
                width_nm=np.empty((0, 0)) if width is None else width.numpy(),
                psf_raw=psf.numpy(), target=target_cpu.numpy(),
                active_gray=active.cpu().numpy(), gain_gray=gains.numpy(), wavelength_nm=np.array(450), history=np.array(history))
            summary.append(dict(index=index, sign=sign, skipped=not bool(active.any()),
                                best_loss=min(history) if history else None, steps=len(history)))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Saved: {out.resolve()}')


if __name__ == '__main__':
    main()
