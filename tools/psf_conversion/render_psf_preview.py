"""Preview saved 450/532 nm raw PSFs, normalized shapes and optimization histories."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    source, out = Path(args.results).resolve(), Path(args.out).resolve()
    tool = Path(__file__).resolve().parent
    if out == source or source in out.parents or out == tool or tool in out.parents:
        parser.error('Output must be fresh and outside source results and tool directory')
    config = json.loads((source / 'config.json').read_text(encoding='utf-8'))
    summary = json.loads((source / 'summary.json').read_text(encoding='utf-8'))
    wavelength = config['wavelength_nm']
    n, patch, dx = int(config['n']), int(config['patch_meta']), float(config['dx_nm']) / 1000
    if not (1 <= patch <= n and np.isfinite(dx) and dx > 0 and config['wavelength_nm'] in (450, 532)):
        raise ValueError('Invalid grid/support/wavelength')
    electrical = np.load(source / 'electrical_kernels.npy', allow_pickle=False, mmap_mode='r')
    if electrical.ndim != 4 or electrical.shape[1:] != (1, 3, 3):
        raise ValueError('This preview requires saved electrical kernels [outputs,1,3,3]')
    records = {}
    for row in summary:
        key = (int(row['index']), row['sign'])
        if key in records or not 0 <= key[0] < len(electrical) or key[1] not in ('pos', 'neg'):
            raise ValueError('Duplicate or invalid summary identity')
        records[key] = row
    actual = set(source.glob('kernel_*_*.npz'))
    expected = {source / f'kernel_{i:03d}_{s}.npz' for i, s in records}
    if actual != expected or not records:
        raise ValueError(f'Incomplete/conflicting artifacts: missing={expected-actual}, unlisted={actual-expected}')
    out.mkdir(parents=True, exist_ok=False)
    start = n // 2 - patch // 2
    support = np.s_[start:start+patch, start:start+patch]
    # Sample i is located at (i - n//2)*dx, matching the original target indices.
    def extent(lo, hi):
        return [(lo-n//2-.5)*dx, (hi-n//2-.5)*dx] * 2
    bounds = extent(start, start+patch)
    # Match nearest-neighbor expansion of the saved 3x3 physical target.
    edges = [bounds[0] + math.ceil(k*patch/3)*dx for k in (1, 2)]
    def grid(ax):
        for edge in edges:
            ax.axvline(edge, color='cyan', linewidth=.6, alpha=.65)
            ax.axhline(edge, color='cyan', linewidth=.6, alpha=.65)
    def save(fig, filename):
        fig.savefig(out / filename, dpi=150)
        plt.close(fig)
    index, thumbnails = [], []
    definition = 'Each full n x n array divided by its own L2 norm before center crop; no camera integration, alignment or refitting'
    for kernel in sorted({i for i, _ in records}):
        branches = []
        for sign in ('pos', 'neg'):
            if (kernel, sign) not in records:
                continue
            row = records[kernel, sign]
            path = source / f'kernel_{kernel:03d}_{sign}.npz'
            with np.load(path, allow_pickle=False) as data:
                raw, target = data['psf_raw'], data['target']
                active, width, history = data['active_gray'], data['width_nm'], data['history']
                skipped = bool(row['skipped'])
                if (raw.shape != (1,n,n) or target.shape != raw.shape
                    or not np.isfinite(raw).all() or not np.isfinite(target).all()
                    or (raw < 0).any() or (target < 0).any()
                    or active.shape != (1,) or bool(active[0]) == skipped
                    or bool(target.any()) == skipped
                    or history.ndim != 1 or len(history) != row['steps']
                    or not np.isfinite(history).all() or (history < 0).any()
                    or data['wavelength_nm'].item() != config['wavelength_nm']
                    or (skipped and (width.shape != (0,0) or len(history) or raw.any() or row['best_loss'] is not None))
                    or (not skipped and (width.shape != (n,n) or not np.isfinite(width).all()
                        or (width < 80).any() or (width > 320).any() or not len(history)
                        or not raw.any() or not np.isclose(min(history), row['best_loss'])))):
                    raise ValueError(f'Invalid/conflicting artifact: {path}')
                raw = raw[0]
                target = target[0]
                raw_norm = float(np.linalg.norm(raw.astype(np.float64)))
                target_norm = float(np.linalg.norm(target.astype(np.float64)))
                shape = raw[support].astype(np.float64) / raw_norm if raw_norm else np.zeros((patch,patch))
                goal = target[support].astype(np.float64) / target_norm if target_norm else np.zeros((patch,patch))
                total = float(raw.sum(dtype=np.float64))
                ref = dict(kernel=kernel, sign=sign, source_npz=str(path), source_sha256=sha256(path),
                           steps=row['steps'], skipped=skipped, wavelength_nm=wavelength)
                branches.append(dict(ref=ref, raw=raw, shape=shape, goal=goal, history=history,
                                     captured_window_center_fraction=float(raw[support].sum(dtype=np.float64))/total if total else None))
        vmax = max(float(b['raw'].max()) for b in branches) or 1.0
        for b in branches:
            ref, raw = b['ref'], b['raw']
            sign = ref['sign']
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained')
            label = 'POSITIVE / POS' if sign == 'pos' else 'NEGATIVE / NEG'
            title = f'Kernel {kernel:03d} | {label} | {wavelength:g} nm | steps={ref["steps"]}'
            if ref['skipped']:
                title += '\nSKIPPED ZERO BRANCH - no optimized metasurface'
            fig.suptitle(title)
            for ax, values, region, subtitle in (
                (axes[0], raw, extent(0,n), 'Full simulation plane'),
                (axes[1], raw[support], bounds, 'Zoom: original target support')):
                im = ax.imshow(values, origin='lower', extent=region, interpolation='nearest', vmin=0, vmax=vmax, cmap='inferno')
                ax.set(title=subtitle, xlabel='x (um)', ylabel='y (um)')
                fig.colorbar(im, ax=ax, label='raw intensity / a.u.')
            fig.supxlabel('Saved psf_raw[0]; linear shared POS/NEG scale; simulated intensity')
            png = f'kernel_{kernel:03d}_{sign}_psf.png'
            save(fig, png)
            index.append(dict(ref, kind='raw', png=png, raw_description='psf_raw[0], unchanged; simulated, not measured power',
                              vmin=0, vmax=vmax, full_extent_um=extent(0,n), zoom_extent_um=bounds,
                              captured_window_center_fraction=b['captured_window_center_fraction']))
            thumbnails.append((ref, b['shape']))
        refs = [b['ref'] for b in branches]
        vmax_shape = max(float(b[key].max()) for b in branches for key in ('goal','shape')) or 1.0
        vmax_diff = max(float(np.abs(b['shape']-b['goal']).max()) for b in branches) or 1.0
        fig, axes = plt.subplots(len(branches), 3, figsize=(13, 4.2*len(branches)), squeeze=False, layout='constrained')
        fig.suptitle(f'Kernel {kernel:03d} | {wavelength:g} nm | center {patch} x {patch} samples | physical target (already rotated)')
        for row_axes, b in zip(axes, branches):
            ref = b['ref']
            for j, (ax, values, title) in enumerate(zip(row_axes, (b['goal'],b['shape'],b['shape']-b['goal']), ('Target','Saved PSF','PSF - target'))):
                im = ax.imshow(values, origin='lower', extent=bounds, interpolation='nearest',
                               cmap='RdBu_r' if j==2 else 'inferno', vmin=-vmax_diff if j==2 else 0, vmax=vmax_diff if j==2 else vmax_shape)
                grid(ax)
                ax.set(title=f'{ref["sign"].upper()} | {title} | {ref["steps"]} steps', xlabel='x (um)', ylabel='y (um)')
                if ref['skipped']:
                    ax.text(.5,.5,'SKIPPED ZERO BRANCH',transform=ax.transAxes,ha='center',color='lime',fontsize=10)
                fig.colorbar(im, ax=ax, label='L2 shape difference' if j==2 else 'L2-normalized shape')
        fig.supxlabel('Shape comparison: each FULL array L2-normalized; no camera integration or refitting')
        png = f'kernel_{kernel:03d}_comparison.png'
        save(fig,png)
        index.append(dict(kind='comparison',kernel=kernel,sources=refs,png=png,definition=definition,vmin=0,vmax=vmax_shape,difference_vmin=-vmax_diff,difference_vmax=vmax_diff,zoom_extent_um=bounds))
        fig, axes = plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
        positive = all((b['history'] > 0).all() for b in branches)
        for b in branches:
            ref, h = b['ref'], b['history']
            label = ref['sign'].upper()+f' ({len(h)} steps)'
            if not len(h):
                for ax in axes: ax.plot([],[],label=label+' SKIPPED ZERO BRANCH')
                continue
            steps = np.arange(1,len(h)+1)
            axes[0].plot(steps,h,label=label)
            axes[1].plot(steps[-200:],h[-200:],label=label)
        if positive and any(len(b['history']) for b in branches):axes[0].set_yscale('log')
        for ax,title in zip(axes,('Complete saved history','Last up to 200 saved steps (linear)')):
            ax.set(title=title,xlabel='Optimization step',ylabel='Saved L2 shape loss')
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
        fig.suptitle(f'Kernel {kernel:03d} | {wavelength:g} nm | POS / NEG loss histories')
        png=f'kernel_{kernel:03d}_loss.png'
        limits=[dict(xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim()),yscale=ax.get_yscale()) for ax in axes]
        save(fig,png)
        index.append(dict(kind='history',kernel=kernel,sources=refs,png=png,definition='Saved history values, no interpolation; full and last up to 200 steps',axes=limits,vmin=min(a['ylim'][0] for a in limits),vmax=max(a['ylim'][1] for a in limits)))
    columns = min(8,len(thumbnails))
    rows = math.ceil(len(thumbnails)/columns)
    fig, axes = plt.subplots(rows,columns,figsize=(max(12,2.6*columns),max(4.8,2.7*rows+.7)),squeeze=False,layout='constrained')
    maximum = max(float(shape.max()) for _,shape in thumbnails) or 1.0
    for ax, (ref, shape) in zip(axes.flat,thumbnails):
        im=ax.imshow(shape,origin='lower',extent=bounds,interpolation='nearest',cmap='inferno',vmin=0,vmax=maximum)
        ax.set_title(f'{ref["kernel"]:03d} {ref["sign"].upper()} | {ref["steps"]} steps',fontsize=9)
        ax.tick_params(labelsize=6)
        if ref['skipped']:ax.text(.5,.5,'SKIPPED\nZERO BRANCH',transform=ax.transAxes,ha='center',color='lime',fontsize=9)
    for ax in list(axes.flat)[len(thumbnails):]:ax.set_visible(False)
    fig.colorbar(im,ax=list(axes.flat)[:len(thumbnails)],label='Full-array L2-normalized PSF shape',shrink=.8)
    fig.suptitle(f'{wavelength:g} nm | {len(thumbnails)} saved branches | center crop / L2-normalized shape montage\nNot raw power or one physical combined PSF',fontsize=13)
    fig.supxlabel('x / y in um; unchanged physical orientation; POS / NEG adjacent')
    png='psf_shape_montage.png'
    save(fig,png)
    index.append(dict(kind='montage',sources=[ref for ref,_ in thumbnails],png=png,definition=definition+'; not raw power or one physical combined PSF',vmin=0,vmax=maximum,rows=rows,columns=columns,zoom_extent_um=bounds))
    (out / 'index.json').write_text(json.dumps(index, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Rendered {len(thumbnails)} saved branches and {len(index)} indexed figures: {out}')


if __name__ == '__main__':
    main()


