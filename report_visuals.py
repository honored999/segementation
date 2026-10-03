"""Small shared diagnostic display helpers; no inference or model changes."""
from pathlib import Path
import textwrap

DPI = 100
BODY = 14
TITLE = 22
DISPLAY_NOTE = ('Independent 1-99% display scale per layer/window; absolute strength is not comparable. '
                'Native grids / nearest; metadata and invalid counts: report.txt')


def setup_font():
    import matplotlib as mpl
    from matplotlib import font_manager
    available={f.name for f in font_manager.fontManager.ttflist}
    font=next((f for f in ('Microsoft YaHei','SimHei','Noto Sans CJK SC') if f in available), None)
    if font is None:
        raise ValueError('report requires an installed CJK font; no dependency is installed automatically')
    mpl.rcParams.update({'font.family':font,'axes.unicode_minus':False})
    return font


def wrap(value, width=55):
    return '\n'.join(part for line in str(value).split('\n') for part in (textwrap.wrap(line,width=width,break_long_words=True) or ['']))


def normalize_intensity(values):
    """Finite-only percentiles, invalid pixels -> zero, stable extreme ranges."""
    import numpy as np
    a=np.asarray(values,dtype=np.float64)
    if a.ndim!=2: raise ValueError('diagnostic display expects a two-dimensional grid')
    finite=np.isfinite(a); count=int(a.size-np.count_nonzero(finite))
    result=np.zeros(a.shape,dtype=np.float32)
    meta=dict(normalization='finite-only 1st-99th percentiles; invalid pixels zero',
              nonfinite_count=count,finite_count=int(np.count_nonzero(finite)))
    if not finite.any(): return result,dict(meta,status='no_finite_values')
    v=a[finite]
    # Scale before percentile interpolation/subtraction to avoid overflow at +/-1e308.
    scale=max(float(np.max(np.abs(v))),1.)
    v=v/scale
    lo,hi=np.percentile(v,(1,99))
    meta.update(percentiles_scaled=[float(lo),float(hi)],value_scale=scale)
    if hi<=lo: return result,dict(meta,status='constant')
    result[finite]=np.clip((v-lo)/(hi-lo),0,1)
    return result,dict(meta,status='finite_with_invalid' if count else 'finite')


def dice_label(value):
    from report_metrics_table import metric_value
    number = metric_value(value)
    return 'N/A' if number is None else f'{number:.3f}'


def stage_sample(value, stage, module, input_index=None):
    """Reduce immediately; the returned object owns no tensor/graph."""
    # Real fp16/bfloat16/fp32/fp64, including finite dtype extremes. Scale
    # fp64 pixels before reduction: normalized channels are in [0, 1], so
    # neither a channel sum nor restoring the scale can overflow. No fp64
    # downcast. Own one temporary fp64 BCHW buffer, then retain only 2D.
    import torch
    with torch.no_grad():
        absolute=value.detach().to(dtype=torch.float64,copy=True).abs_()
        if value.dtype == torch.float64:
            scale=absolute.amax(1,keepdim=True)
            absolute.div_(torch.where(scale > 0,scale,torch.ones_like(scale)))
            reduced=absolute.mean(1).clamp(max=1)*scale[:,0]
        else:
            # Even fp32.max * int64.max is far below fp64.max. Direct
            # fp64 mean avoids unnecessary scaling roundoff for lower dtypes.
            reduced=absolute.mean(1)
        magnitude=reduced[0].cpu().numpy().copy()
    _,display=normalize_intensity(magnitude)
    return dict(stage=stage,module=module,input_index=input_index,
                feature_shape=list(value.shape),aggregation='mean(abs(channel))',
                display=display,magnitude=magnitude,
                feature_nonfinite_count=int((~value.detach().isfinite()).sum().item()))


def metadata_only(value):
    if isinstance(value,dict):
        return {k:metadata_only(v) for k,v in value.items() if k not in {'channels','magnitude','input'}}
    if isinstance(value,tuple): return tuple(metadata_only(v) for v in value)
    if isinstance(value,list): return [metadata_only(v) for v in value]
    return value


def save_figure(fig, output, *, family, coverage=None, kind=None):
    """Save only the requested main PNG, without implicit previews."""
    output = Path(output)
    if 'ppt' in output.parts:
        raise ValueError('PPT output is disabled')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.set_dpi(DPI)
    fig.canvas.draw()
    fig.savefig(output, dpi=DPI, facecolor=fig.get_facecolor())


def validate_visual_outputs(output, required_families):
    from PIL import Image
    output = Path(output)
    if (output/'ppt').exists() or (output/'layout_manifest.json').exists():
        raise ValueError('report must not contain PPT output')
    for family in required_families:
        name = 'encoder_stages_heatmap' if family == 'encoder_stages' else family
        path = output/(name + '.png')
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError('report output incomplete: ' + name)
        with Image.open(path) as image:
            image.verify()


def stage_figure(items, rows, output):
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    setup_font(); output=Path(output)
    packets=[i.get('native',{}).get('encoder_stages') if i.get('native') else None for i in items]
    count=max((len(p['stages']) for p in packets if p),default=0)
    def draw(subset, indices, path):
        fig=plt.figure(figsize=(max(18,3*(len(indices)+1)),3.0*len(subset)+2.8))
        grid=fig.add_gridspec(3*len(subset),len(indices)+1,left=.055,right=.95,
            top=.83,bottom=.18,height_ratios=[r for _ in subset for r in (.48,.62,1)],hspace=.18,wspace=.16)
        fig.suptitle('各级编码器特征强度',fontsize=TITLE,y=.975)
        fig.text(.5,.895,'mean(abs(channel)) diagnostic display | encoder stage numbers are 0-based',ha='center',fontsize=BODY)
        for j,item in enumerate(subset):
            header=fig.add_subplot(grid[3*j,:]); header.axis('off')
            native=item.get('native'); packet=native.get('encoder_stages') if native else None
            label=f"{item['cid']} | {item.get('group','?')} | current Dice {dice_label(rows[item['cid']]['dice'])}"
            label+=f" | original slice {native['original_slice']} | window {native['window_index']}" if native else ' | No GT-positive slice / no native window'
            header.text(0,.5,wrap(label,130),fontsize=BODY,va='center')
            for col,index in enumerate([None,*indices]):
                label_ax=fig.add_subplot(grid[3*j+1,col]); label_ax.axis('off')
                ax=fig.add_subplot(grid[3*j+2,col]); ax.set_xticks([]); ax.set_yticks([])
                if packet is None or index is not None and index>=len(packet['stages']):
                    ax.axis('off'); ax.text(.5,.5,'No native window',ha='center',fontsize=BODY); continue
                if index is None:
                    values=packet['input']; title='Model input window\n'+str(tuple(values.shape))
                    cmap='gray'
                else:
                    s=packet['stages'][index]; values=s['magnitude']; shape=s['feature_shape'][1:]
                    title=f"stage {s['stage']} | {'×'.join(map(str,shape))}\n{s['module']}"
                    if s.get('input_index') is not None: title+=f" input[{s['input_index']}]"
                    if s.get('feature_index') is not None: title+=f"[{s['feature_index']}]"
                    cmap='inferno'
                normalized,meta=normalize_intensity(values)
                ax.imshow(normalized,cmap=cmap,vmin=0,vmax=1,interpolation='nearest')
                label_ax.text(.5,.5,wrap(title,32),ha='center',va='center',fontsize=BODY)
                if meta['nonfinite_count']: ax.set_xlabel(f"invalid={meta['nonfinite_count']} -> zero",fontsize=BODY)
        cax=fig.add_axes((.33,.105,.34,.025)); bar=fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap='inferno'),cax=cax,orientation='horizontal')
        bar.ax.tick_params(labelsize=BODY)
        fig.text(.5,.025,wrap(DISPLAY_NOTE,125),ha='center',fontsize=BODY)
        try: save_figure(fig,path,family='encoder_stages',coverage={'cases':[i['cid'] for i in subset],'stages':indices})
        finally: plt.close(fig)
    draw(items,list(range(count)),output)
