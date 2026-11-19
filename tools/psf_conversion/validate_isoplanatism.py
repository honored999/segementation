"""Independent tilted-plane-wave validation; never optimizes a width map."""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import torch
import isoplanatism_core as c


def resource_preflight():
    import psutil
    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory()
    record = dict(cpu_percent=cpu, ram_percent=ram.percent,
                  ram_available_bytes=ram.available, time_unix=time.time(), gpus=[])
    try:
        proc = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
                               '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
        for line in proc.stdout.strip().splitlines():
            index, util, used, total = map(float, line.split(','))
            record['gpus'].append(dict(physical_index=int(index), util_percent=util, used_mib=used, total_mib=total,
                                       memory_percent=100*used/total))
    except (OSError, subprocess.CalledProcessError) as exc:
        record['gpu_query_unavailable'] = str(exc)
    record['all_gpus'] = list(record['gpus'])
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    record['cuda_visible_devices'] = visible
    if visible is not None:
        selected = {token.strip() for token in visible.split(',') if token.strip()}
        record['gpus'] = [gpu for gpu in record['all_gpus'] if str(gpu['physical_index']) in selected]
    return record


def guard(record, n, paddings, device, bytes_per_px=512):
    # Conservative allocation budget including FFT, diagnostics and plotting arrays.
    # Callers that keep a single window alive may pass their own measured factor.
    estimate = (n+2*max(paddings))**2 * bytes_per_px
    if record['cpu_percent'] >= 80 or record['ram_percent'] >= 80:
        raise c.ValidationError('DEFERRED_RESOURCE_GUARD: CPU/RAM occupied')
    if estimate > record['ram_available_bytes']*.25:
        raise c.ValidationError('DEFERRED_RESOURCE_GUARD: insufficient allocation headroom')
    for gpu in record['gpus']:
        if max(gpu['util_percent'], gpu['memory_percent']) >= 80:
            raise c.ValidationError('DEFERRED_RESOURCE_GUARD: GPU occupied; use tiny CPU tests later')
        if device.startswith('cuda') and gpu['used_mib']*2**20+estimate >= .8*gpu['total_mib']*2**20:
            raise c.ValidationError('DEFERRED_RESOURCE_GUARD: projected VRAM budget')
    if device.startswith('cuda') and not record['gpus']:
        raise c.ValidationError('DEFERRED_RESOURCE_GUARD: GPU resource state unavailable')
    record['allocation_budget_bytes'] = estimate
    record['allocation_bytes_per_px'] = bytes_per_px
    record['decision'] = 'NORMAL_SINGLE_THREAD_SERIAL'


def prepare_output(path, inputs=()):
    out = Path(path).resolve()
    repo = Path(__file__).resolve().parents[2]
    if out == repo or repo in out.parents:
        raise c.ValidationError('Output must be outside this source checkout')
    for source in inputs:
        source = Path(source).resolve()
        if out == source or out in source.parents or source in out.parents:
            raise c.ValidationError('Output overlaps input path')
    if out.exists():
        raise c.ValidationError('Output already exists; choose a fresh directory')
    out.mkdir(parents=True, exist_ok=False)
    return out


def arr(value):
    return value.detach().cpu().numpy()


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def render_full_plane(extended_field, geometry, row, out, reference_peak,
                      reference_padding, identity='optimized-artifact validation', db_floor=-80.0):
    """Display a saved/current field only; never propagate or change energy metrics."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    n, dx_um = geometry.n, geometry.dx_m * 1e6
    field = np.asarray(extended_field)
    m = n + 2 * int(row['padding'])
    if field.shape != (m, m) or not np.iscomplexobj(field) or not np.isfinite(field).all():
        raise ValueError('Full-plane display requires finite complex [n+2padding]^2 field')
    if not math.isfinite(reference_peak) or reference_peak <= 0 or not db_floor < 0:
        raise ValueError('Full-plane display requires positive fixed zero-angle Iref and negative dB floor')
    intensity = np.abs(field)**2
    start = (m - n) // 2
    window_energy = float(intensity[start:start+n, start:start+n].sum(dtype=np.float64)) * geometry.dx_m**2
    extended_energy = float(intensity.sum(dtype=np.float64)) * geometry.dx_m**2
    if extended_energy <= 0:
        raise ValueError('Zero extended-plane intensity')
    energy_check = {}
    for name, actual in [('energy_output_au', window_energy), ('energy_extended_au', extended_energy)]:
        expected = row.get(name)
        if expected is not None:
            if not np.isclose(actual, expected, rtol=5e-6, atol=1e-18):
                raise ValueError(f'Full-plane raw energy inconsistent with saved row: {name}')
            energy_check[name] = dict(expected=expected, displayed_raw_sum=actual)
    ratio = window_energy / extended_energy
    extent = [-m*dx_um/2, m*dx_um/2, m*dx_um/2, -m*dx_um/2]
    patch = int(row['roi_fixed']['size'])
    base = n//2 - patch//2
    target_xy = ((base-n/2)*dx_um, (base-n/2)*dx_um)
    target_width = patch * dx_um
    theory_xy = (target_xy[0] + row['shift_theory_dx_um'], target_xy[1] + row['shift_theory_dy_um'])
    db = 10*np.log10(np.maximum(intensity / reference_peak, 10**(db_floor/10)))
    above = int((intensity > reference_peak).sum())
    below_floor = int((intensity < reference_peak * 10**(db_floor/10)).sum())
    peak_y, peak_x = np.unravel_index(int(np.argmax(intensity)), intensity.shape)
    peak_xy = ((peak_x-(m-1)/2)*dx_um, (peak_y-(m-1)/2)*dx_um)
    fig, axes = plt.subplots(1,2,figsize=(13,7.0),layout='constrained')
    sign = row['artifact'].split('_')[-1].upper()
    fig.suptitle(f'{identity}\n{row["artifact"]} / {sign} | {row["axis"].upper()} {row["angle_deg"]:+g} deg | padding{row["padding"]} | {m} x {m} | {geometry.wavelength_m*1e9:g} nm',fontsize=12)
    for i,(ax,data) in enumerate(zip(axes,(intensity,db))):
        im = ax.imshow(data,origin='upper',extent=extent,interpolation='nearest',cmap='inferno',
                       vmin=0 if i==0 else db_floor,vmax=reference_peak if i==0 else 0)
        ax.set(title='Full plane: raw linear intensity' if i==0 else f'Full plane: 10 log10(I / Iref), floor {db_floor:g} dB',xlabel='x (um)',ylabel='y (um, positive downward)')
        for xy,width,color,style,label in [((-n*dx_um/2,-n*dx_um/2),n*dx_um,'cyan','-','Original n x n output window'),
              (target_xy,target_width,'white','-',f'Original {patch} x {patch} target support'),
              (theory_xy,target_width,'lime','--','Target support + geometric shift')]:
            ax.add_patch(Rectangle(xy,width,width,fill=False,edgecolor=color,linestyle=style,linewidth=1.0,label=label))
        ax.plot(*peak_xy,marker='+',color='red',markersize=7,label='Raw full-plane peak')
        ax.legend(loc='upper left',fontsize=7,framealpha=.7)
        fig.colorbar(im,ax=ax,label='Raw intensity / a.u.' if i==0 else 'dB relative to fixed zero-angle peak',extend='max' if above else 'neither',shrink=.85)
    max_ratio = float(intensity.max()) / reference_peak
    fig.supxlabel(f'Iref = {reference_peak:.7g} a.u. from same artifact, 0 deg, padding{reference_padding}; fixed across angles AND paddings.\n'
        f'Original {n} window / extended energy = {ratio:.6f} ({100*ratio:.3f}%); outside = {1-ratio:.6f} ({100*(1-ratio):.3f}%). Finite-plane capture, NOT device efficiency.\n'
        f'Linear vmax=Iref; log range=[{db_floor:g},0] dB. Above vmax: {above} samples; max/Iref={max_ratio:.5g}. Display-floor samples={below_floor}; raw energy is unmodified.\n'
        'The extended computational plane is also finite: a full-plane view does not exclude wrap-around or sampling aliasing. Input aperture is NOT detector active area.',fontsize=8)
    filename = row['sample_id'] + '_full_plane.png'
    try:
        with (Path(out)/filename).open('xb') as handle:
            fig.savefig(handle,format='png',dpi=140,bbox_inches='tight',pad_inches=.12)
    finally:
        plt.close(fig)
    return dict(png_file=filename,sample_id=row['sample_id'],source_npz=row.get('npz_file'),identity=identity,
                definition='I=abs(extended_field)^2, raw linear plus fixed-Iref 10log10 display; no propagation/recentering/gain',
                shape=[m,m],dx_um=dx_um,extent_um=extent,origin='upper; [y,x]; +y down',reference_peak=reference_peak,
                reference_angle_deg=0,reference_padding=reference_padding,source_baseline_padding=n,
                linear_vmin=0,linear_vmax=reference_peak,db_floor=db_floor,db_vmax=0,above_vmax_samples=above,
                below_display_floor_samples=below_floor,max_over_reference=max_ratio,output_energy_au=window_energy,
                extended_energy_au=extended_energy,output_over_extended=ratio,outside_fraction=1-ratio,
                output_window_um=[-n*dx_um/2,-n*dx_um/2,n*dx_um,n*dx_um],
                target_box_um=[*target_xy,target_width,target_width],theory_target_box_um=[*theory_xy,target_width,target_width],
                theory_box_note='Continuous geometric displacement; existing integer-rounded ROI metrics remain unchanged',
                peak_xy_um=peak_xy,energy_cross_check=energy_check)


def ideal_field(g):
    x = c.axis_coordinates(g.n,g.dx_m)
    yy,xx = torch.meshgrid(x,x,indexing='ij')
    radius2 = xx.square()+yy.square()
    amplitude = torch.exp(-radius2/(g.extent_m/6)**2)
    phase = -2*math.pi/g.wavelength_m*(torch.sqrt(radius2+g.distance_m**2)-g.distance_m)
    return (amplitude*torch.exp(1j*phase)).to(torch.complex64)


def scan(field, g, paddings, axes, angles, roi_size, device, out, label, baseline_padding=None, sample_offset=0):
    baseline_padding = g.n if baseline_padding is None else baseline_padding
    if baseline_padding not in paddings:
        raise c.ValidationError("padding comparison must include explicit baseline")
    engine = c.PropagationEngine(g,device)
    t = c.Thresholds()
    rows, images = [], []
    full_plane_reference = None
    reference_padding = paddings[0]
    for padding in paddings:
        reference_full, reference_field, _ = engine.extended(field,padding)
        reference = c.intensity(reference_field)
        reference_cpu = reference.detach().cpu()
        reference_energy = c.energy(reference,g.dx_m)
        fixed_reference_window = c.roi_window(g.n,roi_size)
        reference_roi_energy = c.roi_sum(reference,fixed_reference_window)*g.dx_m**2
        reference_extended_intensity = c.intensity(reference_full)
        reference_extended_energy = c.energy(reference_extended_intensity,g.dx_m)
        if full_plane_reference is None:
            full_plane_reference = float(reference_extended_intensity.max())
        direct = engine.direct(field,padding)
        crop_error = float((direct-reference_field).abs().max())
        if crop_error > 2e-6*max(1.,float(direct.abs().max())):
            raise c.ValidationError('extended/direct crop consistency failed')
        for axis in axes:
            for angle in sorted(set([0.]+list(angles))):
                tx,ty = (angle,0.) if axis=='x' else (0.,angle)
                tilted = c.apply_carrier(field,g,tx,ty)
                full, propagated, layer = engine.extended(tilted,padding)
                psf = c.intensity(propagated)
                shift = c.estimate_shift(reference_cpu,psf)
                theory = c.lateral_shift_theory(g,tx,ty)
                measured = None
                aligned = mask = None
                if shift.valid:
                    aligned,mask = c.sample_shifted(psf,shift.dy_px,shift.dx_px)
                    measured = c.compare_shapes(reference_cpu,aligned,mask)
                theory_aligned,theory_mask = c.sample_shifted(psf,*theory)
                auxiliary = c.compare_shapes(reference_cpu,theory_aligned,theory_mask)
                validity,reasons = c.classify_validity(measured,auxiliary,shift,t)
                window = c.roi_window(g.n,roi_size)
                following = c.roi_window(g.n,roi_size,*theory)
                boundary = c.boundary_stats(psf)
                extended_boundary = c.boundary_stats(c.intensity(full))
                sampling = c.sampling_diagnostics(tilted,layer,g,tx,ty)
                warning,sampling_reasons = c.classify_sampling(sampling,t)
                e = c.energy(psf,g.dx_m)
                efull = c.energy(c.intensity(full),g.dx_m)
                row = dict(artifact=label,axis=axis,angle_deg=angle,padding=padding,
                    shift_theory_dy_px=theory[0],shift_theory_dx_px=theory[1],
                    shift_measured_dy_px=shift.dy_px,shift_measured_dx_px=shift.dx_px,
                    shift_estimator=asdict(shift),
                    energy_input_au=c.energy(c.intensity(field),g.dx_m),
                    energy_output_au=e,energy_extended_au=efull,
                    energy_output_over_extended=e/efull if efull else None,
                    energy_output_relative_zero=e/reference_energy if reference_energy else None,
                    energy_extended_relative_zero=efull/reference_extended_energy if reference_extended_energy else None,
                    roi_fixed=window,roi_following=following,
                    energy_roi_fixed_au=c.roi_sum(psf,window)*g.dx_m**2,
                    energy_roi_following_au=c.roi_sum(psf,following)*g.dx_m**2,
                    boundary=asdict(boundary),extended_boundary=asdict(extended_boundary),
                    crop_max_abs_error=crop_error,sampling=asdict(sampling),
                    sampling_warning=warning,sampling_reasons=sampling_reasons,
                    numeric_validity=validity,numeric_reasons=reasons,
                    isoplanatism_acceptance='NOT_SPECIFIED',
                    shape_error_measured=None if measured is None else measured.shape_error)
                for key in ('energy_roi_fixed_au','energy_roi_following_au'):
                    row[key.replace('_au','_relative_zero')] = row[key]/reference_roi_energy if reference_roi_energy else None
                for name,comparison in [('measured',measured),('theory',auxiliary)]:
                    row['comparison_'+name] = None if comparison is None else asdict(comparison)
                for a,index in [('y',0),('x',1)]:
                    value = shift.dy_px if a=='y' else shift.dx_px
                    row['shift_theory_d'+a+'_um']=theory[index]*g.dx_m*1e6
                    row['shift_measured_d'+a+'_um']=None if value is None else value*g.dx_m*1e6
                    row['shift_error_d'+a+'_px']=None if value is None else value-theory[index]
                    row['shift_error_d'+a+'_um']=None if value is None else (value-theory[index])*g.dx_m*1e6
                sample_id=f'sample_{sample_offset+len(rows):06d}'
                stem=f'{sample_id}_{label}_{axis}_p{padding}'
                npz_file, png_file = stem+'.npz', stem+'.png'
                if (out/npz_file).exists() or (out/png_file).exists():
                    raise c.ValidationError(f'unexpected artifact name collision: {sample_id}')
                row.update(sample_id=sample_id,npz_file=npz_file,png_file=png_file)
                full_array = arr(full)
                with (out/npz_file).open('xb') as handle:
                    np.savez_compressed(handle,reference=arr(reference),psf=arr(psf),
                        sample_id=np.array(sample_id),angle_deg=np.array(angle),padding=np.array(padding),
                        field=arr(propagated),extended_field=full_array,
                        aligned=np.array([]) if aligned is None else arr(aligned),
                        valid_mask=np.zeros((g.n,g.n),bool) if mask is None else arr(mask),
                        theory_aligned=arr(theory_aligned),theory_mask=arr(theory_mask))
                row.update(full_plane_png_file=None,full_plane_display=None,full_plane_display_error=None)
                try:
                    display = render_full_plane(full_array,g,row,out,full_plane_reference,reference_padding)
                    row.update(full_plane_png_file=display['png_file'],full_plane_display=display)
                except Exception as exc:
                    row['full_plane_display_error'] = f'{type(exc).__name__}: {exc}'
                    print(f'Full-plane display failed for {sample_id}: {exc}',file=sys.stderr,flush=True)
                del full_array
                images.append((stem,arr(reference),arr(psf),None if aligned is None else arr(aligned),None if mask is None else arr(mask)))
                rows.append(row)
    cases = {}
    for row in rows:
        cases.setdefault((row['axis'],row['angle_deg']),[]).append(row)
    for matches in cases.values():
        baseline = next(r for r in matches if r['padding']==baseline_padding)
        comparisons = []
        # Read only two saved center crops at a time, not extra extended fields.
        with np.load(out/baseline['npz_file'],allow_pickle=False) as saved:
            baseline_psf = saved['psf']
        for candidate in sorted(matches,key=lambda r:r['padding']):
            if candidate is baseline:
                continue
            with np.load(out/candidate['npz_file'],allow_pickle=False) as saved:
                candidate_psf = saved['psf']
            comparison = c.padding_stability(dict(baseline,psf=baseline_psf),
                                             dict(candidate,psf=candidate_psf),t)
            comparison['comparison_id'] = baseline['sample_id']+'__'+candidate['sample_id']
            comparisons.append(comparison)
        if len(paddings)==1:
            status,reason='NOT_CHECKED','single padding; no independent comparison'
        elif len(comparisons)!=len(paddings)-1:
            status,reason='indeterminate','missing requested comparison'
        elif any(p['status']=='inconsistent' for p in comparisons):
            status,reason='inconsistent','at least one baseline-vs-candidate comparison inconsistent'
        elif any(p['status']!='consistent' for p in comparisons):
            status,reason='indeterminate','at least one comparison undefined'
        else:
            status,reason='consistent','all requested baseline-vs-candidate comparisons consistent'
        stability=dict(status=status,reason=reason,scope='case-level all requested baseline-vs-candidate comparisons',
                       baseline_padding=baseline_padding,requested_paddings=sorted(paddings),comparisons=comparisons)
        for row in matches:
            row['padding_stability']=stability
            row['padding_comparison_ids']=[p['comparison_id'] for p in comparisons
                                          if row['npz_file'] in (p['baseline_npz_file'],p['candidate_npz_file'])]
            row['truncation'],row['truncation_evidence']=c.classify_truncation(
                c.BoundaryStats(**row['boundary']),row['energy_output_over_extended'],status,t)
            row['physical_interpretation']='indeterminate: sampling aliasing not excluded; no application threshold'
    vmax=max(float(max(ref.max(),psf.max())) for _,ref,psf,_,_ in images)
    for stem,ref,psf,aligned,mask in images:
        panel_png(out,stem,ref,psf,aligned,mask,vmax)
    return rows


# ---------------------------------------------------------------------------
# two-dimensional isoplanatism points
#
# The formal point set is the 2D incidence grid: the centre, the four edge
# midpoints and the four corners.  A point is an incidence *direction*, not a
# lateral object-plane offset: the device pattern stays centred in the input
# grid and the direction enters as a carrier ramp, which each row records in
# ``object_to_input``.  Every point is compared with the centre PSF translated
# by the expected image displacement, so shape, integrated intensity and the
# extra (measured minus expected) displacement stay separate quantities.
# ---------------------------------------------------------------------------

POINT_NAMES=('center','edge_x_pos','edge_x_neg','edge_y_pos','edge_y_neg',
             'corner_pp','corner_pn','corner_np','corner_nn')
CORNER_NAMES=('corner_pp','corner_pn','corner_np','corner_nn')


def formal_points(angle_deg):
    a=float(angle_deg)
    if not math.isfinite(a) or not 0<a<90:
        raise c.ValidationError('point angle must be finite and strictly between 0 and 90 degrees')
    grid=[('center',0.,0.),
          ('edge_x_pos',a,0.),('edge_x_neg',-a,0.),
          ('edge_y_pos',0.,a),('edge_y_neg',0.,-a),
          ('corner_pp',a,a),('corner_pn',a,-a),('corner_np',-a,a),('corner_nn',-a,-a)]
    return [dict(name=name,theta_x_deg=tx,theta_y_deg=ty) for name,tx,ty in grid]


def object_to_input(point,g,padding):
    theory=c.lateral_shift_theory(g,point['theta_x_deg'],point['theta_y_deg'])
    sx=math.sin(math.radians(point['theta_x_deg']))
    sy=math.sin(math.radians(point['theta_y_deg']))
    return dict(
        object_plane_point='incidence direction on the object plane; the device pattern stays centred in the n*n input grid',
        input_phase='carrier ramp exp(i*2*pi/lambda*(sin(theta_x)*x+sin(theta_y)*y)) applied before propagation',
        theta_x_deg=point['theta_x_deg'],theta_y_deg=point['theta_y_deg'],
        sin_theta_x=sx,sin_theta_y=sy,
        sin_theta_over_max=max(abs(sx),abs(sy))/g.max_sin_theta if g.max_sin_theta else None,
        input_grid=f'{g.n}x{g.n} centred in the zero-filled (n+2*{padding})^2 extended plane',
        detector_shift_theory_px=dict(dy=theory[0],dx=theory[1]),
        detector_shift_theory_um=dict(dy=theory[0]*g.dx_m*1e6,dx=theory[1]*g.dx_m*1e6),
        detector_shift_rule='z*tan(theta)/dx, geometric approximation',
        limitation='no lateral object-plane translation and no angular electromagnetic response are modelled')


def scan_points(field,g,paddings,points,roi_size,device,out,label,camera_pixel_m,
                baseline_padding=None,sample_offset=0):
    """Independent 2D point scan; every point is compared with the centre PSF."""
    baseline_padding=g.n if baseline_padding is None else baseline_padding
    if baseline_padding not in paddings:
        raise c.ValidationError('point comparison must include explicit baseline padding')
    engine=c.PropagationEngine(g,device)
    t=c.Thresholds()
    fixed=c.roi_window(g.n,roi_size)
    if not fixed['valid']:
        raise c.ValidationError(f'ROI size {roi_size} is outside the {g.n}px detector crop')
    bins=c.pixel_bin_matrix(roi_size,g.dx_m,camera_pixel_m)
    input_energy=c.energy(c.intensity(field),g.dx_m)
    rows,images=[],[]
    full_plane_reference=None
    reference_padding=paddings[0]
    for padding in paddings:
        reference_full,reference_field,_=engine.extended(field,padding)
        reference=c.intensity(reference_field)
        reference_cpu=reference.detach().cpu()
        reference_energy=c.energy(reference,g.dx_m)
        reference_roi=c.roi_sum(reference,fixed)*g.dx_m**2
        reference_extended=c.intensity(reference_full)
        reference_extended_energy=c.energy(reference_extended,g.dx_m)
        if full_plane_reference is None:
            full_plane_reference=float(reference_extended.max())
        direct=engine.direct(field,padding)
        crop_error=float((direct-reference_field).abs().max())
        if crop_error>2e-6*max(1.,float(direct.abs().max())):
            raise c.ValidationError('extended/direct crop consistency failed')
        extended_window=c.roi_window(g.n+2*padding,roi_size)
        for point in points:
            tx,ty=point['theta_x_deg'],point['theta_y_deg']
            tilted=c.apply_carrier(field,g,tx,ty)
            full,propagated,layer=engine.extended(tilted,padding)
            psf=c.intensity(propagated)
            extended_psf=c.intensity(full)
            shift=c.estimate_shift(reference_cpu,psf)
            theory=c.lateral_shift_theory(g,tx,ty)
            measured=aligned=mask=None
            if shift.valid:
                aligned,mask=c.sample_shifted(psf,shift.dy_px,shift.dx_px)
                measured=c.compare_shapes(reference_cpu,aligned,mask)
            theory_aligned,theory_mask=c.sample_shifted(psf,*theory)
            auxiliary=c.compare_shapes(reference_cpu,theory_aligned,theory_mask)
            validity,reasons=c.classify_validity(measured,auxiliary,shift,t)
            boundary=c.boundary_stats(psf)
            extended_boundary=c.boundary_stats(extended_psf)
            sampling=c.sampling_diagnostics(tilted,layer,g,tx,ty)
            warning,sampling_reasons=c.classify_sampling(sampling,t)
            e=c.energy(psf,g.dx_m)
            efull=c.energy(extended_psf,g.dx_m)
            roi_au=c.roi_sum(psf,fixed)*g.dx_m**2
            crop=psf[fixed['y0']:fixed['y1'],fixed['x0']:fixed['x1']]
            patch=c.pixel_integrate(crop,bins,bins,g.dx_m)
            row=dict(artifact=label,axis='point',point=point['name'],
                theta_x_deg=tx,theta_y_deg=ty,angle_deg=math.hypot(tx,ty),
                padding=padding,roi_size=roi_size,camera_pixel_um=camera_pixel_m*1e6,
                object_to_input=object_to_input(point,g,padding),
                shift_theory_dy_px=theory[0],shift_theory_dx_px=theory[1],
                shift_measured_dy_px=shift.dy_px,shift_measured_dx_px=shift.dx_px,
                shift_estimator=asdict(shift),
                extra_shift_dy_px=None if shift.dy_px is None else shift.dy_px-theory[0],
                extra_shift_dx_px=None if shift.dx_px is None else shift.dx_px-theory[1],
                energy_input_au=input_energy,
                energy_output_au=e,energy_extended_au=efull,
                energy_output_over_extended=e/efull if efull else None,
                energy_output_relative_centre=e/reference_energy if reference_energy else None,
                energy_extended_relative_centre=efull/reference_extended_energy if reference_extended_energy else None,
                roi=dict(fixed=fixed,extended=extended_window),
                energy_roi_fixed_au=roi_au,
                energy_roi_ratio_vs_centre=roi_au/reference_roi if reference_roi else None,
                roi_energy_output=asdict(c.roi_energy(psf,fixed)),
                roi_energy_extended=asdict(c.roi_energy(extended_psf,extended_window)),
                pixel_patch=patch.tolist(),
                boundary=asdict(boundary),extended_boundary=asdict(extended_boundary),
                crop_max_abs_error=crop_error,sampling=asdict(sampling),
                sampling_warning=warning,sampling_reasons=sampling_reasons,
                numeric_validity=validity,numeric_reasons=reasons,
                comparison_vs_centre_measured=None if measured is None else asdict(measured),
                comparison_vs_centre_theory=asdict(auxiliary),
                shape_error_vs_centre=None if measured is None else measured.shape_error,
                isoplanatism_acceptance='NOT_SPECIFIED')
            for axis_name,index in [('y',0),('x',1)]:
                value=shift.dy_px if axis_name=='y' else shift.dx_px
                row['shift_theory_d'+axis_name+'_um']=theory[index]*g.dx_m*1e6
                row['shift_measured_d'+axis_name+'_um']=None if value is None else value*g.dx_m*1e6
                row['shift_error_d'+axis_name+'_px']=None if value is None else value-theory[index]
            sample_id=f'sample_{sample_offset+len(rows):06d}'
            stem=f'{sample_id}_{label}_{point["name"]}_p{padding}'
            npz_file,png_file=stem+'.npz',stem+'.png'
            if (out/npz_file).exists() or (out/png_file).exists():
                raise c.ValidationError(f'unexpected artifact name collision: {sample_id}')
            row.update(sample_id=sample_id,npz_file=npz_file,png_file=png_file)
            full_array=arr(full)
            with (out/npz_file).open('xb') as handle:
                np.savez_compressed(handle,reference=arr(reference),psf=arr(psf),
                    sample_id=np.array(sample_id),point=np.array(point['name']),
                    theta_deg=np.array([tx,ty]),padding=np.array(padding),
                    field=arr(propagated),extended_field=full_array,
                    roi_intensity=arr(crop),pixel_patch=np.asarray(patch,dtype=np.float64),
                    aligned=np.array([]) if aligned is None else arr(aligned),
                    valid_mask=np.zeros((g.n,g.n),bool) if mask is None else arr(mask),
                    theory_aligned=arr(theory_aligned),theory_mask=arr(theory_mask))
            row.update(full_plane_png_file=None,full_plane_display=None,full_plane_display_error=None)
            display_row=dict(row,roi_fixed=row['roi']['fixed'],axis=row['point'])
            try:
                display=render_full_plane(full_array,g,display_row,out,full_plane_reference,reference_padding)
                row.update(full_plane_png_file=display['png_file'],full_plane_display=display)
            except Exception as exc:
                row['full_plane_display_error']=f'{type(exc).__name__}: {exc}'
                print(f'Full-plane display failed for {sample_id}: {exc}',file=sys.stderr,flush=True)
            del full_array,extended_psf
            images.append((stem,arr(reference),arr(psf),None if aligned is None else arr(aligned),
                           None if mask is None else arr(mask)))
            rows.append(row)
    for name in {row['point'] for row in rows}:
        matches=[row for row in rows if row['point']==name]
        if not any(row['padding']==baseline_padding for row in matches):
            raise c.ValidationError(f'point {name} is missing the baseline padding {baseline_padding}')
        baseline=next(row for row in matches if row['padding']==baseline_padding)
        comparisons=[]
        with np.load(out/baseline['npz_file'],allow_pickle=False) as saved:
            baseline_psf=saved['psf']
        for candidate in sorted(matches,key=lambda r:r['padding']):
            if candidate is baseline:
                continue
            with np.load(out/candidate['npz_file'],allow_pickle=False) as saved:
                candidate_psf=saved['psf']
            comparison=c.padding_stability(dict(baseline,psf=baseline_psf),
                                           dict(candidate,psf=candidate_psf),t)
            comparison['comparison_id']=baseline['sample_id']+'__'+candidate['sample_id']
            comparisons.append(comparison)
        if len(paddings)==1:
            status,reason='NOT_CHECKED','single padding; no independent comparison'
        elif len(comparisons)!=len(paddings)-1:
            status,reason='indeterminate','missing requested comparison'
        elif any(p['status']=='inconsistent' for p in comparisons):
            status,reason='inconsistent','at least one baseline-vs-candidate comparison inconsistent'
        elif any(p['status']!='consistent' for p in comparisons):
            status,reason='indeterminate','at least one comparison undefined'
        else:
            status,reason='consistent','all requested baseline-vs-candidate comparisons consistent'
        scope=('corner point re-checked with larger window' if name in CORNER_NAMES
               else 'edge-midpoint re-check with larger window' if name!='center'
               else 'centre control; expected to be independent of registration')
        stability=dict(status=status,reason=reason,scope=scope,point=name,
                       baseline_padding=baseline_padding,requested_paddings=sorted(paddings),
                       comparisons=comparisons)
        for row in matches:
            row['padding_stability']=stability
            row['truncation'],row['truncation_evidence']=c.classify_truncation(
                c.BoundaryStats(**row['boundary']),row['energy_output_over_extended'],status,t)
            row['physical_interpretation']='indeterminate: no isoplanatism tolerance is defined'
    vmax=max(float(max(ref.max(),psf.max())) for _,ref,psf,_,_ in images)
    for stem,ref,psf,aligned,mask in images:
        panel_png(out,stem,ref,psf,aligned,mask,vmax)
    return rows


def pos_neg_rows(rows_by_branch,gains):
    """POS-NEG differential response per (point, padding) at fixed gain."""
    labels=sorted(rows_by_branch)
    if len(labels)!=2:
        raise c.ValidationError('POS/NEG aggregation needs exactly two measured branches')
    left,right=labels
    index={}
    for label,items in rows_by_branch.items():
        for row in items:
            index[(label,row['point'],row['padding'])]=row
    keys=sorted({(row['point'],row['padding']) for row in rows_by_branch[left]})
    rows=[]
    for point,padding in keys:
        a=index.get((left,point,padding))
        b=index.get((right,point,padding))
        if a is None or b is None:
            rows.append(dict(point=point,padding=padding,status='undefined',
                             reason='missing branch sample',threshold='NOT_SPECIFIED'))
            continue
        patch_a=np.asarray(a['pixel_patch'],dtype=np.float64)
        patch_b=np.asarray(b['pixel_patch'],dtype=np.float64)
        na,nb=np.linalg.norm(patch_a),np.linalg.norm(patch_b)
        shape=None
        if na>0 and nb>0:
            shape=float(np.linalg.norm(patch_a/na-patch_b/nb))
        energy=gains[left]*a['energy_roi_fixed_au']-gains[right]*b['energy_roi_fixed_au']
        rows.append(dict(point=point,padding=padding,
            pos_artifact=left,neg_artifact=right,
            pos_gain=gains[left],neg_gain=gains[right],
            energy_au=energy,
            shape=shape,
            shape_definition='||pos/||pos||2 - neg/||neg||2||2 on fixed-ROI camera-pixel integrals',
            energy_definition='gain_pos*ROI_integrated_pos - gain_neg*ROI_integrated_neg',
            shift_dy_px=(a['shift_measured_dy_px']-b['shift_measured_dy_px'])
                if a['shift_measured_dy_px'] is not None and b['shift_measured_dy_px'] is not None else None,
            shift_dx_px=(a['shift_measured_dx_px']-b['shift_measured_dx_px'])
                if a['shift_measured_dx_px'] is not None and b['shift_measured_dx_px'] is not None else None,
            status='metrics_only',threshold='NOT_SPECIFIED',
            reason='no differential isoplanatism tolerance is defined in this repository'))
    return rows


def load_convergence(path):
    source=Path(path)
    data=json.loads(source.read_text(encoding='utf-8'))
    if data.get('run_class')!='fft_window_convergence':
        raise c.ValidationError('convergence manifest is not a window convergence run')
    convergence=data.get('convergence') or {}
    return dict(path=str(source.resolve()),sha256=c.sha256_file(source),
                run_class=data.get('run_class'),status=convergence.get('status'),
                converged_from_fft_size=convergence.get('converged_from_fft_size'),
                fft_sizes=[window['fft_size'] for window in data.get('windows') or []],
                tolerances=data.get('tolerances'),
                tolerances_source=data.get('tolerances_source'),
                dx_review_status=(data.get('dx_review') or {}).get('status'))


def resolve_baseline_padding(geometry,convergence,paddings,explicit=None,config_padding=None):
    """Baseline padding for a point scan: derived from the converged window, never guessed."""
    paddings=sorted(int(p) for p in paddings)
    if explicit is not None and explicit not in paddings:
        raise c.ValidationError(
            f'--baseline-padding {explicit} is not one of the requested paddings {paddings}')
    derived=None; source='explicit --baseline-padding'
    if convergence is not None:
        status=convergence.get('status')
        if status!='CONVERGED':
            raise c.ValidationError(
                f'convergence manifest status is {status}; a formal point scan needs CONVERGED')
        fft=convergence.get('converged_from_fft_size')
        if fft is None:
            raise c.ValidationError('convergence manifest reports no converged FFT size')
        if fft not in (convergence.get('fft_sizes') or []):
            raise c.ValidationError(f'converged FFT {fft} is absent from the manifest window list')
        delta=int(fft)-int(geometry.n)
        if delta<0 or delta%2:
            raise c.ValidationError(
                f'FFT {fft} cannot be written as N_device {geometry.n} plus equal two-sided padding')
        derived=delta//2
        source=f'convergence manifest: FFT {fft} = {geometry.n} + 2*{derived}'
        if derived not in paddings:
            raise c.ValidationError(
                f'converged window FFT {fft} needs baseline padding {derived}; '
                f'requested paddings {paddings} do not include it')
        if explicit is not None and int(explicit)!=derived:
            raise c.ValidationError(
                f'--baseline-padding {explicit} contradicts the converged window FFT {fft} '
                f'(padding {derived})')
    if explicit is not None:
        return int(explicit),source
    if derived is not None:
        return derived,source
    if config_padding is None:
        raise c.ValidationError('no convergence manifest and no source padding: the baseline padding is undefined')
    if config_padding not in paddings:
        raise c.ValidationError('padding comparison must include source baseline padding, '
                                'or pass --baseline-padding at the converged window')
    return int(config_padding),'source config padding'


def identity_checks(config,artifacts,geometry,roi_size,paddings,convergence=None,
                    baseline_padding=None):
    checks=[]
    baseline=min(paddings) if baseline_padding is None else int(baseline_padding)
    def add(name,ok,detail):
        checks.append(dict(name=name,status='pass' if ok else 'fail',detail=detail))
    add('artifact_pair',len(artifacts)==2,f'{len(artifacts)} artifact(s)')
    add('wavelength_nm_matches_config',
        all(a.wavelength_nm==config.wavelength_nm for a in artifacts),
        f'config {config.wavelength_nm} nm')
    add('dx_nm_matches_core_constant',config.dx_nm==c.DX_NM,f'config dx {config.dx_nm} nm')
    add('distance_um_matches_geometry',
        abs(geometry.distance_m*1e6-config.distance_um)<1e-9,
        f'geometry {geometry.distance_m*1e6} um')
    add('detector_roi_matches_patch_meta',roi_size==config.patch_meta,
        f'roi {roi_size} px vs patch_meta {config.patch_meta} px')
    add('artifact_grid_matches_config_n',
        all(tuple(a.psf_raw.shape[-2:])==(config.n,config.n) for a in artifacts if not a.skipped),
        f'config n {config.n} px')
    add('branches_are_pos_and_neg',
        {a.branch.rsplit('_',1)[-1] for a in artifacts}=={'pos','neg'},
        sorted(a.branch for a in artifacts))
    add('same_kernel_directory',
        len({str(Path(a.path).resolve().parent) for a in artifacts})==1,
        'identical fit directory required for a POS/NEG pair')
    if convergence is None:
        add('convergence_manifest_supplied',False,'no --convergence manifest')
    else:
        add('convergence_manifest_supplied',True,convergence['path'])
        add('convergence_status_converged',convergence['status']=='CONVERGED',
            f"status {convergence['status']}")
        add('requested_windows_in_manifest',
            all(geometry.n+2*int(p) in convergence['fft_sizes'] for p in paddings),
            f"requested paddings {sorted(paddings)}")
        if convergence['converged_from_fft_size'] is None:
            add('baseline_window_at_or_above_convergence',False,'no converged FFT size reported')
        else:
            add('baseline_window_at_or_above_convergence',
                geometry.n+2*baseline>=convergence['converged_from_fft_size'],
                f"baseline FFT {geometry.n+2*baseline} vs converged from {convergence['converged_from_fft_size']}")
        add('convergence_dx_review_pass',convergence['dx_review_status']=='PASS',
            f"dx review {convergence['dx_review_status']}")
    failed=[check['name'] for check in checks if check['status']!='pass']
    return dict(status='PASS' if not failed else 'FAIL',checks=checks,failed=failed,
                formal_conclusion_allowed=not failed,
                note='a failed identity or convergence check forbids a formal isoplanatism conclusion')


def formal_conclusion(checks,convergence):
    """Verdict envelope: metrics are reportable, a physical PASS/FAIL never is."""
    reasons=[]
    blocked=False
    if checks is not None and checks['failed']:
        blocked=True
        reasons.extend(f'failed check: {name}' for name in checks['failed'])
    if convergence is None:
        blocked=True
        reasons.append('no window convergence manifest supplied')
    elif convergence['status']!='CONVERGED':
        blocked=True
        reasons.append(f"window convergence status {convergence['status']}")
    reasons.append('no isoplanatism tolerance is defined in this repository')
    return dict(verdict='BLOCKED' if blocked else 'METRICS_ONLY',reasons=reasons,
                acceptance='NOT_SPECIFIED',
                scope='measured points only, kernel000 POS/NEG pairing')


def panel_png(out,stem,ref,psf,aligned,mask,vmax):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(10,3))
    for ax,data,title in zip(axs[:2],[ref,psf],['Reference','Independent ASM']):
        ax.imshow(data,origin='upper',vmin=0,vmax=vmax); ax.set_title(title)
    if aligned is not None:
        r=ref[mask]; a=aligned[mask]
        diff=np.full_like(ref,np.nan,dtype=float)
        if np.linalg.norm(r)>0 and np.linalg.norm(a)>0:
            diff[mask]=a/np.linalg.norm(a)-r/np.linalg.norm(r)
        axs[2].imshow(diff,origin='upper',vmin=-1,vmax=1,cmap='coolwarm')
    axs[2].set_title('Common mask L2 difference [-1,1]')
    fig.tight_layout()
    with (out/(stem+'.png')).open('xb') as handle:
        fig.savefig(handle,format='png')
    plt.close(fig)


def save_rows(out,rows):
    dump_json(out/'metrics.json',rows)
    if rows:
        with (out/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({k:json.dumps(v,allow_nan=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact',type=Path,action='append')
    parser.add_argument('--config',type=Path)
    parser.add_argument('--self-check',action='store_true')
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--angles',type=float,nargs='+',default=c.DEFAULT_ANGLES_DEG)
    parser.add_argument('--axes',choices=['x','y'],nargs='+',default=['x','y'])
    parser.add_argument('--paddings',type=int,nargs='+')
    parser.add_argument('--roi-size',type=int)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--points',action='store_true',
        help='2D incidence points: centre, four edge midpoints, four corners')
    parser.add_argument('--point-angle-deg',type=float,default=0.5,
        help='tilt magnitude of the 2D point grid in degrees')
    parser.add_argument('--convergence',type=Path,
        help='window_convergence manifest.json that fixes the converged window')
    parser.add_argument('--baseline-padding',type=int,
        help='padding treated as the reference; defaults to the source config padding, '
             'and must be raised to the converged window when a convergence manifest is supplied')
    args=parser.parse_args(argv)
    if args.self_check and (args.artifact or args.config):
        parser.error('self-check and real artifacts are separate run classes')
    if not args.self_check and (not args.artifact or not args.config):
        parser.error('BLOCKED_MISSING_OPTIMIZED_ARTIFACT: supply --artifact and --config')
    if args.points and args.self_check:
        parser.error('point mode requires optimized artifacts; the fixture keeps the axis/angle entry')
    if args.baseline_padding is not None and args.self_check:
        parser.error('the fixture uses its own n baseline; --baseline-padding is for real artifacts')
    if not all(math.isfinite(a) and abs(a)<90 for a in args.angles):
        parser.error('angles must be finite and strictly between -90 and 90 degrees')
    torch.set_num_threads(1)
    source=Path(__file__).resolve().parent
    locks=c.verify_locked_sources(source)
    inputs=[]; skipped=[]; artifacts=[]
    convergence=None if args.convergence is None else load_convergence(args.convergence)
    if args.self_check:
        g=c.Geometry(64,400e-9,450e-9,80e-6)
        fields=[('ideal_engineering_fixture',ideal_field(g))]
        paddings=args.paddings or [64,128]
        roi=args.roi_size or 15
    else:
        config=c.load_source_config(args.config)
        g=c.Geometry.from_config(config)
        paddings=args.paddings or [config.padding,config.padding+g.n]
        roi=args.roi_size or config.patch_meta
        fields=[]; inputs=[args.config]+args.artifact
        for path in args.artifact:
            artifact=c.load_artifact(path); c.cross_check_config_artifact(config,artifact)
            artifacts.append(artifact)
            if artifact.skipped:
                skipped.append(dict(artifact=str(path),reason='empty width: zero branch, no physical PSF'))
            else:
                fields.append((artifact.branch,artifact.width_nm))
        if len({name for name,_ in fields})!=len(fields):
            parser.error('artifact labels must be distinct')
    if not paddings or min(paddings)<0 or len(set(paddings))!=len(paddings):
        parser.error('paddings must be distinct nonnegative integers')
    if args.self_check:
        baseline_padding=g.n; baseline_padding_source='fixture geometry n'
        if baseline_padding not in paddings:
            parser.error('padding comparison must include the fixture baseline padding')
    else:
        try:
            baseline_padding,baseline_padding_source=resolve_baseline_padding(
                g,convergence,paddings,args.baseline_padding,config.padding)
        except c.ValidationError as exc:
            parser.error(str(exc))
    c.roi_window(g.n,roi)
    resources=resource_preflight(); guard(resources,g.n,paddings,args.device)
    if not args.self_check:
        fields=[(label,c.build_transmission(width, config.wavelength_nm)) for label,width in fields]
    points=formal_points(args.point_angle_deg) if args.points else None
    camera_pixel_m=None if args.self_check or config is None else float(config.raw['camera_pixel_um'])*1e-6
    checks=None
    if points is not None:
        checks=identity_checks(config,artifacts,g,roi,paddings,convergence,
                               baseline_padding=baseline_padding)
    out=prepare_output(args.out,inputs)
    manifest=dict(status='RUNNING',run_class='synthetic_engineering' if args.self_check else 'optimized_artifact_validation',
        arguments={k:str(v) if isinstance(v,Path) else [str(p) for p in v] if k=='artifact' and v else v for k,v in vars(args).items()},
        geometry=asdict(g),resources=resources,locked_sources=locks,
        baseline_padding=baseline_padding,
        baseline_padding_source=baseline_padding_source,
        padding_comparison="all baseline-vs-candidate raw PSFs, full fixed crop, no registration",
        artifact_naming="unique propagation sample ID; exact angle and filenames in each metrics row",
        source_config=None if args.self_check else config.raw,
        propagation_precision='complex64 original ASM; float64 fitted phase and CPU metrics',
        registration=dict(method='zero-padded linear cross-correlation, 3-point parabolic refinement',
                          interpolation='bilinear zero extension',min_overlap=.5,min_support_fraction=.25,min_support_pixels=9,reference_support='finite positive reference pixels'),
        source_hashes={str(p.relative_to(source)):c.sha256_file(p) for p in [Path(__file__),source/'isoplanatism_core.py',source/'README.md',source/'tests/test_isoplanatism.py']},
        input_hashes={str(p.resolve()):c.sha256_file(p) for p in inputs},
        python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),
        device=args.device,thresholds=c.Thresholds().as_dict(),threshold_rationale=c.THRESHOLD_RATIONALE,
        sampling_limitation=c.SAMPLING_LIMITATION,skipped_artifacts=skipped,
        coordinates='[row,col]=[y,x], pixel centers (index-(n-1)/2)*dx; origin upper, +y down',
        displacement='measured d: off-axis relative to reference; align translation -d; sample at +d',
        roi_rule='fixed center and theory-following integer-rounded intersection; no recentering',
        limitations=['phase-only normal-incidence library; no angular electromagnetic response',
                     'z*tan(theta) is a geometric approximation, not exact arbitrary-phase ASM invariance',
                     'no CMOS area integration; not OLED field coverage or hardware evidence',
                     'diagnostic thresholds are prospective engineering flags, no isoplanatism PASS/FAIL'],
        self_check_tolerances=dict(crop_abs_scaled=2e-6,shift_error_px=.5,zero_shift_px=.1))
    if points is not None:
        manifest.update(mode='points_2d',points=points,point_angle_deg=args.point_angle_deg,
            point_order=list(POINT_NAMES),corner_points=list(CORNER_NAMES),
            identity_checks=checks,convergence=convergence,
            formal_conclusion=formal_conclusion(checks,convergence),
            point_comparison='each point PSF vs the centre PSF translated by the expected image shift',
            pos_neg_rule='fixed-gain differential on fixed-ROI camera-pixel integrals')
    dump_json(out/'manifest.json',manifest)
    rows=[]
    try:
        with torch.no_grad():
            if points is None:
                for label,field in fields:
                    rows.extend(scan(field,g,paddings,args.axes,args.angles,roi,args.device,out,label,baseline_padding=baseline_padding,sample_offset=len(rows)))
            else:
                by_branch={}
                for label,field in fields:
                    branch_rows=scan_points(field,g,paddings,points,roi,args.device,out,label,
                                            camera_pixel_m,baseline_padding=baseline_padding,
                                            sample_offset=len(rows))
                    rows.extend(branch_rows); by_branch[label]=branch_rows
                gains={artifact.branch:artifact.gain_gray for artifact in artifacts}
                if len(by_branch)==2 and all(gains.get(label) is not None for label in by_branch):
                    differential=pos_neg_rows(by_branch,gains)
                    dump_json(out/'pos_neg_metrics.json',differential)
                    manifest['pos_neg']=dict(file='pos_neg_metrics.json',rows=len(differential),
                        definition='fixed-gain POS-NEG differential per point and padding')
                else:
                    manifest['pos_neg']=dict(file=None,rows=0,
                        definition='POS-NEG aggregation skipped: needs two branches with scalar gains')
        save_rows(out,rows)
        if args.self_check:
            checks=dict(crop=all(r['crop_max_abs_error']<=2e-6 for r in rows),
                shift=all(r['shift_error_dx_px'] is not None and abs(r['shift_error_dx_px'])<.5 and abs(r['shift_error_dy_px'])<.5 for r in rows),
                zero=all(abs(r['shift_measured_dx_px'])<.1 and abs(r['shift_measured_dy_px'])<.1 for r in rows if r['angle_deg']==0))
            manifest['self_check_results']=checks
            manifest['self_check_measurements']=dict(max_shift_error_px=max(max(abs(r['shift_error_dx_px']),abs(r['shift_error_dy_px'])) for r in rows),max_crop_error=max(r['crop_max_abs_error'] for r in rows),padding_statuses=sorted(set(r['padding_stability']['status'] for r in rows)))
            if not all(checks.values()):
                raise c.ValidationError('ideal engineering self-check failed; inspect saved evidence')
        manifest['status']='COMPLETED'
    except Exception as exc:
        manifest['status']='FAILED'; manifest['failure']=str(exc)
        raise
    finally:
        dump_json(out/'manifest.json',manifest)
    print(json.dumps(dict(status=manifest['status'],rows=len(rows),out=str(out),self_check_measurements=manifest.get('self_check_measurements'))))
    return manifest


if __name__=='__main__':
    try:
        main()
    except c.ValidationError as exc:
        raise SystemExit(str(exc))
