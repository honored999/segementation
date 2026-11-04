"""Independent tilted-plane-wave validation; never optimizes a width map."""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
import json
import math
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
        proc = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                               '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
        for line in proc.stdout.strip().splitlines():
            util, used, total = map(float, line.split(','))
            record['gpus'].append(dict(util_percent=util, used_mib=used, total_mib=total,
                                       memory_percent=100*used/total))
    except (OSError, subprocess.CalledProcessError) as exc:
        record['gpu_query_unavailable'] = str(exc)
    return record


def guard(record, n, paddings, device):
    # Conservative allocation budget including FFT, diagnostics and plotting arrays.
    estimate = (n+2*max(paddings))**2 * 512
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
    for padding in paddings:
        reference_full, reference_field, _ = engine.extended(field,padding)
        reference = c.intensity(reference_field)
        reference_cpu = reference.detach().cpu()
        reference_energy = c.energy(reference,g.dx_m)
        fixed_reference_window = c.roi_window(g.n,roi_size)
        reference_roi_energy = c.roi_sum(reference,fixed_reference_window)*g.dx_m**2
        reference_extended_energy = c.energy(c.intensity(reference_full),g.dx_m)
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
                with (out/npz_file).open('xb') as handle:
                    np.savez_compressed(handle,reference=arr(reference),psf=arr(psf),
                        sample_id=np.array(sample_id),angle_deg=np.array(angle),padding=np.array(padding),
                        field=arr(propagated),extended_field=arr(full),
                        aligned=np.array([]) if aligned is None else arr(aligned),
                        valid_mask=np.zeros((g.n,g.n),bool) if mask is None else arr(mask),
                        theory_aligned=arr(theory_aligned),theory_mask=arr(theory_mask))
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
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    vmax=max(float(max(ref.max(),psf.max())) for _,ref,psf,_,_ in images)
    for stem,ref,psf,aligned,mask in images:
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
    return rows


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
    args=parser.parse_args(argv)
    if args.self_check and (args.artifact or args.config):
        parser.error('self-check and real artifacts are separate run classes')
    if not args.self_check and (not args.artifact or not args.config):
        parser.error('BLOCKED_MISSING_OPTIMIZED_ARTIFACT: supply --artifact and --config')
    if not all(math.isfinite(a) and abs(a)<90 for a in args.angles):
        parser.error('angles must be finite and strictly between -90 and 90 degrees')
    torch.set_num_threads(1)
    source=Path(__file__).resolve().parent
    locks=c.verify_locked_sources(source)
    inputs=[]; skipped=[]
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
            if artifact.skipped:
                skipped.append(dict(artifact=str(path),reason='empty width: zero branch, no physical PSF'))
            else:
                fields.append((artifact.branch,artifact.width_nm))
        if len({name for name,_ in fields})!=len(fields):
            parser.error('artifact labels must be distinct')
        if config.padding not in paddings:
            parser.error('padding comparison must include source baseline padding')
    if not paddings or min(paddings)<0 or len(set(paddings))!=len(paddings):
        parser.error('paddings must be distinct nonnegative integers')
    baseline_padding = g.n if args.self_check else config.padding
    if baseline_padding not in paddings:
        parser.error("padding comparison must include explicit baseline padding")
    c.roi_window(g.n,roi)
    resources=resource_preflight(); guard(resources,g.n,paddings,args.device)
    if not args.self_check:
        fields=[(label,c.build_transmission(width)) for label,width in fields]
    out=prepare_output(args.out,inputs)
    manifest=dict(status='RUNNING',run_class='synthetic_engineering' if args.self_check else 'optimized_artifact_validation',
        arguments={k:str(v) if isinstance(v,Path) else [str(p) for p in v] if k=='artifact' and v else v for k,v in vars(args).items()},
        geometry=asdict(g),resources=resources,locked_sources=locks,
        baseline_padding=baseline_padding,
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
    dump_json(out/'manifest.json',manifest)
    rows=[]
    try:
        with torch.no_grad():
            for label,field in fields:
                rows.extend(scan(field,g,paddings,args.axes,args.angles,roi,args.device,out,label,baseline_padding=baseline_padding,sample_offset=len(rows)))
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
