"""Check-only (no model construction) and explicit short engineering training."""
import argparse
import math
from pathlib import Path
from experiments.genseg_joint.runtime import TARGET_SHA256


def positive(value):
    number=int(value)
    if number<=0: raise argparse.ArgumentTypeError('Explicit positive steps required')
    return number


def rate(value):
    number=float(value)
    if not math.isfinite(number) or number<=0: raise argparse.ArgumentTypeError('Finite positive value required')
    return number


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    for command in ('check','train'):
        a=sub.add_parser(command)
        for field in ('source','raw-root','targets','baseline-run','output'):
            a.add_argument('--'+field,required=True,type=Path)
        a.add_argument('--targets-sha256',default=TARGET_SHA256,choices=[TARGET_SHA256])
        a.add_argument('--seed',type=int,default=20261018)
        a.add_argument('--gpu',required=command=='train',type=int,help='physical index; check telemetry only, train binds UUID')
        if command=='train':
            a.add_argument('--steps',required=True,type=positive)
            a.add_argument('--pretrain-steps',type=int,default=0,choices=[0],help='this pilot supports coldstart only')
            for key,default in [('g-lr',2e-5),('d-lr',2e-5),('s-lr',1e-5),('arch-lr',1e-4),('l1-weight',100.),
                                ('min-free-vram-gib',12.),('min-free-ram-gib',8.)]:
                a.add_argument('--'+key,type=rate,default=default)
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    if args.gpu is not None:
        from .runtime import gpu_guard
        guard_options={} if args.command=='check' else dict(min_free_vram_gib=args.min_free_vram_gib,min_free_ram_gib=args.min_free_ram_gib)
        gpu_guard(args.gpu,**guard_options)  # telemetry only; no CUDA/model load
    from experiments.genseg30.contract import Source,require,check_output
    from experiments.genseg3mm.targets import TargetDataset
    from .data import inner_split
    from .models import source_identity
    require(0<=args.seed<2**32,'Seed must be uint32')
    source=Source(args.source,args.raw_root)
    source.require_no_unpack_writes()
    # No G/D/S are constructed; old checkpoint bytes are hashed only for target provenance.
    target=TargetDataset(source,args.targets,args.baseline_run,expected_manifest_sha256=args.targets_sha256)
    check_output(args.output,target.protected)
    split=inner_split(target.train,target.val,args.seed)
    identity=source_identity()
    if args.command=='check':
        print('CHECK PASS: inner61/internal15/external19 = '+str(tuple(len(split[k]) for k in ('train','validation','external'))))
        print('No patient image/label arrays, model weights, CUDA context or output directory loaded/created. Target mask caches and provenance hashes audited.')
        print('Fresh G/D/S; coldstart only; native full planes; source='+identity['commit'])
        return 0
    from .runtime import run
    options=vars(args).copy()
    for key in ('command','source','raw_root','targets','baseline_run','targets_sha256'):
        options.pop(key)
    run(target,**options)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
