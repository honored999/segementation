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
    for command in ('check','train','pretrain'):
        a=sub.add_parser(command)
        for field in ('source','raw-root','targets','baseline-run','output'):
            a.add_argument('--'+field,required=True,type=Path)
        a.add_argument('--targets-sha256',default=TARGET_SHA256,choices=[TARGET_SHA256])
        a.add_argument('--seed',type=int,default=20261018)
        a.add_argument('--gpu',required=command!='check',type=int,help='physical index; check telemetry only, train binds UUID')
        if command in ('check','train'):
            a.add_argument('--pretrained-run',type=Path)
            a.add_argument('--pretrained-sha256')
        if command!='check':
            a.add_argument('--steps',required=True,type=positive)
            if command=='train':
                a.add_argument('--pretrain-steps',type=int,default=0,choices=[0],help='legacy cold-start flag; only 0 accepted, no combined phase/resume')
            for key,default in [('g-lr',2e-5),('d-lr',2e-5),('s-lr',1e-5),('arch-lr',1e-4),('l1-weight',100.),
                                ('min-free-vram-gib',12.),('min-free-ram-gib',8.)]:
                if command=='pretrain' and key in ('s-lr','arch-lr'): continue
                a.add_argument('--'+key,type=rate,default=default)
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    from experiments.genseg30.contract import require
    require(args.seed==20261018,'Fixed inner61/15 seed 20261018 required')
    if args.command!='pretrain':
        require((args.pretrained_run is None)==(args.pretrained_sha256 is None),'Paired pretrained-run and pretrained-sha256 required')
        if args.pretrained_sha256 is not None:
            import re
            require(re.fullmatch('[0-9a-f]{64}',args.pretrained_sha256) is not None,'Explicit lowercase checkpoint SHA256 required')
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
    protected=list(target.protected)
    if getattr(args,'pretrained_run',None) is not None: protected.append(args.pretrained_run)
    check_output(args.output,protected)
    split=inner_split(target.train,target.val,args.seed)
    identity=source_identity()
    if args.command=='check':
        if args.pretrained_run is not None:
            from .pretrain import audit_pretrained,policy
            audit_pretrained(args.pretrained_run,args.pretrained_sha256,expected=dict(policy(),seed=args.seed,
                split=split,target_identity=target.identity,target_manifest_sha256=target.manifest_sha256,
                model_config={'G':{'width':64,'downs':8,'seed':args.seed},'D_width':64}))
        print('CHECK PASS: inner61/internal15/external19 = '+str(tuple(len(split[k]) for k in ('train','validation','external'))))
        print('No patient image/label arrays, model weights, CUDA context or output directory loaded/created. Target mask caches and provenance hashes audited.')
        print('Native full planes; completed-pretrain initialization only; no resume; source='+identity['commit'])
        return 0
    from .runtime import run
    options=vars(args).copy()
    for key in ('command','source','raw_root','targets','baseline_run','targets_sha256'):
        options.pop(key)
    run(target,phase='gan_pretrain' if args.command=='pretrain' else 'joint',**options)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
