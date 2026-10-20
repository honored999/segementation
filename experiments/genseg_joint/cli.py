"""Check-only (no model construction) and explicit short engineering training."""
import argparse
import math
from pathlib import Path
from experiments.genseg_joint.convergence import CONVERGENCE_STEPS
from experiments.genseg_joint.runtime import TARGET_SHA256


def positive(value):
    number=int(value)
    if number<=0: raise argparse.ArgumentTypeError('Explicit positive steps required')
    return number


def rate(value):
    number=float(value)
    if not math.isfinite(number) or number<=0: raise argparse.ArgumentTypeError('Finite positive value required')
    return number


def patient_limit(value):
    number=int(value)
    if not 1<=number<=61: raise argparse.ArgumentTypeError('Patient limit must be 1..61')
    return number


def digest(value):
    import re
    if not re.fullmatch('[0-9a-f]{64}',value): raise argparse.ArgumentTypeError('Lowercase SHA256 required')
    return value


class ExperimentParser(argparse.ArgumentParser):
    def parse_args(self,args=None,namespace=None):
        a=super().parse_args(args,namespace)
        if a.command in ('pretrain','diagnose-single-slice'):
            repair=a.strategy=='adam_trainbn_epoch_v1'
            if a.command=='pretrain' and ((repair and a.epochs is None) or (not repair and a.epochs is not None)):
                self.error('--epochs is required only with --strategy adam_trainbn_epoch_v1')
            if a.command=='diagnose-single-slice':
                if a.initialization=='cold':
                    if not repair or a.pretrained_run is not None or a.pretrained_sha256 is not None:
                        self.error('cold requires repair strategy and no pretrained source')
                elif a.pretrained_run is None or a.pretrained_sha256 is None:
                    self.error('gd500 requires paired --pretrained-run and --pretrained-sha256')
                if a.g_objective=='l1_only' and not (repair and a.initialization=='cold'):
                    self.error('--g-objective l1_only requires --strategy adam_trainbn_epoch_v1 and --initialization cold')
                if a.save_mode_previews and not (a.compare_bn_modes or a.compare_bn_calibration):
                    self.error('--save-mode-previews requires --compare-bn-modes or --compare-bn-calibration')
                if a.convergence_audit:
                    if not (repair and a.initialization=='cold' and a.g_objective=='gan_l1'):
                        self.error('--convergence-audit requires --strategy adam_trainbn_epoch_v1, --initialization cold and --g-objective gan_l1')
                    if a.steps!=CONVERGENCE_STEPS:
                        self.error('--convergence-audit requires exactly 500 updates')
        return a


def parser():
    p=ExperimentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    for command in ('check','train','pretrain','preview','diagnose-single-slice','preview-diagnostic'):
        a=sub.add_parser(command)
        for field in ('source','raw-root','targets','baseline-run','output'):
            a.add_argument('--'+field,required=True,type=Path)
        a.add_argument('--targets-sha256',default=TARGET_SHA256,choices=[TARGET_SHA256])
        a.add_argument('--seed',type=int,default=20261018,**({'choices':[20261018]} if command=='diagnose-single-slice' else {}))
        a.add_argument('--gpu',required=command!='check',type=int,help='physical index; check telemetry only, train binds UUID')
        if command=='preview':
            a.add_argument('--phase',choices=('joint','gan_pretrain'),default='joint')
            a.add_argument('--run',required=True,type=Path)
            a.add_argument('--checkpoint-sha256',required=True,type=digest)
            a.add_argument('--patient-limit',type=patient_limit,default=4)
        if command=='preview-diagnostic':
            a.add_argument('--run',required=True,type=Path)
            a.add_argument('--checkpoint-sha256',required=True,type=digest)
        if command in ('pretrain','diagnose-single-slice'):
            a.add_argument('--strategy',choices=('legacy','adam_trainbn_epoch_v1'),default='legacy')
        if command=='pretrain':
            a.add_argument('--epochs',type=positive,help='repair: complete inner61 passes; --steps is max update cap')
        if command=='diagnose-single-slice':
            a.add_argument('--initialization',choices=('gd500','cold'),default='gd500')
            a.add_argument('--compare-bn-modes',action='store_true',help='observe G batchstats with private buffers; eval alone decides quality')
            a.add_argument('--compare-bn-calibration',action='store_true',help='after-only private one-pass BN calibration; implies BN mode comparison; eval quality unchanged')
            a.add_argument('--save-mode-previews',action='store_true',help='after-only PNGs from the existing batchstats/private-calibrated-eval arrays; requires a BN comparison switch; default off')
            a.add_argument('--convergence-audit',action='store_true',
                help='fixed 500-update bounded convergence plan (before/100/250/500) with one diagnostic-only state snapshot; requires --strategy adam_trainbn_epoch_v1 --initialization cold --g-objective gan_l1; default off')
            a.add_argument('--pretrained-run',type=Path)
            a.add_argument('--pretrained-sha256',type=digest)
            a.add_argument('--steps',required=True,type=positive)
            a.add_argument('--g-objective',choices=('gan_l1','l1_only'),default='gan_l1',
                help='explicit G objective; l1_only excludes the adversarial term from G gradients and is cold+repair diagnostic only')
        if command in ('check','train'):
            a.add_argument('--pretrained-run',type=Path)
            a.add_argument('--pretrained-sha256')
        if command not in ('check','preview','diagnose-single-slice','preview-diagnostic'):
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
    if args.command in ('check','train'):
        require((args.pretrained_run is None)==(args.pretrained_sha256 is None),'Paired pretrained-run and pretrained-sha256 required')
        if args.pretrained_sha256 is not None:
            import re
            require(re.fullmatch('[0-9a-f]{64}',args.pretrained_sha256) is not None,'Explicit lowercase checkpoint SHA256 required')
    if args.gpu is not None:
        from .runtime import gpu_guard
        guard_options={} if args.command in ('check','preview','diagnose-single-slice','preview-diagnostic') else dict(min_free_vram_gib=args.min_free_vram_gib,min_free_ram_gib=args.min_free_ram_gib)
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
    if args.command=='diagnose-single-slice':
        from .diagnostic import run_single_slice
        comparison_options={'compare_bn_modes':True} if args.compare_bn_modes else {}
        if args.compare_bn_calibration: comparison_options['compare_bn_calibration']=True
        run_single_slice(target,args.output,pretrained_run=args.pretrained_run,
            pretrained_sha256=args.pretrained_sha256,gpu=args.gpu,steps=args.steps,seed=args.seed,
            strategy=args.strategy,initialization_mode=args.initialization,
            g_objective=args.g_objective,save_mode_previews=args.save_mode_previews,
            convergence_audit=args.convergence_audit,**comparison_options)
        return 0
    if args.command=='preview-diagnostic':
        from .diagnostic_preview import run_preview_diagnostic
        run_preview_diagnostic(target,args.run,args.checkpoint_sha256,args.output,gpu=args.gpu)
        return 0
    if args.command=='preview':
        from .preview import run_preview
        run_preview(target,args.run,args.checkpoint_sha256,args.output,gpu=args.gpu,patient_limit=args.patient_limit,phase=args.phase)
        return 0
    if args.command=='check':
        if args.pretrained_run is not None:
            from .pretrain import audit_pretrained,policy
            audit_pretrained(args.pretrained_run,args.pretrained_sha256,for_joint=True,expected=dict(policy(),seed=args.seed,
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
