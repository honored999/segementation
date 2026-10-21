"""Explicit new joint strategy, isolated from legacy CLI defaults."""
import argparse
import json
from pathlib import Path
from experiments.genseg30.contract import require,Source,check_output
from .runtime import TARGET_SHA256
from .joint_repair import STRATEGY
from .cli import positive,digest


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('check','train','preview'):
        a=sub.add_parser(name)
        a.add_argument('--strategy',required=True,choices=[STRATEGY])
        for field in ('source','raw-root','targets','baseline-run','output'):
            a.add_argument('--'+field,required=True,type=Path)
        a.add_argument('--targets-sha256',default=TARGET_SHA256,choices=[TARGET_SHA256])
        a.add_argument('--gpu',type=int,default=1)
        if name=='preview':
            a.add_argument('--run',required=True,type=Path)
            a.add_argument('--checkpoint-sha256',required=True,type=digest)
            a.add_argument('--patient-limit',type=positive,default=4)
        else:
            a.add_argument('--pretrained-run',required=True,type=Path)
            a.add_argument('--pretrained-sha256',required=True,type=digest)
            a.add_argument('--cache-budget-bytes',required=True,type=positive)
            a.add_argument('--loader-workspace-bytes',required=True,type=positive)
            a.add_argument('--model-reserve-bytes',required=True,type=positive)
        if name=='train':
            a.add_argument('--steps',required=True,type=positive)
    return p


def main(argv=None):
    a=parser().parse_args(argv)
    require(a.gpu>=0,'Nonnegative physical GPU required')
    from .joint_resources import ResourceGuard
    guard=ResourceGuard(a.gpu); guard()
    from experiments.genseg3mm.targets import TargetDataset
    source=Source(a.source,a.raw_root); source.require_no_unpack_writes()
    target=TargetDataset(source,a.targets,a.baseline_run,expected_manifest_sha256=a.targets_sha256)
    protected=list(target.protected)+([a.run] if a.command=='preview' else [a.pretrained_run])
    check_output(a.output,protected)
    if a.command=='check':
        from .joint_runtime import check
        print(json.dumps(check(target,a.pretrained_run,a.pretrained_sha256,a.cache_budget_bytes,a.loader_workspace_bytes,a.model_reserve_bytes),allow_nan=False))
    elif a.command=='train':
        from .joint_runtime import run
        run(target,a.output,pretrained_run=a.pretrained_run,pretrained_sha256=a.pretrained_sha256,
            steps=a.steps,gpu=a.gpu,cache_budget_bytes=a.cache_budget_bytes,model_reserve_bytes=a.model_reserve_bytes,loader_workspace_bytes=a.loader_workspace_bytes)
    else:
        from .preview import run_preview
        run_preview(target,a.run,a.checkpoint_sha256,a.output,gpu=a.gpu,patient_limit=a.patient_limit,phase='joint_repair')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
