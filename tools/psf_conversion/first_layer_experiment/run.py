"""Paired Dataset509 fold-0 first-layer experiment. Run with an explicit fresh output."""
import argparse
import traceback
from pathlib import Path
import os
import shutil

from shared import (
    env_for, gate, need, pin, read, run, save, sha, write,
)

PRIOR = Path('C:/lijialin/results/segementation/runs/frozen_psf8_20261118_01')
BASELINE = '21ae3aee1028da900d6d92d5cf9b07a811ebf6fb3b05a1fb68119025c72e293a'
ENTRY = Path(__file__).resolve().parent
TOOLS = ENTRY.parent
CHECKOUT = TOOLS.parents[1]


def audit_source():
    specs = read(PRIOR/'manifest.private.json')
    spec, = (item for item in specs if item['key'] == '05_509_scratch')
    need(sha(spec['checkpoint']['path']) == spec['checkpoint']['sha256'] == BASELINE, 'WRONG_BASELINE')
    need(spec['pool'] is None and spec['trainer'] == 'nnUNetTrainerEarlyStopping', 'WRONG_TRAINING_PROTOCOL')
    model, raw = Path(spec['model']), Path(spec['raw'])
    split_path = Path(spec['pp'])/spec['dataset']/'splits_final.json'
    need(split_path.resolve() == Path(spec['split']['path']).resolve(), 'WRONG_SPLIT_PATH')
    need(sha(split_path) == spec['split']['sha256'], 'SPLIT_CHANGED')
    split = read(split_path)[0]
    cases = sorted(split['val'])
    need(len(split['train']) == 100 and len(cases) == 28
         and len(set(split['train'] + cases)) == 128, 'WRONG_FOLD_0')
    receipt = read(PRIOR/spec['key']/'audit.json')
    audit_path = Path(receipt.get('record') or receipt['reused']).resolve()
    need(audit_path.parent == (PRIOR/'cache_audits').resolve(), 'WRONG_AUDIT_ROOT')
    audit = read(audit_path)
    identity = audit['identity']
    need(audit['status'] == 'COMPLETED' and audit['verified_cases'] == 128
         and audit['corrected_source_labels'] and audit['cache_compared_to_current_official_preprocessing']
         and Path(identity['split']['path']).resolve() == split_path.resolve()
         and identity['split']['sha256'] == spec['split']['sha256']
         and Path(identity['plans']['path']).resolve() == (model/'plans.json').resolve()
         and sha(model/'plans.json') == identity['plans']['sha256'], 'AUDIT_IDENTITY_CHANGED')
    cache = Path(identity['cache_folder']).resolve()
    need(cache.parent == split_path.resolve().parent, 'WRONG_CACHE_ROOT')
    pins = {Path(path).resolve(): digest for path, digest in audit['source_and_cache_files'].items()}
    need({item.resolve() for item in cache.iterdir() if item.is_file()}
         == {path for path in pins if path.parent == cache}, 'CACHE_FILE_SET_CHANGED')
    required_raw = {path.resolve() for case in split['train'] + cases
                    for path in (raw/'imagesTr'/f'{case}_0000.nii.gz', raw/'labelsTr'/f'{case}.nii.gz')}
    need(required_raw <= pins.keys(), 'AUDIT_MISSING_CURRENT_RAW_INPUTS')
    pins.update({split_path: spec['split']['sha256'], Path(spec['checkpoint']['path']): BASELINE,
                 model/'plans.json': identity['plans']['sha256'], model/'dataset.json': sha(model/'dataset.json')})
    for path, digest in pins.items():
        need(sha(path) == digest, 'AUDITED_SOURCE_CHANGED')
    return spec, cases, pins, audit_path


def predict(job, stage, model, checkpoint, source, env, mode, psf=None):
    dest = job/stage
    args = [TOOLS/'predict_psf_first_layer.py', '--mode', mode, '--model-folder', model,
            '--fold', '0', '--checkpoint', checkpoint, '--input', source, '--output', dest,
            '--label', 'NON_FORMAL' if job.parent.name == 'preflight' else 'FOLD0_DEVELOPMENT']
    if psf is not None:
        args += ['--psf-dir', psf, '--operator', 'pixel_psf_superposition_v1']
    run(f'{job.name}_{stage}', args, env, 12000)
    record = read(dest/'psf_inference_record.json')
    need(record['status'] == 'COMPLETED' and record['first_kernel_size'] == int(job.name[1:])
         and record['checkpoint_sha256'] == sha(Path(model)/'fold_0'/checkpoint)
         and (not psf or record['hook_calls'] > 0), 'INVALID_PREDICTION_RECEIPT')
    return dest


def main(root, mode):
    preflight = mode == 'preflight'
    save(status='RUNNING', stage='SOURCE_AUDIT', formal_eligible=not preflight)
    queues = (PRIOR/'status.json', PRIOR.parent/'no_optics_ablation_20261118_01'/'status.json')
    queue_states = {str(path): read(path)['status'] for path in queues}
    write(root/'existing_queue_status.json', queue_states)
    need(all(status in ('COMPLETED', 'FAILED', 'COMPLETED_WITH_FAILURES')
             for status in queue_states.values()), 'EXISTING_EXPERIMENT_QUEUE_ACTIVE')
    gate('SOURCE_AUDIT', 0)
    spec, cases, pins, audit_path = audit_source()
    raw = Path(spec['raw'])
    inputs = root/'validation_inputs'
    inputs.mkdir()
    used_cases = cases[:1] if preflight else cases
    for case in used_cases:
        path = raw/'imagesTr'/f'{case}_0000.nii.gz'
        target = inputs/path.name
        shutil.copyfile(path, target)
        need(sha(target) == pins[path.resolve()], 'STAGED_IMAGE_CHANGED')
    runtime = Path(spec['runtime'])
    env = env_for(runtime)
    env.update(nnUNet_preprocessed=spec['pp'], nnUNet_raw=str(raw.parent),
               nnUNet_results=str(root/'results'))
    env['PYTHONPATH'] = os.pathsep.join([str(CHECKOUT), str(runtime/'src'), str(runtime/'trainers')])
    source_paths = [*ENTRY.glob('*.py'), *TOOLS.glob('*.py'), *(TOOLS/'optics').glob('*.py'),
                    CHECKOUT/'evaluate_segmentation_metrics.py']
    write(root/'protocol.json', dict(fold=0, validation_volumes=len(used_cases), audit=pin(audit_path),
          baseline=spec['checkpoint'], formal_eligible=not preflight,
          electrical=dict(updates=1 if preflight else 250, lr=0.001, seed=12345,
                          sampling='original trainer, one augmentation process, network eval mode'),
          fit=dict(wavelength_nm=532, fft_size=3600, n=800, dx_nm=400, distance_um=2400,
                   sample_cell=17, steps=1 if preflight else 1000, seed=42, patience=30, lr=0.005),
          source={str(p.relative_to(CHECKOUT)): sha(p) for p in source_paths}))
    score_dirs = {}
    for size in (3, 5, 7):
        job = root/f'k{size}'
        job.mkdir()
        write(job/'spec.private.json', dict(spec, kernel_size=size))
        run(f'k{size}_target', [ENTRY/'training_adapter.py', job, 'target', mode], env, 20000)
        target = read(job/'target_completed.json')['checkpoint']
        checkpoint = Path(target['path'])
        model = checkpoint.parents[1]
        psf = job/'psf532'
        run(f'k{size}_fit', [TOOLS/'fit_psf_gray.py', '--kernels', job/'electrical_kernels.npy',
             '--target-checkpoint-sha256', target['sha256'], '--pixel-superposition',
             '--wavelength-nm', 532, '--fft-size', 3600, '--n', 800, '--dx-nm', 400,
             '--distance-um', 2400, '--steps', 1 if preflight else 1000, '--patience', 30,
             '--lr', 0.005, '--seed', 42, '--device', 'cuda:0', '--out', psf], env, 3000)
        need(len(read(psf/'summary.json')) == 64, 'INCOMPLETE_FIT')
        predictions = {
            'electrical': predict(job, 'electrical', model, checkpoint.name, inputs, env, 'electrical'),
            'fitted': predict(job, 'fitted', model, checkpoint.name, inputs, env, 'fitted', psf),
        }
        hashes = read(predictions['fitted']/'psf_inference_record.json')['psf_sha256']
        write(job/'spec.private.json', dict(spec, kernel_size=size, checkpoint=target,
              model=str(model), psf=str(psf), psf_hashes=hashes))
        run(f'k{size}_backend', [ENTRY/'training_adapter.py', job, 'backend', mode], env, 20000)
        backend = job/'backend'/'model'
        if preflight:
            need(read(job/'preflight.json')['checkpoint_roundtrip'], 'PREFLIGHT_ROUNDTRIP_FAILED')
            best = 'roundtrip_NON_FORMAL.pth'
        else:
            result = read(job/'training_completed.json')
            best = Path(result['checkpoint']['path']).name
            need(result['status'] == 'COMPLETED' and sha(backend/'fold_0'/best)
                 == result['checkpoint']['sha256'], 'INVALID_BACKEND_CHECKPOINT')
        predictions['finetuned'] = predict(job, 'finetuned', backend, best, inputs, env, 'fitted', psf)
        for stage, path in predictions.items():
            dest = job/(stage+'_metrics')
            run(f'k{size}_{stage}_score', [ENTRY/'score.py', path, inputs, raw/'labelsTr', dest], env, 0)
            score_dirs[f'k{size}_{stage}'] = str(dest)
    write(root/'score_dirs.json', score_dirs)
    run('paired_scores', [ENTRY/'score.py', root], env, 0)
    for path, digest in pins.items():
        need(sha(path) == digest, 'AUDITED_SOURCE_CHANGED_DURING_RUN')
    save(status='COMPLETED', stage='DONE', formal_eligible=not preflight)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_parent', type=Path, help='Fresh experiment directory under runs')
    parser.add_argument('mode', choices=('preflight', 'formal'))
    options = parser.parse_args()
    root = options.output_parent.resolve()
    if root.parent != PRIOR.resolve().parent or not root.name.startswith('optical_first_conv_'):
        parser.error('Use a dedicated optical_first_conv_* directory directly under runs')
    if options.mode == 'formal':
        receipt = read(root/'preflight'/'status.json')
        need(receipt['status'] == 'COMPLETED' and receipt['formal_eligible'] is False, 'PREFLIGHT_REQUIRED')
        protocol = read(root/'preflight'/'protocol.json')
        need(all(sha(CHECKOUT/path) == digest for path, digest in protocol['source'].items()),
             'SOURCE_CHANGED_AFTER_PREFLIGHT')
    destination = root/options.mode
    import shared
    shared.ROOT = destination
    destination.mkdir(parents=True, exist_ok=False)
    try:
        main(destination, options.mode)
    except Exception as error:
        if destination.is_dir():
            save(status='FAILED', error=str(error))
        traceback.print_exc()
        raise SystemExit(1)
