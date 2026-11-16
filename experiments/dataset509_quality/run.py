"""Separate, never-overwriting preflight/train entry. No preparation or notifications."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from quality import need, read, sha, write_new

DEPENDENCY_COMMIT = 'c498118e6e3b3e4bb7a1c2380bb777c13f601e04'


def pin(record):
    need(sha(record['path']) == record['sha256'], 'INPUT_PIN_CHANGED')
    return Path(record['path']).resolve()


def check_inputs(c, root):
    need(c['mode'] in ('baseline', 'q', 'quality_filter', 'c'), 'INVALID_MODE')
    need(c['seed'] == 12345 and c['configuration'] == '2d' and type(c['fold']) is int and c['fold'] == 0,
         'FIXED_PROTOCOL_REQUIRED')
    repo = Path(c['lesiongen_repo']).resolve()
    head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    need(head == DEPENDENCY_COMMIT, 'DEPENDENCY_COMMIT_CHANGED')
    # A matching HEAD alone does not establish an unchanged dependency checkout.
    files = ['src/lesiongen/segmentation512.py',
             'trainers/nnUNetTrainerLesionGen512TopK10EarlyStopping.py',
             'trainers/nnUNetTrainerTopK10EarlyStopping.py',
             'trainers/nnUNetTrainerTopK10.py', 'trainers/nnUNetTrainerEarlyStopping.py']
    for rel in files:
        blob = subprocess.check_output(['git', '-C', str(repo), 'show', f'{DEPENDENCY_COMMIT}:{rel}'])
        current = (repo / rel).read_bytes().replace(b'\r\n', b'\n')
        need(current == blob.replace(b'\r\n', b'\n'), 'DIRTY_DEPENDENCY_SOURCE')
    prepared = read(pin(c['preparation']))
    need(prepared['status'] == 'COMPLETE' and prepared['synthetic_planes'] == 478 and
         prepared['validation_full_volumes'] == 28, 'PREPARATION_INVALID')
    base = Path(prepared['base']).resolve()
    pool = Path(prepared['pool_manifest']).resolve().parent
    need(base in pool.parents, 'POOL_OUTSIDE_PREPARED_BASE')
    need(sha(pool / 'manifest.json') == prepared['pool_manifest_sha256'], 'POOL_CHANGED')
    pm = read(pool / 'manifest.json')
    plans_path = pin(c['plans'])
    need(plans_path == Path(prepared['plans']).resolve() and sha(plans_path) == pm['plans_sha256'], 'PLANS_CHANGED')
    split_path = pin(c['split'])
    need(split_path == base / 'splits_final.json' and sha(split_path) == pm['split_sha256'], 'SPLIT_CHANGED')
    split = read(split_path)[0]
    need(len(split['train']) == 100 and len(set(split['train'])) == 100 and len(split['val']) == 28 and
         len(set(split['val'])) == 28 and not set(split['train']) & set(split['val']), 'SPLIT_INVALID')
    need(set(pm['cases']) <= set(split['train']) and sum(r['count'] for r in pm['cases'].values()) == 478,
         'TRAIN_POOL_INVALID')
    for r in pm['cases'].values():
        path = (pool / r['file']).resolve()
        need(pool in path.parents and sha(path) == r['sha256'], 'POOL_CONTENT_CHANGED')
    pretrained = pin(c['pretrained'])
    need(c['pretrained']['dataset_id'] == 508, 'ONLY_508_INITIALIZATION_SUPPORTED')
    protected = [repo, base, pretrained, plans_path, split_path, Path(c['preparation']['path']).resolve()]
    if c['mode'] != 'baseline':
        sidecar = pin(c['curriculum_sidecar'] if c['mode'] == 'c' else c['sidecar'])
        protected.append(sidecar)
        quality = read(sidecar)
        need(quality['pool_manifest_sha256'] == prepared['pool_manifest_sha256'] and
             quality['split_sha256'] == c['split']['sha256'], 'SIDECAR_BINDING_INVALID')
    for p in protected:
        need(root != p and root not in p.parents and p not in root.parents, 'OUTPUT_OVERLAPS_INPUT')
    plans = read(plans_path)
    need(plans['dataset_name'] == 'Dataset514_CI1DWILesionGen30', 'DATASET514_REQUIRED')
    cache = base / plans['configurations']['2d']['data_identifier']
    # Never unpack/write the shared prepared cache from this entry.
    for cid in split['train'] + split['val']:
        need(all((cache / (cid + suffix)).is_file() for suffix in ('.npy', '_seg.npy', '.pkl')),
             'EXISTING_NUMPY_CACHE_BARRIER_REQUIRED')
    policy = c['early_stopping']
    need(type(policy['max_epochs']) is int and type(policy['min_epochs']) is int and
         type(policy['patience']) is int and 1 <= policy['max_epochs'] and 0 <= policy['min_epochs'] <= policy['max_epochs'] and
         policy['patience'] > 0 and 0 <= policy['min_delta'] < 1, 'STOPPING_POLICY_INVALID')
    need(type(c['iterations_per_epoch']) is int and c['iterations_per_epoch'] > 0 and
         type(c['validation_iterations']) is int and c['validation_iterations'] > 0, 'UPDATE_BUDGET_REQUIRED')
    need(type(c['workers']) is int and 0 <= c['workers'] <= 4, 'WORKER_BUDGET_0_TO_4_REQUIRED')
    if c['mode'] == 'c':
        from curriculum import validate_policy
        from quality import digest
        validate_policy(c['curriculum'])
        need(c['workers'] == 0 and c['batch_size'] == 12, 'C_WORKERS0_BATCH12_REQUIRED')
        need(quality['schema'] == 'dataset509-curriculum/v1' and
             quality['policy_sha256'] == digest(c['curriculum']) == digest(quality['policy']) and
             quality['plans_sha256'] == c['plans']['sha256'], 'C_SIDECAR_POLICY_PLANS_CHANGED')
    else:
        need(isinstance(c['synthetic_coefficient'], (float, int)) and
             0 <= c['synthetic_coefficient'] <= 10 and 0 <= c['filter_threshold'] <= 1, 'QUALITY_PARAMETERS_INVALID')
    sys.path[:0] = [str(repo / 'src'), str(repo / 'trainers')]
    return prepared, plans


def run(c, root, stage, gpu_uuid):
    c = dict(c)
    c.setdefault('mode', 'baseline')
    need(not root.exists(), 'NEW_OUTPUT_ROOT_REQUIRED')
    need(gpu_uuid.startswith('GPU-') and ',' not in gpu_uuid and len(gpu_uuid) > 8, 'ONE_GPU_UUID_REQUIRED')
    ready, plans = check_inputs(c, root)
    # Must precede importing nnunetv2.paths (including trainer imports).
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu_uuid, nnUNet_preprocessed=str(Path(ready['base']).parent),
                      nnUNet_results=str(root / 'results'), nnUNet_compile='false',
                      nnUNet_n_proc_DA=str(c['workers']), OMP_NUM_THREADS='1',
                      LESIONGEN512_POOL=str(Path(ready['pool_manifest']).parent),
                      LESIONGEN512_POOL_SHA256=ready['pool_manifest_sha256'])
    import numpy as np
    import torch
    from api_contract import verify
    api = verify()
    need(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'ONE_VISIBLE_CUDA_DEVICE_REQUIRED')
    # Only an immediate headroom guard; main agent still checks all host resources before launch.
    free, total = torch.cuda.mem_get_info(0)
    need(free / total > .2, 'GPU_HEADROOM_BELOW_20_PERCENT')
    root.mkdir(parents=True, exist_ok=False)
    config_path = root / 'config.json'
    write_new(config_path, c)
    os.environ['DATASET509_QUALITY_CONFIG'] = str(config_path)
    write_new(root / 'runtime.json', dict(api=api, torch=torch.__version__, gpu_uuid=gpu_uuid,
                                        stage=stage, formal_eligible=stage == 'train', seed=12345,
                                        dependency_commit=DEPENDENCY_COMMIT))
    trainer = None
    try:
        from trainer import trainer_class
        from nnunetv2.run.load_pretrained_weights import load_pretrained_weights
        random.seed(12345)
        np.random.seed(12345)
        torch.manual_seed(12345)
        torch.cuda.manual_seed_all(12345)
        torch.set_num_threads(1)
        Trainer = trainer_class(c['mode'])
        trainer = Trainer(plans, '2d', 0, read(Path(ready['base']) / 'dataset.json'), torch.device('cuda:0'))
        policy = c['early_stopping']
        for key, attr in [('max_epochs', 'MAX_EPOCHS'), ('min_epochs', 'MIN_TRAINING_EPOCHS'),
                          ('patience', 'PATIENCE'), ('min_delta', 'MIN_DELTA')]:
            setattr(trainer, attr, policy[key])
        trainer.num_epochs = policy['max_epochs']
        trainer.num_iterations_per_epoch = c['iterations_per_epoch']
        trainer.num_val_iterations_per_epoch = c['validation_iterations']
        trainer.initialize()
        if c['mode'] == 'c':
            need(trainer.batch_size == c['batch_size'], 'C_PLANS_BATCH_SIZE_MISMATCH')
        need(trainer.lr_scheduler.max_steps == policy['max_epochs'], 'LR_HORIZON_MISMATCH')
        load_pretrained_weights(trainer.network, c['pretrained']['path'], verbose=True)
        need(trainer.current_epoch == 0 and not trainer.optimizer.state and
             trainer._early_stopping_best_ema is None and trainer._epochs_without_improvement == 0 and
             not trainer._early_stopping_triggered, 'FRESH_OPTIMIZER_AND_STOPPING_REQUIRED')
        if stage == 'preflight':
            trainer.on_train_start()
            training = trainer.train_step(next(trainer.dataloader_train))
            validation = trainer.validation_step(next(trainer.dataloader_val))
            need(np.isfinite(training['loss']).all() and np.isfinite(validation['loss']).all(), 'NONFINITE_PREFLIGHT')
            result = dict(status='COMPLETED', formal_eligible=False, training_steps=1, validation_steps=1,
                          train_loss=float(training['loss']), validation_loss=float(validation['loss']),
                          source_exposure=getattr(trainer, 'exposure', None))
        else:
            trainer.run_training()
            result = dict(status='COMPLETED', formal_eligible=True, completed_epochs=trainer.current_epoch,
                          best_checkpoint=str(Path(trainer.output_folder) / 'checkpoint_best.pth'),
                          original_space_evaluation='PENDING_MAIN_AGENT')
        write_new(root / 'status.json', result)
    except BaseException as exc:
        write_new(root / 'status.json', dict(status='FAILED', error_type=type(exc).__name__, formal_eligible=False))
        raise
    finally:
        if trainer is not None:
            for name in ('dataloader_train', 'dataloader_val'):
                loader = getattr(trainer, name, None)
                if hasattr(loader, '_finish'):
                    loader._finish()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--stage', required=True, choices=['preflight', 'train'])
    p.add_argument('--gpu-uuid', required=True)
    a = p.parse_args()
    run(read(a.config), Path(a.output).resolve(), a.stage, a.gpu_uuid)
