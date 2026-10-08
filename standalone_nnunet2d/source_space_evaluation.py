"""Prepared H2Former fold inference with nnU-Net 2.8.1 source-space export.

No training or GT-dependent inference. Existing raw prediction remains a separate
legacy entry. A pending checkpoint remains pending even after successful export.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from collections.abc import Mapping
import hashlib
import importlib.metadata
import io
import json
import math
import shutil
import tempfile
from pathlib import Path
import pickle
import subprocess
from typing import Sequence

import numpy as np
import torch

from standalone_nnunet2d.alignment_evidence import validate_checkpoint_alignment_metadata
from standalone_nnunet2d.data.dataset import read_splits
from standalone_nnunet2d.data.data_source import _validate_b2nd_store_pair
from standalone_nnunet2d.data.nifti_io import read_nifti
from standalone_nnunet2d.engine.predictor import predict_logits_2d, save_and_validate_prediction
from standalone_nnunet2d.models.factory import build_model, get_model_contract, resolve_checkpoint_model_identity
from standalone_nnunet2d.training.formal_checkpoint import (
    compute_plan_hash, read_formal_payload, validate_initialization_provenance,
    validate_source_schema, validate_model_tensors,
)
from standalone_nnunet2d.training.official_config import DEFAULT_RUN_STATE


def read_fixed_splits():
    return read_splits()


def canonical_splits(folds):
    return [{key: sorted(fold[key]) for key in ('train', 'val')} for fold in folds]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_hashes(hashes):
    for path, expected in hashes.items():
        if sha256(Path(path)) != expected:
            raise ValueError(f'input changed from validated snapshot: {path}')


def _stable_read(path, reader, **kwargs):
    """Bracket path-based readers; detects observed changes, not adversarial ABA."""
    path = Path(path)
    digest = sha256(path)
    value = reader(path, **kwargs)
    _assert_hashes({path: digest})
    return value, digest


def _stable_bytes(path):
    raw, digest = _stable_read(path, Path.read_bytes)
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f'bytes changed during snapshot read: {path}')
    return raw


def read_json(path):
    return json.loads(_stable_bytes(path).decode('utf-8-sig'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True), encoding='utf-8')


def check_new_output(output: Path, protected: Sequence[Path]) -> Path:
    output = output.expanduser().resolve()
    for path in protected:
        root = path.expanduser().resolve()
        if output == root or output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError(f'output overlaps protected input/output: {output} and {root}')
    if output.exists():
        raise ValueError(f'output must be a new directory, already exists: {output}')
    return output


def source_worktree():
    return Path(__file__).resolve().parents[1]


def best_selection(metadata):
    epoch = metadata.get('epoch')
    best_epoch = metadata.get('best_selection_epoch')
    dice = metadata.get('best_selection_dice')
    if (type(epoch) is not int or type(best_epoch) is not int
        or not epoch == best_epoch > 0
        or type(dice) not in (int, float) or not 0 <= dice <= 1 or not math.isfinite(dice)):
        raise ValueError('best selection requires epoch == best_selection_epoch > 0 and finite Dice in [0,1]')
    return {'epoch': epoch, 'best_selection_epoch': best_epoch, 'best_selection_dice': dice,
            'basis': 'training saves best only on strict prepared-space selection Dice improvement'}


def official_managers(plans, dataset):
    if importlib.metadata.version('nnunetv2') != '2.8.1':
        raise ValueError('source-space export requires inspected nnunetv2==2.8.1; no automatic dependency changes')
    if plans.get('image_reader_writer') != 'SimpleITKIO':
        raise ValueError('only verified SimpleITKIO 3D NIfTI geometry is supported')
    forward, backward = plans.get('transpose_forward'), plans.get('transpose_backward')
    if (not isinstance(forward, list) or not isinstance(backward, list)
        or any(type(x) is not int for x in forward + backward)
        or sorted(forward) != [0, 1, 2] or sorted(backward) != [0, 1, 2]
        or [forward[i] for i in backward] != [0, 1, 2]):
        raise ValueError('plans transpose must be explicit inverse permutations')
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    pm = PlansManager(plans)
    cm = pm.get_configuration('2d')
    if len(cm.spacing) != 2 or not np.isfinite(cm.spacing).all() or min(cm.spacing) <= 0:
        raise ValueError('invalid 2d plans spacing')
    return pm, cm, pm.get_label_manager(dataset)


class _PropertiesUnpickler(pickle.Unpickler):
    """Allow official NumPy properties, reject executable arbitrary globals."""
    def find_class(self, module, name):
        if module == 'numpy' and name in ('ndarray', 'dtype'):
            return getattr(np, name)
        if module in ('numpy.core.multiarray', 'numpy._core.multiarray') and name in ('_reconstruct', 'scalar'):
            return getattr(np._core.multiarray, name)
        if module in ('numpy.core.numeric', 'numpy._core.numeric') and name == '_frombuffer':
            return np._core.numeric._frombuffer
        raise ValueError(f'unsupported properties pickle global: {module}.{name}')


def read_properties(path):
    value = _PropertiesUnpickler(io.BytesIO(_stable_bytes(path))).load()
    if not isinstance(value, dict):
        raise ValueError('case properties must be a dictionary')
    return value


def _shape(value, name):
    if (not isinstance(value, (tuple, list, np.ndarray)) or len(value) != 3
        or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) or x <= 0 for x in value)):
        raise ValueError(f'{name} must contain three positive integers')
    return tuple(int(x) for x in value)


def validate_properties(prop, prepared_shape, pm, cm):
    required = ('spacing', 'shape_before_cropping', 'shape_after_cropping_and_before_resampling',
                'bbox_used_for_cropping', 'sitk_stuff')
    if not isinstance(prop, dict) or any(key not in prop for key in required):
        raise ValueError('required restoration properties missing; never infer crop/axis/spacing')
    before = _shape(prop['shape_before_cropping'], 'shape_before_cropping')
    crop = _shape(prop['shape_after_cropping_and_before_resampling'], 'crop shape')
    spacing = np.asarray(prop['spacing'], dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError('properties spacing must contain three positive finite values')
    bbox = prop['bbox_used_for_cropping']
    if not isinstance(bbox, (tuple, list, np.ndarray)) or len(bbox) != 3:
        raise ValueError('invalid crop bbox')
    for axis, interval in enumerate(bbox):
        if (not isinstance(interval, (list, tuple, np.ndarray)) or len(interval) != 2
            or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) for x in interval)
            or not 0 <= interval[0] < interval[1] <= before[axis]
            or interval[1] - interval[0] != crop[axis]):
            raise ValueError('crop bbox conflicts with original/cropped shape')
    geometry = prop['sitk_stuff']
    if not isinstance(geometry, dict):
        raise ValueError('missing SimpleITK properties geometry')
    for key, length in (('spacing', 3), ('origin', 3), ('direction', 9)):
        value = np.asarray(geometry.get(key), dtype=float)
        if value.shape != (length,) or not np.isfinite(value).all():
            raise ValueError(f'invalid properties geometry {key}')
    if not np.allclose(spacing, list(reversed(geometry['spacing'])), rtol=0., atol=1e-6):
        raise ValueError('properties array spacing conflicts with SimpleITK spacing')
    direction = np.asarray(geometry['direction']).reshape(3, 3)
    if not np.allclose(direction.T @ direction, np.eye(3), rtol=0., atol=1e-6):
        raise ValueError('invalid direction cosines')
    from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
    transposed_spacing = spacing[pm.transpose_forward]
    target_spacing = [transposed_spacing[0], *cm.spacing]
    expected = tuple(int(x) for x in compute_new_shape(crop, transposed_spacing, target_spacing))
    if tuple(prepared_shape) != (1, *expected) or min(expected) <= 0:
        raise ValueError(f'prepared shape {prepared_shape} incompatible with properties/plans: {(1, *expected)}')
    return tuple(before[i] for i in pm.transpose_backward)


def verify_reference(prop, prepared_shape, pm, cm, reference):
    original_shape = validate_properties(prop, prepared_shape, pm, cm)
    if reference.array.ndim != 3 or reference.array.shape != original_shape:
        raise ValueError('raw image shape conflicts with properties inverse transpose')
    for key, actual in (('spacing', reference.spacing_xyz), ('origin', reference.origin_xyz), ('direction', reference.direction)):
        if not np.allclose(actual, prop['sitk_stuff'][key], rtol=0., atol=1e-6):
            raise ValueError(f'raw image {key} conflicts with properties')


def restore_logits(logits, prop, pm, cm, lm):
    from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
    logits = np.asarray(logits)
    validate_properties(prop, (1, *logits.shape[1:]), pm, cm)
    if logits.ndim != 4 or logits.shape[0] != 2 or not np.isfinite(logits).all():
        raise ValueError('export needs finite two-class prepared logits')
    old_threads = torch.get_num_threads()
    try:
        # Official inverse resampling on logits, not a resized hard mask.
        return convert_predicted_logits_to_segmentation_with_correct_shape(
            logits, pm, cm, lm, prop, return_probabilities=True, num_threads_torch=1)
    finally:
        torch.set_num_threads(old_threads)


def predict_prepared_logits(model, store, device, *, slice_batch_size, mirror_axes, patch_size, tile_step_size):
    shape = tuple(store.shape)
    if slice_batch_size <= 0 or len(shape) != 4 or shape[0] != 1 or min(shape) <= 0:
        raise ValueError('prepared input must be non-empty [1,Z,H,W] and batch positive')
    result = np.empty((2, *shape[1:]), dtype=np.float32)
    for start in range(0, shape[1], slice_batch_size):
        end = min(start + slice_batch_size, shape[1])
        image = np.asarray(store[0, start:end, :, :], dtype=np.float32)
        if not np.isfinite(image).all():
            raise ValueError('prepared image contains non-finite values')
        # Exactly the training b2nd channel/z slices. No normalization or GT load.
        logits = predict_logits_2d(model, torch.from_numpy(image).unsqueeze(1), device,
            mirror_axes=mirror_axes, patch_size=patch_size, tile_step_size=tile_step_size)
        if tuple(logits.shape) != (end - start, 2, *shape[2:]) or not torch.isfinite(logits).all():
            raise ValueError('invalid prepared inference logits')
        result[:, start:end] = logits.detach().cpu().numpy().transpose(1, 0, 2, 3)
    return result


def _contract(args):
    if args.checkpoint.name != 'checkpoint_best.pth':
        raise ValueError('evaluation requires finetune checkpoint_best.pth')
    payload, checkpoint_hash = read_formal_payload(args.checkpoint)
    md = payload['metadata']
    best_selection(md)
    if resolve_checkpoint_model_identity(md) != ('h2former', 'single_output'):
        raise ValueError('requires base h2former/single_output target checkpoint')
    cfg = md.get('resolved_config')
    if not isinstance(cfg, Mapping) or cfg.get('model') != get_model_contract('h2former').as_dict():
        raise ValueError('target resolved model contract missing/mismatched')
    if type(md.get('fold')) is not int or md['fold'] != args.fold or cfg.get('fold') != args.fold:
        raise ValueError('checkpoint/config fold mismatch')
    run_state, evidence = validate_checkpoint_alignment_metadata(md)
    cfg_state, cfg_evidence = validate_checkpoint_alignment_metadata(cfg)
    if (run_state, evidence) != (cfg_state, cfg_evidence):
        raise ValueError('checkpoint/config alignment contract conflict')
    if run_state == DEFAULT_RUN_STATE and not args.allow_pending:
        raise ValueError('pending checkpoint requires explicit --allow-pending engineering evaluation')
    source = cfg.get('source_identity')
    validate_source_schema(source)
    if source['stage'] != 'finetune' or source['dataset_id'] != 'Dataset501' or source['type'] != 'nnunet_preprocessed_b2nd':
        raise ValueError('requires Dataset501 finetune prepared target source, not external pretrain')
    if cfg.get('data_source', {}).get('type') != source['type'] or cfg['data_source'].get('root') != source['root']:
        raise ValueError('checkpoint data_source/source_identity conflict')
    metadata_hashes = {args.checkpoint.resolve(): checkpoint_hash}
    sidecar = args.checkpoint.parent / 'resolved_config.json'
    if sidecar.is_file():
        sidecar_config, digest = _stable_read(sidecar, read_json)
        metadata_hashes[sidecar.resolve()] = digest
        # JSON sidecars encode checkpoint tuples as arrays, including nested sequences.
        if json.loads(json.dumps(sidecar_config, allow_nan=False)) != json.loads(json.dumps(cfg, allow_nan=False)):
            raise ValueError('training resolved_config.json conflicts with checkpoint config')
    provenance = cfg.get('initialization_provenance')
    validate_initialization_provenance(provenance)
    if provenance is None:
        raise ValueError('external-pretrain initialization provenance is required')
    folds, split_hash = _stable_read(args.splits_file, read_splits)
    metadata_hashes[args.splits_file.resolve()] = split_hash
    if canonical_splits(folds) != canonical_splits(read_fixed_splits()):
        raise ValueError('server split differs from supplied fixed Dataset501 five folds')
    if not 0 <= args.fold < 5:
        raise ValueError('fold must be 0..4')
    fold = folds[args.fold]
    all_val = [case for item in folds for case in item['val']]
    if len(all_val) != len(set(all_val)):
        raise ValueError('five-fold validation sets must be disjoint for OOF')
    if set(all_val) != set(fold['train'] + fold['val']):
        raise ValueError('five-fold validation union must equal Dataset501 cohort')
    plan, plan_hash = _stable_read(args.plans, read_json)
    metadata_hashes[args.plans.resolve()] = plan_hash
    dataset_path = next((path for path in (args.preprocessed_root / 'dataset.json', args.preprocessed_root.parent / 'dataset.json') if path.is_file()), None)
    if dataset_path is None:
        raise ValueError('prepared dataset.json is required')
    dataset, dataset_hash = _stable_read(dataset_path, read_json)
    metadata_hashes[dataset_path.resolve()] = dataset_hash
    if (len(dataset.get('channel_names', {})) != 1 or dataset.get('labels') != {'background': 0, 'lesion': 1}
        or dataset.get('file_ending') != '.nii.gz'):
        raise ValueError('requires single-channel background/lesion NIfTI dataset')
    patient_map = dataset.get('patient_map')
    if source['patient_map_sha256'] is not None:
        if patient_map is None:
            raise ValueError('checkpoint patient map required in dataset.json; no inferred mapping')
        mapped_folds, mapped_hash = _stable_read(args.splits_file, read_splits, patient_map=patient_map)
        if mapped_hash != split_hash or mapped_folds != folds:
            raise ValueError('split changed during patient-map validation')
    checks = {'split_sha256': compute_plan_hash(canonical_splits(folds)),
              'cases_sha256': compute_plan_hash({key: sorted(fold[key]) for key in ('train', 'val')}),
              'plans_sha256': compute_plan_hash(plan), 'dataset_metadata_sha256': compute_plan_hash(dataset),
              'patient_map_sha256': None if patient_map is None else compute_plan_hash(patient_map)}
    for key, expected in checks.items():
        if source[key] != expected:
            raise ValueError(f'checkpoint target source {key} mismatch')
    if (plan.get('dataset_name', '').split('_')[0] != 'Dataset501'
        or plan.get('configurations', {}).get('2d', {}).get('patch_size') != [512, 512]
        or len(plan['configurations']['2d'].get('use_mask_for_norm', [])) != 1):
        raise ValueError('incompatible Dataset501 2d plans')
    pm, cm, lm = official_managers(plan, dataset)
    if lm.has_regions or lm.has_ignore_label or lm.num_segmentation_heads != 2:
        raise ValueError('requires ordinary two-class label manager')
    if args.preprocessed_root.name != cm.data_identifier:
        raise ValueError('prepared directory does not match plans data_identifier')
    cases = tuple(fold['val'])
    for case in cases:
        for path in (args.preprocessed_root / f'{case}.b2nd', args.preprocessed_root / f'{case}_seg.b2nd',
                     args.preprocessed_root / f'{case}.pkl', args.raw_root / 'imagesTr' / f'{case}_0000.nii.gz'):
            if not path.is_file():
                raise FileNotFoundError(f'incomplete fold file pairing: {path}')
    protected = [source_worktree(), args.preprocessed_root, args.raw_root, args.checkpoint.parent, args.plans.parent]
    external_root = Path(provenance['source_identity']['root'])
    protected.extend([external_root, Path(provenance['path']).parent])
    output = check_new_output(args.output_root, protected)
    if args.slice_batch_size < 1 or not 0 < args.tile_step_size <= 1:
        raise ValueError('invalid inference batch/tile step')
    _assert_hashes(metadata_hashes)
    return payload, checkpoint_hash, cfg, cases, pm, cm, lm, output, dataset_path, metadata_hashes


def git_snapshot():
    root = Path(__file__).resolve().parents[1]
    def git(*parts):
        result = subprocess.run(['git', '-C', str(root), *parts], capture_output=True, check=True)
        return result.stdout
    return {'worktree': str(root), 'head': git('rev-parse', 'HEAD').decode().strip(),
            'branch': git('branch', '--show-current').decode().strip(),
            'status': git('status', '--porcelain').decode('utf-8', errors='replace'),
            'tracked_diff_sha256': hashlib.sha256(git('diff', '--binary', 'HEAD')).hexdigest(),
            'implementation_sha256': {str(path.relative_to(root)): sha256(path) for path in
                (Path(__file__).resolve(), root / 'standalone_nnunet2d/predict.py', root / 'standalone_nnunet2d/engine/predictor.py')}}


def _prediction_manifest(root):
    path = root.resolve() / 'prediction_manifest.json'
    manifest, manifest_hash = _stable_read(path, read_json)
    if manifest.get('schema_version') != 1 or manifest.get('entry') != 'h2former_prepared_source_space' or manifest.get('status') != 'complete':
        raise ValueError('only complete prepared source-space manifest can be evaluated')
    split_path = Path(manifest['split']['path'])
    folds, split_hash = _stable_read(split_path, read_splits)
    if (split_hash != manifest['split']['sha256']
        or canonical_splits(folds) != canonical_splits(read_fixed_splits())
        or type(manifest.get('fold')) is not int or not 0 <= manifest['fold'] < 5):
        raise ValueError('manifest split/fold is no longer the fixed Dataset501 split')
    cases = manifest.get('expected_cases', [])
    if cases != folds[manifest['fold']]['val']:
        raise ValueError('manifest does not contain the exact fixed fold validation cohort')
    records = manifest.get('cases', [])
    if not cases or len(cases) != len(set(cases)) or [r['case_id'] for r in records] != cases:
        raise ValueError('incomplete/duplicate case manifest')
    # Reuse case-id syntax validation, then check exact files including duplicates.
    for case in cases:
        if not isinstance(case, str) or Path(case).name != case or any(c in case for c in '/\\:') or case in ('.', '..'):
            raise ValueError('unsafe manifest case ID')
    from evaluate_segmentation_metrics import collect_nifti
    predictions = collect_nifti(root / 'predictions')
    if set(predictions) != set(cases):
        raise ValueError('prediction files do not match complete expected validation cohort')
    _assert_hashes({path: manifest_hash, split_path: split_hash})
    return path, manifest, predictions, manifest_hash


def _same_geometry(pred, ref):
    if pred.array.shape != ref.array.shape:
        raise ValueError('source/GT and prediction shape mismatch')
    for key in ('spacing_xyz', 'origin_xyz', 'direction'):
        if not np.allclose(getattr(pred, key), getattr(ref, key), rtol=0., atol=1e-6):
            raise ValueError(f'source/GT and prediction {key} mismatch')


def _checked_case(record, prediction_path, source_path, gt_path=None):
    case = record['case_id']
    prediction, prediction_hash = _stable_read(prediction_path, read_nifti)
    source, source_hash = _stable_read(source_path, read_nifti)
    if prediction_hash != record['prediction_sha256']:
        raise ValueError(f'prediction changed after export: {case}')
    if source_hash != record['source_sha256']:
        raise ValueError(f'raw source changed after export: {case}')
    if prediction.array.dtype != np.uint8 or not np.isin(prediction.array, (0, 1)).all():
        raise ValueError(f'non-binary or non-uint8 prediction: {case}')
    _same_geometry(prediction, source)
    hashes = {prediction_path: prediction_hash, source_path: source_hash}
    result = {'case_id': case, 'passed': True, 'prediction_sha256': prediction_hash,
              'source_sha256': source_hash}
    if gt_path is not None:
        gt, gt_hash = _stable_read(gt_path, read_nifti)
        _same_geometry(prediction, gt)
        if not np.isin(gt.array, (0, 1)).all():
            raise ValueError(f'GT is not binary 0/1: {case}')
        result['gt_sha256'] = gt_hash
        hashes[gt_path] = gt_hash
    _assert_hashes(hashes)
    return result, hashes


def check_predictions(root: Path, raw_root: Path, gt_root: Path | None = None):
    path, manifest, predictions, manifest_hash = _prediction_manifest(root)
    records = []
    hashes = {path: manifest_hash, Path(manifest['split']['path']): manifest['split']['sha256']}
    for record in manifest['cases']:
        case = record['case_id']
        result, case_hashes = _checked_case(record, predictions[case],
            raw_root / 'imagesTr' / f'{case}_0000.nii.gz',
            None if gt_root is None else gt_root / f'{case}.nii.gz')
        records.append(result)
        hashes.update(case_hashes)
    _assert_hashes(hashes)
    _, _, current_predictions, current_hash = _prediction_manifest(root)
    if current_hash != manifest_hash or current_predictions != predictions:
        raise ValueError('manifest or complete prediction file set changed during check')
    return {'passed': True, 'run_state': manifest['run_state'], 'case_count': len(records),
            'cases': records, 'prediction_manifest_sha256': manifest_hash}


def _copy_metric_snapshot(source, destination, expected_hash):
    _assert_hashes({source: expected_hash})
    shutil.copyfile(source, destination)
    _assert_hashes({source: expected_hash, destination: expected_hash})


def evaluate_predictions(root, raw_root, gt_root, output, f2_mode):
    if f2_mode not in ('paper', 'standard'):
        raise ValueError('unknown F2 mode')
    _, manifest, predictions, manifest_hash = _prediction_manifest(root)
    protected = [source_worktree(), root, raw_root, gt_root,
                 *[Path(p) for p in manifest['protected_roots']]]
    output = check_new_output(output, protected)
    geometry = check_predictions(root, raw_root, gt_root)
    if geometry['prediction_manifest_sha256'] != manifest_hash:
        raise ValueError('manifest changed between validation stages')
    from evaluate_segmentation_metrics import evaluate_case, build_summary
    import pandas as pd
    # One pair at a time, outside all protected inputs and outputs. Only our own
    # temporary copies are removed by TemporaryDirectory; never real sources.
    temp_parent = Path(tempfile.gettempdir()).resolve()
    check_new_output(temp_parent / 'h2-source-metrics-boundary-check', [*protected, output])
    rows = []
    for record, checked in zip(manifest['cases'], geometry['cases']):
        case = record['case_id']
        prediction_path = predictions[case]
        gt_path = gt_root / f'{case}.nii.gz'
        source_path = raw_root / 'imagesTr' / f'{case}_0000.nii.gz'
        with tempfile.TemporaryDirectory(prefix='h2-source-metrics-', dir=temp_parent) as temporary:
            temporary = Path(temporary)
            (temporary / 'prediction').mkdir(); (temporary / 'gt').mkdir()
            prediction_copy = temporary / 'prediction' / f'{case}.nii.gz'
            gt_copy = temporary / 'gt' / f'{case}.nii.gz'
            _copy_metric_snapshot(prediction_path, prediction_copy, checked['prediction_sha256'])
            _copy_metric_snapshot(gt_path, gt_copy, checked['gt_sha256'])
            snapshot_geometry, _ = _checked_case(record, prediction_copy, source_path, gt_copy)
            if snapshot_geometry != checked:
                raise ValueError(f'metric snapshot differs from checked contents: {case}')
            rows.append(evaluate_case(prediction_copy, gt_copy, f2_mode))
            _assert_hashes({prediction_copy: checked['prediction_sha256'], gt_copy: checked['gt_sha256'],
                           prediction_path: checked['prediction_sha256'], gt_path: checked['gt_sha256'],
                           source_path: checked['source_sha256']})
    frame = pd.DataFrame(rows)
    report = build_summary(frame, f2_mode)
    report.update(run_state=manifest['run_state'], experiment_class=manifest['experiment_class'],
                  output_space='original_patient_full_volume', fold=manifest['fold'],
                  prediction_manifest_sha256=manifest_hash, geometry_validation=geometry,
                  expected_cases=manifest['expected_cases'], selection_metric='unchanged prepared-space selection Dice; not this evaluation',
                  metric_input_contract='per-case byte-verified temporary snapshots; originals and complete cohort rechecked before report',
                  metric_implementation_sha256=sha256(source_worktree() / 'evaluate_segmentation_metrics.py'))
    # Includes manifest bytes, split, exact file cohort and all original images,
    # predictions and GT. A changed GT cannot silently acquire a newer report hash.
    if check_predictions(root, raw_root, gt_root) != geometry:
        raise ValueError('validated evaluation inputs changed before report')
    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output / 'case_metrics.csv', index=False)
    pd.DataFrame([{key: value['mean'] for key, value in report['metrics'].items()}]).to_csv(output / 'summary_metrics.csv', index=False)
    write_json(output / 'summary_metrics.json', report)
    return report


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--preprocessed-root', type=Path, required=True)
    parser.add_argument('--plans', type=Path, required=True)
    parser.add_argument('--splits-file', type=Path, required=True)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--metadata-only', action='store_true')
    parser.add_argument('--allow-pending', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--slice-batch-size', type=int, default=1)
    parser.add_argument('--tile-step-size', type=float, default=.5)
    parser.add_argument('--disable-tta', action='store_true')
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    args.preprocessed_root = args.preprocessed_root.expanduser().resolve()
    args.raw_root = args.raw_root.expanduser().resolve()
    payload, checkpoint_hash, cfg, cases, pm, cm, lm, output, dataset_path, input_hashes_by_path = _contract(args)
    if args.metadata_only:
        print(json.dumps({'status': 'metadata_checked', 'case_count': len(cases), 'fold': args.fold,
            'run_state': cfg['run_state'], 'checkpoint_sha256': checkpoint_hash,
            'properties_contents_checked': False, 'real_geometry_checked': False, 'gpu_validated': False,
            'model_loaded': False, 'official_alignment_proven': False}, indent=2))
        return 0
    device = torch.device(args.device)
    model = build_model('h2former', supervision_mode='single_output', inference=True)
    validate_model_tensors(model, payload['model_state_dict'])
    model.load_state_dict(payload['model_state_dict'], strict=True)
    model.to(device)
    selection = best_selection(payload['metadata'])
    del payload
    mirror_axes = () if args.disable_tta else (0, 1)
    manifest = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'schema_version': 1, 'entry': 'h2former_prepared_source_space', 'status': 'running',
        'run_state': cfg['run_state'], 'alignment_evidence': cfg.get('alignment_evidence'),
        'experiment_class': 'pending_engineering_fold_evaluation' if cfg['run_state'] == DEFAULT_RUN_STATE else 'fold_validation',
        'protected_roots': [str(p.resolve()) for p in
            (source_worktree(), args.preprocessed_root, args.raw_root, args.checkpoint.parent, args.plans.parent,
             Path(cfg['initialization_provenance']['source_identity']['root']),
             Path(cfg['initialization_provenance']['path']).parent)],
        'fold': args.fold, 'expected_cases': list(cases), 'cases': [], 'git': git_snapshot(),
        'checkpoint': {'path': str(args.checkpoint.resolve()), 'sha256': checkpoint_hash, 'model': cfg['model'],
                       'source_identity': cfg['source_identity'], 'initialization_provenance': cfg['initialization_provenance'],
                       'selection': selection},
        'plans': {'path': str(args.plans.resolve()), 'sha256': input_hashes_by_path[args.plans.resolve()], 'canonical_sha256': compute_plan_hash(pm.plans)},
        'split': {'path': str(args.splits_file.resolve()), 'sha256': input_hashes_by_path[args.splits_file.resolve()],
                  'canonical_sha256': cfg['source_identity']['split_sha256']},
        'resolved_config_sha256': compute_plan_hash(cfg),
        'environment': {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'SimpleITK', 'blosc2')},
        'dataset_metadata_sha256': input_hashes_by_path[dataset_path.resolve()],
        'dataset_metadata_path': str(dataset_path.resolve()), 'nnunetv2_version': importlib.metadata.version('nnunetv2'),
        'policy': {'input': 'training_prepared_b2nd_no_repeat_normalization', 'configuration': '2d',
                   'mirror_axes': list(mirror_axes), 'mirror_aggregation': 'unflip_logits_then_mean',
                   'patch_size': list(cm.patch_size), 'tile_step_size': args.tile_step_size,
                   'tile_aggregation': 'Gaussian_weighted_logits_fp16_existing_predictor',
                   'slice_batch_size': args.slice_batch_size, 'device': str(device),
                   'export': 'nnunetv2_2.8.1_logits_resample_crop_inverse_transpose',
                   'output_space': 'original_patient_full_volume', 'probabilities_saved': True,
                   'gt_used_for_inference': False, 'selection_semantics_changed': False}}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'predictions').mkdir(); (output / 'probabilities').mkdir()
    manifest_path = output / 'prediction_manifest.json'
    write_json(manifest_path, manifest)
    case = None
    try:
        import blosc2
        for case in cases:
            prepared_path = args.preprocessed_root / f'{case}.b2nd'
            prop_path = args.preprocessed_root / f'{case}.pkl'
            source_path = args.raw_root / 'imagesTr' / f'{case}_0000.nii.gz'
            prop, prop_hash = _stable_read(prop_path, read_properties)
            prepared_hash = sha256(prepared_path)
            store = blosc2.open(urlpath=str(prepared_path), mode='r', dparams={'nthreads': 1})
            seg_store = blosc2.open(urlpath=str(args.preprocessed_root / f'{case}_seg.b2nd'),
                                   mode='r', dparams={'nthreads': 1})
            _validate_b2nd_store_pair(case, store, seg_store)
            del seg_store  # Header pairing only; no GT voxels read or used.
            source, source_hash = _stable_read(source_path, read_nifti)
            verify_reference(prop, store.shape, pm, cm, source)
            input_hashes = {'prepared_sha256': prepared_hash, 'properties_sha256': prop_hash,
                            'source_sha256': source_hash}
            input_hashes_by_path.update({prepared_path: prepared_hash, prop_path: prop_hash, source_path: source_hash})
            logits = predict_prepared_logits(model, store, device, slice_batch_size=args.slice_batch_size,
                        mirror_axes=mirror_axes, patch_size=tuple(cm.patch_size), tile_step_size=args.tile_step_size)
            segmentation, probabilities = restore_logits(logits, prop, pm, cm, lm)
            if (probabilities.shape != (2, *source.array.shape)
                or not np.isfinite(probabilities).all()
                or not np.allclose(probabilities.sum(axis=0), 1., rtol=0., atol=1e-5)):
                raise ValueError('invalid restored probabilities')
            prediction_path = output / 'predictions' / f'{case}.nii.gz'
            geometry = save_and_validate_prediction(prediction_path, segmentation, source)
            probability_path = output / 'probabilities' / f'{case}.npz'
            np.savez_compressed(probability_path, probabilities=probabilities)
            for key, path in (('prepared_sha256', prepared_path), ('properties_sha256', prop_path), ('source_sha256', source_path)):
                if sha256(path) != input_hashes[key]:
                    raise ValueError(f'input changed during prediction: {path}')
            manifest['cases'].append({'case_id': case, 'prepared_shape': list(store.shape),
                'source_shape': list(source.array.shape), **input_hashes, 'prediction_sha256': sha256(prediction_path),
                'probabilities_sha256': sha256(probability_path), 'geometry': geometry})
            write_json(manifest_path, manifest)
            del logits, segmentation, probabilities, store, source
        if len(manifest['cases']) != len(cases):
            raise RuntimeError('not every validation case was exported')
        _assert_hashes(input_hashes_by_path)
        manifest['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
        manifest['status'] = 'complete'
        write_json(manifest_path, manifest)
    except BaseException as exc:
        manifest['status'] = 'failed'
        manifest['failure'] = {'case_id': case, 'error': f'{type(exc).__name__}: {exc}'}
        write_json(manifest_path, manifest)
        raise
    print(f'Exported complete fold {args.fold}: {len(cases)} cases; {cfg["run_state"]}')
    return 0


def cli(argv=None):
    import sys
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in ('check', 'metrics'):
        mode = arguments.pop(0)
        parser = argparse.ArgumentParser(description='Verify complete exported fold before metrics')
        parser.add_argument('--prediction-root', type=Path, required=True)
        parser.add_argument('--raw-root', type=Path, required=True)
        parser.add_argument('--gt-dir', type=Path, required=mode == 'metrics')
        if mode == 'metrics':
            parser.add_argument('--output-root', type=Path, required=True)
            parser.add_argument('--f2-mode', choices=('paper', 'standard'), default='paper')
        args = parser.parse_args(arguments)
        if mode == 'check':
            result = check_predictions(args.prediction_root, args.raw_root, args.gt_dir)
        else:
            result = evaluate_predictions(args.prediction_root, args.raw_root, args.gt_dir, args.output_root, args.f2_mode)
        print(json.dumps(result, indent=2))
        return 0
    return main(arguments)


if __name__ == '__main__':
    raise SystemExit(cli())
