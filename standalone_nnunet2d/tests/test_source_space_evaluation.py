"""Tiny synthetic engineering tests; never real Dataset501 data or weights."""
from copy import deepcopy
import json
import itertools
import pickle
from pathlib import Path

import blosc2
import numpy as np
import pytest
import torch

from standalone_nnunet2d import source_space_evaluation as ev
from standalone_nnunet2d.data.nifti_io import NiftiVolume, read_nifti, write_nifti
from standalone_nnunet2d.models.factory import get_model_contract
from standalone_nnunet2d.training.formal_checkpoint import compute_plan_hash


def plans_dict():
    return {'dataset_name': 'Dataset501_StrokeLesion', 'plans_name': 'nnUNetPlans',
            'image_reader_writer': 'SimpleITKIO', 'transpose_forward': [2, 0, 1],
            'transpose_backward': [1, 2, 0], 'foreground_intensity_properties_per_channel': {},
            'configurations': {'2d': {'data_identifier': 'nnUNetPlans_2d',
                'architecture': {}, 'patch_size': [512, 512], 'spacing': [1., 1.],
                'normalization_schemes': ['ZScoreNormalization'], 'use_mask_for_norm': [False],
                'resampling_fn_data': 'resample_data_or_seg_to_shape',
                'resampling_fn_data_kwargs': {'is_seg': False, 'order': 3, 'order_z': 0, 'force_separate_z': False},
                'resampling_fn_probabilities': 'resample_data_or_seg_to_shape',
                'resampling_fn_probabilities_kwargs': {'is_seg': False, 'order': 1, 'order_z': 0, 'force_separate_z': False}}}}


def properties():
    return {'spacing': [2., 1., 3.], 'shape_before_cropping': [7, 5, 6],
            'shape_after_cropping_and_before_resampling': [4, 3, 4],
            'bbox_used_for_cropping': [[1, 5], [1, 4], [1, 5]],
            'sitk_stuff': {'spacing': [3., 1., 2.], 'origin': [10., -2., 8.],
                           'direction': [0., -1., 0., 1., 0., 0., 0., 0., 1.]}}


def fixture_run(tmp_path, monkeypatch):
    # Substitute only fixed reference membership, never a production guard.
    folds = [{'train': [f'val{j}' for j in range(5) if j != i], 'val': [f'val{i}']} for i in range(5)]
    split = tmp_path / 'splits.json'; split.write_text(json.dumps(folds))
    monkeypatch.setattr(ev, 'read_fixed_splits', lambda: deepcopy(folds))
    pre = tmp_path / 'nnUNetPlans_2d'; pre.mkdir()
    plan = pre / 'plans.json'; plan.write_text(json.dumps(plans_dict()))
    dataset = {'channel_names': {'0': 'DWI'}, 'labels': {'background': 0, 'lesion': 1}, 'file_ending': '.nii.gz'}
    (pre / 'dataset.json').write_text(json.dumps(dataset))
    source = {'stage': 'finetune', 'dataset_id': 'Dataset501', 'source_version': 'synthetic',
              'split_sha256': compute_plan_hash(ev.canonical_splits(folds)),
              'cases_sha256': compute_plan_hash({'train': ['val1', 'val2', 'val3', 'val4'], 'val': ['val0']}),
              'plans_sha256': compute_plan_hash(plans_dict()), 'dataset_metadata_sha256': compute_plan_hash(dataset),
              'patient_map_sha256': None, 'type': 'nnunet_preprocessed_b2nd', 'root': str(pre),
              'channels': 1, 'labels': dataset['labels'], 'patch_size': [512, 512]}
    external = deepcopy(source); external.update(stage='pretrain', dataset_id='Dataset508')
    contract = get_model_contract('h2former').as_dict()
    config = {'fold': 0, 'model': contract, 'source_identity': source,
              'data_source': {'type': source['type'], 'root': str(pre)},
              'run_type': 'official_alignment_pending', 'run_state': 'official_alignment_pending',
              'alignment_evidence': None,
              'initialization_provenance': {'path': 'external.pth', 'sha256': 'a'*64, 'model': contract, 'source_identity': external}}
    metadata = {'epoch': 10, 'best_selection_epoch': 10, 'best_selection_dice': .8, 'fold': 0, 'model_name': 'h2former', 'supervision_mode': 'single_output',
                'resolved_config': config, 'config': config, 'run_type': config['run_type'], 'run_state': config['run_state']}
    checkpoint = tmp_path / 'training' / 'checkpoint_best.pth'; checkpoint.parent.mkdir()
    tiny_model = torch.nn.Conv2d(1, 2, 1)
    with torch.no_grad():
        tiny_model.weight.zero_(); tiny_model.bias.copy_(torch.tensor([0., 20.]))
    torch.save({'format_version': 1, 'model_state_dict': tiny_model.state_dict(),
                'optimizer_state_dict': None, 'metadata': metadata}, checkpoint)
    for name in ('val0.b2nd', 'val0_seg.b2nd', 'val0.pkl'):
        (pre / name).touch()
    raw = tmp_path / 'raw'; (raw / 'imagesTr').mkdir(parents=True)
    ref = NiftiVolume(np.zeros((5, 6, 7), np.float32), (3., 1., 2.), (10., -2., 8.), tuple(properties()['sitk_stuff']['direction']))
    write_nifti(raw / 'imagesTr' / 'val0_0000.nii.gz', ref)
    args = ['--checkpoint', str(checkpoint), '--preprocessed-root', str(pre), '--plans', str(plan),
            '--splits-file', str(split), '--fold', '0', '--raw-root', str(raw),
            '--output-root', str(tmp_path / 'out'), '--allow-pending']
    return args, metadata, pre, ref


def test_official_inverse_transpose_crop_resampling_and_background(tmp_path):
    pm, cm, lm = ev.official_managers(plans_dict(), {'labels': {'background': 0, 'lesion': 1}})
    prop = properties()
    ev.validate_properties(prop, (1, 4, 6, 4), pm, cm)
    logits = np.zeros((2, 4, 6, 4), np.float32); logits[1] = 20
    segmentation, probabilities = ev.restore_logits(logits, prop, pm, cm, lm)
    expected = np.zeros((5, 6, 7), np.uint8); expected[1:4, 1:5, 1:5] = 1
    np.testing.assert_array_equal(segmentation, expected)
    assert probabilities.shape == (2, 5, 6, 7)
    assert probabilities[0, 0, 0, 0] == 1
    assert probabilities[1, 0, 0, 0] == 0


@pytest.mark.parametrize('field,value', [('spacing', [0, 1, 3]), ('bbox_used_for_cropping', [[-1, 3], [1, 4], [1, 5]]),
    ('shape_before_cropping', [7, 5]), ('shape_after_cropping_and_before_resampling', [3, 3, 4])])
def test_invalid_properties_rejected(field, value):
    pm, cm, _ = ev.official_managers(plans_dict(), {'labels': {'background': 0, 'lesion': 1}})
    prop = properties(); prop[field] = value
    with pytest.raises(ValueError): ev.validate_properties(prop, (1, 4, 6, 4), pm, cm)


def test_missing_properties_and_wrong_prepared_shape_rejected():
    pm, cm, _ = ev.official_managers(plans_dict(), {'labels': {'background': 0, 'lesion': 1}})
    for prop, shape in [({}, (1, 4, 6, 4)), (properties(), (1, 4, 3, 4))]:
        with pytest.raises(ValueError): ev.validate_properties(prop, shape, pm, cm)


def test_logits_keep_prepared_intensities_and_no_gt(monkeypatch):
    model = torch.nn.Conv2d(1, 2, 1)
    image = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4) + 10
    seen = []
    def tiles(model, tensor, device, **kwargs):
        seen.append(tensor.numpy().copy())
        return torch.cat([tensor, tensor * 2], dim=1)
    monkeypatch.setattr(ev, 'predict_logits_2d', tiles)
    result = ev.predict_prepared_logits(model, image, torch.device('cpu'), slice_batch_size=1, mirror_axes=(), patch_size=(4, 4), tile_step_size=.5)
    np.testing.assert_array_equal(np.concatenate(seen)[:, 0], image[0])
    np.testing.assert_array_equal(result[1], image[0] * 2)


@pytest.mark.parametrize('change', ['fold', 'source', 'model', 'plans', 'split', 'provenance'])
def test_checkpoint_identity_rejected_before_model(tmp_path, monkeypatch, change):
    args, md, pre, ref = fixture_run(tmp_path, monkeypatch)
    if change == 'fold': md['fold'] = 1
    elif change == 'source': md['config']['source_identity']['stage'] = 'pretrain'
    elif change == 'model': md['model_name'] = 'plain_conv_unet'
    elif change == 'plans': md['config']['source_identity']['plans_sha256'] = 'f'*64
    elif change == 'split': md['config']['source_identity']['split_sha256'] = 'f'*64
    else: md['config']['initialization_provenance'] = None
    checkpoint = Path(args[1]); torch.save({'format_version': 1, 'model_state_dict': {}, 'optimizer_state_dict': None, 'metadata': md}, checkpoint)
    monkeypatch.setattr(ev, 'build_model', lambda *a, **k: pytest.fail('model must not load'))
    with pytest.raises(ValueError): ev.main(args + ['--metadata-only'])


def test_metadata_only_complete_membership_and_pending(tmp_path, monkeypatch):
    args, md, pre, ref = fixture_run(tmp_path, monkeypatch)
    monkeypatch.setattr(ev, 'build_model', lambda *a, **k: pytest.fail('metadata must not build model'))
    assert ev.main(args + ['--metadata-only']) == 0
    assert not (tmp_path / 'out').exists()
    (pre / 'val0.pkl').unlink()
    with pytest.raises(FileNotFoundError): ev.main(args + ['--metadata-only'])


@pytest.mark.parametrize('name', ['raw', 'prepared', 'training', 'out'])
def test_output_protection(tmp_path, name):
    protected = [tmp_path / 'raw', tmp_path / 'prepared', tmp_path / 'training']
    output = tmp_path / name
    if name == 'out': output.mkdir()
    with pytest.raises(ValueError): ev.check_new_output(output, protected)


def test_end_to_end_synthetic_manifest_geometry_metrics(tmp_path, monkeypatch):
    args, md, pre, ref = fixture_run(tmp_path, monkeypatch)
    data = np.ones((1, 4, 6, 4), np.float32) * 7
    blosc2.asarray(data, urlpath=str(pre / 'val0.b2nd'), mode='w')
    blosc2.asarray(np.zeros((1, 4, 6, 4), np.int16), urlpath=str(pre / 'val0_seg.b2nd'), mode='w')
    (pre / 'val0.pkl').write_bytes(pickle.dumps(properties()))
    monkeypatch.setattr(ev, 'build_model', lambda *a, **k: torch.nn.Conv2d(1, 2, 1))
    # Real prepared logits/TTA/Gaussian inference with tiny Conv2d weights.
    assert ev.main(args) == 0
    manifest = json.loads((tmp_path / 'out/prediction_manifest.json').read_text())
    assert manifest['status'] == 'complete'
    assert manifest['run_state'] == 'official_alignment_pending'
    assert len(manifest['checkpoint']['sha256']) == 64
    assert manifest['cases'][0]['geometry']['passed']
    prediction = read_nifti(tmp_path / 'out/predictions/val0.nii.gz')
    np.testing.assert_allclose(prediction.origin_xyz, ref.origin_xyz)
    gt = tmp_path / 'gt'; gt.mkdir(); write_nifti(gt / 'val0.nii.gz', prediction)
    assert ev.check_predictions(tmp_path / 'out', tmp_path / 'raw', gt)['passed']
    report = ev.evaluate_predictions(tmp_path / 'out', tmp_path / 'raw', gt, tmp_path / 'metrics', 'paper')
    assert report['metrics']['dice']['mean'] == 1
    assert report['run_state'] == 'official_alignment_pending'
    assert len(report['prediction_manifest_sha256']) == 64
    bad = NiftiVolume(prediction.array, prediction.spacing_xyz, (0., 0., 0.), prediction.direction)
    write_nifti(gt / 'val0.nii.gz', bad)
    with pytest.raises(ValueError): ev.check_predictions(tmp_path / 'out', tmp_path / 'raw', gt)
    (tmp_path / 'out/predictions/val0.nii.gz').unlink()
    with pytest.raises(ValueError): ev.check_predictions(tmp_path / 'out', tmp_path / 'raw', gt)


def test_pending_latest_and_server_split_are_rejected(tmp_path, monkeypatch):
    args, md, pre, ref = fixture_run(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='pending'):
        ev.main([a for a in args if a != '--allow-pending'] + ['--metadata-only'])
    checkpoint = Path(args[1]); latest = checkpoint.with_name('checkpoint_latest.pth')
    latest.write_bytes(checkpoint.read_bytes())
    bad = list(args); bad[1] = str(latest)
    with pytest.raises(ValueError, match='checkpoint_best'):
        ev.main(bad + ['--metadata-only'])
    split = Path(args[args.index('--splits-file') + 1]); folds = json.loads(split.read_text())
    folds[0]['val'] = ['changed']; split.write_text(json.dumps(folds))
    with pytest.raises(ValueError, match='fixed Dataset501'):
        ev.main(args + ['--metadata-only'])


def test_properties_failure_marks_run_failed_no_success_or_metrics(tmp_path, monkeypatch):
    args, md, pre, ref = fixture_run(tmp_path, monkeypatch)
    blosc2.asarray(np.ones((1, 4, 6, 4), np.float32), urlpath=str(pre / 'val0.b2nd'), mode='w')
    blosc2.asarray(np.zeros((1, 4, 6, 4), np.int16), urlpath=str(pre / 'val0_seg.b2nd'), mode='w')
    (pre / 'val0.pkl').write_bytes(pickle.dumps({}))
    monkeypatch.setattr(ev, 'build_model', lambda *a, **k: torch.nn.Conv2d(1, 2, 1))
    with pytest.raises(ValueError, match='properties missing'):
        ev.main(args)
    manifest = json.loads((tmp_path / 'out/prediction_manifest.json').read_text())
    assert manifest['status'] == 'failed' and manifest['failure']['case_id'] == 'val0'
    with pytest.raises(ValueError, match='complete'):
        ev.check_predictions(tmp_path / 'out', tmp_path / 'raw')


def test_raw_properties_geometry_mismatch(tmp_path):
    pm, cm, _ = ev.official_managers(plans_dict(), {'labels': {'background': 0, 'lesion': 1}})
    ref = NiftiVolume(np.zeros((5, 6, 7)), (3., 1., 2.), (0., 0., 0.))
    with pytest.raises(ValueError, match='origin'):
        ev.verify_reference(properties(), (1, 4, 6, 4), pm, cm, ref)


def test_predict_dispatch_is_explicit_and_keeps_legacy(monkeypatch):
    from standalone_nnunet2d import predict
    calls = []
    monkeypatch.setattr(ev, 'main', lambda args: calls.append(args) or 0)
    assert predict.main(['--preprocessed-root', 'prepared']) == 0
    assert calls == [['--preprocessed-root', 'prepared']]
    calls.clear()
    with pytest.raises(SystemExit): predict.main([])
    assert not calls


def test_restricted_properties_numpy_and_arbitrary_globals(tmp_path):
    path = tmp_path / 'properties.pkl'
    prop = properties(); prop['class_locations'] = {1: np.ones((2, 4), dtype=np.int64)}
    path.write_bytes(pickle.dumps(prop, protocol=5))
    loaded = ev.read_properties(path)
    np.testing.assert_array_equal(loaded['class_locations'][1], prop['class_locations'][1])
    path.write_bytes(pickle.dumps(Path('must-not-load')))
    with pytest.raises(ValueError, match='unsupported properties'):
        ev.read_properties(path)


def test_inverse_spatial_signal_is_not_transposed_as_slice_axis():
    pm, cm, lm = ev.official_managers(plans_dict(), {'labels': {'background': 0, 'lesion': 1}})
    logits = np.zeros((2, 4, 6, 4), np.float32); logits[0] = 10
    logits[0, :2] = 0; logits[1, :2] = 20
    segmentation, _ = ev.restore_logits(logits, properties(), pm, cm, lm)
    expected = np.zeros((5, 6, 7), np.uint8); expected[1:4, 1:5, 1:3] = 1
    np.testing.assert_array_equal(segmentation, expected)


def test_metrics_cannot_write_into_prepared_input(tmp_path, monkeypatch):
    args, md, pre, ref = fixture_run(tmp_path, monkeypatch)
    data = np.ones((1, 4, 6, 4), np.float32)
    blosc2.asarray(data, urlpath=str(pre / 'val0.b2nd'), mode='w')
    blosc2.asarray(np.zeros(data.shape, np.int16), urlpath=str(pre / 'val0_seg.b2nd'), mode='w')
    (pre / 'val0.pkl').write_bytes(pickle.dumps(properties()))
    monkeypatch.setattr(ev, 'build_model', lambda *a, **k: torch.nn.Conv2d(1, 2, 1))
    assert ev.main(args) == 0
    gt = tmp_path / 'gt'; gt.mkdir()
    write_nifti(gt / 'val0.nii.gz', read_nifti(tmp_path / 'out/predictions/val0.nii.gz'))
    with pytest.raises(ValueError, match='overlaps'):
        ev.evaluate_predictions(tmp_path / 'out', tmp_path / 'raw', gt, pre / 'metrics', 'paper')


def test_pair_shape_mismatch_fails_without_loading_gt_values(tmp_path, monkeypatch):
    args, md, pre, ref = fixture_run(tmp_path, monkeypatch)
    blosc2.asarray(np.ones((1, 4, 6, 4), np.float32), urlpath=str(pre / 'val0.b2nd'), mode='w')
    blosc2.asarray(np.zeros((1, 4, 5, 4), np.int16), urlpath=str(pre / 'val0_seg.b2nd'), mode='w')
    (pre / 'val0.pkl').write_bytes(pickle.dumps(properties()))
    monkeypatch.setattr(ev, 'build_model', lambda *a, **k: torch.nn.Conv2d(1, 2, 1))
    with pytest.raises(ValueError, match='shapes differ'):
        ev.main(args)


def test_sidecar_conflict_and_data_identifier_rejected(tmp_path, monkeypatch):
    args, md, pre, ref = fixture_run(tmp_path, monkeypatch)
    sidecar = Path(args[1]).parent / 'resolved_config.json'
    sidecar.write_text(json.dumps({'fold': 4}))
    with pytest.raises(ValueError, match='resolved_config.json'):
        ev.main(args + ['--metadata-only'])
    sidecar.write_text(json.dumps(md['resolved_config']))
    plan_path = Path(args[args.index('--plans') + 1]); plan = plans_dict()
    plan['configurations']['2d']['data_identifier'] = 'wrong_configuration'
    plan_path.write_text(json.dumps(plan))
    md['resolved_config']['source_identity']['plans_sha256'] = compute_plan_hash(plan)
    sidecar.write_text(json.dumps(md['resolved_config']))
    torch.save({'format_version': 1, 'model_state_dict': {}, 'optimizer_state_dict': None, 'metadata': md}, Path(args[1]))
    with pytest.raises(ValueError, match='data_identifier'):
        ev.main(args + ['--metadata-only'])


def test_inference_independent_of_seg_values_and_gt_class_locations(tmp_path, monkeypatch):
    args, md, pre, ref = fixture_run(tmp_path, monkeypatch)
    shape = (1, 4, 6, 4)
    blosc2.asarray(np.ones(shape, np.float32)*7, urlpath=str(pre / 'val0.b2nd'), mode='w')
    monkeypatch.setattr(ev, 'build_model', lambda *a, **k: torch.nn.Conv2d(1, 2, 1))
    results = []
    for idx in range(2):
        blosc2.asarray(np.full(shape, idx, np.int16), urlpath=str(pre / 'val0_seg.b2nd'), mode='w')
        prop = properties(); prop['class_locations'] = {1: np.array([[0, idx, 0, 0]], np.int64)}
        (pre / 'val0.pkl').write_bytes(pickle.dumps(prop))
        run_args = list(args); run_args[run_args.index('--output-root')+1] = str(tmp_path / f'out{idx}')
        assert ev.main(run_args) == 0
        results.append(read_nifti(tmp_path / f'out{idx}/predictions/val0.nii.gz').array)
    np.testing.assert_array_equal(*results)


def test_legacy_raw_prediction_still_runs_with_tiny_checkpoint(tmp_path, monkeypatch):
    from standalone_nnunet2d import predict
    args, md, pre, ref = fixture_run(tmp_path, monkeypatch)
    monkeypatch.setattr(predict, 'build_model', lambda *a, **k: torch.nn.Conv2d(1, 2, 1))
    raw_output = tmp_path / 'legacy-raw'
    assert predict.main(['--checkpoint', args[1], '--raw-root', str(tmp_path / 'raw'),
                         '--case-id', 'val0', '--output-root', str(raw_output), '--allow-pending']) == 0
    restored = read_nifti(raw_output / 'predictions/val0.nii.gz')
    assert restored.array.shape == ref.array.shape
    np.testing.assert_allclose(restored.direction, ref.direction)
    manifest = json.loads((raw_output / 'prediction_manifest.json').read_text())
    assert manifest['policy']['run_state'] == 'official_alignment_pending'
    assert 'entry' not in manifest  # Legacy schema/behavior stays intact.


def test_existing_f2_modes_preserve_asymmetric_fp_fn_definitions(tmp_path):
    from evaluate_segmentation_metrics import evaluate_case
    pred = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8).reshape(1, 2, 3)
    gt = np.array([1, 0, 0, 1, 1, 1], dtype=np.uint8).reshape(1, 2, 3)
    a = tmp_path / 'pred.nii.gz'; b = tmp_path / 'gt.nii.gz'
    write_nifti(a, NiftiVolume(pred, (1., 2., 3.), (0., 0., 0.)))
    write_nifti(b, NiftiVolume(gt, (1., 2., 3.), (0., 0., 0.)))
    assert evaluate_case(a, b, 'paper')['f2'] == pytest.approx(5 / (5 + 4*2 + 3))
    assert evaluate_case(a, b, 'standard')['f2'] == pytest.approx(5 / (5 + 2 + 4*3))


# B1--B4 production-path regressions; review probes remain read-only.
def prepared_run(tmp_path, monkeypatch):
    args, md, pre, ref = fixture_run(tmp_path, monkeypatch)
    blosc2.asarray(np.ones((1, 4, 6, 4), np.float32), urlpath=str(pre / 'val0.b2nd'), mode='w')
    blosc2.asarray(np.zeros((1, 4, 6, 4), np.int16), urlpath=str(pre / 'val0_seg.b2nd'), mode='w')
    (pre / 'val0.pkl').write_bytes(pickle.dumps(properties()))
    monkeypatch.setattr(ev, 'build_model', lambda *a, **k: torch.nn.Conv2d(1, 2, 1))
    return args, md, pre, ref


def exported_run(tmp_path, monkeypatch):
    args, md, pre, ref = prepared_run(tmp_path, monkeypatch)
    assert ev.main(args) == 0
    root = tmp_path / 'out'
    gt = tmp_path / 'gt'; gt.mkdir()
    write_nifti(gt / 'val0.nii.gz', read_nifti(root / 'predictions/val0.nii.gz'))
    return root, tmp_path / 'raw', gt


@pytest.mark.parametrize('updates', [
    {'epoch': 9, 'best_selection_epoch': 3}, {'epoch': 0, 'best_selection_epoch': 0},
    {'epoch': True}, {'epoch': 10.0}, {'best_selection_epoch': '10'},
    {'best_selection_dice': float('nan')}, {'best_selection_dice': float('inf')},
    {'best_selection_dice': -.1}, {'best_selection_dice': 1.1}, {'best_selection_dice': True},
    {'best_selection_dice': None}, {'best_selection_epoch': None}, {'epoch': None},
    {'best_selection_dice': 10**400},
])
def test_b1_nonbest_or_invalid_selection_rejected(tmp_path, monkeypatch, updates):
    args, md, pre, ref = prepared_run(tmp_path, monkeypatch)
    path = Path(args[1]); payload, _ = ev.read_formal_payload(path)
    payload['metadata'].update(updates); torch.save(payload, path)
    monkeypatch.setattr(ev, 'build_model', lambda *a, **k: pytest.fail('must reject before model'))
    with pytest.raises(ValueError, match='selection|best'):
        ev.main(args)
    assert not (tmp_path / 'out').exists()


def test_b1_legal_best_records_selection_basis(tmp_path, monkeypatch):
    args, md, pre, ref = prepared_run(tmp_path, monkeypatch)
    assert ev.main(args) == 0
    manifest = ev.read_json(tmp_path / 'out/prediction_manifest.json')
    assert manifest['checkpoint']['selection']['epoch'] == 10
    assert manifest['checkpoint']['selection']['best_selection_epoch'] == 10
    assert manifest['checkpoint']['selection']['best_selection_dice'] == .8


@pytest.mark.parametrize('entry', ['prediction', 'metrics'])
def test_b2_source_tree_output_rejected_without_writing(tmp_path, monkeypatch, entry):
    candidate = Path(ev.__file__).resolve().parent / 'b2_never_created_output'
    assert not candidate.exists()
    if entry == 'prediction':
        args, md, pre, ref = fixture_run(tmp_path, monkeypatch)
        # Remove the incidental relative external-path protection of cwd.
        payload, _ = ev.read_formal_payload(Path(args[1]))
        payload['metadata']['resolved_config']['initialization_provenance']['path'] = str(tmp_path / 'external' / 'best.pth')
        torch.save(payload, Path(args[1]))
        args[args.index('--output-root') + 1] = str(candidate)
        with pytest.raises(ValueError, match='overlaps'): ev.main(args + ['--metadata-only'])
    else:
        root, raw, gt = exported_run(tmp_path, monkeypatch)
        # Even an older manifest must be protected by the current actual source root.
        manifest_path = root / 'prediction_manifest.json'; manifest = ev.read_json(manifest_path)
        manifest['protected_roots'] = [str(raw)]
        ev.write_json(manifest_path, manifest)
        import evaluate_segmentation_metrics as metrics
        monkeypatch.setattr(metrics, 'build_summary', lambda *a: pytest.fail('output guard must reject before report construction'))
        with pytest.raises(ValueError, match='overlaps'):
            ev.evaluate_predictions(root, raw, gt, candidate, 'paper')
    assert not candidate.exists()


@pytest.mark.parametrize('kind', ['dataset', 'plans', 'split', 'properties'])
@pytest.mark.parametrize('timing', ['read', 'running'])
def test_b3_metadata_mutation_fails_not_complete(tmp_path, monkeypatch, kind, timing):
    args, md, pre, ref = prepared_run(tmp_path, monkeypatch)
    path = {'dataset': pre / 'dataset.json', 'plans': pre / 'plans.json',
            'split': Path(args[args.index('--splits-file') + 1]), 'properties': pre / 'val0.pkl'}[kind]
    def change(): path.write_bytes(path.read_bytes() + (b' ' if kind != 'properties' else b'changed'))
    if timing == 'running':
        original = ev.restore_logits
        def wrapped(*a, **k):
            result = original(*a, **k); change(); return result
        monkeypatch.setattr(ev, 'restore_logits', wrapped)
    else:
        name = 'read_properties' if kind == 'properties' else 'read_splits' if kind == 'split' else 'read_json'
        original = getattr(ev, name)
        def wrapped(p=None, *a, **k):
            result = original(p, *a, **k)
            if p is not None and Path(p) == path: change()
            return result
        monkeypatch.setattr(ev, name, wrapped)
    with pytest.raises(ValueError, match='changed|stable|snapshot'):
        ev.main(args)
    manifest_path = tmp_path / 'out/prediction_manifest.json'
    if manifest_path.exists(): assert ev.read_json(manifest_path)['status'] == 'failed'


@pytest.mark.parametrize('kind', ['gt', 'prediction'])
@pytest.mark.parametrize('target', ['original', 'metric_input'])
def test_b4_changes_between_check_and_computation_rejected(tmp_path, monkeypatch, kind, target):
    root, raw, gt = exported_run(tmp_path, monkeypatch)
    import evaluate_segmentation_metrics as metrics
    actual = metrics.evaluate_case
    original_path = (gt if kind == 'gt' else root / 'predictions') / 'val0.nii.gz'
    def changed(pred_path, gt_path, mode):
        path = original_path if target == 'original' else gt_path if kind == 'gt' else pred_path
        image = read_nifti(path)
        write_nifti(path, NiftiVolume(np.zeros_like(image.array), image.spacing_xyz, image.origin_xyz, image.direction))
        return actual(pred_path, gt_path, mode)
    monkeypatch.setattr(metrics, 'evaluate_case', changed)
    with pytest.raises(ValueError, match='changed|snapshot'):
        ev.evaluate_predictions(root, raw, gt, tmp_path / 'metrics', 'paper')
    assert not (tmp_path / 'metrics/summary_metrics.json').exists()


@pytest.mark.parametrize('kind', ['manifest', 'source', 'gt', 'prediction', 'extra_prediction', 'split'])
def test_b4_changes_before_report_rejected(tmp_path, monkeypatch, kind):
    root, raw, gt = exported_run(tmp_path, monkeypatch)
    import evaluate_segmentation_metrics as metrics
    actual = metrics.build_summary
    def changed(frame, mode):
        result = actual(frame, mode)
        if kind == 'manifest':
            path = root / 'prediction_manifest.json'; path.write_bytes(path.read_bytes() + b' ')
        elif kind == 'extra_prediction':
            write_nifti(root / 'predictions/extra.nii.gz', read_nifti(gt / 'val0.nii.gz'))
        elif kind == 'split':
            path = tmp_path / 'splits.json'; path.write_bytes(path.read_bytes() + b' ')
        else:
            path = {'source': raw / 'imagesTr/val0_0000.nii.gz', 'gt': gt / 'val0.nii.gz',
                    'prediction': root / 'predictions/val0.nii.gz'}[kind]
            image = read_nifti(path)
            write_nifti(path, NiftiVolume(np.ones_like(image.array), image.spacing_xyz, image.origin_xyz, image.direction))
        return result
    monkeypatch.setattr(metrics, 'build_summary', changed)
    with pytest.raises(ValueError):
        ev.evaluate_predictions(root, raw, gt, tmp_path / 'metrics', 'paper')
    assert not (tmp_path / 'metrics/summary_metrics.json').exists()


@pytest.mark.parametrize('kind', ['gt', 'prediction', 'source'])
def test_b4_mutation_during_geometry_read_rejected(tmp_path, monkeypatch, kind):
    root, raw, gt = exported_run(tmp_path, monkeypatch)
    path = {'gt': gt / 'val0.nii.gz', 'prediction': root / 'predictions/val0.nii.gz',
            'source': raw / 'imagesTr/val0_0000.nii.gz'}[kind]
    original = ev.read_nifti
    def changed(p):
        image = original(p)
        if Path(p) == path:
            write_nifti(path, NiftiVolume(np.ones_like(image.array), image.spacing_xyz, image.origin_xyz, image.direction))
        return image
    monkeypatch.setattr(ev, 'read_nifti', changed)
    with pytest.raises(ValueError): ev.check_predictions(root, raw, gt)


@pytest.mark.parametrize('mode', ['paper', 'standard'])
def test_b4_stable_metrics_equal_existing_implementation(tmp_path, monkeypatch, mode):
    root, raw, gt = exported_run(tmp_path, monkeypatch)
    from evaluate_segmentation_metrics import evaluate_case
    expected = evaluate_case(root / 'predictions/val0.nii.gz', gt / 'val0.nii.gz', mode)
    report = ev.evaluate_predictions(root, raw, gt, tmp_path / 'metrics', mode)
    import pandas as pd
    row = pd.read_csv(tmp_path / 'metrics/case_metrics.csv').iloc[0]
    for key, value in expected.items():
        if isinstance(value, str): assert row[key] == value
        else: assert row[key] == pytest.approx(value, nan_ok=True)
    assert report['geometry_validation']['cases'][0]['gt_sha256'] == ev.sha256(gt / 'val0.nii.gz')


@pytest.mark.parametrize('forward', itertools.permutations(range(3)))
def test_all_six_inverse_permutations_preserved(forward):
    plan = plans_dict(); plan['transpose_forward'] = list(forward)
    plan['transpose_backward'] = [int(i) for i in np.argsort(forward)]
    pm, cm, lm = ev.official_managers(plan, {'labels': {'background': 0, 'lesion': 1}})
    prop = properties(); spacing = np.array(prop['spacing'])[list(forward)]
    from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
    shape = compute_new_shape(prop['shape_after_cropping_and_before_resampling'], spacing, [spacing[0], 1., 1.])
    logits = np.zeros((2, *shape), np.float32); logits[1] = 20
    seg, probability = ev.restore_logits(logits, prop, pm, cm, lm)
    expected = np.zeros(prop['shape_before_cropping'], np.uint8); expected[1:5, 1:4, 1:5] = 1
    np.testing.assert_array_equal(seg, expected.transpose(plan['transpose_backward']))
    assert probability.shape == (2, *seg.shape)


@pytest.mark.parametrize('key', ['epoch', 'best_selection_epoch', 'best_selection_dice'])
def test_b1_missing_selection_rejected(tmp_path, monkeypatch, key):
    args, md, pre, ref = fixture_run(tmp_path, monkeypatch)
    path = Path(args[1]); payload, _ = ev.read_formal_payload(path)
    del payload['metadata'][key]; torch.save(payload, path)
    with pytest.raises(ValueError, match='best selection'): ev.main(args + ['--metadata-only'])
    assert not (tmp_path / 'out').exists()


def test_b1_current_training_saver_best_compatible(tmp_path, monkeypatch):
    from standalone_nnunet2d.training.formal_checkpoint import FormalTrainerState, save_formal_checkpoint
    args, md, pre, ref = prepared_run(tmp_path, monkeypatch)
    model = torch.nn.Conv2d(1, 2, 1); optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    state = FormalTrainerState(10, 2500, .7, 0, .8, 10)
    save_formal_checkpoint(model, optimizer, Path(args[1]), state, md['resolved_config'],
                           rng_state={}, checkpoint_root=Path(args[1]).parent)
    assert ev.main(args) == 0
    assert ev.read_json(tmp_path / 'out/prediction_manifest.json')['checkpoint']['selection']['epoch'] == state.epoch


@pytest.mark.parametrize('kind', ['dataset', 'plans', 'properties'])
def test_b3_change_inside_bytes_read_rejected(tmp_path, monkeypatch, kind):
    args, md, pre, ref = prepared_run(tmp_path, monkeypatch)
    path = pre / {'dataset': 'dataset.json', 'plans': 'plans.json', 'properties': 'val0.pkl'}[kind]
    original = Path.read_bytes
    def changed(p):
        value = original(p)
        if p == path: p.write_bytes(value + b' ')
        return value
    monkeypatch.setattr(Path, 'read_bytes', changed)
    with pytest.raises(ValueError, match='changed'): ev.main(args)
    manifest = tmp_path / 'out/prediction_manifest.json'
    if manifest.exists(): assert ev.read_json(manifest)['status'] == 'failed'


def test_b3_properties_changed_after_case_record_rejected(tmp_path, monkeypatch):
    args, md, pre, ref = prepared_run(tmp_path, monkeypatch)
    original = ev.write_json
    def changed(path, value):
        original(path, value)
        if value.get('status') == 'running' and value.get('cases'):
            prop = pre / 'val0.pkl'; prop.write_bytes(prop.read_bytes() + b'changed')
    monkeypatch.setattr(ev, 'write_json', changed)
    with pytest.raises(ValueError, match='changed'): ev.main(args)
    assert ev.read_json(tmp_path / 'out/prediction_manifest.json')['status'] == 'failed'
