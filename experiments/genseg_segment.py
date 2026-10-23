"""Frozen image-only outer19 inference; no GT input, no training or metric changes."""
import argparse
import hashlib
import inspect
import importlib.metadata
import json
import os
import subprocess
from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path
import re
import numpy as np
import torch
from experiments.genseg30.contract import (
    REPO, REFERENCE, REFERENCE_SHA256, require, read_json, write_json, sha256,
    checked_file, capture_identities, assert_identities, check_output)
from experiments.genseg30.models import pad_plane
from experiments.genseg_joint.data import inner_split
from experiments.genseg_joint.joint_checkpoint import validate_protocol, validate_payload
from experiments.genseg_joint.models import author_unet
from experiments.genseg_joint.runtime import implementation_identity, TARGET_SHA256

PROTOCOL = dict(
    id='genseg_epoch80_image_only_support_v2_20261023', threshold=0.5,
    preprocessing='bound_nnUNet_seg_None_transpose_crop_normalize_then_data_and_seg_resample',
    retention='raw_DWI_HD_BET_2.0.1_default_TTA_native_plane_3mm_empty_slice_keep_empty_volume_fail',
    input_order='preprocess_then_keep_fill_baseline_min_then_saved_inner61_scale_no_clip_then_image_mask_cleanup_then_bottom_right_zero_pad256',
    export='threshold_then_image_support_zero_then_seg_inverse_resample_inverse_crop_inverse_axes_then_same_inverse_support_zero_bound_writer',
    support='DefaultPreprocessor.run_case_npy(seg=None)_returned_mask[0]>=0; fresh_bool_cache; same_inverse_geometry',
    difference='supervised_three_class_seg>=0_replaced_by_image_only_0_minus1_mask; resampling_boundaries_and_masked_normalization_statistics_not_pixel_equivalent',
    evidence='outer19_fold0_development; inherited_76_preprocessing_statistics; post_outer19_error_observation_inference_repair_no_independent_test_or_training_gain_claim')
SUPPORT_SOURCE = 'DefaultPreprocessor.run_case_npy(seg=None):returned_mask[0]>=0'

METRICS_SHA256 = 'ab453e1703b9e72828f19a92a417b875506a7acd30a2926fd197739487b787bb'


def require_outer19(identifiers):
    require(sha256(REFERENCE) == REFERENCE_SHA256, 'Canonical split hash mismatch')
    expected = sorted(read_json(REFERENCE)[0]['val'])
    actual = list(identifiers)
    require(len(actual) == len(set(actual)) == 19 and sorted(actual) == expected,
            'Every and only fixed outer19 required')
    return expected


def audit_receipt(run, digest=None, *, nnunet=False):
    """Metadata/hash admission: new S uses meta-only layout, no weight tensors."""
    run = Path(run).resolve(strict=True)
    require(not (run/'failed.json').exists(), 'Run has failed receipt; refused')
    receipt_path = checked_file(run,'complete.json')
    checkpoint = checked_file(run,'joint.pt')
    identities = capture_identities([receipt_path, checkpoint])
    r = read_json(receipt_path)
    require(isinstance(r,dict) and r.get('complete') is True and r.get('format') == (3 if nnunet else 2)
            and r.get('checkpoint') == 'joint.pt', 'Completed explicitly selected schema2/schema3 joint receipt required')
    digest = r.get('checkpoint_sha256') if digest is None else digest
    require(isinstance(digest,str) and re.fullmatch('[0-9a-f]{64}',digest), 'Checkpoint SHA256 required in receipt')
    p = r.get('protocol'); validate_protocol(p,nnunet=nnunet)
    require(r.get('classification') == p['classification'] and
            type(r.get('steps')) is int and r['steps'] == p['steps'], 'Receipt step/classification mismatch')
    require(type(p.get('epochs')) is int and p['epochs'] == 80,
            'Only final completed epoch80 checkpoint permitted')
    planes = p.get('total_valid_planes')
    require(type(planes) is int and planes > 0 and p['steps'] == 80*planes and
            p.get('epoch_progress') == dict(epoch=80,planes=planes,total=planes) and
            p.get('real_sampling') == 'inner61_valid_planes_without_replacement_including_negatives',
            'Incomplete epoch80 budget/progress')
    require(sha256(REFERENCE) == REFERENCE_SHA256, 'Canonical split hash mismatch')
    fixed = read_json(REFERENCE)[0]
    require(p['split'] == inner_split(sorted(fixed['train']), sorted(fixed['val']), 20261018),
            'Fixed patient split mismatch')
    require_outer19(p['split']['external'])
    require(p['target_manifest_sha256'] == TARGET_SHA256, 'Pinned target manifest mismatch')
    require(p['target_identity'].get('reference_sha256') == REFERENCE_SHA256,
            'Target canonical split mismatch')
    require(p['implementation_files_sha256'] == implementation_identity(),
            'Training implementation files differ from saved source identity')
    require(r.get('checkpoint_sha256') == digest == sha256(checkpoint), 'Checkpoint hash mismatch')
    assert_identities(identities)
    return r, identities


def restore_segmenter(values):
    """Strict CPU restoration of pinned author S; never instantiate G or D."""
    model = author_unet()
    refs = model.state_dict()
    require(isinstance(values,dict) and values.keys() == refs.keys(), 'S state keys mismatch')
    for name, ref in refs.items():
        value = values[name]
        require(isinstance(value,torch.Tensor) and value.device.type == 'cpu' and
                value.layout == torch.strided and value.shape == ref.shape and
                value.dtype == ref.dtype and bool(torch.isfinite(value).all()),
                'S tensor shape/dtype/finiteness mismatch: '+name)
    model.load_state_dict(values, strict=True)
    model.eval()
    model.requires_grad_(False)
    return model


def load_audited_segmenter(run, digest, *, target, nnunet=False):
    """Full existing schema/moments/BN audit before S construction; CPU only.

    Caller supplies the audited training metadata binding and preflights RAM.
    Joint schema, Adam, BN, alpha and same-descriptor loading remain unchanged.
    """
    from experiments.genseg_joint.preview import load_checkpoint
    r, identities = audit_receipt(run,digest,nnunet=nnunet)
    from experiments.genseg_joint.joint_checkpoint import audit_joint
    bound, bound_identities = audit_joint(target,run,digest,nnunet=nnunet)
    require(bound == r, 'Bound target receipt changed')
    identities.update(bound_identities)
    payload = load_checkpoint(run,'joint.pt',digest)  # same descriptor, hash before/after, weights_only
    assert_identities(identities)
    validate_payload(payload,r,nnunet=nnunet)  # preserve G/D/S, optimizer, alpha and BN validation
    if nnunet:
        from experiments.genseg_joint.nnunet_s import restore
        model=restore(payload['S'],payload['model_config']['S'])
    else:
        model = restore_segmenter(payload['S'])
    assert_identities(identities)
    require(sha256(Path(run)/'joint.pt') == digest, 'Checkpoint hash changed after restore')
    require(r['protocol']['implementation_files_sha256'] == implementation_identity(),
            'Training implementation changed during restore')
    return model, r


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def infer_prepared_volume(model, normalized_image, *, valid_support, pad_multiple=256, resource_guard=None, audit=None):
    """Engineering primitive only: input must ALREADY satisfy training contract.

    Requires the audited image-side support; takes no labels or lesion ROI.
    It neither establishes nor guesses
    raw-DWI preprocessing. Full planes, including negative/background planes,
    are evaluated; no TTA, threshold search or component postprocessing.
    """
    image = np.asarray(normalized_image)
    require(image.ndim == 3 and all(n > 0 for n in image.shape) and
            image.dtype == np.float32 and np.isfinite(image).all(), 'Finite float32 [Z,H,W] required')
    require(isinstance(valid_support,np.ndarray) and valid_support.dtype==np.bool_ and
            valid_support.shape==image.shape, 'Boolean image support/grid required')
    cleared=0
    require(all(not m.training for m in model.modules()), 'All S modules must be eval')
    require(type(pad_multiple) is int and pad_multiple > 0, 'Positive padding multiple required')
    before = state_digest(model)
    modes = [(m,m.training) for m in model.modules()]
    output = np.empty(image.shape,np.uint8)
    device = next(model.parameters()).device
    with torch.inference_mode():
        for z, plane in enumerate(image):
            if resource_guard is not None: resource_guard()
            descriptor=getattr(model,'genseg_s_descriptor',None)
            if descriptor is None:
                padded = pad_plane(plane,pad_multiple,0)
            else:
                from experiments.genseg_joint.nnunet_s import padded_shape
                h,w=padded_shape(descriptor,plane.shape,multiple=pad_multiple)
                padded=np.pad(plane,((0,h-plane.shape[0]),(0,w-plane.shape[1])),constant_values=0)
            tensor = torch.from_numpy(np.ascontiguousarray(padded[None,None])).to(device)
            logits = model(tensor)
            require(logits.shape == tensor.shape and bool(torch.isfinite(logits).all()),
                    'Expected finite single-channel S logits')
            thresholded=(torch.sigmoid(logits[0,0,:plane.shape[0],:plane.shape[1]]) >= .5).cpu().numpy()
            cleared+=int(np.count_nonzero(thresholded & ~valid_support[z]))
            output[z]=thresholded & valid_support[z]
    require(all(m.training == mode for m,mode in modes) and before == state_digest(model),
            'S parameters/BN/modes changed during inference')
    if audit is not None: audit['cleared_preprocessed_foreground']=cleared
    return output


def restore_binary_geometry(prediction, plans_manager, configuration_manager, properties):
    """Inverse image-side geometry: seg resample -> inverse crop -> inverse axes.

    Binary threshold is applied BEFORE restoration. Uses the existing nnU-Net
    segmentation resampler for binary labels, not raw affine relabeling.
    properties must originate from the image pipeline (never a GT-derived crop).
    """
    from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
    pred = np.asarray(prediction)
    require(pred.ndim == 3 and np.isin(pred,(0,1)).all(), 'Binary 3D prediction required')
    forward = tuple(plans_manager.transpose_forward)
    backward = tuple(plans_manager.transpose_backward)
    require(sorted(forward) == sorted(backward) == [0,1,2] and
            all(forward[backward[i]] == i for i in range(3)), 'Invalid transpose inverse')
    before = tuple(properties['shape_before_cropping'])
    cropped = tuple(properties['shape_after_cropping_and_before_resampling'])
    bbox = properties['bbox_used_for_cropping']
    require(len(before) == len(cropped) == len(bbox) == 3, 'Invalid bbox dimensionality')
    for axis, pair in enumerate(bbox):
        require(len(pair) == 2 and all(isinstance(v,(int,np.integer)) for v in pair) and
                0 <= pair[0] < pair[1] <= before[axis] and pair[1]-pair[0] == cropped[axis],
                'Invalid bbox bounds/shape')
    spacing = [properties['spacing'][i] for i in forward]
    current = list(configuration_manager.spacing)
    if len(current) == 2: current = [spacing[0], *current]
    require(len(current) == len(spacing) == 3 and
            np.isfinite(current+spacing).all() and min(current+spacing) > 0, 'Invalid spacing')
    require(tuple(compute_new_shape(cropped,spacing,current)) == pred.shape,
            'Prediction grid differs from image preprocessing geometry')
    restored = np.asarray(configuration_manager.resampling_fn_seg(
        pred[None].astype(np.uint8),cropped,current,spacing))
    require(restored.shape == (1,*cropped) and np.isin(restored,(0,1)).all(),
            'Invalid inverse-resampled binary mask')
    full = np.zeros(before,np.uint8)
    full[tuple(slice(*pair) for pair in bbox)] = restored[0]
    return full.transpose(backward)


def fresh_output(path, protected):
    target=check_output(path,protected)
    for parent in target.parents:
        require(not any((parent/name).exists() for name in
            ('complete.json','started.json','failed.json','targets.json','generator.json')),
            'Output is inside an existing experiment/artifact directory')
    return target


def source_files(pm, cm):
    """Record executable preprocessing sources, including actual normalizer/backend."""
    from nnunetv2.preprocessing.normalization import default_normalization_schemes as norm
    from nnunetv2.preprocessing.cropping import cropping
    objects = [pm.image_reader_writer_class, cm.preprocessor_class, cropping,
               cm.resampling_fn_data.func, cm.resampling_fn_seg.func]
    for name in cm.normalization_schemes:
        cls = getattr(norm,name,None)
        require(cls is not None, 'Unsupported actual normalizer: '+str(name))
        objects.append(cls)
    paths = {Path(inspect.getfile(obj)).resolve() for obj in objects}
    paths.update(Path(__file__).parent.glob('genseg3mm/*.py'))
    paths.update(Path(__file__).parent.glob('genseg30/*.py'))
    import HD_BET
    paths.update(Path(HD_BET.__file__).resolve().parent.rglob('*.py'))
    paths.add(Path(inspect.getfile(type(pm))).resolve())
    paths.add(Path(inspect.getfile(type(cm))).resolve())
    paths.add(Path(__file__).resolve())
    return {str(p):sha256(p) for p in sorted(paths)}


def bind_metadata(run, source_folder, raw_root, targets, baseline, digest=None, *, nnunet=False):
    """No patient arrays, external properties or label contents are read.
    New S layout is checked on meta device without allocating model weights.

    This is a metadata binding for audit_joint, not TargetDataset reconstruction.
    Training target contents are pinned by the original manifest hash; they are
    never used to construct outer19 inputs. No load_train/training_dataset API.
    """
    from experiments.genseg30.contract import Source
    from experiments.genseg30.pipeline import load_run
    from experiments.genseg3mm.targets import RECIPE, CLASSIFICATION, baseline_training_kwargs, _hd_bet_runtime
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    receipt, identities = audit_receipt(run,digest,nnunet=nnunet)
    source = Source(source_folder,raw_root)
    if nnunet:
        from experiments.genseg_joint.nnunet_s import from_source
        require(receipt['protocol']['model_config']['S']==from_source(source),'S factory differs from actual SOURCE plans/dataset/split')
    require_outer19(source.val)
    manifest_path = checked_file(targets,'targets.json')
    require(sha256(manifest_path) == TARGET_SHA256, 'Pinned target manifest mismatch')
    manifest = read_json(manifest_path)
    require(manifest.get('format') == 1 and manifest.get('complete') is True and
            manifest.get('classification') == CLASSIFICATION and manifest.get('recipe') == RECIPE,
            'Target recipe/schema mismatch')
    source.verify_identity(manifest['source'])
    require(set(manifest['cases']) == set(source.train), 'Training target cohort mismatch')
    record, checkpoint = load_run(source,baseline)
    metadata = record['metadata']; baseline_training_kwargs(metadata,device='cuda:0')
    require(manifest['baseline']['metadata'] == metadata and
            manifest['baseline']['checkpoint_sha256'] == record['checkpoint_sha256'] and
            Path(manifest['baseline']['run']).resolve() == Path(baseline).resolve() and
            manifest['source_inventory'] == metadata['inventory'], 'Baseline metadata binding mismatch')
    require(np.isfinite(manifest['background_fill']) and
            manifest['background_fill'] == metadata['scaling']['min'], 'Baseline fill mismatch')
    identity = dict(source.identity,genseg3mm=dict(classification=CLASSIFICATION,
        target_manifest_sha256=TARGET_SHA256,baseline_checkpoint_sha256=record['checkpoint_sha256'],recipe=RECIPE))
    require(identity == receipt['protocol']['target_identity'], 'Training plans/dataset/split/target identity mismatch')
    pm = PlansManager(source.plans); cm = pm.get_configuration('2d')
    require(cm.preprocessor_class is DefaultPreprocessor, 'Unsupported actual preprocessor; require DefaultPreprocessor seg=None contract')
    require(pm.image_reader_writer_class is SimpleITKIO, 'Unsupported actual reader/backend; require exact native SimpleITKIO grid')
    require(len(cm.use_mask_for_norm) == 1 and type(cm.use_mask_for_norm[0]) is bool and
            len(cm.normalization_schemes) == 1, 'Single channel mask-normalization setting required')
    version, executable, weights, weight_hash = _hd_bet_runtime()
    hd = manifest['hd_bet']
    require(hd.get('version') == version == '2.0.1' and hd.get('default_tta') is True and
            hd.get('weights_sha256') == weight_hash, 'HD-BET recipe/weights mismatch')
    paths = [source.plans_path,source.dataset_path,source.splits,REFERENCE,manifest_path,
             Path(baseline)/'generator.json',checkpoint,weights,Path(run)/'complete.json',Path(run)/'joint.pt']
    identities.update(capture_identities(paths))
    hashes = {str(p):sha256(p) for p in paths}
    def inference_identity():
        files=source_files(pm,cm)
        if nnunet:
            from experiments.genseg_joint.nnunet_s import runtime_source_files
            files.update({str(p.resolve()):sha256(p) for p in runtime_source_files().values()})
        return files
    code = inference_identity()
    def verify(expected, full=True):
        require(expected == identity, 'Target metadata identity changed')
        assert_identities(identities)
        if full:
            require(all(sha256(p)==v for p,v in hashes.items()), 'Bound metadata/checkpoint changed')
        require(inference_identity()==code and implementation_identity()==receipt['protocol']['implementation_files_sha256'],
                'Inference/training sources changed')
    target = SimpleNamespace(train=source.train,val=source.val,identity=identity,
        manifest_sha256=TARGET_SHA256,verify_identity=verify)
    versions={name:importlib.metadata.version(name) for name in ('torch','nnunetv2','hd-bet','numpy','scipy','SimpleITK')}
    audit = dict(runtime_versions=versions,actual_plans_status='VERIFIED_ON_EXECUTING_HOST',source_identity=identity,
        use_mask_for_norm=cm.use_mask_for_norm,normalization_schemes=cm.normalization_schemes,
        configuration=cm.configuration,reader='SimpleITKIO',preprocessor='DefaultPreprocessor',
        metadata_files_sha256=hashes,inference_sources_sha256=code,
        training_sources_sha256=receipt['protocol']['implementation_files_sha256'],
        protected_roots=[str(Path(p).resolve()) for p in source.protected+[Path(run),Path(targets),Path(baseline)]],
        hd_bet=dict(recipe=RECIPE,weights_sha256=weight_hash,weights_file=str(weights),version=version),
        fill=manifest['background_fill'],saved_scaling=receipt['protocol']['scaling'],
        frozen_protocol=PROTOCOL,training_target_arrays_reloaded=False,
        support_geometry=dict(plans=source.plans,plans_file=str(source.plans_path),configuration='2d'))
    if nnunet:
        audit.update(s_architecture=receipt['protocol']['model_config']['S'],checkpoint_format=3,
            s_padding='bottom_right_image0_lcm256_actual_axis_strides; single_logit_sigmoid_ge_0.5')
    verify(identity)
    return SimpleNamespace(receipt=receipt,source=source,target=target,pm=pm,cm=cm,
        manifest=manifest,audit=audit,verify=lambda full=True:verify(identity,full),targets=Path(targets),weights=weights,
        executable=executable,weight_hash=weight_hash)


def preprocess_image(data, properties, pm, cm, dataset_json):
    """Use actual bound implementation; never pass a supervised segmentation."""
    require('class_locations' not in properties, 'External supervised properties forbidden')
    data = np.asarray(data)
    require(data.ndim==4 and data.shape[0]==1 and np.isfinite(data).all(), 'Finite single-channel raw DWI required')
    x, mask, props = cm.preprocessor_class(verbose=False).run_case_npy(
        data,None,deepcopy(properties),pm,cm,dataset_json)
    require(x.dtype==np.float32 and x.shape==mask.shape and np.isfinite(x).all() and
            np.isin(mask,(-1,0)).all() and 'class_locations' not in props,
            'Invalid seg=None preprocessing result')
    return x,mask,props


def prepare_s_input(image, image_mask, keep, *, fill, scale):
    require(image.shape==image_mask.shape and image.shape[0]==1 and keep.dtype==np.bool_ and
            keep.shape==image.shape[1:] and np.isin(image_mask,(-1,0)).all(), 'Input mask/grid mismatch')
    require(np.isfinite(fill) and np.isfinite([scale['min'],scale['max']]).all() and
            scale['max']>scale['min'], 'Invalid fill/saved scale')
    x=image[0].copy(); x[~keep]=fill
    x=(2*(x-scale['min'])/(scale['max']-scale['min'])-1).astype(np.float32)
    x=np.where(image_mask[0]>=0,x,0).astype(np.float32)
    require(np.isfinite(x).all(), 'Nonfinite prepared input')
    return x


def geometry(path):
    import SimpleITK as sitk
    image = sitk.ReadImage(str(path))
    require(image.GetDimension()==3, 'Full 3D image required')
    return dict(size_xyz=list(image.GetSize()),spacing_xyz=list(image.GetSpacing()),
                origin_xyz=list(image.GetOrigin()),direction=list(image.GetDirection()))


def same_geometry(actual, expected):
    require(actual['size_xyz']==expected['size_xyz'] and
            all(np.allclose(actual[k],expected[k],atol=1e-5,rtol=0)
                for k in ('spacing_xyz','origin_xyz','direction')), 'Raw/prediction/GT geometry mismatch')


def check_binary(path):
    import SimpleITK as sitk
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    require(arr.ndim==3 and np.isin(arr,(0,1)).all(), 'Prediction must be finite binary full volume')


def export_prediction(pred, output, raw, pm, cm, props, *, valid_support, audit=None):
    output=Path(output)
    require(not output.exists(), 'Prediction output exists; overwrite refused')
    require(valid_support.dtype==np.bool_ and valid_support.shape==pred.shape, 'Boolean image support/grid required')
    require(not np.any(pred[~valid_support]), 'Preprocessed prediction outside image support')
    restored=restore_binary_geometry(pred,pm,cm,props)
    support=restore_binary_geometry(valid_support.astype(np.uint8),pm,cm,props).astype(bool)
    if audit is not None:
        audit['cleared_restored_foreground']=int(np.count_nonzero(restored[~support]))
        audit['restored_support']=support_identity(support)
    restored[~support]=0
    pm.image_reader_writer_class().write_seg(restored,str(output),props)
    same_geometry(geometry(output),geometry(raw)); check_binary(output)
    # Matching writer and reader must also reproduce the actual label array.
    reread,_=pm.image_reader_writer_class().read_seg(str(output))
    require(np.array_equal(reread[0],restored), 'Writer roundtrip changed binary voxels')
    require(not np.any(reread[0][~support]), 'Written prediction outside image support')
    return geometry(output)


def native_keep(binding, patient, raw, cache, guard):
    """Fresh cache only: existing caches are refused rather than silently reused."""
    import SimpleITK as sitk
    from experiments.genseg3mm.targets import (
        _build_hd_bet_command,mask_image_to_raw_grid,expand_keep_mask_3mm,validate_nifti_mm_orthogonal)
    case=Path(cache)/patient; case.mkdir(exist_ok=False)
    raw_hash=sha256(raw); validate_nifti_mm_orthogonal(raw)
    image=sitk.ReadImage(str(raw))
    las=sitk.DICOMOrient(image,'LAS'); input_file=case/'input_las.nii.gz'
    sitk.WriteImage(las,str(input_file),True)
    command=_build_hd_bet_command(binding.executable,input_file,case/'hd_bet.nii.gz')
    guard(phase='startup')  # subprocess has not started; other workloads must be idle
    write_json(case/'started.json',dict(command=command,raw_sha256=raw_hash,
        weights_sha256=binding.weight_hash,recipe=binding.manifest['recipe'],default_tta=True))
    mask=case/'hd_bet_bet.nii.gz'
    with (case/'hd-bet.log').open('x',encoding='utf-8') as log, (case/'resources.jsonl').open('x',encoding='utf-8') as telemetry:
        process=subprocess.Popen(command,cwd=case,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
        try:
            while True:
                try:
                    code=process.wait(timeout=1.)
                    break
                except subprocess.TimeoutExpired:
                    snapshot=guard(phase='between_units')
                    telemetry.write(json.dumps(snapshot,allow_nan=False)+'\n'); telemetry.flush()
        except BaseException as original:
            # Only this Popen child is ours. Secondary cleanup/diagnostic failures
            # must never replace the original resource or telemetry exception.
            cleanup_errors=[]
            def cleanup_error(stage, error):
                try:
                    cleanup_errors.append(dict(stage=stage,error_type=type(error).__name__))
                except BaseException:
                    pass  # best-effort diagnostics while preserving original
            try: process.terminate()
            except BaseException as error: cleanup_error('terminate',error)
            try: process.wait(timeout=15.)
            except BaseException as error:
                cleanup_error('wait',error)
                try: process.kill()
                except BaseException as error: cleanup_error('kill',error)
                try: process.wait(timeout=15.)
                except BaseException as error: cleanup_error('reap',error)
            try: original.cleanup_errors=cleanup_errors
            except BaseException:
                pass  # some exception objects may reject diagnostic attributes
            raise
    require(code==0 and mask.is_file(), 'HD-BET failed; preserve partial cache')
    require(sha256(binding.weights)==binding.weight_hash and sha256(raw)==raw_hash, 'HD-BET source changed')
    mask_image=sitk.ReadImage(str(mask))
    same_geometry(geometry(mask),geometry(input_file))
    _,brain=mask_image_to_raw_grid(mask_image,image)
    keep=expand_keep_mask_3mm(brain,spacing_xyz=image.GetSpacing())
    receipt=dict(raw_sha256=raw_hash,mask_file=str(mask.resolve()),mask_sha256=sha256(mask),
        input_las_file=str(input_file.resolve()),input_las_sha256=sha256(input_file),
        weights_sha256=binding.weight_hash,recipe=binding.manifest['recipe'],command=command,
        log_sha256=sha256(case/'hd-bet.log'),empty_native_slices=int((~brain.any(axis=(1,2))).sum()))
    write_json(case/'complete.json',receipt)
    return keep,receipt


def support_identity(support):
    return dict(source=SUPPORT_SOURCE,protocol=PROTOCOL['id'],shape=list(support.shape),
                dtype='bool',sha256=hashlib.sha256(support.tobytes()).hexdigest())


def load_prepared(rec):
    """Strict fresh-cache admission; no legacy cache or inferred/all-one support."""
    require(sha256(rec['prepared_file'])==rec['prepared_file_sha256'], 'Derived input cache hash mismatch')
    with np.load(rec['prepared_file'],allow_pickle=False) as arrays:
        require(set(arrays.files)=={'image','valid_support'}, 'Missing/unexpected image support cache fields')
        image=arrays['image']; support=arrays['valid_support']
    require(image.ndim==3 and image.dtype==np.float32 and np.isfinite(image).all(), 'Invalid prepared image')
    require(support.dtype==np.bool_ and support.shape==image.shape and
            rec.get('preprocessed_shape')==[1,*image.shape], 'Invalid image support dtype/shape')
    require(rec.get('valid_support')==support_identity(support), 'Image support source/protocol/shape/hash mismatch')
    require(hashlib.sha256(image.tobytes()).hexdigest()==rec['prepared_sha256'], 'Prepared array changed')
    return image,support


def verify_predictions(output, receipt, raw_folder):
    """Recheck every artifact; complete existence alone never admits scoring."""
    output=Path(output).resolve(strict=True); raw_folder=Path(raw_folder)
    require(receipt.get('complete') is True and receipt.get('protocol')==PROTOCOL,
            'Frozen complete prediction receipt required')
    require(not any((output/name).exists() for name in ('failed.json','partial.json')), 'Failed/partial prediction refused')
    cache_value=receipt.get('cache_root')
    require(isinstance(cache_value,str) and bool(cache_value), 'Missing cache root')
    cache=Path(cache_value).resolve(strict=True)
    require(cache.is_dir() and cache_value==str(cache), 'Canonical absolute cache root required')
    require(cache!=output and cache not in output.parents and output not in cache.parents,
            'Prediction and cache roots must be disjoint')
    ids=require_outer19(receipt.get('external_ids',[]))
    require_outer19(receipt.get('cases',{}).keys())
    files=list(output.glob('*.nii*'))
    require(sorted(p.name for p in files)==[cid+'.nii.gz' for cid in ids], 'Missing/extra prediction files')
    for field in ('metadata_files_sha256','inference_sources_sha256'):
        require(isinstance(receipt.get('audit',{}).get(field),dict) and bool(receipt['audit'][field]), 'Missing inference audit hashes')
    require(receipt.get('state_before')==receipt.get('state_after') and
            isinstance(receipt.get('state_before'),str) and len(receipt['state_before'])==64, 'S state mismatch')
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    chain=receipt.get('audit',{}).get('support_geometry')
    require(isinstance(chain,dict) and chain.get('configuration')=='2d' and
            isinstance(chain.get('plans'),dict), 'Missing image support geometry binding')
    plans_path=chain.get('plans_file')
    hashes=receipt['audit']['metadata_files_sha256']
    require(isinstance(plans_path,str) and plans_path in hashes and
            sha256(plans_path)==hashes[plans_path] and read_json(plans_path)==chain['plans'],
            'Image support geometry plans identity mismatch')
    pm=PlansManager(chain['plans']); cm=pm.get_configuration(chain['configuration'])
    for cid in ids:
        rec=receipt['cases'][cid]
        require(rec.get('prediction_file')==cid+'.nii.gz', 'Case filename mismatch')
        pred=checked_file(output,cid+'.nii.gz'); raw=checked_file(raw_folder,cid+'_0000.nii.gz')
        require(rec.get('prediction_sha256')==sha256(pred) and rec.get('raw_sha256')==sha256(raw), 'Case hash mismatch')
        check_binary(pred); same_geometry(geometry(pred),geometry(raw)); same_geometry(geometry(pred),rec['geometry'])
        case=cache/cid
        require(case.is_dir() and case.resolve(strict=True)==case, 'Invalid cache case directory')
        for field,digest_field,name in (('prepared_file','prepared_file_sha256','prepared.npz'),
                                      ('properties_file','properties_sha256','properties.json')):
            path=Path(rec[field]).resolve(strict=True)
            require(path==case/name, 'Derived input escaped cache case directory')
            require(sha256(path)==rec[digest_field], 'Derived input cache hash mismatch')
        prepared,support=load_prepared(rec)
        del prepared
        props=read_json(rec['properties_file'])
        require('class_locations' not in props, 'Supervised cached properties forbidden')
        native=restore_binary_geometry(support.astype(np.uint8),pm,cm,props).astype(bool)
        require(rec.get('restored_support')==support_identity(native), 'Restored image support identity mismatch')
        require(all(type(rec.get(k)) is int and rec[k]>=0 for k in
                ('cleared_preprocessed_foreground','cleared_restored_foreground')), 'Missing support clearing audit')
        reread,_=pm.image_reader_writer_class().read_seg(str(pred))
        require(reread.shape==(1,*native.shape) and not np.any(reread[0][~native]),
                'Prediction outside restored image support')
        mask=Path(rec['hd_bet']['mask_file']).resolve(strict=True)
        require(mask==case/'hd_bet_bet.nii.gz', 'Derived mask escaped cache case directory')
        require(sha256(mask)==rec['hd_bet']['mask_sha256'], 'Derived mask hash mismatch')
        require(read_json(checked_file(output,cid+'.json'))==rec, 'Per-case manifest mismatch')
    return ids


def record_failure(output, exc, cases):
    """Never leave complete; cleanup/report errors must not replace original error."""
    try:
        (Path(output)/'complete.json').unlink(missing_ok=True)
    except OSError:
        pass  # failed marker additionally makes any remaining complete ineligible
    try:
        if not (Path(output)/'failed.json').exists():
            write_json(Path(output)/'failed.json',dict(complete=False,status='FAILED',
                error_type=type(exc).__name__,reason=str(exc),partial_cases=cases,
                resource_snapshot=getattr(exc,'resource_snapshot',None),
                cleanup_errors=getattr(exc,'cleanup_errors',[])))
    except Exception:
        pass


def predict(binding, run, output, cache, *, model_reserve_bytes, case_reserve_bytes):
    from experiments.genseg_joint.joint_resources import ResourceGuard
    from experiments.genseg3mm.targets import _load_case_raw,map_keep_mask_to_preprocessed
    require(type(model_reserve_bytes) is int and model_reserve_bytes>0 and
            type(case_reserve_bytes) is int and case_reserve_bytes>0, 'Positive explicit RAM reserves required')
    protected=binding.source.protected+[Path(run),binding.weights,
        Path(binding.manifest['baseline']['run'])]
    # Include the target root explicitly, not just the plans/source roots.
    protected.append(Path(binding.targets))
    output=fresh_output(output,protected); cache=fresh_output(cache,protected+[output])
    guard=ResourceGuard(1); initial=guard(model_reserve_bytes+case_reserve_bytes)
    require(not torch.cuda.is_initialized(), 'GPU1 binding must precede CUDA initialization')
    guard.bind()
    require(torch.cuda.is_available(), 'Production prediction requires CUDA')
    torch.cuda.set_per_process_memory_fraction(initial['allocator_fraction'],0)
    ids=require_outer19(binding.source.val)
    raw_folder=Path(binding.source.raw_root)/binding.source.base.name/'imagesTr'
    # Names/role gates precede first outer19 image read.
    for cid in ids: checked_file(raw_folder,cid+'_0000.nii.gz')
    output.mkdir(parents=True,exist_ok=False); cases={}
    try:
        cache.mkdir(parents=True,exist_ok=False)
        write_json(output/'started.json',dict(protocol=PROTOCOL,audit=binding.audit,resources=initial,cache_root=str(cache)))
        binding.verify()
        # Each raw image is preprocessed BEFORE HD-BET; HD-BET always sees raw DWI.
        # Cache only image-derived inputs/properties, then release patient arrays.
        for cid in ids:
            guard(case_reserve_bytes,phase='startup')
            binding.verify(full=False)
            raw,data,props,raw_image=_load_case_raw(binding.source,cid,binding.pm)
            raw_hash=sha256(raw)
            x,valid,props=preprocess_image(data,props,binding.pm,binding.cm,binding.source.dataset_json)
            native,hd=native_keep(binding,cid,raw,cache,guard)
            # native_keep returns the audited original-grid 3mm keep mask.
            keep=map_keep_mask_to_preprocessed(native,binding.pm,binding.cm,props,expected_shape=x.shape)
            prepared=prepare_s_input(x,valid,keep,fill=binding.manifest['background_fill'],scale=binding.receipt['protocol']['scaling'])
            require(sha256(raw)==raw_hash==hd['raw_sha256'], 'Raw changed during preprocessing')
            case_dir=cache/cid; case_dir.mkdir(exist_ok=True)
            array_path=case_dir/'prepared.npz'; properties_path=case_dir/'properties.json'
            support=(valid[0]>=0)
            with array_path.open('xb') as stream: np.savez_compressed(stream,image=prepared,valid_support=support)
            # Fresh reader/preprocessor properties only; no external pickle/class_locations.
            safe_props=json.loads(json.dumps(props,default=lambda v:np.asarray(v).tolist(),allow_nan=False))
            write_json(properties_path,safe_props)
            cases[cid]=dict(hd_bet=hd,raw_sha256=raw_hash,valid_support=support_identity(support),
                prepared_file=str(array_path.resolve()),prepared_file_sha256=sha256(array_path),
                properties_file=str(properties_path.resolve()),properties_sha256=sha256(properties_path),
                preprocessing_properties={k:np.asarray(props[k]).tolist() for k in
                    ('spacing','shape_before_cropping','shape_after_cropping_and_before_resampling','bbox_used_for_cropping')},
                image_mask_values=np.unique(valid).tolist(),preprocessed_shape=list(x.shape),
                keep_sha256=hashlib.sha256(keep.tobytes()).hexdigest(),
                prepared_sha256=hashlib.sha256(prepared.tobytes()).hexdigest())
            del data,x,valid,support,prepared,keep,native,raw_image
        guard(model_reserve_bytes+case_reserve_bytes,phase='startup')
        model,_=load_audited_segmenter(run,binding.receipt['checkpoint_sha256'],target=binding.target,
            **({'nnunet':True} if binding.receipt.get('format')==3 else {}))
        model=model.to('cuda:0'); before=state_digest(model)
        with (output/'resources.jsonl').open('x',encoding='utf-8') as log:
            for cid in ids:
                snapshot=guard(case_reserve_bytes,phase='between_units')
                log.write(json.dumps(snapshot,allow_nan=False)+'\n'); log.flush()
                binding.verify(full=False)
                raw=checked_file(raw_folder,cid+'_0000.nii.gz'); rec=cases[cid]
                raw_hash=sha256(raw)
                require(raw_hash==rec['raw_sha256'] and sha256(rec['prepared_file'])==rec['prepared_file_sha256'] and
                        sha256(rec['properties_file'])==rec['properties_sha256'], 'Raw/derived input cache changed')
                prepared,support=load_prepared(rec)
                props=read_json(rec['properties_file'])
                require('class_locations' not in props, 'Supervised cached properties forbidden')
                def sample_resources():
                    value=guard(phase='between_units')
                    log.write(json.dumps(value,allow_nan=False)+'\n'); log.flush()
                pred=infer_prepared_volume(model,prepared,valid_support=support,resource_guard=sample_resources,audit=rec)
                filename=cid+'.nii.gz'
                geom=export_prediction(pred,output/filename,raw,binding.pm,binding.cm,props,valid_support=support,audit=rec)
                require(sha256(raw)==raw_hash, 'Raw image changed during prediction')
                cases[cid].update(prediction_file=filename,prediction_sha256=sha256(output/filename),geometry=geom)
                write_json(output/(cid+'.json'),cases[cid])
                print('PREDICT '+cid,flush=True)
                del prepared,support,pred
        binding.verify(); after=state_digest(model)
        receipt=dict(complete=True,protocol=PROTOCOL,external_ids=ids,cases=cases,cache_root=str(cache),
            state_before=before,state_after=after,audit=binding.audit,
            checkpoint_sha256=binding.receipt['checkpoint_sha256'])
        verify_predictions(output,receipt,raw_folder)
        write_json(output/'complete.json',receipt)
        verify_predictions(output,read_json(output/'complete.json'),raw_folder)
        return receipt
    except BaseException as exc:
        record_failure(output,exc,cases)
        raise


def accept_for_scoring(output, raw_folder, gt_folder, metrics_output):
    """Independent command: all prediction checks finish before first GT read."""
    output=Path(output).resolve(strict=True)
    receipt=read_json(checked_file(output,'complete.json'))
    protected=[REPO,output,Path(raw_folder),Path(gt_folder)]
    audit=receipt.get('audit',{})
    protected.extend(Path(p).parent for p in audit.get('metadata_files_sha256',{}))
    protected.extend(Path(p) for p in audit.get('protected_roots',[]))
    identity=audit.get('source_identity',{})
    protected.extend(Path(identity[k]) for k in ('source_folder','dataset_base','raw_root') if k in identity)
    ids=verify_predictions(output,receipt,raw_folder)
    protected.append(Path(receipt['cache_root']))
    metrics_output=fresh_output(metrics_output,protected)
    require(sha256(REPO/'evaluate_segmentation_metrics.py')==METRICS_SHA256, 'Original metric script changed')
    # Recheck bound source/code identities as well as predictions.
    for field in ('metadata_files_sha256','inference_sources_sha256'):
        require(bool(audit.get(field)), 'Missing inference audit hashes')
        require(all(sha256(p)==v for p,v in audit[field].items()), 'Audited source changed before scoring')
    require(implementation_identity()==audit.get('training_sources_sha256'), 'Training source changed before scoring')
    for cid in ids:
        gt=checked_file(gt_folder,cid+'.nii.gz')
        same_geometry(geometry(gt),geometry(Path(raw_folder)/(cid+'_0000.nii.gz')))
    print('SCORING_ADMISSION_PASS: exact outer19; raw/GT geometry; hashes; original metrics; fresh output')
    return metrics_output


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('audit','predict'):
        p=sub.add_parser(name)
        p.add_argument('--run',type=Path,required=True)
        p.add_argument('--s-kind',choices=['author_unet','nnunet2d'],default='author_unet')
        p.add_argument('--checkpoint-sha256')  # default comes from complete, cross-checked with joint.pt
        for flag in ('source','raw-root','targets','baseline'):
            p.add_argument('--'+flag,type=Path,required=True)
        if name=='predict':
            p.add_argument('--output',type=Path,required=True)
            p.add_argument('--mask-cache',type=Path,required=True)
            p.add_argument('--model-reserve-gib',type=int,required=True)
            p.add_argument('--case-reserve-gib',type=int,required=True)
    p=sub.add_parser('accept')
    for flag in ('output','raw-images','gt-dir','metrics-output'):
        p.add_argument('--'+flag,type=Path,required=True)
    a=parser.parse_args(argv)
    if a.command=='accept':
        accept_for_scoring(a.output,a.raw_images,a.gt_dir,a.metrics_output)
        return 0
    # Refuse unsafe or old paths even before metadata binding.
    if a.command=='predict':
        protected=[REPO,a.run,a.source.parent,a.raw_root,a.targets,a.baseline]
        fresh_output(a.output,protected); fresh_output(a.mask_cache,protected+[a.output])
    b=bind_metadata(a.run,a.source,a.raw_root,a.targets,a.baseline,a.checkpoint_sha256,nnunet=a.s_kind=='nnunet2d')
    b.targets=a.targets
    if a.command=='audit':
        print(json.dumps(b.audit,indent=2,allow_nan=False))
        print('METADATA_AUDIT_PASS; no patient arrays, allocated model weights, GT or CUDA loaded')
    else:
        predict(b,a.run,a.output,a.mask_cache,
            model_reserve_bytes=a.model_reserve_gib*1024**3,case_reserve_bytes=a.case_reserve_gib*1024**3)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
