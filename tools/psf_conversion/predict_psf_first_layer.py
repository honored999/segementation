"""Official nnUNetPredictor with an optional frozen PSF first convolution.

`--operator legacy_block_input` (the default) keeps the historical U17 block-input
chain; `pixel_psf_superposition_v1` is the opt-in one-PSF-per-pixel protocol.
Both ideal and fitted PSFs share the same selected protocol.
"""
import argparse
import json
import inspect
import os
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.psf_conversion.psf_first_conv import (
    FirstConvHook, KEY, LEGACY_OPERATOR, PIXEL_OPERATOR, load_psfs, operator_record,
    psf_canvas, sha256)
from tools.psf_conversion.first_layer_model import check_artifact_origin, load_network


def initialize_predictor(predictor, args, payload):
    if payload.get('first_layer') is None:
        predictor.initialize_from_trained_model_folder(
            str(args.model_folder), use_folds=(args.fold,), checkpoint_name=args.checkpoint)
        return predictor.network.get_submodule(KEY)
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.inference import predict_from_raw_data as implementation

    dataset = json.loads((args.model_folder/'dataset.json').read_text(encoding='utf-8-sig'))
    plans = PlansManager(str(args.model_folder/'plans.json'))
    configuration = plans.get_configuration(payload['init_args']['configuration'])
    if hasattr(implementation, 'recursive_find_trainer_class_by_name'):
        trainer = implementation.recursive_find_trainer_class_by_name(payload['trainer_name'])
    else:
        import nnunetv2
        trainer = implementation.recursive_find_python_class(
            str(Path(nnunetv2.__path__[0])/'training'/'nnUNetTrainer'),
            payload['trainer_name'], 'nnunetv2.training.nnUNetTrainer')
    build = trainer.build_network_architecture
    inputs = determine_num_input_channels(plans, configuration, dataset)
    outputs = plans.get_label_manager(dataset).num_segmentation_heads
    if 'plans_manager' in inspect.signature(build).parameters:
        network = build(plans, configuration, inputs, outputs, enable_deep_supervision=False)
    else:
        network = build(configuration.network_arch_class_name, configuration.network_arch_init_kwargs,
                        configuration.network_arch_init_kwargs_req_import, inputs, outputs,
                        enable_deep_supervision=False)
    first = load_network(network, payload)
    predictor.manual_initialization(network, plans, configuration, [payload['network_weights']],
                                    dataset, payload['trainer_name'],
                                    payload.get('inference_allowed_mirroring_axes'))
    return first


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['electrical', 'fitted', 'ideal'], required=True)
    parser.add_argument('--operator', choices=[LEGACY_OPERATOR, PIXEL_OPERATOR],
                        default=LEGACY_OPERATOR,
                        help='First-layer protocol; legacy keeps the old U17 block chain')
    parser.add_argument('--model-folder', type=Path, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--psf-dir', type=Path)
    parser.add_argument('--input', type=Path, required=True, help='nnU-Net input directory, or one _0000.nii.gz volume')
    parser.add_argument('--output', type=Path, required=True, help='Fresh independent output directory')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no-tta', action='store_true')
    parser.add_argument('--label', default='NON_FORMAL')
    args = parser.parse_args()
    if args.mode in ('fitted', 'ideal') and args.psf_dir is None:
        parser.error('--psf-dir is required for fitted/ideal mode')
    if Path(args.checkpoint).name != args.checkpoint or args.fold < 0:
        parser.error('checkpoint must be a basename; fold must be nonnegative')
    source = args.input.resolve(strict=True)
    output = args.output.resolve()
    for protected in (source if source.is_dir() else source.parent, args.model_folder.resolve(), args.psf_dir.resolve() if args.psf_dir else args.model_folder.resolve()):
        if output == protected or output in protected.parents or protected in output.parents:
            parser.error('Output must be separate from input/model/PSF sources')
    if source.is_file() and not source.name.endswith('_0000.nii.gz'):
        parser.error('Single volume must have nnU-Net _0000.nii.gz suffix')
    output.mkdir(parents=True, exist_ok=False)
    record = dict(vars(args), status='FAILED', compile=False, tile_step_size=0.5,
                  use_gaussian=True, use_mirroring=not args.no_tta,
                  preprocess_processes=1, export_processes=1)
    # Electrical keeps the real convolution. A hooked PSF replacement records the
    # operator provenance of the artifact that was actually loaded, i.e. its own
    # validated canvas size rather than an assumed one.
    if args.mode == 'electrical':
        record['operator_id'] = 'electrical'
        record['operator'] = 'electrical: checkpoint first-layer Conv2d'
    hook = None
    try:
        os.environ['nnUNet_compile'] = 'false'
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        checkpoint = args.model_folder/f'fold_{args.fold}'/args.checkpoint
        record['checkpoint_sha256'] = sha256(checkpoint)
        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True,
            use_mirroring=not args.no_tta, perform_everything_on_device=args.device.startswith('cuda'),
            device=torch.device(args.device), verbose=False, verbose_preprocessing=False, allow_tqdm=False)
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        optical = payload.get('optical_finetune')
        if optical and (args.mode != 'fitted' or args.operator != PIXEL_OPERATOR):
            raise ValueError('Frozen-PSF checkpoint requires fitted pixel PSF inference')
        first = initialize_predictor(predictor, args, payload)
        record['first_kernel_size'] = first.kernel_size[0]
        predictor.network.eval().requires_grad_(False)
        if args.mode in ('fitted', 'ideal'):
            if payload.get('first_layer') is not None:
                checkpoint_origin = optical['source_checkpoint']['sha256'] if optical else record['checkpoint_sha256']
                check_artifact_origin(args.psf_dir, checkpoint_origin)
            weight = first.weight
            kernel, hashes = load_psfs(args.psf_dir, weight, operator=args.operator,
                                       ideal=args.mode == 'ideal')
            if optical and (hashes != optical['psf_sha256'] or
                            not torch.equal(kernel, payload['optical_effective_kernel'])):
                raise ValueError('Frozen optical state differs from checkpoint')
            n = psf_canvas(args.psf_dir)[0]
            record.update(operator_record(args.operator, n))
            record['psf_canvas'] = [n, n]
            record['psf_sha256'] = hashes
            record['psf_variant'] = ('IDEAL_REPLICATED_TARGET_UNIT_GAIN' if args.mode == 'ideal'
                                     else 'FITTED_RAW_GAIN')
            record['first_weight_sha256'] = __import__('hashlib').sha256(weight.detach().cpu().float().numpy().tobytes()).hexdigest()
            hook = FirstConvHook(predictor.network, kernel)
        inputs = str(source) if source.is_dir() else [[str(source)]]
        outputs = str(output) if source.is_dir() else [str(output/source.name[:-12])]
        predictor.predict_from_files(inputs, outputs, save_probabilities=False, overwrite=False,
            num_processes_preprocessing=1, num_processes_segmentation_export=1)
        if hook is not None and hook.calls == 0:
            raise RuntimeError('First-convolution hook did not execute')
        record['status'] = 'COMPLETED'
    except Exception as error:
        record['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if hook is not None:
            record['hook_calls'] = hook.calls
            hook.remove()
        (output/'psf_inference_record.json').write_text(json.dumps(record, indent=2, default=str), encoding='utf-8')


if __name__ == '__main__':
    main()
