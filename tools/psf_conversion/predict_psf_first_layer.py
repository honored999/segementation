"""Official nnUNetPredictor with optional frozen fitted-PSF first convolution."""
import argparse
import json
import os
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.psf_conversion.psf_first_conv import FirstConvHook, KEY, load_psfs, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['electrical', 'fitted'], required=True)
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
    if args.mode == 'fitted' and args.psf_dir is None:
        parser.error('--psf-dir is required for fitted mode')
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
                  operator='U17 -> physical C(origin=n//2) -> mean R17 stride17 -> s/289 + bias',
                  preprocess_processes=1, export_processes=1)
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
        predictor.initialize_from_trained_model_folder(str(args.model_folder), use_folds=(args.fold,), checkpoint_name=args.checkpoint)
        predictor.network.eval().requires_grad_(False)
        if args.mode == 'fitted':
            weight = predictor.network.get_submodule(KEY).weight
            kernel, hashes = load_psfs(args.psf_dir, weight)
            record['psf_sha256'] = hashes
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
