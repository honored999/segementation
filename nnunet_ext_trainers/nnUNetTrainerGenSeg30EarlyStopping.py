"""Audited train-only slices plus strict boundaries around inherited nnU-Net IO."""
from pathlib import Path
import importlib
import os
import re

import numpy as np
import torch

from nnUNetTrainerEarlyStopping import nnUNetTrainerEarlyStopping
from nnunetv2 import paths as nn_paths
from experiments.genseg30.contract import Source, check_output, checked_file, read_json, require, write_json
from experiments.genseg30.dataset import SlicePoolDataset

NN_MODULE = importlib.import_module('nnunetv2.training.nnUNetTrainer.nnUNetTrainer')
PROVENANCE_KEY = 'genseg30_pool_identity'


def _nn_path(module, name):
    value = getattr(module, name)
    require(value is not None, name + ' must be configured')
    try:
        return Path(os.fspath(value)).expanduser().resolve()
    except (RuntimeError, TypeError) as exc:
        raise ValueError(name + ' must be configured') from exc


def _extra_protected(source, pool=None, receipt=None):
    protected = source.protected + [_nn_path(NN_MODULE, 'nnUNet_preprocessed'),
                                    _nn_path(nn_paths, 'nnUNet_raw')]
    if pool:
        pool = check_output(pool, protected, fresh=False)
        protected.append(pool)
        if (pool / 'manifest.json').is_file():
            record = read_json(checked_file(pool, 'manifest.json'))
            checkpoint = Path(record['generator_checkpoint']).expanduser().resolve()
            protected.append(check_output(checkpoint.parent, protected, fresh=False))
    if receipt:
        protected += [Path(receipt['pool']), Path(receipt['generator_checkpoint']).parent]
    return protected


def _safe_checkpoint(path):
    # Restrict deserialization to tensors/primitives and ordinary NumPy numeric metadata.
    # This retains the original logger/optimizer payload without arbitrary pickle execution.
    core = getattr(np, '_core', np.core).multiarray
    allowed = [core.scalar, core._reconstruct, np.ndarray, np.dtype]
    allowed += list({type(np.dtype(code)) for code in ('f4', 'f8', 'i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8', '?')})
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location='cpu', weights_only=True)


class nnUNetTrainerGenSeg30EarlyStopping(nnUNetTrainerEarlyStopping):
    # Keep EXACT parent argument names: installed __init__ introspects this signature.
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        require(type(fold) is int and fold == 0 and configuration == '2d', 'GenSeg30 only supports 2d fold0')
        clean_plans = {k: v for k, v in plans.items() if k != 'continue_training'}
        require(clean_plans.get('dataset_name') == 'Dataset501_StrokeLesion' and
                clean_plans.get('plans_name') == 'nnUNetPlans', 'Original Dataset501 nnUNetPlans required')
        config = clean_plans.get('configurations', {}).get(configuration, {})
        require(config.get('previous_stage') is None, 'Cascades unsupported')
        source_folder = _nn_path(NN_MODULE, 'nnUNet_preprocessed') / clean_plans['dataset_name'] / config['data_identifier']
        source = Source(source_folder, _nn_path(nn_paths, 'nnUNet_raw'), configuration, fold)
        require(clean_plans == source.plans and dataset_json == source.dataset_json,
                'Actual constructor plans/dataset differs from original source')
        source.require_no_unpack_writes()
        output_base = _nn_path(NN_MODULE, 'nnUNet_results') / clean_plans['dataset_name'] / (
            self.__class__.__name__ + '__' + clean_plans['plans_name'] + '__' + configuration)
        output_folder = output_base / 'fold_0'
        pool = os.environ.get('GENSEG30_POOL')
        # Validate BOTH actual destinations before inherited constructor creates dirs/log files.
        initial_protected = _extra_protected(source, pool)
        check_output(output_base, initial_protected, fresh=False)
        check_output(output_folder, initial_protected, fresh=False)
        receipt_path = output_folder / 'genseg30_pool_identity.json'
        receipt = read_json(receipt_path) if receipt_path.is_file() else None
        protected = _extra_protected(source, pool, receipt)
        check_output(output_base, protected, fresh=False)
        check_output(output_folder, protected, fresh=False)
        self._genseg30_continue_requested = bool(plans.get('continue_training', False))
        self._genseg30_checkpoint_loaded = False
        self._genseg30_constructor_source = source.identity
        if self._genseg30_continue_requested:
            require(receipt is not None, 'Continuation requires an EXISTING result receipt; never establish a new pool')
            require(any(output_folder.glob('checkpoint_*.pth')), 'Continuation requires an existing checkpoint')
        super().__init__(plans, configuration, fold, dataset_json, device)

    def _strict_source(self):
        require(type(self.fold) is int and self.fold == 0 and self.configuration_name == '2d',
                'GenSeg30 only supports Dataset501 2d fold0')
        require(not self.is_cascaded and self.folder_with_segs_from_previous_stage is None, 'Cascades unsupported')
        source = Source(self.preprocessed_dataset_folder, _nn_path(nn_paths, 'nnUNet_raw'), '2d', 0)
        expected_folder = _nn_path(NN_MODULE, 'nnUNet_preprocessed') / source.base.name / source.configuration['data_identifier']
        require(expected_folder.resolve() == source.folder, 'Source differs from actual configured nnUNet_preprocessed')
        require(self.dataset_json == source.dataset_json and self.plans_manager.plans == source.plans,
                'Actual Trainer source/plans/dataset identity mismatch')
        require(self.dataset_class in (None, source.dataset_class), 'Original dataset class must remain unchanged')
        if self.dataset_class is None:
            self.dataset_class = source.dataset_class
        if hasattr(self, '_genseg30_constructor_source'):
            source.verify_identity(self._genseg30_constructor_source)
        return source

    def _guard_output(self, source, pool=None, receipt=None):
        protected = _extra_protected(source, pool, receipt)
        base = getattr(self, 'output_folder_base', str(Path(self.output_folder).parent))
        check_output(base, protected, fresh=False)
        return check_output(self.output_folder, protected, fresh=False)

    def _existing_receipt(self, source):
        output = self._guard_output(source)
        require((output / 'genseg30_pool_identity.json').is_file(),
                'Checkpoint/continuation requires an EXISTING result receipt; no new pool binding')
        receipt = read_json(checked_file(output, 'genseg30_pool_identity.json'))
        require(receipt.get('format') == 1 and receipt.get('trainer_name') == self.__class__.__name__,
                'Result receipt Trainer identity mismatch')
        source.verify_identity(receipt['source'])
        require(re.fullmatch(r'[0-9a-f]{64}', receipt.get('manifest_sha256', '')), 'Invalid receipt pool hash')
        self._guard_output(source, receipt=receipt)
        pin = os.environ.get('GENSEG30_MANIFEST_SHA256')
        pool = os.environ.get('GENSEG30_POOL')
        if pin:
            require(pin.lower() == receipt['manifest_sha256'], 'Checkpoint pool hash differs from active trial')
        if pool:
            require(str(Path(pool).expanduser().resolve()) == receipt['pool'], 'Checkpoint pool differs from active trial')
        return receipt

    def do_split(self):
        source = self._strict_source()
        # Never enter inherited split generation/fallback. Preserve exact server fold0 ordering.
        split = read_json(source.splits)[0]
        return split['train'].copy(), split['val'].copy()

    def perform_actual_validation(self, save_probabilities=False):
        source = self._strict_source()
        self._guard_output(source)
        self.do_split()
        return super().perform_actual_validation(save_probabilities)

    def load_checkpoint(self, filename_or_checkpoint):
        source = self._strict_source()
        receipt = self._existing_receipt(source)  # required BEFORE deserialization/parent initialization
        if isinstance(filename_or_checkpoint, (str, Path)):
            path = Path(filename_or_checkpoint).expanduser().resolve()
            require(path.parent == Path(self.output_folder).resolve() and path.name.startswith('checkpoint_') and
                    path.suffix == '.pth', 'Only this result folder checkpoint may be resumed/validated')
            checkpoint = _safe_checkpoint(path)
        else:
            checkpoint = filename_or_checkpoint
        require(isinstance(checkpoint, dict) and checkpoint.get('trainer_name') == self.__class__.__name__,
                'Checkpoint Trainer identity mismatch')
        args = checkpoint.get('init_args', {})
        require(args.get(PROVENANCE_KEY) == receipt, 'Checkpoint lacks/mismatches the original pool receipt')
        require(type(args.get('fold')) is int and args.get('fold') == 0 and args.get('configuration') == '2d' and
                args.get('dataset_json') == source.dataset_json and
                {k: v for k, v in args.get('plans', {}).items() if k != 'continue_training'} == source.plans,
                'Checkpoint original source/configuration identity mismatch')
        super().load_checkpoint(checkpoint)  # original network/optimizer/EMA/early-stopping restoration
        self._genseg30_checkpoint_loaded = True

    def get_tr_and_val_datasets(self):
        source = self._strict_source()
        # Inherited on_train_start unconditionally calls ORIGINAL unpack_dataset; Blosc2 is no-op.
        source.require_no_unpack_writes()
        pool = os.environ.get('GENSEG30_POOL')
        pin = os.environ.get('GENSEG30_MANIFEST_SHA256', '').lower()
        require(pool and re.fullmatch(r'[0-9a-f]{64}', pin), 'GENSEG30_POOL and external manifest SHA256 pin required')
        output = self._guard_output(source, pool=pool)
        loaded = getattr(self, '_genseg30_checkpoint_loaded', False)
        requested = getattr(self, '_genseg30_continue_requested', False)
        if loaded or requested:
            self._existing_receipt(source)  # refuse deleted/copied identity before pool/source content audit
            require(loaded, 'Continuation must load its existing checkpoint before training')
        else:
            require(not any(output.glob('checkpoint_*.pth')) and getattr(self, 'current_epoch', 0) == 0,
                    'Existing checkpoint/run must load its original receipt before training')
        wrapped = SlicePoolDataset(source.training_dataset(), source, pool, pin)
        manifest = wrapped.manifest
        output = self._guard_output(source, pool=pool)
        receipt = {'format': 1, 'trainer_name': self.__class__.__name__, 'manifest_sha256': pin,
                   'pool': str(Path(pool).resolve()), 'generator_checkpoint': manifest['generator_checkpoint'],
                   'generator_sha256': manifest['generator_sha256'], 'source': source.identity,
                   'synthetic_slices': manifest['synthetic_slices']}
        output.mkdir(parents=True, exist_ok=True)
        receipt_path = output / 'genseg30_pool_identity.json'
        continuing = getattr(self, '_genseg30_checkpoint_loaded', False) or getattr(self, '_genseg30_continue_requested', False)
        checkpoints_exist = any(output.glob('checkpoint_*.pth'))
        if continuing:
            require(self._existing_receipt(source) == receipt, 'Continuation original pool identity mismatch')
        else:
            require(not checkpoints_exist and getattr(self, 'current_epoch', 0) == 0,
                    'Existing checkpoint/run must load matching receipt; cannot silently start fresh')
            if receipt_path.exists():
                require(read_json(receipt_path) == receipt, 'Fresh run pool identity mismatch')
            else:
                try:
                    write_json(receipt_path, receipt)
                except FileExistsError:
                    require(read_json(receipt_path) == receipt, 'Concurrent run pool identity mismatch')
        # Parent save_checkpoint already serializes my_init_kwargs. Add provenance without
        # replacing its serialization, early-stopping, best selection or optimizer semantics.
        self.my_init_kwargs[PROVENANCE_KEY] = receipt
        original_train, original_val = super().get_tr_and_val_datasets()
        require(original_train.identifiers == source.train and original_val.identifiers == source.val,
                'Official split differs from approved source split')
        wrapped.original = original_train
        return wrapped, original_val
