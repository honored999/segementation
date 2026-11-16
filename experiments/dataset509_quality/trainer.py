"""Thin Dataset514 Q adapter. Baseline construction bypasses this class entirely."""
import json
import os
from pathlib import Path
import numpy as np
import torch
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from trainer514 import nnUNetTrainerLesionGen514TopK10EarlyStopping as Baseline
from data import QualityDataset, SourceLoader
from loss import SourceLoss
from quality import need, read, sha, digest, summary


class nnUNetTrainerDataset509Quality(Baseline):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        self.quality_config = read(os.environ['DATASET509_QUALITY_CONFIG'])
        need(self.quality_config['mode'] in ('q', 'quality_filter'), 'BASELINE_MUST_USE_ORIGINAL_TRAINER')
        need(not (torch.distributed.is_initialized() or int(__import__('os').environ.get('WORLD_SIZE', '1')) != 1),
             'DDP_UNSUPPORTED')
        super().__init__(plans, configuration, fold, dataset_json, device)
        need(not self.is_ddp and not self.is_cascaded and not self.label_manager.has_regions,
             'ONLY_SINGLE_GPU_BINARY_2D_SUPPORTED')
        need(dataset_json['channel_names'] == {'0': 'DWI'} and
             dataset_json['labels'] == {'background': 0, 'lesion': 1}, 'DATASET_PROTOCOL_UNSUPPORTED')
        need(not self._do_i_compile(), 'Q_REQUIRES_nnUNet_compile_FALSE')
        self.sidecar = read(self.quality_config['sidecar']['path'])
        need(sha(self.quality_config['sidecar']['path']) == self.quality_config['sidecar']['sha256'], 'SIDECAR_CHANGED')
        need(self.sidecar['scoring_sha256'] == digest(self.sidecar['scoring']), 'SCORING_CHANGED')
        need(self.sidecar['pool_manifest_sha256'] == self.pool_pin, 'SIDECAR_POOL_CHANGED')
        need(self.sidecar['split_sha256'] == sha(Path(self.preprocessed_dataset_folder_base) / 'splits_final.json'),
             'SIDECAR_SPLIT_CHANGED')
        self.my_init_kwargs['quality_identity'] = dict(mode=self.quality_config['mode'],
                                                      sidecar_sha256=self.quality_config['sidecar']['sha256'],
                                                      synthetic_coefficient=self.quality_config['synthetic_coefficient'],
                                                      filter_threshold=self.quality_config['filter_threshold'])
        self.exposure = {'real': 0, 'synthetic': 0, 'positive_weight_synthetic': 0,
                         'synthetic_weight_sum': 0., 'synthetic_weight_bins': [0] * 10,
                         'real_lesion_size_bins': [0] * 6, 'synthetic_lesion_size_bins': [0] * 6}

    def _build_loss(self):
        original = super()._build_loss()
        if self.quality_config['mode'] == 'quality_filter':
            return original
        weights = None
        if self.enable_deep_supervision:
            weights = np.array([1 / 2**i for i in range(len(self._get_deep_supervision_scales()))])
            need(len(weights) > 1, 'DEEP_SUPERVISION_NEEDS_MULTIPLE_LEVELS')
            weights[-1] = 0
            weights = (weights / weights.sum()).tolist()
        return SourceLoss(original, weights, self.quality_config['synthetic_coefficient'], self.label_manager.ignore_label)

    def get_tr_and_val_datasets(self):
        augmented, validation = super().get_tr_and_val_datasets()
        return QualityDataset(augmented.original, self.pool, self.pool_pin, self.sidecar,
                              self.quality_config['mode'], self.quality_config['filter_threshold']), validation

    def get_dataloaders(self):
        # nnU-Net 2.8.1 construction, differing only in training loader class.
        # No module-global replacement; safe across workers and other trainers.
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        patch = self.configuration_manager.patch_size
        scales = self._get_deep_supervision_scales()
        rotation, dummy, initial, axes = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        need(len(patch) == 2 and not dummy, 'ONLY_2D_SUPPORTED')
        common = dict(is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
                      regions=None, ignore_label=self.label_manager.ignore_label)
        training_transform = self.get_training_transforms(
            patch, rotation, scales, axes, dummy,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm, **common)
        validation_transform = self.get_validation_transforms(scales, **common)
        train, val = self.get_tr_and_val_datasets()
        loader_options = dict(oversample_foreground_percent=self.oversample_foreground_percent,
                              sampling_probabilities=None, pad_sides=None,
                              probabilistic_oversampling=self.probabilistic_oversampling)
        dl_tr = SourceLoader(train, self.batch_size, initial, patch, self.label_manager,
                             transforms=training_transform, **loader_options)
        dl_val = nnUNetDataLoader(val, self.batch_size, patch, patch, self.label_manager,
                                 transforms=validation_transform, **loader_options)
        workers = get_allowed_n_proc_DA()
        if workers == 0:
            tr = SingleThreadedAugmenter(dl_tr, None)
            va = SingleThreadedAugmenter(dl_val, None)
        else:
            tr = NonDetMultiThreadedAugmenter(dl_tr, None, num_processes=workers,
                                              num_cached=max(6, workers // 2), seeds=None,
                                              pin_memory=self.device.type == 'cuda', wait_time=.002)
            va = NonDetMultiThreadedAugmenter(dl_val, None, num_processes=max(1, workers // 2),
                                              num_cached=max(3, workers // 4), seeds=None,
                                              pin_memory=self.device.type == 'cuda', wait_time=.002)
        next(tr)
        next(va)
        return tr, va

    def train_step(self, batch):
        need(all(k in batch for k in ('q_source', 'q_quality', 'q_lesion_pixels')), 'SOURCE_METADATA_MISSING')
        source = torch.as_tensor(batch['q_source'])
        weights = torch.as_tensor(batch['q_quality'])
        sizes = torch.as_tensor(batch['q_lesion_pixels'])
        need(source.shape == weights.shape == sizes.shape == (len(batch['data']),), 'SOURCE_BATCH_MISMATCH')
        need(((source == 0) | (source == 1)).all().item() and torch.isfinite(weights).all().item()
             and ((weights >= 0) & (weights <= 1)).all().item()
             and (weights[source == 0] == 1).all().item(), 'INVALID_BATCH_METADATA')
        if isinstance(self.loss, SourceLoss):
            self.loss.metadata = (source, weights)
            self.loss.training_context = True
        try:
            result = super().train_step(batch)
        finally:
            if isinstance(self.loss, SourceLoss):
                self.loss.metadata = None
                self.loss.training_context = False
        # Observation only, after successful optimizer update; no RNG or model reads.
        s, w, z = source.cpu().numpy(), weights.cpu().numpy(), sizes.cpu().numpy()
        for kind, name in ((0, 'real'), (1, 'synthetic')):
            selected = s == kind
            self.exposure[name] += int(selected.sum())
            bins = np.histogram(z[selected], bins=[0, 1, 16, 64, 256, 1024, np.inf])[0]
            key = name + '_lesion_size_bins'
            self.exposure[key] = (np.array(self.exposure[key]) + bins).tolist()
        syn = w[s == 1]
        self.exposure['positive_weight_synthetic'] += int((syn > 0).sum())
        self.exposure['synthetic_weight_sum'] += float(syn.sum())
        hist = np.histogram(syn, bins=np.linspace(0, 1, 11))[0]
        self.exposure['synthetic_weight_bins'] = (np.array(self.exposure['synthetic_weight_bins']) + hist).tolist()
        return result

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        # Logging failure does not change training results or checkpoint selection.
        try:
            with (Path(self.output_folder) / 'quality_exposure.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(dict(epoch=self.current_epoch, cumulative=self.exposure), allow_nan=False) + '\n')
        except OSError:
            self.print_to_log_file('QUALITY_EXPOSURE_LOG_WRITE_FAILED')

    def load_checkpoint(self, filename_or_checkpoint):
        expected = self.my_init_kwargs['quality_identity']
        super().load_checkpoint(filename_or_checkpoint)
        need(self.my_init_kwargs.get('quality_identity') == expected, 'CHECKPOINT_QUALITY_IDENTITY_CHANGED')


def trainer_class(mode='baseline'):
    if mode == 'c':
        from trainer_c import nnUNetTrainerDataset509Curriculum
        return nnUNetTrainerDataset509Curriculum
    need(mode in ('baseline', 'q', 'quality_filter'), 'INVALID_MODE')
    return Baseline if mode == 'baseline' else nnUNetTrainerDataset509Quality
