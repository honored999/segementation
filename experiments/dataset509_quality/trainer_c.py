"""Independent C trainer: sampling curriculum with inherited original Dice+TopK10."""
import json
import os
from pathlib import Path
import numpy as np
import torch
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from trainer514 import nnUNetTrainerLesionGen514TopK10EarlyStopping as Baseline
from data_c import CurriculumDataset, CurriculumLoader
from curriculum import GROUPS, digest, validate_policy, phase
from quality import need, read, sha


class nnUNetTrainerDataset509Curriculum(Baseline):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        self.c_config = read(os.environ['DATASET509_QUALITY_CONFIG'])
        need(self.c_config['mode'] == 'c' and self.c_config['workers'] == 0 and get_allowed_n_proc_DA() == 0,
             'C_ONLY_WORKERS_ZERO_REQUIRED')
        need(not torch.distributed.is_initialized() and int(os.environ.get('WORLD_SIZE', '1')) == 1, 'C_DDP_UNSUPPORTED')
        super().__init__(plans, configuration, fold, dataset_json, device)
        need(not self.is_ddp and not self.is_cascaded and not self.label_manager.has_regions and
             dataset_json['channel_names'] == {'0': 'DWI'} and dataset_json['labels'] == {'background': 0, 'lesion': 1},
             'C_BINARY_SINGLE_GPU_2D_REQUIRED')
        need(not self._do_i_compile(), 'C_COMPILE_UNSUPPORTED')
        reference = self.c_config['curriculum_sidecar']
        need(sha(reference['path']) == reference['sha256'], 'C_SIDECAR_CHANGED')
        self.c_sidecar = read(reference['path'])
        self.policy = self.c_config['curriculum']
        validate_policy(self.policy)
        need(self.c_sidecar['schema'] == 'dataset509-curriculum/v1' and
             digest(self.policy) == self.c_sidecar['policy_sha256'] == digest(self.c_sidecar['policy']), 'C_POLICY_PIN')
        need(self.c_sidecar['pool_manifest_sha256'] == self.pool_pin and
             self.c_sidecar['split_sha256'] == sha(Path(self.preprocessed_dataset_folder_base) / 'splits_final.json'),
             'C_POOL_SPLIT_PIN')
        self.my_init_kwargs['curriculum_identity'] = dict(sidecar_sha256=reference['sha256'], policy_sha256=digest(self.policy))
        self.exposure = dict(source_group_counts=np.zeros((2, 6), dtype=np.int64).tolist(), batches=0,
                             fallback_samples={}, real_foreground_unavailable=0)
        self.c_epoch_counts = np.zeros((2, 6), dtype=np.int64)
        self.c_probability_sum = np.zeros((2, 6), dtype=float)
        self.c_epoch_samples = 0

    # _build_loss and validation_step deliberately inherited unchanged from Baseline.
    def get_tr_and_val_datasets(self):
        augmented, validation = super().get_tr_and_val_datasets()
        return CurriculumDataset(augmented.original, self.pool, self.pool_pin, self.c_sidecar), validation

    def get_dataloaders(self):
        need(get_allowed_n_proc_DA() == 0, 'C_NO_PREFETCH_WORKERS')
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        patch = self.configuration_manager.patch_size
        scales = self._get_deep_supervision_scales()
        rotation, dummy, initial, axes = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        need(len(patch) == 2 and not dummy, 'C_ONLY_2D')
        common = dict(is_cascaded=False, foreground_labels=self.label_manager.foreground_labels,
                      regions=None, ignore_label=None)
        tr_transform = self.get_training_transforms(patch, rotation, scales, axes, False,
                           use_mask_for_norm=self.configuration_manager.use_mask_for_norm, **common)
        va_transform = self.get_validation_transforms(scales, **common)
        train, validation = self.get_tr_and_val_datasets()
        options = dict(oversample_foreground_percent=self.oversample_foreground_percent,
                       sampling_probabilities=None, pad_sides=None,
                       probabilistic_oversampling=self.probabilistic_oversampling)
        dl_tr = CurriculumLoader(train, self.batch_size, initial, patch, self.label_manager,
                                transforms=tr_transform, policy=self.policy, gains=self.c_sidecar['gains'], **options)
        dl_tr.set_epoch(int(self.current_epoch))
        dl_va = nnUNetDataLoader(validation, self.batch_size, patch, patch, self.label_manager,
                                transforms=va_transform, **options)
        tr, va = SingleThreadedAugmenter(dl_tr, None), SingleThreadedAugmenter(dl_va, None)
        next(tr)
        next(va)
        return tr, va

    def on_train_epoch_start(self):
        # No optimizer/LR/epoch reset: use the ordinary nnU-Net hook once.
        super().on_train_epoch_start()
        self.dataloader_train.data_loader.set_epoch(int(self.current_epoch))
        self.c_epoch_counts.fill(0)
        self.c_probability_sum.fill(0)
        self.c_epoch_samples = 0

    def train_step(self, batch):
        need(batch.get('c_epoch') == int(self.current_epoch), 'C_STALE_PREFETCH_EPOCH')
        source = torch.as_tensor(batch['c_source']).cpu().numpy()
        groups = torch.as_tensor(batch['c_group']).cpu().numpy()
        probabilities = torch.as_tensor(batch['c_sampling_probabilities']).cpu().numpy()
        need(source.shape == groups.shape == (len(batch['data']),) and probabilities.shape == (len(source), 2, 6)
             and ((source == 0) | (source == 1)).all() and ((groups >= 0) & (groups < 6)).all()
             and np.isfinite(probabilities).all() and (probabilities >= 0).all()
             and np.allclose(probabilities.sum((1, 2)), 1), 'C_INVALID_BATCH_METADATA')
        result = super().train_step(batch)  # original mixed Dice+TopK10 and ordinary optimizer step
        counts = np.zeros((2, 6), dtype=np.int64)
        np.add.at(counts, (source, groups), 1)
        self.exposure['source_group_counts'] = (np.array(self.exposure['source_group_counts']) + counts).tolist()
        self.exposure['batches'] += 1
        for fallback in batch['c_fallback']:
            self.exposure['fallback_samples'][fallback] = self.exposure['fallback_samples'].get(fallback, 0) + 1
        self.exposure['real_foreground_unavailable'] += int(sum(batch['c_real_foreground_unavailable']))
        self.c_epoch_counts += counts
        self.c_probability_sum += probabilities.sum(0)
        self.c_epoch_samples += len(source)
        return result

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        report = dict(epoch=int(self.current_epoch), alpha=phase(int(self.current_epoch), self.policy),
                      group_names=GROUPS, gain_cap=self.policy['gain_cap'], gains=self.c_sidecar['gains'],
                      realized_source_group_counts=self.c_epoch_counts.tolist(), cumulative=self.exposure,
                      mean_conditional_source_group_probabilities=(self.c_probability_sum / max(self.c_epoch_samples, 1)).tolist())
        try:
            with (Path(self.output_folder) / 'curriculum_exposure.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(report, allow_nan=False) + '\n')
        except OSError:
            self.print_to_log_file('C_EXPOSURE_LOG_WRITE_FAILED')

    def load_checkpoint(self, filename_or_checkpoint):
        expected = self.my_init_kwargs['curriculum_identity']
        super().load_checkpoint(filename_or_checkpoint)
        need(self.my_init_kwargs.get('curriculum_identity') == expected, 'C_CHECKPOINT_IDENTITY_CHANGED')
