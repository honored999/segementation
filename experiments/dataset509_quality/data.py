"""Capture actual sampled Z at upstream get_bbox; leave cropping and DA upstream."""
import copy
import numpy as np
import torch
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from lesiongen.segmentation512 import AugmentedDataset
from quality import need, read, sha


class QualityDataset(AugmentedDataset):
    def __init__(self, original, pool, pool_pin, sidecar, mode, threshold=.5):
        super().__init__(original, pool, pool_pin)
        self.sidecar = sidecar
        self.mode = mode
        self.threshold = threshold
        need(sidecar['pool_manifest_sha256'] == pool_pin, 'SIDECAR_POOL_MISMATCH')
        need(set(sidecar['cases']) == set(self.manifest['cases']), 'SIDECAR_CASE_MISMATCH')
        for cid, record in self.manifest['cases'].items():
            need(len(sidecar['cases'][cid]) == record['count'], 'SIDECAR_LENGTH_MISMATCH')
        self.current = None

    def load_case(self, identifier):
        data, seg, previous, props = super().load_case(identifier)
        n_extra = self.manifest['cases'].get(identifier, {}).get('count', 0)
        n_real = data.shape[1] - n_extra
        weights = np.asarray([e['weight'] for e in self.sidecar['cases'].get(identifier, [])], dtype=np.float32)
        sizes = np.asarray([e['lesion_pixels'] for e in self.sidecar['cases'].get(identifier, [])], dtype=np.int64)
        need(len(weights) == n_extra and np.isfinite(weights).all() and
             ((weights >= 0) & (weights <= 1)).all(), 'INVALID_SIDECAR_WEIGHTS')
        # Record native full-plane lesion size for synthetic exposure; real is current full-plane size.
        real_sizes = (seg[0, :n_real] == 1).sum(axis=(1, 2))
        if self.mode == 'quality_filter' and n_extra:
            keep = np.flatnonzero(weights >= self.threshold)
            indices = np.concatenate((np.arange(n_real), n_real + keep))
            data, seg = data[:, indices], seg[:, indices]
            sizes = sizes[keep]
            weights = np.ones(len(keep), dtype=np.float32)
            props = copy.deepcopy(props)
            from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
            props['class_locations'] = DefaultPreprocessor._sample_foreground_locations(seg, [1])
        self.current = dict(source=np.r_[np.zeros(n_real, dtype=np.int8), np.ones(len(weights), dtype=np.int8)],
                            quality=np.r_[np.ones(n_real, dtype=np.float32), weights],
                            lesion_pixels=np.r_[real_sizes, sizes])
        need(len(self.current['source']) == data.shape[1], 'SOURCE_DEPTH_MISMATCH')
        return data, seg, previous, props


class SourceLoader(nnUNetDataLoader):
    def get_bbox(self, data_shape, *args, **kwargs):
        lower, upper = super().get_bbox(data_shape, *args, **kwargs)
        need(self.patch_size_was_2d and self.patch_size[0] == 1, 'ONLY_2D_SUPPORTED')
        z = int(lower[0])
        current = self._data.current
        need(current is not None and 0 <= z < len(current['source']) and upper[0] == z + 1,
             'SAMPLED_PLANE_UNKNOWN')
        self._sample_metadata.append([current[k][z] for k in ('source', 'quality', 'lesion_pixels')])
        return lower, upper

    def generate_train_batch(self):
        self._sample_metadata = []
        batch = super().generate_train_batch()
        need(len(self._sample_metadata) == len(batch['data']), 'SOURCE_BATCH_MISMATCH')
        batch['q_source'] = torch.tensor([r[0] for r in self._sample_metadata], dtype=torch.int64)
        batch['q_quality'] = torch.tensor([r[1] for r in self._sample_metadata], dtype=torch.float32)
        batch['q_lesion_pixels'] = torch.tensor([r[2] for r in self._sample_metadata], dtype=torch.int64)
        return batch
