"""C changes actual Z sampling; official crop/DA and mixed loss remain unchanged."""
import os
from pathlib import Path
import numpy as np
import torch
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from lesiongen.segmentation512 import AugmentedDataset
from quality import need, sha
from curriculum import case_current, probabilities, validate_policy


class CurriculumDataset(AugmentedDataset):
    def __init__(self, original, pool, pool_pin, sidecar):
        super().__init__(original, pool, pool_pin)
        self.sidecar = sidecar
        need(sidecar['pool_manifest_sha256'] == pool_pin and
             set(sidecar['cases']) == set(self.identifiers), 'C_DATASET_BINDING')
        self.current = None
        self.current_seg = None
        for cid, entry in sidecar['cases'].items():
            folder = Path(self.source_folder)
            need(sha(folder / (cid + '.pkl')) == entry['properties_sha256'] and
                 sha(folder / (cid + '_seg.npy')) == entry['real_seg_sha256'], 'C_REAL_CACHE_CHANGED')
            need(len(entry['synthetic']) == self.manifest['cases'].get(cid, {}).get('count', 0), 'C_SYNTHETIC_COUNT')

    def load_case(self, identifier):
        data, seg, previous, properties = super().load_case(identifier)
        entry = self.sidecar['cases'][identifier]
        n_real = len(entry['real'])
        need(list(data[:, :n_real].shape) == entry['real_shape'], 'C_REAL_SHAPE_CHANGED')
        current = case_current(entry)
        need(len(current['source']) == data.shape[1], 'C_PLANE_METADATA_MISMATCH')
        actual = (seg[0] == 1).sum(axis=(1, 2))
        need(np.array_equal(actual, current['foreground_pixels']), 'C_FOREGROUND_MAPPING_CHANGED')
        self.current, self.current_seg = current, seg
        return data, seg, previous, properties


class CurriculumLoader(nnUNetDataLoader):
    def __init__(self, *args, policy, gains, **kwargs):
        need(int(os.environ.get('nnUNet_n_proc_DA', '0')) == 0, 'C_WORKERS_ZERO_REQUIRED')
        validate_policy(policy)
        self.policy, self.gains, self.epoch = dict(policy), list(gains), 0
        super().__init__(*args, **kwargs)
        need(self.patch_size_was_2d and not self.has_ignore, 'C_BINARY_2D_NO_IGNORE_REQUIRED')

    def set_epoch(self, epoch):
        from curriculum import phase
        phase(epoch, self.policy)
        self.epoch = epoch

    def get_bbox(self, data_shape, force_fg, class_locations, overwrite_class=None, verbose=False):
        need(getattr(self, 'thread_id', 0) == 0, 'C_BACKGROUND_WORKER_UNSUPPORTED')
        current = self._data.current
        p, state = probabilities(current, self.epoch, self.policy, self.gains, force_fg)
        z = int(np.random.choice(len(p), p=p))
        # Rebase actual chosen-plane foreground coordinates to a depth-one volume.
        # Upstream then chooses its ordinary XY bbox on that exact plane.
        coords = np.argwhere(self._data.current_seg[0, z] == 1)
        locations = np.column_stack((np.zeros((len(coords), 2), dtype=np.int64), coords))
        lower, upper = super().get_bbox((1, *data_shape[1:]), force_fg, {1: locations},
                                        overwrite_class=overwrite_class, verbose=verbose)
        need(lower[0] == 0 and upper[0] == 1, 'C_2D_DEPTH_BBOX')
        lower[0], upper[0] = z, z + 1
        joint = np.zeros((2, 6), dtype=float)
        np.add.at(joint, (current['source'], current['group']), p)
        self._c_metadata.append(dict(source=int(current['source'][z]), group=int(current['group'][z]), z=z,
                                     joint=joint.tolist(), fallback=state['fallback'],
                                     real_foreground_unavailable=state['real_foreground_unavailable']))
        return lower, upper

    def generate_train_batch(self):
        self._c_metadata = []
        batch = super().generate_train_batch()
        need(len(self._c_metadata) == len(batch['data']), 'C_BATCH_METADATA_LENGTH')
        for key in ('source', 'group', 'z'):
            batch['c_' + key] = torch.tensor([r[key] for r in self._c_metadata], dtype=torch.int64)
        batch['c_sampling_probabilities'] = torch.tensor([r['joint'] for r in self._c_metadata], dtype=torch.float64)
        batch['c_fallback'] = [r['fallback'] for r in self._c_metadata]
        batch['c_real_foreground_unavailable'] = [r['real_foreground_unavailable'] for r in self._c_metadata]
        batch['c_epoch'] = self.epoch
        return batch
