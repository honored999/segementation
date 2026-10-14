"""Training-only sampling containers; appended z is never physical geometry."""
from copy import deepcopy
from pathlib import Path

import numpy as np

from .contract import assert_identities, checked_file, require, validate_arrays
from .pipeline import audit_pool, flip_plane


class SlicePoolDataset:
    def __init__(self, original, source, pool, expected_sha256):
        require(set(original.identifiers) == set(source.train), 'Only original training patient IDs allowed')
        require(expected_sha256, 'External manifest hash pin required')
        self.manifest, self._identities = audit_pool(source, pool, require_quality=True,
                                expected_sha256=expected_sha256, return_identities=True)
        self.original, self.source = original, source
        self.pool = Path(pool).resolve()
        self.identifiers = original.identifiers.copy()
        self.source_folder = original.source_folder
        self.folder_with_segs_from_previous_stage = None
        patient_files = {p: source.case_files(p) + [checked_file(self.pool, self.manifest['cases'][p]['file'])] +
                         [checked_file(self.pool, f'{p}_{i:06d}_preview.png')
                          for i in range(self.manifest['cases'][p]['synthetic_slices'])] for p in self.identifiers}
        all_patient_paths = {str(path) for paths in patient_files.values() for path in paths}
        self._global_paths = [path for path in self._identities if path not in all_patient_paths]
        self._patient_paths = {p: [str(path) for path in paths] for p, paths in patient_files.items()}
        assert_identities(self._identities)

    def __getitem__(self, identifier):
        return self.load_case(identifier)

    def load_case(self, identifier):
        require(identifier in self.identifiers, 'Foreign/validation patient cannot enter training wrapper')
        paths = self._global_paths + self._patient_paths[identifier]
        # Full SHA256 content audit occurs at startup. Unchanged batches use only stat identities.
        # On any change, reject rather than silently refreshing the trusted hash or caching all patients.
        assert_identities(self._identities, paths)
        case = self.manifest['cases'][identifier]
        with np.load(checked_file(self.pool, case['file']), allow_pickle=False) as arrays:
            extra, targets = arrays['data'], arrays['seg']
        data, seg, properties = self.source.load_train(self.original, identifier)
        assert_identities(self._identities, paths)
        require(list(data.shape) == case['source_shape'], 'Original source shape changed')
        validate_arrays(extra, targets, allow_empty=True)
        require(extra.shape == (1, case['synthetic_slices'], *data.shape[2:]), 'Native synthetic shape mismatch')
        for i, record in enumerate(case['records']):
            require(np.array_equal(targets[0, i], flip_plane(seg[0, record['source_z']], record)),
                    'Paired synthetic mask does not match recorded training-source transform')
        data, seg = np.concatenate((data, extra), axis=1), np.concatenate((seg, targets), axis=1)
        properties = deepcopy(properties)
        from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
        properties['class_locations'] = DefaultPreprocessor._sample_foreground_locations(seg, [1])
        properties['genseg30_sampling_only'] = True
        assert_identities(self._identities, paths)
        return data, seg, None, properties
