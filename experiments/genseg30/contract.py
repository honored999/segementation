"""Fail-closed Dataset501 fold0 provenance and native-array contracts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

REPO = Path(__file__).resolve().parents[2]
REFERENCE = REPO / 'standalone_nnunet2d/reference/splits_final.json'
REFERENCE_SHA256 = '02a84d777e4eb9f481c3a9d90529c1673c828e14f2699fd0e898596ce21574ed'
CLASSIFICATION = 'exploratory_dataset501_fold0_offline_conditional_gan'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _windows_change_time(path):
    # Windows stat ctime is creation time. Query NTFS metadata ChangeTime explicitly.
    import ctypes
    from ctypes import wintypes
    class BasicInfo(ctypes.Structure):
        _fields_ = [('CreationTime', ctypes.c_longlong), ('LastAccessTime', ctypes.c_longlong),
                    ('LastWriteTime', ctypes.c_longlong), ('ChangeTime', ctypes.c_longlong),
                    ('FileAttributes', wintypes.DWORD)]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
        wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.GetFileInformationByHandleEx.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                   wintypes.LPVOID, wintypes.DWORD]
    kernel.GetFileInformationByHandleEx.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    name = str(path)
    handle = kernel.CreateFileW(name, 0x80, 0x7, None, 3, 0x02000000, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ValueError('Read-only Windows file identity query unavailable')
    try:
        info = BasicInfo()
        if not kernel.GetFileInformationByHandleEx(handle, 0, ctypes.byref(info), ctypes.sizeof(info)):
            raise ValueError('Windows FileBasicInfo ChangeTime unavailable; immutable-input guard refused')
        require(info.ChangeTime > 0, 'Windows ChangeTime token unavailable; immutable-input guard refused')
        return int(info.ChangeTime)
    finally:
        kernel.CloseHandle(handle)


def file_identity(path):
    path = Path(path)
    try:
        resolved = path.resolve(strict=True)
        stat = resolved.stat()
    except OSError as exc:
        raise ValueError('Audited immutable file is missing') from exc
    return (str(resolved), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
            stat.st_dev, stat.st_ino, stat.st_mode,
            _windows_change_time(resolved) if os.name == 'nt' else stat.st_ctime_ns)


def capture_identities(paths):
    return {str(path): file_identity(path) for path in paths}


def assert_identities(identities, paths=None):
    for path in identities if paths is None else paths:
        require(file_identity(path) == identities[str(path)],
                'Audited manifest/quality/source/pool changed; immutable training input required')


def read_json(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return json.load(stream)


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def valid_id(value):
    require(isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', value),
            'Unsafe patient identifier')
    return value


def validate_splits(actual, reference, expected_sizes=(76, 19)):
    require(isinstance(actual, list) and isinstance(reference, list) and len(actual) == len(reference) == 5,
            'Existing fixed five-fold splits required; never generate splits')
    for fold, expected in zip(actual, reference):
        require(isinstance(fold, dict) and set(fold) == {'train', 'val'}, 'Invalid split fields')
        for key in ('train', 'val'):
            require(isinstance(fold[key], list), 'Split identifiers must be lists')
            for identifier in fold[key]:
                valid_id(identifier)
            require(len(set(fold[key])) == len(fold[key]), 'Duplicate split identifiers')
            require(sorted(fold[key]) == sorted(expected[key]), 'Canonical reference split mismatch')
        require(not set(fold['train']) & set(fold['val']), 'Train/validation overlap')
    train, val = sorted(actual[0]['train']), sorted(actual[0]['val'])
    require((len(train), len(val)) == expected_sizes, 'Dataset501 fold0 requires 76 train / 19 validation')
    return train, val


def allocate_counts(z_counts):
    require(bool(z_counts), 'Empty training cohort')
    require(all(type(z) is int and z > 0 for z in z_counts.values()), 'Positive integer source z counts required')
    # Integer arithmetic avoids float rounding changing ceil or remainder ties.
    result = {p: 3 * z // 10 for p, z in z_counts.items()}
    target = (3 * sum(z_counts.values()) + 9) // 10
    order = sorted(z_counts, key=lambda p: (-(3 * z_counts[p] % 10), p))
    for patient in order[:target - sum(result.values())]:
        result[patient] += 1
    return result


def check_output(path, protected, fresh=True):
    target = Path(path).expanduser().resolve()
    for entry in protected:
        root = Path(entry).expanduser().resolve()
        require(target != root and root not in target.parents and target not in root.parents,
                'Output must be outside and not an ancestor of repository/source/raw/protected paths')
    if fresh:
        require(not target.exists(), 'Fresh output directory required; no overwrite or resume')
    return target


def checked_file(root, name):
    root = Path(root).resolve()
    require(isinstance(name, str) and Path(name).name == name and name not in ('', '.', '..'),
            'Only local filenames allowed')
    path = (root / name).resolve()
    require(path.parent == root and path.name == name and path.is_file(),
            'Missing file or path aliases another patient or escapes declared folder')
    return path


def validate_arrays(data, seg, allow_empty=False):
    require(data.ndim == seg.ndim == 4 and data.shape == seg.shape and data.shape[0] == 1,
            'Native matching [1,Z,H,W] single-channel arrays required')
    require(all(s > 0 for s in data.shape[2:]) and (data.shape[1] > 0 or allow_empty), 'Empty source shape')
    require(np.issubdtype(data.dtype, np.floating) and np.isfinite(data).all(), 'Finite native floats required')
    require(np.issubdtype(seg.dtype, np.signedinteger) and np.isin(seg, [-1, 0, 1]).all(),
            'Signed binary/-1 segmentation required')


class Source:
    """Metadata checks precede dataset construction and any source-array reads."""
    def __init__(self, folder, raw_root, configuration='2d', fold=0):
        from importlib.metadata import version
        require(version('nnunetv2') == '2.8.1', 'This exploration is pinned to nnunetv2==2.8.1')
        require(type(fold) is int and fold == 0 and configuration == '2d', 'Only 2d fold0 is approved')
        self.folder = Path(folder).expanduser().resolve()
        self.base = self.folder.parent
        self.raw_root = Path(raw_root).expanduser().resolve()
        if os.environ.get('nnUNet_raw'):
            require(self.raw_root == Path(os.environ['nnUNet_raw']).expanduser().resolve(),
                    '--raw-root must match the configured nnUNet_raw protection root')
        require(self.folder.is_dir(), 'Missing original preprocessed folder')
        require(sha256(REFERENCE) == REFERENCE_SHA256, 'Pinned canonical reference split is missing or changed')
        self.splits = checked_file(self.base, 'splits_final.json')
        self.train, self.val = validate_splits(read_json(self.splits), read_json(REFERENCE))
        self.dataset_path = checked_file(self.base, 'dataset.json')
        self.plans_path = checked_file(self.base, 'nnUNetPlans.json')
        self.dataset_json, self.plans = read_json(self.dataset_path), read_json(self.plans_path)
        require(self.plans.get('dataset_name') == self.base.name == 'Dataset501_StrokeLesion',
                'Only the established Dataset501_StrokeLesion source is permitted')
        channels = self.dataset_json.get('channel_names', {})
        labels = self.dataset_json.get('labels', {})
        require(set(channels) == {'0'} and str(channels['0']).lower() == 'dwi', 'DWI-only source required')
        require(len(labels) == 2 and set(labels.values()) == {0, 1} and
                all(type(v) is int for v in labels.values()), 'Binary labels without regions/ignore class required')
        config = self.plans.get('configurations', {}).get('2d', {})
        require(config.get('data_identifier') == self.folder.name and len(config.get('patch_size', [])) == 2,
                'Original nnUNetPlans 2d configuration/data identifier mismatch')
        require(config.get('previous_stage') is None, 'Cascades are unsupported')
        from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
        self.dataset_class = infer_dataset_class(str(self.folder))
        identifiers = self.dataset_class.get_identifiers(str(self.folder))
        require(set(identifiers) == set(self.train + self.val), 'Source has missing/foreign/test patients')
        self.configuration = config
        self.protected = [REPO, self.base, self.raw_root]
        if os.environ.get('nnUNet_preprocessed'):
            self.protected.append(Path(os.environ['nnUNet_preprocessed']))
        self.identity = {'source_folder': str(self.folder), 'dataset_base': str(self.base),
                         'raw_root': str(self.raw_root), 'dataset_name': self.base.name,
                         'configuration': '2d', 'fold': 0, 'backend': self.dataset_class.__name__,
                         'dataset_sha256': sha256(self.dataset_path), 'plans_sha256': sha256(self.plans_path),
                         'splits_sha256': sha256(self.splits), 'reference_sha256': REFERENCE_SHA256,
                         'train_ids': self.train, 'val_ids': self.val}
        # Check filename provenance for ALL cases without loading their data or properties.
        for identifier in self.train + self.val:
            self.case_files(identifier)

    def case_files(self, identifier):
        require(identifier in self.train + self.val, 'Foreign source patient')
        from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
        if self.dataset_class is nnUNetDatasetBlosc2:
            names = [identifier + '.b2nd', identifier + '_seg.b2nd', identifier + '.pkl']
        else:
            names = [identifier + '.npz', identifier + '.pkl']
            names += [identifier + suffix for suffix in ('.npy', '_seg.npy')
                      if (self.folder / (identifier + suffix)).exists()]
        return [checked_file(self.folder, name) for name in names]

    def inventory(self, identifier):
        require(identifier in self.train, 'Generator may only inventory training patients')
        return {path.name: sha256(path) for path in self.case_files(identifier)}

    def verify_identity(self, identity):
        require(identity == self.identity, 'Source/config/split/cohort provenance mismatch')

    def verify_inventory(self, inventory):
        require(set(inventory) == set(self.train), 'Foreign/incomplete training inventory')
        for identifier in self.train:
            require(self.inventory(identifier) == inventory[identifier], 'Original training source changed')

    def training_dataset(self):
        return self.dataset_class(str(self.folder), self.train.copy(), None)

    def load_train(self, dataset, identifier):
        require(identifier in self.train and set(dataset.identifiers) == set(self.train), 'Train-only reads required')
        data, seg, previous, properties = dataset.load_case(identifier)
        require(previous is None, 'Cascades unsupported')
        data, seg = np.asarray(data[:]), np.asarray(seg[:])
        validate_arrays(data, seg)
        return data, seg, properties

    def require_no_unpack_writes(self):
        from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
        require(self.dataset_class is nnUNetDatasetBlosc2,
                'Trainer requires Blosc2: official NPZ unpack may create/repair/delete source files; refused')
