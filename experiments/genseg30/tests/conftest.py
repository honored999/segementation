from pathlib import Path
import json
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.genseg30 import contract


@pytest.fixture
def source_factory(tmp_path, monkeypatch):
    train = [f'fixture{i:03d}' for i in range(76)]
    val = [f'heldout{i:03d}' for i in range(19)]
    splits = [{'train': train, 'val': val}] * 5
    reference = tmp_path / 'fixed_reference.json'
    reference.write_text(json.dumps(splits))
    monkeypatch.setattr(contract, 'REFERENCE', reference)
    monkeypatch.setattr(contract, 'REFERENCE_SHA256', contract.sha256(reference))
    monkeypatch.delenv('nnUNet_preprocessed', raising=False)

    def create(backend='npz'):
        base = tmp_path / backend / 'Dataset501_StrokeLesion'
        folder = base / 'nnUNetPlans_2d'
        folder.mkdir(parents=True)
        (base / 'splits_final.json').write_text(json.dumps(splits))
        (base / 'dataset.json').write_text(json.dumps({'channel_names': {'0': 'DWI'},
            'labels': {'background': 0, 'lesion': 1}}))
        (base / 'nnUNetPlans.json').write_text(json.dumps({'dataset_name': base.name,
            'plans_name': 'nnUNetPlans', 'configurations': {'2d': {'data_identifier': folder.name, 'patch_size': [16, 16],
                'architecture': {'network_class_name': 'dynamic_network_architectures.architectures.unet.PlainConvUNet',
                    'arch_kwargs': {'n_stages': 2, 'features_per_stage': [2, 4],
                        'conv_op': 'torch.nn.modules.conv.Conv2d', 'kernel_sizes': [[3, 3], [3, 3]],
                        'strides': [[1, 1], [2, 2]], 'n_conv_per_stage': [1, 1], 'n_conv_per_stage_decoder': [1],
                        'conv_bias': True, 'norm_op': 'torch.nn.modules.instancenorm.InstanceNorm2d',
                        'norm_op_kwargs': {'eps': 1e-5, 'affine': True}, 'dropout_op': None,
                        'dropout_op_kwargs': None, 'nonlin': 'torch.nn.LeakyReLU', 'nonlin_kwargs': {'inplace': True}},
                    '_kw_requires_import': ['conv_op', 'norm_op', 'dropout_op', 'nonlin']}}}}))
        for identifier in train + val:
            z = 3 if identifier == train[0] else 1
            image = np.linspace(-2, 3, z * 9 * 11, dtype=np.float32).reshape(1, z, 9, 11)
            seg = np.zeros(image.shape, dtype=np.int16)
            seg[:, :, 2:4, 3:5] = 1
            seg[:, :, 0, 0] = -1
            if backend == 'npz':
                np.savez(folder / (identifier + '.npz'), data=image, seg=seg)
            else:
                import blosc2
                blosc2.asarray(image, urlpath=str(folder / (identifier + '.b2nd')))
                blosc2.asarray(seg, urlpath=str(folder / (identifier + '_seg.b2nd')))
            with (folder / (identifier + '.pkl')).open('wb') as stream:
                pickle.dump({'class_locations': {1: np.argwhere(seg == 1)}, 'fixture': True}, stream)
        monkeypatch.setenv('nnUNet_preprocessed', str(base.parent))
        monkeypatch.setenv('nnUNet_raw', str(tmp_path / 'raw'))
        return contract.Source(folder, tmp_path / 'raw')
    return create
