import json

import numpy as np
import pytest
import torch
from torch import nn

from tools.psf_conversion.first_layer_model import check_artifact_origin, load_network, resize_first_conv
from tools.psf_conversion.psf_first_conv import KEY, PIXEL_OPERATOR, FirstConvHook, load_psfs


def network():
    model = nn.Module()
    parent = model
    for name in KEY.split('.')[:-1]:
        child = nn.Module()
        parent.add_module(name, child)
        parent = child
    parent.add_module('conv', nn.Conv2d(1, 32, 3, padding=1))
    parent.add_module('all_modules', nn.Sequential(parent.conv))
    model.add_module('decoder_encoder_alias', model.encoder)
    return model


@pytest.mark.parametrize('size', [3, 5, 7])
def test_kernel_geometry_roundtrip(tmp_path, size):
    model = network()
    old = model.get_submodule(KEY)
    pulses = torch.zeros(2, 1, 11, 13)
    pulses[0, 0, 5, 6] = 1
    pulses[1, 0, 0, -1] = 1
    reference = old(pulses)
    conv = resize_first_conv(model, size)
    forward = model.get_submodule(KEY.rsplit('.', 1)[0]).all_modules
    assert forward[0] is conv
    assert model.decoder_encoder_alias is model.encoder
    torch.testing.assert_close(forward(pulses), reference)
    with torch.no_grad():
        conv.weight[0, 0, 0, -1] += 1.0
    checkpoint = tmp_path/'target.pth'
    torch.save(dict(network_weights=model.state_dict(), first_layer={'kernel_size': size}), checkpoint)
    clone = network()
    loaded = load_network(clone, torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(conv(pulses), loaded(pulses))
    broken = torch.load(checkpoint, weights_only=True)
    alias = KEY.removesuffix('conv') + 'all_modules.0.weight'
    broken['network_weights'][alias] = broken['network_weights'][alias] + 1
    with pytest.raises(ValueError, match='aliases disagree'):
        load_network(network(), broken)
    weights = conv.weight.detach().numpy()
    patch, n = 17*size, 128
    start = n//2 - patch//2
    config = dict(n=n, patch_meta=patch, wavelength_nm=532, target_rotated_180=True,
                  fft_size=256, padding=64, grayscale=True, first_kernel_size=size,
                  sample_cell=17, target_checkpoint_sha256='correct')
    (tmp_path/'config.json').write_text(json.dumps(config), encoding='utf-8')
    scales = np.abs(weights).max(axis=(1, 2, 3)).clip(1e-12)
    np.save(tmp_path/'electrical_kernels.npy', weights)
    np.save(tmp_path/'shared_scales.npy', scales)
    for channel in range(32):
        for sign, polarity in [('pos', 1), ('neg', -1)]:
            target = np.zeros((1, n, n), dtype=np.float32)
            component = np.maximum(polarity*weights[channel, 0, ::-1, ::-1]/scales[channel], 0)
            target[0, start:start+patch, start:start+patch] = component.repeat(17, 0).repeat(17, 1)
            active = bool(component.any())
            np.savez(tmp_path/f'kernel_{channel:03d}_{sign}.npz', psf_raw=target,
                     target=target, gain_gray=np.array([float(active)], dtype=np.float32),
                     active_gray=np.array([active]), wavelength_nm=np.array(532))
    check_artifact_origin(tmp_path, 'correct')
    with pytest.raises(ValueError, match='different checkpoint'):
        check_artifact_origin(tmp_path, 'wrong')
    with pytest.raises(ValueError, match='Checkpoint first layer differs'):
        load_psfs(tmp_path, conv.weight + 1, operator=PIXEL_OPERATOR)
    electrical = loaded(pulses)
    for ideal in (False, True):
        kernel, hashes = load_psfs(tmp_path, loaded.weight, operator=PIXEL_OPERATOR, ideal=ideal)
        hook = FirstConvHook(clone, kernel)
        assert len(hashes) == 67
        assert kernel.shape[-1] > size
        torch.testing.assert_close(loaded(pulses), electrical)
        hook.remove()
