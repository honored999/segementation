"""Construction and checkpoint metadata for the converted first layer."""
from torch import nn
from torch.nn import functional as F

from .psf_first_conv import KEY, KERNEL_SIZES, sha256


def resize_first_conv(network, size):
    old = network.get_submodule(KEY)
    if (type(size) is not int or size not in KERNEL_SIZES
            or not isinstance(old, nn.Conv2d)
            or (old.in_channels, old.out_channels) != (1, 32)
            or old.kernel_size != (3, 3) or old.padding != (1, 1)
            or old.stride != (1, 1) or old.dilation != (1, 1)
            or old.groups != 1 or old.padding_mode != 'zeros'):
        raise ValueError('Invalid first-convolution architecture or kernel size')
    if size == 3:
        return old
    # The nnU-Net block registers this Conv2d both as conv and in all_modules.
    # The decoder can also hold a reference to the same encoder.
    pad = (size-3)//2
    old.weight = nn.Parameter(F.pad(old.weight.detach(), (pad,)*4))
    old.kernel_size = (size, size)
    old.padding = (size//2,)*2
    return old


def load_network(network, payload):
    metadata = payload.get('first_layer')
    if metadata is not None:
        resize_first_conv(network, metadata['kernel_size'])
    first = network.get_submodule(KEY)
    weights = payload['network_weights']
    if metadata is not None:
        import torch
        aliases = first_weight_names(network)
        if any(not torch.equal(weights[name], weights[KEY+'.weight']) for name in aliases):
            raise ValueError('Checkpoint first-weight aliases disagree')
    network.load_state_dict(payload['network_weights'], strict=True)
    return first


def first_weight_names(network):
    weight = network.get_submodule(KEY).weight
    return {name for name, parameter in network.named_parameters(remove_duplicate=False)
            if parameter is weight}


def check_artifact_origin(directory, expected):
    import json
    config = json.loads((directory/'config.json').read_text(encoding='utf-8'))
    if config.get('target_checkpoint_sha256') != expected:
        raise ValueError('PSF was fitted to a different checkpoint')
