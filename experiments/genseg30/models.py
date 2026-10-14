"""Pinned GenSeg modules in a private Python context; no upstream edits."""
from __future__ import annotations

import importlib.util
import json
import sys
import types
import uuid

import numpy as np
import torch

from .contract import REPO, require, sha256

UPSTREAM_COMMIT = '6480e178a5d44cb838f53f7951aae48138b8f5eb'
FILES = {
    'architecture_pix2pix/genotypes.py': '81636600f2f727218f50f575d5ef14a2d6af51674f38689bdf9c261539a8fb2d',
    'architecture_pix2pix/operations.py': '6d39ef853542c8c3e12bc1848b47b8b4455dfff93a0159fd6297be715d34327e',
    'models_pix2pix/networks.py': '6d1124db13719511811700fe9596cb658b6e8669a3a8064d44a97b1fae60f275',
}


def import_networks():
    root = REPO / 'reference/GenSeg'
    for name, digest in FILES.items():
        require(sha256(root / name) == digest, 'Pinned upstream module changed: ' + name)
    aliases = ['architecture_pix2pix', 'architecture_pix2pix.genotypes', 'architecture_pix2pix.operations']
    old = {name: sys.modules.get(name) for name in aliases}
    old_path = sys.path.copy()
    try:
        package = types.ModuleType(aliases[0])
        package.__path__ = [str(root / 'architecture_pix2pix')]
        sys.modules[aliases[0]] = package
        for name, path in [(aliases[1], 'architecture_pix2pix/genotypes.py'),
                           (aliases[2], 'architecture_pix2pix/operations.py'),
                           ('_genseg30_' + uuid.uuid4().hex, 'models_pix2pix/networks.py')]:
            spec = importlib.util.spec_from_file_location(name, root / path)
            module = importlib.util.module_from_spec(spec)
            if name in aliases:
                sys.modules[name] = module
            spec.loader.exec_module(module)
        return module  # each model retains its own networks globals, even after a second import
    finally:
        sys.path[:] = old_path
        for name, value in old.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def pad_plane(plane, multiple, fill):
    require(plane.ndim == 2 and multiple > 0, 'Expected one full 2D plane')
    height, width = plane.shape
    out = np.full((((height + multiple - 1) // multiple) * multiple,
                   ((width + multiple - 1) // multiple) * multiple), fill, dtype=plane.dtype)
    out[:height, :width] = plane
    return out


class Generator:
    def __init__(self, width=64, downs=8, seed=20261014, device='cpu'):
        require(type(width) is int and width > 0 and type(downs) is int and 5 <= downs <= 8,
                'Positive width and 5..8 generator downsamplings required')
        require(type(seed) is int and 0 <= seed < 2**32, 'Seed must be uint32')
        self.config = {'width': width, 'downs': downs, 'seed': seed}
        self.device = torch.device(device)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.networks = import_networks()
            self.net = self.networks.UnetGenerator(1, 1, downs, width,
                norm_layer=self.networks.get_norm_layer('batch'), use_dropout=False)
            self.networks.init_weights(self.net, 'normal', .02)
        self.net.to(self.device)
        self.restore_arch({name: getattr(self.networks, name) for name in ('conv_arch', 'upconv_arch')})

    def restore_arch(self, values):
        for name in ('conv_arch', 'upconv_arch'):
            value = values[name]
            require(value.shape == getattr(self.networks, name).shape and torch.isfinite(value).all().item(),
                    'Architecture tensor shape/finiteness mismatch')
            setattr(self.networks, name, value.detach().clone().to(self.device).requires_grad_(False))
        # Upstream caches lists of global tensors. Rebind ALL aliases after movement/reload.
        self.networks._conv_arch_parameters = [self.networks.conv_arch]
        self.networks._upconv_arch_parameters = [self.networks.upconv_arch]
        self.networks._arch_parameters = [self.networks.upconv_arch, self.networks.conv_arch]

    def discriminator(self, width=64):
        net = self.networks.NLayerDiscriminator(2, width, n_layers=3,
                norm_layer=self.networks.get_norm_layer('batch'))
        self.networks.init_weights(net, 'normal', .02)
        return net.to(self.device)

    def save(self, path, metadata, discriminator=None):
        payload = {'format': 1, 'upstream_commit': UPSTREAM_COMMIT, 'upstream_files': FILES,
                   'config': self.config, 'metadata_json': json.dumps(metadata, allow_nan=False),
                   'generator': self.net.state_dict(),
                   'arch': {name: getattr(self.networks, name).detach().cpu().clone()
                            for name in ('conv_arch', 'upconv_arch')}}
        if discriminator is not None:
            payload['discriminator'] = discriminator.state_dict()
        with path.open('xb') as stream:
            torch.save(payload, stream)

    @classmethod
    def load(cls, path, device='cpu'):
        # Only tensors/primitive metadata; never load arbitrary pickle objects from generator files.
        payload = torch.load(path, map_location='cpu', weights_only=True)
        require(payload['format'] == 1 and payload['upstream_commit'] == UPSTREAM_COMMIT and
                payload['upstream_files'] == FILES, 'Generator upstream identity mismatch')
        generator = cls(**payload['config'], device=device)
        generator.net.load_state_dict(payload['generator'], strict=True)
        generator.restore_arch(payload['arch'])
        require(all(torch.isfinite(v).all().item() for v in generator.net.state_dict().values()),
                'Nonfinite generator weights')
        return generator, json.loads(payload['metadata_json'])


def train_step(generator, discriminator, optimizer_g, optimizer_d, condition, target, l1_weight, valid=None):
    if valid is None:
        valid = torch.ones_like(target)
    require(torch.count_nonzero(valid).item() > 0, 'Slice contains no valid pixels')
    fake = generator.net(condition)
    require(fake.shape == target.shape and torch.isfinite(fake).all().item(), 'Invalid generator output')
    gan = generator.networks.GANLoss('vanilla').to(generator.device)
    for parameter in discriminator.parameters():
        parameter.requires_grad_(True)
    optimizer_d.zero_grad(set_to_none=True)
    real_input = torch.cat((condition, target * valid), 1)
    fake_input = torch.cat((condition, fake * valid), 1)
    loss_d = .5 * (gan(discriminator(real_input), True) + gan(discriminator(fake_input.detach()), False))
    loss_d.backward()
    optimizer_d.step()
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)
    optimizer_g.zero_grad(set_to_none=True)
    loss_g = gan(discriminator(fake_input), True) + l1_weight * ((fake - target).abs() * valid).sum() / valid.sum()
    loss_g.backward()
    optimizer_g.step()
    result = {'generator_loss': float(loss_g.detach()), 'discriminator_loss': float(loss_d.detach())}
    require(all(np.isfinite(x) for x in result.values()), 'Nonfinite GAN loss')
    return result
