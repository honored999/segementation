"""Unmodified pinned author networks, with private import/global contexts."""
import importlib.util
import sys
import types
import uuid
import hashlib
import json
import torch
from experiments.genseg30.models import Generator as BaseGenerator, FILES, UPSTREAM_COMMIT
from experiments.genseg30.contract import REPO, require, sha256

UNET_FILES={
 'unet/unet_model.py':'1e6bf74e9a629a41a1e7fa86db3d7255f384d15629100f19f468588de006a53a',
 'unet/unet_parts.py':'4b0441f549caf588642f7e5ef3ba2dfb1032c1ded0ba350e46f96f43caa93ea1'}


def source_identity():
    # Canonical pinned LF hashes plus observed raw hashes, no arbitrary whitespace normalization.
    files=FILES|UNET_FILES
    observed={}
    for name,digest in files.items():
        raw=(REPO/'reference/GenSeg'/name).read_bytes()
        canonical=hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()
        require(canonical==digest,'Pinned upstream module changed: '+name)
        observed[name]=hashlib.sha256(raw).hexdigest()
    return {'commit':UPSTREAM_COMMIT,'files':files,'raw_files':observed,'canonicalization':'CRLF_to_LF_only'}


def private_networks():
    source_identity()
    root=REPO/'reference/GenSeg'
    aliases=['architecture_pix2pix','architecture_pix2pix.genotypes','architecture_pix2pix.operations']
    old={k:sys.modules.get(k) for k in aliases}; old_path=sys.path.copy()
    try:
        package=types.ModuleType(aliases[0]); package.__path__=[str(root/'architecture_pix2pix')]
        sys.modules[aliases[0]]=package
        for name,path in [(aliases[1],'architecture_pix2pix/genotypes.py'),
                          (aliases[2],'architecture_pix2pix/operations.py'),
                          ('_genseg_joint_'+uuid.uuid4().hex,'models_pix2pix/networks.py')]:
            spec=importlib.util.spec_from_file_location(name,root/path)
            module=importlib.util.module_from_spec(spec)
            if name in aliases: sys.modules[name]=module
            spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:]=old_path
        for name,value in old.items():
            if value is None: sys.modules.pop(name,None)
            else: sys.modules[name]=value


class Generator(BaseGenerator):
    def __init__(self,width=64,downs=8,seed=20261018,device='cpu'):
        require(type(width) is int and width>0 and type(downs) is int and 5<=downs<=8,'Invalid generator dimensions')
        self.config={'width':width,'downs':downs,'seed':seed}; self.device=torch.device(device)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.networks=private_networks()
            self.net=self.networks.UnetGenerator(1,1,downs,width,norm_layer=self.networks.get_norm_layer('batch'),use_dropout=False)
            self.networks.init_weights(self.net,'normal',.02)
        self.net.to(self.device)
        self.restore_arch({name:getattr(self.networks,name) for name in ('conv_arch','upconv_arch')})


def author_unet():
    source_identity()
    root=REPO/'reference/GenSeg/unet'
    name='_genseg_joint_unet_'+uuid.uuid4().hex
    package=types.ModuleType(name); package.__path__=[str(root)]
    sys.modules[name]=package
    old_path=sys.path.copy()
    try:
        for part in ('unet_parts','unet_model'):
            spec=importlib.util.spec_from_file_location(name+'.'+part,root/(part+'.py'))
            mod=importlib.util.module_from_spec(spec)
            sys.modules[name+'.'+part]=mod
            spec.loader.exec_module(mod)
        return mod.UNet(1,1)
    finally:
        sys.path[:]=old_path
        for key in (name,name+'.unet_parts',name+'.unet_model'):
            sys.modules.pop(key,None)


def create_models(*,width=64,downs=8,discriminator_width=64,seed=20261018,device='cpu'):
    source_identity()
    torch.manual_seed(seed)
    g=Generator(width,downs,seed,device)
    d=g.discriminator(discriminator_width)
    s=author_unet().to(device)
    return g,d,s
