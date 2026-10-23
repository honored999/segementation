"""Explicit live-Adam joint adaptation; legacy JointEngine is untouched."""
from contextlib import contextmanager
from copy import deepcopy
import time
import torch
from experiments.genseg30.contract import require
from .core import (FunctionalModel, bind_alpha, clean, virtual_chain, segmentation_loss,
                   gan_generator_loss, stage_alpha_update, norm, PretrainEngine)

STRATEGY = 'joint_adam_trainbn_positive_sgdproxy_v1'
CLASSIFICATION = 'engineering_genseg_joint_adam_positive_not_formal'


@contextmanager
def private_eval(*nets):
    modes = [(m, m.training) for net in nets for m in net.modules()]
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []):
        try:
            for net in nets:
                net.eval()
            yield
        finally:
            for module, mode in modes:
                module.training = mode


class RepairJointEngine:
    strategy = STRATEGY

    def __init__(self, generator, discriminator, segmenter):
        require(getattr(generator, 'pretrain_strategy', None) == 'adam_trainbn_epoch_v1',
                'New joint strategy requires audited repair G/D weights')
        self.g, self.d, self.s = generator, discriminator, segmenter
        self.s_descriptor=getattr(segmenter,'genseg_s_descriptor',None)
        if self.s_descriptor is not None:
            from .nnunet_s import audit_norm, STRATEGY as NN_STRATEGY
            audit_norm(segmenter,self.s_descriptor)
            self.strategy=NN_STRATEGY
        self.steps = 0
        self.poisoned = False
        self.failure_stage = None
        self.rates = dict(g=2e-5, d=2e-5, s=1e-5, alpha=1e-4)
        self.l1_weight = 100.
        # Reuse the actual G1/D3 Adam implementation, with fresh moments.
        self.gan = PretrainEngine(generator, discriminator, strategy='adam_trainbn_epoch_v1')
        n = generator.networks
        n.conv_arch.requires_grad_(False)
        n.upconv_arch = n.upconv_arch.detach().clone().requires_grad_(True)
        n._upconv_arch_parameters = [n.upconv_arch]
        n._arch_parameters = [n.upconv_arch, n.conv_arch]
        self._alpha_state = deepcopy(torch.optim.Adam([n.upconv_arch], lr=1e-4,
            betas=(.5, .999), eps=1e-8, weight_decay=0.).state_dict())
        self.s.train()
        self.s_optimizer = torch.optim.Adam(self.s.parameters(), lr=1e-5,
            betas=(.5, .999), eps=1e-8, weight_decay=0.)
        self.initial_bn = {key: {k: int(v) for k,v in net.state_dict().items()
            if k.endswith('num_batches_tracked')} for key,net in self.networks()}

    def networks(self):
        return (('G', self.g.net), ('D', self.d), ('S', self.s))

    def hyperloss(self, real, label, validation, val_label, fake_label, alpha):
        """Eval-only private single-SGD proxy; preserves live modes, RNG and grads."""
        with private_eval(self.g.net, self.d, self.s):
            gf, df, sf = (FunctionalModel(n) for _, n in self.networks())
            gp, dp, sp = gf.parameters(), df.parameters(), sf.parameters()
            condition = (label == 1).to(real.dtype)
            fake_condition = (fake_label == 1).to(real.dtype)
            valid, fake_valid = (label >= 0).to(real.dtype), (fake_label >= 0).to(real.dtype)
            def forward(p, x):
                with bind_alpha(self.g, alpha):
                    return gf(p, x)
            def gl(p):
                return gan_generator_loss(df, dp, condition, forward(p, condition), real, valid, 100.)
            def sl(p, vg):
                fake = forward(vg, fake_condition)
                return segmentation_loss(sf(p, real), label) + segmentation_loss(sf(p, fake*fake_valid), fake_label)
            return virtual_chain(gp, sp, gl, sl,
                lambda p: segmentation_loss(sf(p, validation), val_label), 2e-5, 1e-5)[0]

    def step(self, real, label, validation, val_label, fake_label):
        require(not self.poisoned, 'Joint engine poisoned; no retry')
        real, validation = clean(real, label), clean(validation, val_label)
        clean(torch.zeros_like(fake_label), fake_label)
        require(bool((fake_label == 1).any()), 'Synthetic branch requires positive mask')
        timings = {}
        try:
            self.failure_stage = 'virtual_hypergradient'
            started = time.perf_counter()
            alpha = self.g.networks.upconv_arch
            vl = self.hyperloss(real, label, validation, val_label, fake_label, alpha)
            ag, = torch.autograd.grad(vl, alpha)
            require(bool(torch.isfinite(ag).all()) and norm([ag]) > 0, 'Absent/nonfinite/zero hypergradient')
            new_alpha, staged = stage_alpha_update(alpha, ag, self._alpha_state)
            require(bool(torch.isfinite(new_alpha).all()), 'Nonfinite alpha')
            validation_value = float(vl.detach())
            del vl
            timings['virtual_hypergradient'] = time.perf_counter()-started
            self.failure_stage = 'actual_GD'
            started = time.perf_counter()
            # New alpha bound privately while actual GAN updates retain G1/D3.
            with bind_alpha(self.g, new_alpha):
                metrics = self.gan.step(real, label)
            timings['actual_GD'] = time.perf_counter()-started
            self.failure_stage = 'actual_S'
            started = time.perf_counter()
            with private_eval(self.g.net), torch.no_grad(), bind_alpha(self.g, new_alpha):
                fake = self.g.net((fake_label == 1).to(real.dtype))
                fake = fake * (fake_label >= 0).to(real.dtype)
            self.s.train()
            owned=self.s_optimizer.param_groups
            require(len(owned)==1 and len(owned[0]['params'])==len(list(self.s.parameters())) and
                all(a is b for a,b in zip(owned[0]['params'],self.s.parameters())), 'S Adam parameter ownership changed')
            loss = segmentation_loss(self.s(real), label) + segmentation_loss(self.s(fake), fake_label)
            require(bool(torch.isfinite(loss)), 'Nonfinite S loss')
            inactive=set(getattr(self.s,'genseg_inactive_parameters',()))
            grads = torch.autograd.grad(loss, tuple(self.s.parameters()),allow_unused=bool(inactive))
            require(all((v is None)==(k in inactive) for k,v in zip(dict(self.s.named_parameters()),grads)),
                'Unexpected disconnected/active actual S parameter')
            require(all(bool(torch.isfinite(v).all()) for v in grads if v is not None), 'Nonfinite S gradient')
            self.s_optimizer.zero_grad(set_to_none=True)
            for p, grad in zip(self.s.parameters(), grads):
                p.grad = None if grad is None else grad.detach()
            self.s_optimizer.step()
            with torch.no_grad():
                alpha.copy_(new_alpha)
            self._alpha_state = staged
            for key, net in self.networks():
                require(all(bool(torch.isfinite(v).all()) for v in net.state_dict().values()), 'Nonfinite live '+key)
            require(all(bool(torch.isfinite(v).all()) for item in self.s_optimizer.state.values()
                for v in item.values() if isinstance(v, torch.Tensor)), 'Nonfinite S moments')
            next_step = self.steps+1
            for key, net in self.networks():
                count = {'G':1, 'D':3, 'S':2}[key]
                require(all(int(net.state_dict()[k]) == v+count*next_step
                    for k,v in self.initial_bn[key].items()), 'Joint BN count mismatch '+key)
            if self.s_descriptor is not None:
                from .nnunet_s import audit_norm
                audit_norm(self.s,self.s_descriptor)
            self.steps = next_step
            self.failure_stage = None
            timings['actual_S'] = time.perf_counter()-started
            return dict(metrics, s_loss=float(loss.detach()), alpha_gradient_norm=norm([ag]),
                internal_validation_loss=validation_value, s_gradient_norm=norm([v for v in grads if v is not None]),
                forwards={'virtual':{'G':2,'D':1,'S':3}, 'actual_train':{'G':1,'D':3,'S':2},
                          'actual_eval_auxiliary':{'G':1,'D':0,'S':0}},
                bn_updates={'G':1,'D':3,'S':0 if self.s_descriptor is not None else 2}, stage_host_wall_seconds=timings,
                timing_policy='host_enqueue_wall; CUDA execution/synchronization charged to unit boundary')
        except BaseException:
            self.poisoned = True
            raise

    def optimizer_state(self):
        require(not self.poisoned, 'Cannot save poisoned joint engine')
        state = self.gan.optimizer_state()
        state['s'] = dict(algorithm='Adam', lr=1e-5, betas=(.5,.999), eps=1e-8,
            weight_decay=0., steps=self.steps, parameter_names=list(dict(self.s.named_parameters())),
            state_dict=deepcopy(self.s_optimizer.state_dict()))
        state['alpha'] = dict(algorithm='staged_Adam', steps=self.steps, state_dict=deepcopy(self._alpha_state))
        return state
