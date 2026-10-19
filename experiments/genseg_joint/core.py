"""Legacy functional SGD/staged alpha Adam and explicit live GAN Adam.

Legacy updates stage candidates before mutation. The repair pretrainer instead
uses live trainBN/Adam; failures poison it and do not promise atomic rollback.
"""
from contextlib import contextmanager
from copy import deepcopy
import math
import torch
import torch.nn.functional as F
from torch.func import functional_call
from experiments.genseg30.contract import require


def virtual_sgd(parameters, loss, lr):
    grads = torch.autograd.grad(loss, tuple(parameters.values()), create_graph=True)
    return {k: p-lr*g for (k,p),g in zip(parameters.items(),grads)}


def virtual_chain(gp, sp, generator_loss, segmentation_train_loss, validation_loss, g_lr, s_lr):
    vg = virtual_sgd(gp, generator_loss(gp), g_lr)
    vs = virtual_sgd(sp, segmentation_train_loss(sp, vg), s_lr)
    return validation_loss(vs), vg, vs


def stage_alpha_update(alpha, gradient, saved_state):
    """Step a private clone with private Adam moments; never mutate alpha/saved state."""
    candidate=alpha.detach().clone().requires_grad_(True)
    optimizer=torch.optim.Adam([candidate],lr=saved_state['param_groups'][0]['lr'],
                               betas=(.5,.999),eps=1e-8,weight_decay=0.)
    optimizer.load_state_dict(deepcopy(saved_state))
    candidate.grad=gradient.detach().clone()
    optimizer.step()
    state=deepcopy(optimizer.state_dict())
    require(all(bool(torch.isfinite(v).all()) for item in state['state'].values()
                for v in item.values() if isinstance(v,torch.Tensor)), 'Nonfinite staged alpha Adam state')
    return candidate.detach(),state


def segmentation_loss(logits, target):
    valid = target >= 0
    require(bool(valid.any()), 'No valid segmentation pixels')
    y = target.clamp_min(0)
    bce = F.binary_cross_entropy_with_logits(logits[valid], y[valid])
    prob = logits.sigmoid()[valid]; label = y[valid]
    dice = 1-(2*(prob*label).sum()+1)/(prob.sum()+label.sum()+1)
    return bce+dice


class FunctionalModel:
    def __init__(self, net):
        self.net = net
        self.buffers = {k:v.detach().clone() for k,v in net.named_buffers()}

    def parameters(self):
        return {k:v.detach().clone().requires_grad_(True) for k,v in self.net.named_parameters()}

    def __call__(self, parameters, x):
        # Even training-mode BN may modify only private buffers, never module live state.
        buffers = {k:v.clone() for k,v in self.buffers.items()}
        return functional_call(self.net, (parameters,buffers), (x,), strict=True)


@contextmanager
def bind_alpha(generator, alpha):
    n = generator.networks
    old = (n.upconv_arch, n._upconv_arch_parameters, n._arch_parameters)
    n.upconv_arch = alpha
    n._upconv_arch_parameters = [alpha]
    n._arch_parameters = [alpha,n.conv_arch]
    try:
        yield
    finally:
        n.upconv_arch, n._upconv_arch_parameters, n._arch_parameters = old


def norm(tensors):
    # float64 accumulation prevents float32 squared-gradient underflow in diagnostics.
    return math.sqrt(sum(float(t.detach().double().square().sum()) for t in tensors))


def clean(image, label):
    require(image.shape == label.shape and image.ndim == 4, 'Expected matching NCHW tensors')
    require(bool(torch.isfinite(image).all()) and bool(torch.isfinite(label).all()), 'Nonfinite batch')
    require(bool(((label == -1)|(label == 0)|(label == 1)).all()), 'Binary/-1 labels required')
    require(bool((label >= 0).any()), 'No valid batch pixels')
    return torch.where(label >= 0, image, torch.zeros_like(image))


class JointEngine:
    def __init__(self, generator, discriminator, segmenter, *, g_lr=2e-5, d_lr=2e-5,
                 s_lr=1e-5, arch_lr=1e-4, l1_weight=100.):
        require(getattr(generator,'pretrain_strategy','legacy')=='legacy',
                'Repair pretraining cannot enter legacy joint: stage B learning gate and stage C adaptation are pending')
        self.g,self.d,self.s = generator,discriminator,segmenter
        self.rates = dict(g=g_lr,d=d_lr,s=s_lr,alpha=arch_lr)
        require(all(math.isfinite(v) and v > 0 for v in self.rates.values()), 'Positive finite rates required')
        require(math.isfinite(l1_weight) and l1_weight >= 0, 'Finite nonnegative L1 weight required')
        self.l1_weight=l1_weight
        self.steps=0
        for net in (generator.net,discriminator,segmenter):
            net.eval()  # Explicit engineering deviation: frozen BN everywhere.
        alpha = generator.networks.upconv_arch.detach().clone().requires_grad_(True)
        generator.networks.upconv_arch=alpha
        generator.networks._upconv_arch_parameters=[alpha]
        generator.networks._arch_parameters=[alpha,generator.networks.conv_arch]
        self._alpha_state=deepcopy(torch.optim.Adam([alpha],lr=arch_lr,betas=(.5,.999),
                                                   eps=1e-8,weight_decay=0.).state_dict())

    def optimizer_state(self):
        state={k:{'algorithm':'stateless_SGD','lr':v,'momentum':0.,'weight_decay':0.,
                   'steps':self.steps,'state':{}} for k,v in self.rates.items() if k!='alpha'}
        state['alpha']={'algorithm':'Adam','lr':self.rates['alpha'],'betas':(.5,.999),
                        'eps':1e-8,'weight_decay':0.,'steps':self.steps,'state_dict':deepcopy(self._alpha_state)}
        return state

    def step(self, real, label, validation, val_label, fake_label):
        real=clean(real,label); validation=clean(validation,val_label)
        clean(torch.zeros_like(fake_label),fake_label)
        condition=(label == 1).to(real.dtype)
        fake_condition=(fake_label == 1).to(real.dtype)
        valid=(label >= 0).to(real.dtype)
        fake_valid=(fake_label >= 0).to(real.dtype)
        gf,df,sf=FunctionalModel(self.g.net),FunctionalModel(self.d),FunctionalModel(self.s)
        gp,dp,sp=gf.parameters(),df.parameters(),sf.parameters()
        alpha=self.g.networks.upconv_arch
        def forward_g(p,x,a):
            with bind_alpha(self.g,a):
                return gf(p,x)
        def gl(p,a):
            fake=forward_g(p,condition,a)
            return gan_generator_loss(df,dp,condition,fake,real,valid,self.l1_weight)
        def sl(p,pg,a):
            fake=forward_g(pg,fake_condition,a)
            return segmentation_loss(sf(p,real),label)+segmentation_loss(sf(p,fake*fake_valid),fake_label)
        vl,vg,vs=virtual_chain(gp,sp,lambda p:gl(p,alpha),lambda p,pg:sl(p,pg,alpha),
            lambda p:segmentation_loss(sf(p,validation),val_label),self.rates['g'],self.rates['s'])
        ag,=torch.autograd.grad(vl,alpha)
        self.last_diagnostics={'alpha_gradient_norm':norm([ag]),'internal_validation_loss':float(vl.detach())}
        require(bool(torch.isfinite(ag).all()) and norm([ag]) > 0, 'Absent/nonfinite/zero architecture hypergradient')
        new_alpha,staged_alpha_state=stage_alpha_update(alpha,ag,self._alpha_state)
        change=norm([new_alpha-alpha.detach()])
        self.last_diagnostics['alpha_change_norm']=change
        require(change > 0, 'Architecture update rounded to zero; incomplete pilot, not success')
        # Release the large higher-order graph before actual updates.
        del vl,vg,vs
        new_g_loss=gl(gp,new_alpha)
        gg=torch.autograd.grad(new_g_loss,tuple(gp.values()))
        # Stage actual G's SGD candidate; S sees updated G and the new architecture.
        actual_gp={k:(p-self.rates['g']*grad).detach() for (k,p),grad in zip(gp.items(),gg)}
        with torch.no_grad():
            fake_s=forward_g(actual_gp,fake_condition,new_alpha)
            fake_d=forward_g(gp,condition,new_alpha)
        new_s_loss=segmentation_loss(sf(sp,real),label)+segmentation_loss(sf(sp,fake_s*fake_valid),fake_label)
        sg=torch.autograd.grad(new_s_loss,tuple(sp.values()))
        d_loss=.5*(adversarial(df(dp,torch.cat((condition,real*valid),1)),True)+
                    adversarial(df(dp,torch.cat((condition,fake_d*valid),1)),False))
        dg=torch.autograd.grad(d_loss,tuple(dp.values()))
        candidates=[]
        for net,p,grads,key in ((self.g.net,gp,gg,'g'),(self.d,dp,dg,'d'),(self.s,sp,sg,'s')):
            values={k:(v-self.rates[key]*grad).detach() for (k,v),grad in zip(p.items(),grads)}
            require(all(bool(torch.isfinite(v).all()) for v in values.values()), 'Nonfinite SGD candidate; no live update')
            candidates.append((net,values))
        require(all(bool(torch.isfinite(v)) for v in (new_g_loss,new_s_loss,d_loss)), 'Nonfinite loss')
        require(bool(torch.isfinite(new_alpha).all()), 'Nonfinite alpha candidate')
        metrics={**self.last_diagnostics,
                 'g_gradient_norm':norm(gg),'d_gradient_norm':norm(dg),'s_gradient_norm':norm(sg),
                 'g_loss':float(new_g_loss.detach()),'s_loss':float(new_s_loss.detach()),'d_loss':float(d_loss.detach())}
        # Commit validated live tensors first, then the already validated private Adam state.
        with torch.no_grad():
            for net,values in candidates:
                for k,p in net.named_parameters():
                    p.copy_(values[k])
            alpha.copy_(new_alpha)
        self._alpha_state=staged_alpha_state
        self.steps+=1
        return metrics


class PretrainEngine:
    """Legacy staged SGD or explicit live Adam/trainBN combination (not atomic)."""
    def __init__(self, generator, discriminator, *, g_lr=2e-5, d_lr=2e-5, l1_weight=100., strategy="legacy"):
        require(all(math.isfinite(v) and v>0 for v in (g_lr,d_lr)), 'Positive finite rates required')
        require(math.isfinite(l1_weight) and l1_weight>=0, 'Finite nonnegative L1 weight required')
        self.g,self.d=generator,discriminator
        self.rates={'g':g_lr,'d':d_lr}; self.l1_weight=l1_weight; self.steps=0
        require(strategy in ('legacy','adam_trainbn_epoch_v1'),'Unknown pretraining strategy')
        self.strategy=strategy; self.poisoned=False
        generator.pretrain_strategy=strategy
        if strategy=='adam_trainbn_epoch_v1':
            require(g_lr==d_lr==2e-5 and l1_weight==100.,'Repair uses fixed Adam rates and L1')
            generator.net.train(); discriminator.train()
            self.optimizers={key:torch.optim.Adam(net.parameters(),lr=self.rates[key],
                betas=(.5,.999),eps=1e-8,weight_decay=0.) for key,net in (('g',generator.net),('d',discriminator))}
        else:
            generator.net.eval(); discriminator.eval()
        generator.restore_arch({k:getattr(generator.networks,k) for k in ('conv_arch','upconv_arch')})

    def optimizer_state(self):
        require(not self.poisoned,'Cannot save poisoned live Adam engine')
        if self.strategy!='legacy':
            return {key:{'algorithm':'Adam','lr':self.rates[key],'betas':(.5,.999),'eps':1e-8,
                'weight_decay':0.,'steps':self.steps,'parameter_names':list(dict(net.named_parameters())),
                'state_dict':deepcopy(self.optimizers[key].state_dict())}
                for key,net in (('g',self.g.net),('d',self.d))}
        return {k:{'algorithm':'stateless_SGD','lr':v,'momentum':0.,'weight_decay':0.,
                   'steps':self.steps,'state':{}} for k,v in self.rates.items()}

    def _adam_step(self,real,label):
        # G once; D(fake for G), D(real), D(detached fake). Both gradients use
        # pre-update parameters. BN buffers are live, with G1/D3 per layer.
        try:
            require(all(m.training for n in (self.g.net,self.d) for m in n.modules()),'Repair requires train mode')
            condition=(label==1).to(real.dtype); valid=(label>=0).to(real.dtype)
            before={key:{name:int(m.num_batches_tracked) for name,m in net.named_modules()
                if isinstance(m,torch.nn.BatchNorm2d)} for key,net in (('g',self.g.net),('d',self.d))}
            gp=tuple(self.g.net.parameters()); dp=tuple(self.d.parameters())
            require(all(len(opt.param_groups)==1 and len(opt.param_groups[0]['params'])==len(params) and
                all(a is b for a,b in zip(opt.param_groups[0]['params'],params))
                for opt,params in ((self.optimizers['g'],gp),(self.optimizers['d'],dp))),
                'Adam parameter ownership changed')
            fake=self.g.net(condition)
            g_loss=adversarial(self.d(torch.cat((condition,fake*valid),1)),True)+self.l1_weight*((fake-real).abs()*valid).sum()/valid.sum()
            d_loss=.5*(adversarial(self.d(torch.cat((condition,real*valid),1)),True)+
                       adversarial(self.d(torch.cat((condition,fake.detach()*valid),1)),False))
            require(bool(torch.isfinite(g_loss)) and bool(torch.isfinite(d_loss)),'Nonfinite GAN loss')
            gg=torch.autograd.grad(g_loss,gp); dg=torch.autograd.grad(d_loss,dp)
            require(all(bool(torch.isfinite(v).all()) for v in (*gg,*dg)),'Nonfinite Adam gradient')
            for key,params,grads in (('g',gp,gg),('d',dp,dg)):
                self.optimizers[key].zero_grad(set_to_none=True)
                for param,grad in zip(params,grads): param.grad=grad.detach()
                self.optimizers[key].step()
            for key,net,count in (('g',self.g.net,1),('d',self.d,3)):
                require(all(int(m.num_batches_tracked)-before[key][name]==count
                    for name,m in net.named_modules() if isinstance(m,torch.nn.BatchNorm2d)),
                    'Unexpected per-layer BN update count')
                require(all(bool(torch.isfinite(v).all()) for v in net.state_dict().values()),'Nonfinite live Adam/BN state')
                require(all(bool(torch.isfinite(v).all()) for state in self.optimizers[key].state.values()
                    for v in state.values() if isinstance(v,torch.Tensor)),'Nonfinite Adam moments')
            self.steps+=1
            return {'g_loss':float(g_loss.detach()),'d_loss':float(d_loss.detach()),
                'g_gradient_norm':norm(gg),'d_gradient_norm':norm(dg),'g_bn_forwards':1,'d_bn_forwards':3}
        except BaseException:
            # Some live parameters/buffers/moments may already have changed.
            # No retry, rollback claim, successful checkpoint or continued updates.
            self.poisoned=True
            raise

    def step(self, real, label):
        require(not self.poisoned,'Live Adam engine is poisoned; create a fresh run')
        real=clean(real,label)
        if self.strategy!='legacy': return self._adam_step(real,label)
        condition=(label==1).to(real.dtype); valid=(label>=0).to(real.dtype)
        gf,df=FunctionalModel(self.g.net),FunctionalModel(self.d)
        gp,dp=gf.parameters(),df.parameters()
        fake=gf(gp,condition)
        g_loss=gan_generator_loss(df,dp,condition,fake,real,valid,self.l1_weight)
        d_loss=.5*(adversarial(df(dp,torch.cat((condition,real*valid),1)),True)+
                   adversarial(df(dp,torch.cat((condition,fake.detach()*valid),1)),False))
        require(bool(torch.isfinite(g_loss)) and bool(torch.isfinite(d_loss)), 'Nonfinite GAN loss')
        gg=torch.autograd.grad(g_loss,tuple(gp.values()))
        dg=torch.autograd.grad(d_loss,tuple(dp.values()))
        candidates=[]
        for net,p,grads,key in ((self.g.net,gp,gg,'g'),(self.d,dp,dg,'d')):
            values={k:(v-self.rates[key]*grad).detach() for (k,v),grad in zip(p.items(),grads)}
            require(all(bool(torch.isfinite(v).all()) for v in values.values()), 'Nonfinite SGD candidate; no live update')
            candidates.append((net,values))
        metrics={'g_loss':float(g_loss.detach()),'d_loss':float(d_loss.detach()),
                 'g_gradient_norm':norm(gg),'d_gradient_norm':norm(dg)}
        with torch.no_grad():
            for net,values in candidates:
                for k,p in net.named_parameters(): p.copy_(values[k])
        self.steps+=1
        return metrics


def adversarial(logits, positive):
    return F.binary_cross_entropy_with_logits(logits,torch.ones_like(logits) if positive else torch.zeros_like(logits))


def gan_generator_loss(df,dp,condition,fake,real,valid,l1_weight):
    return adversarial(df(dp,torch.cat((condition,fake*valid),1)),True)+l1_weight*((fake-real).abs()*valid).sum()/valid.sum()
