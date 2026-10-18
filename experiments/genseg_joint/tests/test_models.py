import torch
import pytest
from experiments.genseg_joint.models import create_models
from experiments.genseg_joint.core import JointEngine


def test_actual_pinned_generator_author_unet_same_joint_step(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(48)
    g,d,s=create_models(width=1, downs=5, discriminator_width=1, seed=48)
    # Reduced G/D only. S remains the exact 31M-parameter author UNet.
    engine=JointEngine(g,d,s)
    x=torch.randn(1,1,64,64)*.2
    y=torch.zeros_like(x); y[:,:,20:40,20:40]=1; y[:,:,:3]=-1
    state={k:v.clone() for k,v in s.state_dict().items()}
    alpha=g.networks.upconv_arch.detach().clone()
    unused=g.networks.conv_arch.clone()
    metrics=engine.step(x,y,x*.7,torch.flip(y,[-1]),torch.flip(y,[-2]))
    print('ACTUAL AUTHOR UNET SYNTHETIC METRICS',metrics)
    assert metrics['alpha_gradient_norm']>0 and metrics['alpha_change_norm']>0
    assert not torch.equal(alpha,g.networks.upconv_arch)
    assert torch.equal(unused,g.networks.conv_arch)
    assert any(not torch.equal(v,state[k]) for k,v in s.named_parameters())
    assert all(torch.equal(v,state[k]) for k,v in s.named_buffers())
    assert engine.optimizer_state()['g']['steps']==1
    assert engine.optimizer_state()['alpha']['algorithm']=='Adam'
    assert engine.optimizer_state()['alpha']['state_dict']['state'][0]['step'].item()==1
    from experiments.genseg_joint.runtime import save_checkpoint
    from experiments.genseg_joint.core import stage_alpha_update
    # Small weight placeholders avoid serializing 125MB solely to test actual alpha moments.
    from types import SimpleNamespace
    tiny=torch.nn.Linear(1,1)
    checkpoint_engine=SimpleNamespace(g=SimpleNamespace(net=tiny,config=g.config,networks=g.networks),
        d=tiny,s=tiny,steps=engine.steps,optimizer_state=engine.optimizer_state)
    save_checkpoint(tmp_path,checkpoint_engine,{'test':'actual_author_alpha_default_rates'}, {})
    payload=torch.load(tmp_path/'joint.pt',weights_only=True)
    saved=payload['optimizers']['alpha']
    assert saved['lr']==1e-4 and saved['betas']==(.5,.999) and saved['eps']==1e-8
    assert saved['state_dict']['state'][0]['exp_avg'].abs().sum()>0
    test_grad=torch.ones_like(alpha)*1e-12
    first,state1=stage_alpha_update(g.networks.upconv_arch,test_grad,engine.optimizer_state()['alpha']['state_dict'])
    second,state2=stage_alpha_update(payload['arch']['upconv_arch'],test_grad,saved['state_dict'])
    torch.testing.assert_close(first,second,atol=0,rtol=0)
    assert saved['state_dict']['state'][0]['step'].item()==1
    assert state1['state'][0]['step'].item()==2
    torch.testing.assert_close(state1['state'][0]['exp_avg'],
        .5*saved['state_dict']['state'][0]['exp_avg']+.5*test_grad,atol=1e-20,rtol=1e-6)
    torch.testing.assert_close(state1['state'][0]['exp_avg_sq'],
        .999*saved['state_dict']['state'][0]['exp_avg_sq']+.001*test_grad.square(),atol=1e-32,rtol=1e-5)
    for key in ('step','exp_avg','exp_avg_sq'):
        torch.testing.assert_close(state1['state'][0][key],state2['state'][0][key],atol=0,rtol=0)
    # Fail prior to mutation: all G/D/S, alpha, buffer and optimizer states survive.
    snapshots=[{k:v.clone() for k,v in net.state_dict().items()} for net in (g.net,d,s)]
    a=g.networks.upconv_arch.clone(); opt=engine.optimizer_state()
    with pytest.raises(ValueError,match='Nonfinite'):
        engine.step(x*float('nan'),y,x,y,y)
    assert all(all(torch.equal(v,snap[k]) for k,v in net.state_dict().items()) for net,snap in zip((g.net,d,s),snapshots))
    assert torch.equal(a,g.networks.upconv_arch)
    assert_optimizer_equal(opt,engine.optimizer_state())
    # Reject AFTER alpha Adam staging: no moment/step or G/D/S candidate leaks.
    import experiments.genseg_joint.core as core
    old=core.norm
    def reject_after_staging(tensors):
        result=old(tensors)
        if engine.last_diagnostics.get('alpha_change_norm',0)>0:
            raise RuntimeError('poststage numerical failure')
        return result
    engine.last_diagnostics={}
    monkeypatch.setattr(core,'norm',reject_after_staging)
    with pytest.raises(RuntimeError,match='poststage'):
        engine.step(x,y,x*.7,torch.flip(y,[-1]),torch.flip(y,[-2]))
    assert all(all(torch.equal(v,snap[k]) for k,v in net.state_dict().items()) for net,snap in zip((g.net,d,s),snapshots))
    assert torch.equal(a,g.networks.upconv_arch)
    assert_optimizer_equal(opt,engine.optimizer_state())
    # The Adam adaptation must never bypass the existing zero-change refusal.
    monkeypatch.setattr(core,'norm',old)
    monkeypatch.setattr(core,'stage_alpha_update',lambda alpha,gradient,state:(alpha.detach().clone(),state))
    with pytest.raises(ValueError,match='rounded to zero'):
        engine.step(x,y,x*.7,torch.flip(y,[-1]),torch.flip(y,[-2]))
    assert all(all(torch.equal(v,snap[k]) for k,v in net.state_dict().items()) for net,snap in zip((g.net,d,s),snapshots))
    assert torch.equal(a,g.networks.upconv_arch)
    assert_optimizer_equal(opt,engine.optimizer_state())


def assert_optimizer_equal(left,right):
    if isinstance(left,torch.Tensor):
        assert torch.equal(left,right)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left: assert_optimizer_equal(left[key],right[key])
    elif isinstance(left,(tuple,list)):
        assert len(left)==len(right)
        for a,b in zip(left,right): assert_optimizer_equal(a,b)
    else: assert left==right
