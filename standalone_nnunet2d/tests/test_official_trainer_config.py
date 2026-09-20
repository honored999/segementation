from __future__ import annotations
import dataclasses
import pytest
import torch
from torch import nn
from standalone_nnunet2d.training import official_config
from standalone_nnunet2d.models.factory import H2FORMER, H2FORMER_LITE_UPERNET, PLAIN_CONV_UNET, PLAIN_CONV_UNET_LITE_UPERNET
from standalone_nnunet2d.training.official_config import OfficialTrainerSchedule, PolyLRScheduler, deep_supervision_weights, make_official_optimizer

def test_official_optimizer_matches_inspected_trainer() -> None:
 model=nn.Conv2d(1,2,1); optimizer=make_official_optimizer(model)
 assert optimizer.param_groups[0]['lr']==pytest.approx(.01)
 assert optimizer.param_groups[0]['momentum']==pytest.approx(.99)
 assert optimizer.param_groups[0]['nesterov'] is True
 assert optimizer.param_groups[0]['weight_decay']==pytest.approx(3e-5)

def test_deep_supervision_weights_disable_lowest_scale() -> None:
 weights=deep_supervision_weights(7)
 assert weights[-1]==0 and sum(weights)==pytest.approx(1.)

def test_poly_lr_matches_official_formula() -> None:
 model=nn.Conv2d(1,2,1); optimizer=make_official_optimizer(model); scheduler=PolyLRScheduler(optimizer,1000)
 scheduler.step(500)
 assert optimizer.param_groups[0]['lr']==pytest.approx(.01*(1-.5)**.9)

def test_poly_lr_accepts_named_new_max_steps() -> None:
 model=nn.Conv2d(1,2,1); optimizer=make_official_optimizer(model); scheduler=PolyLRScheduler(optimizer,max_steps=1000)
 scheduler.step(500)
 assert optimizer.param_groups[0]['lr']==pytest.approx(.01*(1-.5)**.9)

def test_poly_lr_accepts_legacy_positional_initial_lr_as_consistency_check() -> None:
 model=nn.Conv2d(1,2,1); optimizer=make_official_optimizer(model); scheduler=PolyLRScheduler(optimizer,.01,1000)
 scheduler.step(500)
 assert optimizer.param_groups[0]['lr']==pytest.approx(.01*(1-.5)**.9)

def test_poly_lr_accepts_legacy_named_initial_lr_as_consistency_check() -> None:
 model=nn.Conv2d(1,2,1); optimizer=make_official_optimizer(model)
 scheduler=PolyLRScheduler(optimizer,initial_lr=.01,max_steps=1000)
 scheduler.step(500)
 assert optimizer.param_groups[0]['lr']==pytest.approx(.01*(1-.5)**.9)

def test_poly_lr_rejects_mismatched_legacy_initial_lr() -> None:
 model=nn.Conv2d(1,2,1); optimizer=make_official_optimizer(model)
 with pytest.raises(ValueError,match='initial_lr.*optimizer'):
  PolyLRScheduler(optimizer,initial_lr=.02,max_steps=1000)

def test_adamw_legacy_initial_lr_mismatch_is_not_silently_ignored() -> None:
 model=nn.Conv2d(1,2,1)
 optimizer=make_official_optimizer(model,model_name=H2FORMER,optimizer_name='adamw')
 with pytest.raises(ValueError,match='initial_lr.*optimizer'):
  PolyLRScheduler(optimizer,.01,1000)

@pytest.mark.parametrize('max_steps',[0,-1,1.5])
def test_poly_lr_rejects_invalid_max_steps(max_steps: float) -> None:
 model=nn.Conv2d(1,2,1); optimizer=make_official_optimizer(model)
 with pytest.raises(ValueError,match='max_steps'):
  PolyLRScheduler(optimizer,max_steps=max_steps)

def test_poly_lr_captures_each_optimizer_group_initial_lr() -> None:
 model=nn.Sequential(nn.Conv2d(1,2,1),nn.Conv2d(2,2,1))
 optimizer=torch.optim.SGD([
  {'params':model[0].parameters(),'lr':.01},
  {'params':model[1].parameters(),'lr':.001},
 ],momentum=.99,nesterov=True,weight_decay=3e-5)
 scheduler=PolyLRScheduler(optimizer,1000)
 scheduler.step(500)
 assert [group['lr'] for group in optimizer.param_groups]==pytest.approx([
  .01*(1-.5)**.9,.001*(1-.5)**.9,
 ])

@pytest.mark.parametrize("model_name", [H2FORMER, H2FORMER_LITE_UPERNET])
def test_adamw_preset_is_exactly_h2former_family(model_name: str) -> None:
 model=nn.Conv2d(1,2,1)
 optimizer=make_official_optimizer(model,model_name=model_name,optimizer_name='adamw')
 group=optimizer.param_groups[0]
 assert isinstance(optimizer,torch.optim.AdamW)
 assert group['lr']==pytest.approx(1e-4)
 assert group['betas']==(.9,.999)
 assert group['eps']==pytest.approx(1e-8)
 assert group['weight_decay']==pytest.approx(3e-5)
 scheduler=PolyLRScheduler(optimizer,1000)
 scheduler.step(0)
 assert optimizer.param_groups[0]['lr']==pytest.approx(1e-4)

@pytest.mark.parametrize("model_name", [PLAIN_CONV_UNET, PLAIN_CONV_UNET_LITE_UPERNET])
def test_adamw_rejected_for_plain_model_family(model_name: str) -> None:
 model=nn.Conv2d(1,2,1)
 with pytest.raises(ValueError,match='H2Former'):
  make_official_optimizer(model,model_name=model_name,optimizer_name='adamw')

def test_official_schedule_matches_inspected_defaults() -> None:
 schedule=OfficialTrainerSchedule()
 assert (schedule.num_epochs,schedule.num_iterations_per_epoch,schedule.num_val_iterations_per_epoch,schedule.oversample_foreground_percent)==(1000,250,50,.33)

def test_policy_records_argmax_tta_and_case_macro_contract() -> None:
 policy = official_config.OfficialInferencePolicy()

 assert policy.postprocessing == "argmax"
 assert policy.mirror_axes == (0, 1)
 assert policy.tile_step_size == pytest.approx(0.5)
 assert policy.aggregation == "case_macro_mean"

 with pytest.raises(dataclasses.FrozenInstanceError):
  policy.postprocessing = "sigmoid"
