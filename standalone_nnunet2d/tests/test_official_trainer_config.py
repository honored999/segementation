from __future__ import annotations
import dataclasses
import pytest
import torch
from torch import nn
from standalone_nnunet2d.training import official_config
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
 model=nn.Conv2d(1,2,1); optimizer=make_official_optimizer(model); scheduler=PolyLRScheduler(optimizer,.01,1000)
 scheduler.step(500)
 assert optimizer.param_groups[0]['lr']==pytest.approx(.01*(1-.5)**.9)

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
