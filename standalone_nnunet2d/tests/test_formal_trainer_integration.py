from __future__ import annotations
import json
from uuid import uuid4
import pytest
import torch
from torch import nn
from standalone_nnunet2d.engine.checkpoint import PROJECT_OUTPUTS_DIRECTORY
from standalone_nnunet2d import formal_train
from standalone_nnunet2d.losses.compound import DiceCrossEntropyLoss
from standalone_nnunet2d.training.formal_checkpoint import FormalTrainerState, load_formal_checkpoint, save_formal_checkpoint
from standalone_nnunet2d.training.formal_trainer import run_formal_epoch, run_formal_validation
from standalone_nnunet2d.training.official_config import OfficialTrainerSchedule, PolyLRScheduler, make_official_optimizer


def test_formal_trainer_epoch_and_validation_integrate() -> None:
 model=nn.Conv2d(1,2,1); optimizer=make_official_optimizer(model); scheduler=PolyLRScheduler(optimizer,.01,1000); schedule=OfficialTrainerSchedule(num_iterations_per_epoch=1,num_val_iterations_per_epoch=1)
 batch=(torch.randn(1,1,4,4),torch.randint(0,2,(1,4,4))); loss=DiceCrossEntropyLoss()
 train,lr=run_formal_epoch(model,[batch],loss,optimizer,scheduler,torch.device('cpu'),1,schedule)
 validation=run_formal_validation(model,[batch],loss,torch.device('cpu'),schedule)
 assert train.batch_count==1 and validation.batch_count==1 and lr<.01


def test_formal_training_persists_resolved_pending_configuration(tmp_path) -> None:
 schedule=OfficialTrainerSchedule(num_iterations_per_epoch=1,num_val_iterations_per_epoch=1)
 config=formal_train.build_formal_config(fold=2,epochs=4,schedule=schedule)
 path=tmp_path/'resolved_config.json'
 formal_train.write_resolved_config(path,config)
 resolved=json.loads(path.read_text(encoding='utf-8'))
 assert resolved['run_state']=='official_alignment_pending'
 assert resolved['run_state']!='official_aligned'
 assert resolved['plan_hash']==config['plan_hash']
 assert resolved['policies']==config['policies']


def test_formal_trainer_deterministically_continues_after_checkpoint() -> None:
 torch.manual_seed(123)
 model=nn.Sequential(nn.Conv2d(1,2,1),nn.Dropout2d(p=.5))
 optimizer=make_official_optimizer(model)
 scheduler=PolyLRScheduler(optimizer,.01,1000)
 schedule=OfficialTrainerSchedule(num_iterations_per_epoch=1,num_val_iterations_per_epoch=1)
 batch1=(torch.randn(1,1,4,4),torch.randint(0,2,(1,4,4)))
 batch2=(torch.randn(1,1,4,4),torch.randint(0,2,(1,4,4)))
 loss=DiceCrossEntropyLoss()
 run_formal_epoch(model,[batch1],loss,optimizer,scheduler,torch.device('cpu'),0,schedule)
 state=FormalTrainerState(epoch=1,global_step=1,best_validation_dice=.4,fold=0)
 path=PROJECT_OUTPUTS_DIRECTORY/f'pytest-formal-continuation-{uuid4().hex}.pth'
 save_formal_checkpoint(model,optimizer,scheduler,path,state,{'plan':'test'},plan_hash='test-plan',policies={'batch_size':1})
 expected,_=run_formal_epoch(model,[batch2],loss,optimizer,scheduler,torch.device('cpu'),1,schedule)
 expected_parameters=[parameter.detach().clone() for parameter in model.parameters()]

 restored_model=nn.Sequential(nn.Conv2d(1,2,1),nn.Dropout2d(p=.5))
 restored_optimizer=make_official_optimizer(restored_model)
 restored_scheduler=PolyLRScheduler(restored_optimizer,.01,1000)
 restored=load_formal_checkpoint(restored_model,restored_optimizer,restored_scheduler,path,fold=0,plan_hash='test-plan',policies={'batch_size':1})
 actual,_=run_formal_epoch(restored_model,[batch2],loss,restored_optimizer,restored_scheduler,torch.device('cpu'),restored.state.epoch,schedule)

 assert actual.mean_loss==pytest.approx(expected.mean_loss,abs=0.0,rel=0.0)
 assert all(torch.equal(expected_parameter,actual_parameter) for expected_parameter,actual_parameter in zip(expected_parameters,restored_model.parameters()))
