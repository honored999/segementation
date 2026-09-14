from __future__ import annotations
import json
from types import SimpleNamespace
from uuid import uuid4
import pytest
import torch
from torch import nn
from standalone_nnunet2d.engine.checkpoint import PROJECT_OUTPUTS_DIRECTORY
from standalone_nnunet2d import formal_train
from standalone_nnunet2d.losses.compound import DiceCrossEntropyLoss
from standalone_nnunet2d.models.factory import DEEP_SUPERVISION, H2FORMER, PLAIN_CONV_UNET, SINGLE_OUTPUT
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
 assert resolved['model']['name']=='plain_conv_unet'
 assert resolved['model']['supervision_mode']=='deep_supervision'


def test_batch_size_is_recorded_in_config_and_changes_plan_hash() -> None:
 schedule=OfficialTrainerSchedule(num_iterations_per_epoch=1,num_val_iterations_per_epoch=1)
 default=formal_train.build_formal_config(fold=0,epochs=1,schedule=schedule)
 explicit=formal_train.build_formal_config(fold=0,epochs=1,schedule=schedule,batch_size=4)

 assert default['batch_size']==12
 assert explicit['batch_size']==4
 assert default['plan_hash']!=explicit['plan_hash']


@pytest.mark.parametrize('batch_size',[0,-1])
def test_build_formal_config_rejects_nonpositive_batch_size(batch_size: int) -> None:
 schedule=OfficialTrainerSchedule(num_iterations_per_epoch=1,num_val_iterations_per_epoch=1)

 with pytest.raises(ValueError,match='batch_size'):
  formal_train.build_formal_config(fold=0,epochs=1,schedule=schedule,batch_size=batch_size)


def test_main_passes_explicit_batch_size_to_loader_before_training(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
 seen: list[int] = []
 tiny_model=nn.Conv2d(1,2,1)

 monkeypatch.setattr(formal_train,'load_2d_plan_config',lambda path:((4,4),(False,)))
 monkeypatch.setattr(formal_train,'build_formal_datasets',lambda *args,**kwargs:('train','val'))

 def fake_build_loaders(train,val,*,performance,batch_size):
  del train,val,performance
  seen.append(batch_size)
  return [],[]

 monkeypatch.setattr(formal_train,'build_formal_loaders',fake_build_loaders)
 monkeypatch.setattr(formal_train,'build_model',lambda *args,**kwargs:tiny_model)
 monkeypatch.setattr(formal_train,'run_formal_epochs',lambda **kwargs:iter(()))

 assert formal_train.main([
  '--raw-root',str(tmp_path/'raw'),
  '--output-root',str(tmp_path/'output'),
  '--plans',str(tmp_path/'plans.json'),
  '--device','cpu',
  '--epochs','1',
  '--batch-size','4',
  '--confirm-run',
 ])==0
 assert seen==[4]


def test_main_passes_output_root_to_formal_checkpoint_operations(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
 output_root=tmp_path/'formal-run'
 resume_path=output_root/'checkpoint_latest.pth'
 seen_load_roots: list[object] = []
 seen_save_operations: list[tuple[object,object]] = []
 tiny_model=nn.Conv2d(1,2,1)

 monkeypatch.setattr(formal_train,'load_2d_plan_config',lambda path:((4,4),(False,)))
 monkeypatch.setattr(formal_train,'build_formal_datasets',lambda *args,**kwargs:('train','val'))
 monkeypatch.setattr(formal_train,'build_formal_loaders',lambda *args,**kwargs:([],[]))
 monkeypatch.setattr(formal_train,'build_model',lambda *args,**kwargs:tiny_model)

 def fake_load(*args,**kwargs):
  seen_load_roots.append(kwargs['checkpoint_root'])
  return SimpleNamespace(state=FormalTrainerState(0,0,-1.,0))

 def fake_save(*args,**kwargs):
  seen_save_operations.append((args[3],kwargs['checkpoint_root']))
  return args[3]

 monkeypatch.setattr(formal_train,'load_formal_checkpoint',fake_load)
 monkeypatch.setattr(formal_train,'save_formal_checkpoint',fake_save)
 monkeypatch.setattr(
  formal_train,
  'run_formal_epochs',
  lambda **kwargs: iter(((0,SimpleNamespace(mean_loss=.5),SimpleNamespace(dice=.2),.001),)),
 )

 assert formal_train.main([
  '--raw-root',str(tmp_path/'raw'),
  '--output-root',str(output_root),
  '--plans',str(tmp_path/'plans.json'),
  '--device','cpu',
  '--epochs','1',
  '--resume',str(resume_path),
  '--confirm-run',
 ])==0
 assert seen_load_roots==[output_root]
 assert seen_save_operations==[
  (output_root/'checkpoint_latest.pth',output_root),
  (output_root/'checkpoint_best.pth',output_root),
 ]


def test_stage3_resolved_config_records_all_three_training_contracts() -> None:
 schedule=OfficialTrainerSchedule(num_iterations_per_epoch=1,num_val_iterations_per_epoch=1)
 plain=formal_train.build_formal_config(fold=0,epochs=1,schedule=schedule)
 matched=formal_train.build_formal_config(fold=0,epochs=1,schedule=schedule,model_name=PLAIN_CONV_UNET,supervision_mode=SINGLE_OUTPUT)
 h2=formal_train.build_formal_config(fold=0,epochs=1,schedule=schedule,model_name=H2FORMER)
 assert plain['model']['supervision_mode']==DEEP_SUPERVISION
 assert plain['model']['deep_supervision'] is True
 assert plain['model']['loss_name']=='DeepSupervisionLoss'
 assert matched['model']['supervision_mode']==SINGLE_OUTPUT
 assert matched['model']['deep_supervision'] is False
 assert matched['model']['loss_name']=='DiceCrossEntropyLoss'
 assert h2['model']['supervision_mode']==SINGLE_OUTPUT
 assert h2['model']['deep_supervision'] is False
 assert h2['model']['loss_name']=='DiceCrossEntropyLoss'
 assert len({plain['plan_hash'],matched['plan_hash'],h2['plan_hash']})==3


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
