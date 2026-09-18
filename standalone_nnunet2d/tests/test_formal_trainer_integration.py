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

def test_formal_config_records_optimizer_selection_and_policies() -> None:
 schedule=OfficialTrainerSchedule(num_iterations_per_epoch=1,num_val_iterations_per_epoch=1)
 default=formal_train.build_formal_config(fold=0,epochs=1,schedule=schedule)
 h2=formal_train.build_formal_config(fold=0,epochs=1,schedule=schedule,model_name=H2FORMER,optimizer_name='adamw')
 assert default['optimizer']=={'name':'SGD','lr':.01,'momentum':.99,'nesterov':True,'weight_decay':3e-5}
 assert h2['optimizer']=={'name':'AdamW','lr':1e-4,'betas':(.9,.999),'eps':1e-8,'weight_decay':3e-5}
 assert h2['policies']['scheduler']['initial_lr']==pytest.approx(1e-4)
 assert h2['selection']=={'interval_epochs':10,'mirror_axes':[],'aggregation':'case_macro_mean'}
 assert h2['early_stopping']=={'max_epochs':1000,'start_epoch':100,'patience':10,'min_delta':.001}
 assert default['plan_hash']!=h2['plan_hash']

def test_formal_config_rejects_adamw_for_non_h2former() -> None:
 schedule=OfficialTrainerSchedule(num_iterations_per_epoch=1,num_val_iterations_per_epoch=1)
 with pytest.raises(ValueError,match='H2Former'):
  formal_train.build_formal_config(fold=0,epochs=1,schedule=schedule,optimizer_name='adamw')


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
 assert seen_save_operations==[(output_root/'checkpoint_latest.pth',output_root)]


def test_main_skips_training_selection_and_optimizer_update_for_terminal_resume(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
 counters = {"optimizer_steps": 0, "train_loop_calls": 0, "selection_calls": 0}
 output_root=tmp_path/'terminal-resume'
 tiny_model=nn.Conv2d(1,2,1)

 monkeypatch.setattr(formal_train,'load_2d_plan_config',lambda path:((4,4),(False,)))
 monkeypatch.setattr(formal_train,'build_formal_datasets',lambda *args,**kwargs:('train','val'))
 monkeypatch.setattr(formal_train,'build_formal_loaders',lambda *args,**kwargs:([],[]))
 monkeypatch.setattr(formal_train,'build_model',lambda *args,**kwargs:tiny_model)

 def make_counting_optimizer(*args,**kwargs):
  optimizer=torch.optim.SGD(tiny_model.parameters(),lr=.01)
  original_step=optimizer.step
  def step(*step_args,**step_kwargs):
   counters['optimizer_steps']+=1
   return original_step(*step_args,**step_kwargs)
  optimizer.step=step
  return optimizer

 monkeypatch.setattr(formal_train,'make_official_optimizer',make_counting_optimizer)
 monkeypatch.setattr(
  formal_train,
  'load_formal_checkpoint',
  lambda *args,**kwargs: SimpleNamespace(
   state=FormalTrainerState(
    epoch=110,
    global_step=110,
    best_validation_dice=.4,
    fold=0,
    best_selection_dice=.73,
    best_selection_epoch=100,
    early_stop_reference_dice=.72,
    checks_without_improvement=10,
   )
  ),
 )

 def fail_if_training_runs(**kwargs):
  counters['train_loop_calls']+=1
  raise AssertionError('terminal early-stop resume must not enter the training loop')

 def fail_if_selection_runs(*args,**kwargs):
  counters['selection_calls']+=1
  raise AssertionError('terminal early-stop resume must not run selection')

 monkeypatch.setattr(formal_train,'run_formal_epochs',fail_if_training_runs)
 monkeypatch.setattr(formal_train,'select_fold',fail_if_selection_runs)

 assert formal_train.main([
  '--raw-root',str(tmp_path/'raw'),
  '--output-root',str(output_root),
  '--plans',str(tmp_path/'plans.json'),
  '--device','cpu',
  '--epochs','1000',
  '--resume',str(output_root/'checkpoint_latest.pth'),
  '--confirm-run',
 ])==0
 assert counters=={"optimizer_steps":0,"train_loop_calls":0,"selection_calls":0}
 assert (output_root/'training_log.csv').read_text(encoding='utf-8')=='epoch,global_step,train_loss,validation_dice,best_dice,selection_dice,lr\n'


def test_main_reports_non_h2former_adamw_as_argparse_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
 monkeypatch.setattr(formal_train,'load_2d_plan_config',lambda path:((4,4),(False,)))

 with pytest.raises(SystemExit) as error:
  formal_train.main([
   '--raw-root',str(tmp_path/'raw'),
   '--output-root',str(tmp_path/'output'),
   '--plans',str(tmp_path/'plans.json'),
   '--device','cpu',
   '--model',PLAIN_CONV_UNET,
   '--optimizer','adamw',
  ])

 assert error.value.code==2
 captured=capsys.readouterr()
 assert 'AdamW is only supported for H2Former' in captured.err
 assert 'Traceback' not in captured.err


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
