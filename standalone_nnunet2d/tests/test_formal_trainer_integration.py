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


def test_multichannel_formal_config_is_explicitly_experimental_and_not_aligned() -> None:
 schedule=OfficialTrainerSchedule(num_iterations_per_epoch=1,num_val_iterations_per_epoch=1)
 config=formal_train.build_formal_config(fold=0,epochs=2,schedule=schedule,input_channels=3)
 assert config['input_channels']==3
 assert config['experimental_extension']=='multichannel'
 assert config['run_state']=='official_alignment_pending'


@pytest.mark.parametrize('declared,channels,expected', [((True,),3,(True,True,True)),((False,False),2,(False,False))])
def test_formal_training_expands_single_plan_norm_flag_only_for_dataset_channels(declared,channels,expected) -> None:
 assert formal_train.resolve_use_mask_for_channels(declared,channels)==expected


def test_formal_training_rejects_plan_norm_flags_with_wrong_channel_count() -> None:
 with pytest.raises(ValueError, match='use_mask_for_norm.*3'):
  formal_train.resolve_use_mask_for_channels((True,False),3)


def test_formal_training_resolves_input_channels_before_pending_config(
    tmp_path, capsys
) -> None:
 raw_root=tmp_path/'raw'
 raw_root.mkdir()
 (raw_root/'dataset.json').write_text(json.dumps({'channel_names': {'0':'DWI','1':'ADC','2':'FLAIR'}}),encoding='utf-8')
 result=formal_train.main(['--raw-root',str(raw_root),'--output-root',str(tmp_path/'output'),'--plans',str(formal_train.Path(__file__).resolve().parents[1]/'reference'/'nnUNetPlans.json'),'--device','cpu'])
 assert result==0
 payload=json.loads(capsys.readouterr().out)
 assert payload['config']['input_channels']==3
 assert payload['config']['experimental_extension']=='multichannel'


def test_formal_training_rejects_multichannel_alignment_evidence_before_pending_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
 raw_root=tmp_path/'raw'
 raw_root.mkdir()
 (raw_root/'dataset.json').write_text(json.dumps({'channel_names': {'0':'DWI','1':'ADC'}}),encoding='utf-8')
 monkeypatch.setattr(formal_train, 'resolve_alignment_state', lambda *_: ('official_aligned', {'fixture':'evidence'}))
 with pytest.raises(ValueError, match='multichannel.*official alignment evidence'):
  formal_train.main(['--raw-root',str(raw_root),'--output-root',str(tmp_path/'output'),'--plans',str(formal_train.Path(__file__).resolve().parents[1]/'reference'/'nnUNetPlans.json'),'--device','cpu'])


def test_formal_resume_rejects_checkpoint_input_contract_before_model_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
 raw_root=tmp_path/'raw'
 raw_root.mkdir()
 (raw_root/'dataset.json').write_text(
  json.dumps({'channel_names': {'0':'DWI'}}), encoding='utf-8'
 )
 plans=tmp_path/'plans.json'
 plans.write_text(
  json.dumps({'configurations': {'2d': {'patch_size': [32, 32], 'use_mask_for_norm': [False]}}}),
  encoding='utf-8',
 )
 checkpoint=tmp_path/'resume.pth'
 torch.save(
  {
   'format_version': 1,
   'model_state_dict': {},
   'optimizer_state_dict': None,
   'metadata': {
    'input_channels': 4,
    'resolved_config': {
     'input_mode': 'dwi_adc_bilateral',
     'input_channels': 4,
     'physical_input_channels': 2,
     'effective_model_input_channels': 4,
    },
   },
  },
  checkpoint,
 )
 events: list[str] = []

 monkeypatch.setattr(formal_train, 'build_formal_datasets', lambda *args, **kwargs: ([], []))
 monkeypatch.setattr(formal_train, 'build_formal_loaders', lambda *args, **kwargs: ([], []))

 def model_sentinel(*args: object, **kwargs: object) -> None:
  events.append('model')
  raise AssertionError('model construction must not occur before resume validation')

 def load_sentinel(*args: object, **kwargs: object) -> None:
  events.append('load')
  raise AssertionError('checkpoint load must not occur before resume validation')

 monkeypatch.setattr(formal_train, 'PlainConvUNet2D', model_sentinel)
 monkeypatch.setattr(formal_train, 'load_formal_checkpoint', load_sentinel)

 with pytest.raises(
  ValueError,
  match='runtime input_mode=dwi_bilateral conflicts with checkpoint input_mode=dwi_adc_bilateral',
 ):
  formal_train.main([
   '--raw-root', str(raw_root),
   '--output-root', str(tmp_path/'output'),
   '--plans', str(plans),
   '--device', 'cpu',
   '--input-mode', 'dwi_bilateral',
   '--resume', str(checkpoint),
   '--confirm-run',
  ])

 assert events == []


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
