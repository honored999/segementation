"""Tiny CPU engineering evidence, never real-data or performance evidence."""
from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

EXT = Path(__file__).resolve().parents[1]
ROOT = EXT.parent
sys.path.insert(0, str(EXT)) if str(EXT) not in sys.path else None
torch.set_num_threads(1)
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnUNetTrainerMixins import UPerNetDecoder, select_upernet_feature_indices, validate_upernet_feature_indices
from nnUNetTrainerUPerNetSelectedStagesTopK10EarlyStopping import nnUNetTrainerUPerNetSelectedStagesTopK10EarlyStopping as Trainer
from nnUNetTrainerUPerNetTopK10EarlyStopping import nnUNetTrainerUPerNetTopK10EarlyStopping as OldTrainer
from create_upernet_stage_plans import create_upernet_stage_plans

AUTO = object()

def plans(indices=AUTO, *, inherited=False, n=8, strides=None):
    # Defaults are metadata-only; callers supply valid geometry for actual models.
    strides = strides or ([[1, 1]] + [[2, 2]]*2 + [[1, 1]]*(n-3))
    cfg = {
        'data_identifier': 'OriginalPlans_2d', 'patch_size': [16, 16], 'batch_size': 1,
        'batch_dice': True, 'spacing': [1,1], 'normalization_schemes': ['ZScoreNormalization'],
        'use_mask_for_norm': [False], 'preprocessor_name': 'DefaultPreprocessor',
        'architecture': {'network_class_name': 'dynamic_network_architectures.architectures.unet.PlainConvUNet',
            'arch_kwargs': {'n_stages': n, 'features_per_stage': [2]*n,
                'conv_op': 'torch.nn.Conv2d', 'kernel_sizes': [[3,3]]*n, 'strides': strides,
                'n_conv_per_stage': [1]*n, 'n_conv_per_stage_decoder': [1]*(n-1),
                'conv_bias': False, 'norm_op': 'torch.nn.InstanceNorm2d',
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True}, 'dropout_op': None,
                'dropout_op_kwargs': None, 'nonlin': 'torch.nn.ReLU', 'nonlin_kwargs': {'inplace': True}},
            '_kw_requires_import': ['conv_op', 'norm_op', 'dropout_op', 'nonlin']}}
    # Eight-stage stride-2 geometry is checked separately without allocating full-size tensors.
    if indices is not AUTO:
        cfg['upernet_feature_indices'] = indices
    configurations = {'2d': cfg}
    if inherited:
        configurations = {'base': cfg, '2d': {'inherits_from': 'base'}}
    return {'plans_name': 'OriginalPlans', 'dataset_name': 'Dataset999_Synthetic',
            'transpose_forward': [0,1,2], 'transpose_backward': [0,1,2],
            'image_reader_writer': 'SimpleITKIO', 'configurations': configurations}


def build(indices=AUTO, *, n=4, strides=None, old=False):
    # Official small-channel model; no real plans or medical data.
    data = plans(indices, n=n, strides=strides or [[1,1]]+[[2,2]]*(n-1))
    manager = PlansManager(data)
    cfg = manager.get_configuration('2d')
    return (OldTrainer if old else Trainer).build_network_architecture(manager, cfg, 1, 3, False)


@pytest.mark.parametrize('indices', [[1,3,5,7], [1,3,5,6], [2,3,4,5], [0,7], list(range(8))])
def test_indices_validate_eight_stage_geometry(indices):
    assert validate_upernet_feature_indices(indices, [[1,1]]+[[2,2]]*7, 8) == tuple(indices)
    manager = PlansManager(plans(indices, strides=[[1,1]]+[[2,2]]*7))
    # Build small channels only; no full-size eight-stage forward.
    model = Trainer.build_network_architecture(manager, manager.get_configuration('2d'), 1, 3, False)
    assert model.selected_feature_indices == tuple(indices)
    assert len(model.encoder.stages) == 8
    assert model.decoder.fusion[0].in_channels == len(indices)*128


@pytest.mark.parametrize('indices', [[], [1], [True,2], [1,2.0], ['1',2], [None,2], None,
                                      [1,1], [2,1], [-1,2], [0,8], {'x':1}, [0,1,2,3,4,5,6,7,8]])
def test_invalid_indices_rejected_before_official_allocation(indices, monkeypatch):
    import nnUNetTrainerUPerNetSelectedStagesTopK10EarlyStopping as module
    called = []
    monkeypatch.setattr(module, 'get_network_from_plans', lambda *a, **k: called.append(True))
    manager = PlansManager(plans(indices, strides=[[1,1]]+[[2,2]]*7))
    with pytest.raises(ValueError):
        Trainer.build_network_architecture(manager, manager.get_configuration('2d'), 1, 3, False)
    assert not called


@pytest.mark.parametrize('strides', [[[1,1],[1,1],[2,2]], [[1,1],[2,1],[2,2]]])
def test_equal_or_one_axis_scale_rejected(strides):
    with pytest.raises(ValueError, match='strict'):
        validate_upernet_feature_indices([0,1], strides, 3)


def test_default_and_inheritance_and_old_ignores_key():
    data = plans([0,2,3], n=4, strides=[[1,1]]+[[2,2]]*3, inherited=True)
    manager = PlansManager(data)
    model = Trainer.build_network_architecture(manager, manager.get_configuration('2d'), 1, 3, False)
    assert model.selected_feature_indices == (0,2,3)
    assert 'upernet_feature_indices' not in manager.get_configuration('2d').network_arch_init_kwargs
    auto = plans(strides=[[1,1]]+[[2,2]]*7)
    manager = PlansManager(auto)
    default = Trainer.build_network_architecture(manager, manager.get_configuration('2d'), 1, 3, False)
    assert default.selected_feature_indices == (1,3,5,7)
    manager = PlansManager(plans([0,1], strides=[[1,1]]+[[2,2]]*7))
    old = OldTrainer.build_network_architecture(manager, manager.get_configuration('2d'), 1, 3, False)
    assert old.selected_feature_indices == (1,3,5,7)
    assert '_extra_state' not in old.state_dict()


@pytest.mark.parametrize('indices,size,strides', [([0,3],(17,25),[[1,1],[2,2],[2,1],[2,2]]),
                                                 ([0,1,2,3],(17,25),[[1,1]]+[[2,2]]*3)])
def test_tiny_forward_backward_full_encoder_and_accounting(indices,size,strides):
    model = build(indices, strides=strides).train()
    counts = {'encoder':0,'decoder':0}
    stages = []
    chosen = []
    handles = []
    def counter(group):
        def hook(module, inp, output):
            counts[group] += output.numel()
        return hook
    for group in counts:
        for module in getattr(model, group).modules():
            if isinstance(module, nn.Conv2d):
                handles.append(module.register_forward_hook(counter(group)))
    for i, stage in enumerate(model.encoder.stages):
        handles.append(stage.register_forward_hook(lambda m,x,y,i=i: stages.append(i)))
    def capture(m, inp):
        chosen.extend(inp[0])
        for feature in inp[0]:
            feature.retain_grad()
    handles.append(model.decoder.register_forward_pre_hook(capture))
    image = torch.randn(1,1,*size, requires_grad=True)
    try:
        logits = model(image)
        assert logits.shape == (1,3,*size)
        assert torch.isfinite(logits).all()
        nn.CrossEntropyLoss()(logits, torch.randint(0,3,(1,*size))).backward()
        assert stages == list(range(4))
        assert all(f.grad is not None and torch.isfinite(f.grad).all() and f.grad.abs().sum()>0 for f in chosen)
        assert torch.isfinite(image.grad).all()
        assert counts['decoder'] == int(model.decoder.compute_conv_feature_map_size(size))
        assert sum(counts.values()) == int(model.compute_conv_feature_map_size(size))
    finally:
        for h in handles: h.remove()


def test_unselected_last_stage_still_executes():
    model = build([0,2]).eval()
    called = []
    handle = model.encoder.stages[3].register_forward_hook(lambda *args: called.append(True))
    with torch.inference_mode():
        model(torch.randn(1,1,17,17))
    handle.remove()
    assert called == [True]


@pytest.mark.parametrize('n', [2,4,8])
def test_n_level_decoder_tiny_backward_accounting(n):
    # Tiny direct decoder uses genuinely decreasing adjacent integer sizes 10..3,
    # and actual stride-1 valid 3x3 convolution geometry; avoids full-size eight-stage fixtures.
    geometry = tuple((((3,3),(1,1),(0,0),(1,1)),) for _ in range(n))
    sizes = [2*(n-i)+3 for i in range(n)]
    decoder = UPerNetDecoder([2]*n,3,lambda c: nn.InstanceNorm2d(c,affine=True), conv_bias=False,
                            encoder_spatial_geometry=geometry, selected_feature_indices=tuple(range(n)))
    features = [torch.randn(1,2,s,s, requires_grad=True) for s in sizes]
    count = []
    handles = [m.register_forward_hook(lambda m,x,y: count.append(y.numel())) for m in decoder.modules() if isinstance(m,nn.Conv2d)]
    logits = decoder(features,output_size=(sizes[0]+2,sizes[0]+2))
    logits.square().mean().backward()
    for h in handles: h.remove()
    assert all(f.grad is not None and torch.isfinite(f.grad).all() and f.grad.abs().sum()>0 for f in features)
    assert sum(count) == int(decoder.compute_conv_feature_map_size((sizes[0]+2,sizes[0]+2)))
    assert len(decoder.lateral_projections) == n-1


@pytest.mark.parametrize('bad', ['count','channels','batch','ndim','spatial'])
def test_decoder_actual_tensor_validation(bad):
    decoder = UPerNetDecoder([2,2],3,lambda c: nn.Identity(),conv_bias=False)
    features = [torch.randn(1,2,9,9),torch.randn(1,2,5,5)]
    if bad=='count': features.pop()
    if bad=='channels': features[1]=torch.randn(1,3,5,5)
    if bad=='batch': features[1]=torch.randn(2,2,5,5)
    if bad=='ndim': features[1]=torch.randn(2,5,5)
    if bad=='spatial': features[1]=torch.randn(1,2,9,5)
    with pytest.raises(ValueError): decoder(features,output_size=(9,9))


def test_four_level_frozen_baseline_keys_shapes_values_and_count():
    original = subprocess.run(['git','show','8a41ce3:nnunet_ext_trainers/nnUNetTrainerMixins.py'], cwd=ROOT,
                              capture_output=True,check=True,encoding='utf-8').stdout
    baseline = ModuleType('frozen_baseline')
    exec(compile(original,'frozen_baseline','exec'),baseline.__dict__)
    manager = PlansManager(plans([1,2,3,4], n=5, strides=[[1,1]]+[[2,2]]*4))
    cfg = manager.get_configuration('2d')
    torch.manual_seed(31)
    old = baseline.UPerNetArchitectureMixin.build_network_architecture(manager,cfg,1,3,False).eval()
    torch.manual_seed(31)
    current = OldTrainer.build_network_architecture(manager,cfg,1,3,False).eval()
    torch.manual_seed(31)
    new = Trainer.build_network_architecture(manager,cfg,1,3,False).eval()
    old_state = old.state_dict()
    assert list(old_state) == list(current.state_dict())
    assert list(old_state) == [k for k in new.state_dict() if k != '_extra_state']
    assert all(torch.equal(v,current.state_dict()[k]) and torch.equal(v,new.state_dict()[k]) for k,v in old_state.items())
    image = torch.randn(1,1,32,32)
    with torch.inference_mode():
        assert torch.equal(old(image), current(image))
        assert torch.equal(old(image), new(image))
    assert old.compute_conv_feature_map_size((32,32)) == current.compute_conv_feature_map_size((32,32)) == new.compute_conv_feature_map_size((32,32))


DATASET = {'channel_names': {'0':'synthetic'}, 'labels': {'background':0,'a':1,'b':2}, 'file_ending':'.nii.gz'}

class Logger:
    def __init__(self): self.loaded=None
    def get_checkpoint(self): return {'synthetic':True}
    def load_checkpoint(self,state): self.loaded=state


def fake_trainer(model):
    trainer = object.__new__(Trainer)
    trainer.network = model
    trainer.optimizer = torch.optim.SGD(model.parameters(),lr=.01,momentum=.99)
    trainer.grad_scaler = None
    trainer.logger = Logger()
    trainer.was_initialized = True
    trainer.is_ddp = False
    trainer.device = torch.device('cpu')
    trainer.local_rank = 0
    trainer.disable_checkpointing = False
    trainer.current_epoch = 20
    trainer.my_init_kwargs = {'configuration':'2d'}
    trainer._best_ema = .4
    trainer.inference_allowed_mirroring_axes = (0,1)
    trainer._early_stopping_best_ema = .5
    trainer._epochs_without_improvement = 7
    trainer._early_stopping_triggered = False
    return trainer


def checkpoint(model):
    t = fake_trainer(model)
    return {'network_weights': deepcopy(model.state_dict()), 'optimizer_state':t.optimizer.state_dict(),
            'grad_scaler_state':None, 'logging': {'synthetic':True}, '_best_ema':.4,'current_epoch':21,
            'init_args': {'configuration':'2d'}, 'trainer_name':Trainer.__name__,
            'inference_allowed_mirroring_axes': (0,1), 'early_stopping_state':t._early_stopping_checkpoint_state()}


def tensor_state(model):
    return {k:v.clone() for k,v in model.state_dict().items() if isinstance(v,torch.Tensor)}


@pytest.mark.parametrize('mutation',['other','missing','corrupt','bool'])
def test_identity_rejected_before_parameters_optimizer_and_logger(mutation):
    model = build([0,2])
    before = tensor_state(model)
    state = deepcopy(model.state_dict())
    if mutation=='other':
        state = build([1,3]).state_dict()
        assert {k:tuple(v.shape) for k,v in state.items() if isinstance(v,torch.Tensor)} == {k:tuple(v.shape) for k,v in before.items()}
    if mutation=='missing': del state['_extra_state']
    if mutation=='corrupt': state['_extra_state']['fpn_channels'] = 64
    if mutation=='bool': state['_extra_state']['indices'][0] = False
    for k,v in state.items():
        if isinstance(v,torch.Tensor): state[k] = torch.full_like(v,9)
    with pytest.raises(ValueError,match='identity'): model.load_state_dict(state)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    t = fake_trainer(model)
    cp = checkpoint(model)
    cp['network_weights']=state
    cp['optimizer_state']={'invalid':True}
    with pytest.raises(ValueError,match='identity'): t.load_checkpoint(cp)
    assert t.current_epoch == 20 and t.logger.loaded is None
    assert t.optimizer.param_groups[0]['lr']==.01
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())


def test_round_trip_file_training_and_val_and_module_prefix(tmp_path):
    original = build([0,2])
    trainer = fake_trainer(original)
    trainer.save_checkpoint(str(tmp_path/'checkpoint.pth'))
    rebuilt = build([0,2])
    receiver = fake_trainer(rebuilt)
    receiver.load_checkpoint(str(tmp_path/'checkpoint.pth'))
    assert receiver.current_epoch == 21 and receiver._epochs_without_improvement==7
    assert all(torch.equal(v,rebuilt.state_dict()[k]) for k,v in tensor_state(original).items())
    cp = torch.load(tmp_path/'checkpoint.pth',weights_only=False)
    cp['network_weights'] = {'module.'+k:v for k,v in cp['network_weights'].items()}
    receiver.load_checkpoint(cp)
    receiver.set_deep_supervision_enabled(True)
    assert rebuilt.decoder.deep_supervision is False


def test_real_predictor_rebuild_and_multifold_weights(tmp_path, monkeypatch):
    monkeypatch.setenv('nnUNet_extTrainer',str(EXT))
    monkeypatch.setenv('nnUNet_compile','false')
    data = plans([0,2],n=4,strides=[[1,1]]+[[2,2]]*3)
    data['plans_name']='nnUNetPlansUPerNetStages_s02'
    (tmp_path/'plans.json').write_text(json.dumps(data),encoding='utf-8')
    (tmp_path/'dataset.json').write_text(json.dumps(DATASET),encoding='utf-8')
    model = build([0,2])
    for fold in (0,1):
        folder = tmp_path/f'fold_{fold}'
        folder.mkdir()
        torch.save(checkpoint(model),folder/'checkpoint_final.pth')
    predictor = nnUNetPredictor(device=torch.device('cpu'),perform_everything_on_device=False,use_mirroring=False,verbose=False)
    predictor.initialize_from_trained_model_folder(str(tmp_path),(0,1))
    assert predictor.network.selected_feature_indices == (0,2)
    # Real official fold loop and sliding-window logic with tiny CPU input.
    output = predictor.predict_logits_from_preprocessed_data(torch.randn(1,1,16,16))
    assert output.shape==(3,1,16,16) and torch.isfinite(output).all()
    before = tensor_state(predictor.network)
    torch.save(checkpoint(build([1,3])),tmp_path/'fold_1/checkpoint_final.pth')
    predictor.initialize_from_trained_model_folder(str(tmp_path),(0,1))
    with pytest.raises(ValueError,match='identity'):
        predictor.predict_logits_from_preprocessed_data(torch.randn(1,1,16,16))
    assert all(torch.equal(v,predictor.network.state_dict()[k]) for k,v in before.items())
    torch.save(checkpoint(build([1,3])),tmp_path/'fold_0/checkpoint_final.pth')
    with pytest.raises(ValueError,match='identity'):
        predictor.initialize_from_trained_model_folder(str(tmp_path),(0,))


@pytest.mark.parametrize('inherited',[False,True])
def test_plan_copy_roundtrip_and_cli(tmp_path,inherited):
    source = tmp_path/'original.json'
    data = plans(inherited=inherited,strides=[[1,1]]+[[2,2]]*7)
    source.write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
    before = source.read_bytes()
    output = tmp_path/'nnUNetPlansUPerNetStages_s1356.json'
    create_upernet_stage_plans(source,output,'2d',[1,3,5,6])
    result = json.loads(output.read_text(encoding='utf-8'))
    expected = deepcopy(data)
    expected['plans_name']=output.stem
    expected['configurations']['2d']['upernet_feature_indices']=[1,3,5,6]
    assert result==expected and source.read_bytes()==before
    assert PlansManager(result).get_configuration('2d').data_identifier=='OriginalPlans_2d'
    other = tmp_path/'nnUNetPlansUPerNetStages_s07.json'
    run = subprocess.run([sys.executable,str(EXT/'create_upernet_stage_plans.py'),'--source',str(source),
                          '--output',str(other),'--feature-indices','0','7'],capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr
    assert json.loads(other.read_text(encoding='utf-8'))['configurations']['2d']['upernet_feature_indices']==[0,7]


@pytest.mark.parametrize('problem',['same','exists','name','configuration','invalid','architecture','nested_key'])
def test_plan_copy_refuses_and_preserves_source(tmp_path,problem):
    data = plans(strides=[[1,1]]+[[2,2]]*7)
    if problem=='architecture': data['configurations']['2d']['architecture']['network_class_name']='other'
    if problem=='nested_key': data['configurations']['2d']['architecture']['arch_kwargs']['upernet_feature_indices']=[0,7]
    source=tmp_path/'original.json'
    source.write_text(json.dumps(data),encoding='utf-8')
    before=source.read_bytes()
    output=tmp_path/'nnUNetPlansUPerNetStages_s07.json'
    if problem=='same': output=source
    if problem=='exists': output.write_text('existing',encoding='utf-8')
    if problem=='name': output=tmp_path/'wrong.json'
    with pytest.raises((ValueError,FileExistsError)):
        create_upernet_stage_plans(source,output,'missing' if problem=='configuration' else '2d',
                                  [7,0] if problem=='invalid' else [0,7])
    assert source.read_bytes()==before
    if problem=='exists': assert output.read_text(encoding='utf-8')=='existing'
    elif problem!='same': assert not output.exists()


def test_fresh_official_discovery():
    env = os.environ.copy()
    env['nnUNet_extTrainer']=str(EXT)
    code = """import inspect
from importlib.metadata import version
from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
assert version('nnunetv2') == '2.8.1'
cls = recursive_find_trainer_class_by_name('nnUNetTrainerUPerNetSelectedStagesTopK10EarlyStopping')
assert issubclass(cls,nnUNetTrainer)
assert tuple(inspect.signature(cls.build_network_architecture).parameters) == ('plans_manager','configuration_manager','num_input_channels','num_output_channels','enable_deep_supervision')
print('FRESH_SELECTED_STAGES_DISCOVERY_OK')
"""
    result = subprocess.run([sys.executable,'-c',code],env=env,cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'FRESH_SELECTED_STAGES_DISCOVERY_OK' in result.stdout



def test_eight_stage_tiny_full_network_backward_and_count():
    # 129 is the smallest odd square yielding >1 spatial element at final stride-128.
    # Two encoder channels; FPN128 and the complete eight-stage execution are preserved.
    model = build(list(range(8)),n=8).train()
    observed=[]
    handles=[m.register_forward_hook(lambda m,x,y: observed.append(y.numel())) for m in model.modules() if isinstance(m,nn.Conv2d)]
    image=torch.randn(1,1,129,129,requires_grad=True)
    logits=model(image)
    logits.square().mean().backward()
    for h in handles: h.remove()
    assert logits.shape==(1,3,129,129) and torch.isfinite(logits).all()
    assert torch.isfinite(image.grad).all() and image.grad.abs().sum()>0
    assert sum(observed)==int(model.compute_conv_feature_map_size((129,129)))


def test_actual_cpu_trainer_initialize_resume_and_validation_rebuild(tmp_path,monkeypatch):
    monkeypatch.setenv('nnUNet_results',str(tmp_path/'results'))
    monkeypatch.setenv('nnUNet_preprocessed',str(tmp_path/'preprocessed'))
    monkeypatch.setenv('nnUNet_compile','false')
    monkeypatch.setenv('nnUNet_wandb_enabled','0')
    monkeypatch.setenv('nnUNet_mlflow_enabled','0')
    # Tiny synthetic npz supplies the format sentinel for official dataset-class detection.
    folder=tmp_path/'preprocessed/Dataset999_Synthetic/OriginalPlans_2d'
    folder.mkdir(parents=True)
    np.savez_compressed(folder/'synthetic.npz', data=np.zeros((1,1,16,16),dtype=np.float32))
    data=plans([0,2],n=4,strides=[[1,1]]+[[2,2]]*3)
    data['plans_name']='nnUNetPlansUPerNetStages_s02'
    data['continue_training']=False
    original=Trainer(deepcopy(data),'2d',0,DATASET,torch.device('cpu'))
    original.initialize()
    assert original.enable_deep_supervision is False
    assert original.optimizer.param_groups[0]['lr']==.01
    assert original.initial_lr==.01 and original.weight_decay==3e-5
    assert original.oversample_foreground_percent==.33 and original.num_epochs==1000
    original.current_epoch=320
    original._early_stopping_best_ema=.7
    original._epochs_without_improvement=11
    cp_path=tmp_path/'actual_checkpoint.pth'
    original.save_checkpoint(str(cp_path))
    resumed=Trainer(deepcopy(data),'2d',0,DATASET,torch.device('cpu'))
    assert not resumed.was_initialized
    resumed.load_checkpoint(str(cp_path))
    assert resumed.was_initialized and resumed.current_epoch==321
    assert resumed._epochs_without_improvement==11
    assert all(torch.equal(v,resumed.network.state_dict()[k]) for k,v in tensor_state(original.network).items())
    # --val also reconstructs via Trainer and inherited load, then toggles single-output lifecycle.
    resumed.set_deep_supervision_enabled(False)
    assert resumed.network.decoder.deep_supervision is False
    log=Path(resumed.log_file).read_text(encoding='utf-8')
    assert 'effective selected-stage identity' in log and '"indices": [0, 2]' in log


def test_compiled_wrapper_and_nested_root_identity_guards(tmp_path):
    from torch._dynamo import OptimizedModule
    model=build([0,2])
    wrapped=torch.compile(model,backend='eager')
    assert isinstance(wrapped,OptimizedModule)
    trainer=fake_trainer(wrapped)
    trainer.save_checkpoint(str(tmp_path/'compiled.pth'))
    trainer.load_checkpoint(str(tmp_path/'compiled.pth'))
    cp=torch.load(tmp_path/'compiled.pth',weights_only=False)
    cp['network_weights']=build([1,3]).state_dict()
    with pytest.raises(ValueError,match='identity'): trainer.load_checkpoint(cp)
    parent=nn.Sequential(model)
    before=tensor_state(model)
    invalid={'0.'+k:v for k,v in build([1,3]).state_dict().items()}
    with pytest.raises(ValueError,match='identity'): parent.load_state_dict(invalid)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())


@pytest.mark.parametrize('bad_source',['duplicate','missing_name','target_not_dict'])
def test_source_plans_ambiguity_rejected(tmp_path,bad_source):
    source=tmp_path/'original.json'
    data=plans(strides=[[1,1]]+[[2,2]]*7)
    if bad_source=='missing_name': del data['plans_name']
    if bad_source=='target_not_dict': data['configurations']['2d']=[]
    text=json.dumps(data)
    if bad_source=='duplicate': text=text.replace('"plans_name": "OriginalPlans"','"plans_name": "OriginalPlans", "plans_name": "OtherPlans"')
    source.write_text(text,encoding='utf-8')
    output=tmp_path/'nnUNetPlansUPerNetStages_s07.json'
    with pytest.raises(ValueError): create_upernet_stage_plans(source,output,'2d',[0,7])
    assert not output.exists() and source.read_text(encoding='utf-8')==text



@pytest.fixture
def cpu_gloo_group():
    """ASCII FileStore avoids this Windows build's Unicode rendezvous failure."""
    from datetime import timedelta
    import tempfile
    import torch.distributed as dist
    assert not dist.is_initialized(), 'test must not join or disturb an existing group'
    with tempfile.TemporaryDirectory(prefix='upernet-ddp-') as directory:
        rendezvous = str(Path(directory)/'rendezvous')
        assert rendezvous.isascii(), 'CPU Gloo test requires an ASCII temporary root'
        store = dist.FileStore(rendezvous, 1)
        dist.init_process_group('gloo', store=store, rank=0, world_size=1,
                                timeout=timedelta(seconds=20))
        try:
            yield dist.group.WORLD
        finally:
            dist.destroy_process_group()
            del store


def cpu_ddp_initialize(trainer, monkeypatch, *, compiled):
    """Replace only CUDA-dependent base initialization with its CPU DDP equivalent.

    Exercise the real new Trainer.initialize integration after the official order
    (compile then DDP). The initializer's parent lookup is patched, never the fix.
    Real CUDA/SyncBatchNorm/multi-rank behavior is separately deferred.
    """
    import weakref
    created = []
    def base_initialize(self):
        network = self.network
        if compiled:
            network = torch.compile(network, backend='eager')
        self.network = nn.parallel.DistributedDataParallel(network)
        created.append(weakref.ref(self.network))
        self.was_initialized = True
    monkeypatch.setattr(OldTrainer, 'initialize', base_initialize)
    trainer.was_initialized = False
    trainer.is_ddp = True
    trainer.print_to_log_file = lambda *a, **k: None
    trainer.initialize()
    return created[0]


@pytest.mark.parametrize('compiled', [False, True], ids=['plain','eager-compiled'])
@pytest.mark.parametrize('indices', [[0,3], [0,2]], ids=['last-stage-control','unused-trailing'])
def test_selected_trainer_ddp_two_iterations(cpu_gloo_group, monkeypatch, indices, compiled):
    from torch._dynamo import OptimizedModule
    model = build(indices)
    trainer = fake_trainer(model)
    optimizer = trainer.optimizer
    parameter_ids = [id(p) for p in model.parameters()]
    created_wrapper = cpu_ddp_initialize(trainer, monkeypatch, compiled=compiled)
    wrapper = trainer.network
    assert isinstance(wrapper, nn.parallel.DistributedDataParallel)
    assert wrapper.process_group is cpu_gloo_group
    assert isinstance(wrapper.module, OptimizedModule) == compiled
    # Preserve the last-stage default DDP path. Do not fake grads on unused stages.
    called = []
    handle = model.encoder.stages[-1].register_forward_hook(lambda *a: called.append(True))
    try:
        for _ in range(2):
            trainer.optimizer.zero_grad(set_to_none=True)
            output = wrapper(torch.randn(1,1,17,17))
            output.square().mean().backward()
            assert output.shape == (1,3,17,17) and torch.isfinite(output).all()
        assert called == [True,True]
        assert wrapper.find_unused_parameters == (indices[-1] < 3)
        assert (created_wrapper() is wrapper) == (indices[-1] == 3)
        if indices[-1] < 3:
            assert created_wrapper() is None
        assert trainer.optimizer is optimizer
        assert [id(p) for p in wrapper.parameters()] == parameter_ids
        for index, stage in enumerate(model.encoder.stages):
            for parameter in stage.parameters():
                if index > indices[-1]:
                    assert parameter.grad is None
                else:
                    assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    finally:
        handle.remove()


@pytest.mark.parametrize('compiled', [False,True], ids=['plain','eager-compiled'])
def test_selected_trainer_real_ddp_checkpoint_identity(cpu_gloo_group, monkeypatch, tmp_path, compiled):
    original = build([0,2])
    sender = fake_trainer(original)
    cpu_ddp_initialize(sender, monkeypatch, compiled=compiled)
    destination = tmp_path/'ddp.pth'
    sender.save_checkpoint(str(destination))
    rebuilt = build([0,2])
    receiver = fake_trainer(rebuilt)
    cpu_ddp_initialize(receiver, monkeypatch, compiled=compiled)
    receiver.load_checkpoint(str(destination))
    assert all(torch.equal(v,rebuilt.state_dict()[k]) for k,v in tensor_state(original).items())
    cp = torch.load(destination,weights_only=False)
    # The official loader accepts module-prefixed states; guards must also apply.
    cp['network_weights'] = {'module.'+k:v for k,v in cp['network_weights'].items()}
    receiver.load_checkpoint(cp)
    before = tensor_state(rebuilt)
    cp['network_weights'] = build([1,3]).state_dict()
    cp['optimizer_state'] = {'invalid':True}
    cp['logging'] = {'must_not_load':True}
    cp['current_epoch'] = 999
    with pytest.raises(ValueError,match='identity'):
        receiver.load_checkpoint(cp)
    assert receiver.current_epoch == 21
    assert receiver.logger.loaded != cp['logging']
    assert receiver.optimizer.param_groups[0]['lr'] == .01
    assert all(torch.equal(v,rebuilt.state_dict()[k]) for k,v in before.items())
    with pytest.raises(ValueError,match='identity'):
        receiver.network.load_state_dict({'module.'+k:v for k,v in cp['network_weights'].items()})
    assert all(torch.equal(v,rebuilt.state_dict()[k]) for k,v in before.items())
