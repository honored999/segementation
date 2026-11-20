"""Isolated training adaptation: fixed optical kernel and fresh backend optimizer."""
import gc,importlib,json,os,random,sys,types
from pathlib import Path
import numpy as np
import torch
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from shared import read,write,need,sha,pin,KEY
from tools.psf_conversion.psf_first_conv import FirstConvHook,load_psfs,PIXEL_OPERATOR
from tools.psf_conversion.first_layer_model import check_artifact_origin, load_network, resize_first_conv
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetNumpy
class ReadOnlyNumpyDataset(nnUNetDatasetNumpy):
    @staticmethod
    def unpack_dataset(*args,**kwargs):return None

CONV=KEY.removesuffix('.weight')

def tensor_sha(x):
    return __import__('hashlib').sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def frozen_state(t):
    conv=t.network.get_submodule(CONV)
    need(all(not x.requires_grad and x.grad is None for x in conv.parameters()),'OPTICAL_GRADIENT_LEAK')
    state={k:tensor_sha(v) for k,v in conv.state_dict().items()}
    need(state==t._optical['first_conv_state_sha256'],'OPTICAL_STATE_CHANGED')
    need(tensor_sha(t._optical_hook.kernel)==t._optical['effective_kernel_sha256'],'OPTICAL_KERNEL_CHANGED')
    return state

def prepare_trainer(spec, model_dir, *, lr):
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.6, 0)
    need(torch.cuda.device_count() == 1, 'GPU_MAPPING_INVALID')
    model = Path(spec['model'])
    plans = read(model/'plans.json')
    plans['continue_training'] = False
    Trainer = getattr(importlib.import_module(spec['trainer']), spec['trainer'])
    trainer = Trainer(plans, '2d', 0, read(model/'dataset.json'), torch.device('cuda:0'))
    trainer.MAX_EPOCHS = trainer.num_epochs = 1000
    trainer.MIN_TRAINING_EPOCHS = 0
    trainer.PATIENCE = 30
    trainer.MIN_DELTA = 1e-4
    trainer.initial_lr = lr
    trainer.output_folder_base = str(model_dir)
    trainer.output_folder = str(model_dir/'fold_0')
    trainer.log_file = str(model_dir/'fold_0'/'training_log.txt')
    Path(trainer.output_folder).mkdir(parents=True, exist_ok=False)
    trainer.logger = type(trainer.logger)(trainer.output_folder, False)
    trainer.logger.update_config(dict(plans=plans, configuration='2d', fold=0, dataset=trainer.dataset_json))
    trainer.initialize()
    need(trainer.dataset_class.__name__ == 'nnUNetDatasetNumpy', 'UNSUPPORTED_CACHE_DATASET_CLASS')
    trainer.dataset_class = ReadOnlyNumpyDataset
    trainer.plot_network_architecture = types.MethodType(lambda self: None, trainer)
    return trainer


def build(spec, job, preflight):
    ckpt = Path(spec['checkpoint']['path'])
    need(sha(ckpt) == spec['checkpoint']['sha256'], 'SOURCE_CHECKPOINT_CHANGED')
    out = job/'backend'/'model'
    trainer = prepare_trainer(spec, out, lr=.001)
    payload = torch.load(ckpt, map_location='cpu', weights_only=False, mmap=True)
    conv = load_network(trainer.network, payload)
    trainer.inference_allowed_mirroring_axes = payload.get('inference_allowed_mirroring_axes')
    trainer.first_layer = payload['first_layer']
    del payload
    gc.collect()
    check_artifact_origin(Path(spec['psf']), spec['checkpoint']['sha256'])
    kernel,hashes=load_psfs(Path(spec['psf']),conv.weight,operator=PIXEL_OPERATOR,ideal=False)
    need(hashes==spec['psf_hashes'] and len(hashes)==67,'PSF_IDENTITY_CHANGED')
    hook=FirstConvHook(trainer.network,kernel)
    trainer.network.requires_grad_(True);conv.requires_grad_(False)
    params=[v for v in trainer.network.parameters() if v.requires_grad]
    trainer.optimizer=torch.optim.SGD(params,lr=.001,weight_decay=trainer.weight_decay,momentum=.99,nesterov=True)
    trainer.lr_scheduler=PolyLRScheduler(trainer.optimizer,.001,1000)
    optical_ids={id(x) for x in conv.parameters()}
    need(not optical_ids&{id(x) for g in trainer.optimizer.param_groups for x in g['params']},'OPTICAL_IN_OPTIMIZER')
    need(trainer.current_epoch==0 and not trainer.optimizer.state and trainer._early_stopping_best_ema is None
         and trainer._epochs_without_improvement==0 and not trainer._early_stopping_triggered,'STALE_TRAINING_STATE')
    trainer._optical_hook=hook
    trainer._optical=dict(operator_id=PIXEL_OPERATOR,psf_directory=spec['psf'],psf_sha256=hashes,
      first_conv_state_sha256={k:tensor_sha(v) for k,v in conv.state_dict().items()},
      effective_kernel_sha256=tensor_sha(kernel),source_checkpoint=spec['checkpoint'],
      fixed_bias=True,lr=.001,min_epochs=0,patience=30,min_delta=.0001,max_epochs=1000,
      mandatory_psf_inference=True,classification='fixed fold0 development backend finetuning')
    write(out/'optical_finetune.json',trainer._optical)
    def save_checkpoint(self,filename):
        frozen_state(self)
        checkpoint=dict(network_weights=self.network.state_dict(),optimizer_state=self.optimizer.state_dict(),
          grad_scaler_state=self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
          logging=self.logger.get_checkpoint(),_best_ema=self._best_ema,current_epoch=self.current_epoch+1,
          init_args=self.my_init_kwargs,trainer_name=self.__class__.__name__,
          inference_allowed_mirroring_axes=self.inference_allowed_mirroring_axes,
          early_stopping_state=self._early_stopping_checkpoint_state(),optical_finetune=self._optical,
          optical_effective_kernel=self._optical_hook.kernel.cpu(),first_layer=self.first_layer)
        torch.save(checkpoint,filename)
    trainer.save_checkpoint=types.MethodType(save_checkpoint,trainer)
    end=trainer.on_epoch_end
    def epoch_end(self):
        end();frozen_state(self)
        write(job/'train_progress.json',dict(status='RUNNING',completed_epochs=self.current_epoch,
          ema_fg_dice=float(self.logger.get_value('ema_fg_dice',step=-1)),
          train_loss=float(self.logger.get_value('train_losses',step=-1)),
          val_loss=float(self.logger.get_value('val_losses',step=-1)),
          best_monitored_ema=self._early_stopping_best_ema,epochs_without_improvement=self._epochs_without_improvement,
          early_stopping_triggered=self._early_stopping_triggered,lr=self.optimizer.param_groups[0]['lr'],
          policy=dict(min_epochs=0,patience=30,min_delta=1e-4,max_epochs=1000)))
    trainer.on_epoch_end=types.MethodType(epoch_end,trainer)
    return trainer

def finish_loaders(t):
    for x in (getattr(t, "dataloader_train", None), getattr(t, "dataloader_val", None)):
        if hasattr(x,'_finish'):x._finish()

def train(spec,job,preflight):
    trainer=build(spec,job,preflight)
    try:
        if preflight:
            from types import SimpleNamespace
            from nnUNetTrainerEarlyStopping import nnUNetTrainerEarlyStopping
            fake=SimpleNamespace(current_epoch=0,MIN_TRAINING_EPOCHS=0,PATIENCE=30,MIN_DELTA=.0001,
              _early_stopping_best_ema=None,_epochs_without_improvement=0,_early_stopping_triggered=False,
              logger=SimpleNamespace(get_value=lambda *a,**k:.8),print_to_log_file=lambda *a:None)
            for epoch in range(31):
                fake.current_epoch=epoch;nnUNetTrainerEarlyStopping._update_early_stopping(fake)
                need(fake._early_stopping_triggered==(epoch==30),'EARLY_STOP_PLATEAU_OFF_BY_ONE')
            trainer.on_train_start();trainer.network.train()
            batch=next(trainer.dataloader_train)
            candidate=next((n,x) for n,x in trainer.network.named_parameters() if x.requires_grad and 'norm' in n)
            name,param=candidate;before=param.detach().clone()
            result=trainer.train_step(batch)
            need(np.isfinite(result['loss']).all(),'NONFINITE_PREFLIGHT_TRAIN_LOSS')
            need(param.grad is not None and torch.isfinite(param.grad).all() and bool(param.grad.abs().max()>0),'BACKEND_GRADIENT_MISSING')
            need(not torch.equal(before,param.detach()),'BACKEND_DID_NOT_UPDATE')
            frozen_state(trainer)
            trainer.network.eval()
            with torch.no_grad():val=trainer.validation_step(next(trainer.dataloader_val))
            need(np.isfinite(val['loss']).all(),'NONFINITE_PREFLIGHT_VAL_LOSS')
            x=batch['data'][:1].to(trainer.device)
            with torch.no_grad():a=trainer.network(x);a=(a[0] if isinstance(a,(list,tuple)) else a).detach().clone()
            path=Path(trainer.output_folder)/'roundtrip_NON_FORMAL.pth';trainer.save_checkpoint(str(path))
            trainer.load_checkpoint(str(path));frozen_state(trainer)
            with torch.no_grad():b=trainer.network(x);b=b[0] if isinstance(b,(list,tuple)) else b
            torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
            saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            need(saved['optical_finetune']==trainer._optical and tensor_sha(saved['optical_effective_kernel'])==trainer._optical['effective_kernel_sha256'],'ROUNDTRIP_OPTICAL_METADATA_LOST')
            write(job/'preflight.json',dict(status='COMPLETED',formal_eligible=False,train_loss=float(result['loss']),
              val_loss=float(val['loss']),backend_parameter=name,backend_updates=True,optical_unchanged=True,
              checkpoint_roundtrip=True,early_stop_plateau_triggers_epoch31=True,policy=trainer._optical))
        else:
            trainer.run_training();frozen_state(trainer)
            best=Path(trainer.output_folder)/'checkpoint_best.pth';need(best.is_file(),'BEST_CHECKPOINT_MISSING')
            saved=torch.load(best,map_location='cpu',weights_only=False,mmap=True)
            need(saved['optical_finetune']==trainer._optical,'BEST_OPTICAL_METADATA_CHANGED')
            for k,digest in trainer._optical['first_conv_state_sha256'].items():
                need(tensor_sha(saved['network_weights'][CONV+'.'+k])==digest,'BEST_OPTICAL_STATE_CHANGED')
            write(job/'training_completed.json',dict(status='COMPLETED',completed_epochs=trainer.current_epoch,
              early_stopping_triggered=trainer._early_stopping_triggered,best_checkpoint_epoch=saved['current_epoch'],
              checkpoint=pin(best),model_folder=trainer.output_folder_base,optical_finetune=trainer._optical))
    finally:finish_loaders(trainer)

def train_scratch(spec, job, preflight):
    """Train every parameter from random initialization before fitting optics."""
    model = job/'electrical_target'/'model'
    trainer = prepare_trainer(spec, model, lr=.01)
    conv = resize_first_conv(trainer.network, spec['kernel_size'])
    # PlainConvUNet initializes convolutional weights with He initialization.
    from dynamic_network_architectures.initialization.weight_init import InitWeights_He
    InitWeights_He(1e-2)(conv)
    parameters = list(trainer.network.parameters())
    need(all(p.requires_grad for p in parameters), 'FROZEN_ELECTRICAL_PARAMETER')
    trainer.optimizer, trainer.lr_scheduler = trainer.configure_optimizers()
    ids = {id(p) for group in trainer.optimizer.param_groups for p in group['params']}
    need(ids == {id(p) for p in parameters} and not trainer.optimizer.state
         and trainer.current_epoch == 0, 'INVALID_SCRATCH_OPTIMIZER')
    stage = Path(trainer.output_folder)
    metadata = dict(kernel_size=spec['kernel_size'], initialization='random nnU-Net He',
                    training='all electrical network parameters', seed=12345,
                    initial_lr=.01, min_epochs=0, patience=30, min_delta=1e-4, max_epochs=1000)
    write(model/'electrical_scratch.json', metadata)
    end = trainer.on_epoch_end

    def epoch_end(self):
        end()
        write(job/'electrical_progress.json', dict(completed_epochs=self.current_epoch,
              ema_fg_dice=float(self.logger.get_value('ema_fg_dice', step=-1)),
              train_loss=float(self.logger.get_value('train_losses', step=-1)),
              val_loss=float(self.logger.get_value('val_losses', step=-1)),
              lr=self.optimizer.param_groups[0]['lr'],
              epochs_without_improvement=self._epochs_without_improvement,
              early_stopping_triggered=self._early_stopping_triggered))

    trainer.on_epoch_end = types.MethodType(epoch_end, trainer)
    try:
        if preflight:
            trainer.on_train_start()
            trainer.network.train()
            backend = next(p for name, p in trainer.network.named_parameters()
                           if not name.startswith(CONV) and 'norm' in name)
            before = tuple(p.detach().clone() for p in (conv.weight, backend))
            loss = trainer.train_step(next(trainer.dataloader_train))['loss']
            need(np.isfinite(loss).all(), 'NONFINITE_ELECTRICAL_LOSS')
            for previous, current in zip(before, (conv.weight, backend)):
                need(current.grad is not None and torch.isfinite(current.grad).all()
                     and not torch.equal(previous, current.detach()), 'ELECTRICAL_WEIGHT_DID_NOT_UPDATE')
            trainer.network.eval()
            with torch.no_grad():
                val = trainer.validation_step(next(trainer.dataloader_val))['loss']
            need(np.isfinite(val).all(), 'NONFINITE_ELECTRICAL_VALIDATION')
            best = stage/'checkpoint_electrical_NON_FORMAL.pth'
            trainer.save_checkpoint(str(best))
        else:
            trainer.run_training()
            best = stage/'checkpoint_best.pth'
        need(best.is_file(), 'ELECTRICAL_BEST_CHECKPOINT_MISSING')
        payload = torch.load(best, map_location='cpu', weights_only=False, mmap=True)
        weight = payload['network_weights'][KEY]
        need(tuple(weight.shape) == (32, 1, spec['kernel_size'], spec['kernel_size'])
             and torch.isfinite(weight).all(), 'INVALID_ELECTRICAL_BEST_KERNEL')
        export = stage/'checkpoint_target.pth'
        # Preserve the native best checkpoint. The inference artifact adds size metadata
        # and carries the exact same network weights without optimizer state.
        fields = ('network_weights', 'init_args', 'trainer_name', 'inference_allowed_mirroring_axes')
        torch.save(dict({key: payload[key] for key in fields}, first_layer=metadata), export)
        np.save(job/'electrical_kernels.npy', weight.detach().float().numpy())
        write(job/'target_completed.json', dict(status='COMPLETED', checkpoint=pin(export),
              native_best_checkpoint=pin(best), best_checkpoint_epoch=payload['current_epoch'],
              completed_epochs=trainer.current_epoch, formal_eligible=not preflight,
              scratch=True, all_parameters_trainable=True, policy=metadata))
    finally:
        finish_loaders(trainer)


if __name__ == '__main__':
    job = Path(sys.argv[1])
    phase, mode = sys.argv[2:]
    spec = read(job/'spec.private.json')
    if phase == 'scratch':
        train_scratch(spec, job, mode == 'preflight')
    elif phase == 'backend':
        train(spec, job, mode == 'preflight')
    else:
        raise ValueError(f'Unknown phase: {phase}')
