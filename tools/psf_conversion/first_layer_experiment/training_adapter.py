"""Isolated training adaptation: fixed optical kernel and fresh backend optimizer."""
import gc,importlib,json,os,random,sys,types
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from shared import read,write,need,sha,pin,KEY
from tools.psf_conversion.psf_first_conv import FirstConvHook,load_psfs,PIXEL_OPERATOR
from tools.psf_conversion.first_layer_model import check_artifact_origin, first_weight_names, load_network, resize_first_conv
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

def build(spec,job,preflight, *, target=False):
    random.seed(12345);np.random.seed(12345);torch.manual_seed(12345);torch.cuda.manual_seed_all(12345)
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.6,0)
    need(torch.cuda.device_count()==1,'GPU_MAPPING_INVALID')
    model=Path(spec['model']);ckpt=Path(spec['checkpoint']['path'])
    need(sha(ckpt)==spec['checkpoint']['sha256'],'SOURCE_CHECKPOINT_CHANGED')
    plans=read(model/'plans.json');plans['continue_training']=False
    Trainer=getattr(importlib.import_module(spec['trainer']),spec['trainer'])
    trainer=Trainer(plans,'2d',0,read(model/'dataset.json'),torch.device('cuda:0'))
    trainer.MAX_EPOCHS=1000;trainer.MIN_TRAINING_EPOCHS=0;trainer.PATIENCE=30;trainer.MIN_DELTA=1e-4
    trainer.num_epochs=1000;trainer.initial_lr=.001
    out=job/('electrical_target' if target else 'backend')/'model'
    trainer.output_folder_base=str(out);trainer.output_folder=str(out/'fold_0')
    trainer.log_file=str(Path(trainer.output_folder)/'training_log.txt')
    Path(trainer.output_folder).mkdir(parents=True,exist_ok=False)
    trainer.logger=type(trainer.logger)(trainer.output_folder,False)
    trainer.logger.update_config(dict(plans=plans,configuration='2d',fold=0,dataset=trainer.dataset_json))
    trainer.initialize()
    # Trusted project checkpoint pinned above; transfer network only, never optimizer/logging/epoch state.
    payload=torch.load(ckpt,map_location='cpu',weights_only=False,mmap=True)
    conv = load_network(trainer.network, payload)
    trainer.inference_allowed_mirroring_axes=payload.get('inference_allowed_mirroring_axes')
    if target:
        conv = resize_first_conv(trainer.network, spec['kernel_size'])
        trainer.first_layer = dict(kernel_size=spec['kernel_size'],
            initialization='centered baseline weights, zero exterior',
            initialization_checkpoint=spec['checkpoint'])
    else:
        trainer.first_layer = payload['first_layer']
    del payload
    gc.collect()
    need(trainer.dataset_class.__name__ == 'nnUNetDatasetNumpy', 'UNSUPPORTED_CACHE_DATASET_CLASS')
    trainer.dataset_class = ReadOnlyNumpyDataset
    trainer.plot_network_architecture = types.MethodType(lambda self: None, trainer)
    if target:
        trainer.network.requires_grad_(False)
        conv.weight.requires_grad_(True)
        trainer.optimizer = torch.optim.SGD([conv.weight], lr=0.001, momentum=0.99,
                                            nesterov=True, weight_decay=trainer.weight_decay)
        return trainer
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

def train_target(spec, job, preflight):
    """Update only the first weight with the source trainer's loss and augmentation."""
    trainer = build(spec, job, preflight, target=True)
    updates = 1 if preflight else 250
    try:
        random.seed(12345)
        np.random.seed(12345)
        torch.manual_seed(12345)
        trainer.on_train_start()
        trainer.network.eval()
        conv = trainer.network.get_submodule(CONV)
        before = tensor_sha(conv.weight)
        weight_names = first_weight_names(trainer.network)
        frozen = {name: tensor_sha(value) for name, value in trainer.network.state_dict().items()
                  if name not in weight_names}
        for _ in range(updates):
            result = trainer.train_step(next(trainer.dataloader_train))
            need(np.isfinite(result['loss']).all(), 'NONFINITE_ELECTRICAL_LOSS')
        need(tensor_sha(conv.weight) != before, 'FIRST_WEIGHT_DID_NOT_UPDATE')
        need(frozen == {name: tensor_sha(value) for name, value in trainer.network.state_dict().items()
                        if name not in weight_names}, 'FROZEN_ELECTRICAL_STATE_CHANGED')
        out = Path(trainer.output_folder)
        checkpoint = out/'checkpoint_target.pth'
        torch.save(dict(network_weights=trainer.network.state_dict(), first_layer=trainer.first_layer,
            init_args=trainer.my_init_kwargs, trainer_name=trainer.__class__.__name__,
            inference_allowed_mirroring_axes=trainer.inference_allowed_mirroring_axes), checkpoint)
        np.save(job/'electrical_kernels.npy', conv.weight.detach().float().cpu().numpy())
        write(job/'target_completed.json', dict(checkpoint=pin(checkpoint), updates=updates,
              nominal_updates=250, lr=0.001, seed=12345, last_loss=float(result['loss']),
              formal_eligible=not preflight, frozen_backend_and_bias=True))
    finally:
        finish_loaders(trainer)


if __name__ == '__main__':
    job = Path(sys.argv[1])
    phase, mode = sys.argv[2:]
    spec = read(job/'spec.private.json')
    if phase == 'target':
        train_target(spec, job, mode == 'preflight')
    elif phase == 'backend':
        train(spec, job, mode == 'preflight')
    else:
        raise ValueError(f'Unknown phase: {phase}')
