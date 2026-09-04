from __future__ import annotations
import torch
from torch import nn
from standalone_nnunet2d.losses.compound import DiceCrossEntropyLoss
from standalone_nnunet2d.training.formal_trainer import run_formal_epoch
from standalone_nnunet2d.training.official_config import OfficialTrainerSchedule, PolyLRScheduler, make_official_optimizer
def test_formal_epoch_uses_poly_lr_and_updates_model() -> None:
 model=nn.Conv2d(1,2,1); optimizer=make_official_optimizer(model); scheduler=PolyLRScheduler(optimizer,.01,1000); before=model.weight.detach().clone()
 result,lr=run_formal_epoch(model,[(torch.randn(1,1,4,4),torch.randint(0,2,(1,4,4)))],DiceCrossEntropyLoss(),optimizer,scheduler,torch.device('cpu'),10,OfficialTrainerSchedule())
 assert result.batch_count==1 and lr<.01 and not torch.equal(before,model.weight.detach())
