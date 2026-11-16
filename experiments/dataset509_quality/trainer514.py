"""Dataset514 identity adapter; reuse the validated train-only pool mechanics."""
import os
from pathlib import Path
import torch
from nnUNetTrainerLesionGen512TopK10EarlyStopping import nnUNetTrainerLesionGen512TopK10EarlyStopping
from nnUNetTrainerTopK10EarlyStopping import nnUNetTrainerTopK10EarlyStopping
from lesiongen.segmentation512 import sha,need


class nnUNetTrainerLesionGen514TopK10EarlyStopping(nnUNetTrainerLesionGen512TopK10EarlyStopping):
    def __init__(self,plans,configuration,fold,dataset_json,device=torch.device('cuda')):
        need(configuration == '2d' and type(fold) is int and fold == 0
             and plans['dataset_name'] == 'Dataset514_CI1DWILesionGen30','FIXED514_FOLD0_2D_REQUIRED')
        self.pool=Path(os.environ['LESIONGEN512_POOL']);self.pool_pin=os.environ['LESIONGEN512_POOL_SHA256']
        need(sha(self.pool/'manifest.json') == self.pool_pin,'POOL_PIN_REQUIRED')
        plans=dict(plans);plans['continue_training']=False
        nnUNetTrainerTopK10EarlyStopping.__init__(self,plans,configuration,fold,dataset_json,device)
        self.my_init_kwargs['lesiongen512_pool_sha256']=self.pool_pin
