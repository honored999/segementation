"""Fail-closed runtime fingerprint of the inspected, locally tested official API."""
import hashlib
import importlib.metadata
import inspect
from pathlib import Path
from quality import need, read


def snapshot():
    from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    targets = dict(loader_bbox=nnUNetDataLoader.get_bbox, loader_batch=nnUNetDataLoader.generate_train_batch,
                   loaders=nnUNetTrainer.get_dataloaders, train_step=nnUNetTrainer.train_step,
                   validation_step=nnUNetTrainer.validation_step,
                   transforms=nnUNetTrainer.get_training_transforms)
    return dict(nnunetv2=importlib.metadata.version('nnunetv2'),
                source_sha256={k: hashlib.sha256(inspect.getsource(v).replace('\r\n', '\n').encode()).hexdigest()
                               for k, v in targets.items()})


def verify():
    expected = read(Path(__file__).with_name('api_contract.json'))
    actual = snapshot()
    need(actual == expected, 'NNUNET_API_DRIFT: run focused_checks.py and review before updating contract')
    return actual
