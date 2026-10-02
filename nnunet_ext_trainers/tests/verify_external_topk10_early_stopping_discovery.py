"""Fresh official resolver check; do not preload external Trainer modules."""
from importlib.metadata import version
import sys

assert version('nnunetv2') == '2.8.1'
assert 'nnUNetTrainerTopK10EarlyStopping' not in sys.modules
from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name

resolved = recursive_find_trainer_class_by_name('nnUNetTrainerTopK10EarlyStopping')
assert resolved.__name__ == 'nnUNetTrainerTopK10EarlyStopping'
assert resolved.__module__ == 'nnUNetTrainerTopK10EarlyStopping'
assert [c.__name__ for c in resolved.__mro__[:4]] == [
    'nnUNetTrainerTopK10EarlyStopping', 'nnUNetTrainerEarlyStopping',
    'nnUNetTrainerTopK10', 'nnUNetTrainer']
print('TOPK10_EARLY_STOPPING_RESOLVER_OK nnunetv2=2.8.1 MRO=' +
      ' -> '.join(c.__name__ for c in resolved.__mro__))
