"""Mode-specific names in a fresh run root; old Q job specs keep their defaults."""
from pathlib import Path
from quality import need


def job_layout(root, spec):
    root = Path(root)
    mode = spec.get('mode', 'q')
    need(mode in ('baseline', 'q', 'quality_filter', 'c'), 'JOB_MODE_UNSUPPORTED')
    names = dict(config=spec.get('config_name', mode + '.formal.json'),
                 training=spec.get('training_name', mode + '_training'),
                 metrics=spec.get('metrics_name', mode + '_metrics'))
    for name in names.values():
        need(isinstance(name, str) and bool(name) and name not in ('.', '..') and
             '/' not in name and '\\' not in name and ':' not in name, 'JOB_LAYOUT_LEAF_NAMES_REQUIRED')
    need(len(set(names.values())) == 3, 'JOB_LAYOUT_COLLISION')
    return dict(mode=mode, **{key: root / name for key, name in names.items()})
