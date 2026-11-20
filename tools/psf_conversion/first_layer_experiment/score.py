"""Native-space scoring with explicit cohort pairing and undefined HD95 handling."""
import json
import math
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from evaluate_segmentation_metrics import build_summary, evaluate_case

EXTRA = ('precision', 'tp', 'fp', 'fn')


def score(pred, inputs, gt, output):
    expected = {p.name.removesuffix('_0000.nii.gz') for p in inputs.glob('*_0000.nii.gz')}
    actual = {p.name.removesuffix('.nii.gz') for p in pred.glob('*.nii.gz')}
    if not expected or actual != expected:
        raise ValueError('Predicted cases differ from the fixed evaluation cohort')
    rows = []
    for case in sorted(expected):
        prediction = pred/f'{case}.nii.gz'
        x = nib.load(inputs/f'{case}_0000.nii.gz')
        y = nib.load(prediction)
        if x.shape != y.shape or not np.allclose(x.affine, y.affine, rtol=0, atol=1e-5):
            raise ValueError('Prediction geometry differs from its input')
        if not np.isin(np.asarray(y.dataobj), [0, 1]).all():
            raise ValueError('Invalid binary segmentation')
        row = evaluate_case(prediction, gt/f'{case}.nii.gz', 'paper')
        row['precision'] = row['tp']/row['pred_voxels'] if row['pred_voxels'] else 0.0
        if not row['pred_voxels'] or not row['gt_voxels']:
            row['hd95_mm'] = math.nan
        rows.append(row)
    frame = pd.DataFrame(rows)
    summary = build_summary(frame, 'paper')
    for name in EXTRA:
        summary['metrics'][name] = dict(mean=float(frame[name].mean()), valid_cases=len(frame))
        if name != 'precision':
            summary['metrics'][name]['total'] = int(frame[name].sum())
    for stats in summary['metrics'].values():
        for name, value in stats.items():
            if isinstance(value, float) and not math.isfinite(value):
                stats[name] = None
    output.mkdir()
    frame.to_csv(output/'case_metrics.csv', index=False)
    (output/'summary_metrics.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')


def pair(root):
    directories = json.loads((root/'score_dirs.json').read_text(encoding='utf-8'))
    frames, summaries = {}, {}
    cohort = None
    for key, value in directories.items():
        path = Path(value)
        summaries[key] = json.loads((path/'summary_metrics.json').read_text(encoding='utf-8'))
        frame = pd.read_csv(path/'case_metrics.csv').set_index('case_id').sort_index()
        if not frame.index.is_unique:
            raise ValueError('Duplicate case in stage metrics')
        if cohort is None:
            cohort = frame.index
        if not frame.index.equals(cohort):
            raise ValueError('Stage cohorts differ')
        frames[key] = frame[list(summaries[key]['metrics'])]
    pd.concat(frames, axis=1).to_csv(root/'paired_metrics.private.csv')
    (root/'summary_metrics.json').write_text(json.dumps(summaries, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    args = [Path(arg) for arg in sys.argv[1:]]
    if len(args) == 1:
        pair(args[0])
    else:
        score(*args)
