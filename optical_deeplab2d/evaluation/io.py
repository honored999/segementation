from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np

def write_evaluation(rows: list[dict], output_dir: Path) -> dict:
    from .metrics import binary_metrics, summarize_by_patient
    output_dir.mkdir(parents=True, exist_ok=True); images=[{"patient":r['patient'], **binary_metrics(r['target'],r['prediction'])} for r in rows]; patients=summarize_by_patient(rows); global_metrics=binary_metrics(np.concatenate([r['target'].ravel() for r in rows]),np.concatenate([r['prediction'].ravel() for r in rows])); summary={"global":global_metrics,"mean_image_dice":float(np.mean([r['dice'] for r in images])),"mean_patient_dice":float(np.mean([r['dice'] for r in patients]))}
    for name,data in [('image_metrics.csv',images),('patient_metrics.csv',patients)]:
        with (output_dir/name).open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=data[0].keys());w.writeheader();w.writerows(data)
    (output_dir/'summary_metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');return summary
