from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
if __package__ in {None, ""}:
 sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from optical_deeplab2d.datasets.dataset_2d import read_manifest, load_sample
def main() -> None:
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args(); rows=read_manifest(a.data_root); [load_sample(row) for row in rows]; report={'samples':len(rows),'patients':len({r.patient for r in rows}),'positive':sum(r.has_mask for r in rows),'negative':sum(not r.has_mask for r in rows),'patient_id_rule':'manifest patient column'};a.output_dir.mkdir(parents=True,exist_ok=True);(a.output_dir/'dataset_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
