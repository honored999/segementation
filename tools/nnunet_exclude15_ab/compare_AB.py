import csv
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
config = json.loads((root/"experiment_config.json").read_text(encoding="utf-8-sig"))
seed = int(sys.argv[1]) if len(sys.argv)>1 else 0
runs = [Path(config["results_parent"])/f"seed_{seed}"/arm for arm in ("A","B")]
manifests = [json.loads((p/"experiment_manifest.json").read_text(encoding="utf-8")) for p in runs]
if any(m["status"]!="COMPLETE" or not m["canonical_metadata_unchanged"] for m in manifests):
    raise ValueError("Both runs must complete with unchanged canonical metadata")
if manifests[0]["seed"] != manifests[1]["seed"] or manifests[0]["pretrained_sha256"]!=manifests[1]["pretrained_sha256"]:
    raise ValueError("Seed or pretraining weights differ")
if manifests[0]["selection_validation"]!=manifests[1]["selection_validation"] or manifests[0]["evaluation"]["evaluation_cases"]!=manifests[1]["evaluation"]["evaluation_cases"]:
    raise ValueError("Selection/evaluation cohorts differ")
rows=[]
for p in runs:
    with (p/"evaluation19_case_metrics.csv").open(encoding="utf-8-sig",newline="") as f:
        rows.append({r["case_id"]:r for r in csv.DictReader(f)})
if set(rows[0])!=set(rows[1]) or len(rows[0])!=19:
    raise ValueError("Expected same 19 evaluation cases")
dest=runs[0].parent/"paired19_A_vs_B.csv"
with dest.open("x",encoding="utf-8-sig",newline="") as f:
    writer=csv.writer(f);writer.writerow(["case_id","A_Dice","B_Dice","B_minus_A_percentage_points"])
    for case in sorted(rows[0]):
        a,b=[float(r[case]["dice_0_to_1"]) for r in rows]
        writer.writerow([case,a,b,100*(b-a)])
a,b=[m["evaluation"]["primary_original19_dice"] for m in manifests]
print(f"Original 19: A={a:.4%}; B={b:.4%}; B-A={100*(b-a):+.3f} percentage points")
a17,b17=[m["evaluation"]["secondary_remaining17_dice"] for m in manifests]
print(f"Remaining 17 (secondary): A={a17:.4%}; B={b17:.4%}; B-A={100*(b17-a17):+.3f} percentage points")
print("Paired CSV:",dest)
print("Exploratory result; a single pair does not establish data quality or stable benefit.")