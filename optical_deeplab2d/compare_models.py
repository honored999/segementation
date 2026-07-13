from __future__ import annotations
import argparse,json
from pathlib import Path
def main() -> None:
 p=argparse.ArgumentParser(description='Compare two summary_metrics.json files.');p.add_argument('hybrid',type=Path);p.add_argument('electronic',type=Path);a=p.parse_args();print(json.dumps({'hybrid':json.loads(a.hybrid.read_text()),'electronic':json.loads(a.electronic.read_text())},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
