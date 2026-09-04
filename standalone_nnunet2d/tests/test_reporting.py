from __future__ import annotations
import json
from standalone_nnunet2d.metrics.reporting import write_case_reports

def test_reporting_writes_summary_and_ranked_case_files(tmp_path) -> None:
    records=[{'case_id':'a','dice':.2,'iou':.1,'precision':.2,'recall':.3,'gt_voxels':1,'pred_voxels':2,'tp':1,'fp':1,'fn':0},{'case_id':'b','dice':.8,'iou':.7,'precision':.8,'recall':.9,'gt_voxels':2,'pred_voxels':2,'tp':2,'fp':0,'fn':0}]
    write_case_reports(records,tmp_path,fold=0,checkpoint_path=tmp_path/'best.pth')
    summary=json.loads((tmp_path/'summary.json').read_text())
    assert summary['median_dice']==.5 and summary['best_case']=='b'
    assert 'case_id=b' in (tmp_path/'best_cases.txt').read_text()
