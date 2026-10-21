# Next

Updated: 2026-10-21; branch-local continuation.

## Current focus
- Consolidate already reported HD-BET3mm ISLES-initialized and scratch results; preparation/training/evaluation are no longer blanket pending tasks.

## Next actions
1. Retrieve or verify both evaluation_best_fold0 folders on the server: summary_metrics.csv/json, case_metrics.csv, exact fixed-19 cohort and best-checkpoint provenance. Preserve all existing outputs.
2. Confirm the original ISLES508 source hash, derived-data receipt, unchanged splits/plans and matched Trainer/stopping settings. Pair case-level results and report the full metric vector; do not interpret Dice alone as uniform improvement.
3. For a skull-stripping effect study, first establish a matched original-DWI control with the same initialization hash and training protocol. Historical unmatched runs are descriptive references only.
4. Before resuming GenSeg, read `docs/GenSeg_生成质量修复与训练协议调整_主代理交接_20261019.md` if available and recheck its current contents. This ignored document does not travel via git pull. Preserve completed diagnostics and keep Adam/BN changes as explicit, separately tested hypotheses.

## Blockers / evidence limits
- Server result files/checkpoint hashes were not accessed in this update; recorded numbers rely on prior server-returned evidence.
- No formal five-fold OOF outcome is established by the recorded fold-0 comparison.
- Historical whole-repository pytest collection errors were previously recorded; their current status is unknown. No tests or resource/model probes were run for this documentation update.
- Master has no tracked canonical project memory. Do not describe this branch snapshot or other unmerged worktrees as integrated state.
