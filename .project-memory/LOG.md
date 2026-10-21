# Log

## 2026-10-21 - Branch-local progress consolidation
- Updated STATUS/GOALS/NEXT to the HD-BET3mm comparison milestone; removed superseded blanket pending-run directions and compressed older engineering history.
- Evidence: current clean branch at fb6f8ca; prior 2026-10-18 session through 2026-10-20 records server preparation95/95, fine-tuning/scratch fixed19 results and completed GenSeg100 diagnostic. Server artifacts were not freshly verified.
- Recorded Dice 74.7072% versus 71.5864% with boundary-metric tradeoffs; single-fold development evidence only. Next: paired provenance audit and a matched original-DWI control before skull-stripping attribution.
- Documentation-only update by main agent; no source/tests/raw data/checkpoints changed, no runtime validation, commit or push. Master memory absent and untouched.

## 2026-10-19 - ISLES508 initialization steering
- Added separate source-audit/hash/protocol runbook, retaining normal deep supervision and TopK10+early stopping; commit fb6f8ca.
- Embedded-command AST checks and independent Level 3 STATIC review PASS were recorded. Subsequent server outcomes are summarized above; historical local checks did not verify server runtime.

## 2026-10-19 - HD-BET real-DWI fold-0 preparation
- Added isolated95 derived-data builder, frozen mask processing/native3mm rule, protected GT/plans/splits and inherited preprocessing compatibility; commit 40309bd.
- Historical AST/diff checks and independent Level 3 STATIC review PASS. No local real-patient runtime or effectiveness claim.

## 2026-10-18 - GenSeg staged training and diagnostics
- Added G/D pretraining handoff, completed-checkpoint previews, bounded utilization retry and fixed-slice diagnostics. Historical scoped synthetic tests/static reviews are engineering evidence only.
- Prior receipt confirms100 diagnostic steps with no weights saved; gray GD500/joint previews do not pass an augmentation quality gate or prove a BN cause.

## 2026-10-14 - GenSeg reference and augmentation preparation
- Pinned upstream source with provenance; prepared train-only30% slice augmentation with fixed validation boundaries.
- Historical synthetic checks and independent review passed. Generated-data quality and segmentation benefit were not established.
