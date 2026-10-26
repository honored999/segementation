# Next

Scope: branch-local codex/upernet-no-stage7; not canonical master memory.

## Current focus

- Deliver the two-stage drop-deepest/DS/TopK10 BAT; independent static review PASS. Trainer 01ec900 supports optional deep supervision; historical single-output runs remain distinct.

## Next actions

1. Synchronize the matching server worktree; do not claim runtime validation.
2. Run run_plainconv_depth_pretrain_finetune.bat on the confirmed server. It activates nnunet5090, selects physical GPU1 by UUID, reuses existing preprocessed data/splits, trains both fold0 stages with best validation, then computes two original-full-volume metric sets.
3. Preserve any failure output. The BAT starts fresh and does not automatically resume or overwrite previous runs.
4. Collect both printed metric directories and run_receipt.json. Source and target results describe different cohorts; fold0 is development evidence, not five-fold OOF.

## Pending evidence

- Server paths/dependencies, real training, weight initialization and resulting full-volume metrics remain NOT_RUN locally.
- Source raw data, original plans/splits and old experiments remain protected.
