# Next

## Current focus

- Decide whether to run a separately authorized, isolated Lite-UPerNet
  engineering preflight or experiment. Implementation itself is complete.

## Next actions

1. Keep both baseline models and their checkpoints unchanged.
2. If experimentation is requested, define isolated output names and preserve
   the fixed patient-level five-fold split and matched single-output contract.
3. Treat any synthetic smoke/preflight evidence separately from formal OOF
   results.

## Blockers

- None for the Lite-UPerNet implementation.
- Whole-repository `python -m pytest -q` still encounters six pre-existing
  `non_teacher_student_files` import-collection errors outside this task diff.

## 2026-10-12 CKPD immediate actions (branch-local)

1. Deploy the small CKPD package separately using docs/NNUNET_CKPD_SERVER_STEPS_ZH.md.
2. Select and record the intended Dataset508 source checkpoint; confirm cohort independence and fixed source/target splits.
3. Run bounded source-train calibration with host/GPU resource guards, then isolated Dataset501 fold-0 CKPD and same-weight full-tuning control.
4. Evaluate original full-volume metrics and preserve per-case provenance. Fold-0 is development evidence, not five-fold OOF.

Blockers to real-server evidence: source weight path and patient-cohort provenance are not verified locally; actual server data and CUDA execution are unavailable in this checkout.

## 2026-10-13 CKPD immediate recovery

Update the server worktree using fast-forward-only pull, then use the existing CKPD result root and --val --val_best to validate the saved best checkpoint. Do not restart training for this metadata compatibility issue. After validation, evaluate original full-volume metrics and compare with the same-source-checkpoint full-tuning control; cohort provenance and formal five-fold evidence remain distinct requirements.
