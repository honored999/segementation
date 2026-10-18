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

## Branch-local immediate action: GenSeg30 (2026-10-14)

1. Transfer only reviewed code to the server after requested Git integration; run experiments/genseg30/README.md compatibility preflight on the existing Dataset501 source.
2. If Blosc2/fixed fold0 checks pass, train generator, export exactly 30% extra train slices and inspect/approve image-mask quality.
3. Run matched plain EarlyStopping A and GenSeg30 B fold0 and evaluate best checkpoints on the original 19 real patients. No server access or real-data run was performed locally; historical A protocol is unverified.

## Branch-local immediate action: GD500 vs joint100 preview (2026-10-18)

1. Data computer pull --ff-only, then follow reference/GENSEG-JOINT-GD-PREVIEW-RUNBOOK.md to preview completed GD500 gan.pt into fresh genseg_gd_preview_500steps_trial01 using --phase gan_pretrain and the receipt's exact hash.
2. Compare same-name PNG third panels against existing joint500plus100 preview; copy new preview folder locally for read-only review. If GD already gray, prioritize pretraining diagnosis; if only joint gray, prioritize joint diagnosis. Neither pattern proves BN/optimizer causality by itself.
3. Pause budget expansion pending stage localization; no training settings/threshold/split changes. Keep all old runs and end-only-save/no-resume limits. Local verification remains STATIC ONLY; first new preview runtime belongs on data computer.

Current evidence: user reported continuous joint3 and GD500/joint100 steps; local copied final joint preview confirms 500-step pretrain initialization and 100 joint steps, and inspected images lack useful anatomy. This is engineering/exploratory evidence, not formal evaluation.
