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

## Branch-local immediate action: fixed-slice diagnostic (2026-10-18)

1. Data computer pull --ff-only and follow reference/GENSEG-JOINT-SINGLE-SLICE-RUNBOOK.md: first fresh3-step diagnose-single-slice from the approved GD500 checkpoint, keeping all original settings.
2. On diagnostic-only complete=true/completed_steps=3 with finite probes and no resource/audit error, run a NEW100-step diagnostic from the same original GD500 (not from3); no weights are saved and no resume exists.
3. Copy before.png/after.png/diagnostic.json/logs locally. Compare image structure, normalized L1, spatial variation and layer gradients; gray GD500 is already confirmed. Do not expand joint budget or change BN/optimizer/split until evidence supports a controlled next experiment.

Local verification remains STATIC ONLY. Single-training-sample observations do not prove generalization, model incapacity or a BN root cause. Master canonical memory is absent/unmodified; this is branch-local continuation.
