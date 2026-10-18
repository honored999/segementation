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

## Branch-local immediate action: GD pretraining handoff (2026-10-18)

1. Publish reviewed feature branch locally, then data computer git pull --ff-only; follow reference/GENSEG-JOINT-PRETRAIN-RUNBOOK.md.
2. In fresh HD-BET_outputs directories, run1 GD pretrain update, audit completed gan.pt/receipt hash, then1 joint update from that same-inner61 checkpoint. Preserve raw data, original splits and all old runs/weights.
3. Inspect target-PC receipts/resources/finite alpha change. Long budgets and train-only generated-image QA follow only after the phase check; no external19 final-test claim and no arbitrary old76 warm start.

## Branch-local immediate action: joint preview (2026-10-18)

1. Data computer pull --ff-only, then reference/GENSEG-JOINT-PREVIEW-RUNBOOK.md to preview the completed1+1 joint checkpoint in a new HD-BET_outputs directory.
2. Inspect lesion/empty PNGs and preview.json for anatomy, label alignment and artifacts. No generated dataset has been exported; no image-quality approval or segmentation benefit yet.
3. Confirm multi-step resource behavior and define a small longer budget separately; no training/guard/resume changes were included in this preview task.

## Branch-local immediate action: increased training (2026-10-18)

1. Pull code on data computer, follow reference/GENSEG-JOINT-SHORT-RUNBOOK.md for fresh-directory GD3/joint3 preflight; stop on any resource/numerical error.
2. Only after that succeeds, fresh GD500 -> new joint100 -> same-slice preview. Preserve1+1 baseline, raw/splits/targets and all old weights. Budget is exploratory, no performance claim.
3. Local checks remain STATIC ONLY per user; all runtime validation uses data computer. Current code saves weights only at successful run end and cannot resume interruptions.
