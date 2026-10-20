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

## Historical GenSeg30 action (2026-10-14; superseded)

1. Transfer only reviewed code to the server after requested Git integration; run experiments/genseg30/README.md compatibility preflight on the existing Dataset501 source.
2. If Blosc2/fixed fold0 checks pass, train generator, export exactly 30% extra train slices and inspect/approve image-mask quality.
3. Run matched plain EarlyStopping A and GenSeg30 B fold0 and evaluate best checkpoints on the original 19 real patients. No server access or real-data run was performed locally; historical A protocol is unverified.

## Historical fixed-slice diagnostic (2026-10-18; completed receipt supplied)

1. Data computer pull --ff-only and follow reference/GENSEG-JOINT-SINGLE-SLICE-RUNBOOK.md: first fresh3-step diagnose-single-slice from the approved GD500 checkpoint, keeping all original settings.
2. On diagnostic-only complete=true/completed_steps=3 with finite probes and no resource/audit error, run a NEW100-step diagnostic from the same original GD500 (not from3); no weights are saved and no resume exists.
3. Copy before.png/after.png/diagnostic.json/logs locally. Compare image structure, normalized L1, spatial variation and layer gradients; gray GD500 is already confirmed. Do not expand joint budget or change BN/optimizer/split until evidence supports a controlled next experiment.

Local verification remains STATIC ONLY. Single-training-sample observations do not prove generalization, model incapacity or a BN root cause. Master canonical memory is absent/unmodified; this is branch-local continuation.

## Branch-local current priority (2026-10-19)

- Local copy of single-slice diagnostic.json confirms complete=true, completed_steps=100, weights_saved=false. Prior pending-run instructions above are historical; do not rerun or expand joint budgets by default. This is diagnostic engineering evidence, not efficacy or a proven BN/optimizer root cause.
- User approved first running real-DWI HD-BET3mm nnU-Net TopK10+early stopping with normal deep supervision, then steered initialization to the original ISLES508 checkpoint (fresh full-parameter fine-tuning, not CKPD or private-data resume) on original fold0 (76 training/19 development validation), before further generative redesign. Existing masks cover76 only; remaining19 need the same frozen image-only processing. No raw/split/oldweight mutation.
- Implementation and independent Level3 STATIC acceptance complete; sync the committed branch and follow preparation in reference/HDBET-NNUNET-FOLD0-RUNBOOK.md and the steering addendum reference/HDBET-NNUNET-FINETUNE-FOLD0-RUNBOOK.md (independent STATIC review PASS; source runtime still pending). Server check -> fresh prepare95 -> train76 QA -> isolated roots -> preprocess-check/guard -> preprocess2d inherited plans -> ready/discovery/guard -> train fold0 --val_best -> exact19 original-space evaluation.
- New server output root: C:\lijialin\HD-BET_outputs\nnunet_hdbet3mm_isles508_finetune_fold0_trial01. Local validation stays STATIC ONLY by user request; runtime/tests/real preprocessing/training remain deferred. Matched original-DWI Trainer control required for causal attribution when old settings differ.
- Generative original-DWI conditioning/residual and lesion-template transfer remain unvalidated candidate directions, not part of this nnU-Net preprocessing comparison.
- Original ISLES508 fold0 checkpoint path was recovered from the user's historical chat; actual current existence/content/hash still requires server audit. The supplied nnUNet_results_ISLES508_pretrained_ES and CKPD_ES roots contain private fine-tuning outputs, not the original ISLES checkpoint. Do not initialize from either arbitrarily.
- Before preparation: run the addendum's read-only source existence check, then guarded SERVER CPU source-metadata/hash audit. Compare same-source originalDWI vs strippedDWI; old ordinary runs have no verified identical source hash, so causal attribution remains limited. No LR/freeze/optimizer/architecture change in this first screen.

## Branch-local repair continuation (2026-10-19)

1. Publish only codex/genseg-quality-repair; server uses a separate checkout per reference/GENSEG-QUALITY-REPAIR-RUNBOOK.md. Never pull shared-runtime changes into the ongoing fine-tuning checkout or alter its recorded hashes.
2. Run the new fixed-budget cold/GD500-recovery training-slice diagnostics on the data computer, preserve failed gates. Review same-scale structure/lesion alignment and fixed inner61 multi-plane previews before expanding budget.
3. Phase C remains pending real learning gates and a separate actual/virtual BN/optimizer design. New pretrain cannot enter legacy joint. No automatic long training or efficacy claim; local CPU pytest permission is task-scoped.

- Immediate repair continuation: pull the probe-context follow-up only in the repair checkout, use a fresh diagnostic output (trial02) and rerun the same fixed-budget command. Preserve the failed pre-update trial; if exact check still fails, collect the newly reported max_abs_diff/device/context before changing tolerances.
- Latest probe contract supersedes the exact-output requirement above: update only repair checkout, then use a fresh cold100 trial03 directory. Repeat numerical comparison is now fixed atol1e-7/rtol0; state purity stays exact. Preserve old failures; wait for actual CUDA diagnostic and quality results before further work.
- Latest continuation2026-10-20: pull probe-local cuDNN fix only into repair checkout and first run a fresh1-step cold diagnostic; inspect before/after and backend restored state. Do not directly repeat100. Old100 steps completed but after QA failed, no recoverable weights; retain failed artifacts and await1-step evidence before next budget.

## Current GenSeg repair continuation (2026-10-20; supersedes earlier pending checks)
- Deterministic probe cold100 already completed on server; learning-quality gate failed. Preserve that run; no resume/weights available.
- Pull only repair checkout and run same-seed/same-target/same100-update diagnostic with --compare-bn-modes into a new isolated directory per RUNBOOK. Compare after eval vs private BN batchstats statistics, retain original quality result; no epoch/joint expansion or automatic parameter changes.
