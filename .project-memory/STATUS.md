# Status

Updated: 2026-10-21

## Scope and evidence
- Branch-local snapshot on `codex/genseg-reference`, based on `fb6f8ca`.
- `master` has no tracked `.project-memory/`; this does not represent integration into master.
- Repository state checked locally on 2026-10-21. Server results below come from prior user-supplied receipts/results recorded through 2026-10-20; not revalidated against server files in this update.

## Current focus
- HD-BET3mm Dataset501 fold-0 preparation and original-ISLES508 initialization runbooks are implemented; independent Level 3 STATIC review PASS is recorded in reference/HDBET-NNUNET-FOLD0-REVIEW.json and reference/HDBET-NNUNET-FINETUNE-FOLD0-REVIEW.json.
- Prior server evidence reports derived preparation and inherited-plan 2D preprocessing completed for 95/95 cases, followed by ISLES-initialized full fine-tuning and a separate scratch control.
- Protocol: fixed 76 training / 19 development-validation cases, ordinary deep supervision, Dice+TopK10, early stopping, best-checkpoint evaluation in original full-volume space with `--f2-mode paper`.
- Fine-tuning uses the original Dataset508 checkpoint via `-pretrained_weights`, with fresh optimizer/scheduler/stopping state; private-data fine-tuned or CKPD checkpoints are not interchangeable source choices.

## Recorded development results
- Fixed-19 macro Dice: ISLES-initialized 74.7072%; scratch 71.5864% (difference +3.1208 percentage points). Recall difference +4.6669 points; scratch has better LCD/HD95. This supports an initialization overlap/recall advantage within this comparison, not uniform metric improvement or a skull-stripping causal claim.
- Evidence roots on server: `C:\lijialin\HD-BET_outputs\nnunet_hdbet3mm_isles508_finetune_fold0_trial01\evaluation_best_fold0` and `C:\lijialin\HD-BET_outputs\nnunet_hdbet3mm_scratch_fold0_trial01\evaluation_best_fold0`; retain summary_metrics.csv/json and case_metrics.csv.
- These are single-fold development observations, not formal five-fold OOF evidence. Source hash, exact cohort/protocol and case-level pairing must accompany subsequent comparisons.

## GenSeg and earlier engineering state
- Reference source is pinned to upstream `6480e178a5d44cb838f53f7951aae48138b8f5eb`; source inclusion alone is not effective augmentation.
- GenSeg30, staged G/D-to-joint training, preview and diagnostic tooling have historical synthetic/static validation and independent review; no current full-suite rerun is implied.
- Reviewed GD500/joint previews remained gray. Completed fixed-slice diagnostic receipt reports 100 steps, complete=true, weights_saved=false; this is diagnostic evidence only. BN/optimizer causes remain hypotheses; do not expand failed outputs by default.
- Standalone PlainConvUNet/H2Former and Lite-UPerNet engineering capabilities remain historical implemented work; their earlier synthetic tests do not establish clinical benefit or current checkout-wide passing status.

## Scientific and operational invariants
- Dataset501 contains 95 cases (94 was a typo); preserve existing patient-level five-fold splits. Ordinary fold-0 is 76/19; GenSeg inner training/development is 61/15 within the 76 training cases.
- Raw images/labels, fixed splits and old checkpoints remain protected. Derived masks/images/predictions/results use separate roots; GT never influences inference inputs; prediction-guided Stage 2 training requires OOF predictions.
- HD-BET may resample internally for mask prediction; native extracted DWI retains original in-mask values and zeros the exterior. Equal grids do not establish registration.
- Current local experimental workflow remains STATIC ONLY unless separately authorized. Prefer newconda for Python; request direct execution approval as instructed. No runtime/model/data/test execution occurred in this memory update.
