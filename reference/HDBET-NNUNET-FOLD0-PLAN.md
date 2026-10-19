# HD-BET 3mm nnU-Net fold0 implementation plan

Approved 2026-10-18: native Dataset501 DWI -> fixed image-only HD-BET2.0.1/defaultTTA/in-plane3mm/empty-plane-retain -> separate full95 dataset -> inherited Dataset501 2d plans -> TopK10+early stopping with normal deep supervision, fixed original76/19. Existing19 are development validation, not pristine test. No GenSeg images or weights.

## Ownership and constraints
- Parent: protocol, runbook, acceptance, branch memory and Git. Leaf dataset worker: experiments/hdbet_nnunet only. Leaf Trainer worker: new combination Trainer + focused tests only. Separate read-only reviewer. No nested agents.
- Local STATIC ONLY: AST/text/diff review; no tests/import probes/models/CUDA/patient arrays/resource probes. Server commands must clearly separate read-only check, prepare, preprocess, preflight and train. Record runtime validation as deferred.
- Root checkout codex/genseg-reference starts a5c35d9 clean. Preserve all original data, splits, caches and weights. Fresh output root C:\lijialin\HD-BET_outputs\nnunet_hdbet3mm_fold0_trial01; separate nnUNet_raw/preprocessed/results below it. Reuse Dataset501 name in isolated environment roots, avoiding dataset-ID collisions and unnecessary plan changes.

## Task 1 dataset builder leaf
- Create experiments/hdbet_nnunet/__init__.py and cli.py plus at most one utility module if necessary. Read root AGENTS. Reuse existing genseg3mm raw-geometry, mask mapping, expand_keep_mask_3mm, HD-BET runtime/command and selected-GPU guard helpers; do not modify old modules.
- CLI check/prepare with --raw-dataset, --preprocessed-dataset, --targets, --targets-sha256 and --output, --gpu default1. Fixed canonical5fold semantic split equality (CRLF safe), exact95 IDs and Dataset501 DWI labels metadata. Validate pinned targets complete/recipe and all76 train cache source raw hashes/native mask receipts, confinement. Check no patient arrays/model/CUDA/output creation; hash/header metadata allowed as documented. Actual prepare refuses any existing output.
- Actual prepare: resource guard before arrays/each case; reuse hash-validated original-space HD-BET masks for76; run identical pinnedHD-BET weights defaultTTA only for19. Apply native3mm rule image-only; zero outside; exact raw geometry/dtype/scaled-value safety. Copy labels byte-exact; no GT-based repairs. Store masks, per-case provenance, all-slice QA incl native empty planes; label coverage is report-only and never changes mask.
- All95 NIfTI full native grid output, isolated Dataset501_StrokeLesion/{imagesTr,labelsTr,dataset.json}. Copy original plans/splits/dataset metadata into isolated preprocessed Dataset501_StrokeLesion; baseline fingerprint is copied both to provenance and the required preprocessed compatibility path, explicitly inherited fixed-baseline metadata rather than a measured fingerprint of the derived images. No replanning; geometry target spacing/patch/batch/network/normalization settings retained. Any changed size/cropping is declared preprocessing consequence. No training launch. Complete receipt only after saved readback/source integrity recheck.
- Provide useful focused tests as files but DO NOT run them locally. Check should audit safety/version/identity without reading patient arrays or downloading weights; prepare guard before model/arrays. Any blocker fail with diagnostics; retain incomplete outputs without resume/overwrite. No destructive actions.

## Task 2 Trainer leaf
- Create nnunet_ext_trainers/nnUNetTrainerTopK10EarlyStopping.py. Minimal cooperative multiple inheritance of existing nnUNetTrainerEarlyStopping and nnUNetTrainerTopK10; ordinary deep supervision retained. Preserve _build_loss binding and early stop lifecycle/checkpoints. No existing file edits. Focused contract tests as new file only; no execution locally.

## Task 3 parent review + server runbook
- Inspect ownership diff for geometry, data boundaries, fixed masks/hash, metadata-only check, separate roots, full95 coverage, splits installed before training, inherited plans unchanged, Trainer discovery and validation_best flags per installed2.8.1 source.
- Document Windows CMD nnunet5090 paths and new environment roots AFTER builder; no train before prepare/QA/preprocess/fixed split preflight. nnUNetv2_preprocess -d501 -c2d -np2 (NOT plan_and_preprocess). set nnUNet_extTrainer correct repo path; GPU1 explicit selection; no -pretrained_weights or --c. Original full-volume19 evaluation/checkpoint_best separate from training monitor.
- Independent Level3 STATIC reviewer assesses final file snapshot and docs. Fix through original workers; parent source implementation prohibited. AST/diff checks only; tests deferred server-side. Update branch memory only; master untouched. Stage explicit files, inspect diff/cached/check, commit and push per standing authorization; server pull --ff-only.
## Continuation 2026-10-19
- Trainer leaf completed both new files with static-only evidence. Dataset and reviewer channels hit usage limits before any dataset file was written; same workers resumed after explicit user continuation. Parent remains coordination/docs/Git only.
- Implementation mode and normal deep supervision were explicitly approved; no additional protocol approval required.

## Final command boundaries
- check / ready / preprocess-check: stdlib metadata and file hashes only; no resource query, patient arrays, models or output creation. prepare imports native processing only after guard.
- preprocess-check requires exact isolated roots and absent target configuration; run `preprocess-check && guard && nnUNetv2_preprocess` so failures prevent official destructive rebuilds.
- ready refuses any existing Trainer fold0 directory. trainer-check is SERVER ONLY: resource guard then actual official resolver and method-source identity; no Trainer instance/model/checkpoint.
- Fresh training chain: ready && guard && trainer-check --gpu1 && guard && official train with --val_best. Fixed19 assertion && guard && original-space evaluation; no automatic continuation after failed checks.
- Native NIfTI retains original copied qform/sform encodings; do not reconstruct quaternion through set_qform. Identity-scaled numeric DWI only; unsupported scaling fails rather than silently requantizing.
- QA and optional GT coverage report are train76 only. Original fingerprint is copied under provenance and the official Trainer-required preprocessed compatibility filename; both remain explicitly inherited fixed-baseline metadata, not a derived-fingerprint measurement.