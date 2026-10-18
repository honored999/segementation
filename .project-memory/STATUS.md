# Status

Updated: 2026-09-21

## Current state

- The active feature branch contains standalone 2D PlainConvUNet and H2Former
  model families with explicit model/supervision checkpoint identities.
- Existing H2Former uses single-output supervision; PlainConvUNet supports its
  default deep supervision and a matched single-output mode.
- Independent `h2former_lite_upernet` and `plain_conv_unet_lite_upernet`
  single-output variants are implemented at `53472ae`. Both use the shared
  lightweight PPM/FPN decoder while preserving the baseline model identities.

## Verified capabilities/results

- Fresh main-agent validation: standalone suite `424 passed in 47.67s`; root
  `tests/` suite `26 passed in 6.67s`.
- Independent Level 3 review of `bbe65b5..53472ae` returned PASS with no
  blocking findings.
- Synthetic complexity evidence shows both new decoders have fewer parameters
  than their corresponding baseline decoders. This is engineering evidence,
  not medical-performance evidence.

## Verified constraints

- Dataset501 remains the established DWI-only baseline.
- Preserve the existing patient-level five-fold split and original full-volume
  formal-evaluation semantics.
- Synthetic validation is engineering evidence only.

## Active work

- Lite-UPerNet implementation is accepted. No real-data training, preflight,
  fold evaluation, or formal five-fold OOF evaluation has been run.

## Branch-local GenSeg30 preparation (2026-10-14)

- codex/genseg-reference contains a synthetic-validated offline GenSeg30 slice-pool workflow and separate plain EarlyStopping Trainer. Adds ceil(30% of training-only 2D slices), retaining patient-level fold0 grouping and real validation geometry.
- Local engineering evidence: 74 combined synthetic and EarlyStopping tests passed; final independent 15-file snapshot PASS. Server compatibility, generated-image QA, training and quantitative benefit remain unverified. Original Blosc2 preprocessed backend required.

## Branch-local GD pretraining handoff (2026-10-18)

- User confirmed Dataset501 still95 (94 was a typo); fixed76/19 and inner61/15 remain unchanged.
- Added inner61-only G/D pretraining with frozen architecture/BN, completed gan.pt receipt and strict same-protocol handoff to fresh author U-Net/alphaAdam. Cold-start remains supported; interrupted-run resume is absent.
- Worker guarded CPU tests73 joint +15 target +63 offline passed separately; parent43 key tests and three CLI helps passed. Independent Level3 review PASS on the exact five-file snapshot in reference/GENSEG-JOINT-PRETRAIN-REVIEW.json.
- Target-PC native512x512 one-step cold-start pilot success was user-reported; user subsequently reported completed1+1 pretrain-to-joint CUDA receipts with finite alpha change; those remote files were not read locally. No performance/OOF claim.
- This is feature-branch state only; master canonical memory not modified.

## Branch-local joint preview (2026-10-18)

- Added completed-joint-checkpoint preview: only G inference, saved learned alpha/scaling, deterministic first4 inner61 cases, native four-panel PNG and provenance manifest. No inner15/external19 image reads or training/data-export changes.
- Guarded worker32 focused/105 joint affected PASS; parent32 focused and CLI PASS. Parent synthetic visual inspection and final exact-three-file independent Level3 PASS; receipt reference/GENSEG-JOINT-PREVIEW-REVIEW.json.
- Target-PC real-patient/CUDA preview not run locally. User can view PNGs after pulling branch and executing PREVIEW-RUNBOOK. PNG output is qualitative only, not a generated training dataset or formal metric.

## Branch-local multi-step preparation (2026-10-18; static only)

- Added synchronize/unused-cache release before unchanged per-step guard; fixed allocator ceiling to minimum initial/pre-model budget. No resource thresholds or numerical/data settings changed; no cooldown/retry.
- Latest user instruction: local static review only. Three files AST/diff checks and independent Level3 PASS(STATIC); pytest/model/CUDA/numerical execution NOT RUN. Pre-override test start interrupted without result.
- Proposed observation budget: target-PC3+3 first, then new GD500 and joint100 plus same-inner61 previews. Server multi-step success and image-quality benefit unverified; no resume/periodic recovery added.
