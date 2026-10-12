# Log

## 2026-09-21 - Lite UPerNet variants accepted

- Changed: added isolated H2Former and PlainConvUNet Lite-UPerNet single-output
  variants, shared decoder, model/checkpoint contracts, tests, and complexity
  reporting through commit `53472ae`.
- Validation: fresh main-agent runs reported `424 passed` for
  `standalone_nnunet2d/tests` and `26 passed` for root `tests/`; independent
  Level 3 review returned PASS with no blockers.
- Result: implementation accepted as synthetic engineering evidence only; no
  real medical data, training, preflight, or formal evaluation was performed.
- Next: run an isolated experiment only if explicitly authorized.

## 2026-09-20 - Lite UPerNet implementation plan

- Changed: corrected PPM normalization for 1x1 pooled features and added the
  task-by-task TDD implementation plan.
- Validation: plan checked for spec coverage, placeholders, interface
  consistency, and scope; no production code or real-data run.
- Result: implementation remains not started and requires manual worker dispatch.
- Next: collect and inspect the implementation worker HANDOFF.

## 2026-09-20 - Lite UPerNet decoder design

- Changed: documented two independent single-output Lite-UPerNet model variants.
- Validation: design checked against current model, supervision, and checkpoint
  contracts; no production code or real-data run.
- Result: design approved in conversation and pending written-spec review.
- Next: create the implementation plan after user review.

## 2026-10-12 - CKPD branch engineering accepted

- Added opt-in external CKPD Trainer, source-train-only bounded calibrator, checkpoint contracts, tests and server runbooks in isolated codex/nnunet-ckpd-finetune based on 414cdac.
- Fresh parent focused/affected CPU suite: 51 passed in 22.23s; independent Level 3 review PASS after fixing five initial blockers.
- Production source/tests were delegated to leaf implementation/fixer workers; parent handled coordination and acceptance only.
- No real data, CUDA calibration/training, production checkpoint or formal evaluation; raw data/splits and main checkout unchanged. Project-memory updates here are branch-local; master integration/push not performed.

## 2026-10-12 - CKPD optional dataset name fix

- Server diagnostic confirmed valid Dataset508 plans with dataset.json name omitted. Corrected calibrator's optional-field guard without changing data or source identity contracts.
- Leaf fixer added missing/null/conflicting-name regressions: RED 2 failures / 1 pass; GREEN calibrator 10 passed. Parent affected CPU suite 30 passed in7.96s; independent focused review PASS.
- No production source/test edits by parent, no patient-data modifications or real calibration/training. State is branch-local, not master integration.
