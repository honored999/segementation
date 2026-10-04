# Log

## 2026-10-04 - Configurable-depth ordinary PlainConvUNet Trainer

- Changed: one separate external Trainer reads encoder_last_stage from saved plans/configuration, constructs retained encoder stages and corresponding official UNetDecoder directly, and inherits TopK10/early-stopping/single-output policy. Added short derived-plans/server-command instructions; old UPerNet Trainers unchanged.
- Validation: independent exact-snapshot static review PASS; source/diff/hash inspection only. User explicitly requested no tests; no tests/imports/model construction/runtime validation were executed. Earlier NoStage7 test results do not apply to this new Trainer.
- Result: source implementation completed by a leaf subagent. Main edited only branch-local records. Based on f1b9ca8; user subsequently authorized committing and pushing this accepted addition. Canonical master memory unchanged. Raw data, source plans and fixed splits untouched.
- Next: deliver source, create independent derived plans on server, and compare ordinary-decoder depths with fresh per-configuration outputs and original full-volume validation. Runtime compatibility and performance remain unverified.

## 2026-10-03 - NoStage7 UPerNet branch-local engineering acceptance

- Changed: separate external Trainer removes only returned encoder stage7 and fixes decoder inputs(1,3,5,6),preserving baseline policy and plans. Existing Trainer source untouched.
- Validation: newconda CPU synthetic10passed; affected original UPerNet25passed; independent exact-snapshot read-only Level3 PASS. Per-command resource preflight below80percent.
- Result: local engineering accepted; no real-data/checkpoint/CUDA validation,training or formal evaluation. Production/test implementation delegated; main edited coordination and branch-local memory only. Local engineering stage had no commit,push or master merge; user subsequently authorized committing and pushing this branch.
- Next: training-host compatibility/resource checks and fresh fold0 final-checkpoint/full-volume development comparison; five-fold OOF required for formal conclusions.

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
