# Log

## 2026-10-14 - GenSeg reference source snapshot (branch-local)

- Changed: downloaded 342 upstream source/config/documentation files under
  `reference/GenSeg` on `codex/genseg-reference`, pinned to upstream
  `6480e178a5d44cb838f53f7951aae48138b8f5eb`; preserved license and provenance.
- Validation: all upstream Git blob hashes matched; local SHA256 and file count
  verified. Datasets, binaries, caches and nested Git metadata excluded.
- Result: reference snapshot recorded on its dedicated branch; no dependency installation, source
  integration, training or formal evaluation. Canonical master memory untouched.
- Next: inspect/adapt only under a separately defined implementation scope.

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
