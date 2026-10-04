# Log

## 2026-10-04 - user-authorized branch delivery

- Changed: review status in README, Linux server resource-guarded CPU validation instructions and branch-local delivery state; no source/test changes.
- Validation: fresh precommit affected external-Trainer CPU suite 111 passed, exit 0.
- Scope: commit/push authorized; no merge. User executes server download/validation. Canonical memory untouched; server/CUDA/multi-rank/formal evidence pending.

## 2026-10-04 - independent re-review accepted, branch-local/unintegrated

- Reviewed: unchanged implementation snapshot 9cb7266a68b87cf77ce125f86318e2e6db51863610044f0489d2cba0368f39ba; all 37 files/24 ignored files and independent source hashes verified.
- Result: manual independent Level 3 PASS, original DDP P1 CLOSED, no new blocker. Independent tests 6 + 9 passed (exit 0); prior focused 61 and affected 111 passed.
- Changed after review: branch-local STATUS/NEXT/LOG and task-local acceptance/snapshot documentation only; source/tests untouched. No agents, commits, pushes, merges or real-data execution. Canonical memory untouched.
- Remaining: separately authorized integration and server/CUDA/multi-rank/real-data/performance/formal validation.

## 2026-10-04 — DDP P1 fix, branch-local/unintegrated

- Independent review: original snapshot BLOCKING on unused trailing encoder parameters.
- Changed: new Trainer conditional DDP reducer rebuilding and normalized checkpoint delegation; tiny CPU Gloo regressions, README and task-local re-review snapshot. No changes to old Trainers/shared decoder in this fix.
- Validation: RED reproduced second-forward failure; fix regressions 6 passed, focused 61 passed, affected 111 passed (exit 0). CUDA/multi-rank/performance/real data deferred.
- Result: implementation fix ready, independent re-review PENDING; no commit/push/merge/agents, canonical memory untouched.
- Next: user manually dispatches focused Level 3 re-review.

## 2026-10-03 — selected UPerNet stages, branch-local only

- Changed: resolved-plan stage selection, N-layer shared decoder, new network state identity/loading guard and exclusive plans-copy CLI. User explicitly assigned direct implementation; no agents.
- Validation: tiny synthetic CPU focused 55 passed; legacy official 25 passed; affected external suite 105 passed. Checkpoint/predictor/odd accounting evidence recorded in task notes.
- Result: implementation ready, uncommitted/unmerged; independent review PENDING. No server/CUDA/real-data/performance evidence. Canonical master memory absent in Git and untouched.
- Next: manual Level 3 review of exact source/SHA256 snapshot.

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
