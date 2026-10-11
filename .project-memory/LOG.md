# Log

## 2026-10-11 - Exploratory nnU-Net A/B package archived (branch-local)

- Changed: versioned reusable A/B scripts, frozen Trainer snapshots, public metadata, configuration template and focused tests. Private configuration and patient splits remain ignored local files.
- Validation: original archive independent review PASS after fixing Windows LF/CRLF drift; simulated Windows Git checkout matched all Trainer hashes. Source-check repair independently reviewed PASS; 10 focused engineering checks passed.
- Result: codex/nnunet-exclude15-ab archive only, not integrated into master. No real data, checkpoints or results published; no server training or formal performance claim. Production source/test implementation delegated; parent changed only Git attributes, documentation and branch-local coordination records.
- Next: sync an isolated server checkout, copy private configuration/split snapshots, preserve aborted outputs and use a new results_parent before metadata checks and paired training.

## 2026-10-11 - External Trainer source check repaired (branch-local)

- Changed: tools/nnunet_exclude15_ab retains its expected external Trainer module before nnU-Net lookup; verifies the resolved source path and exact returned class identity. Trainer snapshots, split and training/evaluation semantics remain unchanged.
- Validation: newconda actual-import-cleanup regressions and affected integrity tests passed 10/10; original TypeError reproduced before repair. Independent read-only review of source SHA256 A42E8B3B8CF5E3F56C1EF19E03961250DFE13EAF60AEE872AE659EAF1A96ACEF returned PASS.
- Result: local engineering fix accepted and included in this branch archive; not integrated into master. No real data, model, checkpoint or training execution.
- Next: transfer the corrected run_experiment.py to the existing server package, preserve the aborted run, choose a new isolated results_parent and run metadata checks before paired A/B rerun. Canonical integration-branch memory has no tracked files and was not initialized by this task.

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
