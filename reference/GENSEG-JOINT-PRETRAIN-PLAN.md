# G/D pretraining handoff implementation plan

**Goal:** Add an explicitly budgeted inner61-only G/D pretraining phase and load its completed same-protocol checkpoint into a fresh-U-Net joint run.
**Architecture:** Reuse existing protected TargetDataset/Cohort, pinned networks and functional staged SGD logic. Keep cold-start joint behavior. A distinct completed-pretrain handoff is initialization, not interrupted-run resume.
**Tech stack:** existing Python/PyTorch; newconda; guarded serial CPU synthetic tests. No dependency installation or upstream edits.

Approved by user on 2026-10-18: only G/D pretraining followed by joint training; first try. Total95 confirmed; original76/19 and inner61/15 unchanged. Long-run budgets are not approved. First target-PC engineering command is 1 pretrain update then 1 joint update in fresh separate directories. Parent writes docs/Git acceptance; leaf owns production/tests. No nested agents. Commands use require_escalated directly per user.

## Task 1: Frozen-architecture GAN update

Files: experiments/genseg_joint/core.py and tests/test_pretrain.py (new).
- [x] RED: a tiny actual pinned generator/discriminator pretrain step must update finite G/D weights, keep both architecture tensors unchanged, never train S, and preserve live G/D state on invalid/nonfinite input/candidate failure.
- [x] Implement the smallest reusable staged ordinary GAN SGD step using the current masked GAN+100 L1 loss, frozen BN and current G/D rates. No virtual/meta gradients during pretraining; no alpha optimizer update. Reuse current helpers; avoid a second divergent GAN loss implementation where feasible.
- [x] GREEN: use run_guarded.ps1 with the exact new test file. No architecture/search/model/preprocessing policy change.

## Task 2: Completed-pretrain receipt and strict handoff

Files: experiments/genseg_joint/pretrain.py (new if needed), runtime.py, tests/test_pretrain.py.
- [x] RED: completed G/D checkpoint roundtrip; reject joint.pt as pretrain, incomplete receipt, wrong hash, nonmatching target/source/split/scaling/model/optimizer policy and malformed/nonfinite architecture or model tensors BEFORE live weight mutation.
- [x] Add pretrain orchestration with explicit steps, same resource/UUID checks and immutable-source audits, no outer15/19 patient arrays, native full planes, inner61 scaling and fixed alpha. Save G/D, architecture tensors, optimizer policy/count, seeds/RNG and protocol/source identity. Completion receipt LAST; fresh protected output only.
- [x] Add completed-pretrain loading for joint initialization. Require explicit expected checkpoint SHA256; use safe tensor loading and schema validation. Validate complete metadata and all model-state tensors before committing; no partial mutation on failure. Fresh U-Net, fresh alpha-Adam moments, fresh explicitly recorded joint RNG semantics. No arbitrary previous joint checkpoint, old76 weights or resume.
- [x] Exercise tiny actual-network pretrain -> serialize -> load -> same production joint step. Use metadata-only fixtures for rejection tests, without large checkpoint duplication.

## Task 3: CLI and regression compatibility

Files: experiments/genseg_joint/cli.py, runtime.py and tests/test_runtime.py/test_pretrain.py.
- [x] Expose pretrain with explicit positive --steps and normal source/targets/output/GPU arguments; train accepts paired --pretrained-run / --pretrained-sha256. Keep old --pretrain-steps 0 accepted; reject ambiguous combined modes. check-mode support may audit handoff metadata/hash but constructs no model/CUDA and reads no patient image/label arrays.
- [x] Validate mode/hash/supported budget/seed before expensive construction or output writes. Pretrain step counters and joint step counters remain distinct; phase/classification in protocol and receipt unambiguous.
- [x] Focused regressions then affected joint/genseg3mm/genseg30 tests through existing run_guarded.ps1 (fresh resource preflight each invocation, newconda, single CPU thread, no CUDA tests locally).

## Task 4: Parent acceptance and delivery

Files: reference/GENSEG-JOINT-PRETRAIN-DESIGN.md, this plan, new PRETRAIN-RUNBOOK.md and review receipt. No canonical master-memory write from this feature branch; update branch-local memory only with verified feature status, clearly scoped, if appropriate.
- [x] Parent inspects final diff and test evidence, records exact changed-file hashes.
- [x] Separate read-only leaf reviewer inspects exact snapshot for data roles, numerical behavior, checkpoint handoff, fresh outputs and cold-start compatibility. Resolve blockers through implementation owner; rerun affected tests/re-review changed risk paths.
- [x] Write CMD commands for C:/lijialin/segementation/.worktrees/genseg-reference using existing source/targets/baseline paths and fresh outputs beneath C:/lijialin/HD-BET_outputs. Include local-first commit/push and data-computer git pull --ff-only. Explicit hash extraction before handoff. Only 1+1 engineering updates, no long-run/effect claim.
- [ ] Inspect status/diff/check, stage explicit intended files, inspect cached diff/check, commit and push codex/genseg-reference as user authorized. Do not stage raw images, splits, checkpoints, caches or unrelated edits. Verify remote commit and final Git state.

Worker HANDOFF must give COMPLETED/BLOCKED/FAILED, exact files, RED/GREEN/affected command results, resource evidence, assumptions/limits and remaining risks. Parent must not claim CUDA/real-data phase handoff succeeded locally. Existing reported target-PC cold-start pilot success is separate historical evidence.
