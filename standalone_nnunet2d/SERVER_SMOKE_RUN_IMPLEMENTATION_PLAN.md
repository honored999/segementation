# Server Smoke Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` task-by-task.

**Goal:** Bind an explicitly confirmed experiment request to a one-case, one-epoch server smoke run.

**Architecture:** A dedicated runner will limit the supplied fold to its first train/validation cases, compose existing dataset/model/loss/epoch/checkpoint utilities, and return JSON-safe results. `experiment.py` will call it only for `--confirm-run`.

### Task 1: Specify synthetic runner behavior

- [ ] Add `tests/test_smoke_runner.py` with tiny injected model, synthetic one-item loaders, SGD, and temporary output-root tests proving one train/validation pass, metadata, and output confinement.
- [ ] Run the test to establish RED because `engine.smoke_runner` does not exist.

### Task 2: Implement the constrained runner

- [ ] Create `engine/smoke_runner.py` with `run_smoke_epoch` accepting injected model/loaders/loss/optimizer/device/output path, calling existing epoch functions once, saving a checkpoint/report only below outputs, and returning JSON-safe aggregates.
- [ ] Extend `experiment.py` so only `--confirm-run` constructs the fixed-fold one-case datasets and calls the runner; leave the default branch read-only.
- [ ] Run focused synthetic tests.

### Task 3: Document and verify

- [ ] Document server-only data access and the exact one-case/one-epoch limit.
- [ ] Run the complete pytest suite and `git diff --check`; do not commit automatically.
