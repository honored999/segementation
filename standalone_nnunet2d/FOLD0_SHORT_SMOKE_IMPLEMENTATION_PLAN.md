# Fold 0 Short Smoke Implementation Plan

**Goal:** Execute the bounded two-epoch fold 0 smoke workflow and preserve validated artifacts.

1. Add synthetic tests for bounded batch iteration, latest/best checkpoint selection, and restored-mask equality.
2. Implement a `fold0_short_smoke.py` command using fixed split IDs, max eight train batches, first three validation cases, and smoke-only SGD.
3. Save latest/best checkpoints with epoch, global step, fold, best Dice, config, and `smoke_run_only` metadata.
4. Reload best state into a fresh model, verify same-input argmax identity, and run bounded case validation.
5. Emit training log, resolved config, CSV/summary, overlays, and best/worst reports; run full tests before server use.
