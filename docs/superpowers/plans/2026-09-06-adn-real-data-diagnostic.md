# ADN real-data diagnostic Implementation Plan

> **For agentic workers:** This plan is executed inline by the sole authorized agent; no subagents, commits, or pushes are permitted. Steps use checkbox syntax for TDD tracking.

**Goal:** Add a minimal synthetic-tested NIfTI orientation adapter, ADN diagnostic checkpoint/training helpers, and isolated QC CLI without changing the existing ADN network or segmentation pipeline.

**Architecture:** `nifti_adapter.py` owns strict global orientation assignment and exact array-only canonicalization. `diagnostic.py` owns normalization, split reading, output safety, checkpoint contract, loss-only training, and QC data preparation. Two thin tools expose argparse CLIs and never access labels.

**Tech Stack:** Python, NumPy, SimpleITK-backed `NiftiVolume`, PyTorch, matplotlib, pytest.

---

### Task 1: RED tests and design evidence

**Files:**
- Create: `standalone_nnunet2d/tests/test_adn_nifti_adapter.py`
- Create: `standalone_nnunet2d/tests/test_adn_diagnostic.py`
- Create: this spec and plan

- [x] Encode identity, permutation/flip reversibility, case005-style direction, accepted oblique direction, all orientation rejection gates, split-only/no-label behavior, checkpoint roles/history/contract, QC artifacts/fields/input immutability, required CLI arguments, and output safety.
- [x] Run focused tests with `D:\Anaconda\envs\newconda\python.exe -m pytest ... -q`; the expected RED is missing `nifti_adapter`/`diagnostic` modules.

### Task 2: Implement strict adapter

**Files:**
- Create: `standalone_nnunet2d/brain_alignment/nifti_adapter.py`
- Modify only if needed: `standalone_nnunet2d/brain_alignment/__init__.py`

- [ ] Add `OrientationError(reason, details)`, exhaustive six-permutation assignment, finite/orthogonal/uniqueness/dominance/margin checks, canonical labels, and reversible provenance.
- [ ] Canonicalize with `np.transpose` plus `np.flip` only; preserve raw geometry and exact inverse restoration.
- [ ] Run the adapter-focused tests and confirm GREEN.

### Task 3: Implement shared diagnostic logic and checkpoint contract

**Files:**
- Create: `standalone_nnunet2d/brain_alignment/diagnostic.py`

- [ ] Add per-volume finite normalization, explicit fold-0 split loading, path-overlap/non-empty output guards, ADN contract validation, compact checkpoint persistence/loading, and loss-only training with batch size one.
- [ ] Save latest/best checkpoints and JSON/JSONL history with reproducibility metadata; never read `labelsTr`.
- [ ] Run the diagnostic unit/checkpoint tests and fix only contract failures.

### Task 4: Implement training and QC entry points

**Files:**
- Create: `standalone_nnunet2d/tools/train_adn_alignment.py`
- Create: `standalone_nnunet2d/tools/adn_alignment_qc.py`

- [ ] Expose required argparse inputs and explicit output safeguards.
- [ ] Produce three deterministic four-panel PNGs and `summary.json` with raw/scaled parameters, `rz_degree`, `tx`, before/after flip losses, raw geometry, canonicalization decision, slice indices, and checkpoint.
- [ ] Run CLI-focused tests and inspect generated synthetic artifacts.

### Task 5: High-risk verification

- [ ] Run the focused changed tests, `python -m py_compile` on new Python files, `git diff --check`, and the relevant existing ADN tests.
- [ ] Perform a read-only self-review of orientation, checkpoint, data-access, and output-boundary paths.
- [ ] Confirm no changes to `adn_transform.py`, segmentation production files, labels, raw data, or `.project-memory`; report final `git status` without committing.
