# ADN Acquisition-Preserving LR Canonicalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace strict world-axis ADN canonicalization with an acquisition-plane-preserving, safely assigned left-right model-space contract.

**Architecture:** Keep canonical `D` on source voxel z, assign canonical `W` only from the uniquely separated in-plane voxel axis with the largest LPS X component, and use the remaining in-plane axis for `H`. Apply only transpose and flips: `D` to LPS +Z, `W` to LPS +X, and `H` so `(W x H) dot D > 0`; retain exact inverse operations and source geometry as provenance.

**Tech Stack:** Python, NumPy, PyTorch checkpoint metadata, pytest, SimpleITK-compatible LPS direction conventions.

---

### Task 1: Establish RED coverage for the new orientation contract

**Files:**
- Modify: `standalone_nnunet2d/tests/test_adn_nifti_adapter.py`
- Modify: `standalone_nnunet2d/tests/test_adn_diagnostic.py`

- [x] Add synthetic tests for fixed source-z depth, LR in voxel x/y, LR sign, right-handed H, strong AP-SI obliquity, z-as-LR rejection, LR margin rejection, malformed directions, exact restoration, and JSON provenance.
- [x] Update checkpoint/QC assertions to require `acquisition_preserving_lr` and model-space `tx`/`rz` semantics.
- [x] Run `python -m pytest -q standalone_nnunet2d/tests/test_adn_nifti_adapter.py standalone_nnunet2d/tests/test_adn_diagnostic.py` in `newconda` and verify expected failures originate from the old strict contract.

### Task 2: Implement the minimal acquisition-preserving adapter

**Files:**
- Modify: `standalone_nnunet2d/brain_alignment/nifti_adapter.py`
- Modify: `standalone_nnunet2d/brain_alignment/diagnostic.py`
- Modify only if needed: `standalone_nnunet2d/tools/adn_alignment_qc.py`
- Modify only if needed: `standalone_nnunet2d/tools/train_adn_alignment.py`

- [x] Replace global world-axis assignment and `cos(20 degree)` gate with finite/orthogonality checks plus unique best LR axis and `LR margin >= 0.20`.
- [x] Reject voxel z as LR; preserve source z as `D`; construct permutation only between source x/y for `H/W`.
- [x] Flip `D` to LPS +Z, `W` to LPS +X, and `H` only to satisfy `(W x H) dot D > 0`.
- [x] Record all required source axes, LR measures, source direction vectors, AP/SI components/obliquity, operations, unchanged geometry, and exact inverse operations in JSON-safe provenance.
- [x] Update checkpoint/QC descriptions to identify `tx` as model-space LR and `rz` as acquisition/model in-plane rotation, not physical 3D registration.
- [x] Run the focused tests and verify GREEN.

### Task 3: Validate, review, document, and publish

**Files:**
- Modify: `docs/superpowers/specs/2026-09-06-adn-real-data-diagnostic-design.md`
- Modify: `.project-memory/STATUS.md`
- Modify: `.project-memory/NEXT.md`
- Modify: `.project-memory/LOG.md`

- [x] Run affected ADN tests, then the full repository pytest suite with an isolated temporary root.
- [x] Obtain independent read-only Level 3 review of orientation mathematics, `D=z`, LR assignment, reversibility, and leakage; apply a minimal fixer and re-review if BLOCKING.
- [x] After PASS, rerun fresh focused/affected/full validation as justified.
- [x] Update design and project memory with verified synthetic-only evidence and the server preflight continuation.
- [ ] Verify `adn_transform.py` is unchanged, inspect status/diffs/checks, stage explicit files, commit as `fix(adn): preserve acquisition plane for LR canonicalization`, and push `codex/adn-transform-alignment`.
