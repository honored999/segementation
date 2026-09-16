# Foreground50 DetailRefine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and validate an isolated Foreground50 Trainer whose only architecture change is zero-initialized residual refinement of the final full-resolution decoder feature.

**Architecture:** Build the official initialized network first, then replace only its highest-resolution segmentation head with a local serializable wrapper around the original head. The wrapper derives all structural types and channel counts from the constructed network and leaves every lower-resolution deep-supervision head untouched.

**Tech Stack:** Python, PyTorch 2.11.0+cu126, nnunetv2 2.8.1, dynamic-network-architectures 0.4.4, pytest.

---

### Task 1: Lock the architecture contract with failing tests

**Files:**
- Create: `nnunet_ext_trainers/tests/test_trainer_foreground50_detail_refine.py`

- [ ] Add synthetic-constructor tests for output shape/order, zero projection,
  baseline parity, paired backbone initialization, staged gradients, strict state
  loading, external discovery, deep-supervision switching and predictor calls.
- [ ] Run the focused test and record the expected import/feature failure (RED).

### Task 2: Implement the minimal module and Trainer

**Files:**
- Create: `nnunet_ext_trainers/detail_refinement.py`
- Create: `nnunet_ext_trainers/nnUNetTrainerForeground50DetailRefine.py`

- [ ] Implement the fixed-width residual feature refiner and segmentation-head
  replacement without hooks or site-package edits.
- [ ] Override only `build_network_architecture`, delegating baseline construction
  to `nnUNetTrainerForeground50` before attaching the head.
- [ ] Run the focused tests to GREEN.

### Task 3: Document executable experiment usage

**Files:**
- Modify: `nnunet_ext_trainers/README.md`

- [ ] Document verified dependency/runtime contracts and exact training, resume,
  validation and prediction setup with isolated result names.
- [ ] Document the fixed fold-0 A/B comparison and its engineering-only screen.
- [ ] Separate local verification from server-only plans/checkpoint/resource checks.

### Task 4: Validate and measure

- [ ] Run focused tests, related external-Trainer tests, and affected tests under
  `newconda` with an ASCII pytest temp root.
- [ ] Instantiate the real reference-plan constructor and report measured baseline
  and refined parameter counts.
- [ ] Probe matched GPU measurement eligibility without changing batch/patch.
- [ ] Run `git diff --check`, inspect status/diff, and create one explicit commit.

### Task 5: Independent Level 3 review and acceptance

- [ ] Send the exact final commit to a separate read-only reviewer context.
- [ ] Resolve any BLOCKING findings, revalidate, recommit and re-review.
- [ ] Synchronize only verified engineering state to project memory if appropriate;
  never record an unrun experiment as a result.
