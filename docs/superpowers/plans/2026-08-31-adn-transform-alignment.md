# ADN Transformation Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal, standalone, differentiable ADN transformation network for canonical model-space brain alignment.

**Architecture:** Keep parameter/matrix construction, warping, losses, and the encoder separate in `brain_alignment`.  Tests use generated tensors only and never touch dataset paths.

**Tech Stack:** Python, PyTorch, pytest.

---

### Task 1: Specify and prove the public geometry API

**Files:**
- Create: `standalone_nnunet2d/tests/test_adn_transform_alignment.py`

- [ ] Write tests for zero-to-identity matrices, both inverse multiplication orders, positive and negative x sampling translations, z rotation, identity/cube warp, 5-D shape preservation, batch support, and nearest labels.
- [ ] Run `D:\Anaconda\envs\newconda\python.exe -m pytest standalone_nnunet2d/tests/test_adn_transform_alignment.py -q` and verify collection fails because the new module does not exist.

### Task 2: Implement transform primitives

**Files:**
- Create: `standalone_nnunet2d/brain_alignment/__init__.py`
- Create: `standalone_nnunet2d/brain_alignment/adn_transform.py`

- [ ] Add a named `W`-axis flip, a six-range value object, ADN-order homogeneous matrices, explicit output-to-input sampling documentation, `affine_grid`/`grid_sample` warp, and independent losses.
- [ ] Re-run the focused tests and make all geometry/warp/loss tests pass.

### Task 3: Implement the encoder and result API

**Files:**
- Modify: `standalone_nnunet2d/brain_alignment/adn_transform.py`
- Modify: `standalone_nnunet2d/tests/test_adn_transform_alignment.py`

- [ ] Add the four official downsampling/residual stages, the adaptive-pool MRI adaptation note, near-identity output-head initialization, minimum-shape validation, and an inference result object.
- [ ] Add and run CPU tests for raw/scaled parameter bounds, variable shapes, finite independent loss, and backward gradients.

### Task 4: Verify scope

**Files:**
- Verify only: the three new module/test files and these design documents.

- [ ] Run only the focused synthetic test file with newconda, inspect `git diff --check`, and do not commit.
