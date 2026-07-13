# Optical DeepLab2D Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated 2D DWI binary-lesion experiment that fairly compares ideal optical-convolution DeepLabV3+ with an electronic baseline.

**Architecture:** A manifest-first dataset layer validates and groups samples by patient. Model wrappers share one DeepLabV3+ factory while differing only in their input front end. Training, evaluation and inference use resolved configuration/checkpoint metadata so server runs are reproducible.

**Tech Stack:** Python 3.11, PyTorch, segmentation-models-pytorch, Albumentations, NumPy, Pillow, PyYAML, pytest.

---

### Task 1: Package scaffold and configuration

**Files:** Create `optical_deeplab2d/__init__.py`, `configs/hybrid_ideal.yaml`, `configs/electronic_baseline.yaml`, `requirements.txt`, `tests/test_config.py`.

- [ ] Write a failing test asserting both YAML files expose the same training/data defaults and distinct `model.type` values.
- [ ] Run `conda run -n newconda python -m pytest optical_deeplab2d/tests/test_config.py -q`; expect import/config failure.
- [ ] Add the package markers, conservative server dependency declarations and two resolved-default templates.
- [ ] Re-run the test; expect pass.

### Task 2: Manifest validation and patient folds

**Files:** Create `datasets/dataset_2d.py`, `datasets/split.py`, `datasets/inspect_dataset.py`, `tests/test_dataset.py`.

- [ ] Write failing tests for manifest pairing, direct `patient` identity, mask binarization, dimension mismatch rejection and a five-fold split with no patient overlap.
- [ ] Run the focused dataset tests; expect missing-module failures.
- [ ] Implement typed manifest records, image readers, paired sample validation, statistics and deterministic `seed=2026` grouped folds persisted as JSON.
- [ ] Re-run focused tests; expect pass.

### Task 3: Normalization and paired augmentation

**Files:** Create `datasets/transforms.py`, extend `tests/test_dataset.py`.

- [ ] Write failing tests proving training percentile normalization is stable, resize preserves binary masks and validation uses no random transform.
- [ ] Implement configured robust normalization and Albumentations-backed paired transforms with an import-time actionable dependency error.
- [ ] Re-run focused tests; expect pass.

### Task 4: Models and loss

**Files:** Create `models/optical_conv.py`, `models/hybrid_deeplabv3plus.py`, `models/electronic_deeplabv3plus.py`, `training/losses.py`, `tests/test_model_shape.py`, `tests/test_optical_gradient.py`, `tests/test_loss.py`.

- [ ] Write failing tests for signed 5x5 optical weights, logits shape restoration on a mocked/available backend, nonzero finite optical gradients, electronic 1-to-3 repeat, and finite loss for required empty/nonempty cases.
- [ ] Implement a shared SMP factory with explicit MobileNetV2-to-ResNet18 fallback reporting, exact hybrid front end, logits-only output, and BCE/Dice loss with per-sample Dice.
- [ ] Run focused model/loss tests on small tensors; expect pass. Do not run 512x512 checks locally.

### Task 5: Reproducible training and checkpoints

**Files:** Create `training/seed.py`, `training/checkpoint.py`, `training/trainer.py`, `train.py`, `tests/test_checkpoint.py`.

- [ ] Write failing tests for checkpoint metadata roundtrip, positive/negative sampling weights, optimizer parameter groups and resume config mismatch detection.
- [ ] Implement seed control, clipped pos-weight calculation, AMP-safe train/validation loops, patient-level best metric selection, OOM advice, resolved configuration and exact checkpoint payload.
- [ ] Run focused tests plus CLI `--help`; expect pass without training.

### Task 6: Evaluation, visualization and inference

**Files:** Create `evaluation/metrics.py`, `evaluation/visualization.py`, `evaluation/postprocess.py`, `evaluate.py`, `infer_image.py`, `compare_models.py`, `tests/test_metrics.py`.

- [ ] Write failing tests for global/image/patient aggregation, specified empty-mask Dice rules and checkpoint-driven inference metadata.
- [ ] Implement metric exports, selected validation/kernel visualization, inference outputs and comparison of saved summary JSON files.
- [ ] Run focused tests and CLI `--help`; expect pass without checkpoint/data training.

### Task 7: Documentation and server handoff

**Files:** Create `README.md`, extend test files as necessary.

- [ ] Document data layout, manifest ID rule, dependency installation, server commands, local-test boundary and exact required server preflight sequence.
- [ ] Run `python -m compileall optical_deeplab2d` and the complete local test suite in `newconda`.
- [ ] Inspect `git diff --check` and commit the completed module only after fresh evidence is recorded.
