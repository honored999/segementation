# batchgeneratorsv2 Augmentation Alignment Implementation Plan

> **For agentic workers:** Execute serially with one editing agent. Do not commit or push unless explicitly requested.

**Goal:** Replace the handwritten 2D augmentation approximation with a standalone `batchgeneratorsv2` adapter that matches the server oracle transform configuration.

**Architecture:** The adapter constructs the server-confirmed 2D `ComposeTransforms` sequence directly from `batchgeneratorsv2`, using the plan patch size and mask-normalization configuration. It accepts one channel-first image/segmentation pair and a seed, returns NumPy arrays, and is used by both formal patch training and transform capture. No `nnunetv2` trainer dependency is added.

**Tech Stack:** Python, NumPy, Torch, batchgeneratorsv2, pytest.

---

### Task 1: Declare and test the transform adapter

**Files:**
- Modify: `standalone_nnunet2d/requirements.txt`
- Modify: `standalone_nnunet2d/training/official_augmentation.py`
- Modify: `standalone_nnunet2d/tests/test_official_augmentation.py`

- [ ] Write a failing test that injects the official 2D configuration, requires `batchgeneratorsv2`, and asserts channel-first Torch input plus NumPy image/label output.
- [ ] Run the focused test and confirm it fails because the handwritten adapter cannot build the official transform.
- [ ] Add a pinned `batchgeneratorsv2` requirement and replace the handwritten composition with the exact server-confirmed transform sequence. Derive rotation, patch size, and `use_mask_for_norm` from the provided plan configuration; seed NumPy immediately before invoking the transform.
- [ ] Run the focused test and `test_official_augmentation.py` until green.

### Task 2: Wire training and capture to the adapter

**Files:**
- Modify: `standalone_nnunet2d/training/formal_dataset.py`
- Modify: `standalone_nnunet2d/standalone_capture.py`
- Modify: `standalone_nnunet2d/tests/test_formal_dataset.py`
- Modify: `standalone_nnunet2d/tests/test_standalone_capture.py`

- [ ] Write failing tests showing formal patches and standalone capture supply the plan-derived configuration and a deterministic seed to the adapter.
- [ ] Run the focused tests and confirm expected failure.
- [ ] Pass the plan configuration into the adapter, retain the existing foreground patch sampler, and keep inference untouched.
- [ ] Run the focused tests until green.

### Task 3: Verify and hand off

**Files:**
- Test: `standalone_nnunet2d/tests/`

- [ ] Run the full pytest suite, py_compile for changed modules, and `git diff --check`.
- [ ] Provide server commands to install the pinned package, regenerate transform oracle/standalone artifacts in fresh output folders, and run the parity report.

### Task 4: Convert one fixed official checkpoint

**Files:**
- Create: `standalone_nnunet2d/tools/convert_official_checkpoint.py`
- Create: `standalone_nnunet2d/tests/test_convert_official_checkpoint.py`

- [ ] Write failing tests for semantic key mapping from official nnU-Net names to standalone names. Cover encoder convolutions/norms, decoder convolutions/norms, transposed convolutions, and segmentation heads.
- [ ] Write failing tests that reject missing keys, unexpected standalone targets, shape mismatches, and mapped counts other than exactly 148 tensors.
- [ ] Implement a converter that reads only `network_weights`, maps each target key by its semantic path, validates every tensor shape against `PlainConvUNet2D(load_model_config(), deep_supervision=False).state_dict()`, and writes a minimal standalone `format_version=1` checkpoint with `model_state_dict` and pending metadata.
- [ ] Record the source checkpoint SHA-256, fold, mapping policy, and `official_alignment_pending`; never copy optimizer state and never mark the converted file aligned.
- [ ] Run the focused converter tests and `py_compile` until green.

### Task 5: Expose standalone inference capture CLI

**Files:**
- Modify: `standalone_nnunet2d/standalone_capture.py`
- Modify: `standalone_nnunet2d/tests/test_standalone_capture.py`

- [ ] Write failing CLI tests for `--mode inference`, requiring oracle root, raw root, converted checkpoint, plans, output root, device, and slice batch size.
- [ ] Keep transform capture as the default mode for backward compatibility; inference mode must call the existing `capture_standalone_inference` function without changing inference behavior.
- [ ] Run the complete standalone capture tests and `py_compile` until green.

### Task 6: Final parity verification handoff

**Files:**
- Test: `standalone_nnunet2d/tests/`

- [ ] Run the full pytest suite, compile every changed Python module, and run `git diff --check`.
- [ ] Provide server commands to convert the fixed official fold-0 checkpoint, capture oracle inference, capture standalone inference with the converted checkpoint, and generate the inference parity report.
- [ ] Retain `official_alignment_pending` unless both the already-passed transform report and the new inference report have `status: "passed"`.
