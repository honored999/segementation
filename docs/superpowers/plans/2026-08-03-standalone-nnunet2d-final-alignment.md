# Standalone nnU-Net 2D Final Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the standalone training-to-OOF workflow and gate its official-alignment label on reproducible server-side comparisons with installed nnU-Net v2.

**Architecture:** Keep standalone runtime free of `nnunetv2`. Replace approximate transform/sampling behavior with focused plan-derived modules, add full-volume inference and fold/OOF commands, and create an oracle artifact contract that can be generated only on the server with nnU-Net v2 installed. Every external result is marked pending until a parity report passes.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, SciPy, SimpleITK, pytest, CSV, JSON, NIfTI.

---

## File map

- Modify: `standalone_nnunet2d/training/official_config.py` — immutable transform/inference policy and resolved metadata.
- Create: `standalone_nnunet2d/training/formal_transforms.py` — paired spatial/intensity operations and deep-supervision target creation.
- Create: `standalone_nnunet2d/training/batch_sampler.py` — deterministic batch-level case/foreground patch decisions.
- Modify: `standalone_nnunet2d/training/formal_dataset.py` — consume precomputed patch requests and transform outputs.
- Modify: `standalone_nnunet2d/training/formal_checkpoint.py` — preserve scheduler and RNG state.
- Modify: `standalone_nnunet2d/formal_train.py` — construct formal loaders, persist the complete resolved configuration, and retain pending status.
- Modify: `standalone_nnunet2d/engine/predictor.py` — full-resolution logits, mirroring, tiled aggregation, and source-space output.
- Replace: `standalone_nnunet2d/predict.py` — formal checkpoint prediction CLI.
- Replace: `standalone_nnunet2d/validate_cv.py` — fold prediction/validation and OOF aggregation CLI.
- Create: `standalone_nnunet2d/oracle_capture.py` — server-only nnU-Net v2 capture command.
- Create: `standalone_nnunet2d/tools/parity_report.py` — compare standalone/oracle artifacts and write a pass/fail JSON report.
- Modify: `standalone_nnunet2d/metrics/segmentation_metrics.py` and `standalone_nnunet2d/metrics/crossval_summary.py` — explicit final metric policy and report fields.
- Modify: `standalone_nnunet2d/REPRODUCTION_NOTES.md` and `standalone_nnunet2d/README.md` — commands, labels, and evidence boundaries.
- Create: `standalone_nnunet2d/tests/test_documentation_contract.py` — prevent documentation from promoting pending runs.

### Task 1: Formal policy and metric-contract primitives

**Files:**
- Modify: `standalone_nnunet2d/training/official_config.py`
- Modify: `standalone_nnunet2d/metrics/segmentation_metrics.py`
- Modify: `standalone_nnunet2d/metrics/crossval_summary.py`
- Test: `standalone_nnunet2d/tests/test_official_trainer_config.py`
- Test: `standalone_nnunet2d/tests/test_metrics.py`

- [ ] **Step 1: Write failing policy/metric tests**

```python
def test_policy_records_argmax_tta_and_case_macro_contract() -> None:
    policy = OfficialInferencePolicy()
    assert policy.postprocessing == "argmax"
    assert policy.mirror_axes == (0, 1)
    assert policy.aggregation == "case_macro_mean"

def test_oof_summary_records_empty_mask_policy() -> None:
    record = {"case_id": "case001", "Dice": 1.0, "IoU": 1.0,
              "TP": 0, "FP": 0, "FN": 0, "TN": 4}
    summary = summarize_oof_cases([record])
    assert summary["metric_policy"]["both_empty"] == "dice=1"
    assert summary["aggregation"] == "case_macro_mean"
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `conda run -n newconda pytest standalone_nnunet2d/tests/test_official_trainer_config.py standalone_nnunet2d/tests/test_metrics.py -q`

Expected: FAIL because `OfficialInferencePolicy` and the required OOF fields do not exist.

- [ ] **Step 3: Implement minimal immutable contracts**

```python
@dataclass(frozen=True)
class OfficialInferencePolicy:
    postprocessing: str = "argmax"
    mirror_axes: tuple[int, int] = (0, 1)
    tile_step_size: float = 0.5
    aggregation: str = "case_macro_mean"

METRIC_POLICY = {
    "foreground": 1,
    "postprocessing": "argmax",
    "both_empty": "dice=1",
    "one_empty": "dice=0",
    "aggregation": "case_macro_mean",
}
```

Return `metric_policy` and `aggregation` in every OOF summary without changing
the existing confusion-count calculation.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `conda run -n newconda pytest standalone_nnunet2d/tests/test_official_trainer_config.py standalone_nnunet2d/tests/test_metrics.py -q`

Expected: PASS.

- [ ] **Step 5: Commit only when the user explicitly requests it**

```bash
git add standalone_nnunet2d/training/official_config.py standalone_nnunet2d/metrics/segmentation_metrics.py standalone_nnunet2d/metrics/crossval_summary.py standalone_nnunet2d/tests/test_official_trainer_config.py standalone_nnunet2d/tests/test_metrics.py
git commit -m "feat: define formal inference and metric policy"
```

### Task 2: Exactable transform operators and batch patch requests

**Files:**
- Create: `standalone_nnunet2d/training/formal_transforms.py`
- Create: `standalone_nnunet2d/training/batch_sampler.py`
- Modify: `standalone_nnunet2d/training/formal_dataset.py`
- Test: `standalone_nnunet2d/tests/test_formal_transforms.py`
- Test: `standalone_nnunet2d/tests/test_batch_sampler.py`
- Test: `standalone_nnunet2d/tests/test_formal_dataset.py`

- [ ] **Step 1: Write failing transform and sampler tests**

```python
def test_spatial_transform_uses_initial_patch_then_crops_final_patch() -> None:
    image = np.arange(64, dtype=np.float32).reshape(8, 8)
    label = np.zeros((8, 8), dtype=np.int16)
    result = apply_formal_spatial_transform(image, label, rng=np.random.default_rng(4),
                                            initial_patch_size=(12, 12), patch_size=(8, 8))
    assert result.image.shape == (8, 8)
    assert set(np.unique(result.label)) <= {-1, 0, 1}

def test_batch_request_has_oracle_comparable_foreground_slots() -> None:
    requests = FormalBatchSampler(case_ids=("a", "b", "c", "d"), batch_size=4,
                                  foreground_slots=(2, 3), seed=7).batch(0)
    assert [request.force_foreground for request in requests] == [False, False, True, True]
```

- [ ] **Step 2: Run new tests and verify RED**

Run: `conda run -n newconda pytest standalone_nnunet2d/tests/test_formal_transforms.py standalone_nnunet2d/tests/test_batch_sampler.py -q`

Expected: FAIL because the modules and public APIs do not exist.

- [ ] **Step 3: Implement focused operators and request objects**

```python
@dataclass(frozen=True)
class PatchRequest:
    case_id: str
    force_foreground: bool
    z_index: int
    center_yx: tuple[int, int]

def apply_formal_spatial_transform(image, label, *, rng, initial_patch_size, patch_size):
    image, label = crop_or_pad(image, label, _centre(image.shape), initial_patch_size)
    image, label = _rotate_and_scale_pair(image, label, rng)
    return SpatialResult(*crop_or_pad(image, label, _centre(image.shape), patch_size))
```

Implement every intensity operator as a separate function with explicit
probability and parameters. Preserve label `-1` until the final remove-label
operation. `FormalPatchDataset` receives a `PatchRequest` instead of making a
second independent foreground/random decision.

- [ ] **Step 4: Add fixed-seed regression fixtures**

Add tests for nearest-neighbour label interpolation, independent rotation and
scaling decisions, both gamma stages, low-resolution shape preservation,
mirroring, remove-label behavior, no-foreground fallback, and reproducible
batch requests. Fixtures must compare labels exactly and image values with
`np.testing.assert_allclose`.

- [ ] **Step 5: Run transform/dataset tests and verify GREEN**

Run: `conda run -n newconda pytest standalone_nnunet2d/tests/test_formal_transforms.py standalone_nnunet2d/tests/test_batch_sampler.py standalone_nnunet2d/tests/test_formal_dataset.py -q`

Expected: PASS.

- [ ] **Step 6: Commit only when the user explicitly requests it**

```bash
git add standalone_nnunet2d/training/formal_transforms.py standalone_nnunet2d/training/batch_sampler.py standalone_nnunet2d/training/formal_dataset.py standalone_nnunet2d/tests/test_formal_transforms.py standalone_nnunet2d/tests/test_batch_sampler.py standalone_nnunet2d/tests/test_formal_dataset.py
git commit -m "feat: add formal transform and batch sampler"
```

### Task 3: Reproducible formal trainer state

**Files:**
- Modify: `standalone_nnunet2d/training/formal_checkpoint.py`
- Modify: `standalone_nnunet2d/training/formal_trainer.py`
- Modify: `standalone_nnunet2d/formal_train.py`
- Test: `standalone_nnunet2d/tests/test_formal_checkpoint.py`
- Test: `standalone_nnunet2d/tests/test_formal_trainer_integration.py`

- [ ] **Step 1: Write failing resume-state tests**

```python
def test_formal_checkpoint_restores_scheduler_and_rng_state() -> None:
    scheduler = PolyLRScheduler(optimizer, .01, 1000)
    scheduler.step(17)
    state = FormalTrainerState(epoch=18, global_step=4500, best_validation_dice=.4, fold=0)
    save_formal_checkpoint(model, optimizer, scheduler, path, state, config, rng_state={"numpy": {}})
    restored = load_formal_checkpoint(restored_model, restored_optimizer, restored_scheduler, path, fold=0)
    assert restored.state == state
    assert restored.scheduler_step == 17
```

- [ ] **Step 2: Run checkpoint test and verify RED**

Run: `conda run -n newconda pytest standalone_nnunet2d/tests/test_formal_checkpoint.py -q`

Expected: FAIL because scheduler and RNG state are not part of the current checkpoint API.

- [ ] **Step 3: Extend checkpoint payload and training metadata**

Save `scheduler_state`, Python/NumPy/Torch CPU/CUDA RNG states, plan hash,
policy dictionaries, and `run_state`. Restore all states before the next
batch. Formal training must write `run_state="official_alignment_pending"`,
never `official_aligned`.

- [ ] **Step 4: Verify deterministic continuation**

Add a test that performs one update, saves, reloads into an equivalent model,
performs the same supplied next batch, and asserts equal parameters and loss.

- [ ] **Step 5: Run formal trainer tests and verify GREEN**

Run: `conda run -n newconda pytest standalone_nnunet2d/tests/test_formal_checkpoint.py standalone_nnunet2d/tests/test_formal_trainer.py standalone_nnunet2d/tests/test_formal_trainer_integration.py -q`

Expected: PASS.

- [ ] **Step 6: Commit only when the user explicitly requests it**

```bash
git add standalone_nnunet2d/training/formal_checkpoint.py standalone_nnunet2d/training/formal_trainer.py standalone_nnunet2d/formal_train.py standalone_nnunet2d/tests/test_formal_checkpoint.py standalone_nnunet2d/tests/test_formal_trainer_integration.py
git commit -m "feat: persist formal trainer reproducibility state"
```

### Task 4: Formal full-volume inference

**Files:**
- Modify: `standalone_nnunet2d/engine/predictor.py`
- Replace: `standalone_nnunet2d/predict.py`
- Test: `standalone_nnunet2d/tests/test_predictor.py`
- Test: `standalone_nnunet2d/tests/test_predict_command.py`

- [ ] **Step 1: Write failing inference tests**

```python
def test_predict_volume_averages_mirrored_logits_before_argmax() -> None:
    prediction = predict_volume(_OrientationModel(), volume, torch.device("cpu"), mirror_axes=(0, 1))
    np.testing.assert_array_equal(prediction, expected_mask)

def test_prediction_command_writes_source_space_mask(tmp_path) -> None:
    result = main(["--checkpoint", str(checkpoint), "--raw-root", str(raw_root),
                   "--case-id", "case001", "--output-root", str(output_root), "--device", "cpu"])
    assert result == 0
    assert read_nifti(output_root / "predictions" / "case001.nii.gz").direction == source.direction
```

- [ ] **Step 2: Run new tests and verify RED**

Run: `conda run -n newconda pytest standalone_nnunet2d/tests/test_predictor.py standalone_nnunet2d/tests/test_predict_command.py -q`

Expected: FAIL because mirror-aware inference and the prediction command do not exist.

- [ ] **Step 3: Implement predictor primitives**

```python
def predict_logits_2d(model, image, device, *, mirror_axes):
    logits = _full_resolution_logits(model(image))
    total, count = logits.clone(), 1
    for axes in mirror_combinations(mirror_axes):
        mirrored = torch.flip(image, dims=tuple(axis + 2 for axis in axes))
        total += torch.flip(_full_resolution_logits(model(mirrored)), dims=tuple(axis + 2 for axis in axes))
        count += 1
    return total / count
```

Use a tiled accumulator only when a slice exceeds the configured patch size;
for a 512×512 source slice it must use exactly one tile. Convert logits to
labels once after aggregation. Load formal checkpoints into a model configured
for full-resolution tensor output and reject non-formal/pending metadata only
when the caller explicitly asks for an aligned run.

- [ ] **Step 4: Implement the CLI and metadata validation**

The command accepts checkpoint, raw root, one-or-more case IDs or a fold,
output root, device, and `--allow-pending`. It writes `prediction_manifest.json`
with checkpoint metadata, preprocessing/inference policy, each source path,
and NIfTI validation status.

- [ ] **Step 5: Run inference tests and verify GREEN**

Run: `conda run -n newconda pytest standalone_nnunet2d/tests/test_predictor.py standalone_nnunet2d/tests/test_predict_command.py standalone_nnunet2d/tests/test_nifti_roundtrip.py -q`

Expected: PASS.

- [ ] **Step 6: Commit only when the user explicitly requests it**

```bash
git add standalone_nnunet2d/engine/predictor.py standalone_nnunet2d/predict.py standalone_nnunet2d/tests/test_predictor.py standalone_nnunet2d/tests/test_predict_command.py
git commit -m "feat: add formal full-volume prediction"
```

### Task 5: Fold validation and strict OOF aggregation

**Files:**
- Create: `standalone_nnunet2d/engine/formal_validation.py`
- Replace: `standalone_nnunet2d/validate_cv.py`
- Modify: `standalone_nnunet2d/metrics/crossval_summary.py`
- Test: `standalone_nnunet2d/tests/test_formal_validation.py`
- Test: `standalone_nnunet2d/tests/test_validate_cv.py`

- [ ] **Step 1: Write failing fold/OOF tests**

```python
def test_validate_fold_writes_one_record_per_held_out_case(tmp_path) -> None:
    report = validate_fold(model, raw_root, fold=0, output_root=tmp_path, device=torch.device("cpu"))
    assert report["case_count"] == len(load_fold_cases(0, "val"))
    assert report["aggregation"] == "case_macro_mean"

def test_oof_command_rejects_missing_or_duplicate_held_out_cases(tmp_path) -> None:
    with pytest.raises(ValueError, match="95 unique"):
        aggregate_oof(tmp_path)
```

- [ ] **Step 2: Run new tests and verify RED**

Run: `conda run -n newconda pytest standalone_nnunet2d/tests/test_formal_validation.py standalone_nnunet2d/tests/test_validate_cv.py -q`

Expected: FAIL because `formal_validation` and `validate_cv` are deferred.

- [ ] **Step 3: Implement fold-level full-volume validation**

For every `load_fold_cases(fold, "val")` case: predict, save NIfTI, read GT,
compute one `case_metric_record`, and write deterministic CSV/JSON reports.
Do not call `run_validation_epoch`; that is an online patch metric only.

- [ ] **Step 4: Implement OOF aggregation command**

Read exactly five fold report JSON/CSV files, verify that each of the 95 supplied
validation IDs appears once, create `oof_per_case_metrics.csv` and
`oof_summary.json`, and preserve metric policy, case count, failed case count,
and `run_state`.

- [ ] **Step 5: Run validation tests and verify GREEN**

Run: `conda run -n newconda pytest standalone_nnunet2d/tests/test_formal_validation.py standalone_nnunet2d/tests/test_validate_cv.py standalone_nnunet2d/tests/test_metrics.py -q`

Expected: PASS.

- [ ] **Step 6: Commit only when the user explicitly requests it**

```bash
git add standalone_nnunet2d/engine/formal_validation.py standalone_nnunet2d/validate_cv.py standalone_nnunet2d/metrics/crossval_summary.py standalone_nnunet2d/tests/test_formal_validation.py standalone_nnunet2d/tests/test_validate_cv.py
git commit -m "feat: validate folds and aggregate OOF metrics"
```

### Task 6: Server oracle artifact capture and parity report

**Files:**
- Create: `standalone_nnunet2d/oracle_capture.py`
- Create: `standalone_nnunet2d/tools/parity_report.py`
- Create: `standalone_nnunet2d/tests/test_parity_report.py`
- Modify: `standalone_nnunet2d/README.md`

- [ ] **Step 1: Write failing artifact-comparison tests**

```python
def test_parity_report_fails_on_label_difference_and_passes_declared_tolerance(tmp_path) -> None:
    write_artifact(tmp_path / "oracle", image=np.array([1.0]), label=np.array([1]))
    write_artifact(tmp_path / "standalone", image=np.array([1.0 + 1e-7]), label=np.array([1]))
    report = compare_artifacts(tmp_path / "oracle", tmp_path / "standalone", image_atol=1e-6)
    assert report["status"] == "passed"
    np.save(tmp_path / "standalone" / "label.npy", np.array([0]))
    assert compare_artifacts(tmp_path / "oracle", tmp_path / "standalone")["status"] == "failed"
```

- [ ] **Step 2: Run parity test and verify RED**

Run: `conda run -n newconda pytest standalone_nnunet2d/tests/test_parity_report.py -q`

Expected: FAIL because the parity reporter does not exist.

- [ ] **Step 3: Implement a versioned artifact manifest and reporter**

The manifest contains artifact version, `nnunetv2` version, plans hash, fixed
seed, case ID, transform/sampling policy, array names/shapes/dtypes, and NIfTI
metadata. The reporter compares mandatory keys, exact integer arrays, floating
arrays with explicit tolerances, masks exactly, and emits component-level
diagnostics plus `run_state`.

- [ ] **Step 4: Implement the server-only capture entry point**

Import `nnunetv2` only inside `oracle_capture.py`. It must reject execution
when unavailable and must never invoke training. Provide capture modes:
`preprocess`, `sample`, `transform`, `deep_supervision`, and `inference`.
The README command requires explicit paths and writes below this project's
`outputs/oracle/` directory.

- [ ] **Step 5: Run parity tests and static import guard**

Run: `conda run -n newconda pytest standalone_nnunet2d/tests/test_parity_report.py -q`

Expected: PASS. Also run `rg -n "import nnunetv2|from nnunetv2" standalone_nnunet2d -g '*.py'` and verify the only allowed match is `oracle_capture.py`.

- [ ] **Step 6: Commit only when the user explicitly requests it**

```bash
git add standalone_nnunet2d/oracle_capture.py standalone_nnunet2d/tools/parity_report.py standalone_nnunet2d/tests/test_parity_report.py standalone_nnunet2d/README.md
git commit -m "feat: add nnunet oracle parity gate"
```

### Task 7: Documentation, full verification, and server handoff

**Files:**
- Modify: `standalone_nnunet2d/REPRODUCTION_NOTES.md`
- Modify: `standalone_nnunet2d/README.md`
- Test: all `standalone_nnunet2d/tests`

- [ ] **Step 1: Add documentation assertions before updating docs**

```python
def test_documentation_does_not_describe_pending_runs_as_official() -> None:
    text = Path("standalone_nnunet2d/REPRODUCTION_NOTES.md").read_text(encoding="utf-8")
    assert "official_alignment_pending" in text
    assert "parity report" in text.lower()
```

- [ ] **Step 2: Run test and verify RED**

Run: `conda run -n newconda pytest standalone_nnunet2d/tests/test_documentation_contract.py -q`

Expected: FAIL because the documentation contract test and current explanation are absent.

- [ ] **Step 3: Document exact command sequence and result labels**

Document local tests; server oracle capture; standalone fold-0 formal training;
formal fold-0 prediction/validation; parity report; five-fold training and OOF
aggregation. State that only a passed transform and inference parity report
permits `official_aligned`; no smoke result or online `validation_dice` is a
final benchmark.

- [ ] **Step 4: Run the full local verification suite**

Run: `conda run -n newconda pytest standalone_nnunet2d/tests -q`

Expected: PASS with no failures. Then run:

```bash
conda run -n newconda python -m py_compile standalone_nnunet2d/formal_train.py standalone_nnunet2d/predict.py standalone_nnunet2d/validate_cv.py standalone_nnunet2d/oracle_capture.py
```

Expected: exit code 0.

- [ ] **Step 5: Perform the server handoff checks**

Run the documented server commands in this order: oracle capture; parity
report; one fold-0 prediction export; fold-0 case report. Do not start the
five-fold run until the artifact report is `passed`. Archive JSON/CSV/NIfTI
paths in the final run manifest.

- [ ] **Step 6: Commit only when the user explicitly requests it**

```bash
git add standalone_nnunet2d/REPRODUCTION_NOTES.md standalone_nnunet2d/README.md standalone_nnunet2d/tests/test_documentation_contract.py
git commit -m "docs: document final alignment verification workflow"
```
