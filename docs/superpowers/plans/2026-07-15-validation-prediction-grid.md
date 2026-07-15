# Validation Prediction Grid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render a deterministic nnU-Net-style DWI/GT/prediction validation grid from the best-checkpoint evaluation rows.

**Architecture:** Visualization helpers select named representative slice dictionaries and render one three-column matplotlib figure. The evaluator retains image, target, prediction and metadata for validation rows, then calls the helper after metrics export.

**Tech Stack:** Python, NumPy, Matplotlib, pytest.

---

### Task 1: Selection and grid renderer

**Files:** Modify `optical_deeplab2d/evaluation/visualization.py`; create `optical_deeplab2d/tests/test_visualization.py`.

- [ ] **Step 1: Write a failing test**

```python
def test_select_representative_rows_is_unique_and_returns_png(tmp_path):
    rows = synthetic_validation_rows()
    selected = select_representative_rows(rows, seed=2026, limit=6)
    output = tmp_path / "validation_predictions_best.png"
    save_validation_grid(selected, output)
    assert len({row['sample_id'] for row in selected}) == len(selected)
    assert output.exists() and output.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n newconda python -m pytest optical_deeplab2d/tests/test_visualization.py -q`

Expected: FAIL because selection and grid functions do not exist.

- [ ] **Step 3: Implement minimal renderer**

Implement `select_representative_rows(rows, seed, limit)` using largest/smallest positive area, lowest Dice, empty-mask maximum false positives and seeded random completion. Implement `save_validation_grid(rows, output)` with three columns titled DWI, GT Mask and Prediction.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n newconda python -m pytest optical_deeplab2d/tests/test_visualization.py -q`

Expected: PASS.

### Task 2: Evaluation integration

**Files:** Modify `optical_deeplab2d/evaluate.py`; extend `optical_deeplab2d/tests/test_visualization.py`.

- [ ] **Step 1: Add an evaluator contract test**

```python
def test_evaluation_grid_uses_validation_predictions(tmp_path):
    # evaluator passes retained image, target, prediction and metadata rows
    assert (tmp_path / 'validation_predictions_best.png').exists()
```

- [ ] **Step 2: Integrate after metric export**

Retain original DWI and probability-derived prediction alongside patient metadata; call the renderer once after `write_evaluation` and write only to `--output-dir`.

- [ ] **Step 3: Verify**

Run: `conda run -n newconda python -m pytest optical_deeplab2d/tests -q` and `conda run -n newconda python -m compileall -q optical_deeplab2d`.

- [ ] **Step 4: Commit**

Run: `git add optical_deeplab2d docs/superpowers/plans/2026-07-15-validation-prediction-grid.md && git commit -m "feat: add validation prediction grid"`.
