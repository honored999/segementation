# Segmentation Metrics and OOF Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reliable binary Dice/IoU metrics and auditable out-of-fold case aggregation without creating or reading predictions.

**Architecture:** `segmentation_metrics.py` converts validated binary masks into confusion counts and scalar metrics. `crossval_summary.py` aggregates JSON-safe case records and extracts the supplied official baseline without assuming it has been reproduced.

**Tech Stack:** Python 3.14, NumPy, pytest, supplied JSON references; no NIfTI or nnU-Net runtime dependency.

---

### Task 1: Binary segmentation metrics (completed)

**Files:**
- Modify: `standalone_nnunet2d/metrics/segmentation_metrics.py`
- Modify: `standalone_nnunet2d/tests/test_metrics.py`

- [ ] **Step 1: Write failing overlap and empty-mask tests**

```python
def test_binary_metrics_report_perfect_overlap() -> None:
    result = binary_segmentation_metrics(np.array([[0, 1], [1, 0]]), np.array([[0, 1], [1, 0]]))
    assert result["TP"] == 2 and result["FP"] == 0 and result["FN"] == 0
    assert result["Dice"] == pytest.approx(1.0) and result["IoU"] == pytest.approx(1.0)

def test_binary_metrics_define_empty_masks_as_perfect_agreement() -> None:
    result = binary_segmentation_metrics(np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8))
    assert result["Dice"] == 1.0 and result["IoU"] == 1.0
```

- [ ] **Step 2: Run the test and verify it fails because the metric function is absent**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests/test_metrics.py -v`

Expected: import failure for `binary_segmentation_metrics`.

- [ ] **Step 3: Implement validated binary confusion metrics**

```python
import numpy as np
from numpy.typing import ArrayLike

def binary_segmentation_metrics(prediction: ArrayLike, reference: ArrayLike) -> dict[str, float | int]:
    pred, ref = np.asarray(prediction), np.asarray(reference)
    if pred.shape != ref.shape: raise ValueError("prediction and reference must have equal shapes")
    if not np.isin(pred, (0, 1)).all() or not np.isin(ref, (0, 1)).all(): raise ValueError("masks must contain only 0 and 1")
    tp, fp, fn, tn = int(((pred == 1) & (ref == 1)).sum()), int(((pred == 1) & (ref == 0)).sum()), int(((pred == 0) & (ref == 1)).sum()), int(((pred == 0) & (ref == 0)).sum())
    dice = 1.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
    iou = 1.0 if tp + fp + fn == 0 else tp / (tp + fp + fn)
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "Dice": dice, "IoU": iou}
```

- [ ] **Step 4: Run metric tests and verify they pass**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests/test_metrics.py -v`

Expected: overlap and empty-mask cases pass.

### Task 2: JSON-safe case records and OOF aggregation (completed)

**Files:**
- Modify: `standalone_nnunet2d/metrics/segmentation_metrics.py`
- Modify: `standalone_nnunet2d/metrics/crossval_summary.py`
- Modify: `standalone_nnunet2d/tests/test_metrics.py`

- [ ] **Step 1: Write failing case-record and duplicate-case tests**

```python
def test_case_record_is_json_safe_and_oof_summary_averages_metrics() -> None:
    records = [case_metric_record("case001", np.array([0, 1]), np.array([0, 1])), case_metric_record("case002", np.array([0, 0]), np.array([0, 1]))]
    summary = summarize_oof_cases(records)
    assert summary["case_count"] == 2
    assert summary["foreground_mean"]["Dice"] == pytest.approx(0.5)

def test_oof_summary_rejects_duplicate_case_ids() -> None:
    record = {"case_id": "case001", "Dice": 1.0, "IoU": 1.0, "TP": 1, "FP": 0, "FN": 0, "TN": 1}
    with pytest.raises(ValueError, match="duplicate"):
        summarize_oof_cases([record, record])
```

- [ ] **Step 2: Run the tests and verify they fail because record/summary functions are absent**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests/test_metrics.py -v`

Expected: import failure for `case_metric_record` or `summarize_oof_cases`.

- [ ] **Step 3: Implement record construction and strict aggregation**

```python
from collections.abc import Mapping, Sequence

def case_metric_record(case_id: str, prediction: ArrayLike, reference: ArrayLike) -> dict[str, str | float | int]:
    if not case_id: raise ValueError("case_id must not be empty")
    return {"case_id": case_id, **binary_segmentation_metrics(prediction, reference)}

def summarize_oof_cases(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    case_ids = [str(record.get("case_id", "")) for record in records]
    if not case_ids or any(not case_id for case_id in case_ids): raise ValueError("records require non-empty case_id values")
    if len(case_ids) != len(set(case_ids)): raise ValueError("duplicate case IDs are not valid OOF results")
    return {"case_count": len(records), "foreground_mean": {metric: float(np.mean([float(record[metric]) for record in records])) for metric in ("Dice", "IoU")}, "metric_per_case": [dict(record) for record in records]}
```

- [ ] **Step 4: Run aggregation tests and verify they pass**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests/test_metrics.py -v`

Expected: JSON-safe records, mean aggregation, and duplicate rejection pass.

### Task 3: Official baseline extraction and documentation (completed)

**Files:**
- Modify: `standalone_nnunet2d/metrics/crossval_summary.py`
- Modify: `standalone_nnunet2d/tests/test_metrics.py`
- Modify: `standalone_nnunet2d/README.md`
- Modify: `standalone_nnunet2d/REPRODUCTION_NOTES.md`

- [ ] **Step 1: Write a failing supplied-baseline test**

```python
def test_reference_baseline_matches_supplied_summary() -> None:
    baseline = extract_reference_baseline(Path("standalone_nnunet2d/reference/summary.json"))
    assert baseline == {"Dice": pytest.approx(0.731103738314918), "IoU": pytest.approx(0.5923877518050135), "case_count": 95}
```

- [ ] **Step 2: Run the test and verify it fails because the extractor is absent**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests/test_metrics.py::test_reference_baseline_matches_supplied_summary -v`

Expected: import failure for `extract_reference_baseline`.

- [ ] **Step 3: Implement exact baseline extraction**

```python
import json
from pathlib import Path

def extract_reference_baseline(summary_path: Path) -> dict[str, float | int]:
    with summary_path.open(encoding="utf-8") as handle: summary = json.load(handle)
    foreground, per_case = summary.get("foreground_mean"), summary.get("metric_per_case")
    if not isinstance(foreground, dict) or not isinstance(per_case, list) or "Dice" not in foreground or "IoU" not in foreground:
        raise ValueError("summary must contain foreground Dice, IoU, and metric_per_case")
    return {"Dice": float(foreground["Dice"]), "IoU": float(foreground["IoU"]), "case_count": len(per_case)}
```

- [ ] **Step 4: Document that metrics do not create predictions or a reproduced score, then run all tests**

```powershell
conda run -n newconda python -m pytest standalone_nnunet2d/tests -v
```

Expected: all tests pass; the documentation identifies the supplied 95-case baseline as comparison-only.

## Plan self-review

The plan covers strict mask validation, empty foreground behavior, confusion
counts, JSON-safe per-case records, duplicate rejection, OOF means, actual
reference-schema extraction, documentation, and full regression. It excludes
thresholding, argmax, NIfTI I/O, model execution, output writes, and training.

## Execution record

- [x] Strict binary confusion/Dice/IoU metrics implemented and tested.
- [x] JSON-safe case records and duplicate-protected OOF aggregation implemented and tested.
- [x] Supplied 95-case Dice/IoU baseline extraction implemented and tested.
- [x] Comparison-only documentation added; no predictions or training were run.
- [x] Full regression suite passed: 21 tests.
