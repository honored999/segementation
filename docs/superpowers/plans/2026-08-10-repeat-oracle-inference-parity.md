# Repeat-Oracle Inference Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible inference parity gate that requires exact agreement on official-stable voxels while explicitly bounding standalone output by labels observed across at least three independent official CUDA oracle runs.

**Architecture:** Preserve the existing one-oracle exact comparison for transform and diagnostic use. Add a separate repeated-inference comparison function, dispatch to it only when the CLI receives three or more distinct `--oracle-root` arguments, and keep every report at `official_alignment_pending`.

**Tech Stack:** Python 3.10+, NumPy, argparse, JSON, pytest.

**Git constraint:** Do not commit, push, or create a PR. Preserve all existing uncommitted changes.

---

## File map

- Modify: `standalone_nnunet2d/tools/parity_report.py` — repeated-oracle validation, stable/unstable mask comparison, and CLI dispatch.
- Modify: `standalone_nnunet2d/tests/test_parity_report.py` — core repeated-gate and backward-compatibility tests.
- Modify: `standalone_nnunet2d/tests/test_documentation_contract.py` — documentation and pending-state assertions.
- Modify: `standalone_nnunet2d/README.md` — user-facing repeated inference command and evidence boundary.
- Modify: `standalone_nnunet2d/REPRODUCTION_NOTES.md` — server workflow and report interpretation.

### Task 1: Repeated-oracle comparison core

**Files:**
- Modify: `standalone_nnunet2d/tools/parity_report.py`
- Modify: `standalone_nnunet2d/tests/test_parity_report.py`

- [ ] **Step 1: Add failing fixture helpers and passing variability test**

Extend `_write_artifact` calls with inference manifests and create three distinct
oracle roots. The test data must contain one stable foreground voxel, two
official-unstable voxels, and a standalone mask whose unstable labels each
occur in at least one oracle repeat.

```python
from standalone_nnunet2d.tools.parity_report import (
    compare_artifacts,
    compare_repeated_oracle_inference,
    main,
)


def _write_repeated_inference_artifacts(
    tmp_path: Path,
    oracle_masks: tuple[np.ndarray, ...],
    standalone_mask: np.ndarray,
) -> tuple[tuple[Path, ...], Path]:
    image = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    label = np.array([[0, 0], [1, 0]], dtype=np.int16)
    oracle_roots: list[Path] = []
    for index, mask in enumerate(oracle_masks):
        root = tmp_path / f"oracle_{index}"
        _write_artifact(
            root,
            image=image,
            label=label,
            mask=mask,
            manifest_overrides={
                "transform_policy": {"mode": "inference", "implementation": "oracle"},
                "sampling_policy": {"seed": 17, "fold": 0, "implementation": "oracle"},
            },
        )
        oracle_roots.append(root)
    standalone_root = tmp_path / "standalone"
    _write_artifact(
        standalone_root,
        image=image,
        label=label,
        mask=standalone_mask,
        manifest_overrides={
            "transform_policy": {"mode": "inference", "implementation": "standalone"},
            "sampling_policy": {"seed": 17, "fold": 0, "implementation": "standalone"},
        },
    )
    return tuple(oracle_roots), standalone_root


def test_repeated_oracle_gate_accepts_only_observed_labels_on_unstable_voxels(
    tmp_path: Path,
) -> None:
    oracle_masks = (
        np.array([[0, 0], [1, 0]], dtype=np.uint8),
        np.array([[1, 0], [1, 1]], dtype=np.uint8),
        np.array([[1, 0], [1, 0]], dtype=np.uint8),
    )
    standalone_mask = np.array([[0, 0], [1, 1]], dtype=np.uint8)
    oracle_roots, standalone_root = _write_repeated_inference_artifacts(
        tmp_path, oracle_masks, standalone_mask
    )

    report = compare_repeated_oracle_inference(
        oracle_roots, standalone_root, image_atol=0.0
    )

    assert report["status"] == "passed"
    assert report["parity_policy"] == "repeat_oracle_stability_v1"
    assert report["oracle_repeat_count"] == 3
    assert report["oracle_unstable_voxel_count"] == 2
    assert report["oracle_unstable_voxel_coordinates"] == [[0, 0], [1, 1]]
    assert report["stable_mask_mismatch_count"] == 0
    assert report["unobserved_standalone_label_count"] == 0
    assert report["run_state"] == "official_alignment_pending"
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
conda run -n newconda python -m pytest -q standalone_nnunet2d/tests/test_parity_report.py -k repeated_oracle_gate_accepts
```

Expected: FAIL because `compare_repeated_oracle_inference` does not exist.

- [ ] **Step 3: Implement manifest/root validation and repeated mask policy**

Add these public/private interfaces without changing `compare_artifacts`:

```python
REPEATED_INFERENCE_POLICY = "repeat_oracle_stability_v1"
MINIMUM_ORACLE_REPEATS = 3


def _resolved_distinct_roots(oracle_roots: Sequence[Path]) -> tuple[Path, ...]:
    roots = tuple(Path(root).resolve() for root in oracle_roots)
    if len(roots) < MINIMUM_ORACLE_REPEATS:
        raise ValueError("repeated inference parity requires at least three oracle roots")
    if len(set(roots)) != len(roots):
        raise ValueError("repeated inference parity requires distinct oracle roots")
    return roots


def _coordinates(mask: np.ndarray) -> list[list[int]]:
    return np.argwhere(mask).astype(int, copy=False).tolist()
```

The implementation must:

1. Reject nonzero, negative, or non-finite `image_atol`.
2. Resolve and reject duplicate roots.
3. Load every manifest and require `transform_policy.mode == "inference"`.
4. Compare every repeat and standalone manifest with the first oracle using
   `_manifest_value_differences`.
5. Compare `image` and `label` exactly across all roots with `_array_matches`
   and `image_atol=0.0`.
6. Require all masks to share shape, dtype, and integer dtype.
7. Stack oracle masks with the leading repeat dimension, derive
   `stable = np.all(stack == stack[0], axis=0)`, and derive
   `unstable = ~stable`.
8. Count standalone mismatches on `stable`.
9. At unstable coordinates, accept a standalone label only when
   `np.any(stack == standalone_mask[None], axis=0)` is true.
10. Emit pairwise mask difference records in deterministic repeat-index order:

```python
pairwise = [
    {
        "left_index": left,
        "right_index": right,
        "difference_count": int(np.count_nonzero(stack[left] != stack[right])),
    }
    for left in range(len(roots))
    for right in range(left + 1, len(roots))
]
```

Return all fields required by the design, component diagnostics, and
`run_state=RUN_STATE`. Never emit `official_aligned`.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the same command from Step 2.

Expected: PASS.

- [ ] **Step 5: Add failing rejection and compatibility tests**

Add separate tests for each behavior:

```python
def test_repeated_oracle_gate_rejects_stable_voxel_mismatch(tmp_path: Path) -> None:
    oracle_masks = tuple(
        np.array([[0, 0], [1, 0]], dtype=np.uint8) for _ in range(3)
    )
    standalone_mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    roots, standalone = _write_repeated_inference_artifacts(
        tmp_path, oracle_masks, standalone_mask
    )
    report = compare_repeated_oracle_inference(roots, standalone)
    assert report["status"] == "failed"
    assert report["stable_mask_mismatch_count"] == 1
    assert report["stable_mask_mismatch_coordinates"] == [[0, 1]]


def test_repeated_oracle_gate_rejects_unobserved_unstable_label(tmp_path: Path) -> None:
    oracle_masks = (
        np.array([[0, 0], [1, 0]], dtype=np.uint8),
        np.array([[1, 0], [1, 0]], dtype=np.uint8),
        np.array([[0, 0], [1, 0]], dtype=np.uint8),
    )
    standalone_mask = np.array([[2, 0], [1, 0]], dtype=np.uint8)
    roots, standalone = _write_repeated_inference_artifacts(
        tmp_path, oracle_masks, standalone_mask
    )
    report = compare_repeated_oracle_inference(roots, standalone)
    assert report["status"] == "failed"
    assert report["unobserved_standalone_label_count"] == 1
    assert report["unobserved_standalone_label_coordinates"] == [[0, 0]]


@pytest.mark.parametrize("repeat_count", [0, 1, 2])
def test_repeated_oracle_gate_requires_three_distinct_roots(
    tmp_path: Path, repeat_count: int
) -> None:
    masks = tuple(np.zeros((2, 2), dtype=np.uint8) for _ in range(3))
    roots, standalone = _write_repeated_inference_artifacts(
        tmp_path, masks, np.zeros((2, 2), dtype=np.uint8)
    )
    with pytest.raises(ValueError, match="at least three"):
        compare_repeated_oracle_inference(roots[:repeat_count], standalone)


def test_repeated_oracle_gate_rejects_duplicate_roots(tmp_path: Path) -> None:
    masks = tuple(np.zeros((2, 2), dtype=np.uint8) for _ in range(3))
    roots, standalone = _write_repeated_inference_artifacts(
        tmp_path, masks, np.zeros((2, 2), dtype=np.uint8)
    )
    with pytest.raises(ValueError, match="distinct"):
        compare_repeated_oracle_inference([roots[0], roots[0], roots[0]], standalone)


def test_repeated_oracle_gate_rejects_non_inference_mode(tmp_path: Path) -> None:
    masks = tuple(np.zeros((2, 2), dtype=np.uint8) for _ in range(3))
    roots, standalone = _write_repeated_inference_artifacts(
        tmp_path, masks, np.zeros((2, 2), dtype=np.uint8)
    )
    manifest_path = roots[1] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["transform_policy"] = {"mode": "transform", "implementation": "oracle"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="inference"):
        compare_repeated_oracle_inference(roots, standalone)


def test_single_oracle_transform_comparison_remains_exact(tmp_path: Path) -> None:
    oracle, standalone = _write_pair(tmp_path)
    np.save(standalone / "mask.npy", np.array([[0, 1], [0, 0]], dtype=np.uint8))
    report = compare_artifacts(oracle, standalone)
    assert report["status"] == "failed"
    assert report["diagnostics"] == ["mask: integer values differ"]
```

Also cover inconsistent image, label, manifest metadata, mask shape/dtype, and
`image_atol != 0.0`. Each failure must identify its component and preserve the
pending run state when a report is returned.

- [ ] **Step 6: Run new tests and verify RED, then complete minimal validation**

Run:

```powershell
conda run -n newconda python -m pytest -q standalone_nnunet2d/tests/test_parity_report.py
```

Expected before implementation completion: the new rejection tests FAIL for
missing validation. Implement only the validations enumerated in Step 5, then
rerun until the file passes.

### Task 2: Backward-compatible repeatable CLI

**Files:**
- Modify: `standalone_nnunet2d/tools/parity_report.py`
- Modify: `standalone_nnunet2d/tests/test_parity_report.py`

- [ ] **Step 1: Write failing CLI dispatch tests**

```python
def test_cli_one_oracle_uses_existing_exact_comparison(capsys, tmp_path: Path) -> None:
    oracle, standalone = _write_pair(tmp_path)
    assert main([
        "--oracle-root", str(oracle),
        "--standalone-root", str(standalone),
    ]) == 0
    assert json.loads(capsys.readouterr().out)["oracle_root"] == str(oracle.resolve())


def test_cli_three_oracles_use_repeated_stability_policy(capsys, tmp_path: Path) -> None:
    masks = tuple(np.zeros((2, 2), dtype=np.uint8) for _ in range(3))
    oracle_roots, standalone = _write_repeated_inference_artifacts(
        tmp_path, masks, np.zeros((2, 2), dtype=np.uint8)
    )
    argv = [
        "--oracle-root", str(oracle_roots[0]),
        "--oracle-root", str(oracle_roots[1]),
        "--oracle-root", str(oracle_roots[2]),
        "--standalone-root", str(standalone),
        "--image-atol", "0",
    ]
    assert main(argv) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["parity_policy"] == "repeat_oracle_stability_v1"


def test_cli_two_oracles_are_rejected(capsys, tmp_path: Path) -> None:
    masks = tuple(np.zeros((2, 2), dtype=np.uint8) for _ in range(3))
    oracle_roots, standalone = _write_repeated_inference_artifacts(
        tmp_path, masks, np.zeros((2, 2), dtype=np.uint8)
    )
    with pytest.raises(SystemExit) as error:
        main([
            "--oracle-root", str(oracle_roots[0]),
            "--oracle-root", str(oracle_roots[1]),
            "--standalone-root", str(standalone),
        ])
    assert error.value.code == 2
    assert "one or at least three" in capsys.readouterr().err
```

- [ ] **Step 2: Run CLI tests and verify RED**

Run:

```powershell
conda run -n newconda python -m pytest -q standalone_nnunet2d/tests/test_parity_report.py -k cli
```

Expected: FAIL because `--oracle-root` currently accepts one value only and
`main` always calls `compare_artifacts`.

- [ ] **Step 3: Implement repeatable argument and explicit dispatch**

Change only the parser and `main` dispatch:

```python
parser.add_argument("--oracle-root", required=True, action="append", type=Path)

oracle_roots = arguments.oracle_root
if len(oracle_roots) == 1:
    report = compare_artifacts(
        oracle_roots[0], arguments.standalone_root, image_atol=arguments.image_atol
    )
elif len(oracle_roots) >= MINIMUM_ORACLE_REPEATS:
    report = compare_repeated_oracle_inference(
        oracle_roots,
        arguments.standalone_root,
        image_atol=arguments.image_atol,
    )
else:
    parser.error("provide one oracle root or at least three distinct oracle roots")
```

Keep report writing and exit-code behavior unchanged.

- [ ] **Step 4: Run CLI and complete parity tests**

Run:

```powershell
conda run -n newconda python -m pytest -q standalone_nnunet2d/tests/test_parity_report.py
```

Expected: PASS.

### Task 3: Documentation contract and final verification

**Files:**
- Modify: `standalone_nnunet2d/README.md`
- Modify: `standalone_nnunet2d/REPRODUCTION_NOTES.md`
- Modify: `standalone_nnunet2d/tests/test_documentation_contract.py`

- [ ] **Step 1: Write failing documentation contract test**

Require both documents to name the repeated policy, minimum repeat count,
stable/unstable rule, and pending-state boundary:

```python
@pytest.mark.parametrize("document", [README_PATH, REPRODUCTION_NOTES_PATH])
def test_documentation_records_repeat_oracle_inference_gate(document: Path) -> None:
    text = document.read_text(encoding="utf-8")
    assert "repeat_oracle_stability_v1" in text
    assert "at least three" in text.lower()
    assert "stable voxel" in text.lower()
    assert "official_alignment_pending" in text
```

- [ ] **Step 2: Run documentation test and verify RED**

Run:

```powershell
conda run -n newconda python -m pytest -q standalone_nnunet2d/tests/test_documentation_contract.py
```

Expected: FAIL because the documents still show a single-oracle inference gate.

- [ ] **Step 3: Update only the inference parity instructions**

In both documents:

1. Keep the existing single-root transform command.
2. Label a one-root inference comparison as diagnostic only.
3. Show a repeated command with three independent roots:

```powershell
conda run -n newconda python -m standalone_nnunet2d.tools.parity_report `
  --oracle-root standalone_nnunet2d\outputs\oracle_inference_run1\inference\case005 `
  --oracle-root standalone_nnunet2d\outputs\oracle_inference_run2\inference\case005 `
  --oracle-root standalone_nnunet2d\outputs\oracle_inference_run3\inference\case005 `
  --standalone-root standalone_nnunet2d\outputs\standalone_inference\inference\case005 `
  --image-atol 0 `
  --output standalone_nnunet2d\outputs\inference_repeat_parity_report.json
```

4. State that stable voxels compare exactly, unstable voxels accept only labels
   observed in the repeats, every unstable coordinate is reported, and the
   report itself remains pending.
5. Do not claim that existing historical training output is aligned.

- [ ] **Step 4: Run documentation and focused tests**

Run:

```powershell
conda run -n newconda python -m pytest -q standalone_nnunet2d/tests/test_documentation_contract.py standalone_nnunet2d/tests/test_parity_report.py
```

Expected: PASS.

- [ ] **Step 5: Run final verification**

Run:

```powershell
conda run -n newconda python -m pytest -q
$pyFiles = @(rg --files standalone_nnunet2d -g '*.py')
conda run -n newconda python -m py_compile $pyFiles
git diff --check
git status --short
```

Expected: all tests pass, `py_compile` exits 0, `git diff --check` reports no
errors, and only intended plus pre-existing uncommitted files appear. Do not
commit or push.

- [ ] **Step 6: Prepare server handoff without assuming access**

Report the runtime file that must be uploaded, the third independent oracle
capture command, and the repeated parity command containing three
`--oracle-root` arguments. Do not call the result official until the repeated
report is actually generated and passes.
