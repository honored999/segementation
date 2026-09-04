# Safe Checkpoint Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit, locally constrained checkpoint save/load interface for the standalone project without enabling a training loop.

**Architecture:** `engine/checkpoint.py` will own one format version and validate that every write path resolves below `standalone_nnunet2d/outputs/`. The serialized payload will contain only model state, optional optimizer state, and caller-supplied metadata. Loading will validate the version and caller-declared metadata expectations before restoring state.

**Tech Stack:** Python 3.14, PyTorch, pytest, pathlib.

---

### Task 1: Specify checkpoint behavior with failing tests

**Files:**
- Create: `standalone_nnunet2d/tests/test_checkpoint.py`
- Modify: `standalone_nnunet2d/engine/checkpoint.py`

- [x] **Step 1: Write the failing round-trip test**

```python
def test_checkpoint_round_trip_restores_model_optimizer_and_metadata(tmp_path: Path) -> None:
    source = nn.Conv2d(1, 2, kernel_size=1)
    source_optimizer = torch.optim.SGD(source.parameters(), lr=0.1, momentum=0.9)
    _prime_optimizer_state(source, source_optimizer)
    checkpoint_path = _project_outputs_path("round-trip.pt")

    save_checkpoint(source, source_optimizer, checkpoint_path, {"fold": 0})

    target = nn.Conv2d(1, 2, kernel_size=1)
    target_optimizer = torch.optim.SGD(target.parameters(), lr=0.1, momentum=0.9)
    metadata = load_checkpoint(target, target_optimizer, checkpoint_path, {"fold": 0})
    assert metadata == {"fold": 0}
    assert torch.equal(source.weight, target.weight)
```

- [x] **Step 2: Run the test to verify it fails**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests/test_checkpoint.py -v`

Expected: FAIL because `save_checkpoint` and `load_checkpoint` do not exist.

- [x] **Step 3: Write validation tests**

```python
def test_save_checkpoint_rejects_path_outside_project_outputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outputs"):
        save_checkpoint(nn.Conv2d(1, 2, 1), None, tmp_path / "outside.pt")


def test_load_checkpoint_rejects_metadata_mismatch() -> None:
    checkpoint_path = _project_outputs_path("metadata.pt")
    save_checkpoint(nn.Conv2d(1, 2, 1), None, checkpoint_path, {"fold": 0})
    with pytest.raises(ValueError, match="metadata"):
        load_checkpoint(nn.Conv2d(1, 2, 1), None, checkpoint_path, {"fold": 1})
```

- [x] **Step 4: Run the validation tests to verify they fail**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests/test_checkpoint.py -v`

Expected: FAIL because the checkpoint interface is still absent.

### Task 2: Implement the constrained payload format

**Files:**
- Modify: `standalone_nnunet2d/engine/checkpoint.py`
- Test: `standalone_nnunet2d/tests/test_checkpoint.py`

- [x] **Step 1: Add the public save/load functions**

```python
def save_checkpoint(model, optimizer, path, metadata=None) -> Path:
    resolved_path = _resolve_output_path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "metadata": dict(metadata or {}),
    }, resolved_path)
    return resolved_path


def load_checkpoint(model, optimizer, path, expected_metadata=None) -> dict[str, object]:
    payload = torch.load(_resolve_output_path(path), map_location="cpu", weights_only=False)
    _validate_payload(payload, expected_metadata)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and payload["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return dict(payload["metadata"])
```

- [x] **Step 2: Add private path/payload validation helpers**

```python
def _resolve_output_path(path: str | Path) -> Path:
    candidate = Path(path).resolve()
    candidate.relative_to(PROJECT_OUTPUTS_DIRECTORY)
    return candidate


def _validate_payload(payload: object, expected_metadata: Mapping[str, object] | None) -> None:
    if not isinstance(payload, dict) or payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported checkpoint format")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata must be a dictionary")
    for key, expected_value in (expected_metadata or {}).items():
        if metadata.get(key) != expected_value:
            raise ValueError("checkpoint metadata does not match expectations")
```

- [x] **Step 3: Run the focused checkpoint tests**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests/test_checkpoint.py -v`

Expected: PASS with three tests.

### Task 3: Document and verify the interface

**Files:**
- Modify: `standalone_nnunet2d/README.md`
- Modify: `standalone_nnunet2d/REPRODUCTION_NOTES.md`
- Modify: `standalone_nnunet2d/CHECKPOINT_IMPLEMENTATION_PLAN.md`

- [x] **Step 1: Document the non-training boundary**

Add a README paragraph stating that checkpoint functions are caller-invoked, write only below `outputs/`, and do not create a training loop or copy external weights.

- [x] **Step 2: Record the interface contract in reproduction notes**

Document the format-version, model/optimizer/metadata payload fields, and metadata expectation behavior.

- [x] **Step 3: Mark completed plan items**

Replace each completed `- [ ]` item above with `- [x]` after its associated verification has passed.

- [x] **Step 4: Run complete validation**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests -v`

Expected: PASS for the full standalone test suite.

- [x] **Step 5: Inspect the diff before handoff**

Run: `git diff --check` and `git status --short`

Expected: no whitespace errors; report all files left uncommitted for user review.
