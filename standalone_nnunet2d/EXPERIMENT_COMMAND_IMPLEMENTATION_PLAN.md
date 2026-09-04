# Explicit Experiment Command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a validated experiment request command that remains non-training until a later phase.

**Architecture:** `experiment.py` parses into a frozen request, constrains output paths below `outputs/`, and composes the existing preflight report. Default execution is reporting only; `--confirm-run` returns a documented nonzero deferred status without side effects.

**Tech Stack:** Python 3.14, argparse, dataclasses, pathlib, json, pytest.

---

### Task 1: Add failing command tests

**Files:**
- Create: `standalone_nnunet2d/tests/test_experiment.py`

- [ ] **Step 1: Write the valid default-request test**

```python
def test_experiment_reports_valid_request_without_training(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    raw, preprocessed, results = _roots(tmp_path)
    output = _output_path("default")
    assert main(_arguments(raw, preprocessed, results, output)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["request"]["fold"] == 0
    assert payload["request"]["epochs"] == 1
    assert payload["execution"] == "not-confirmed"
    assert not output.exists()
```

- [ ] **Step 2: Write invalid/confirmed request tests**

```python
def test_experiment_rejects_invalid_fold(tmp_path: Path) -> None:
    raw, preprocessed, results = _roots(tmp_path)
    with pytest.raises(SystemExit) as error:
        main(_arguments(raw, preprocessed, results, _output_path("bad-fold"), fold="5"))
    assert error.value.code == 2


def test_experiment_rejects_output_outside_project_outputs(tmp_path: Path) -> None:
    raw, preprocessed, results = _roots(tmp_path)
    with pytest.raises(SystemExit) as error:
        main(_arguments(raw, preprocessed, results, tmp_path / "outside"))
    assert error.value.code == 2


def test_confirmed_experiment_is_deferred_without_writing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    raw, preprocessed, results = _roots(tmp_path)
    output = _output_path("confirmed")
    assert main(_arguments(raw, preprocessed, results, output, confirm=True)) == 3
    assert json.loads(capsys.readouterr().out)["execution"] == "deferred"
    assert not output.exists()
```

- [ ] **Step 3: Run tests to verify RED**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests/test_experiment.py -v`

Expected: FAIL because `experiment.main` does not exist.

### Task 2: Implement the non-training command contract

**Files:**
- Create: `standalone_nnunet2d/experiment.py`
- Test: `standalone_nnunet2d/tests/test_experiment.py`

- [ ] **Step 1: Add request parsing and output-path constraint**

```python
@dataclass(frozen=True)
class ExperimentRequest:
    raw_root: Path
    preprocessed_root: Path
    results_root: Path
    output_root: Path
    fold: int
    epochs: int
    device: str | None


def _output_path(value: Path) -> Path:
    resolved = value.expanduser().resolve()
    try:
        resolved.relative_to(PROJECT_OUTPUTS_DIRECTORY.resolve())
    except ValueError as error:
        raise argparse.ArgumentTypeError("output root must be under standalone_nnunet2d/outputs") from error
    return resolved
```

- [ ] **Step 2: Add JSON report and deferred confirmation branch**

```python
report = inspect_server_readiness(request.raw_root, request.preprocessed_root, request.results_root, device=request.device)
payload = {"request": _json_request(request), "readiness": report, "execution": "deferred" if args.confirm_run else "not-confirmed"}
print(json.dumps(payload, indent=2, sort_keys=True))
return 3 if args.confirm_run else (0 if report["ready"] else 2)
```

- [ ] **Step 3: Run focused tests**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests/test_experiment.py -v`

Expected: PASS for four tests.

### Task 3: Document and verify

**Files:**
- Modify: `standalone_nnunet2d/README.md`
- Modify: `standalone_nnunet2d/REPRODUCTION_NOTES.md`
- Modify: `standalone_nnunet2d/EXPERIMENT_COMMAND_IMPLEMENTATION_PLAN.md`

- [ ] **Step 1: Document the request arguments and exit codes 0, 2, and 3**
- [ ] **Step 2: State that `--confirm-run` remains deferred and creates no output**
- [ ] **Step 3: Mark completed items and run `conda run -n newconda python -m pytest standalone_nnunet2d/tests -v` plus `git diff --check`**
