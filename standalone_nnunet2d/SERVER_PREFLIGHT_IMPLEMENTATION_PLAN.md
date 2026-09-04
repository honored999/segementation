# Server Preflight and Safe Dry-Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only server readiness report and dry-run CLI without enabling formal training.

**Architecture:** `tools/server_preflight.py` will collect path, PyTorch-device, and local-reference facts into one JSON-safe report. `dry_run.py` will only parse explicit paths, print the report, and signal readiness through its exit code.

**Tech Stack:** Python 3.14, PyTorch, pathlib, argparse, json, pytest.

---

### Task 1: Define readiness behavior with failing tests

**Files:**
- Create: `standalone_nnunet2d/tests/test_server_preflight.py`
- Create: `standalone_nnunet2d/tests/test_dry_run.py`

- [x] **Step 1: Write the failing reusable-function tests**

```python
def test_inspect_server_readiness_reports_cpu_and_reference_plan(tmp_path: Path) -> None:
    roots = [tmp_path / name for name in ("raw", "preprocessed", "results")]
    for root in roots:
        root.mkdir()

    report = inspect_server_readiness(*roots, device="cpu")

    assert report["ready"] is True
    assert report["device"]["selected"] == "cpu"
    assert report["plan"]["patch_size"] == [512, 512]
    assert report["plan"]["batch_size"] == 12


def test_inspect_server_readiness_diagnoses_missing_directory(tmp_path: Path) -> None:
    raw_root, preprocessed_root = tmp_path / "raw", tmp_path / "preprocessed"
    raw_root.mkdir()
    preprocessed_root.mkdir()

    report = inspect_server_readiness(raw_root, preprocessed_root, tmp_path / "missing", device="cpu")

    assert report["ready"] is False
    assert any("results" in message for message in report["diagnostics"])


def test_inspect_server_readiness_rejects_unavailable_cuda_device(tmp_path: Path) -> None:
    roots = [tmp_path / name for name in ("raw", "preprocessed", "results")]
    for root in roots:
        root.mkdir()

    report = inspect_server_readiness(*roots, device="cuda:999")

    assert report["ready"] is False
    assert any("cuda:999" in message for message in report["diagnostics"])
```

- [x] **Step 2: Run the function tests to verify RED**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests/test_server_preflight.py -v`

Expected: FAIL because `server_preflight` does not exist.

- [x] **Step 3: Write the failing CLI test**

```python
def test_dry_run_prints_json_and_returns_zero_when_ready(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    roots = [tmp_path / name for name in ("raw", "preprocessed", "results")]
    for root in roots:
        root.mkdir()

    exit_code = main([
        "--raw-root", str(roots[0]),
        "--preprocessed-root", str(roots[1]),
        "--results-root", str(roots[2]),
        "--device", "cpu",
    ])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["ready"] is True


def test_dry_run_rejects_run_flag(tmp_path: Path) -> None:
    roots = [tmp_path / name for name in ("raw", "preprocessed", "results")]
    for root in roots:
        root.mkdir()
    with pytest.raises(SystemExit) as error:
        main([
            "--raw-root", str(roots[0]),
            "--preprocessed-root", str(roots[1]),
            "--results-root", str(roots[2]),
            "--run",
        ])
    assert error.value.code == 2
```

- [x] **Step 4: Run the CLI test to verify RED**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests/test_dry_run.py -v`

Expected: FAIL because `dry_run` does not exist.

### Task 2: Implement read-only preflight and CLI

**Files:**
- Create: `standalone_nnunet2d/tools/server_preflight.py`
- Create: `standalone_nnunet2d/dry_run.py`
- Test: `standalone_nnunet2d/tests/test_server_preflight.py`
- Test: `standalone_nnunet2d/tests/test_dry_run.py`

- [x] **Step 1: Add `inspect_server_readiness`**

```python
def inspect_server_readiness(raw_root: Path, preprocessed_root: Path, results_root: Path, *, device: str | None = None) -> dict[str, object]:
    cuda_available = torch.cuda.is_available()
    selected = torch.device(device or ("cuda" if cuda_available else "cpu"))
    diagnostics = [f"{name} directory is missing: {path}" for name, path in {
        "raw": raw_root, "preprocessed": preprocessed_root, "results": results_root,
    }.items() if not path.is_dir()]
    diagnostics.extend(_device_diagnostics(selected, cuda_available, torch.cuda.device_count() if cuda_available else 0))
    plan = _plan_facts(diagnostics)
    return {
        "ready": not diagnostics,
        "diagnostics": diagnostics,
        "paths": {"raw": str(raw_root), "preprocessed": str(preprocessed_root), "results": str(results_root)},
        "device": {"selected": str(selected), "cuda_available": cuda_available, "gpu_count": torch.cuda.device_count() if cuda_available else 0},
        "plan": plan,
    }
```

- [x] **Step 2: Add plan/device helper validation**

```python
def _device_diagnostics(device: torch.device, cuda_available: bool, gpu_count: int) -> list[str]:
    if device.type != "cuda":
        return []
    index = 0 if device.index is None else device.index
    if not cuda_available or index >= gpu_count:
        return [f"requested CUDA device is unavailable: {device}"]
    return []


def _plan_facts(diagnostics: list[str]) -> dict[str, object]:
    try:
        inspection = inspect_reference(DEFAULT_REFERENCE_DIR)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        diagnostics.append(f"reference plan is unreadable: {error}")
        return {}
    return {
        "patch_size": list(inspection.patch_size),
        "batch_size": inspection.batch_size,
        "stages": inspection.n_stages,
    }
```

- [x] **Step 3: Add the explicit dry-run CLI**

```python
def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only standalone nnU-Net server preflight")
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--preprocessed-root", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--device")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(arguments)
    if args.run:
        parser.error("--run is not supported; dry_run never starts training")
    report = inspect_server_readiness(args.raw_root, args.preprocessed_root, args.results_root, device=args.device)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready"] else 2
```

- [x] **Step 4: Run focused preflight tests**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests/test_server_preflight.py standalone_nnunet2d/tests/test_dry_run.py -v`

Expected: PASS for five tests.

### Task 3: Document and fully verify the safe boundary

**Files:**
- Modify: `standalone_nnunet2d/README.md`
- Modify: `standalone_nnunet2d/REPRODUCTION_NOTES.md`
- Modify: `standalone_nnunet2d/SERVER_PREFLIGHT_IMPLEMENTATION_PLAN.md`

- [x] **Step 1: Document the exact dry-run command and exit behavior**

Add a command example with all three roots and explain exit 0 versus exit 2; state that no directory or model artifact is created.

- [x] **Step 2: Record the reported plan/device facts and deferred training boundary**

Document path checks, CUDA facts, patch/batch reporting, and the rejected `--run` flag.

- [x] **Step 3: Mark completed plan items**

Replace each completed checkbox with `- [x]` after its corresponding verification passes.

- [x] **Step 4: Run the full suite and inspect changes**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests -v`, then `git diff --check` and `git status --short`.

Expected: all tests pass, no whitespace errors, and no automatic commit.
