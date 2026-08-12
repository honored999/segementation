# Alignment Evidence Promotion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking. Do not commit or push.

**Goal:** Validate the two passed parity reports once, embed their evidence in new formal checkpoints, and propagate `official_aligned` through prediction, fold validation, and strict five-fold OOF aggregation.

**Architecture:** A focused `alignment_evidence.py` module owns report parsing, validation, hashing, and embedded-record validation. Training resolves pending versus aligned state before side effects. Downstream commands trust aligned checkpoints only after validating the embedded record and propagate it without changing metric or inference behavior.

**Tech Stack:** Python, JSON, hashlib, PyTorch checkpoint metadata, pytest.

---

### Task 1: Alignment evidence validation core

**Files:**
- Create: `standalone_nnunet2d/alignment_evidence.py`
- Create: `standalone_nnunet2d/tests/test_alignment_evidence.py`

- [ ] Write tests for a valid exact transform report and valid
  `repeat_oracle_stability_v1` report producing `official_aligned` evidence.
- [ ] Run `conda run -n newconda python -m pytest -q standalone_nnunet2d/tests/test_alignment_evidence.py` and verify RED because the module/API is absent.
- [ ] Implement these public contracts:

```python
OFFICIAL_ALIGNED = "official_aligned"
ALIGNMENT_EVIDENCE_POLICY = "transform_exact_plus_repeat_oracle_stability_v1"

def build_alignment_evidence(
    transform_report_path: Path,
    inference_report_path: Path,
) -> dict[str, Any]: ...

def validate_alignment_evidence_record(value: object) -> dict[str, Any]: ...

def resolve_alignment_state(
    transform_report_path: Path | None,
    inference_report_path: Path | None,
) -> tuple[str, dict[str, Any] | None]: ...
```

- [ ] Add separate failing tests and minimal implementations for malformed
  JSON, missing report pairs, failed components, nonzero tolerance, wrong
  inference policy, fewer than three or duplicate roots, stable mismatches, and
  unobserved labels.
- [ ] Run the focused test to GREEN, then `py_compile` both files and
  `git diff --check`.

### Task 2: Formal training and checkpoint integration

**Files:**
- Modify: `standalone_nnunet2d/formal_train.py`
- Modify: `standalone_nnunet2d/training/formal_checkpoint.py`
- Modify: `standalone_nnunet2d/tests/test_formal_checkpoint.py`
- Modify: `standalone_nnunet2d/tests/test_performance_phase1.py`

- [ ] First add failing tests proving the parser exposes both report flags,
  `build_formal_config` records aligned evidence in its plan hash, and missing
  one report fails before output creation.
- [ ] Add failing checkpoint tests proving aligned save/resume succeeds only
  with valid embedded evidence and pending behavior remains unchanged.
- [ ] Run the two focused test files and verify expected RED failures.
- [ ] Resolve evidence before `--confirm-run` side effects; set printed,
  configured, and saved `run_type`/`run_state` from the resolved state.
- [ ] Permit `save_formal_checkpoint` and `load_formal_checkpoint` to handle
  only pending-without-evidence or aligned-with-valid-evidence; reject all
  inconsistent combinations. Include evidence in metadata and expected resume
  checks.
- [ ] Run focused tests to GREEN, then py_compile changed files and
  `git diff --check`.

### Task 3: Prediction, fold validation, and OOF propagation

**Files:**
- Modify: `standalone_nnunet2d/predict.py`
- Modify: `standalone_nnunet2d/engine/formal_validation.py`
- Modify: `standalone_nnunet2d/validate_cv.py`
- Modify: `standalone_nnunet2d/tests/test_predict_command.py`
- Modify: `standalone_nnunet2d/tests/test_formal_validation.py`
- Modify: `standalone_nnunet2d/tests/test_validate_cv.py`

- [ ] Add failing prediction tests: pending still needs `--allow-pending`, valid
  aligned evidence is accepted and copied, and aligned-without-evidence fails.
- [ ] Add failing fold tests for aligned state/evidence propagation into
  `fold_<N>_report.json`.
- [ ] Add failing OOF tests: five identical aligned evidence records yield an
  aligned summary; mixed states, missing evidence, or differing evidence fail.
- [ ] Run focused tests and verify RED for missing propagation.
- [ ] Add one shared checkpoint-state validator, pass state/evidence into
  `validate_fold`, and derive aggregate state from all five reports without
  changing full-volume prediction or metric policy.
- [ ] Run focused tests to GREEN, then py_compile changed files and
  `git diff --check`.

### Task 4: Documentation and server workflow

**Files:**
- Modify: `standalone_nnunet2d/README.md`
- Modify: `standalone_nnunet2d/REPRODUCTION_NOTES.md`
- Modify: `standalone_nnunet2d/tests/test_documentation_contract.py`

- [ ] Add failing documentation tests requiring both new CLI flags, explicit
  no-retroactive-promotion language, throughput profile, five folds, direct
  `validate_cv.py fold`, and strict 95-case aggregation.
- [ ] Run the documentation contract and verify RED.
- [ ] Document source synchronization, new output roots, fold 0-4 training,
  direct full-volume fold validation without duplicate `predict.py`, and final
  aggregation commands using the user-provided Windows paths.
- [ ] Run the documentation contract to GREEN and run py_compile/diff checks.

### Final verification

- [ ] Run `conda run -n newconda python -m pytest -q`.
- [ ] Run `conda run -n newconda python -m py_compile` over every Python file
  below `standalone_nnunet2d`.
- [ ] Run `git diff --check` and inspect `git status --short`.
- [ ] Report changed files, fresh test counts, hashes of server runtime files,
  and exact fold 0-4 commands. Never call pre-existing training artifacts
  official reproduction.
