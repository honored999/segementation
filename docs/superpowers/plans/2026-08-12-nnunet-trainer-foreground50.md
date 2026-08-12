# nnU-Net Trainer foreground sampling 50% Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a portable official nnU-Net v2 Trainer extension that changes only foreground-patch oversampling to 50%, together with server-side TDD verification and reproducible fold-0 commands.

**Architecture:** Keep the standalone reproduction unchanged. Add a small external Trainer package under the workspace that is discovered by the official runtime using `nnUNet_extTrainer` on Windows. The Trainer subclass has one responsibility: it delegates everything to official `nnUNetTrainer` while replacing its foreground oversampling value with `0.50`.

**Tech Stack:** Python, official `nnunetv2`, pytest, Windows PowerShell, Conda environment `nnunet5090`, `nnUNet_extTrainer`.

---

## File structure

- Create: `nnunet_ext_trainers/nnUNetTrainerForeground50.py` — external official Trainer subclass; no standalone imports.
- Create: `nnunet_ext_trainers/tests/test_trainer_foreground50.py` — official-runtime integration test for inheritance and effective setting.
- Create: `nnunet_ext_trainers/README.md` — server synchronization, test, train, resume, and full-volume fold-0 validation commands.
- Modify: `docs/superpowers/specs/2026-08-12-nnunet-trainer-foreground50-design.md` — only if the installed official Trainer version proves a different minimal override is required; record the evidence.

The extension directory is copied or synchronized as source to the server. Do not edit the installed `nnunetv2` package. Do not modify `standalone_nnunet2d/`, existing outputs, plans, dataset splits, or batch size.

### Task 1: Write server-runtime RED test

**Files:**
- Create: `nnunet_ext_trainers/tests/test_trainer_foreground50.py`

- [ ] **Step 1: Write the failing test before creating the Trainer module**

```python
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnUNetTrainerForeground50 import nnUNetTrainerForeground50


def test_foreground50_trainer_inherits_and_overrides_only_sampling_rate() -> None:
    assert issubclass(nnUNetTrainerForeground50, nnUNetTrainer)
    assert nnUNetTrainerForeground50.OVERSAMPLE_FOREGROUND_PERCENT == 0.50
```

- [ ] **Step 2: Run the focused test in the server environment and verify RED**

```powershell
conda run -n nnunet5090 python -m pytest nnunet_ext_trainers\tests\test_trainer_foreground50.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'nnUNetTrainerForeground50'`. If it instead fails because the official base-class import path differs, record the installed `nnunetv2` version and corrected import path before continuing; do not write a compatibility shim.

### Task 2: Implement the minimal external Trainer

**Files:**
- Create: `nnunet_ext_trainers/nnUNetTrainerForeground50.py`
- Test: `nnunet_ext_trainers/tests/test_trainer_foreground50.py`

- [ ] **Step 1: Implement only the subclass constant and initialization override**

```python
from __future__ import annotations

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerForeground50(nnUNetTrainer):
    """Official nnU-Net Trainer with 50% foreground-patch oversampling."""

    OVERSAMPLE_FOREGROUND_PERCENT = 0.50

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.oversample_foreground_percent = self.OVERSAMPLE_FOREGROUND_PERCENT
```

Do not override network construction, plans, batch size, loss, augmentation, optimizer, scheduler, epoch count, random seed policy, validation, or inference.

- [ ] **Step 2: Run the focused test and verify GREEN**

```powershell
conda run -n nnunet5090 python -m pytest nnunet_ext_trainers\tests\test_trainer_foreground50.py -q
```

Expected: `1 passed`.

- [ ] **Step 3: Verify the value is effective after real official initialization**

Extend the test with a real `nnUNetTrainerForeground50(...)` construction using the official installed version's smallest valid plans/configuration fixture. Assert the initialized instance has `oversample_foreground_percent == 0.50`. First run this added assertion and ensure it fails if the value is not effective, then make only the minimal override required by the installed source to pass it.

- [ ] **Step 4: Re-run the focused integration test**

```powershell
conda run -n nnunet5090 python -m pytest nnunet_ext_trainers\tests\test_trainer_foreground50.py -q
```

Expected: all tests pass. Save the exact command output and `python -c "import nnunetv2; print(nnunetv2.__file__)"` result in the experiment log.

### Task 3: Verify external discovery and document the frozen experiment command

**Files:**
- Create: `nnunet_ext_trainers/README.md`
- Test: `nnunet_ext_trainers/tests/test_trainer_foreground50.py`

- [ ] **Step 1: Add a discovery test that uses the runtime's external Trainer resolver**

```python
def test_foreground50_trainer_is_discoverable_from_external_directory(monkeypatch) -> None:
    monkeypatch.setenv("nnUNet_extTrainer", str(ROOT))
    # Import through the exact official resolver used by the installed release.
    # Assert the resolved class is nnUNetTrainerForeground50.
```

Before completing this test, inspect the installed `nnunetv2` resolver and replace the comment with its exact import and assertion. Run it first with an unset/misspelled `nnUNet_extTrainer` value and observe resolution failure. Then set `nnUNet_extTrainer` to the extension directory and verify resolution succeeds. Do not test a hand-written import in place of the official resolver.

- [ ] **Step 2: Add the PowerShell operating instructions**

Document commands equivalent to:

```powershell
$env:nnUNet_extTrainer = "C:\path\to\nnunet_ext_trainers"
conda run -n nnunet5090 python -m pytest "C:\path\to\nnunet_ext_trainers\tests\test_trainer_foreground50.py" -q
nnUNetv2_train 501 2d 0 -tr nnUNetTrainerForeground50
```

Document that the actual dataset identifier/configuration must match the baseline run, and the source extension path must remain available for training, resume, validation, and inference. Include the exact output naming convention `nnUNetTrainerForeground50__nnUNetPlans__2d/fold_0`; do not reuse the original trainer directory.

- [ ] **Step 3: Run discovery and focused integration checks in the server environment**

```powershell
$env:nnUNet_extTrainer = (Resolve-Path "nnunet_ext_trainers").Path
conda run -n nnunet5090 python -m pytest nnunet_ext_trainers\tests\test_trainer_foreground50.py -q
```

Expected: all tests pass and the reported resolved class is `nnUNetTrainerForeground50`.

### Task 4: Run the isolated fold-0 screen and validate its full volumes

**Files:**
- Create: server experiment directory only, outside the Git worktree's tracked source tree.
- Read: baseline fold-0 full-volume report and custom-trainer fold-0 validation outputs.

- [ ] **Step 1: Record immutable run inputs**

Capture the official `nnunetv2` version, Trainer extension file SHA256, plans SHA256, split file SHA256, command line, CUDA/GPU identity, and output directory before training. Confirm batch size remains 12 and the output directory is new.

- [ ] **Step 2: Run the custom Trainer for fold 0 only**

```powershell
$env:nnUNet_extTrainer = "C:\path\to\nnunet_ext_trainers"
nnUNetv2_train 501 2d 0 -tr nnUNetTrainerForeground50
```

Use the same baseline epoch schedule and performance settings. If resuming, use only the exact custom Trainer output directory and unchanged extension source; otherwise start with a fresh directory. Do not resume the baseline default-Trainer checkpoint under the custom Trainer.

- [ ] **Step 3: Run complete-volume fold-0 validation**

Use the official full-volume validation/prediction route against all 19 fold-0 validation cases, saving predictions separately from the baseline. Evaluate class 1 after argmax for each reconstructed 3D patient: both empty = 1, one empty = 0, `2TP/(2TP+FP+FN)` otherwise. Save per-case scores and their equal-case macro mean.

- [ ] **Step 4: Decide promotion using the declared score only**

Compare the saved full-volume 19-case macro foreground Dice with `0.73225151385`. Promote only if the score is at least `0.74225151385`; otherwise record the negative screen and stop before five-fold training. Never use online patch/global validation Dice to promote the variant.

### Task 5: Final verification and handoff

**Files:**
- Read: `nnunet_ext_trainers/nnUNetTrainerForeground50.py`
- Read: `nnunet_ext_trainers/tests/test_trainer_foreground50.py`
- Read: `nnunet_ext_trainers/README.md`
- Read: custom fold-0 full-volume result artifacts.

- [ ] **Step 1: Confirm source scope**

```powershell
git status --short
git diff -- nnunet_ext_trainers docs\superpowers
```

Expected: only the extension package, test, documentation, and already-approved planning/spec files are changed. Preserve unrelated pre-existing files exactly.

- [ ] **Step 2: Run all portable source checks available locally**

```powershell
conda run -n newconda python -m py_compile nnunet_ext_trainers\nnUNetTrainerForeground50.py
```

Expected: exit code 0. The authoritative integration test remains the `nnunet5090` test from Tasks 2–3 because `newconda` does not provide `nnunetv2`.

- [ ] **Step 3: Report evidence and do not commit**

Report the custom Trainer path, exact override, server test output, version/path provenance, fold-0 full-volume result field, case count, failed-case count, baseline delta, and promotion decision. Do not commit, push, relabel the standalone baseline, or claim a five-fold result.

## Plan self-review

The plan covers all validated design requirements: isolated Trainer-only change, external discoverability, strict server-side RED/GREEN validation, untouched standalone baseline, full-volume case-macro evaluation, explicit promotion threshold, resume constraints, and no commit/push. It deliberately leaves server execution conditional on access to `nnunet5090`; the local worktree cannot establish server runtime behavior.
