# Train and Validation Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit, tested single-epoch training and binary validation functions without enabling formal training.

**Architecture:** `trainer.py` will iterate caller-supplied batches through the existing `train_step` and return a mean-loss aggregate. `validator.py` will run caller-supplied batches under `no_grad`, select the full-resolution logits, aggregate binary confusion counts, restore the model's original mode, and return mean loss plus global Dice/IoU.

**Tech Stack:** Python 3.14, PyTorch, NumPy, pytest.

---

### Task 1: Add failing epoch-level tests

**Files:**
- Modify: `standalone_nnunet2d/tests/test_trainer.py`
- Create: `standalone_nnunet2d/tests/test_validator.py`

- [x] **Step 1: Write the failing train-epoch tests**

```python
import pytest


def _tiny_model_and_optimizer() -> tuple[nn.Conv2d, torch.optim.Optimizer]:
    model = nn.Conv2d(1, 2, kernel_size=1)
    return model, torch.optim.SGD(model.parameters(), lr=0.1)


def test_run_train_epoch_updates_model_and_averages_batch_losses() -> None:
    model, optimizer = _tiny_model_and_optimizer()
    before = model.weight.detach().clone()
    batches = [
        (torch.randn(2, 1, 4, 4), torch.randint(0, 2, (2, 4, 4))),
        (torch.randn(2, 1, 4, 4), torch.randint(0, 2, (2, 4, 4))),
    ]

    result = run_train_epoch(model, batches, DiceCrossEntropyLoss(), optimizer, torch.device("cpu"))

    assert result.batch_count == 2
    assert result.mean_loss >= 0
    assert result.output_shapes == ((2, 2, 4, 4),)
    assert not torch.equal(before, model.weight.detach())


def test_run_train_epoch_rejects_empty_batches() -> None:
    model, optimizer = _tiny_model_and_optimizer()
    with pytest.raises(ValueError, match="empty"):
        run_train_epoch(model, [], DiceCrossEntropyLoss(), optimizer, torch.device("cpu"))
```

- [x] **Step 2: Run the train-epoch tests to verify they fail**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests/test_trainer.py -v`

Expected: FAIL because `run_train_epoch` does not exist.

- [x] **Step 3: Write the failing validation tests**

```python
import pytest


class _PerfectBinaryModel(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.stack((1 - image[:, 0], image[:, 0]), dim=1) * 20


def test_run_validation_epoch_reports_perfect_metrics_and_restores_train_mode() -> None:
    model = _PerfectBinaryModel()
    model.train()
    batch = (torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]]), torch.tensor([[[0, 1], [1, 0]]]))

    result = run_validation_epoch(model, [batch], DiceCrossEntropyLoss(), torch.device("cpu"))

    assert result.batch_count == 1
    assert result.dice == 1.0
    assert result.iou == 1.0
    assert model.training


def test_run_validation_epoch_rejects_empty_batches() -> None:
    with pytest.raises(ValueError, match="empty"):
        run_validation_epoch(_PerfectBinaryModel(), [], DiceCrossEntropyLoss(), torch.device("cpu"))
```

- [x] **Step 4: Run the validation tests to verify they fail**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests/test_validator.py -v`

Expected: FAIL because `run_validation_epoch` does not exist.

### Task 2: Implement explicit epoch aggregates

**Files:**
- Modify: `standalone_nnunet2d/engine/trainer.py`
- Modify: `standalone_nnunet2d/engine/validator.py`
- Test: `standalone_nnunet2d/tests/test_trainer.py`
- Test: `standalone_nnunet2d/tests/test_validator.py`

- [x] **Step 1: Add `TrainEpochResult` and `run_train_epoch`**

```python
from collections.abc import Iterable

@dataclass(frozen=True)
class TrainEpochResult:
    batch_count: int
    mean_loss: float
    output_shapes: tuple[tuple[int, ...], ...]


def run_train_epoch(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
    loss_fn: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> TrainEpochResult:
    results = [train_step(model, batch, loss_fn, optimizer, device) for batch in batches]
    if not results:
        raise ValueError("training batch iterable is empty")
    return TrainEpochResult(
        batch_count=len(results),
        mean_loss=sum(result.loss for result in results) / len(results),
        output_shapes=results[-1].output_shapes,
    )
```

- [x] **Step 2: Add `ValidationEpochResult` and `run_validation_epoch`**

```python
from collections.abc import Iterable

@dataclass(frozen=True)
class ValidationEpochResult:
    batch_count: int
    mean_loss: float
    dice: float
    iou: float


def run_validation_epoch(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
    loss_fn: nn.Module,
    device: torch.device,
) -> ValidationEpochResult:
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            records = [_validate_batch(model, batch, loss_fn, device) for batch in batches]
    finally:
        model.train(was_training)
    if not records:
        raise ValueError("validation batch iterable is empty")
    true_positive = sum(int(record[1]["TP"]) for record in records)
    false_positive = sum(int(record[1]["FP"]) for record in records)
    false_negative = sum(int(record[1]["FN"]) for record in records)
    denominator = 2 * true_positive + false_positive + false_negative
    union = true_positive + false_positive + false_negative
    return ValidationEpochResult(
        batch_count=len(records),
        mean_loss=sum(record[0] for record in records) / len(records),
        dice=1.0 if denominator == 0 else 2 * true_positive / denominator,
        iou=1.0 if union == 0 else true_positive / union,
    )
```

- [x] **Step 3: Implement full-resolution binary validation**

```python
def _full_resolution_logits(outputs: Tensor | tuple[Tensor, ...]) -> Tensor:
    return outputs if isinstance(outputs, Tensor) else outputs[0]


def _validate_batch(model, batch, loss_fn, device) -> tuple[float, dict[str, float | int]]:
    image, target = (value.to(device) for value in batch)
    logits = _full_resolution_logits(model(image))
    if logits.ndim != 4 or logits.shape[1] != 2 or logits.shape[0] != target.shape[0] or logits.shape[2:] != target.shape[1:]:
        raise ValueError("validation requires full-resolution two-channel binary logits")
    loss = loss_fn(logits, target)
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite validation loss")
    return float(loss.cpu()), binary_segmentation_metrics(logits.argmax(dim=1).cpu().numpy(), target.cpu().numpy())
```

- [x] **Step 4: Run focused epoch tests**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests/test_trainer.py standalone_nnunet2d/tests/test_validator.py -v`

Expected: PASS for the five epoch-level tests.

### Task 3: Document and fully verify the non-training interface

**Files:**
- Modify: `standalone_nnunet2d/README.md`
- Modify: `standalone_nnunet2d/REPRODUCTION_NOTES.md`
- Modify: `standalone_nnunet2d/TRAIN_VALIDATION_ENGINE_IMPLEMENTATION_PLAN.md`

- [x] **Step 1: Describe epoch functions without a runnable training command**

Add documentation that `run_train_epoch` and `run_validation_epoch` require caller-supplied batches and do not change `train.py` or enable formal training.

- [x] **Step 2: Record binary validation semantics**

Document that validation uses full-resolution two-channel logits, argmax masks,
global confusion-count Dice/IoU, CPU tensor conversion for metrics, and mode
restoration.

- [x] **Step 3: Mark completed plan items**

Replace each completed `- [ ]` item above with `- [x]` after its associated verification has passed.

- [x] **Step 4: Run complete validation and inspect the worktree**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests -v`, then `git diff --check` and `git status --short`.

Expected: all standalone tests pass, no whitespace errors, and no automatic commit.
