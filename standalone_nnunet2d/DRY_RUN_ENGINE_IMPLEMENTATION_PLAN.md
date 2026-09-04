# Single-Step Dry-Run Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a testable one-batch optimization step while retaining the formal-training block.

**Architecture:** `engine/trainer.py` provides a small `train_step`; callers supply model, compound/deep-supervision loss, optimizer, and batch. It neither iterates epochs nor reads data.

**Tech Stack:** PyTorch, pytest.

---

### Task 1: Test-first one-step engine

**Files:**
- Modify: `standalone_nnunet2d/engine/trainer.py`
- Create: `standalone_nnunet2d/tests/test_trainer.py`

- [ ] **Step 1: Write a failing parameter-update test**

```python
def test_train_step_updates_tiny_model_parameters() -> None:
    model = nn.Conv2d(1, 2, 1); optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.weight.detach().clone()
    result = train_step(model, (torch.randn(2, 1, 4, 4), torch.randint(0, 2, (2, 4, 4))), DiceCrossEntropyLoss(), optimizer, torch.device("cpu"))
    assert result.loss >= 0 and not torch.equal(before, model.weight.detach())
```

- [ ] **Step 2: Run and verify failure because `train_step` is absent**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests/test_trainer.py -v`

- [ ] **Step 3: Implement the bounded interface**

```python
@dataclass(frozen=True)
class TrainStepResult: loss: float; output_shapes: tuple[tuple[int, ...], ...]

def train_step(model: nn.Module, batch: tuple[Tensor, Tensor], loss_fn: nn.Module, optimizer: Optimizer, device: torch.device) -> TrainStepResult:
    model.train(); image, target = (value.to(device) for value in batch); optimizer.zero_grad(set_to_none=True)
    outputs = model(image); loss = loss_fn(outputs, target)
    if not torch.isfinite(loss): raise FloatingPointError("non-finite training loss")
    loss.backward(); optimizer.step()
    levels = (outputs,) if isinstance(outputs, Tensor) else tuple(outputs)
    return TrainStepResult(float(loss.detach().cpu()), tuple(tuple(level.shape) for level in levels))
```

- [ ] **Step 4: Run the test and full suite**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests -v`

Expected: all tests pass; `train.py` is unchanged and still exits.

## Plan self-review

The only side effect is the one explicit optimizer step in its unit test. No
epoch loop, dataset traversal, checkpoint, or CLI training path is added.
