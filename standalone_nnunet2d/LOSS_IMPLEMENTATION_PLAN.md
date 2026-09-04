# Loss and Deep-Supervision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add pure-PyTorch batch Dice, Dice-plus-cross-entropy, and explicit-weight deep-supervision losses without adding a training loop.

**Architecture:** `dice.py` validates logits/targets and computes foreground batch Dice from softmax probabilities. `compound.py` combines it with PyTorch cross-entropy. `deep_supervision.py` resizes integer targets by nearest neighbor and normalizes only caller-provided positive scale weights.

**Tech Stack:** Python 3.14, PyTorch, pytest; no `nnunetv2` or `dynamic_network_architectures` imports.

---

### Task 1: Batch Dice loss (completed)

**Files:**
- Modify: `standalone_nnunet2d/losses/dice.py`
- Modify: `standalone_nnunet2d/tests/test_loss.py`

- [ ] **Step 1: Write failing Dice behavior tests**

```python
def test_batch_dice_is_near_zero_for_confident_correct_foreground() -> None:
    logits = torch.tensor([[[[10.0, -10.0]], [[-10.0, 10.0]]]], requires_grad=True)
    target = torch.tensor([[[0, 1]]])
    loss = SoftDiceLoss(batch_dice=True, include_background=False)(logits, target)
    assert loss.item() < 1e-4

def test_batch_dice_rejects_wrong_target_shape() -> None:
    with pytest.raises(ValueError, match="target"):
        SoftDiceLoss()(torch.randn(1, 2, 4, 4), torch.zeros(1, 1, 4, 4, dtype=torch.long))
```

- [ ] **Step 2: Run the test and verify it fails due to the absent loss class**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests/test_loss.py -v`

Expected: import failure for `SoftDiceLoss`.

- [ ] **Step 3: Implement the validated batch Dice class**

```python
def _validate_logits_and_target(logits: Tensor, target: Tensor) -> None:
    if logits.ndim != 4 or target.ndim != 3 or logits.shape[0] != target.shape[0] or logits.shape[2:] != target.shape[1:]:
        raise ValueError("logits must be (B, C, H, W) and target must be matching (B, H, W)")
    if logits.shape[1] < 2 or not torch.isfinite(logits).all():
        raise ValueError("logits require at least two finite classes")
    if target.numel() and (target.min() < 0 or target.max() >= logits.shape[1]):
        raise ValueError("target labels must be in [0, C)")

class SoftDiceLoss(nn.Module):
    def __init__(self, *, smooth: float = 1e-5, batch_dice: bool = True, include_background: bool = False) -> None:
        super().__init__()
        if smooth <= 0: raise ValueError("smooth must be positive")
        self.smooth, self.batch_dice, self.include_background = smooth, batch_dice, include_background

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        _validate_logits_and_target(logits, target)
        probabilities = torch.softmax(logits, dim=1)
        target_one_hot = F.one_hot(target.long(), num_classes=logits.shape[1]).movedim(-1, 1).to(probabilities.dtype)
        reduce_dims = (0, 2, 3) if self.batch_dice else (2, 3)
        intersection = (probabilities * target_one_hot).sum(reduce_dims)
        denominator = probabilities.sum(reduce_dims) + target_one_hot.sum(reduce_dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        if not self.include_background: dice = dice[1:] if self.batch_dice else dice[:, 1:]
        return 1.0 - dice.mean()
```

- [ ] **Step 4: Run the Dice tests and verify they pass**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests/test_loss.py -v`

Expected: Dice behavior tests pass.

### Task 2: Compound Dice and cross-entropy loss (completed)

**Files:**
- Modify: `standalone_nnunet2d/losses/compound.py`
- Modify: `standalone_nnunet2d/tests/test_loss.py`

- [ ] **Step 1: Write a failing compound-loss gradient test**

```python
def test_compound_loss_is_finite_and_backpropagates() -> None:
    logits = torch.randn(2, 2, 8, 8, requires_grad=True)
    target = torch.randint(0, 2, (2, 8, 8))
    loss = DiceCrossEntropyLoss()(logits, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
```

- [ ] **Step 2: Run the test and verify it fails due to the absent compound class**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests/test_loss.py::test_compound_loss_is_finite_and_backpropagates -v`

Expected: import failure for `DiceCrossEntropyLoss`.

- [ ] **Step 3: Implement the additive compound loss**

```python
class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, *, dice_weight: float = 1.0, ce_weight: float = 1.0, dice_kwargs: dict[str, object] | None = None) -> None:
        super().__init__()
        if dice_weight < 0 or ce_weight < 0 or dice_weight + ce_weight == 0: raise ValueError("at least one loss weight must be positive")
        self.dice_weight, self.ce_weight = dice_weight, ce_weight
        self.dice = SoftDiceLoss(**(dice_kwargs or {}))

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        return self.dice_weight * self.dice(logits, target) + self.ce_weight * F.cross_entropy(logits, target.long())
```

- [ ] **Step 4: Run the targeted compound test and verify it passes**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests/test_loss.py::test_compound_loss_is_finite_and_backpropagates -v`

Expected: `1 passed`.

### Task 3: Explicit-weight deep supervision (completed)

**Files:**
- Modify: `standalone_nnunet2d/losses/deep_supervision.py`
- Modify: `standalone_nnunet2d/tests/test_loss.py`

- [ ] **Step 1: Write failing target-resize and weight-normalization tests**

```python
def test_deep_supervision_normalizes_explicit_weights_and_resizes_targets() -> None:
    outputs = (torch.randn(1, 2, 8, 8, requires_grad=True), torch.randn(1, 2, 4, 4, requires_grad=True))
    target = torch.randint(0, 2, (1, 8, 8))
    loss = DeepSupervisionLoss(DiceCrossEntropyLoss(), weights=(2.0, 1.0))(outputs, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert outputs[0].grad is not None and outputs[1].grad is not None

def test_deep_supervision_rejects_mismatched_weight_count() -> None:
    with pytest.raises(ValueError, match="weights"):
        DeepSupervisionLoss(DiceCrossEntropyLoss(), weights=(1.0,))( (torch.randn(1, 2, 8, 8), torch.randn(1, 2, 4, 4)), torch.zeros(1, 8, 8, dtype=torch.long))
```

- [ ] **Step 2: Run the tests and verify they fail due to the absent deep-supervision class**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests/test_loss.py -v`

Expected: import failure for `DeepSupervisionLoss`.

- [ ] **Step 3: Implement nearest-neighbor target alignment and normalized aggregation**

```python
def resize_target_nearest(target: Tensor, spatial_shape: tuple[int, int]) -> Tensor:
    if target.shape[1:] == spatial_shape: return target.long()
    return F.interpolate(target.unsqueeze(1).float(), size=spatial_shape, mode="nearest").squeeze(1).long()

class DeepSupervisionLoss(nn.Module):
    def __init__(self, base_loss: nn.Module, *, weights: Sequence[float]) -> None:
        super().__init__()
        if not weights or any(weight <= 0 for weight in weights): raise ValueError("weights must all be positive")
        self.base_loss, self.weights = base_loss, tuple(weight / sum(weights) for weight in weights)

    def forward(self, outputs: Tensor | Sequence[Tensor], target: Tensor) -> Tensor:
        levels = (outputs,) if isinstance(outputs, Tensor) else tuple(outputs)
        if len(levels) != len(self.weights): raise ValueError("number of weights must match deep-supervision outputs")
        return sum(weight * self.base_loss(logits, resize_target_nearest(target, logits.shape[2:])) for logits, weight in zip(levels, self.weights, strict=True))
```

- [ ] **Step 4: Run all loss tests and verify they pass**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests/test_loss.py -v`

Expected: all Dice, compound, and deep-supervision tests pass.

### Task 4: Documentation and full regression (completed)

**Files:**
- Modify: `standalone_nnunet2d/configs/default.yaml`
- Modify: `standalone_nnunet2d/REPRODUCTION_NOTES.md`
- Modify: `standalone_nnunet2d/README.md`

- [ ] **Step 1: Add explicit non-official loss configuration**

```yaml
loss:
  dice_weight: 1.0
  cross_entropy_weight: 1.0
  dice_smooth: 1.0e-5
  include_background: false
  deep_supervision_weights: [1.0] # Must be explicitly expanded after official verification.
```

- [ ] **Step 2: Document that deep-supervision weights are caller-provided and training remains disabled**

Append the following to `REPRODUCTION_NOTES.md` and summarize it in `README.md`:

```markdown
The current Dice smooth value (1e-5), foreground-only Dice setting, and
Dice-plus-CE sum are explicit local choices. Deep-supervision weights are
caller-provided and are not claimed to match official nnU-Net values. Training
remains disabled.
```

- [ ] **Step 3: Run the complete suite**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests -v`

Expected: all existing model/data/reference tests and the new loss tests pass; no training command is invoked.

## Plan self-review

The plan covers all approved loss interfaces, explicit handling of unconfirmed
weights, valid gradients, target alignment, configuration documentation, and a
full regression run. It intentionally does not create any trainer, optimizer,
augmentation, sampling policy, checkpoint, prediction, or external-data write.

## Execution record

- [x] Foreground batch Soft Dice and input validation implemented with tests.
- [x] Configurable Dice-plus-cross-entropy loss implemented with gradient test.
- [x] Explicit-weight deep-supervision aggregation and nearest target resize implemented with tests.
- [x] Configuration and reproduction notes updated without enabling training.
- [x] Full regression suite passed: 16 tests.
