# Configurable Augmentation and Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic foreground-aware slice selection and opt-in synchronized 2D augmentation without enabling training.

**Architecture:** `sampling.py` owns RNG-driven slice index decisions. `augmentation.py` owns immutable configuration and image/label paired transforms. `dataset.py` will accept these interfaces but retain on-demand one-case loading.

**Tech Stack:** Python 3.14, NumPy, PyTorch, pytest; no external data access in tests.

---

### Task 1: Foreground-aware sampling

**Files:**
- Modify: `standalone_nnunet2d/data/sampling.py`
- Modify: `standalone_nnunet2d/tests/test_dataset.py`

- [ ] **Step 1: Write failing foreground/fallback tests**

```python
def test_foreground_sampler_chooses_foreground_slice_when_probability_is_one() -> None:
    labels = np.zeros((3, 2, 2), dtype=np.int16); labels[2, 0, 0] = 1
    assert select_slice_index(labels, np.random.default_rng(7), foreground_probability=1.0) == 2

def test_foreground_sampler_falls_back_to_valid_uniform_index_without_foreground() -> None:
    assert 0 <= select_slice_index(np.zeros((3, 2, 2), dtype=np.int16), np.random.default_rng(7), foreground_probability=1.0) < 3
```

- [ ] **Step 2: Run tests and verify failure because `select_slice_index` is absent**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests/test_dataset.py -v`

Expected: import failure for `select_slice_index`.

- [ ] **Step 3: Implement validated, caller-RNG sampling**

```python
def select_slice_index(labels: np.ndarray, rng: np.random.Generator, *, foreground_probability: float = 0.0) -> int:
    if labels.ndim != 3: raise ValueError("labels must be (z, y, x)")
    if not 0.0 <= foreground_probability <= 1.0: raise ValueError("foreground_probability must be in [0, 1]")
    foreground = np.flatnonzero(np.any(labels != 0, axis=(1, 2)))
    candidates = foreground if foreground.size and rng.random() < foreground_probability else np.arange(labels.shape[0])
    return int(rng.choice(candidates))
```

- [ ] **Step 4: Run sampling tests and verify they pass**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests/test_dataset.py -v`

Expected: foreground and fallback tests pass.

### Task 2: Opt-in paired augmentation

**Files:**
- Modify: `standalone_nnunet2d/data/augmentation.py`
- Create: `standalone_nnunet2d/tests/test_augmentation.py`

- [ ] **Step 1: Write failing identity and synchronized-flip tests**

```python
def test_default_augmentation_is_identity() -> None:
    image, label = np.arange(4, dtype=np.float32).reshape(2, 2), np.array([[0, 1], [1, 0]], dtype=np.int16)
    augmented_image, augmented_label = augment_slice(image, label, np.random.default_rng(1), AugmentationConfig())
    np.testing.assert_array_equal(augmented_image, image); np.testing.assert_array_equal(augmented_label, label)

def test_horizontal_flip_is_synchronized_for_image_and_label() -> None:
    image, label = np.array([[1, 2], [3, 4]], dtype=np.float32), np.array([[0, 1], [1, 0]], dtype=np.int16)
    config = AugmentationConfig(horizontal_flip_probability=1.0)
    result_image, result_label = augment_slice(image, label, np.random.default_rng(1), config)
    np.testing.assert_array_equal(result_image, image[:, ::-1]); np.testing.assert_array_equal(result_label, label[:, ::-1])
```

- [ ] **Step 2: Run tests and verify failure due to absent augmentation API**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests/test_augmentation.py -v`

Expected: import failure for `AugmentationConfig` or `augment_slice`.

- [ ] **Step 3: Implement immutable config and paired transforms**

```python
@dataclass(frozen=True)
class AugmentationConfig:
    horizontal_flip_probability: float = 0.0
    vertical_flip_probability: float = 0.0
    intensity_scale_range: tuple[float, float] = (1.0, 1.0)

def augment_slice(image: np.ndarray, label: np.ndarray, rng: np.random.Generator, config: AugmentationConfig) -> tuple[np.ndarray, np.ndarray]:
    if image.shape != label.shape or image.ndim != 2: raise ValueError("image and label must be matched 2D arrays")
    result_image, result_label = image.copy(), label.copy()
    if rng.random() < config.horizontal_flip_probability: result_image, result_label = result_image[:, ::-1], result_label[:, ::-1]
    if rng.random() < config.vertical_flip_probability: result_image, result_label = result_image[::-1, :], result_label[::-1, :]
    low, high = config.intensity_scale_range
    return (result_image * rng.uniform(low, high)).astype(image.dtype, copy=False), result_label
```

- [ ] **Step 4: Run augmentation tests and verify they pass**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests/test_augmentation.py -v`

Expected: identity and synchronized-flip tests pass.

### Task 3: Dataset integration and documentation

**Files:**
- Modify: `standalone_nnunet2d/data/dataset.py`
- Modify: `standalone_nnunet2d/configs/default.yaml`
- Modify: `standalone_nnunet2d/README.md`
- Modify: `standalone_nnunet2d/REPRODUCTION_NOTES.md`

- [ ] **Step 1: Add optional `rng`, `foreground_probability`, and `augmentation_config` constructor parameters**

```python
slice_index = select_slice_index(label, self.rng, foreground_probability=self.foreground_probability)
image_slice, label_slice = augment_slice(select_axial_slice(image, slice_index), select_axial_slice(label, slice_index), self.rng, self.augmentation_config)
```

- [ ] **Step 2: Document disabled-by-default local choices**

```yaml
sampling:
  foreground_probability: 0.0
augmentation:
  horizontal_flip_probability: 0.0
  vertical_flip_probability: 0.0
  intensity_scale_range: [1.0, 1.0]
```

- [ ] **Step 3: Run the full suite**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests -v`

Expected: all prior tests and new sampling/augmentation tests pass without external data access or training.

## Plan self-review

The plan covers deterministic RNG use, foreground fallback, probability
validation, synchronized label-safe geometry, disabled defaults, dataset
integration, configuration, and full regression. It excludes unverified
official parameter schedules, NIfTI bulk reads, cache generation, and training.
