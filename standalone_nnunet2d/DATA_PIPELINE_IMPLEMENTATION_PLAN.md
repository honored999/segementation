# Read-Only 2D Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bounded, read-only Dataset501 NIfTI data pipeline that produces normalized 2D image/label slices from the fixed supplied folds.

**Architecture:** `nifti_io.py` owns one-file SimpleITK conversion and metadata. `preprocessing.py` owns in-plane resampling and normalization. `dataset.py` composes those operations only for one requested case and `sampling.py` selects a single axial slice. No component writes external data or preloads/caches all cases.

**Tech Stack:** Python 3.14, PyTorch, NumPy, SimpleITK, pytest, supplied JSON references.

---

## File responsibilities

- `data/nifti_io.py`: typed `NiftiVolume`, one-file read/write for tests, and geometry validation.
- `data/preprocessing.py`: in-plane image/segmentation resampling and finite-value Z-score normalization.
- `data/sampling.py`: deterministic/random axial slice selection with explicit bounds checks.
- `data/dataset.py`: fixed-fold case membership, on-demand case loading, and a PyTorch slice dataset.
- `tools/inspect_dataset.py`: path status and optional one-case metadata only.
- `tests/test_nifti_roundtrip.py`, `tests/test_preprocessing.py`, and `tests/test_dataset.py`: synthetic, temporary-file tests only.

### Task 1: NIfTI adapter (completed)

**Files:**
- Modify: `standalone_nnunet2d/data/nifti_io.py`
- Modify: `standalone_nnunet2d/tests/test_nifti_roundtrip.py`

- [ ] **Step 1: Write the failing round-trip test**

```python
def test_nifti_round_trip_preserves_array_and_metadata(tmp_path: Path) -> None:
    source = NiftiVolume(np.arange(24, dtype=np.float32).reshape(2, 3, 4), (0.5, 0.6, 5.0), (1.0, 2.0, 3.0))
    path = tmp_path / "case001_0000.nii.gz"
    write_nifti(path, source)
    restored = read_nifti(path)
    np.testing.assert_array_equal(restored.array, source.array)
    assert restored.spacing_xyz == source.spacing_xyz
    assert restored.origin_xyz == source.origin_xyz
```

- [ ] **Step 2: Run the test and verify it fails because the adapter symbols are absent**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests/test_nifti_roundtrip.py -v`

Expected: import failure for `NiftiVolume`, `read_nifti`, or `write_nifti`.

- [ ] **Step 3: Implement the adapter**

```python
@dataclass(frozen=True)
class NiftiVolume:
    array: np.ndarray
    spacing_xyz: tuple[float, float, float]
    origin_xyz: tuple[float, float, float]

def read_nifti(path: Path) -> NiftiVolume:
    if not path.is_file(): raise FileNotFoundError(f"NIfTI file does not exist: {path}")
    image = sitk.ReadImage(str(path))
    return NiftiVolume(sitk.GetArrayFromImage(image), tuple(image.GetSpacing()), tuple(image.GetOrigin()))

def write_nifti(path: Path, volume: NiftiVolume) -> None:
    image = sitk.GetImageFromArray(volume.array)
    image.SetSpacing(volume.spacing_xyz)
    image.SetOrigin(volume.origin_xyz)
    sitk.WriteImage(image, str(path))
```

- [ ] **Step 4: Run the round-trip test and verify it passes**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests/test_nifti_roundtrip.py -v`

Expected: `1 passed`.

### Task 2: Plan-driven preprocessing (completed)

**Files:**
- Modify: `standalone_nnunet2d/data/preprocessing.py`
- Create: `standalone_nnunet2d/tests/test_preprocessing.py`

- [ ] **Step 1: Write failing normalization and label-discreteness tests**

```python
def test_z_score_normalize_returns_zero_mean_unit_variance() -> None:
    normalized = z_score_normalize(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert normalized.mean() == pytest.approx(0.0, abs=1e-6)
    assert normalized.std() == pytest.approx(1.0, abs=1e-6)

def test_resample_segmentation_keeps_integer_labels() -> None:
    result = resample_inplane(np.array([[[0, 1], [1, 0]]], dtype=np.int16), (1.0, 1.0, 5.0), (0.5, 0.5))
    assert set(np.unique(result.array)).issubset({0, 1})
```

- [ ] **Step 2: Run the tests and verify they fail because preprocessing functions are absent**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests/test_preprocessing.py -v`

Expected: import failure for `z_score_normalize` and `resample_inplane`.

- [ ] **Step 3: Implement minimal preprocessing**

```python
def z_score_normalize(image: np.ndarray) -> np.ndarray:
    if not np.isfinite(image).all(): raise ValueError("image contains non-finite values")
    std = float(image.std())
    if std == 0.0: return np.zeros_like(image, dtype=np.float32)
    return ((image.astype(np.float32) - image.mean()) / std).astype(np.float32)

def resample_inplane(volume: NiftiVolume, target_spacing_xy: tuple[float, float], is_seg: bool) -> NiftiVolume:
    image = to_sitk(volume)
    old_size = image.GetSize(); old_spacing = image.GetSpacing()
    new_size = (round(old_size[0] * old_spacing[0] / target_spacing_xy[0]), round(old_size[1] * old_spacing[1] / target_spacing_xy[1]), old_size[2])
    resampled = sitk.Resample(image, new_size, sitk.Transform(), sitk.sitkNearestNeighbor if is_seg else sitk.sitkBSpline, image.GetOrigin(), (target_spacing_xy[0], target_spacing_xy[1], old_spacing[2]), image.GetDirection(), 0, image.GetPixelID())
    array = sitk.GetArrayFromImage(resampled)
    return NiftiVolume(array.astype(np.int16 if is_seg else np.float32), tuple(resampled.GetSpacing()), tuple(resampled.GetOrigin()))
```

- [ ] **Step 4: Run preprocessing tests and verify they pass**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests/test_preprocessing.py -v`

Expected: `2 passed`.

### Task 3: Fixed-fold, on-demand slice dataset (completed)

**Files:**
- Modify: `standalone_nnunet2d/data/sampling.py`
- Modify: `standalone_nnunet2d/data/dataset.py`
- Create: `standalone_nnunet2d/tests/test_dataset.py`

- [ ] **Step 1: Write failing fixed-fold and slice tests**

```python
def test_fold_dataset_exposes_only_requested_validation_cases(tmp_path: Path) -> None:
    dataset = StrokeSliceDataset(tmp_path, fold=0, split="val", case_ids=("case001",), target_spacing_xy=(0.5, 0.5))
    assert dataset.case_ids == ("case001",)

def test_select_axial_slice_checks_bounds() -> None:
    image = np.zeros((3, 8, 8), dtype=np.float32)
    with pytest.raises(IndexError, match="slice index"):
        select_axial_slice(image, 3)
```

- [ ] **Step 2: Run the tests and verify they fail because the dataset and sampler are absent**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests/test_dataset.py -v`

Expected: import failure for `StrokeSliceDataset` or `select_axial_slice`.

- [ ] **Step 3: Implement the fixed-fold interfaces**

```python
def select_axial_slice(volume: np.ndarray, index: int) -> np.ndarray:
    if not 0 <= index < volume.shape[0]: raise IndexError(f"slice index {index} is outside [0, {volume.shape[0]})")
    return volume[index]

class StrokeSliceDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, raw_root: Path, fold: int, split: Literal["train", "val"], case_ids: tuple[str, ...] | None = None, target_spacing_xy: tuple[float, float] = (0.4892368018627167, 0.4892368018627167)) -> None:
        self.case_ids = case_ids or load_fold_cases(fold, split)
        self.raw_root = validate_raw_root(raw_root)
        self.target_spacing_xy = target_spacing_xy

    def load_case(self, case_id: str) -> tuple[np.ndarray, np.ndarray]:
        image = read_nifti(self.raw_root / "imagesTr" / f"{case_id}_0000.nii.gz")
        label = read_nifti(self.raw_root / "labelsTr" / f"{case_id}.nii.gz")
        if image.array.shape != label.array.shape: raise ValueError(f"geometry mismatch for {case_id}")
        return z_score_normalize(resample_inplane(image, self.target_spacing_xy, False).array), resample_inplane(label, self.target_spacing_xy, True).array
```

- [ ] **Step 4: Run dataset tests and verify they pass**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests/test_dataset.py -v`

Expected: `2 passed`.

### Task 4: Bounded inspection and integration verification (completed)

**Files:**
- Modify: `standalone_nnunet2d/tools/inspect_dataset.py`
- Modify: `standalone_nnunet2d/README.md`
- Modify: `standalone_nnunet2d/REPRODUCTION_NOTES.md`

- [ ] **Step 1: Add an inspect command that reports root/child availability and accepts at most one `--case-id`**

```python
parser.add_argument("--raw-root", type=Path, required=True)
parser.add_argument("--case-id")
if args.case_id:
    image, label = dataset.load_case(args.case_id)
    print({"case_id": args.case_id, "image_shape": image.shape, "label_shape": label.shape})
```

- [ ] **Step 2: Document the no-training guarantee and the server-side command**

```powershell
conda run -n newconda python standalone_nnunet2d/tools/inspect_dataset.py --raw-root C:\...\Dataset501_StrokeLesion --case-id case001
conda run -n newconda python -m pytest standalone_nnunet2d/tests -v
```

- [ ] **Step 3: Run the complete suite**

Run: `conda run -n newconda python -m pytest standalone_nnunet2d/tests -v`

Expected: all model, reference, NIfTI, preprocessing, and dataset tests pass; no external dataset is read by the tests.

## Plan self-review

The tasks cover all confirmed design components: one-file I/O, 2D-plan spacing,
image/label interpolation, Z-score behavior, fixed-fold membership, on-demand
slice selection, bounded inspection, and synthetic tests. The plan deliberately
excludes foreground oversampling, caching, augmentation, loss, training,
prediction, and external-data writes.

## Execution record

- [x] NIfTI round-trip behavior implemented and tested with temporary files.
- [x] Plan-driven in-plane resampling and full-image Z-score implemented and tested.
- [x] Fixed-fold, on-demand deterministic central-slice Dataset implemented and tested.
- [x] Bounded path/one-case inspection implemented and tested.
- [x] Complete regression suite passed: 11 tests, with no external dataset read.
