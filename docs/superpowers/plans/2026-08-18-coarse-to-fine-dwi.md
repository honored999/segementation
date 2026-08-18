# Two-Stage Coarse-to-Fine DWI Lesion Segmentation Implementation Plan

> **For implementers:** Execute this plan in the stated TDD order. The implementation must use scoped file ownership and the repository's main-agent → implementation → tests → independent review → fixer → validation workflow when the plan is executed. This planning task itself does not spawn subagents and does not commit.

**Goal:** Build a self-contained two-stage acute ischemic stroke DWI segmentation pipeline that uses the default 2D nnU-Net's complete five-fold out-of-fold (OOF) Stage 1 predictions to generate prediction-guided Dataset504 ROIs, trains the default 2D nnU-Net on the cropped data, restores Stage 2 predictions into the original full-volume space, and compares Stage 1 and Stage 2 only with original-volume patient metrics.

**Architecture:** `coarse_to_fine_dwi` owns SimpleITK NIfTI I/O, array-space XY crop/restore, prediction-only ROI derivation, Dataset504 generation, and full-volume evaluation. The Dataset504 writer consumes explicit Dataset501 paths, an explicit Stage 1 OOF prediction directory, and the existing fixed split file; it never reads a GT mask to locate an ROI. Stage 2 restoration uses the manifest's reversible crop coordinates and the original image as the metadata reference, so evaluation never scores cropped-space masks.

**Tech Stack:** Python 3, NumPy, SimpleITK, standard library `argparse`/`csv`/`json`/`pathlib`, pytest, and the server's existing nnU-Net v2 environment. The local repository does not contain the previously referenced `standalone_nnunet2d` package, so no implementation step may import or repair it.

---

## Audited context and non-negotiable protocol

The following facts were checked in the current checkout before this plan was written:

- `AGENTS.md` is present and currently has a user-owned uncommitted modification. That change, the untracked `.vscode/` directory, and the untracked `non_teacher_student_files/` directory are outside this plan and must remain untouched.
- `SERVER_PROJECT_STRUCTURE.md` locally verifies Dataset501 as DWI-only with 95 cases and a fixed five-fold patient split. It records historical server paths and nnU-Net conventions, but explicitly marks the actual server raw/preprocessed/results directories, complete checkpoints, and complete OOF NIfTI files as pending server verification.
- `standalone_nnunet2d/reference/dataset.json`, `standalone_nnunet2d/reference/splits_final.json`, and `standalone_nnunet2d/metrics/segmentation_metrics.py` are absent from this checkout. Their absence is not evidence that the server resources are absent. The new component must therefore use direct SimpleITK/NumPy logic and explicit runtime inputs.
- No local real Dataset501 image, label, checkpoint, or OOF NIfTI may be fabricated for this work. Synthetic NIfTI fixtures are permitted only for engineering tests and must be labelled synthetic in test names/comments and never reported as experimental results.

Formal acceptance requires all of the following at runtime:

1. Exactly 95 unique Stage 1 OOF predictions exist, with the exact Dataset501 case-ID set.
2. The fixed split file has exactly five patient-level folds, 95 unique IDs, disjoint train/validation IDs per fold, and each case appears in validation exactly once across folds.
3. Every Dataset504 ROI is derived from the Stage 1 prediction only. GT masks may be copied and cropped for Stage 2 supervision, but no GT foreground coordinate may enter ROI computation.
4. ROI localization uses the union of all Stage 1 foreground voxels in XY across every z slice. There is no largest-component filtering.
5. An empty Stage 1 prediction produces a full-XY ROI and a manifest flag, not an error and not a GT-derived fallback.
6. NIfTI array shapes and spacing/origin/direction metadata are checked before pairing, cropping, restoring, or scoring.
7. Stage 2 predictions are restored to the original full `(z, y, x)` shape and original reference metadata before comparison.
8. The formal metric space is original full-volume patient space only; Dataset504 cropped labels are never used as evaluation GT.

## Target file map

The implementation phase must create exactly these files and no other source/test/documentation files:

| File | Responsibility |
| --- | --- |
| `coarse_to_fine_dwi/__init__.py` | Public package version and explicit exports only. |
| `coarse_to_fine_dwi/nifti.py` | SimpleITK read/write, `(z,y,x)` volume wrapper, strict compatibility checks, XY crop, and full-volume restore. |
| `coarse_to_fine_dwi/roi.py` | Binary prediction validation and prediction-only all-foreground XY ROI computation with margin, minimum size, clipping, and full-XY fallback. |
| `coarse_to_fine_dwi/dataset.py` | Dataset501 discovery, exact OOF/split validation, Dataset504 writer, manifest serialization, and output-boundary checks. |
| `coarse_to_fine_dwi/evaluate.py` | Full-volume binary metrics and Stage 1 versus restored Stage 2 CSV/JSON reporting. |
| `coarse_to_fine_dwi/cli/generate_dataset.py` | Server-facing Dataset504 generation CLI. |
| `coarse_to_fine_dwi/cli/restore_predictions.py` | Server-facing Stage 2 cropped-prediction restoration CLI. |
| `coarse_to_fine_dwi/cli/compare_predictions.py` | Server-facing original-volume Stage 1/Stage 2 comparison CLI. |
| `tests/test_coarse_to_fine_roi.py` | Synthetic NIfTI, binary-mask, crop, restore, and ROI tests. |
| `tests/test_coarse_to_fine_dataset.py` | Synthetic Dataset501/OOF/split and Dataset504 manifest tests. |
| `tests/test_coarse_to_fine_evaluate.py` | Synthetic full-volume metric and CSV/JSON tests. |
| `coarse_to_fine_dwi/README.md` | Verified local limitations, protocol contract, and parameterized server runbook. |

Do not create a replacement `standalone_nnunet2d` package, modify existing training scripts, modify existing tests, add a dependency file, or add real/generated data to the repository.

## API contract to implement before CLI wiring

Use these stable signatures so the tests and later tasks share one interface. Type aliases may be implemented with standard typing syntax; no third-party package beyond NumPy and SimpleITK is needed.

```python
# coarse_to_fine_dwi/nifti.py
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class NiftiVolume:
    array: np.ndarray                 # exactly (z, y, x)
    spacing_xyz: tuple[float, float, float]
    origin_xyz: tuple[float, float, float]
    direction: tuple[float, ...]      # exactly 9 values, row-major

def read_nifti(path: Path) -> NiftiVolume: ...
def write_nifti(volume: NiftiVolume, path: Path) -> None: ...
def assert_compatible(reference: NiftiVolume, candidate: NiftiVolume, *, context: str) -> None: ...
def crop_xy(volume: NiftiVolume, bbox_xy: tuple[int, int, int, int]) -> NiftiVolume: ...
def restore_xy(cropped: NiftiVolume, bbox_xy: tuple[int, int, int, int], reference: NiftiVolume) -> NiftiVolume: ...
```

`read_nifti` must reject a non-3D image, a zero-sized dimension, or a non-finite array. SimpleITK's `GetArrayFromImage` is the sole array conversion and must produce `(z,y,x)`. `write_nifti` must create parent directories only below the caller's already validated output root and must write the stored spacing, origin, and direction without resampling.

`crop_xy` accepts half-open `(x0, y0, x1, y1)` coordinates, validates `0 <= x0 < x1 <= X` and `0 <= y0 < y1 <= Y`, returns `array[:, y0:y1, x0:x1]`, preserves z spacing and direction, preserves x/y spacing, and shifts physical origin by `D @ [x0 * sx, y0 * sy, 0]`, where `D` is the 3×3 direction matrix. `restore_xy` validates that the cropped shape exactly equals `(Z, y1-y0, x1-x0)`, allocates a zero array of the reference shape, inserts the crop at the bbox, and returns the reference spacing/origin/direction exactly. It must reject a cropped volume whose metadata does not equal the expected crop metadata.

```python
# coarse_to_fine_dwi/roi.py
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Roi:
    bbox_xy: tuple[int, int, int, int]
    used_full_xy_fallback: bool
    foreground_voxels: int
    margin_px: int
    min_width_px: int
    min_height_px: int

def validate_binary_prediction(mask: np.ndarray) -> np.ndarray: ...
def compute_prediction_roi(
    prediction: np.ndarray,
    *,
    image_shape_zyx: tuple[int, int, int],
    margin_px: int,
    min_width_px: int,
    min_height_px: int,
) -> Roi: ...
```

`validate_binary_prediction` must require a 3D finite NumPy array and allow only exact values in `{0, 1}` (including boolean and integer arrays). It returns a `uint8` binary copy and rejects NaN, infinity, negative values, non-integral floats, and values such as 2. `compute_prediction_roi` must require prediction shape equal to `image_shape_zyx`, non-negative `margin_px`, positive minimum dimensions, and a positive `(Z,Y,X)` shape. It finds `np.where(prediction > 0)` across all z slices, computes one XY union bbox, expands it by the margin, clips it to the image, then expands the clipped bbox around its center to satisfy minimum width/height while preserving bounds. If there are no foreground voxels, it returns `(0, 0, X, Y)` with `used_full_xy_fallback=True` and `foreground_voxels=0`. It must not label connected components or retain only the largest component.

```python
# coarse_to_fine_dwi/dataset.py
from pathlib import Path

def case_id_from_nifti_name(path: Path) -> str: ...
def validate_fixed_splits(splits_path: Path, case_ids: set[str], *, expected_folds: int = 5) -> dict: ...
def build_dataset504(
    *,
    dataset501_raw: Path,
    stage1_oof_dir: Path,
    splits_path: Path,
    output_root: Path,
    margin_px: int,
    min_width_px: int,
    min_height_px: int,
) -> Path: ...
```

The source dataset must contain `imagesTr`, `labelsTr`, and `dataset.json` under the explicitly supplied root. Image IDs must be obtained from DWI channel files with the exact `_0000.nii.gz` or `_0000.nii` suffix; label IDs must match after removing the extension. Prediction IDs must match the case ID exactly after removing only `.nii.gz` or `.nii`; the discovery function must reject duplicate candidates, missing IDs, and extra IDs. A prediction named with a channel suffix is not silently normalized because that could hide a wrong directory; the server command must first normalize/verify the actual OOF artifact names into the explicitly supplied prediction directory.

`validate_fixed_splits` must parse a JSON list of exactly five objects, each with `train` and `val` lists, coerce no IDs, reject duplicates within a fold, require train/val disjointness, require the union to equal `case_ids`, and require each case to occur in validation exactly once across all folds. The function returns the parsed JSON object only after validation; it must not regenerate or reorder folds.

`build_dataset504` must reject an existing non-empty output directory, validate the source `dataset.json` as one-channel DWI metadata with 95 training cases, validate exactly 95 unique Stage 1 OOF predictions, validate the fixed split file against the same 95 IDs, and then for each case:

1. Read the DWI image and Stage 1 prediction; assert shape and all spatial metadata match.
2. Validate the prediction as binary and compute the ROI from prediction only.
3. Read the GT label only after the ROI is known and assert it matches the image in shape and metadata.
4. Crop image and GT with the same bbox; never inspect GT values for bbox computation.
5. Write cropped image as `imagesTr/{case_id}_0000.nii.gz` and cropped label as `labelsTr/{case_id}.nii.gz` below the new output root.
6. Emit `dataset.json` derived from the source metadata with the Dataset504 name and the actual training count, preserving the source channel/label definitions rather than inventing a schema.
7. Copy the validated fixed split JSON byte-for-byte to `splits_final.json` in the Dataset504 output and write `roi_manifest.json` atomically after all cases succeed.

The manifest must contain one JSON object per case with these exact fields: `case_id`, `source_image`, `source_label`, `stage1_prediction`, `bbox_xy`, `source_shape_zyx`, `cropped_shape_zyx`, `spacing_xyz`, `origin_xyz`, `direction`, `margin_px`, `min_width_px`, `min_height_px`, `foreground_voxels`, and `used_full_xy_fallback`. Paths must be recorded as normalized absolute paths for auditability, while all generated files remain under the caller's output root. The output manifest must record `case_count=95`, the sorted case-ID list, the SHA-256 hash of the copied split file, and a protocol field stating `roi_source: stage1_oof_prediction` and `gt_used_for_roi: false`.

```python
# coarse_to_fine_dwi/evaluate.py
from pathlib import Path
from typing import Any
import numpy as np

def binary_case_metrics(ground_truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]: ...
def compare_full_volume_predictions(
    *,
    labels_dir: Path,
    stage1_dir: Path,
    stage2_restored_dir: Path,
    output_dir: Path,
    expected_case_count: int = 95,
) -> tuple[Path, Path]: ...
```

`binary_case_metrics` must require equal 3D shapes and binary masks. For foreground class 1, calculate TP, FP, FN, TN, Dice, IoU, precision, recall, GT voxel count, and prediction voxel count. Use Dice/IoU/precision/recall equal to 1.0 for both-empty masks and 0.0 when exactly one mask is empty. The evaluator must read original `labelsTr` and compare exact case-ID sets against both prediction directories, require exactly 95 cases by default, check shape and all metadata against the original label/reference image, and reject cropped predictions or duplicate/extra/missing IDs. The CSV must contain one row per case with `case_id`, `gt_voxels`, `stage1_voxels`, `stage2_voxels`, `stage1_tp`, `stage1_fp`, `stage1_fn`, `stage1_dice`, `stage1_iou`, `stage1_precision`, `stage1_recall`, `stage2_tp`, `stage2_fp`, `stage2_fn`, `stage2_dice`, `stage2_iou`, `stage2_precision`, `stage2_recall`, `dice_delta`, `iou_delta`, `precision_delta`, and `recall_delta`. The JSON must contain `protocol.space=original_full_volume`, `protocol.gt_source=Dataset501_labelsTr`, `protocol.case_aggregation=equal_case_macro`, `case_count`, sorted `case_ids`, Stage 1 and Stage 2 case-macro means, pooled global counts/metrics, and per-metric Stage 2-minus-Stage 1 deltas. The evaluator must set `formal_eligible=true` only after all exact-count, ID, shape, metadata, and protocol checks pass; synthetic tests must set/report only engineering evidence and must not be called formal.

## TDD implementation sequence

### Task 1: Add the failing ROI and NIfTI tests

**Files:**

- Create: `tests/test_coarse_to_fine_roi.py`
- Read only: `AGENTS.md`, `SERVER_PROJECT_STRUCTURE.md`
- Must not modify: every source file, all existing tests, all datasets, all output directories, and Git state.

- [ ] **Step 1: Write tests for binary validation and prediction-only ROI union.**

Use `pytest.importorskip("SimpleITK")` at module import, import the future APIs from `coarse_to_fine_dwi.nifti` and `coarse_to_fine_dwi.roi`, and add tests with these concrete expectations:

```python
def test_roi_uses_all_foreground_components_and_all_z_slices():
    prediction = np.zeros((4, 20, 30), dtype=np.uint8)
    prediction[0, 2:4, 3:5] = 1
    prediction[3, 12:15, 20:24] = 1
    roi = compute_prediction_roi(
        prediction,
        image_shape_zyx=prediction.shape,
        margin_px=0,
        min_width_px=1,
        min_height_px=1,
    )
    assert roi.bbox_xy == (3, 2, 24, 15)
    assert roi.foreground_voxels == 18
    assert roi.used_full_xy_fallback is False

def test_empty_prediction_falls_back_to_full_xy():
    prediction = np.zeros((3, 8, 11), dtype=np.uint8)
    roi = compute_prediction_roi(
        prediction,
        image_shape_zyx=prediction.shape,
        margin_px=6,
        min_width_px=7,
        min_height_px=7,
    )
    assert roi.bbox_xy == (0, 0, 11, 8)
    assert roi.used_full_xy_fallback is True
    assert roi.foreground_voxels == 0

@pytest.mark.parametrize("bad", [np.array([[[0, 2]]]), np.array([[[0, np.nan]]]), np.array([[[0, 0.5]]])])
def test_non_binary_prediction_is_rejected(bad):
    with pytest.raises(ValueError, match="binary"):
        validate_binary_prediction(bad)
```

Also test margin clipping and minimum dimensions on a one-voxel lesion, a bbox at each image boundary, a shape mismatch, and negative/zero parameter rejection. Include a test with two spatially separated foreground components whose union bbox must contain both; this is the regression guard against largest-component filtering.

- [ ] **Step 2: Write tests for crop/restore shape and metadata.**

Create a synthetic `NiftiVolume` with shape `(3, 10, 12)`, spacing `(0.7, 0.8, 2.0)`, non-identity direction, and a non-zero origin. Crop `(2, 3, 9, 8)`, assert cropped shape `(3, 5, 7)`, assert origin equals the direction-matrix physical shift, restore it, and assert the original array and all reference metadata are recovered. Add failures for invalid bbox, wrong cropped shape, and wrong crop metadata. Write/read one `.nii.gz` fixture through SimpleITK and assert the same array and metadata.

- [ ] **Step 3: Run the focused tests and record the expected red state.**

Run:

```powershell
conda run -n newconda python -m pytest -q tests/test_coarse_to_fine_roi.py
```

Expected before implementation: collection fails because `coarse_to_fine_dwi` and its imported functions do not exist. Do not treat a wrapper exit code without a visible pytest summary as evidence.

### Task 2: Implement self-contained NIfTI and ROI primitives

**Files:**

- Create: `coarse_to_fine_dwi/__init__.py`
- Create: `coarse_to_fine_dwi/nifti.py`
- Create: `coarse_to_fine_dwi/roi.py`
- Modify: none of the existing files.

- [ ] **Step 1: Implement the smallest package and NIfTI wrapper.**

Implement the exact API above. Use `sitk.ReadImage`, `sitk.GetArrayFromImage`, and `sitk.GetImageFromArray`. Store metadata as immutable tuples. Use `np.asarray(...).copy()` at the package boundary. Compare shape exactly and spacing/origin/direction with `np.array_equal` for direction and `np.allclose(..., rtol=0, atol=1e-6)` for floating-point spacing/origin, with error messages naming the context and field. Do not resample, transpose, reorient, normalize, threshold, or use a GT mask in this module.

- [ ] **Step 2: Implement crop and restore with physical-origin correctness.**

Validate half-open coordinates against the reference array dimensions. In `crop_xy`, calculate the new origin with the 3×3 direction matrix and x/y spacing, keep the z extent unchanged, and return a copy. In `restore_xy`, first derive the expected crop metadata by applying the same origin shift to the reference; reject a mismatch before inserting into a zero-filled full-size array. The output metadata must be copied from the reference so restored predictions are in the original space.

- [ ] **Step 3: Implement strict binary validation and ROI calculation.**

Convert only after checking `ndim == 3`, `np.isfinite`, and `np.isin(mask, [0, 1])`. Derive coordinates with `np.where(mask == 1)`, take min/max over x and y regardless of z, apply inclusive max plus one to form half-open bounds, then apply margin and min-size expansion with deterministic integer arithmetic and boundary clipping. Never call a connected-component routine and never inspect a label/GT argument; the function has no GT parameter.

- [ ] **Step 4: Rerun the focused tests.**

Run the same command from Task 1. Expected: the visible pytest summary reports all tests in `tests/test_coarse_to_fine_roi.py` passed.

### Task 3: Add the failing Dataset504 tests

**Files:**

- Create: `tests/test_coarse_to_fine_dataset.py`
- Read only: the new `coarse_to_fine_dwi/nifti.py` and `coarse_to_fine_dwi/roi.py` APIs.
- Must not modify: existing source/tests, reference metadata, real data, and Git state.

- [ ] **Step 1: Build synthetic Dataset501 fixtures with explicit metadata.**

Define test-local helpers that write a 3D DWI image, GT label, and Stage 1 prediction with SimpleITK. Create a small valid synthetic case set for fast writer tests and a separate 95-case fixture for formal cardinality checks. The 95-case fixture must use deterministic IDs `case_000` through `case_094`, five fixed folds of 19 validation IDs, and no real patient content. Keep all fixture paths under pytest's `tmp_path`.

- [ ] **Step 2: Test exact OOF and fixed-split validation before writing.**

Add tests that the writer rejects one missing prediction, one duplicate prediction candidate, one extra prediction, a split with six folds, overlapping train/val IDs, and a case appearing in validation twice. Add a positive test asserting the validated split is not regenerated or shuffled.

- [ ] **Step 3: Test manifest, crop contents, metadata, and empty fallback.**

Use a non-empty prediction whose two disconnected components occur on different z slices and assert the generated Dataset504 image/label contains the union bbox, not one component. Use an empty Stage 1 prediction for another case and assert the manifest records `used_full_xy_fallback: true`, bbox `(0, 0, X, Y)`, and the cropped shape equal to the full XY shape. Change only the GT foreground location between two otherwise identical fixtures and assert the ROI bbox is unchanged, proving GT cannot localize the crop. Assert the output contains `imagesTr`, `labelsTr`, derived `dataset.json`, byte-identical `splits_final.json`, and `roi_manifest.json`; assert no file is written into the source Dataset501 directory.

- [ ] **Step 4: Run the Dataset504 tests before implementing the writer.**

Run:

```powershell
conda run -n newconda python -m pytest -q tests/test_coarse_to_fine_dataset.py
```

Expected before `dataset.py` exists: visible collection errors identify the missing module/functions. Preserve the red evidence, then implement only the requested file.

### Task 4: Implement Dataset504 discovery, validation, and manifest writing

**Files:**

- Create: `coarse_to_fine_dwi/dataset.py`
- Modify: none of the existing files.

- [ ] **Step 1: Implement deterministic case-ID discovery.**

Implement `case_id_from_nifti_name` for only `.nii.gz` and `.nii`, and require exact DWI `_0000` matching for source images. Enumerate files in sorted order, reject duplicate IDs, reject missing labels, and reject extra predictions. Do not glob a guessed server results tree; all roots are explicit function arguments.

- [ ] **Step 2: Implement fixed-split validation.**

Parse the split JSON without changing list order. Require five folds, each `train` and `val` list to contain strings from the Dataset501 case set, no within-fold duplicates, no train/val overlap, and validation occurrence count exactly one for all 95 cases. Raise `ValueError` with the fold/case causing the failure. Do not generate a replacement split.

- [ ] **Step 3: Implement safe Dataset504 generation.**

Resolve source, prediction, split, and output paths. Reject an output path that resolves inside the Dataset501 source tree and reject an existing non-empty output directory. Read and validate the source `dataset.json` fields needed to establish one DWI channel, 95 training cases, and a single foreground label. For each case, perform the sequence in the API contract and write only derived outputs under the new root. Use a temporary sibling manifest file and replace it only after every case succeeds; never overwrite source files.

- [ ] **Step 4: Implement manifest and metadata audit fields.**

Serialize the exact manifest fields from the API contract, sorted by case ID. Include a split SHA-256 and protocol flags. Store source paths for audit but do not claim that the paths exist outside the current invocation. Preserve the source dataset's label/channel schema; do not invent unverified Dataset501 metadata.

- [ ] **Step 5: Run focused then affected tests.**

Run:

```powershell
conda run -n newconda python -m pytest -q tests/test_coarse_to_fine_roi.py tests/test_coarse_to_fine_dataset.py
```

Expected: a visible summary with zero failures. If a fixture fails because an existing project dependency is missing, report that environment failure and do not weaken production validation.

### Task 5: Add the failing full-volume evaluation tests

**Files:**

- Create: `tests/test_coarse_to_fine_evaluate.py`
- Read only: the new `coarse_to_fine_dwi/nifti.py` API.
- Must not modify: existing metric scripts or any source/test outside the target file list.

- [ ] **Step 1: Test metric semantics.**

Add tests for exact overlap, partial overlap, both-empty masks, GT-only foreground, prediction-only foreground, and shape mismatch. Assert the exact TP/FP/FN/TN and the empty-mask rules. Include a test that passes a crop-shaped prediction against a full-shaped label and expects rejection.

- [ ] **Step 2: Test Stage 1/Stage 2 CSV and JSON output.**

Create two synthetic original volumes with distinct spacing/origin/direction, labels, Stage 1 predictions, and restored Stage 2 predictions. Run `compare_full_volume_predictions`, assert one CSV row per case, exact column names, Stage 2-minus-Stage 1 deltas, equal case-macro mean, pooled global metrics, sorted IDs, `original_full_volume` protocol, and `formal_eligible` only for the exact expected count. Add missing/extra/duplicate ID and metadata mismatch failures.

- [ ] **Step 3: Run the focused tests before implementing evaluation.**

Run:

```powershell
conda run -n newconda python -m pytest -q tests/test_coarse_to_fine_evaluate.py
```

Expected before `evaluate.py` exists: visible collection errors for the missing module/functions.

### Task 6: Implement full-volume metrics and comparison artifacts

**Files:**

- Create: `coarse_to_fine_dwi/evaluate.py`
- Modify: none of the existing files.

- [ ] **Step 1: Implement binary metrics locally.**

Use boolean NumPy operations on full 3D arrays. Keep per-case metrics separate from pooled totals. Implement the exact empty-mask conventions in the API contract and return ordinary JSON-serializable numbers. Do not import the absent `standalone_nnunet2d.metrics` package and do not silently choose a different metric field such as slice Dice or online validation Dice.

- [ ] **Step 2: Implement exact-ID full-volume comparison.**

Discover labels and predictions by exact case ID, require the three sets to match and have 95 IDs by default, read the original label/reference image metadata, and assert every prediction matches before computing any metric. Write CSV and JSON only under a validated output directory. Name artifacts `stage1_vs_stage2_case_metrics.csv` and `stage1_vs_stage2_summary.json`.

- [ ] **Step 3: Rerun evaluation and affected tests.**

Run:

```powershell
conda run -n newconda python -m pytest -q tests/test_coarse_to_fine_roi.py tests/test_coarse_to_fine_dataset.py tests/test_coarse_to_fine_evaluate.py
```

Expected: visible summary with zero failures and no tests using real server data.

### Task 7: Add CLI entry points and CLI-level validation

**Files:**

- Create: `coarse_to_fine_dwi/cli/generate_dataset.py`
- Create: `coarse_to_fine_dwi/cli/restore_predictions.py`
- Create: `coarse_to_fine_dwi/cli/compare_predictions.py`
- Create: `coarse_to_fine_dwi/cli/__init__.py` only if Python package discovery requires it; otherwise do not create an extra file.
- Modify: `tests/test_coarse_to_fine_dataset.py` and `tests/test_coarse_to_fine_evaluate.py` only if adding CLI subprocess assertions is necessary; keep those changes within the already authorized test files.

- [ ] **Step 1: Implement `generate_dataset.py`.**

Expose these required flags: `--dataset501-raw`, `--stage1-oof-dir`, `--splits`, `--output-root`, `--margin-px`, `--min-width-px`, and `--min-height-px`. Call `build_dataset504`, print the output root, manifest path, `case_count`, and fallback count, and return a non-zero exit code for validation errors. Do not accept a GT/label path for ROI generation.

- [ ] **Step 2: Implement `restore_predictions.py`.**

Expose `--manifest`, `--cropped-predictions`, `--dataset501-raw`, and `--output-dir`. Load every manifest row, require one exact cropped Stage 2 prediction per case, read the original DWI image as the restore reference, validate cropped prediction shape/metadata against the recorded bbox, restore to full space, validate the restored shape/metadata, and write `{case_id}.nii.gz` under the output root. Reject missing/extra/duplicate predictions and reject a manifest whose protocol says anything other than `stage1_oof_prediction`.

- [ ] **Step 3: Implement `compare_predictions.py`.**

Expose `--dataset501-raw`, `--stage1-oof-dir`, `--stage2-restored-dir`, and `--output-dir`, with `--expected-case-count` defaulting to 95. Call `compare_full_volume_predictions` using `labelsTr` from Dataset501. Print the two artifact paths and the validated case count; do not accept Dataset504 labels as a GT argument.

- [ ] **Step 4: Test CLI help and an end-to-end synthetic invocation.**

Add subprocess checks to the existing coarse-to-fine test files using `conda run -n newconda python -m ... --help` only if the package import path is stable in the checkout. Otherwise test the `main(argv)` functions directly. The end-to-end synthetic flow must be `build_dataset504` → synthetic cropped Stage 2 prediction → `restore_predictions` logic → `compare_full_volume_predictions`, and must assert that the final evaluator sees full original shapes.

### Task 8: Write the component README and server runbook

**Files:**

- Create: `coarse_to_fine_dwi/README.md`
- Modify: none of the repository-level documentation files.

- [ ] **Step 1: Document verified versus pending facts.**

State that local references verify Dataset501 DWI-only, 95 cases, and fixed five-fold splits; state that the actual server raw/preprocessed/results paths, complete default-Trainer checkpoints, retained validation NIfTI files, and complete OOF set remain pending until checked on the server. State explicitly that no local numerical Stage 1 or Stage 2 result is produced by this implementation task.

- [ ] **Step 2: Document the formal data-lineage contract.**

Document the exact chain `Stage 1 default DWI 2D nnU-Net OOF prediction → binary validation → all-foreground XY union ROI → Dataset504 crop → Stage 2 default 2D nnU-Net → cropped prediction → manifest-based restore → Dataset501 original full-volume GT evaluation`. Include the no-GT-localization rule, no-largest-component rule, empty-prediction full-XY fallback, fixed split requirement, exact 95-case requirement, and full-volume case-macro metric semantics.

- [ ] **Step 3: Document local synthetic validation commands.**

Include these exact commands and require visible pytest summaries:

```powershell
conda run -n newconda python -m pytest -q tests/test_coarse_to_fine_roi.py
conda run -n newconda python -m pytest -q tests/test_coarse_to_fine_dataset.py
conda run -n newconda python -m pytest -q tests/test_coarse_to_fine_evaluate.py
conda run -n newconda python -m pytest -q tests/test_coarse_to_fine_roi.py tests/test_coarse_to_fine_dataset.py tests/test_coarse_to_fine_evaluate.py
conda run -n newconda python -m pytest -q
```

The README must say that synthetic tests validate engineering behavior only and are not formal patient results.

- [ ] **Step 4: Add the parameterized Windows server runbook.**

Use environment variables rather than hard-coded historical paths. The operator must set values after server inspection:

```powershell
$env:NNUNET_ENV = "nnunet5090"
$env:STAGE1_DATASET = "Dataset501_StrokeLesion"
$env:STAGE1_TRAINER = "nnUNetTrainer"
$env:STAGE1_PLANS = "nnUNetPlans"
$env:STAGE1_OOF_DIR = "D:\derived\stroke\stage1_oof_default_5fold"
$env:DATASET504_NAME = "Dataset504_StrokeLesion_CoarseToFine"
$env:DATASET504_RAW = Join-Path $env:nnUNet_raw $env:DATASET504_NAME
$env:DATASET504_PREPROCESSED = Join-Path $env:nnUNet_preprocessed $env:DATASET504_NAME
$env:STAGE2_CROPPED_OOF_DIR = "D:\derived\stroke\stage2_cropped_oof_default_5fold"
$env:STAGE2_RESTORED_DIR = "D:\derived\stroke\stage2_restored_oof_default_5fold"
$env:CTF_EVAL_DIR = "D:\derived\stroke\coarse_to_fine_eval"
$Dataset501 = Join-Path $env:nnUNet_raw $env:STAGE1_DATASET
$Splits501 = Join-Path (Join-Path $env:nnUNet_preprocessed $env:STAGE1_DATASET) "splits_final.json"
```

The README must require these read-only checks before any generation or training:

```powershell
Get-ChildItem Env:nnUNet_raw,Env:nnUNet_preprocessed,Env:nnUNet_results,Env:NNUNET_ENV,Env:STAGE1_OOF_DIR
Test-Path (Join-Path $Dataset501 "imagesTr")
Test-Path (Join-Path $Dataset501 "labelsTr")
Test-Path (Join-Path $Dataset501 "dataset.json")
Test-Path $Splits501
Get-ChildItem -LiteralPath $env:nnUNet_results -Recurse -Filter "*.pth" | Select-Object FullName
Get-ChildItem -LiteralPath $env:STAGE1_OOF_DIR -Filter "*.nii*" | Measure-Object
```

If any required check is false, stop and report the missing server resource; do not substitute in-sample predictions or a fold-0 result.

After the checks pass, run the local component from the repository checkout on the server:

```powershell
conda run -n $env:NNUNET_ENV python -m coarse_to_fine_dwi.cli.generate_dataset `
  --dataset501-raw $Dataset501 `
  --stage1-oof-dir $env:STAGE1_OOF_DIR `
  --splits $Splits501 `
  --output-root $env:DATASET504_RAW `
  --margin-px 16 `
  --min-width-px 128 `
  --min-height-px 128
```

The README must explain that the command is valid only when the Stage 1 directory contains exactly the 95 held-out predictions selected by the fixed five-fold split. A directory containing predictions from one model applied to all 95 cases is not OOF and must be rejected or withheld from this command.

Then preprocess and train Stage 2 with the default 2D configuration, preserving the fixed split. First run:

```powershell
conda run -n $env:NNUNET_ENV nnUNetv2_plan_and_preprocess -d $env:DATASET504_NAME --verify_dataset_integrity
```

Before training, verify that the preprocessed Dataset504 split is byte-identical to the validated `splits_final.json` emitted by the writer. If nnU-Net generated a separate copy, compare hashes and stop on mismatch; do not replace a split silently. Train the five fixed folds only after the Stage 2 default-Trainer checkpoint configuration has been confirmed:

```powershell
0..4 | ForEach-Object {
  conda run -n $env:NNUNET_ENV nnUNetv2_train $env:DATASET504_NAME 2d $env:STAGE1_TRAINER $_ --npz
}
```

The README must state that the cropped Stage 2 OOF directory must be assembled from each fold's held-out validation predictions only, one prediction per case, with case IDs matching the Dataset504 manifest. It must not be assembled from a single fold's predictions on all cases. Because the server document does not locally verify checkpoint folder names or retained validation filenames, the runbook must first list and inspect the actual server artifacts rather than assuming a path.

Restore and evaluate with:

```powershell
conda run -n $env:NNUNET_ENV python -m coarse_to_fine_dwi.cli.restore_predictions `
  --manifest (Join-Path $env:DATASET504_RAW "roi_manifest.json") `
  --cropped-predictions $env:STAGE2_CROPPED_OOF_DIR `
  --dataset501-raw $Dataset501 `
  --output-dir $env:STAGE2_RESTORED_DIR

conda run -n $env:NNUNET_ENV python -m coarse_to_fine_dwi.cli.compare_predictions `
  --dataset501-raw $Dataset501 `
  --stage1-oof-dir $env:STAGE1_OOF_DIR `
  --stage2-restored-dir $env:STAGE2_RESTORED_DIR `
  --output-dir $env:CTF_EVAL_DIR `
  --expected-case-count 95
```

Only a JSON summary that reports `case_count: 95`, exact IDs, `original_full_volume`, and `formal_eligible: true` may be considered a completed formal comparison. A fold-0 screening run, a smoke/preflight run, an in-sample Stage 1 prediction set, a missing validation case, or a synthetic fixture is engineering evidence and must not be reported as the two-stage result.

### Task 9: Independent review, fixer gate, and final validation

**Files:**

- Read only for review: all files listed in the target file map, `AGENTS.md`, and `SERVER_PROJECT_STRUCTURE.md`.
- Modify only if a concrete blocker is found: the smallest affected file from the target file map; do not modify existing project files.

- [ ] **Step 1: Review protocol and leakage boundaries.**

Check that no function accepts GT coordinates for ROI derivation, no largest-component filter exists, the empty Stage 1 fallback is full XY, all 95 IDs and five folds are exact, and no CLI can accidentally evaluate cropped labels as original GT.

- [ ] **Step 2: Review metadata and output safety.**

Check shape/spacing/origin/direction validation before every pairing and restore, physical-origin correction for cropped images, reference metadata on restore, normalized output containment, no source overwrite, and atomic manifest finalization.

- [ ] **Step 3: Run all tests and static checks.**

Run:

```powershell
conda run -n newconda python -m pytest -q tests/test_coarse_to_fine_roi.py tests/test_coarse_to_fine_dataset.py tests/test_coarse_to_fine_evaluate.py
conda run -n newconda python -m pytest -q
git diff --check -- docs/superpowers/plans/2026-08-18-coarse-to-fine-dwi.md
```

The implementation phase must require a visible pytest summary and exit code 0 before claiming completion. If a review blocker is found, apply only the minimum fix, rerun the focused test, the three-file affected suite, then the full suite, and repeat the independent review.

- [ ] **Step 4: Confirm repository safety.**

The implementation phase must inspect `git status --short` and `git diff --stat` and verify that only the target file map changed. It must not commit, push, reset, restore, stash, clean, or alter the user's pre-existing `AGENTS.md`, `.vscode/`, `non_teacher_student_files/`, data, results, or worktree registrations.

## Self-review checklist for this plan

- [x] Defines every requested source, CLI, test, and README path with a single responsibility.
- [x] Uses test-first order: failing tests → focused implementation → affected tests → CLI integration → independent review → fixer if needed → full validation.
- [x] Specifies exact 95-case/5-fold/OOF checks and does not substitute in-sample predictions.
- [x] Specifies prediction-only ROI union, no largest-component filtering, margin/minimum-size rules, and empty full-XY fallback.
- [x] Specifies strict shape and NIfTI metadata checks, reversible crop/restore, and original full-volume evaluation.
- [x] Distinguishes locally verified Dataset501/split facts from pending server paths/checkpoints/OOF artifacts.
- [x] Contains concrete local `conda run -n newconda python -m pytest` commands and parameterized server commands using nnU-Net environment variables.
- [x] Does not rely on the absent `standalone_nnunet2d` package or invent its files.
- [x] Does not require a commit for this planning task.
