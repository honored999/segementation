# H2Former prepared-input source-space fold evaluation

## Scope and baseline

Implementation base: `609165d4994aa160561f985f506f6303956e7ec5`.
Local worktree: `E:\study\研一\work14-图像分割\segementation\.worktrees\h2former-source-space-evaluation`.
Branch: `codex/h2former-source-space-evaluation`. No commit, push, merge, retraining,
real patient data access or real checkpoint loading occurred during implementation.
The existing implementation worktree remains untouched.

Files: `standalone_nnunet2d/predict.py` (explicit prepared CLI dispatch),
`standalone_nnunet2d/source_space_evaluation.py` (new evaluation entry),
`standalone_nnunet2d/tests/test_source_space_evaluation.py` (tiny synthetic tests),
and this document. `engine/predictor.py`, model, trainer, selection, fixed splits,
raw inputs and canonical project memory are unchanged. The main agent must arrange
independent Level 3 review against the final uncommitted snapshot before acceptance.
Implementer self-inspection is not an independent review.

## Preprocessing and export contract

Inspected local runtime: newconda, nnunetv2 2.8.1, blosc2 4.12.0, torch 2.11.0+cu126,
SimpleITK 2.5.5, pytest 9.0.3. No dependencies installed/upgraded.
Primary contract is installed official nnunetv2 code:

- `preprocessing/preprocessors/default_preprocessor.py:DefaultPreprocessor.run_case_npy`:
  transpose image/spacing, nonzero image crop, normalization **before** resampling,
  and configured resampling. Records `shape_before_cropping`,
  `shape_after_cropping_and_before_resampling`, `bbox_used_for_cropping`, original
  array-order `spacing` and reader geometry. It does not record a dedicated
  shape-after-resampling field: the b2nd header supplies that actual shape.
- `training/dataloading/nnunet_dataset.py:nnUNetDatasetBlosc2`: official siblings
  `CASE.b2nd`, `CASE_seg.b2nd`, `CASE.pkl`. The latter is a properties dictionary.
- `inference/export_prediction.py:convert_predicted_logits_to_segmentation_with_correct_shape`:
  resample **class logits** to cropped pre-resampling shape using the configured
  probabilities resampling function and transposed spacing; compute softmax and
  classes; background-fill crop; inverse transpose. Probability restoration also
  fills the outside-crop background probability with 1.
- `imageio/simpleitk_reader_writer.py:SimpleITKIO`: array order is Z,Y,X, geometry
  is stored in `sitk_stuff`; array spacing is the reverse of xyz spacing.

Training reads the existing b2nd image channel/z slices. Prediction reads these
same stored values and performs no second normalization, raw preprocessing or
GT-derived ROI. Reuses existing `predict_logits_2d`: 512x512 sliding tiles, centered
zero padding, step 0.5, fp16 Gaussian importance/accumulation, default in-plane TTA
on axes 0,1, unflip logits then mean. `--disable-tta` is explicit and recorded;
`--tile-step-size` and positive `--slice-batch-size` are recorded. Final classes are
formed by official export after logits resampling, not by resizing a final mask.
Prepared logits are kept until export; restored float probabilities are saved in
`probabilities/CASE.npz` in original Z,Y,X order with class dimension first.

Minimal supported reader contract is **SimpleITKIO + 3D .nii.gz**. Other readers,
reorientation, invalid/missing properties or contradictory shape/spacing/bbox
are refused. Transpose permutations must be explicit inverses. Prepared shape
must equal official `compute_new_shape` for the crop and target plans spacing;
2D keeps the transposed slice spacing unchanged. The prepared directory basename
must match plans `data_identifier`. Original NIfTI must independently match the
inverse-transposed full shape and properties spacing/origin/direction. Export uses
these verified original image attributes with the existing write/readback check:
exact shape, uint8 binary mask, spacing/origin/direction at rtol=0, atol=1e-6.
Missing values are never inferred. Properties pickle accepts only native data and
needed NumPy globals, rejects other executable global classes.

The image/seg b2nd headers must pair in shape and dtype; **seg voxels and
class_locations never contribute to inference**. GT NIfTI is read only by the
post-export check/metrics stage. Existing `select_fold` still scores prepared
volumes without TTA and still controls checkpoint selection; its Dice is not the
source-space Dice. This implementation does not claim inference parity with an
nnU-Net network or change the training selection contract.

## Identity, completeness and provenance

Only `checkpoint_best.pth` from Dataset501 **finetune** is accepted. In addition
to the filename, selection metadata must contain native integer epochs with
`epoch == best_selection_epoch > 0` and a native numeric (not boolean), finite
`best_selection_dice` in [0,1]. The manifest records these values and the current
training saver basis: best is saved on strict prepared-space selection Dice
improvement. Missing or contradictory selection metadata is refused. This rejects
ordinary renamed latest contents with a different epoch; it does not authenticate
weights against deliberate forgery of all metadata or distinguish a latest saved
at the same best-selection epoch. Training and checkpoint selection are unchanged.
The checkpoint must also have strict
base h2former/single_output, matching metadata/config fold, target source schema,
prepared source type, external initialization provenance and supplied plans,
dataset.json, patient-map (when present), whole fixed split and selected cohort
hashes. An available sibling `resolved_config.json` must equal checkpoint config.
The original external checkpoint is not opened. Source roots may relocate; their
canonical content identities may not change. Full model keys/shapes/dtypes/finite
values are checked before strict model loading. Latest, pretrain, legacy, mismatched
fold/plans/source and missing provenance are refused before model construction.

The server must explicitly supply `splits_final.json`; every train/val membership
across its five folds must equal the repository fixed reference (order-independent
identity). Cases are the whole selected validation fold, no subsets or case-ID
selection. Validation folds must be disjoint and union to the cohort; no split
creation. All image, b2nd pair and properties paths must exist even in metadata mode.
Any per-case error raises nonzero and leaves `status=failed` with case/error and any
partial outputs; that directory cannot be treated as complete or reused.

New outputs must not exist and must not overlap the resolved actual source worktree,
raw/prepared/plans directories,
checkpoint directory and recorded external checkpoint/source. Existing prediction
or output directories cannot be reused.
Metrics output obeys the prediction manifest's protected roots too. Never retry by
reusing a partial directory: choose a genuinely new run suffix after investigating.
No automatic cleanup or overwrite. Manifest records Git HEAD/branch/status, binary
tracked diff SHA256 and implementation hashes, checkpoint SHA256/identity/source/
initialization, raw-file and canonical plans/split hashes, cohort, properties and
prepared/source/prediction/probability SHA256, environment, UTC timestamps, exact
inference policy and per-case geometry. JSON and properties are parsed from
captured bytes whose SHA256 is checked against
bracketing file hashes; path-based split validation and NIfTI reads are bracketed
by hashes too. Recorded metadata hashes come from those initial stable reads,
not a later independent hash. All dataset/plans/split/checkpoint and every case's
properties/prepared/source hashes are rechecked before `complete`; a detected
change fails explicitly, leaving a failed manifest if export already started.
No latest hash is substituted to accept mixed input versions.
Post-export check demands the exact manifest cohort/files, unchanged original
image/prediction hashes, binary dtype, and original/GT geometry. It revalidates the
server fixed split; incomplete or extra/duplicate predictions are refused.

`--metadata-only` reads restricted checkpoint payload **on CPU** (including stored
tensors; not a zero-memory parser), config, plans, dataset.json and explicit split,
checks case file existence, prints JSON and creates no model or output directory.
It does **not** read properties contents or medical arrays, validate real geometry,
run GPU kernels or prove preprocessing parity. Budget CPU RAM for the checkpoint.
`official_alignment_pending` requires `--allow-pending`; manifests and metrics
retain that marker and `pending_engineering_fold_evaluation`. Official exporter use
alone does not establish `official_aligned` or formal five-fold clinical evidence.

## Metrics compatibility

The metrics command imports the exact existing `evaluate_segmentation_metrics.py`
`evaluate_case` and `build_summary`. It adds completeness/geometry/provenance guards
without altering any formulas. Metric inputs use one independent temporary
prediction/GT byte snapshot pair at a time (same case filenames). Snapshot hashes
must match the already checked originals. Binary/uint8 and strict geometry checks
run on those snapshot contents, and the unchanged metric implementation reads the
snapshots. Snapshot and original hashes are rechecked after calculation; manifest,
fixed split, exact complete prediction file set and all original prediction/source/
GT contents are revalidated before report writing. Any observed mismatch raises
before a successful report is created. Temporary disk usage is bounded by one
compressed prediction/GT pair, outside protected inputs/outputs; set TEMP/TMP to
an independent scratch disk and preflight space/RAM on the server. Only temporary
copies are cleaned up, never real sources or prediction outputs. GT is first
accessed after prediction export and never affects crop or inference.

These checks detect ordinary concurrent replacement and persistent changes at the
checked boundaries. They do not lock inputs, prevent adversarial change-and-restore
between observations (ABA), or prevent edits after the final check/after report
creation. The report hashes identify the checked snapshot version; they are not a
claim that source paths cannot subsequently change. They also do not establish
real-server compatibility or full-fold scientific validation.

Reports case_metrics.csv, summary_metrics.csv and
summary_metrics.json, including alignment state, fold, manifest SHA256, geometry
and GT hashes, metric-script SHA256. Summary CSV uses the existing internal units
(fractions for Dice/IoU/F2/Recall); multiply by 100 for percentage presentation.
Case records also retain TP/FP/FN, lesion/voxel counts and volumes in mL.

- Dice, IoU, Recall and F2: higher is better; macro over patients.
- AVD: absolute volume difference / GT volume *100 (%), lower is better.
- LCD: absolute component-count difference, full 26-connectivity, lower is better.
- HD95: pooled bidirectional surface-distance 95th percentile (mm), lower is better.
- `--f2-mode paper` (existing default/comparable printed SrSNet equation):
  `5TP/(5TP+4FP+FN)`.
- `--f2-mode standard`: conventional recall-weighted beta=2,
  `5TP/(5TP+FP+4FN)`. Do not combine the two modes as one metric.

Existing NaN policy is preserved: undefined ratios and HD95 for exactly one empty
mask remain NaN; macro excludes NaN and reports valid_cases per metric. Both-empty
HD95 is 0; both-empty Dice is undefined. No silent case exclusion: every case has
a row, with the denominator count explicit per metric. Existing script tolerates
geometry at 1e-5; this wrapper first enforces the stricter 1e-6/rtol=0 project rule.

## Server preparation (manual, after independent review)

Known training worktree:
`C:\lijialin\segementation\.worktrees\h2former-external-pretrain-es`.
Known finetune files:
`C:\lijialin\experiments\h2former-isles-finetune-bs4-run01\fold_0\checkpoint_best.pth`,
`checkpoint_latest.pth`, `resolved_config.json`, `training_log.csv`.
These paths are supplied by the user, **not locally or remotely verified**.

Use a separate server worktree from the exact baseline; no pull of a nonexistent
published evaluation branch. Run these Windows **CMD** commands only after checking
the destination/branch do not already exist and the baseline commit is available:

```bat
git -C "C:\lijialin\segementation" worktree list --porcelain
git -C "C:\lijialin\segementation" show -s --oneline 609165d4994aa160561f985f506f6303956e7ec5
git -C "C:\lijialin\segementation" worktree add -b codex/h2former-source-space-evaluation "C:\lijialin\segementation\.worktrees\h2former-source-space-evaluation" 609165d4994aa160561f985f506f6303956e7ec5
```

Manually transfer **only the four reviewed files** with original relative paths
into that new worktree and compare final HANDOFF SHA256 (e.g. `certutil -hashfile
"FULL_FILE_PATH" SHA256`). No evaluation implementation is committed/published;
a Git update of the old branch does not transfer this uncommitted snapshot.
If a branch/path already exists, stop and inspect; do not force/reset/recreate it.

**Unconfirmed raw path:** replace every occurrence of
`C:\CONFIRM_DATASET501_RAW_ROOT` below with the independently confirmed original
Dataset501 root containing imagesTr and labelsTr. It is deliberately a placeholder,
not a claimed actual path. Confirm paired originals correspond to the prepared
Dataset501 cohort. Verify actual server nnunetv2 is 2.8.1; do not install/upgrade
without separate authorization. Outputs below must be brand-new independent paths.

Before each server validation/prediction stage check resources; at/above 80% or
unsafe estimated peak defer/reduce the operation. Batch starts at 1, not training
batch4. Export holds full prepared two-class logits and restored probabilities on
CPU: substantial RAM headroom is necessary. CPU batch and threads affect resources,
not metric definitions. No server execution is claimed here.

```bat
call conda activate nnunet5090
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
powershell -NoProfile -Command "$c=Get-CimInstance Win32_Processor; $o=Get-CimInstance Win32_OperatingSystem; $c | Select-Object LoadPercentage; Write-Output ('RAM percent: ' + (100*(1-$o.FreePhysicalMemory/$o.TotalVisibleMemorySize)))"
python -c "import importlib.metadata as m; print({n:m.version(n) for n in ('nnunetv2','torch','blosc2','SimpleITK')})"
```

### 1. Metadata check (no predictions, no geometry/GPU proof)

```bat
call conda activate nnunet5090
cd /d "C:\lijialin\segementation\.worktrees\h2former-source-space-evaluation"
set "CUDA_VISIBLE_DEVICES=0"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
python -m standalone_nnunet2d.predict --checkpoint "C:\lijialin\experiments\h2former-isles-finetune-bs4-run01\fold_0\checkpoint_best.pth" --preprocessed-root "C:\lijialin\models3d\nnUNet\nnUNet_preprocessed\Dataset501_StrokeLesion\nnUNetPlans_2d" --plans "C:\lijialin\models3d\nnUNet\nnUNet_preprocessed\Dataset501_StrokeLesion\nnUNetPlans.json" --splits-file "C:\lijialin\models3d\nnUNet\nnUNet_preprocessed\Dataset501_StrokeLesion\splits_final.json" --raw-root "C:\CONFIRM_DATASET501_RAW_ROOT" --fold 0 --output-root "C:\lijialin\experiments\h2former-source-space-eval-fold0-run01" --device cpu --slice-batch-size 1 --allow-pending --metadata-only
```

### 2. Predict complete fold using prepared inputs

```bat
call conda activate nnunet5090
cd /d "C:\lijialin\segementation\.worktrees\h2former-source-space-evaluation"
set "CUDA_VISIBLE_DEVICES=0"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
python -m standalone_nnunet2d.predict --checkpoint "C:\lijialin\experiments\h2former-isles-finetune-bs4-run01\fold_0\checkpoint_best.pth" --preprocessed-root "C:\lijialin\models3d\nnUNet\nnUNet_preprocessed\Dataset501_StrokeLesion\nnUNetPlans_2d" --plans "C:\lijialin\models3d\nnUNet\nnUNet_preprocessed\Dataset501_StrokeLesion\nnUNetPlans.json" --splits-file "C:\lijialin\models3d\nnUNet\nnUNet_preprocessed\Dataset501_StrokeLesion\splits_final.json" --raw-root "C:\CONFIRM_DATASET501_RAW_ROOT" --fold 0 --output-root "C:\lijialin\experiments\h2former-source-space-eval-fold0-run01" --device cuda:0 --slice-batch-size 1 --tile-step-size 0.5 --allow-pending
```

GPU 0 is exposed as logical cuda:0. Do not pass cuda:1 merely because a physical
index differs. No --confirm-run or training action is involved. Require manifest
status complete; on failure stop before metrics. The former raw predict command
without --preprocessed-root is legacy behavior and is **not** this evaluation.

### 3. Original-space geometry and complete-case check

```bat
call conda activate nnunet5090
cd /d "C:\lijialin\segementation\.worktrees\h2former-source-space-evaluation"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
python -m standalone_nnunet2d.source_space_evaluation check --prediction-root "C:\lijialin\experiments\h2former-source-space-eval-fold0-run01" --raw-root "C:\CONFIRM_DATASET501_RAW_ROOT" --gt-dir "C:\CONFIRM_DATASET501_RAW_ROOT\labelsTr"
```

### 4. All existing metrics, with paper F2 for comparability

```bat
call conda activate nnunet5090
cd /d "C:\lijialin\segementation\.worktrees\h2former-source-space-evaluation"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
python -m standalone_nnunet2d.source_space_evaluation metrics --prediction-root "C:\lijialin\experiments\h2former-source-space-eval-fold0-run01" --raw-root "C:\CONFIRM_DATASET501_RAW_ROOT" --gt-dir "C:\CONFIRM_DATASET501_RAW_ROOT\labelsTr" --output-root "C:\lijialin\experiments\h2former-source-space-metrics-fold0-run01-paper" --f2-mode paper
```

For independently labelled standard F2 results use a different fresh output:

```bat
call conda activate nnunet5090
cd /d "C:\lijialin\segementation\.worktrees\h2former-source-space-evaluation"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
python -m standalone_nnunet2d.source_space_evaluation metrics --prediction-root "C:\lijialin\experiments\h2former-source-space-eval-fold0-run01" --raw-root "C:\CONFIRM_DATASET501_RAW_ROOT" --gt-dir "C:\CONFIRM_DATASET501_RAW_ROOT\labelsTr" --output-root "C:\lijialin\experiments\h2former-source-space-metrics-fold0-run01-standard" --f2-mode standard
```

The wrapper reuses existing metrics directly and embeds pending/provenance in the
same JSON report. Directly calling the old metrics script would lose those
completeness guards and marker propagation; use the wrapper as evaluation entry.

## Five-fold OOF contract and outstanding evidence

Only a fold0 checkpoint is currently known. Fold0 is a development/engineering
result, not complete OOF. For five-fold OOF, separately produce fold0..4 with each
fold's own finetuned checkpoint_best, matching target fold identity, identical
plans/dataset/source/split and prediction policy. Each manifest must be complete,
geometry-checked, held-out validation membership exact; their disjoint case union
must equal the fixed cohort exactly once. Aggregate per-case results once over that
union, not mean fold means when sizes differ. Never reuse fold0 weights for folds1..4
or relabel five copies as OOF. No OOF orchestration/automatic aggregation added.

Still pending: independent Level 3 review, confirmed raw server root, actual
properties content/reader and preprocessing provenance, real checkpoint/state,
server dependency/runtime compatibility, GPU/RAM capacity, full-fold source-space
geometry and metrics, and all remaining folds. Metadata/synthetic passing does not
close these gaps. Where target Dataset501 source metadata has only a patient-map hash
but no map in dataset.json, supply the original trusted metadata via the upstream
source contract; this entry refuses to infer the patient map.

## Historical local validation evidence (engineering only; predates B1--B4 fixes)

All commands used explicit escalation approval and newconda Python. GPU kernels,
training and real data/weights were not used. Every pytest run had a new CPU/RAM/
GPU/VRAM preflight and one CPU thread (OMP_NUM_THREADS=1, MKL_NUM_THREADS=1).
Temporary synthetic files were isolated under D:\codex-pytest-temp.

- Initial RED: new module absent; import collection error, exit 1, 3.46s.
- First integration attempt: 5 failed, 1 passed, 13 fixture errors (3.78s): missing
  pytest parent directory and incomplete synthetic plans architecture. Corrected
  fixtures/environment, no weakened production checks.
- Next: 17 passed, 2 failed (3.87s): output guard correctly rejected fixture outputs
  within plans parent. Moved synthetic plans under protected prepared root.
- Focused: 25 passed (4.00s), exit 0.
- Additional guard RED: 2 failed, 25 deselected (3.76s); metrics protection and
  b2nd-pair-header shape checks missing. Added minimal production guards.
- Focused after fix: 27 passed (4.47s), exit 0. Four warnings are installed
  batchgenerators/SciPy deprecations, no geometry/prediction failures.

CPU samples ranged 18.5..27.5%; RAM 59.6..62.7%; GPU 1..30%; VRAM 913..1160 MiB of
6144 MiB in completed test preflights. psutil is not installed in newconda; CIM
provided CPU/RAM instead, no package installation. Remaining fresh final/affected
validation records are appended below after execution. No broad/full suite planned.

Final focused/affected evidence (2026-10-07):

- Fresh focused before last two added regressions: 29 passed, 4 warnings, 5.31s,
  exit 0. Preflight CPU 23%, RAM 62.215%, GPU 28%, VRAM 985/6144 MiB.
- Final affected: 82 passed, 4 warnings, 8.38s, exit 0. Includes all 31 new tests
  plus existing predictor, prepared selection, full-volume validation, metrics and
  tiny/metadata prediction-loader regression tests. Preflight CPU 11.5%, RAM
  61.949%, GPU 1%, VRAM 1019/6144 MiB. No 80% downgrade required. Excluded the two
  existing predict-command tests that instantiate full PlainConvUNet checkpoints;
  the new tiny-checkpoint legacy raw CLI test covers dispatch/source geometry.
- No full suite, real H2Former checkpoint/model compatibility, CUDA kernel/memory,
  real full-fold export/geometry, server formal metrics or five-fold OOF execution.
  Those remain pending; synthetic tests do not constitute real-data validation.

Exact final affected invocation (local PowerShell, approval/escalation used):

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& 'D:/Anaconda/envs/newconda/python.exe' -m pytest -q standalone_nnunet2d/tests/test_source_space_evaluation.py standalone_nnunet2d/tests/test_predictor.py standalone_nnunet2d/tests/test_formal_selection.py standalone_nnunet2d/tests/test_formal_validation.py standalone_nnunet2d/tests/test_metrics.py standalone_nnunet2d/tests/test_predict_command.py::test_prediction_loader_uses_explicit_single_output_metadata standalone_nnunet2d/tests/test_predict_command.py::test_prediction_loader_legacy_checkpoint_defaults_to_plain_conv standalone_nnunet2d/tests/test_predict_command.py::test_prediction_loader_rejects_incomplete_or_conflicting_model_identity standalone_nnunet2d/tests/test_predict_command.py::test_prediction_command_rejects_invalid_checkpoint_alignment_metadata --basetemp D:/codex-pytest-temp/stroke-lesion-segmentation/h2-source-affected
```

Review snapshot is the exact base HEAD plus the four uncommitted changed-file
SHA256 values in HANDOFF. `git diff --binary HEAD` covers tracked predict.py;
three untracked files must also be inspected in full. Nothing is staged.
Independent reviewer should trace identity guards -> prepared image-only slices
-> logits fusion -> official resampling/crop/transpose -> source geometry -> exact
case manifest -> GT metrics, and independently assess the server evidence gaps.

The repository `.gitignore:7` ignores `docs/`. This delivered document exists
but is ignored/untracked and does not appear in ordinary `git status --short`.
It is explicitly included in the four-file snapshot hashes and must be transferred
and reviewed. No .gitignore edit or forced staging was performed.
