# Two-stage coarse-to-fine DWI lesion segmentation

This package contains the prediction-guided ROI, reversible XY crop/restore,
and original-volume comparison pieces for a two-stage acute ischemic stroke
DWI segmentation experiment.

The local implementation is validated with synthetic NIfTI fixtures only. The
formal server experiment has not been run by this checkout: the real Dataset501
NIfTI files, default-Trainer five-fold checkpoints, complete Stage 1 OOF
predictions, and server runtime are not locally available. No number in this
document is a formal experiment result.

## Protocol and data lineage

The formal data flow is:

```text
Dataset501 DWI imagesTr (read-only)
  -> Stage 1 default nnUNetTrainer, 2d, fixed five-fold validation
  -> exactly one OOF prediction for each of the 95 cases
  -> prediction-only XY ROI, with full z preserved
  -> derived Dataset504_StrokeLesion_CoarseToFine
  -> Stage 2 default nnUNetTrainer, 2d, fixed five-fold training
  -> cropped Stage 2 validation predictions
  -> manifest-driven restore to Dataset501 full-volume space
  -> full-volume Stage 1 vs Stage 2 evaluation against original labelsTr
```

The following rules are part of the protocol, not optional conveniences:

- Dataset501 and its labels are source data and remain read-only.
- The established `splits_final.json` is used unchanged: five patient-level
  folds, 76 training cases and 19 validation cases per fold, 95 unique cases
  overall.
- Stage 1 must be the default DWI `nnUNetTrainer` and must provide complete
  five-fold OOF predictions. A fold-0 result, an in-sample prediction, or a
  prediction from another Trainer is not an acceptable Stage 1 source.
- The ROI is computed from the Stage 1 prediction foreground union over all z
  slices. Ground truth is not used to locate the ROI. An empty prediction
  falls back to the complete XY field of view.
- Dataset504 is derived output. Its `manifest.json` records the fixed split,
  case IDs, ROI, fallback status, and Stage 1 prediction source. Its output
  directory must not overlap Dataset501 or the Stage 1 prediction directory.
- Stage 2 predictions are restored with the Dataset504 manifest and checked
  against the original DWI shape, spacing, origin, and direction.
- Final metrics are computed in original full-volume `(z, y, x)` space against
  the original Dataset501 `labelsTr`, with equal case weighting. The evaluator
  reports per-case TP/FP/FN, Dice, IoU, precision, recall, case-macro means,
  pooled/global counts, and Stage 2 minus Stage 1 deltas. Both-empty masks
  score 1; one-empty masks score 0.

## What is verified locally and what is pending

Verified from the current checkout:

- `coarse_to_fine_dwi.dataset.build_dataset504` enforces the fixed split and
  exact 95-case OOF ID set, computes prediction-only ROIs, and writes the
  derived Dataset504 plus manifest.
- `coarse_to_fine_dwi.nifti` uses SimpleITK metadata with arrays in `(z, y, x)`
  order and implements reversible XY cropping.
- `python -m coarse_to_fine_dwi.cli.generate_dataset` builds the prediction-
  guided Dataset504 and writes `manifest.json` plus `roi_manifest.json`.
- `python -m coarse_to_fine_dwi.cli.restore_predictions` restores cropped
  predictions to original space.
- `python -m coarse_to_fine_dwi.cli.compare_predictions` compares original
  full-volume Stage 1 and restored Stage 2 predictions.
- The local tests use synthetic volumes and do not establish model performance.

Pending server verification:

- The actual `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results` roots.
- The presence and exact names of all five default-Trainer Stage 1 checkpoints.
- Whether five-fold validation NIfTI predictions already exist and whether
  they are truly out-of-fold, rather than in-sample or fold-mixed outputs.
- The server Conda environment, `nnunetv2` version, GPU/CUDA runtime, and the
  exact checkpoint filename retained by each run.
- Whether Dataset504 has been created. It is planned output, not an existing
  server resource.

The historical paths in `SERVER_PROJECT_STRUCTURE.md` are references, not a
server directory listing. Replace every placeholder below with paths confirmed
on the target server; do not infer a missing file or directory from the
templates.

## Synthetic/local validation

Run from the coarse-to-fine worktree. These commands do not read patient data,
start training, or create a formal result:

```powershell
conda run -n newconda python -m pytest -q `
  tests/test_coarse_to_fine_roi.py `
  tests/test_coarse_to_fine_dataset.py `
  tests/test_coarse_to_fine_evaluate.py

conda run -n newconda python -m pytest standalone_nnunet2d/tests -q
```

The first command is the focused coarse-to-fine synthetic suite. The second
is the affected standalone regression suite. Require a visible pytest summary
and zero failures; an empty wrapper result is not sufficient evidence.

## Parameterized server procedure

The following is a command template, not a command run by this checkout. It
uses only configurable variables and fresh derived-output roots. Set the
variables after checking the server and before running any data-producing step.

```powershell
# Confirm these values on the server. Do not copy historical paths blindly.
$ServerEnv = '<confirmed-server-conda-environment>'
$RepoRoot = '<confirmed-server-checkout-root>'
$NnUNetRawRoot = '<confirmed-nnUNet_raw-root>'
$NnUNetPreprocessedRoot = '<confirmed-nnUNet_preprocessed-root>'
$NnUNetResultsRoot = '<confirmed-nnUNet_results-root>'
$DerivedRoot = '<fresh-derived-output-root>'

$Dataset501Id = 501
$Dataset504Id = 504
$Dataset501Raw = Join-Path $NnUNetRawRoot 'Dataset501_StrokeLesion'
$Dataset504Raw = Join-Path $NnUNetRawRoot 'Dataset504_StrokeLesion_CoarseToFine'
$SplitFile = Join-Path $RepoRoot 'standalone_nnunet2d\reference\splits_final.json'

$RunRoot = Join-Path $DerivedRoot '<unique-run-id>'
$Stage1WorkRoot = Join-Path $RunRoot 'stage1_fold_runs'
$Stage1OofDir = Join-Path $RunRoot 'stage1_oof'
$Stage1Provenance = Join-Path $RunRoot 'stage1_provenance.json'
$Stage2WorkRoot = Join-Path $RunRoot 'stage2_fold_runs'
$Stage2CroppedPredictions = Join-Path $RunRoot 'stage2_cropped_predictions'
$Stage2RestoredDir = Join-Path $RunRoot 'stage2_restored_full_volume'
$EvaluationDir = Join-Path $RunRoot 'full_volume_evaluation'

# These names must be confirmed from the server result directories.
$Stage1CheckpointName = '<confirmed-stage1-checkpoint-filename>'
$Stage2CheckpointName = '<confirmed-stage2-checkpoint-filename>'
$RoiMargin = 0
$RoiMinWidth = 1
$RoiMinHeight = 1

if (Test-Path -LiteralPath $RunRoot) {
  throw "RunRoot already exists; choose a fresh derived-output root: $RunRoot"
}
New-Item -ItemType Directory -Path $RunRoot | Out-Null
New-Item -ItemType Directory -Path $Stage1WorkRoot, $Stage1OofDir, $Stage2WorkRoot, $Stage2CroppedPredictions | Out-Null

$env:nnUNet_raw = $NnUNetRawRoot
$env:nnUNet_preprocessed = $NnUNetPreprocessedRoot
$env:nnUNet_results = $NnUNetResultsRoot
Set-Location -LiteralPath $RepoRoot
```

### 1. Server preflight and Stage 1 provenance gate

Run these read-only checks first. Stop if an entry point, dataset directory,
fold result, or checkpoint cannot be confirmed.

```powershell
Get-Command nnUNetv2_plan_and_preprocess, nnUNetv2_train, nnUNetv2_predict
conda run -n $ServerEnv python -c "import nnunetv2; print(nnunetv2.__version__)"
conda run -n $ServerEnv python standalone_nnunet2d/tools/inspect_dataset.py --raw-root $Dataset501Raw
Get-ChildItem -LiteralPath (Join-Path $NnUNetResultsRoot 'Dataset501_StrokeLesion') -Directory
Get-ChildItem -LiteralPath $Dataset501Raw -Directory
```

The default-Trainer result folders and checkpoint files must be inspected for
all folds before using them. The exact folder names and checkpoint filenames
are pending until this check succeeds. Do not convert a fold-0 output into an
OOF source.

### 2. Produce or collect complete Stage 1 OOF predictions

If a server audit has already confirmed a complete, default-Trainer, fixed
five-fold OOF prediction directory, set `$Stage1OofDir` to a fresh derived copy
of that directory and verify that it contains exactly the 95 case IDs. If it
does not exist, the following template predicts each validation fold using its
confirmed default-Trainer checkpoint. It stages validation images outside the
read-only Dataset501 tree and never uses labels to select inputs.

```powershell
$splits = Get-Content -Raw -LiteralPath $SplitFile | ConvertFrom-Json

foreach ($fold in 0..4) {
  $foldInput = Join-Path $Stage1WorkRoot "fold_$fold\input"
  $foldOutput = Join-Path $Stage1WorkRoot "fold_$fold\prediction"
  New-Item -ItemType Directory -Path $foldInput, $foldOutput | Out-Null

  foreach ($caseId in @($splits[$fold].val)) {
    $source = Join-Path $Dataset501Raw "imagesTr\${caseId}_0000.nii.gz"
    Copy-Item -LiteralPath $source -Destination $foldInput
  }

  conda run -n $ServerEnv nnUNetv2_predict `
    -i $foldInput `
    -o $foldOutput `
    -d $Dataset501Id `
    -c 2d `
    -f $fold `
    -tr nnUNetTrainer `
    -chk $Stage1CheckpointName

  Get-ChildItem -LiteralPath $foldOutput -Filter '*.nii.gz' -File |
    ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $Stage1OofDir }
}
```

Before proceeding, inspect the resulting names and metadata. The Dataset504
builder will reject missing/extra IDs and shape or spatial-metadata mismatch;
do not rename or silently repair predictions to bypass those checks.

### 3. Build the prediction-guided Dataset504

Use the Dataset504 builder CLI from the repository checkout. The provenance
file is required and must describe the Stage 1 source; an unverified record
keeps the generated manifest non-formal (`formal_eligible: false`). Do not
write a hand-authored assertion merely to enable formal reporting.

```powershell
conda run -n $ServerEnv python -m coarse_to_fine_dwi.cli.generate_dataset `
  --dataset501-raw $Dataset501Raw `
  --stage1-oof-dir $Stage1OofDir `
  --output-root $Dataset504Raw `
  --splits $SplitFile `
  --margin-px $RoiMargin `
  --min-roi-size $RoiMinWidth $RoiMinHeight `
  --stage1-provenance $Stage1Provenance
```

This writes only derived Dataset504 data, `dataset.json`, the fixed
`splits_final.json`, `manifest.json`, and `roi_manifest.json`. It does not
alter Dataset501. Missing or extra OOF case IDs, fixed-split mismatches, shape
or spatial-metadata mismatches, an existing destination, or overlap with a
source root cause the CLI to fail; missing predictions are never silently
skipped. An empty prediction uses the documented full-XY fallback.

### 4. Plan, preprocess, and train default Stage 2 2D nnU-Net

These are official nnU-Net server entry points and must be confirmed by the
preflight above. They are not implemented by this package and are not the
standalone PyTorch `formal_train.py` command.

```powershell
conda run -n $ServerEnv nnUNetv2_plan_and_preprocess `
  -d $Dataset504Id `
  --verify_dataset_integrity

foreach ($fold in 0..4) {
  conda run -n $ServerEnv nnUNetv2_train `
    $Dataset504Id `
    2d `
    $fold `
    -tr nnUNetTrainer
}
```

Use a fresh result namespace or a server-confirmed existing one. Do not mix
outputs with another Trainer, plans identifier, dataset, or interrupted run.
Training completion alone is not a formal result; the next step must produce
the five validation-fold prediction sets and then restore them to original
space.

### 5. Predict Stage 2 on each held-out cropped fold

Stage 2 validation inputs are selected by the fixed split, not by GT. The
following stages only the cropped Dataset504 images for each validation fold,
then collects one cropped prediction per case into a new common directory.

```powershell
$splits = Get-Content -Raw -LiteralPath $SplitFile | ConvertFrom-Json

foreach ($fold in 0..4) {
  $foldInput = Join-Path $Stage2WorkRoot "fold_$fold\input"
  $foldOutput = Join-Path $Stage2WorkRoot "fold_$fold\prediction"
  New-Item -ItemType Directory -Path $foldInput, $foldOutput | Out-Null

  foreach ($caseId in @($splits[$fold].val)) {
    $source = Join-Path $Dataset504Raw "imagesTr\${caseId}_0000.nii.gz"
    Copy-Item -LiteralPath $source -Destination $foldInput
  }

  conda run -n $ServerEnv nnUNetv2_predict `
    -i $foldInput `
    -o $foldOutput `
    -d $Dataset504Id `
    -c 2d `
    -f $fold `
    -tr nnUNetTrainer `
    -chk $Stage2CheckpointName

  Get-ChildItem -LiteralPath $foldOutput -Filter '*.nii.gz' -File |
    ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $Stage2CroppedPredictions }
}
```

### 6. Restore and evaluate in original full-volume space

Restore is manifest-driven and writes to a derived directory outside raw data:

```powershell
conda run -n $ServerEnv python -m coarse_to_fine_dwi.cli.restore_predictions `
  --manifest (Join-Path $Dataset504Raw 'manifest.json') `
  --cropped-predictions $Stage2CroppedPredictions `
  --dataset501-raw $Dataset501Raw `
  --output-dir $Stage2RestoredDir

conda run -n $ServerEnv python -m coarse_to_fine_dwi.cli.compare_predictions `
  --dataset501-raw $Dataset501Raw `
  --stage1-oof-dir $Stage1OofDir `
  --stage2-restored-dir $Stage2RestoredDir `
  --output-dir $EvaluationDir `
  --expected-case-count 95
```

The compare CLI writes `stage1_vs_stage2_case_metrics.csv` and
`stage1_vs_stage2_summary.json`. It always evaluates original full-volume
arrays. Its current CLI has no provenance-file argument, so its summary is
`formal_eligible: false`; do not edit that field by hand or call the output an
official result.

The lower-level API can accept a provenance mapping, but only use it after an
independent server audit has produced and verified the required record. The
record must establish all of the following: `verified: true`, default
`nnUNetTrainer`, `complete_5_fold_oof`, prediction-only ROI, fixed five-fold
patient-level splits, five folds, and 95 cases. Supplying an unverified or
hand-written assertion would violate the experiment protocol. Extending the
CLI to ingest such a record is outside this documentation-only change.

## Acceptance boundary

This README and the current package do not establish that the server has the
data, checkpoints, OOF predictions, or a completed training run. The formal
experiment remains pending until the server checklist is satisfied, all five
folds are independently accounted for, and the final full-volume output is
reviewed. Synthetic tests are engineering validation only and must never be
reported as a Dice result for the stroke cohort.
