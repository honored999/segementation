# Reproduction Notes

## Directly supplied by `nnUNetPlans.json`

The 2D patch size, batch size, spacing, normalization, mask-for-normalization,
resampling names/kwargs, PlainConvUNet class name, eight stages, feature list,
kernels, strides, encoder/decoder convolution counts, convolution bias,
InstanceNorm2d settings, no-dropout setting, LeakyReLU inplace setting, and
`batch_dice=true` are read directly from the reference JSON.

## Fold and benchmark provenance

`splits_final.json` is used without random re-splitting: five folds each have
76 train and 19 validation cases, with 95 unique validation cases exactly once.
`summary.json` supplies the official 95-case foreground Dice baseline
0.731103738314918 and IoU baseline 0.5923877518050135.

## Defaults and outstanding confirmation

The plans omit `LeakyReLU.negative_slope`; the implementation uses 0.01, the
PyTorch `nn.LeakyReLU` default observed in the current `newconda` environment.
The `newconda` environment did not contain `nnunetv2` or
`dynamic_network_architectures` during the initial inspection. The requested
`nnunet5090` Conda environment was also unavailable at its declared location,
so official source could not be used to independently confirm the exact
deep-supervision head selection/order or trainer defaults. The current
implementation returns heads in full-to-low-resolution order: 512, 256, 128,
64, 32, 16, then 8 pixels.

The exact official augmentation, foreground oversampling, inference mirroring,
sliding-window, deep-supervision weighting, optimization schedule, and
preprocessing defaults remain unconfirmed without a compatible official source
installation. Local implementations only expose explicit caller configuration;
they are not claims of source-verified official defaults.

## Read-only data pipeline

The current data pipeline reads one explicitly requested image/label pair at a
time, uses the fixed supplied folds, resamples x/y to the plan spacing, and
applies full-image Z-score normalization. By default it extracts a deterministic
central axial slice; callers may explicitly opt into foreground slice sampling
and paired augmentation. It does not cache volumes or start training.
Segmentation resampling uses linear interpolation followed by rounding to
preserve discrete labels; this is a practical interpretation of the plan's
segmentation `order=1` until the official preprocessing source is available.

## Loss and deep supervision

The current loss implementation uses foreground-only batch Soft Dice with
smooth value `1e-5`, plus categorical cross-entropy at equal weights. These are
explicit local choices, not claims of source-verified official defaults.
Deep-supervision weights are mandatory caller-provided positive values that are
normalized to sum to one; the project does not supply an alleged official
seven-scale schedule. This phase implements loss computation and gradient tests
only—no optimizer, training loop, or formal training run is enabled.

## Metrics and OOF summaries

Binary Dice and IoU are computed only from caller-supplied 0/1 masks, with
empty prediction/reference masks defined as perfect agreement. OOF summaries
require unique case IDs and report case-level mean Dice/IoU. The reference JSON
baseline remains 95 cases with Dice `0.731103738314918` and IoU
`0.5923877518050135`; the project has not generated predictions or claimed to
reproduce those values.

## Sampling and augmentation

Foreground slice selection and paired flips/intensity scaling are optional,
caller-configured features that default to disabled. They use supplied random
generators and preserve label geometry; their probabilities and ranges are not
claimed to match unverified official nnU-Net settings. No training run is
enabled by this addition.

## Checkpoint interface

Checkpoint persistence is explicit and local-only: `save_checkpoint` and
`load_checkpoint` accept only paths resolving below `standalone_nnunet2d/outputs/`.
Version-1 payloads contain `model_state_dict`, optional `optimizer_state_dict`,
and caller-supplied dictionary metadata. Loading maps tensors to CPU, validates
the format and required fields, and rejects any caller-specified metadata value
that does not match the stored metadata. These utilities neither schedule a
checkpoint nor enable formal training.

## Explicit epoch utilities

`run_train_epoch` performs one optimizer update for each caller-supplied batch
through the existing `train_step`, then returns the batch count, mean loss, and
final output shapes. `run_validation_epoch` is binary-only: it selects the
full-resolution logits (the first deep-supervision output when present), requires
two channels and target-compatible `(N, H, W)` labels, applies argmax masks, and
calculates Dice/IoU from global TP/FP/FN counts. Validation runs under
`torch.no_grad()` and restores the model's original top-level training/evaluation
mode. These utilities do not construct loaders, discover cases, write results,
or enable the disabled formal training command.

## Server preflight and safe dry run

`python -m standalone_nnunet2d.dry_run` accepts explicit raw, preprocessed, and
results directory paths and reports normalized path status, selected-device CUDA
facts, GPU names/count, and the local reference plan's patch size, batch size,
and stage count. It does not recurse into the supplied directories or open
dataset cases. A ready report exits 0; a missing directory, unavailable requested
CUDA device, or unreadable reference plan exits 2. `--run` is rejected, so this
preflight cannot start a formal training run or create outputs.

## Experiment command contract

`experiment.py` validates the three data roots, an output root confined below
`standalone_nnunet2d/outputs/`, a fixed fold 0--4, and a positive epoch count.
Its normal result is JSON-only: exit 0 when preflight is ready and exit 2 for
invalid arguments or failed readiness. `--confirm-run` is an explicit consent
boundary reserved for the future training phase; it currently exits 3 after
reporting `execution=deferred`, with no DataLoader, optimizer, model artifact,
or directory creation.

## Documentation contract and verification workflow

The local documentation gate is:

```powershell
conda run -n newconda python -m pytest standalone_nnunet2d/tests/test_documentation_contract.py -q
conda run -n newconda python -m pytest standalone_nnunet2d/tests -q
conda run -n newconda python -m py_compile standalone_nnunet2d/formal_train.py standalone_nnunet2d/predict.py standalone_nnunet2d/validate_cv.py standalone_nnunet2d/oracle_capture.py
```

The first command checks this file and `README.md`; the second is the full local
test suite; the third is the static syntax check for the formal command entry
points. A smoke result or an online validation value is not an official
reproduction. Smoke results and online validation are not official reproduction.

All formal training, prediction, fold-validation, and OOF results remain
`official_alignment_pending` before alignment. A parity report also keeps that
run state even when its comparison status is `passed`. Only a passed transform
parity report together with a passed `repeat_oracle_stability_v1` report
permits a later artifact/result to be labeled `official_aligned`; a single-root
inference report is diagnostic only, and one report alone is not enough.

## Server oracle capture (server-only)

Server oracle capture is permitted only in an environment with the official
`nnunetv2` installation. These commands are documented for the server handoff
only and are not run in `newconda` or by local verification. Capture the
transform/preprocessing artifact first:

```powershell
conda run -n <server-env> python -m standalone_nnunet2d.oracle_capture `
  --mode preprocess `
  --raw-root C:\path\to\Dataset501_StrokeLesion `
  --preprocessed-root C:\path\to\Dataset501_StrokeLesion `
  --results-root C:\path\to\nnUNet_results `
  --output-root standalone_nnunet2d\outputs\oracle `
  --case-id case_0001 `
  --fold 0 `
  --seed 17
```

Then capture the matching inference artifact using the official model folder:

```powershell
conda run -n <server-env> python -m standalone_nnunet2d.oracle_capture `
  --mode inference `
  --raw-root C:\path\to\Dataset501_StrokeLesion `
  --preprocessed-root C:\path\to\Dataset501_StrokeLesion `
  --results-root C:\path\to\nnUNet_results `
  --output-root standalone_nnunet2d\outputs\oracle `
  --case-id case_0001 `
  --fold 0 `
  --seed 17 `
  --model-folder C:\path\to\nnUNet_results\Dataset501_StrokeLesion\... `
  --device cuda `
  --checkpoint-name checkpoint_best.pth
```

The capture entry point also accepts `sample`, `transform`, and
`deep_supervision` modes. The handoff sequence is transform capture, three
independent inference captures, local transform parity report, repeated local
inference parity report, and then one fold-0 local prediction/validation
export. Do not begin five-fold training until both parity reports have status
`passed`.

## Local parity report

Run the two reports in `newconda` after the corresponding server artifacts and
standalone artifacts exist:

```powershell
conda run -n newconda python -m standalone_nnunet2d.tools.parity_report `
  --oracle-root standalone_nnunet2d\outputs\oracle\preprocess\case_0001 `
  --standalone-root standalone_nnunet2d\outputs\standalone\preprocess\case_0001 `
  --image-atol 0 `
  --output C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\transform_parity_case005_v3.json

conda run -n newconda python -m standalone_nnunet2d.tools.parity_report `
  --oracle-root standalone_nnunet2d\outputs\oracle\inference\case_0001 `
  --standalone-root standalone_nnunet2d\outputs\standalone\inference\case_0001 `
  --image-atol 0 `
  --output standalone_nnunet2d\outputs\inference_parity_report.json
```

The single-root inference command above is an exact comparison for diagnosis
only. The accepted inference gate is `repeat_oracle_stability_v1` and requires
at least three independent oracle runs, each captured by a separate server
`nnunetv2` invocation under a different root:

```powershell
conda run -n newconda python -m standalone_nnunet2d.tools.parity_report `
  --oracle-root standalone_nnunet2d\outputs\oracle_inference_run1\inference\case_0001 `
  --oracle-root standalone_nnunet2d\outputs\oracle_inference_run2\inference\case_0001 `
  --oracle-root standalone_nnunet2d\outputs\oracle_inference_run3\inference\case_0001 `
  --standalone-root standalone_nnunet2d\outputs\standalone_inference\inference\case_0001 `
  --image-atol 0 `
  --output standalone_nnunet2d\outputs\inference_repeat_parity_report.json
```

The stable voxel exact rule is that every stable voxel must match the
unanimous official label exactly. An unstable voxel only accepts labels
observed across oracle repeats. The report must report all unstable coordinates
and pairwise differences. Do not use a Dice threshold, morphological cleanup,
nonzero tolerance, retry-until-match behavior, or a hidden fallback.

Inference artifacts must include the `inference_context` containing the fold,
official source-checkpoint SHA256, and normalized device. Existing inference
artifacts without this context must be recaptured; they are not automatically
compatible and the gate has no legacy fallback.

Both reports must say `status: "passed"` before an external alignment decision,
but the report remains `official_alignment_pending`. A passing repeated report
does not automatically relabel historical runs `official_aligned`; existing
training, prediction, validation, and OOF results remain pending.

## Fold-0 formal train, prediction, and validation

This legacy pending workflow is retained for compatibility and diagnostic only.
It must not be used as the final `official_aligned` workflow or to label new
results `official_aligned`:

```powershell
conda run -n newconda python standalone_nnunet2d\formal_train.py `
  --raw-root C:\path\to\Dataset501_StrokeLesion `
  --plans C:\path\to\nnUNetPlans.json `
  --output-root standalone_nnunet2d\outputs\formal\fold_0 `
  --fold 0 `
  --device cuda:0 `
  --epochs 1000 `
  --confirm-run

conda run -n newconda python standalone_nnunet2d\predict.py `
  --checkpoint standalone_nnunet2d\outputs\formal\fold_0\checkpoint_best.pth `
  --raw-root C:\path\to\Dataset501_StrokeLesion `
  --fold 0 `
  --output-root standalone_nnunet2d\outputs\formal\fold_0_predictions `
  --device cuda:0 `
  --allow-pending

conda run -n newconda python standalone_nnunet2d\validate_cv.py fold `
  --checkpoint standalone_nnunet2d\outputs\formal\fold_0\checkpoint_best.pth `
  --raw-root C:\path\to\Dataset501_StrokeLesion `
  --fold 0 `
  --output-root standalone_nnunet2d\outputs\formal\crossval `
  --device cuda:0 `
  --allow-pending
```

The fold-0 validation command writes `fold_0_report.json` and
`fold_0_case_metrics.csv` in the shared `crossval` directory. Its metrics and
the prediction manifest remain `official_alignment_pending` until the two
parity reports have passed.

## Five-fold training and OOF sequence

The following legacy pending commands are diagnostic only, not the final
aligned five-fold workflow. They must not be used as the final `official_aligned` workflow.
Repeat them for folds `1`, `2`, `3`, and `4`, using a
fold-specific training and prediction directory while keeping the validation
output root shared:

```powershell
conda run -n newconda python standalone_nnunet2d\formal_train.py --raw-root C:\path\to\Dataset501_StrokeLesion --plans C:\path\to\nnUNetPlans.json --output-root standalone_nnunet2d\outputs\formal\fold_<FOLD> --fold <FOLD> --device cuda:0 --epochs 1000 --confirm-run
conda run -n newconda python standalone_nnunet2d\predict.py --checkpoint standalone_nnunet2d\outputs\formal\fold_<FOLD>\checkpoint_best.pth --raw-root C:\path\to\Dataset501_StrokeLesion --fold <FOLD> --output-root standalone_nnunet2d\outputs\formal\fold_<FOLD>_predictions --device cuda:0 --allow-pending
conda run -n newconda python standalone_nnunet2d\validate_cv.py fold --checkpoint standalone_nnunet2d\outputs\formal\fold_<FOLD>\checkpoint_best.pth --raw-root C:\path\to\Dataset501_StrokeLesion --fold <FOLD> --output-root standalone_nnunet2d\outputs\formal\crossval --device cuda:0 --allow-pending
```

Replace `<FOLD>` with `1`, `2`, `3`, and `4` in order. After the shared
directory contains exactly `fold_0_report.json` through `fold_4_report.json`,
aggregate all 95 unique validation cases:

```powershell
conda run -n newconda python standalone_nnunet2d\validate_cv.py aggregate `
  --output-root standalone_nnunet2d\outputs\formal\crossval
```

This writes `oof_per_case_metrics.csv` and `oof_summary.json`; the OOF result
remains `official_alignment_pending` until both the transform and inference
parity reports are passed.

## Final official-aligned five-fold workflow

Synchronize the current final source before training, but do not copy
`outputs/`. Use a fresh output root:
`C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\official_aligned_5fold`.
Only when both `--transform-parity-report` and `--inference-parity-report` are
supplied and validated may new checkpoints use `official_aligned`. Omitting
both reports leaves the run `official_alignment_pending`; providing only one
is rejected. The epoch 999/fold 0 is not retroactively upgraded. Parity reports
themselves remain `official_alignment_pending`; only newly generated artifacts
may be labeled `official_aligned`. Smoke and online validation are never
official.

Use these evidence files. If the transform report was saved under a different
name, replace that path consistently:

```text
C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\transform_parity_case005_v3.json
C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\inference_repeat_parity_case005_ctx.json
```

The raw root is
`C:\lijialin\models3d\nnUNet\nnUNet_raw\Dataset501_StrokeLesion` and the plans
file is
`C:\lijialin\models3d\nnUNet\nnUNet_preprocessed\Dataset501_StrokeLesion\nnUNetPlans.json`.
The batch size is fixed at 12. The throughput profile does not enable AMP,
TF32, or compile.

Run each fold once:

```powershell
conda run -n newconda python standalone_nnunet2d\formal_train.py --raw-root "C:\lijialin\models3d\nnUNet\nnUNet_raw\Dataset501_StrokeLesion" --plans "C:\lijialin\models3d\nnUNet\nnUNet_preprocessed\Dataset501_StrokeLesion\nnUNetPlans.json" --output-root "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\official_aligned_5fold\formal\fold_0" --fold 0 --device cuda:0 --epochs 1000 --performance-profile throughput --transform-parity-report "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\transform_parity_case005_v3.json" --inference-parity-report "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\inference_repeat_parity_case005_ctx.json" --confirm-run
conda run -n newconda python standalone_nnunet2d\formal_train.py --raw-root "C:\lijialin\models3d\nnUNet\nnUNet_raw\Dataset501_StrokeLesion" --plans "C:\lijialin\models3d\nnUNet\nnUNet_preprocessed\Dataset501_StrokeLesion\nnUNetPlans.json" --output-root "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\official_aligned_5fold\formal\fold_1" --fold 1 --device cuda:0 --epochs 1000 --performance-profile throughput --transform-parity-report "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\transform_parity_case005_v3.json" --inference-parity-report "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\inference_repeat_parity_case005_ctx.json" --confirm-run
conda run -n newconda python standalone_nnunet2d\formal_train.py --raw-root "C:\lijialin\models3d\nnUNet\nnUNet_raw\Dataset501_StrokeLesion" --plans "C:\lijialin\models3d\nnUNet\nnUNet_preprocessed\Dataset501_StrokeLesion\nnUNetPlans.json" --output-root "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\official_aligned_5fold\formal\fold_2" --fold 2 --device cuda:0 --epochs 1000 --performance-profile throughput --transform-parity-report "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\transform_parity_case005_v3.json" --inference-parity-report "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\inference_repeat_parity_case005_ctx.json" --confirm-run
conda run -n newconda python standalone_nnunet2d\formal_train.py --raw-root "C:\lijialin\models3d\nnUNet\nnUNet_raw\Dataset501_StrokeLesion" --plans "C:\lijialin\models3d\nnUNet\nnUNet_preprocessed\Dataset501_StrokeLesion\nnUNetPlans.json" --output-root "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\official_aligned_5fold\formal\fold_3" --fold 3 --device cuda:0 --epochs 1000 --performance-profile throughput --transform-parity-report "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\transform_parity_case005_v3.json" --inference-parity-report "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\inference_repeat_parity_case005_ctx.json" --confirm-run
conda run -n newconda python standalone_nnunet2d\formal_train.py --raw-root "C:\lijialin\models3d\nnUNet\nnUNet_raw\Dataset501_StrokeLesion" --plans "C:\lijialin\models3d\nnUNet\nnUNet_preprocessed\Dataset501_StrokeLesion\nnUNetPlans.json" --output-root "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\official_aligned_5fold\formal\fold_4" --fold 4 --device cuda:0 --epochs 1000 --performance-profile throughput --transform-parity-report "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\transform_parity_case005_v3.json" --inference-parity-report "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\inference_repeat_parity_case005_ctx.json" --confirm-run
```

After each training command, run fold validation directly. It does not require
a separate `predict.py` run. Each command performs full-volume prediction,
writes the prediction manifest, and writes the fold report. Fold reports share
one `crossval` directory:

```powershell
conda run -n newconda python standalone_nnunet2d\validate_cv.py fold --checkpoint "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\official_aligned_5fold\formal\fold_0\checkpoint_best.pth" --raw-root "C:\lijialin\models3d\nnUNet\nnUNet_raw\Dataset501_StrokeLesion" --fold 0 --output-root "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\official_aligned_5fold\crossval" --device cuda:0
conda run -n newconda python standalone_nnunet2d\validate_cv.py fold --checkpoint "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\official_aligned_5fold\formal\fold_1\checkpoint_best.pth" --raw-root "C:\lijialin\models3d\nnUNet\nnUNet_raw\Dataset501_StrokeLesion" --fold 1 --output-root "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\official_aligned_5fold\crossval" --device cuda:0
conda run -n newconda python standalone_nnunet2d\validate_cv.py fold --checkpoint "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\official_aligned_5fold\formal\fold_2\checkpoint_best.pth" --raw-root "C:\lijialin\models3d\nnUNet\nnUNet_raw\Dataset501_StrokeLesion" --fold 2 --output-root "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\official_aligned_5fold\crossval" --device cuda:0
conda run -n newconda python standalone_nnunet2d\validate_cv.py fold --checkpoint "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\official_aligned_5fold\formal\fold_3\checkpoint_best.pth" --raw-root "C:\lijialin\models3d\nnUNet\nnUNet_raw\Dataset501_StrokeLesion" --fold 3 --output-root "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\official_aligned_5fold\crossval" --device cuda:0
conda run -n newconda python standalone_nnunet2d\validate_cv.py fold --checkpoint "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\official_aligned_5fold\formal\fold_4\checkpoint_best.pth" --raw-root "C:\lijialin\models3d\nnUNet\nnUNet_raw\Dataset501_StrokeLesion" --fold 4 --output-root "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\official_aligned_5fold\crossval" --device cuda:0
```

The aggregate requires fold_0_report.json through fold_4_report.json, exactly
95 unique IDs, zero failed cases, and identical evidence. It writes an
`official_aligned` `oof_summary.json`:

```powershell
conda run -n newconda python standalone_nnunet2d\validate_cv.py aggregate --output-root "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\official_aligned_5fold\crossval"
cd "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\official_aligned_5fold\crossval"
type oof_summary.json
```
