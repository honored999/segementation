# Standalone nnU-Net 2D Reproduction

This project is a pure-PyTorch foundation for reproducing the 2D `PlainConvUNet`
configuration of `Dataset501_StrokeLesion`. The standalone package does not
import `nnunetv2` or `dynamic_network_architectures` during normal use;
`oracle_capture.py` is the separate server-only, lazy-import exception.

Current progress: reference validation, environment inspection, JSON-driven
network construction, CPU model-shape tests, a read-only data pipeline, and
Dice-plus-cross-entropy deep-supervision loss modules, binary/OOF metrics,
optional caller-configured slice augmentation, a single explicit dry-run
optimization step, formal training/inference/evaluation entry points, and a
safe checkpoint interface are implemented. Formal run artifacts remain
`official_alignment_pending` until both required parity reports pass.

Run the checks from the worktree root (using the required environment):

```powershell
conda run -n newconda python standalone_nnunet2d/tools/inspect_environment.py
conda run -n newconda python standalone_nnunet2d/tools/inspect_reference.py
conda run -n newconda python standalone_nnunet2d/tools/inspect_dataset.py --raw-root C:\path\to\Dataset501_StrokeLesion
conda run -n newconda python -m pytest standalone_nnunet2d/tests/test_documentation_contract.py -q
conda run -n newconda python -m pytest standalone_nnunet2d/tests -q
conda run -n newconda python -m py_compile standalone_nnunet2d/formal_train.py standalone_nnunet2d/predict.py standalone_nnunet2d/validate_cv.py standalone_nnunet2d/oracle_capture.py
```

The three external Dataset501 directories are strictly read-only. Do not copy
the NIfTI dataset, NNUNet outputs, large NPZ files, or model weights into this
repository; any future generated artifacts belong under `outputs/`.

`inspect_dataset.py` only checks paths unless a single `--case-id` is supplied.
It never scans all images. Data loading supports the read-only, on-demand slice
pipeline; formal training, prediction, and fold/OOF validation are explicit
commands documented below.

The loss module exposes foreground batch Dice and cross-entropy. Its
deep-supervision weights are caller-provided rather than asserted to match
unavailable official source defaults.

Metric helpers consume explicitly supplied binary masks and can aggregate
case-level OOF records. They do not create predictions or claim a reproduced
score; the bundled official 95-case Dice/IoU values are comparison baselines.

The data Dataset can opt into foreground-aware slice selection and synchronized
paired augmentation through explicit configuration. Both are disabled by
default. Formal training is explicit and all resulting artifacts remain pending
until the parity gate described below passes.

`engine.checkpoint.save_checkpoint` and `load_checkpoint` are caller-invoked
utilities, not a training loop. They only accept paths below
`standalone_nnunet2d/outputs/`, serialize a versioned model/optional-optimizer/
metadata payload, and never copy external model weights into the repository.

`engine.trainer.run_train_epoch` and `engine.validator.run_validation_epoch`
only consume caller-supplied tensor batches. They do not create a DataLoader,
schedule epochs, write checkpoints, or alter the disabled `train.py` command;
they are composable building blocks for a later, explicitly requested formal
run.

Before using a server, run the read-only preflight with explicit paths:

```powershell
conda run -n newconda python -m standalone_nnunet2d.dry_run `
  --raw-root C:\path\to\Dataset501_StrokeLesion `
  --preprocessed-root C:\path\to\Dataset501_StrokeLesion `
  --results-root C:\path\to\Dataset501_StrokeLesion `
  --device cuda
```

It prints JSON and exits 0 only if all directories, the requested device, and
the local reference plan are ready; it exits 2 otherwise. The command neither
creates directories nor reads cases, allocates a model, writes artifacts, or
starts training. Its `--run` flag is deliberately rejected.

`python -m standalone_nnunet2d.experiment` adds required `--output-root`,
`--fold 0..4`, and positive `--epochs` validation. Its explicit
`--confirm-run` path is a smoke run only; a smoke result is not an official
reproduction and remains `official_alignment_pending`.

## Documentation contract and result labels

The exact local verification commands are the documentation-contract test, the
full local test suite, and the four-file `py_compile` check shown above. A local
test pass, a smoke result, or an online `validation_dice` value is not an
official reproduction. Smoke results and online validation are not official
reproduction.

Smoke and online validation are never official. Before formal training,
synchronize the current final source from this worktree to the runtime
checkout. Sync the source tree only; do not copy `outputs/`, checkpoints,
caches, or other generated artifacts. Use a fresh output root for the aligned
run:

```text
C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\official_aligned_5fold
```

All formal training, prediction, fold-validation, and OOF artifacts are labeled
`official_alignment_pending` before alignment is established. The local parity
report also remains `official_alignment_pending` even when its comparison status
is `passed`. Only a passed transform parity report together with a passed
`repeat_oracle_stability_v1` report permits a later artifact/result to be
labeled `official_aligned`; a single-root inference report is diagnostic only,
and neither report alone is sufficient.

For a new training run, only when both `--transform-parity-report` and
`--inference-parity-report` are supplied and both reports pass may training
write `official_aligned`. Omitting both reports leaves the run
`official_alignment_pending`; supplying only one report is also rejected.
The epoch 999/fold 0 is not retroactively upgraded, and no historical artifact
is relabeled by a later parity result. Parity reports
themselves remain `official_alignment_pending` as evidence; only newly
generated artifacts may be labeled `official_aligned`.

## Server oracle capture (server-only)

The server oracle capture command must run only in an environment where the
official `nnunetv2` installation is available. Do not run it in `newconda` and
do not treat its output as official until the matching local parity reports have
passed. The command supports `preprocess`, `sample`, `transform`,
`deep_supervision`, and `inference` and writes below the selected output root.

Capture a transform/preprocessing artifact on the server:

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

Capture the corresponding inference artifact on the server. The model folder
must point to the official trained model:

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

The server handoff order is: capture the transform artifact, capture the
inference artifact, run each local parity report, then run one local fold-0
prediction/validation export. No five-fold run starts before both parity reports
have status `passed`.

## Local parity reports

Run the transform parity report in `newconda` against the server artifact and a
standalone artifact produced from the same case, seed, and policy:

```powershell
conda run -n newconda python -m standalone_nnunet2d.tools.parity_report `
  --oracle-root standalone_nnunet2d\outputs\oracle\preprocess\case_0001 `
  --standalone-root standalone_nnunet2d\outputs\standalone\preprocess\case_0001 `
  --image-atol 0 `
  --output C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\transform_parity_case005_v3.json
```

Run a single-root inference comparison with the corresponding inference artifacts
only as a diagnostic exact comparison:

```powershell
conda run -n newconda python -m standalone_nnunet2d.tools.parity_report `
  --oracle-root standalone_nnunet2d\outputs\oracle\inference\case_0001 `
  --standalone-root standalone_nnunet2d\outputs\standalone\inference\case_0001 `
  --image-atol 0 `
  --output standalone_nnunet2d\outputs\inference_parity_report.json
```

The accepted inference gate is `repeat_oracle_stability_v1`. It requires at
least three independent oracle runs, with each official artifact produced by a
separate server `nnunetv2` invocation and stored below a different root:

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
and pairwise differences. It uses no Dice threshold, morphological cleanup,
nonzero tolerance, retry-until-match behavior, or hidden fallback.

Inference artifacts must include the `inference_context` containing the fold,
official source-checkpoint SHA256, and normalized device. Existing inference
artifacts without this context must be recaptured; they are not automatically
compatible and the gate has no legacy fallback.

Both the transform report and the repeated inference report must have
`status: "passed"` before an external alignment decision. The report remains
`official_alignment_pending`, and a passing repeated report does not
automatically relabel historical runs `official_aligned`; existing training,
prediction, validation, and OOF artifacts remain pending.

## Final five-fold aligned training and OOF sequence

Use these Windows paths for the current run. If the actual transform report
filename differs, replace that path consistently in every command. The batch
size is fixed at 12. The throughput profile does not enable AMP, TF32, or
compile; it only selects the approved loader-throughput settings.

After the current final source has been synchronized and the fresh output root
is empty or newly created, run each formal train command once, for folds 0--4:

```powershell
conda run -n newconda python standalone_nnunet2d\formal_train.py --raw-root "C:\lijialin\models3d\nnUNet\nnUNet_raw\Dataset501_StrokeLesion" --plans "C:\lijialin\models3d\nnUNet\nnUNet_preprocessed\Dataset501_StrokeLesion\nnUNetPlans.json" --output-root "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\official_aligned_5fold\formal\fold_0" --fold 0 --device cuda:0 --epochs 1000 --performance-profile throughput --transform-parity-report "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\transform_parity_case005_v3.json" --inference-parity-report "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\inference_repeat_parity_case005_ctx.json" --confirm-run
conda run -n newconda python standalone_nnunet2d\formal_train.py --raw-root "C:\lijialin\models3d\nnUNet\nnUNet_raw\Dataset501_StrokeLesion" --plans "C:\lijialin\models3d\nnUNet\nnUNet_preprocessed\Dataset501_StrokeLesion\nnUNetPlans.json" --output-root "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\official_aligned_5fold\formal\fold_1" --fold 1 --device cuda:0 --epochs 1000 --performance-profile throughput --transform-parity-report "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\transform_parity_case005_v3.json" --inference-parity-report "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\inference_repeat_parity_case005_ctx.json" --confirm-run
conda run -n newconda python standalone_nnunet2d\formal_train.py --raw-root "C:\lijialin\models3d\nnUNet\nnUNet_raw\Dataset501_StrokeLesion" --plans "C:\lijialin\models3d\nnUNet\nnUNet_preprocessed\Dataset501_StrokeLesion\nnUNetPlans.json" --output-root "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\official_aligned_5fold\formal\fold_2" --fold 2 --device cuda:0 --epochs 1000 --performance-profile throughput --transform-parity-report "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\transform_parity_case005_v3.json" --inference-parity-report "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\inference_repeat_parity_case005_ctx.json" --confirm-run
conda run -n newconda python standalone_nnunet2d\formal_train.py --raw-root "C:\lijialin\models3d\nnUNet\nnUNet_raw\Dataset501_StrokeLesion" --plans "C:\lijialin\models3d\nnUNet\nnUNet_preprocessed\Dataset501_StrokeLesion\nnUNetPlans.json" --output-root "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\official_aligned_5fold\formal\fold_3" --fold 3 --device cuda:0 --epochs 1000 --performance-profile throughput --transform-parity-report "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\transform_parity_case005_v3.json" --inference-parity-report "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\inference_repeat_parity_case005_ctx.json" --confirm-run
conda run -n newconda python standalone_nnunet2d\formal_train.py --raw-root "C:\lijialin\models3d\nnUNet\nnUNet_raw\Dataset501_StrokeLesion" --plans "C:\lijialin\models3d\nnUNet\nnUNet_preprocessed\Dataset501_StrokeLesion\nnUNetPlans.json" --output-root "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\official_aligned_5fold\formal\fold_4" --fold 4 --device cuda:0 --epochs 1000 --performance-profile throughput --transform-parity-report "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\transform_parity_case005_v3.json" --inference-parity-report "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\inference_repeat_parity_case005_ctx.json" --confirm-run
```

The raw root is
`C:\lijialin\models3d\nnUNet\nnUNet_raw\Dataset501_StrokeLesion` and the
plans file is
`C:\lijialin\models3d\nnUNet\nnUNet_preprocessed\Dataset501_StrokeLesion\nnUNetPlans.json`.

After each training command, run the matching fold validation immediately; it
does not require a separate `predict.py` run. Do not run a duplicate
`predict.py` command. The fold-validation command uses
the aligned checkpoint without `--allow-pending`, performs full-volume
prediction, and writes the prediction manifest and fold report. Fold reports
share one `crossval` directory:

```powershell
conda run -n newconda python standalone_nnunet2d\validate_cv.py fold --checkpoint "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\official_aligned_5fold\formal\fold_0\checkpoint_best.pth" --raw-root "C:\lijialin\models3d\nnUNet\nnUNet_raw\Dataset501_StrokeLesion" --fold 0 --output-root "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\official_aligned_5fold\crossval" --device cuda:0
conda run -n newconda python standalone_nnunet2d\validate_cv.py fold --checkpoint "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\official_aligned_5fold\formal\fold_1\checkpoint_best.pth" --raw-root "C:\lijialin\models3d\nnUNet\nnUNet_raw\Dataset501_StrokeLesion" --fold 1 --output-root "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\official_aligned_5fold\crossval" --device cuda:0
conda run -n newconda python standalone_nnunet2d\validate_cv.py fold --checkpoint "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\official_aligned_5fold\formal\fold_2\checkpoint_best.pth" --raw-root "C:\lijialin\models3d\nnUNet\nnUNet_raw\Dataset501_StrokeLesion" --fold 2 --output-root "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\official_aligned_5fold\crossval" --device cuda:0
conda run -n newconda python standalone_nnunet2d\validate_cv.py fold --checkpoint "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\official_aligned_5fold\formal\fold_3\checkpoint_best.pth" --raw-root "C:\lijialin\models3d\nnUNet\nnUNet_raw\Dataset501_StrokeLesion" --fold 3 --output-root "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\official_aligned_5fold\crossval" --device cuda:0
conda run -n newconda python standalone_nnunet2d\validate_cv.py fold --checkpoint "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\official_aligned_5fold\formal\fold_4\checkpoint_best.pth" --raw-root "C:\lijialin\models3d\nnUNet\nnUNet_raw\Dataset501_StrokeLesion" --fold 4 --output-root "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\official_aligned_5fold\crossval" --device cuda:0
```

After all five fold reports exist, aggregate only when the directory contains
`fold_0_report.json` through `fold_4_report.json`. The aggregate requires
fold_0_report.json through fold_4_report.json, exactly 95 unique IDs, zero
failed cases, and identical evidence across all five aligned reports. It
writes an `official_aligned` `oof_summary.json` and
the OOF case metrics:

```powershell
conda run -n newconda python standalone_nnunet2d\validate_cv.py aggregate --output-root "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\official_aligned_5fold\crossval"
cd "C:\lijialin\segementation\.worktrees\standalone-nnunet2d\standalone_nnunet2d\outputs\official_aligned_5fold\crossval"
type oof_summary.json
```
